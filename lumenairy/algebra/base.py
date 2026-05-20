"""lumenairy.algebra.base -- Operator and CompositeOperator base classes.

Implements the Nazarathy/Shamir-style optical operator algebra over
LumenAiry's existing array-first propagator infrastructure.  An
:class:`Operator` carries 2x2 ABCD matrices (separate ``_abcd_x`` and
``_abcd_y`` for anamorphic systems), composes via ``*`` (matrix
multiply on ABCDs, chain-and-delegate on the field-application
side), and applies to a :class:`~lumenairy.sources.Source` or bare
``(E, dx, wavelength)`` triple via ``__call__`` / :meth:`apply`.

Convention
----------
``A * B`` means "apply ``B`` first, then ``A``".  The ABCD of
``A * B`` is ``A.abcd @ B.abcd`` -- matrix-on-the-left, matching the
:func:`lumenairy.raytrace.system_abcd` convention where each
refractive / transfer matrix multiplies the running product on the
left.  Field-application order is therefore ``A(B(field))``.

References
----------
- Nazarathy, M. & Shamir, J., "Fourier optics described by operator
  algebra," JOSA 70 (2), 150-159 (1980).  The foundational paper.
- LumenAiry ABCD ground truth: :func:`lumenairy.raytrace.system_abcd`,
  :func:`lumenairy.raytrace.lens_abcd`.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    # v5.0.1 (audit P1-NEW-V4-1): TYPE_CHECKING guard for forward reference
    # to lumenairy.sources.Source used in ``Operator.__call__`` annotations.
    # The runtime ``__call__`` body does its own lazy ``from ..sources import
    # Source as _Source`` to avoid a top-level circular import; ruff F821
    # needs a typing-time name binding so the string-quoted ``'Source'`` in
    # the signature resolves.
    from ..sources import Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_abcd(M: Any, name: str) -> np.ndarray:
    """Coerce ``M`` to a real float64 (2, 2) ndarray with sanity checks.

    Used by every primitive ``__init__`` to keep operator ABCDs in
    a uniform representation regardless of how the caller supplied
    them (Python list-of-lists, ``np.array`` with int dtype, ...).
    """
    arr = np.asarray(M, dtype=np.float64)
    if arr.shape != (2, 2):
        raise ValueError(
            f"{name}: ABCD must be a (2, 2) matrix, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name}: ABCD contains non-finite entries:\n{arr}"
        )
    return arr


def _coerce_propagation_output(
    out: Any,
    *,
    dx_default: float,
    dy_default: float,
) -> Tuple[np.ndarray, float, float]:
    """Coerce a propagator return into ``(E_out, dx_out, dy_out)``.

    Mirrors the logic in :func:`lumenairy.propagators.dispatch._coerce_field`
    but threads explicit dx/dy defaults through for kernels that return
    a bare ndarray (ASM, RS) versus a tuple (Fresnel, Fraunhofer, SAS).
    """
    # Tuple / list returned by fresnel / fraunhofer / SAS:
    # (E, dx_out, dy_out) or (E, dx_out) for the resample helper.
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        first = out[0]
        if isinstance(first, np.ndarray):
            dx_out = dx_default
            dy_out = dy_default
            if len(out) >= 2:
                try:
                    dx_out = float(out[1])
                except (TypeError, ValueError):
                    pass
            if len(out) >= 3:
                try:
                    dy_out = float(out[2])
                except (TypeError, ValueError):
                    pass
            else:
                # Square-grid kernels return only (E, dx_out); preserve
                # the dx_out/dy_out coupling unless the caller specified
                # an anamorphic input pitch.
                if dy_default == dx_default:
                    dy_out = dx_out
            return first, dx_out, dy_out
    # Bare ndarray (ASM / RS).
    if isinstance(out, np.ndarray):
        return out, dx_default, dy_default
    # PropagationResult from the dispatcher's return_result path.
    field_attr = getattr(out, 'field', None)
    if isinstance(field_attr, np.ndarray):
        dx_attr = getattr(out, 'dx', None)
        dy_attr = getattr(out, 'dy', None)
        return (
            field_attr,
            float(dx_attr) if dx_attr is not None else dx_default,
            float(dy_attr) if dy_attr is not None else dy_default,
        )
    raise TypeError(
        f"Operator._apply: unexpected propagator return type "
        f"{type(out).__name__!r}; expected ndarray, "
        f"(E, dx_out[, dy_out]) tuple, or PropagationResult."
    )


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class Operator:
    """Base class for composable optical operators in the Nazarathy/
    Shamir algebraic formalism.

    Each Operator carries 2x2 ABCD matrices for the x and y directions
    (real-valued, supports anamorphic systems) and implements an
    :meth:`_apply` hook that delegates to existing LumenAiry
    propagators / element functions.

    Operators compose via ``*`` (matrix multiply on ABCDs, chain of
    :meth:`_apply` calls right-to-left) and apply to a field via
    :meth:`__call__`.

    Convention follows Nazarathy & Shamir, "Fourier optics described
    by operator algebra," JOSA 70 (2), 1980.  The ABCD layout matches
    :func:`lumenairy.raytrace.system_abcd` -- ``A * B`` produces the
    composite ABCD ``A.abcd @ B.abcd`` and application order is
    ``A(B(field))``.

    Subclasses must:

    1. Populate ``self._abcd_x`` and ``self._abcd_y`` in their
       constructor (use :func:`_validate_abcd`).
    2. Override :meth:`_apply` to apply this operator's physics to a
       complex field, returning ``(E_out, dx_out, dy_out)``.

    See :class:`CompositeOperator` for the multi-stage chain class
    that ``__mul__`` produces.
    """

    # Subclasses set these in __init__:
    _abcd_x: np.ndarray  # shape (2, 2), real, float64
    _abcd_y: np.ndarray  # shape (2, 2), real, float64

    # -----------------------------------------------------------------
    # ABCD readout
    # -----------------------------------------------------------------

    @property
    def abcd(self) -> np.ndarray:
        """System ABCD if isotropic (``_abcd_x == _abcd_y``).

        Raises
        ------
        ValueError
            If the operator is anamorphic.  Use :attr:`abcd_x` /
            :attr:`abcd_y` instead, or read :attr:`is_anamorphic`
            first.
        """
        if np.allclose(self._abcd_x, self._abcd_y):
            return self._abcd_x.copy()
        raise ValueError(
            "Operator.abcd: anamorphic system (abcd_x != abcd_y); "
            "use .abcd_x / .abcd_y instead.  Inspect .is_anamorphic "
            "to branch on this case."
        )

    @property
    def abcd_x(self) -> np.ndarray:
        """x-direction ABCD (always defined)."""
        return self._abcd_x.copy()

    @property
    def abcd_y(self) -> np.ndarray:
        """y-direction ABCD (always defined)."""
        return self._abcd_y.copy()

    def _guard_isotropic(self, prop: str) -> None:
        if not np.allclose(self._abcd_x, self._abcd_y):
            raise ValueError(
                f"Operator.{prop}: anamorphic system; the {prop} "
                f"scalar is only defined when abcd_x == abcd_y.  Use "
                f".abcd_x / .abcd_y for the per-axis ABCDs."
            )

    @property
    def A(self) -> float:
        """ABCD ``A`` element (anamorphic-guarded scalar)."""
        self._guard_isotropic('A')
        return float(self._abcd_x[0, 0])

    @property
    def B(self) -> float:
        """ABCD ``B`` element (anamorphic-guarded scalar)."""
        self._guard_isotropic('B')
        return float(self._abcd_x[0, 1])

    @property
    def C(self) -> float:
        """ABCD ``C`` element (anamorphic-guarded scalar)."""
        self._guard_isotropic('C')
        return float(self._abcd_x[1, 0])

    @property
    def D(self) -> float:
        """ABCD ``D`` element (anamorphic-guarded scalar)."""
        self._guard_isotropic('D')
        return float(self._abcd_x[1, 1])

    @property
    def efl(self) -> float:
        """Effective focal length [m].

        Defined as ``-1 / C`` where ``C`` is the ABCD lower-left
        entry.  Returns ``+inf`` if ``|C| < 1e-30`` (afocal).
        Matches the :func:`lumenairy.raytrace.system_abcd`
        convention.

        For anamorphic systems, raises ``ValueError`` -- read
        :attr:`abcd_x` / :attr:`abcd_y` and compute the per-axis EFL
        explicitly.
        """
        self._guard_isotropic('efl')
        C = float(self._abcd_x[1, 0])
        if abs(C) < 1e-30:
            return float('inf')
        return -1.0 / C

    @property
    def is_anamorphic(self) -> bool:
        """``True`` iff the operator has different ABCDs in x and y."""
        return not np.allclose(self._abcd_x, self._abcd_y)

    # -----------------------------------------------------------------
    # Composition
    # -----------------------------------------------------------------

    def __mul__(self, other: 'Operator') -> 'CompositeOperator':
        """Compose two operators.  ``A * B`` means "apply ``B`` first,
        then ``A``"; the resulting ABCD is ``A.abcd @ B.abcd``.

        Flattens nested :class:`CompositeOperator` chains so deeply-
        nested expressions don't produce a tree of intermediates.
        """
        if not isinstance(other, Operator):
            return NotImplemented
        left_chain = (
            self._chain if isinstance(self, CompositeOperator) else [self]
        )
        right_chain = (
            other._chain if isinstance(other, CompositeOperator) else [other]
        )
        # right_chain is applied first; left_chain is applied last.
        # Application order in CompositeOperator._apply walks the
        # internal ``_chain`` left-to-right, so right (applied first)
        # comes first in the list.
        return CompositeOperator(right_chain + left_chain)

    def __rmul__(self, other: Any) -> Any:
        # Scalar pre-multiplication is intentionally NOT supported --
        # operators in this algebra represent physical transforms, not
        # complex amplitudes.  Defer to NotImplemented so a useful
        # TypeError surfaces.
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    # -----------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------

    def __call__(
        self,
        source: Union['Source', Tuple[Any, float, float], Tuple[Any, float, float, float]],
    ) -> Union['Source', Tuple[Any, float, float], Tuple[Any, float, float, float]]:
        """Apply this operator to a :class:`Source` or bare tuple.

        Accepted inputs:

        - :class:`~lumenairy.sources.Source` -- returns a new Source.
        - ``(E, dx, wavelength)`` tuple -- returns ``(E_out, dx_out,
          wavelength)``.
        - ``(E, dx, dy, wavelength)`` tuple -- returns ``(E_out,
          dx_out, dy_out, wavelength)``.

        The Source path threads ``dy`` through automatically (default
        ``dy = dx`` for square grids).  Use :meth:`apply` if you want
        the bare ndarray entry point with named kwargs.
        """
        # Source path.
        from ..sources import Source as _Source

        if isinstance(source, _Source):
            dx_in = float(source.dx)
            dy_in = float(source.dy) if source.dy is not None else dx_in
            wl = float(source.wavelength)
            E_out, dx_out, dy_out = self._apply(
                source.E, dx=dx_in, dy=dy_in, wavelength=wl,
            )
            return _Source(
                E=E_out,
                dx=dx_out,
                dy=dy_out,
                wavelength=wl,
                source_point=source.source_point,
                name=(source.name or 'Source') + f"->{type(self).__name__}",
            )

        # Tuple path.
        if isinstance(source, tuple):
            if len(source) == 3:
                E, dx_in, wl = source
                dx_in = float(dx_in)
                wl = float(wl)
                E_out, dx_out, _ = self._apply(
                    E, dx=dx_in, dy=dx_in, wavelength=wl,
                )
                return (E_out, dx_out, wl)
            if len(source) == 4:
                E, dx_in, dy_in, wl = source
                dx_in = float(dx_in)
                dy_in = float(dy_in)
                wl = float(wl)
                E_out, dx_out, dy_out = self._apply(
                    E, dx=dx_in, dy=dy_in, wavelength=wl,
                )
                return (E_out, dx_out, dy_out, wl)
            raise TypeError(
                f"Operator.__call__: tuple input must be "
                f"(E, dx, wavelength) or (E, dx, dy, wavelength); "
                f"got tuple of length {len(source)}."
            )

        raise TypeError(
            f"Operator.__call__: expected a Source or "
            f"(E, dx, wavelength) / (E, dx, dy, wavelength) tuple; "
            f"got {type(source).__name__}."
        )

    def apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        wavelength: float,
        dy: Optional[float] = None,
    ) -> np.ndarray:
        """Bare-ndarray entry point.

        Parameters
        ----------
        E : ndarray, complex
            Input field.
        dx : float
            Grid spacing in x [m].
        wavelength : float
            Vacuum wavelength [m].
        dy : float, optional
            Grid spacing in y [m].  Defaults to ``dx``.

        Returns
        -------
        E_out : ndarray, complex
            Field after applying this operator.  The output grid
            pitch may differ from the input pitch (e.g. for Fresnel-
            backed :class:`~lumenairy.algebra.FreeSpace`); use
            :meth:`__call__` with a Source if you need the updated
            pitch threaded through automatically.
        """
        dy_in = float(dy) if dy is not None else float(dx)
        E_out, _, _ = self._apply(
            E, dx=float(dx), dy=dy_in, wavelength=float(wavelength),
        )
        return E_out

    # -----------------------------------------------------------------
    # Subclass hook
    # -----------------------------------------------------------------

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Apply this operator to a complex field.

        Returns
        -------
        (E_out, dx_out, dy_out)
            The propagated field and the (possibly changed) grid
            spacings.  Most operators preserve ``dx`` and ``dy`` --
            the exceptions are Fresnel-backed
            :class:`~lumenairy.algebra.FreeSpace`,
            :class:`~lumenairy.algebra.FourierTransform`, and
            :class:`~lumenairy.algebra.Magnify`, which deliberately
            change the output pitch.

        Subclasses MUST override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._apply must be overridden."
        )


# ---------------------------------------------------------------------------
# CompositeOperator
# ---------------------------------------------------------------------------


class CompositeOperator(Operator):
    """A chain of operators produced by composition.

    ``CompositeOperator`` stores the flattened chain so callers can
    inspect intermediate stages (:meth:`stages`,
    :meth:`apply_with_intermediates`) and pre-computes the composite
    ABCD for fast readout.

    Convention
    ----------
    ``self._chain[0]`` is applied first; ``self._chain[-1]`` is
    applied last.  Visually: read left-to-right is the application
    order; the math counterpart ``A * B * C`` produces a chain
    ``[C, B, A]`` (rightmost-of-product applied first).

    Constructed implicitly via ``op_a * op_b``; users normally do
    not instantiate this class directly.
    """

    def __init__(self, chain: List[Operator]):
        if not chain:
            raise ValueError(
                "CompositeOperator: chain must contain at least one "
                "operator."
            )
        for i, op in enumerate(chain):
            if not isinstance(op, Operator):
                raise TypeError(
                    f"CompositeOperator: chain[{i}] is "
                    f"{type(op).__name__}, expected Operator subclass."
                )
        self._chain: List[Operator] = list(chain)
        # Composite ABCD: apply chain left-to-right, so the matrix
        # product accumulates on the LEFT of the running product
        # (matches ``A.abcd @ B.abcd`` for ``A * B``).
        abcd_x = np.eye(2, dtype=np.float64)
        abcd_y = np.eye(2, dtype=np.float64)
        for op in self._chain:
            abcd_x = op._abcd_x @ abcd_x
            abcd_y = op._abcd_y @ abcd_y
        self._abcd_x = abcd_x
        self._abcd_y = abcd_y

    def __repr__(self) -> str:
        # Display in math order (right-applied-first) so a user who
        # types ``A * B * C`` sees ``CompositeOperator(A * B * C)``.
        names = [type(op).__name__ for op in reversed(self._chain)]
        return "CompositeOperator(" + " * ".join(names) + ")"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Walk the chain left-to-right, threading ``dx`` and ``dy``
        through each stage.
        """
        for op in self._chain:
            E, dx, dy = op._apply(E, dx=dx, dy=dy, wavelength=wavelength)
        return E, dx, dy

    def stages(self) -> List[Operator]:
        """Return a copy of the underlying operator chain.

        The list is in application order (``[0]`` applied first,
        ``[-1]`` applied last).  Use this for visualization or to
        introspect a composed system.
        """
        return list(self._chain)

    def apply_with_intermediates(
        self,
        E: np.ndarray,
        *,
        dx: float,
        wavelength: float,
        dy: Optional[float] = None,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Apply the chain, returning a list of intermediate states.

        Returns
        -------
        list of (E_i, dx_i, dy_i)
            One entry per stage in :attr:`_chain`, in application
            order.  ``result[0]`` is the field after the first
            operator; ``result[-1]`` is the final output.

        Useful for through-stage visualization (e.g. plotting the
        field at each step of a 4f system).
        """
        dy_in = float(dy) if dy is not None else float(dx)
        out: List[Tuple[np.ndarray, float, float]] = []
        E_curr = E
        dx_curr = float(dx)
        dy_curr = dy_in
        wl = float(wavelength)
        for op in self._chain:
            E_curr, dx_curr, dy_curr = op._apply(
                E_curr, dx=dx_curr, dy=dy_curr, wavelength=wl,
            )
            out.append((E_curr, dx_curr, dy_curr))
        return out


__all__ = ['Operator', 'CompositeOperator']
