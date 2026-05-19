"""lumenairy.algebra.primitives -- core operator primitives.

This module defines the primitive optical operators used to build
LumenAiry algebraic expressions.  Each primitive carries its closed-
form ABCD (matching Nazarathy/Shamir 1980 conventions) and delegates
the field-application step to an existing LumenAiry function.

Primitives
----------

- :class:`FreeSpace` -- ``[[1, d], [0, 1]]``, delegates to
  :func:`lumenairy.propagate`.
- :class:`ThinLens` -- ``[[1, 0], [-1/f, 1]]``, delegates to
  :func:`lumenairy.apply_thin_lens`.
- :class:`CylindricalLens` -- per-axis ABCD, delegates to
  :func:`lumenairy.apply_cylindrical_lens`.
- :class:`Magnify` -- ``[[1/a, 0], [0, a]]`` (anamorphic when
  ``a_x != a_y``), delegates to
  :func:`lumenairy.resample_field` with a ``sqrt(a_x * a_y)``
  amplitude prefactor.
- :class:`FourierTransform` -- physical optical Fourier transform
  realized as the literal 3-stage chain
  ``FreeSpace(f) * ThinLens(f) * FreeSpace(f)`` (Goodman §5.2).
  v4.15.2 closes a field-vs-ABCD inconsistency in v4.15.1 where
  the ABCD claimed 3-stage but ``_apply`` ran only 2 stages.

References
----------
- Nazarathy, M. & Shamir, J., "Fourier optics described by operator
  algebra," JOSA 70 (2), 150-159 (1980).  See §IV for ``V``,
  ``F``, ``Q``, ``R`` operator family.
- ABCD ground truth: :func:`lumenairy.raytrace.system_abcd`.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import (
    Operator,
    _coerce_propagation_output,
    _validate_abcd,
)


# ---------------------------------------------------------------------------
# FreeSpace
# ---------------------------------------------------------------------------


class FreeSpace(Operator):
    """Free-space propagation by distance ``d``.

    ABCD::

        [[1, d],
         [0, 1]]

    Parameters
    ----------
    distance : float
        Propagation distance ``d`` [m].  Must be real and finite.
        Negative distances are honoured by the underlying ASM /
        Rayleigh-Sommerfeld backends (back-propagation); the Fresnel
        and Fraunhofer kernels are forward-only.
    method : str, default ``'auto'``
        Propagator backend to delegate to.  One of
        :data:`lumenairy.VALID_METHODS`.  ``'auto'`` lets the
        dispatcher pick the best method given the geometry.

    Notes
    -----
    Delegates :meth:`_apply` to
    :func:`lumenairy.propagators.dispatch.propagate` for a Source-
    driven call chain.  The ABCD is independent of the chosen
    method.
    """

    def __init__(self, distance: float, *, method: str = 'auto') -> None:
        try:
            d = float(distance)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"FreeSpace: distance must be a real number, got "
                f"{distance!r}."
            ) from e
        if not np.isfinite(d):
            raise ValueError(
                f"FreeSpace: distance must be finite, got {d}."
            )
        self.distance = d
        self.method = str(method)
        M = _validate_abcd([[1.0, d], [0.0, 1.0]], 'FreeSpace')
        self._abcd_x = M
        self._abcd_y = M.copy()

    def __repr__(self) -> str:
        return f"FreeSpace(distance={self.distance:.6g}, method={self.method!r})"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        # Zero-distance fast path: no FFT round-trip needed.
        if self.distance == 0.0:
            return E, dx, dy
        from ..propagators.dispatch import propagate
        # v4.15.2 (audit P1-NEW-C): thread ``dy`` into the dispatcher
        # when it differs from ``dx``, so anamorphic chains (e.g.
        # ``Magnify(a_x, a_y) * FreeSpace(d)``) propagate on the
        # correct y-axis grid pitch.  Pre-fix the call dropped
        # ``dy`` and the underlying kernel silently defaulted to
        # ``dy = dx``.  ``propagate`` forwards ``**method_kwargs``
        # to the chosen kernel; ASM / Fresnel / Fraunhofer / RS all
        # accept ``dy`` (Optional[float] = None, defaulting to dx).
        # SAS does NOT accept ``dy`` (square-grid kernel) -- we only
        # forward the kwarg when needed so the common ``dy == dx``
        # case continues to work with any auto-selected method.
        kw = {}
        anamorphic = (dy is not None and float(dy) != float(dx))
        if anamorphic:
            kw['dy'] = float(dy)
        # v4.15.3 (audit P1-NEW-F1-1): when ``dy != dx`` (anamorphic
        # input), the SAS kernel does NOT accept the ``dy`` kwarg --
        # it is a square-grid-only kernel.  With ``self.method ==
        # 'auto'`` (the default), the dispatcher routes to SAS in the
        # far-field regime (``Q = N*dx**2/(lambda*z) > 1``), and the
        # forwarded ``dy`` kwarg raises
        # ``TypeError: sas_propagate() got an unexpected keyword
        # argument 'dy'``.  Force ``method='asm'`` whenever ``dy !=
        # dx``: ASM is anamorphic-aware and round-trips correctly on
        # non-square grids.  The user implicitly opted into
        # anamorphic support by passing ``dy``; making ``auto`` a
        # hint (not a contract) is the cleanest fix.  If the user
        # explicitly chose ``method='sas'`` on an anamorphic grid,
        # we still forward ``dy`` so the underlying kernel's own
        # error is what surfaces -- the explicit ``method=`` choice
        # is the user's responsibility to keep in sync with the
        # input geometry.
        method = self.method
        if anamorphic and method == 'auto':
            method = 'asm'
        out = propagate(
            E,
            z=self.distance,
            wavelength=wavelength,
            dx=dx,
            method=method,
            **kw,
        )
        return _coerce_propagation_output(
            out, dx_default=dx, dy_default=dy,
        )


# ---------------------------------------------------------------------------
# ThinLens
# ---------------------------------------------------------------------------


class ThinLens(Operator):
    """Paraxial thin lens with focal length ``f``.

    ABCD::

        [[1,    0],
         [-1/f, 1]]

    ``f = np.inf`` produces the identity ABCD (no optical power) and
    a no-op :meth:`_apply` -- useful as a placeholder in algebraic
    expressions.

    Parameters
    ----------
    f : float
        Focal length [m].  Positive = converging.

    Notes
    -----
    Delegates to :func:`lumenairy.apply_thin_lens` with
    ``lens_model='paraxial'`` (the closed-form quadratic phase
    matching the ABCD).  For higher-fidelity lens models
    (nonparaxial / aplanatic / decentered), build the algebra
    expression with :class:`FreeSpace` segments and an explicit
    :func:`apply_real_lens` call -- the operator-algebra path is
    paraxial-only by design.
    """

    def __init__(self, f: float) -> None:
        try:
            fval = float(f)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"ThinLens: focal length must be a real number, got "
                f"{f!r}."
            ) from e
        if not (np.isfinite(fval) or np.isinf(fval)):
            raise ValueError(
                f"ThinLens: focal length must be finite or +/-inf, "
                f"got {fval} (NaN?)."
            )
        if fval == 0.0:
            raise ValueError(
                "ThinLens: focal length must be non-zero "
                "(use FreeSpace(0) for a no-op identity or pass np.inf "
                "for zero power)."
            )
        self.f = fval
        if np.isinf(fval):
            inv_f = 0.0
        else:
            inv_f = 1.0 / fval
        M = _validate_abcd([[1.0, 0.0], [-inv_f, 1.0]], 'ThinLens')
        self._abcd_x = M
        self._abcd_y = M.copy()

    def __repr__(self) -> str:
        return f"ThinLens(f={self.f:.6g})"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        if np.isinf(self.f):
            return E, dx, dy
        from ..elements.lenses import apply_thin_lens
        E_out = apply_thin_lens(
            E, f=self.f, wavelength=wavelength, dx=dx, dy=dy,
        )
        return E_out, dx, dy


# ---------------------------------------------------------------------------
# CylindricalLens
# ---------------------------------------------------------------------------


class CylindricalLens(Operator):
    """Anamorphic thin lens with separate focal lengths in x and y.

    ABCD::

        abcd_x = [[1, 0], [-1/f_x, 1]]
        abcd_y = [[1, 0], [-1/f_y, 1]]

    Either focal length may be ``np.inf`` (flat in that axis).
    ``f_x == f_y`` reduces to a rotationally symmetric
    :class:`ThinLens`.

    Parameters
    ----------
    f_x : float, default ``np.inf``
        Focal length along x [m].  Positive = converging.
    f_y : float, default ``np.inf``
        Focal length along y [m].

    Notes
    -----
    Delegates :meth:`_apply` to two
    :func:`lumenairy.apply_cylindrical_lens` calls (one per axis)
    so anamorphic prescriptions are handled in a single primitive.
    Internally the application short-circuits when either focal
    length is infinite -- only the focusing axis gets a phase
    mask applied.
    """

    def __init__(
        self,
        f_x: float = np.inf,
        f_y: float = np.inf,
    ) -> None:
        try:
            fxv = float(f_x)
            fyv = float(f_y)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"CylindricalLens: focal lengths must be real numbers, "
                f"got f_x={f_x!r}, f_y={f_y!r}."
            ) from e
        if fxv == 0.0 or fyv == 0.0:
            raise ValueError(
                "CylindricalLens: focal lengths must be non-zero "
                "(use np.inf for a flat axis)."
            )
        if np.isnan(fxv) or np.isnan(fyv):
            raise ValueError(
                f"CylindricalLens: focal lengths must be finite or "
                f"+/-inf, got f_x={fxv}, f_y={fyv}."
            )
        self.f_x = fxv
        self.f_y = fyv

        inv_fx = 0.0 if np.isinf(fxv) else 1.0 / fxv
        inv_fy = 0.0 if np.isinf(fyv) else 1.0 / fyv
        self._abcd_x = _validate_abcd(
            [[1.0, 0.0], [-inv_fx, 1.0]], 'CylindricalLens.abcd_x',
        )
        self._abcd_y = _validate_abcd(
            [[1.0, 0.0], [-inv_fy, 1.0]], 'CylindricalLens.abcd_y',
        )

    def __repr__(self) -> str:
        return (
            f"CylindricalLens(f_x={self.f_x:.6g}, f_y={self.f_y:.6g})"
        )

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        from ..elements.lenses import apply_cylindrical_lens
        E_curr = E
        if not np.isinf(self.f_x):
            E_curr = apply_cylindrical_lens(
                E_curr, f=self.f_x, wavelength=wavelength,
                dx=dx, dy=dy, axis='x',
            )
        if not np.isinf(self.f_y):
            E_curr = apply_cylindrical_lens(
                E_curr, f=self.f_y, wavelength=wavelength,
                dx=dx, dy=dy, axis='y',
            )
        return E_curr, dx, dy


# ---------------------------------------------------------------------------
# Magnify
# ---------------------------------------------------------------------------


class Magnify(Operator):
    """Geometric magnification operator (Nazarathy/Shamir ``V[a]``).

    ABCD::

        [[1/a, 0],
         [0,   a]]

    Rescales the field's grid pitch from ``dx`` to ``dx / a`` (and
    ``dy`` to ``dy / a``).  Under the Nazarathy/Shamir 1980
    ``V[a]`` convention, ``V[a]: (x, y) -> (x/a, y/a)``, so the
    output is **shrunk by a factor of ``a``** when ``a > 1`` and
    **magnified by a factor of ``1/a``** when ``0 < a < 1``.

    For anamorphic stretches, pass distinct ``a_x`` and ``a_y``.

    Parameters
    ----------
    a_x : float
        Magnification along x.  Must be positive and finite.
    a_y : float, optional
        Magnification along y.  Defaults to ``a_x`` (isotropic).

    Output grid
    -----------
    On an ``N``-sample fixed grid, the output pixel pitch becomes::

        new_dx = dx / a_x,   new_dy = dy / a_y

    and the output extent is::

        N * new_dx = N * dx / a_x,   N * new_dy = N * dy / a_y

    Thus:

    - ``a > 1``: output is **shrunk** by ``a`` (demagnifier in
      optical-system terms).  Pitch is ``dx/a`` (finer);
      extent is ``N*dx/a`` (smaller).
    - ``0 < a < 1``: output is **magnified** by ``1/a``.  Pitch is
      ``dx/a`` (coarser); extent is ``N*dx/a`` (larger).

    Notes
    -----
    Application is a unitary rescale on a fixed-N grid: the field
    values are multiplied by ``sqrt(a_x * a_y)`` and the announced
    output pitch contracts by ``a``.  This preserves total power::

        |E_out|^2 * dx_out * dy_out  ==  |E_in|^2 * dx_in * dy_in.

    No interpolation onto a coarser/finer support is performed --
    the field values stay at the same array indices and only the
    grid-pitch metadata changes (see the implementation note inside
    :meth:`_apply`).

    The ABCD ``diag(1/a, a)`` is the **corrected** Nazarathy/Shamir
    form -- the original 1980 paper had a transcription slip in
    the off-diagonal sign; the corrected diagonal form is what
    matches a physical magnifier in the operator algebra (the
    height-to-angle off-diagonal is zero because pure magnification
    doesn't bend rays).
    """

    def __init__(self, a_x: float, a_y: Optional[float] = None) -> None:
        try:
            ax = float(a_x)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Magnify: a_x must be a real number, got {a_x!r}."
            ) from e
        if a_y is None:
            ay = ax
        else:
            try:
                ay = float(a_y)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Magnify: a_y must be a real number or None, got "
                    f"{a_y!r}."
                ) from e
        if not (np.isfinite(ax) and np.isfinite(ay)):
            raise ValueError(
                f"Magnify: a_x and a_y must be finite, got a_x={ax}, "
                f"a_y={ay}."
            )
        if ax <= 0.0 or ay <= 0.0:
            raise ValueError(
                f"Magnify: a_x and a_y must be positive, got a_x={ax}, "
                f"a_y={ay} (negative magnification corresponds to an "
                f"image inversion; use a separate ThinLens-pair "
                f"4f-inverter for sign flips)."
            )
        self.a_x = ax
        self.a_y = ay
        self._abcd_x = _validate_abcd(
            [[1.0 / ax, 0.0], [0.0, ax]], 'Magnify.abcd_x',
        )
        self._abcd_y = _validate_abcd(
            [[1.0 / ay, 0.0], [0.0, ay]], 'Magnify.abcd_y',
        )

    def __repr__(self) -> str:
        return f"Magnify(a_x={self.a_x:.6g}, a_y={self.a_y:.6g})"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        # Nazarathy/Shamir V[a_x, a_y]f(x, y) = sqrt(a_x*a_y) *
        # f(a_x*x, a_y*y) (unitary scaling).  The field's coordinate
        # support contracts by 1/a in each axis; the output grid
        # therefore covers a smaller physical extent at finer pitch.
        # On a fixed-N grid:
        #
        #   new_dx = dx / a_x,   new_dy = dy / a_y
        #
        # and at output pixel j (physical position p_j = (j-N/2)*new_dx)
        # the field value is
        #
        #   E_out[j] = sqrt(a_x*a_y) * E_in(a_x * p_j) = sqrt(a_x*a_y) * E_in[j]
        #
        # because a_x * p_j = a_x * (j-N/2) * dx/a_x = (j-N/2) * dx,
        # which is exactly the input grid position at index j.  No
        # interpolation is needed -- the algebra collapses to a
        # uniform amplitude rescale plus a grid-pitch announcement.
        new_dx = dx / self.a_x
        new_dy = dy / self.a_y
        prefactor = float(np.sqrt(self.a_x * self.a_y))
        # Preserve dtype: a complex64 input should stay complex64.
        out_dtype = E.dtype
        E_out = (prefactor * E).astype(out_dtype, copy=False)
        return E_out, new_dx, new_dy


# ---------------------------------------------------------------------------
# FourierTransform
# ---------------------------------------------------------------------------


class FourierTransform(Operator):
    """Physical optical Fourier transform (3-stage ``2f-lens-2f`` setup).

    ABCD (3-stage ``FreeSpace(f) * ThinLens(f) * FreeSpace(f)``)::

        [[0,    f],
         [-1/f, 0]]

    Parameters
    ----------
    f_focal : float
        Focal length of the Fourier lens [m].  Must be positive and
        finite.

    Notes
    -----
    Implements the canonical 3-stage Fourier-transforming geometry
    of Goodman §5.2 ("Fourier Transforming Properties of a Thin
    Lens"): the input plane sits one focal length before a lens
    of focal length ``f``, and the output plane is read one focal
    length after the lens.  In this geometry the field at the
    output plane is the **pure** scaled optical Fourier transform
    of the input field with no residual quadratic-phase factor --
    only this 3-stage geometry achieves that on both ABCD and
    field axes simultaneously.

    Realization
    -----------
    The v4.15.2 ``_apply`` implementation invokes the literal
    3-stage chain ``FreeSpace(f) -> ThinLens(f) -> FreeSpace(f)``
    by delegating to those primitives.  Earlier (v4.15.1) shipped a
    2-stage shortcut ``ThinLens(f) -> fresnel_propagate(f)`` which
    matched the 3-stage ABCD but left a residual
    ``exp(+i*k/(2f)*r^2)`` quadratic phase on the output -- the
    field-vs-ABCD inconsistency closed by audit P1-NEW-A.

    Performance note: the 3-stage path runs one additional Fresnel
    propagation compared to the 2-stage shortcut.  For a single
    Fourier transform the cost is ~2 FFTs (the two FreeSpace
    legs) plus one phase multiply.  Users who want the 2-stage
    ``ThinLens(f) -> propagate(f)`` "lens-then-propagate"
    hardware semantics (back-focal-plane shortcut, with the
    output-plane quadratic phase) can compose the operators
    directly as ``ThinLens(f) * FreeSpace(f)`` -- this is a
    genuine 2-stage chain whose ABCD is ``[[1, f], [-1/f, 0]]``
    (not ``[[0, f], [-1/f, 0]]``) and whose field carries the
    expected output-plane quadratic phase.

    References
    ----------
    Goodman, J. W., *Introduction to Fourier Optics*, 4th ed.,
    §5.2.3 "Fourier Transforming Properties of a Thin Lens"
    (W. H. Freeman, 2017).
    """

    def __init__(self, f_focal: float) -> None:
        try:
            ff = float(f_focal)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"FourierTransform: f_focal must be a real number, got "
                f"{f_focal!r}."
            ) from e
        if not np.isfinite(ff) or ff <= 0.0:
            raise ValueError(
                f"FourierTransform: f_focal must be positive and "
                f"finite (got {ff})."
            )
        self.f = ff
        # v4.15.2: the propagator backend for the two FreeSpace legs
        # is fixed to ``'auto'`` so the dispatcher chooses Fresnel
        # (the natural choice for z = f at typical optical scales);
        # users who want a specific kernel can compose
        # FreeSpace(f, method='asm') * ThinLens(f) * FreeSpace(f, method='asm')
        # by hand.
        self.method = 'auto'
        M = _validate_abcd(
            [[0.0, ff], [-1.0 / ff, 0.0]], 'FourierTransform',
        )
        self._abcd_x = M
        self._abcd_y = M.copy()

    def __repr__(self) -> str:
        return f"FourierTransform(f_focal={self.f:.6g})"

    def _apply(
        self,
        E: np.ndarray,
        *,
        dx: float,
        dy: float,
        wavelength: float,
    ) -> Tuple[np.ndarray, float, float]:
        # v4.15.2 (audit P1-NEW-A): invoke the literal 3-stage chain
        # ``FreeSpace(f) -> ThinLens(f) -> FreeSpace(f)`` so the
        # output field matches the ABCD claim (Goodman §5.2).  The
        # v4.15.1 2-stage shortcut (``ThinLens(f) -> fresnel(f)``)
        # left a residual ``exp(+i*k/(2f)*r^2)`` quadratic phase on
        # the output, so users composing ``FourierTransform(f)`` with
        # phase-sensitive downstream operators got different fields
        # than the equivalent 3-stage chain.
        #
        # v4.15.3 (audit P1-NEW-F1-1): the two FreeSpace legs below
        # carry ``self.method == 'auto'`` -- the dispatcher's regime-
        # dependent kernel pick.  ``FreeSpace._apply`` itself now
        # forces ``method='asm'`` whenever the input grid is
        # anamorphic (``dy != dx``), so this 3-stage chain is
        # anamorphic-safe by composition: passing a non-square input
        # grid no longer triggers the SAS-anamorphic
        # ``TypeError: sas_propagate() got an unexpected keyword
        # argument 'dy'`` crash that the v4.15.2 closure exposed.
        fs = FreeSpace(self.f, method=self.method)
        tl = ThinLens(self.f)
        E1, dx1, dy1 = fs._apply(
            E, dx=dx, dy=dy, wavelength=wavelength,
        )
        E2, dx2, dy2 = tl._apply(
            E1, dx=dx1, dy=dy1, wavelength=wavelength,
        )
        E3, dx3, dy3 = fs._apply(
            E2, dx=dx2, dy=dy2, wavelength=wavelength,
        )
        return E3, float(dx3), float(dy3)


__all__ = [
    'FreeSpace',
    'ThinLens',
    'CylindricalLens',
    'Magnify',
    'FourierTransform',
]
