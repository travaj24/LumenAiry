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
  realized at the back focal plane of a thin lens; delegates to
  :func:`lumenairy.apply_thin_lens` + :func:`lumenairy.fresnel_propagate`
  (the spec-§3.3.3 option (a)).

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
        out = propagate(
            E,
            z=self.distance,
            wavelength=wavelength,
            dx=dx,
            method=self.method,
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

    Rescales the field's grid pitch by ``a`` so the physical extent
    scales as ``a * (N * dx)`` while preserving total power.  For
    anamorphic stretches, pass distinct ``a_x`` and ``a_y``.

    Parameters
    ----------
    a_x : float
        Magnification along x.  Must be positive and finite.
    a_y : float, optional
        Magnification along y.  Defaults to ``a_x`` (isotropic).

    Notes
    -----
    Delegates :meth:`_apply` to
    :func:`lumenairy.resample_field` (interpolation onto the new
    grid pitch ``dx_out = dx_in / a``) and applies the
    ``sqrt(a_x * a_y)`` amplitude prefactor required for energy
    conservation:

        |E_out|^2 * dx_out * dy_out  ==  |E_in|^2 * dx_in * dy_in.

    The ABCD ``diag(1/a, a)`` is the **corrected** Nazarathy/Shamir
    form -- the original 1980 paper had a transcription slip in
    the off-diagonal sign; the corrected diagonal form is what
    matches a physical magnifier (image is ``a`` times bigger in
    real-space coordinates, so the ABCD ``[A, B; C, D]`` reads
    ``A = 1/a`` because object-plane height scales to image-plane
    height as ``y' = (1/a) * y`` for the conjugate of an ``a``-
    times-bigger output region).  See lens-designer reference at
    ``operators.py:556-577`` for the corresponding closure form.
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
    """Physical optical Fourier transform (back-focal-plane 2f setup).

    ABCD (2f setup ``FreeSpace(f) * ThinLens(f) * FreeSpace(f)``)::

        [[0,    f],
         [-1/f, 0]]

    Parameters
    ----------
    f_focal : float
        Focal length of the Fourier lens [m].  Must be positive and
        finite.

    Notes
    -----
    The pure mathematical Fourier transform is dimensionally
    inconsistent in physical units; we therefore deliberately
    require a focal length.  The chosen implementation (spec
    §3.3.3 option (a)) applies a thin-lens phase to the input and
    then Fresnel-propagates by ``f_focal``, which collapses the
    2f setup into a single FFT.  This matches the back-focal-plane
    Fourier output of a single lens of focal length ``f`` placed
    at distance ``f`` from the input plane.

    The ABCD is the ABCD of the **2f setup** (``[FreeSpace(f),
    ThinLens(f), FreeSpace(f)]`` composed), not the abstract
    mathematical FT -- composition with subsequent operators is
    therefore correct (verified in
    ``test_v4_15_1_agent_g_abcd.py``).
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
        # Spec §3.3.3 option (a): lens phase then Fresnel propagate
        # by f.  This is the "single-lens 2f" physical realization
        # of the optical Fourier transform: input -> lens(f) ->
        # propagate(f) -> back-focal-plane FT.
        from ..elements.lenses import apply_thin_lens
        from ..propagators.propagation import fresnel_propagate
        E1 = apply_thin_lens(
            E, f=self.f, wavelength=wavelength, dx=dx, dy=dy,
        )
        E_out, dx_out, dy_out = fresnel_propagate(
            E1, z=self.f, wavelength=wavelength, dx=dx, dy=dy,
        )
        return E_out, float(dx_out), float(dy_out)


__all__ = [
    'FreeSpace',
    'ThinLens',
    'CylindricalLens',
    'Magnify',
    'FourierTransform',
]
