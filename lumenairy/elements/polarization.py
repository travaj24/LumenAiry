"""
Vector Optics / Polarization Module
====================================

Jones-vector representation for polarized optical fields. Extends the
scalar field functions to 2-component (Ex, Ey) vector fields.

Main class: :class:`JonesField` — wraps two scalar fields and provides
propagation, optical element application, and polarization analysis.

Polarization elements provided:
    - apply_polarizer
    - apply_waveplate (HWP, QWP, arbitrary retardance)
    - apply_jones_matrix (arbitrary 2×2 transformation)
    - apply_rotator
    - apply_polarizing_beam_splitter (two orthogonal output ports)

Sources:
    - create_linear_polarized
    - create_circular_polarized
    - create_elliptical_polarized

Analysis:
    - stokes_parameters
    - degree_of_polarization
    - polarization_ellipse

Convention: Jones vectors written as column vectors [Ex, Ey]^T.
Angles measured from +x axis, counter-clockwise.

All scalar propagators and optical elements can be applied to a
JonesField by calling its methods, which dispatch to the underlying
scalar functions for each component independently (Ex and Ey propagate
or transform identically through non-polarizing elements).

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

# Scalar propagation and element functions (dispatched per component)
from ..propagators.propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_batch,
    angular_spectrum_propagate_tilted,
    fraunhofer_propagate,
    fresnel_propagate,
    scalable_angular_spectrum_propagate,
)
from .elements import (
    apply_aperture,
    apply_mask,
    apply_mirror,
)
from .lenses import (
    apply_real_lens,
    apply_spherical_lens,
    apply_thin_lens,
)

# =============================================================================
# JONES FIELD CLASS
# =============================================================================

class JonesField:
    """
    A two-component (Ex, Ey) complex field representing a polarized
    optical beam on a discrete 2D grid.

    Attributes
    ----------
    Ex : ndarray (complex, N×N)
        Electric field component along x.
    Ey : ndarray (complex, N×N)
        Electric field component along y.
    dx : float
        Grid spacing in x [m].
    dy : float
        Grid spacing in y [m]. Defaults to dx.

    Notes
    -----
    Non-polarizing optical elements (lenses, apertures, free-space
    propagation) act identically on Ex and Ey. The corresponding methods
    on :class:`JonesField` simply dispatch the scalar functions to each
    component.

    Polarization-dependent elements (polarizers, waveplates, Jones
    matrices) mix Ex and Ey according to their 2×2 Jones matrix.

    Warnings
    --------
    v5.4.6 (audit P3-18): the "non-polarizing" per-component dispatch is
    exact only inside the paraxial, low-AOI envelope.  It does NOT model:

    * s/p Fresnel splitting -- ``apply_real_lens(fresnel=True)`` /
      ``slant_correction=True`` at large AOI (it applies an s/p-averaged
      unpolarized power coefficient; a UserWarning is raised).
    * the longitudinal Ez component / Debye-Wolf depolarization at high NA
      (NA > ~0.3) -- use
      :func:`lumenairy.propagators.vector_diffraction.richards_wolf_focus`.
    * the polarization-basis rotation at large carrier tilt in
      :meth:`propagate_tilted` (paraxial / small-tilt only).
    * circular-polarization handedness inversion on reflection in
      :meth:`apply_mirror`.
    """

    def __init__(self, Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: Optional[float] = None) -> None:
        if Ex.shape != Ey.shape:
            raise ValueError(f"Ex and Ey must have the same shape, "
                             f"got {Ex.shape} and {Ey.shape}")
        # v4.14.2 (P1-NEW-8): validate that ``Ex`` / ``Ey`` are 2-D
        # and that ``dx`` / ``dy`` are positive finite reals.  Pre-
        # v4.14.2 a 1-D field or a non-positive pitch propagated all
        # the way to the FFT in :meth:`propagate`, which raised an
        # opaque shape / value error far from the construction site.
        # Match the input-guard pattern used in
        # :func:`_validate_grid_params` in
        # :mod:`lumenairy.propagators.propagation`.
        if Ex.ndim != 2:
            raise ValueError(
                f"JonesField: Ex / Ey must be 2-D arrays; got "
                f"Ex.ndim={Ex.ndim} with shape {Ex.shape}.  "
                f"Polarized propagation expects a (Ny, Nx) grid.")
        dx_f = float(dx)
        if not np.isfinite(dx_f) or dx_f <= 0.0:
            raise ValueError(
                f"JonesField: dx must be a positive finite real "
                f"[m]; got dx={dx!r}.  (LumenAiry uses metres; for "
                f"a 2 um pitch pass dx=2e-6, not 2.)")
        if dy is not None:
            dy_f = float(dy)
            if not np.isfinite(dy_f) or dy_f <= 0.0:
                raise ValueError(
                    f"JonesField: dy must be a positive finite real "
                    f"[m] or None; got dy={dy!r}.")
        # 4.11.2: ``dtype=complex`` aliases to ``complex128`` and
        # silently promoted complex64 inputs.  Preserve the caller's
        # dtype if it's already complex; otherwise cast through the
        # global default (which honours precision='single').
        Ex_arr = np.asarray(Ex)
        Ey_arr = np.asarray(Ey)
        if np.iscomplexobj(Ex_arr) and np.iscomplexobj(Ey_arr):
            self.Ex = Ex_arr
            self.Ey = Ey_arr
        else:
            from ..propagators.propagation import get_default_complex_dtype
            cdt = get_default_complex_dtype()
            self.Ex = Ex_arr.astype(cdt)
            self.Ey = Ey_arr.astype(cdt)
        self.dx = dx_f
        self.dy = float(dy) if dy is not None else dx_f

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.Ex.shape

    def intensity(self) -> np.ndarray:
        """Total intensity |Ex|^2 + |Ey|^2."""
        return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

    def power(self) -> float:
        """Total integrated power."""
        return float(np.sum(self.intensity()) * self.dx * self.dy)

    def copy(self) -> 'JonesField':
        """Return a deep copy of the field."""
        return JonesField(self.Ex.copy(), self.Ey.copy(), self.dx, self.dy)

    # ------------------------------------------------------------------
    # Polarization analysis (bound forms of the module-level helpers)
    # ------------------------------------------------------------------

    def stokes_parameters(self) -> Dict[str, np.ndarray]:
        """Per-pixel Stokes parameters dict ``{'S0', 'S1', 'S2', 'S3'}``.

        Equivalent to module-level :func:`stokes_parameters`; exposed
        as a bound method so JonesField pipelines can chain
        ``jf.propagate(...).stokes_parameters()`` without round-tripping
        through the scalar API.
        """
        return stokes_parameters(self)

    def degree_of_polarization(self) -> np.ndarray:
        """Per-pixel DOP map.  Equivalent to module-level
        :func:`degree_of_polarization`."""
        return degree_of_polarization(self)

    def polarization_ellipse(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel polarization-ellipse ``(orientation, ellipticity)``.

        Equivalent to module-level :func:`polarization_ellipse`.
        """
        return polarization_ellipse(self)

    # ------------------------------------------------------------------
    # Propagation (scalar functions applied to each component)
    # ------------------------------------------------------------------

    # Below this grid size the batched-FFT dispatch overhead exceeds
    # the savings from sharing an H build, so JonesField.propagate
    # falls back to two sequential ASM calls (the second hits the H
    # cache too, so it's essentially free).  Empirically determined
    # on a 6-core CPU; adjust via :func:`set_jones_batch_threshold`.
    _BATCH_PROPAGATE_MIN_N = 512

    def propagate(self, z: float, wavelength: float, bandlimit: bool = True) -> 'JonesField':
        """Propagate via the angular spectrum method.

        For grids at or above ``_BATCH_PROPAGATE_MIN_N`` (default 512)
        uses :func:`angular_spectrum_propagate_batch` so Ex/Ey share a
        single H build + one batched FFT pair.  Below that the
        sequential path is faster because FFT-call dispatch dominates
        compute -- both components hit the H cache on the second call
        anyway.
        """
        same_layout = (self.Ex.shape == self.Ey.shape
                       and self.Ex.dtype == self.Ey.dtype)
        N_max = max(self.Ex.shape) if self.Ex.ndim >= 2 else 0
        if same_layout and N_max >= self._BATCH_PROPAGATE_MIN_N:
            stack = np.stack([self.Ex, self.Ey], axis=0)
            out = angular_spectrum_propagate_batch(
                stack, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
            self.Ex, self.Ey = out[0], out[1]
        else:
            self.Ex = angular_spectrum_propagate(
                self.Ex, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
            self.Ey = angular_spectrum_propagate(
                self.Ey, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
        return self

    def propagate_tilted(
        self,
        z: float,
        wavelength: float,
        tilt_x: float = 0,
        tilt_y: float = 0,
        bandlimit: bool = True,
    ) -> 'JonesField':
        """Propagate via off-axis ASM.

        v5.4.6 (audit P3-20): valid in the paraxial / small-tilt regime
        (|tilt| < ~5 deg).  Ex and Ey are dispatched with the same carrier
        tilt and NO rotation of the (Ex, Ey) basis into the tilted
        propagation frame, so at larger tilts the implied transverse
        polarization basis and the (small) Ez component are not modeled.
        """
        self.Ex = angular_spectrum_propagate_tilted(
            self.Ex, z, wavelength, self.dx, self.dy,
            tilt_x=tilt_x, tilt_y=tilt_y, bandlimit=bandlimit)
        self.Ey = angular_spectrum_propagate_tilted(
            self.Ey, z, wavelength, self.dx, self.dy,
            tilt_x=tilt_x, tilt_y=tilt_y, bandlimit=bandlimit)
        return self

    def propagate_fresnel(self, z: float, wavelength: float) -> 'JonesField':
        """Propagate via single-FFT Fresnel. Returns new grid spacings."""
        self.Ex, dx_out, dy_out = fresnel_propagate(
            self.Ex, z, wavelength, self.dx, self.dy)
        self.Ey, _, _ = fresnel_propagate(
            self.Ey, z, wavelength, self.dx, self.dy)
        self.dx = dx_out
        self.dy = dy_out
        return self

    def propagate_fraunhofer(self, z: float, wavelength: float) -> 'JonesField':
        """Propagate to the far-field via Fraunhofer. Returns new grid spacings."""
        self.Ex, dx_out, dy_out = fraunhofer_propagate(
            self.Ex, z, wavelength, self.dx, self.dy)
        self.Ey, _, _ = fraunhofer_propagate(
            self.Ey, z, wavelength, self.dx, self.dy)
        self.dx = dx_out
        self.dy = dy_out
        return self

    def sas_propagate(
        self,
        z: float,
        wavelength: float,
        pad: int = 2,
        skip_final_phase: bool = False,
    ) -> 'JonesField':
        """Propagate via the Scalable Angular Spectrum Method.

        Applies :func:`scalable_angular_spectrum_propagate` to ``Ex``
        and ``Ey`` independently.  Both components share the same
        output grid (same ``z``, ``wavelength``, ``dx``, ``pad``), so
        this is the polarization-aware analogue of a scalar SAS call.
        The field's ``dx`` / ``dy`` are updated to the new output pitch
        (``lambda * z / (pad * N * dx)``) so downstream element calls
        see the correct coordinates.

        Parameters
        ----------
        z : float
            Propagation distance [m].
        wavelength : float
            Wavelength [m].
        pad : int, default 2
            Zero-padding factor for the SAS kernel.
        skip_final_phase : bool, default False
            Skip the outer quadratic phase; intensity is correct but
            field phase is not.

        Returns
        -------
        self
            Modified in place (same convention as the other propagate
            methods on this class).
        """
        if self.dx != self.dy:
            raise ValueError(
                "JonesField.sas_propagate requires dx == dy "
                f"(got dx={self.dx}, dy={self.dy}).  SAS assumes a "
                f"square grid pitch.")
        self.Ex, dx_out, dy_out = scalable_angular_spectrum_propagate(
            self.Ex, z, wavelength, self.dx,
            pad=pad, skip_final_phase=skip_final_phase)
        self.Ey, _, _ = scalable_angular_spectrum_propagate(
            self.Ey, z, wavelength, self.dx,
            pad=pad, skip_final_phase=skip_final_phase)
        self.dx = dx_out
        self.dy = dy_out
        return self

    # ------------------------------------------------------------------
    # Non-polarizing elements (dispatched per component)
    # ------------------------------------------------------------------

    def apply_thin_lens(self, f: float, wavelength: float, **kwargs: Any) -> 'JonesField':
        """Apply a thin lens to both components.

        v5.4.6 (audit P3-21): scalar per-component dispatch is exact in the
        paraxial regime.  Under high-NA focusing (NA > ~0.3, f/# < ~1.7)
        the vectorial Ez component and Debye-Wolf depolarization are NOT
        modeled; use ``vector_diffraction.richards_wolf_focus`` instead.
        Same caveat applies to :meth:`apply_spherical_lens`.
        """
        self.Ex = apply_thin_lens(self.Ex, f=f, wavelength=wavelength, dx=self.dx, dy=self.dy, **kwargs)
        self.Ey = apply_thin_lens(self.Ey, f=f, wavelength=wavelength, dx=self.dx, dy=self.dy, **kwargs)
        return self

    def apply_spherical_lens(self, **kwargs: Any) -> 'JonesField':
        # GL-nit (AUDIT_GLASS_POLARIZATION): a missing ``wavelength`` raised a
        # bare ``KeyError('wavelength')``; raise a TypeError naming the missing
        # argument (matching how the sibling methods take it positionally).
        if 'wavelength' not in kwargs:
            raise TypeError(
                "JonesField.apply_spherical_lens() missing required keyword "
                "argument 'wavelength'.")
        wl = kwargs['wavelength']
        rest = {k: v for k, v in kwargs.items() if k != 'wavelength'}
        self.Ex = apply_spherical_lens(self.Ex, wavelength=wl,
                                        dx=self.dx, dy=self.dy, **rest)
        self.Ey = apply_spherical_lens(self.Ey, wavelength=wl,
                                        dx=self.dx, dy=self.dy, **rest)
        return self

    def apply_real_lens(
        self,
        prescription: Dict[str, Any],
        wavelength: float,
        bandlimit: bool = True,
        slant_correction: bool = False,
        fresnel: bool = False,
        absorption: bool = False,
    ) -> 'JonesField':
        """Apply a multi-surface real lens to both components.

        See :func:`lumenairy.lenses.apply_real_lens` for parameter
        documentation.

        Warning
        -------
        v5.4.6 (audit P2-3 / P3-25): per-component dispatch is only
        non-polarizing for the default (phase-screen) path.  With
        ``fresnel=True`` (or ``slant_correction=True`` at large AOI) the
        scalar kernel applies an s/p-AVERAGED unpolarized power
        coefficient ``T_eff = 0.5(|t_s|^2 + |t_p|^2)`` to each component
        independently, so the s-vs-p Fresnel split is collapsed BEFORE the
        JonesField sees the result (a 100%-x-polarized beam through a
        Brewster surface is then physically wrong).  A ``UserWarning`` is
        emitted in that case; a rigorous per-pixel Jones-matrix Fresnel
        path is queued for v5.5.
        """
        if fresnel or slant_correction:
            import warnings
            warnings.warn(
                "JonesField.apply_real_lens: fresnel/slant_correction apply "
                "an s/p-averaged unpolarized power coefficient per component, "
                "collapsing the polarization-dependent Fresnel split.  The "
                "result is only correct for unpolarized / low-AOI input; for "
                "a polarized high-AOI pipeline this is physically wrong.",
                UserWarning, stacklevel=2)
        # v5.4.6 (audit P2-4): forward dy=self.dy so anamorphic JonesFields
        # (dx != dy) keep the correct y-axis pitch (apply_thin_lens already did).
        self.Ex = apply_real_lens(
            self.Ex, prescription=prescription, wavelength=wavelength,
            dx=self.dx, dy=self.dy,
            bandlimit=bandlimit, slant_correction=slant_correction,
            fresnel=fresnel, absorption=absorption)
        self.Ey = apply_real_lens(
            self.Ey, prescription=prescription, wavelength=wavelength,
            dx=self.dx, dy=self.dy,
            bandlimit=bandlimit, slant_correction=slant_correction,
            fresnel=fresnel, absorption=absorption)
        return self

    def apply_mirror(self, wavelength: float, **kwargs: Any) -> 'JonesField':
        """Apply an ideal (phase-only) mirror to both components.

        v5.4.6 (audit P3-19): the scalar mirror applies only the sag phase
        (no metallic R_s/R_p split), so per-component dispatch is correct
        for the phase part -- use ``apply_jones_matrix`` for a metallic
        coating.  Note that physical reflection inverts circular-
        polarization handedness (S3 -> -S3) because the propagation
        direction reverses; this method leaves the (Ex, Ey) basis
        unchanged, so the caller owns any handedness/frame convention.
        """
        # v5.4.6 (audit P2-4): forward dy=self.dy for anamorphic grids.
        self.Ex = apply_mirror(self.Ex, wavelength, self.dx, dy=self.dy, **kwargs)
        self.Ey = apply_mirror(self.Ey, wavelength, self.dx, dy=self.dy, **kwargs)
        return self

    def apply_aperture(
        self,
        shape: str = 'circular',
        params: Optional[Dict[str, Any]] = None,
        xc: float = 0,
        yc: float = 0,
    ) -> 'JonesField':
        # v5.4.6 (audit P2-4): forward dy=self.dy for anamorphic grids.
        self.Ex = apply_aperture(self.Ex, self.dx, shape=shape, params=params,
                                 xc=xc, yc=yc, dy=self.dy)
        self.Ey = apply_aperture(self.Ey, self.dx, shape=shape, params=params,
                                 xc=xc, yc=yc, dy=self.dy)
        return self

    def apply_mask(self, mask: np.ndarray) -> 'JonesField':
        self.Ex = apply_mask(self.Ex, mask)
        self.Ey = apply_mask(self.Ey, mask)
        return self


# =============================================================================
# POLARIZATION-DEPENDENT ELEMENTS
# =============================================================================

# v4.14.3 (P1-NEW-2): canonical "argument-not-supplied" sentinel for the
# five polarization helpers' (``angle``, ``angle_deg``) conflict-detection
# branch.  Matches the v4.14.1 ``_ZeroApertureMaskSentinel`` ``__slots__=()``
# singleton pattern documented in ``optimize/core.py:2022-2031``.  We use
# a dedicated class (rather than the more common ``_UNSET = object()``)
# so the ``repr`` is informative if it ever leaks into an error message,
# and so the sentinel can carry a ``__reduce__`` if it ever needs to
# cross a pickle boundary (no current callsite does -- the sentinel is
# function-local default-value plumbing).
# v4.15.1 (Agent E): now inherits from ``_deprecation._Sentinel`` to
# share the singleton-name registry + pickle-safe ``__reduce__``
# protocol.  See ``_deprecation.py`` and the migration note on
# ``_ZeroApertureMaskSentinel`` in ``optimize/core.py``.
from .._deprecation import _Sentinel as _Sentinel


class _AngleUnsetSentinel(_Sentinel):
    """Singleton sentinel meaning "the caller did not supply this angle".

    Used by :func:`apply_polarizer`, :func:`apply_waveplate`,
    :func:`apply_half_wave_plate`, :func:`apply_quarter_wave_plate`, and
    :func:`apply_rotator` to distinguish "user explicitly passed
    ``angle=0.0``" from "user passed nothing".  The former plus an
    ``angle_deg=N`` second kwarg with ``N != 0`` is a conflict and must
    raise; the latter accepts ``angle_deg=N`` silently.  Pre-v4.14.3
    the conflict check tested ``if angle != 0.0`` and so silently
    accepted ``angle=0.0, angle_deg=90`` as a 90-degree rotation,
    discarding the explicit ``angle=0`` half of the request.
    """

    __slots__ = ()

    def __init__(self) -> None:
        # Use the existing repr-friendly name as the singleton key.
        super().__init__('_ANGLE_UNSET')


_ANGLE_UNSET = _AngleUnsetSentinel()


def _resolve_angle(
    func_name: str,
    angle: Any,
    angle_deg: Optional[float],
) -> float:
    """Resolve ``(angle, angle_deg)`` to a single radian value.

    Shared helper for the 5 ``apply_*`` polarization helpers.  Returns
    ``0.0`` when neither is supplied; otherwise returns the (radian)
    value of whichever was supplied.  If both are supplied with
    disagreeing numeric values raises :class:`ValueError`; agreement
    is checked to ``atol=1e-12`` rad.
    """
    angle_supplied = angle is not _ANGLE_UNSET
    if angle_deg is None:
        return float(angle) if angle_supplied else 0.0
    angle_from_deg = float(np.radians(angle_deg))
    if angle_supplied:
        # Both supplied -- they must agree to floating tolerance.
        if not np.isclose(float(angle), angle_from_deg,
                          rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{func_name}: conflicting angle specification -- "
                f"angle={angle} rad and angle_deg={angle_deg} "
                f"(=> {angle_from_deg} rad) disagree.  Pass only one.")
    return angle_from_deg


def apply_jones_matrix(field: 'JonesField', matrix: Union[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]) -> 'JonesField':
    """
    Apply an arbitrary 2×2 Jones matrix to a JonesField.

    The transformation is:

        [Ex']   [J00 J01] [Ex]
        [Ey'] = [J10 J11] [Ey]

    Parameters
    ----------
    field : JonesField
        Input polarized field.
    matrix : ndarray (complex, 2×2) or callable
        The Jones matrix. If callable, must accept (x, y) meshgrids and
        return a spatially-varying Jones array.  v5.4.6 (audit P3-26):
        BOTH layouts are accepted -- the canonical ``(2, 2, Ny, Nx)`` and
        the ``(Ny, Nx, 2, 2)`` layout that ``meshgrid``-based callables
        produce naturally (the latter is transposed internally via
        ``np.moveaxis``).

    Returns
    -------
    JonesField
        Transformed field (modified in place and returned).
    """
    if callable(matrix):
        x = (np.arange(field.shape[1]) - field.shape[1] / 2) * field.dx
        y = (np.arange(field.shape[0]) - field.shape[0] / 2) * field.dy
        X, Y = np.meshgrid(x, y)
        J = matrix(X, Y)  # expect (2, 2, N, N)
        # 4.10.2: validate shape.  Pre-4.10.2 silently broadcast any
        # shape and produced wrong answers without an error.  The
        # expected layout is (2, 2, Ny, Nx) so that J[0,0] is a 2-D
        # array matching field.Ex.shape.
        J = np.asarray(J)
        expected = (2, 2, field.shape[0], field.shape[1])
        if J.shape != expected:
            # Permit the swapped layout (Ny, Nx, 2, 2) that some
            # callers produce naturally with meshgrid -- transpose.
            if J.shape == (field.shape[0], field.shape[1], 2, 2):
                J = np.moveaxis(J, (-2, -1), (0, 1))
            else:
                raise ValueError(
                    f"apply_jones_matrix: callable returned shape "
                    f"{J.shape}, expected {expected} (2x2 Jones matrix "
                    f"with per-pixel spatial extent).")
        Ex_new = J[0, 0] * field.Ex + J[0, 1] * field.Ey
        Ey_new = J[1, 0] * field.Ex + J[1, 1] * field.Ey
    else:
        J = np.asarray(matrix, dtype=complex)
        if J.shape != (2, 2):
            raise ValueError(
                f"apply_jones_matrix: matrix array shape {J.shape}, "
                f"expected (2, 2).  Use a callable for spatially-"
                f"varying matrices.")
        Ex_new = J[0, 0] * field.Ex + J[0, 1] * field.Ey
        Ey_new = J[1, 0] * field.Ex + J[1, 1] * field.Ey

    field.Ex = Ex_new
    field.Ey = Ey_new
    return field


def apply_polarizer(
    field: 'JonesField',
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
) -> 'JonesField':
    """
    Apply an ideal linear polarizer at the specified transmission angle.

    Parameters
    ----------
    field : JonesField
        Input field.
    angle : float, default 0
        Transmission axis angle [radians], measured from +x axis.
    angle_deg : float, optional
        Transmission axis angle in degrees.  When supplied it is used
        in place of the radian ``angle``; if BOTH are given and disagree
        (``angle != radians(angle_deg)``) a ``ValueError`` is raised (see
        Raises).  4.7+: ``_deg`` is the canonical user-facing angle unit.

    Returns
    -------
    JonesField

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` are supplied with values
        that disagree (i.e. ``angle != radians(angle_deg)``).  Passing
        ``angle_deg`` alone is accepted.  v4.14.3 (P1-NEW-2): symmetric
        with :func:`apply_rotator` / :func:`apply_waveplate` and the
        half/quarter-wave-plate wrappers.
    """
    angle = _resolve_angle('apply_polarizer', angle, angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    # Projection matrix for linear polarizer
    J = np.array([[c*c,  c*s],
                  [c*s,  s*s]], dtype=complex)
    return apply_jones_matrix(field, J)


def apply_waveplate(
    field: 'JonesField',
    retardance: float,
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
) -> 'JonesField':
    """
    Apply a waveplate (linear retarder) with arbitrary retardance.

    Parameters
    ----------
    field : JonesField
        Input field.
    retardance : float
        Phase retardance between fast and slow axes [radians].
        - pi/2 = quarter-wave plate
        - pi   = half-wave plate
    angle : float, default 0
        Fast-axis angle [radians], measured from +x axis.
    angle_deg : float, optional
        Fast-axis angle in degrees.  When supplied it is used in place
        of ``angle``; if both are given and disagree a ``ValueError`` is
        raised (see Raises).

    Returns
    -------
    JonesField

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` are supplied with values
        that disagree (i.e. ``angle != radians(angle_deg)``).  Passing
        ``angle_deg`` alone is accepted.  v4.14.3 (P1-NEW-2): symmetric
        with :func:`apply_rotator` / :func:`apply_polarizer` and the
        half/quarter-wave-plate wrappers.

    Notes
    -----
    The Jones matrix for a waveplate with fast axis at angle theta and
    retardance phi (under the library's exp(-i omega t) convention) is::

        J = R(theta) * diag(1, exp(+i*phi)) * R(-theta)

    where R(theta) = [[cos t, -sin t], [sin t, cos t]] is the 2D
    rotation matrix.  Under the library-wide ``exp(-i omega t)`` /
    ``exp(+i k n z)`` field convention (the one berreman.py declares
    "PUBLIC throughout" and the propagators' ``exp(+i k OPL)`` phase
    uses), traversing a thickness ``d`` of index ``n`` multiplies the
    phasor by ``exp(+i k0 n d)``, so the SLOW axis (larger ``n``,
    arrives *later*: a time delay ``tau`` contributes
    ``exp(+i omega tau)`` when the carrier is ``exp(-i omega t)``)
    accumulates POSITIVE relative phase ``exp(+i*phi)``.

    BEHAVIOR CHANGE (audit P2-15, post-v5.17.0): from v4.7 through
    v5.17.0 this function used ``exp(-i*retardance)`` on the slow axis
    -- the ``exp(+i omega t)`` (EE-convention) sign -- with a docstring
    that incorrectly attributed it to ``exp(-i omega t)``.  That made
    the Jones-element family the CONJUGATE of the library's own
    rigorous solver Jones: ``berreman_jones_1d`` on a uniaxial
    quarter-wave slab (``eps = diag(no^2, ne^2, no^2)``,
    ``d = lambda/(4 (ne - no))``, index-matched half-spaces) returns
    transmission Jones ``diag(e^{i k0 no d}, e^{i k0 ne d})`` --
    slow-relative-fast phase ``+pi/2`` -- and the same slab with its
    fast axis at +45 deg maps x-pol to ``Ey/Ex = -i`` (S3 = -1),
    while the pre-fix ``apply_waveplate`` gave ``Ey/Ex = +i``
    (S3 = +1): circular handedness flipped between the element and
    solver families for the same physical device.  The element sign
    now matches the solver family (``berreman_jones_1d`` /
    ``BerremanStack`` / ``rcwa_jones_1d``), so solver-derived Jones
    matrices drop into JonesField pipelines without conjugation.

    Consequences: a QWP with fast axis at +45 deg on x-pol now yields
    S3 = -1 (``create_circular_polarized``'s 'left'); use fast axis
    at -45 deg for S3 = +1 ('right').  Half-wave plates are unaffected
    (``exp(+-i pi) = -1`` either way), as are all
    retardance-magnitude / intensity results.  The v5.4.6 (P3-22)
    "DECOUPLED, mutually consistent" note predates the Berreman /
    RCWA retarder Jones (v5.14.4) and is superseded by this
    cross-family alignment; see CONVENTIONS.md section 7.
    """
    angle = _resolve_angle('apply_waveplate', angle, angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    # Slow axis picks up POSITIVE relative phase exp(+i phi) under the
    # library's exp(-i omega t) / exp(+i k n z) convention -- matching
    # berreman_jones_1d / rcwa_jones_1d transmission Jones (audit
    # P2-15, post-v5.17.0).  History: pre-4.7 exp(+i phi); v4.7
    # flipped to exp(-i phi) (the EE-convention sign, misattributed
    # to exp(-i omega t)); the audit fix restores exp(+i phi) so the
    # element family agrees with the solver family on handedness.
    e = np.exp(+1j * retardance)

    # R(theta) * diag(1, e) * R(-theta)  -- fast axis at angle theta,
    # slow axis perpendicular and delayed by `retardance` radians.
    J = np.array([
        [c*c + e*s*s,     c*s*(1 - e)],
        [c*s*(1 - e),     s*s + e*c*c],
    ], dtype=complex)
    return apply_jones_matrix(field, J)


def apply_half_wave_plate(
    field: 'JonesField',
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
) -> 'JonesField':
    """Convenience wrapper: half-wave plate (retardance = pi).

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` are supplied with values
        that disagree.  v4.14.3 (P1-NEW-2): symmetric with the other
        polarization helpers.
    """
    angle = _resolve_angle('apply_half_wave_plate', angle, angle_deg)
    return apply_waveplate(field, np.pi, angle)


def apply_quarter_wave_plate(
    field: 'JonesField',
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
) -> 'JonesField':
    """Convenience wrapper: quarter-wave plate (retardance = pi/2).

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` are supplied with values
        that disagree.  v4.14.3 (P1-NEW-2): symmetric with the other
        polarization helpers.
    """
    angle = _resolve_angle('apply_quarter_wave_plate', angle, angle_deg)
    return apply_waveplate(field, np.pi / 2, angle)


def apply_rotator(
    field: 'JonesField',
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
) -> 'JonesField':
    """
    Apply a polarization rotator (e.g. Faraday rotator).

    Rotates the polarization state by the given angle without changing
    its ellipticity.

    Parameters
    ----------
    field : JonesField
        Input field.
    angle : float, default 0
        Rotation angle [radians].
    angle_deg : float, optional
        Rotation angle in degrees.  When supplied it is used in place of
        the radian ``angle``; if both are given and disagree a
        ``ValueError`` is raised (see Raises).  4.7+ convention: ``_deg``
        is the canonical user-facing angle unit.  Matches the
        ``apply_polarizer`` / ``apply_waveplate`` /
        ``apply_half_wave_plate`` / ``apply_quarter_wave_plate``
        signatures; pre-v4.14.2 ``apply_rotator`` was the only
        angle-taking helper missing this kwarg (sibling-gap caught
        by AUDIT_V4_14_1_2026_05_17 P1-NEW-7).

    Returns
    -------
    JonesField

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` are supplied with values
        that disagree (i.e. ``angle != radians(angle_deg)``).  Passing
        ``angle_deg`` alone is accepted.

        v4.14.3 (P1-NEW-2): the pre-v4.14.3 check used ``if angle !=
        0.0`` as a proxy for "angle was supplied", which silently
        accepted ``apply_rotator(field, angle=0.0, angle_deg=90)`` as
        a 90-degree rotation and discarded the explicit ``angle=0``
        half of the request.  v4.14.3 switches to the
        :data:`_ANGLE_UNSET` singleton sentinel so explicit zeros are
        distinguished from defaults; the same sentinel is now used by
        the four sibling helpers (``apply_polarizer``,
        ``apply_waveplate``, ``apply_half_wave_plate``,
        ``apply_quarter_wave_plate``) for API consistency.
    """
    angle = _resolve_angle('apply_rotator', angle, angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    J = np.array([[c, -s],
                  [s,  c]], dtype=complex)
    return apply_jones_matrix(field, J)


def apply_polarizing_beam_splitter(
    field: 'JonesField',
    angle: Any = _ANGLE_UNSET,
    *,
    angle_deg: Optional[float] = None,
    extinction_ratio: Optional[float] = None,
) -> 'Tuple[JonesField, JonesField]':
    """
    Split a field into two orthogonally-polarized output ports of a
    polarizing beam splitter (PBS).

    The TRANSMITTED port passes the linear polarization along the
    transmission axis (``angle``, the "p" component); the REFLECTED port
    carries the orthogonal polarization (``angle + 90 deg``, the "s"
    component).  For an ideal PBS the two ports are complementary
    projections, so power is conserved exactly
    (``|E_t|^2 + |E_r|^2 == |E_in|^2`` per pixel).

    Parameters
    ----------
    field : JonesField
        Input field.
    angle : float, default 0
        Transmission-axis angle [radians] from +x (the p-axis).
    angle_deg : float, optional
        Transmission-axis angle in degrees.  When supplied it is used in
        place of the radian ``angle``; if both are given and disagree a
        ``ValueError`` is raised (see Raises).  (The canonical ``_deg``
        convention used by the other polarization helpers.)
    extinction_ratio : float, optional
        Finite polarization extinction ratio (the wanted:unwanted POWER
        ratio in each port), modelling a real PBS's leakage.  ``None``
        (default) is the ideal PBS (infinite extinction).  A value of,
        e.g., ``1000`` lets ``1/1001`` of the wrong polarization's power
        leak into each port; power is still conserved between the two
        ports.

    Returns
    -------
    (transmitted, reflected) : tuple of JonesField
        The two output ports.

    Raises
    ------
    ValueError
        If both ``angle`` and ``angle_deg`` disagree, or if
        ``extinction_ratio`` is non-positive.

    Notes
    -----
    This models the PBS as a polarization *separator*: the reflected
    port's Jones vector is expressed in the same lab (x, y) basis as the
    input (no handedness flip).  If a specific reflection geometry needs
    the mirror coordinate flip, apply it to the returned reflected port.
    """
    angle = _resolve_angle('apply_polarizing_beam_splitter', angle, angle_deg)
    if extinction_ratio is not None and extinction_ratio <= 0:
        raise ValueError(
            "apply_polarizing_beam_splitter: extinction_ratio must be "
            "positive (the wanted:unwanted power ratio), got "
            f"{extinction_ratio}.")
    # Power leakage fraction of the wrong polarization into each port.
    leak = 0.0 if extinction_ratio is None else 1.0 / (1.0 + extinction_ratio)
    a = np.sqrt(1.0 - leak)   # amplitude of the wanted polarization
    b = np.sqrt(leak)         # amplitude of the leaked (wrong) polarization
    c = np.cos(angle)
    s = np.sin(angle)
    # J = R(angle) diag(major, minor) R(-angle); transmitted is strong along
    # the p (transmission) axis, reflected strong along the orthogonal s axis.
    Jt = np.array([[a * c * c + b * s * s, (a - b) * c * s],
                   [(a - b) * c * s,       a * s * s + b * c * c]], dtype=complex)
    Jr = np.array([[b * c * c + a * s * s, (b - a) * c * s],
                   [(b - a) * c * s,       b * s * s + a * c * c]], dtype=complex)
    # apply_jones_matrix mutates its field in place, so each output port must
    # operate on an independent copy of the input (otherwise the second call
    # would see the first port's already-transformed field).
    return (apply_jones_matrix(field.copy(), Jt),
            apply_jones_matrix(field.copy(), Jr))


# =============================================================================
# POLARIZED SOURCES
# =============================================================================

def create_linear_polarized(
    scalar_field: np.ndarray,
    dx: float,
    angle: float = 0.0,
    dy: Optional[float] = None,
) -> 'JonesField':
    """
    Create a linearly polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution (e.g. from create_gaussian_beam).
    dx : float
        Grid spacing [m].
    angle : float, default 0
        Polarization angle [radians] from +x axis.
        0 = x-polarized, pi/2 = y-polarized, pi/4 = diagonal.
    dy : float, optional

    Returns
    -------
    JonesField
    """
    Ex = scalar_field * np.cos(angle)
    Ey = scalar_field * np.sin(angle)
    return JonesField(Ex, Ey, dx, dy)


def create_circular_polarized(
    scalar_field: np.ndarray,
    dx: float,
    handedness: str = 'right',
    dy: Optional[float] = None,
) -> 'JonesField':
    """
    Create a circularly polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution.
    dx : float
        Grid spacing [m].
    handedness : {'right', 'left'}, default 'right'
        Handedness of the circular polarization, defined by S3 sign
        under the library's ``S3 = -2 Im(Ex Ey*)`` Stokes convention:
        - 'right' (RHC): Jones vector ``(1, +i)/sqrt(2)``; S3 = +1.
        - 'left'  (LHC): Jones vector ``(1, -i)/sqrt(2)``; S3 = -1.

        This matches ``apply_waveplate(QWP, fast axis at -45 deg)``
        acting on a linear x-polarized input (which produces
        ``(1, +i)/sqrt(2)`` up to a global phase under the library's
        ``exp(-i omega t)`` time convention) and the
        ``vector_diffraction.richards_wolf_focus`` circular-pol
        branch.  (Audit P2-15, post-v5.17.0: ``apply_waveplate`` was
        realigned to the Berreman/RCWA solver Jones, so a QWP with
        fast axis at **+45 deg** on x-pol now gives 'left' (S3 = -1);
        pre-fix it gave 'right'.)
    dy : float, optional

    Returns
    -------
    JonesField

    Notes
    -----
    4.11.1: the 4.10 "fix" to this function flipped the handedness
    branches so that 'right' produced ``(1, -i)/sqrt(2)``, which gave
    ``S3 = -1`` under the library's own Stokes formula and contradicted
    the hard-coded right-circular Jones vector in
    ``vector_diffraction.py``.  4.11.1 restores the pre-4.10 form
    where 'right' obeys ``S3 > 0``.

    Audit P2-15 (post-v5.17.0): ``apply_waveplate``'s retarder sign
    was flipped to match the Berreman/RCWA solver Jones (slow axis
    ``exp(+i*phi)``), so the QWP recipe that reproduces
    ``create_circular_polarized('right')`` on x-pol is now fast axis
    at **-45 deg** (pre-fix: +45 deg).  This function's own Jones
    vectors and its agreement with ``vector_diffraction.py`` are
    unchanged.
    """
    Ex = scalar_field / np.sqrt(2)
    if handedness.lower().startswith('r'):
        Ey = scalar_field * 1j / np.sqrt(2)
    else:
        Ey = scalar_field * (-1j) / np.sqrt(2)
    return JonesField(Ex, Ey, dx, dy)


def create_elliptical_polarized(
    scalar_field: np.ndarray,
    dx: float,
    ellipticity: float = 0.0,
    orientation: float = 0.0,
    dy: Optional[float] = None,
) -> 'JonesField':
    """
    Create an elliptically polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution.
    dx : float
        Grid spacing [m].
    ellipticity : float, default 0
        Ellipticity angle chi [radians]. 0 = linear, ±pi/4 = circular.
    orientation : float, default 0
        Major-axis angle psi [radians] from +x axis.
    dy : float, optional

    Returns
    -------
    JonesField

    Notes
    -----
    The Jones vector for an elliptical polarization is::

        [Ex]   [cos(psi)  -sin(psi)] [cos(chi)]
        [Ey] = [sin(psi)   cos(psi)] [i sin(chi)]
    """
    cp = np.cos(orientation)
    sp = np.sin(orientation)
    cc = np.cos(ellipticity)
    sc = np.sin(ellipticity)
    Ex = scalar_field * (cp * cc - 1j * sp * sc)
    Ey = scalar_field * (sp * cc + 1j * cp * sc)
    return JonesField(Ex, Ey, dx, dy)


# =============================================================================
# POLARIZATION ANALYSIS
# =============================================================================

def stokes_parameters(field: 'JonesField') -> Dict[str, np.ndarray]:
    """
    Compute the Stokes parameters (S0, S1, S2, S3) of a JonesField.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    S : dict
        Dictionary with keys 'S0', 'S1', 'S2', 'S3', each a 2D array.
        S0 = |Ex|^2 + |Ey|^2         (total intensity)
        S1 = |Ex|^2 - |Ey|^2         (horizontal vs vertical)
        S2 = 2*Re(Ex * conj(Ey))     (±45 deg linear)
        S3 = -2*Im(Ex * conj(Ey))    (circular, right-hand positive)
    """
    Ex = field.Ex
    Ey = field.Ey
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = -2 * np.imag(Ex * np.conj(Ey))
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}


def degree_of_polarization(field: 'JonesField') -> np.ndarray:
    """
    Compute the degree of polarization (DOP).

    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0

    For fully coherent fields from a single source, DOP = 1 everywhere
    (where S0 > 0). Values less than 1 indicate depolarization, which
    occurs only for partially coherent / incoherent sources or through
    depolarizing elements.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    dop : ndarray (real, N×N)
        Local degree of polarization (0 to 1).
    """
    S = stokes_parameters(field)
    total = np.sqrt(S['S1']**2 + S['S2']**2 + S['S3']**2)
    # Avoid division by zero
    dop = np.where(S['S0'] > 1e-30, total / np.maximum(S['S0'], 1e-30), 0.0)
    return dop


def polarization_ellipse(field: 'JonesField') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the polarization ellipse parameters (orientation, ellipticity)
    at each grid point.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    orientation : ndarray (real, N×N)
        Orientation angle psi of the polarization ellipse major axis
        [radians], from +x axis. Range: [-pi/2, pi/2].
    ellipticity : ndarray (real, N×N)
        Ellipticity angle chi [radians]. Range: [-pi/4, pi/4].
        - chi = 0: linear polarization
        - chi = ±pi/4: circular polarization

    Notes
    -----
    Relationship to Stokes parameters::

        tan(2*psi) = S2 / S1
        sin(2*chi) = S3 / sqrt(S1^2 + S2^2 + S3^2)
    """
    S = stokes_parameters(field)
    orientation = 0.5 * np.arctan2(S['S2'], S['S1'])
    total = np.sqrt(S['S1']**2 + S['S2']**2 + S['S3']**2)
    sin_2chi = np.where(total > 1e-30, S['S3'] / np.maximum(total, 1e-30), 0.0)
    sin_2chi = np.clip(sin_2chi, -1.0, 1.0)
    ellipticity = 0.5 * np.arcsin(sin_2chi)
    return orientation, ellipticity


# =============================================================================
# JONES-PUPIL -> STOKES / DOP HELPERS
# =============================================================================
# v5.4.7 (audit AUDIT_V5_4_6 #10): relocated here from
# ``lumenairy/ui/jones_pupil_dock.py`` (which imports PySide6 at module
# scope).  These are pure-NumPy helpers and belong in the polarization
# domain; living here lets CI exercise them without a Qt install.  The
# dock re-imports them so its behaviour is unchanged.

def jones_pupil_to_stokes_unpolarized(J: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-pixel Stokes maps from a Jones *pupil* under unpolarized input.

    A Jones pupil is the full 2x2 transfer matrix at each spatial point,
    so collapsing it to a Stokes image requires choosing the input
    polarization.  This uses the canonical UNPOLARIZED-input output Stokes
    (Mueller column 0 -- the intensity-average of the x-input and y-input
    output Stokes), with the 1/2 normalisation and the library's
    ``S3 = -2 Im(Ex conj Ey)`` sign (see CONVENTIONS.md section 7)::

        S0 =  0.5 (|J00|^2 + |J01|^2 + |J10|^2 + |J11|^2)
        S1 =  0.5 (|J00|^2 + |J01|^2 - |J10|^2 - |J11|^2)
        S2 =  Re(J00 conj(J10) + J01 conj(J11))
        S3 = -Im(J00 conj(J10) + J01 conj(J11))

    Parameters
    ----------
    J : ndarray (complex, Ny, Nx, 2, 2)
        Jones pupil (e.g. from ``compute_jones_pupil``).

    Returns
    -------
    dict with keys ``'S0'``, ``'S1'``, ``'S2'``, ``'S3'`` -- real (Ny, Nx).
    """
    J00 = J[..., 0, 0]
    J01 = J[..., 0, 1]
    J10 = J[..., 1, 0]
    J11 = J[..., 1, 1]
    a00 = np.abs(J00) ** 2
    a01 = np.abs(J01) ** 2
    a10 = np.abs(J10) ** 2
    a11 = np.abs(J11) ** 2
    S0 = 0.5 * (a00 + a01 + a10 + a11)
    S1 = 0.5 * (a00 + a01 - a10 - a11)
    S2 = np.real(J00 * np.conj(J10) + J01 * np.conj(J11))
    S3 = -np.imag(J00 * np.conj(J10) + J01 * np.conj(J11))
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}


def stokes_to_dop(stokes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-pixel DOP / DOLP / DOCP from a Stokes dict.

    ``DOP = sqrt(S1^2 + S2^2 + S3^2) / S0``,
    ``DOLP = sqrt(S1^2 + S2^2) / S0``, ``DOCP = |S3| / S0``.
    Pixels with ``S0 <= 1e-30`` (background) are set to 0.
    """
    S0 = stokes['S0']
    S1 = stokes['S1']
    S2 = stokes['S2']
    S3 = stokes['S3']
    safe = np.maximum(S0, 1e-30)
    mask = S0 > 1e-30
    dop = np.where(mask, np.sqrt(S1 ** 2 + S2 ** 2 + S3 ** 2) / safe, 0.0)
    dolp = np.where(mask, np.sqrt(S1 ** 2 + S2 ** 2) / safe, 0.0)
    docp = np.where(mask, np.abs(S3) / safe, 0.0)
    return {'DOP': dop, 'DOLP': dolp, 'DOCP': docp}


# ===========================================================================
# Engine -> propagator bridge: build a JonesField from per-order plane-wave
# amplitudes (AUDIT_V5_24_2 S5-5).  The carrier / power-normalization math was
# verified convention-correct on ``RCWAResult`` (v5.5.3, audit P1); it lived
# only there, so PMM / Berreman consumers had to hand-assemble it.  Factored
# here as the SINGLE source of truth, keyed on the shared
# ``per_order_amplitudes`` dict contract, and delegated to by
# ``RCWAResult.to_multiorder_field`` (byte-identical).
# ===========================================================================

def _order_power_scale(ax, ay, kz_m, kx_m, ky_m, kz_inc, kx0, ky0, incident):
    """Scale ``s`` so the deposited tangential power ``|s ax|^2 + |s ay|^2``
    equals one diffraction order's TRUE efficiency.

    A diffracted order carries power ``flux*(|ax|^2+|ay|^2+|az|^2)/einc_sq``
    with the Poynting flux weight ``flux = Re(kz_m/kz_inc)``, the longitudinal
    field ``az = -(kx ax + ky ay)/kz`` (large for steep orders), and the
    incident ``|E|^2 = einc_sq``.  Depositing the raw tangential
    ``|ax|^2+|ay|^2`` drops all three, so the reconstructed field violates
    energy conservation and can show the wrong dominant order.  Returns ``0``
    for an evanescent order (no propagating power).  All ``k`` are normalised
    by ``k0``; ``incident`` is the ``(2,)`` Jones drive."""
    tang = float(abs(ax) ** 2 + abs(ay) ** 2)
    flux = float(np.real(kz_m / kz_inc)) if kz_inc != 0 else 0.0
    if tang < 1e-300 or flux <= 0.0:      # evanescent / no tangential power
        return 0.0
    az = -(kx_m * ax + ky_m * ay) / (kz_m if abs(kz_m) > 1e-12 else 1.0)
    inc = np.asarray(incident, dtype=np.complex128).reshape(2)
    ez_inc = (-(kx0 * inc[0] + ky0 * inc[1]) / kz_inc) if kz_inc != 0 else 0.0
    einc_sq = float(abs(inc[0]) ** 2 + abs(inc[1]) ** 2 + abs(ez_inc) ** 2)
    eff = flux * (tang + float(abs(az) ** 2)) / (einc_sq if einc_sq else 1.0)
    return float(np.sqrt(eff / tang)) if eff > 0.0 else 0.0


def _plane_wave_carrier(kx_m, ky_m, wavelength, nx, ny, dx, dy):
    """Unit plane-wave carrier ``exp(i (kx_m x + ky_m y))`` of one order on a
    centred ``(ny, nx)`` grid.  ``kx_m`` / ``ky_m`` are stored normalised by
    ``k0 = 2*pi/wavelength``."""
    k0 = 2.0 * np.pi / wavelength
    kx = k0 * float(np.real(kx_m))                     # physical [1/m]
    ky = k0 * float(np.real(ky_m))
    xg = (np.arange(nx) - nx // 2) * dx
    yg = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(xg, yg)                         # (ny, nx)
    return np.exp(1j * (kx * X + ky * Y)).astype(np.complex128)


def _order_lanczos_sigma(orders):
    """Per-order Lanczos sigma factors ``sinc(m/(Mx+1)) sinc(n/(My+1))`` (1-D:
    the y-factor is 1), indexed by flat order index -- damps the high orders
    smoothly to suppress Gibbs ringing in the reconstructed real-space
    field."""
    o = np.asarray(orders)
    if o.ndim == 2:
        mx = max(1, int(np.abs(o[:, 0]).max()))
        my = max(1, int(np.abs(o[:, 1]).max()))
        return (np.sinc(o[:, 0] / (mx + 1.0))
                * np.sinc(o[:, 1] / (my + 1.0)))
    mx = max(1, int(np.abs(o).max()))
    return np.sinc(o / (mx + 1.0))


def _order_flat_index(orders, order):
    """Flat index of a diffraction order in an ``orders`` array (1-D ``(N,)``
    of ``m`` or 2-D ``(N, 2)`` of ``(m, n)``); ``order`` may be an ``int``
    (``n`` defaults to 0) or an ``(m, n)`` pair."""
    o = np.asarray(orders)
    if o.ndim == 2:
        if np.ndim(order) == 0:
            hit = np.where((o[:, 0] == int(order)) & (o[:, 1] == 0))[0]
        else:
            m, n = order
            hit = np.where((o[:, 0] == int(m)) & (o[:, 1] == int(n)))[0]
    else:
        if np.ndim(order) == 0:
            hit = np.where(o == int(order))[0]
        else:
            m, n = order
            hit = (np.where(o == int(m))[0] if int(n) == 0
                   else np.asarray([], dtype=int))
    if hit.size == 0:
        raise ValueError(
            f"jones_field_from_orders: order {order!r} is outside the "
            f"retained range; increase n_orders on the solve.")
    return int(hit[0])


def jones_field_from_orders(amps, nx, ny, dx, *, incident=(1.0, 0.0),
                            normalize="power", orders=None, filter="none",
                            dy=None):
    """Reconstruct a diffracted field as a :class:`JonesField` from an engine's
    per-order plane-wave amplitudes -- the engine->propagator bridge for ANY
    solver exposing the ``per_order_amplitudes`` dict contract (RCWA, PMM,
    2-D PMM), not just :class:`RCWAResult` (AUDIT_V5_24_2 S5-5).

    ``E(x, y) = sum_m A_m exp(i (kx_m x + ky_m y))`` over the requested
    ``orders`` (default: every PROPAGATING order, ``Re(kz) > 0``), on a centred
    ``(ny, nx)`` grid of pitch ``dx`` (``dy`` defaults to ``dx``).

    Parameters
    ----------
    amps : dict
        A ``per_order_amplitudes(port)`` dict: ``orders`` (``(N,)`` or
        ``(N, 2)``), ``Ex`` / ``Ey`` each ``(2, N)`` (row 0 = response to
        incident lab ``E_x``, row 1 to ``E_y``), ``kx`` / ``ky`` / ``kz``
        normalised by ``k0``, ``wavelength``, and the incidence terms
        ``kz_inc`` / ``kx0`` / ``ky0`` (the incident-medium specular ``kz`` and
        transverse wavevectors the ``normalize='power'`` flux weight needs --
        the transmission-port ``kz`` is the SUBSTRATE ``kz``, not the incident
        one).  The dict already selects the reflection or transmission port.
    nx, ny : int
        Output grid shape ``(ny, nx)``.
    dx : float
        Grid pitch [m] (``dy`` defaults to ``dx``).
    incident : (2,) complex, optional
        Incident Jones vector ``(E_x, E_y)``.  Default ``(1, 0)``.
    normalize : {'power', 'field'}, optional
        ``'power'`` (default) scales each order so ``sum |A_m|^2`` equals the
        sum of the propagating-order diffraction EFFICIENCIES (energy-correct,
        right dominant order); ``'field'`` deposits the raw tangential boundary
        amplitudes (whose ``|.|^2`` is NOT power -- it drops the Poynting flux
        weight and the longitudinal component).
    orders : sequence, optional
        Explicit orders to superpose (each an ``int`` or ``(m, n)``); default
        is every propagating order.  Evanescent explicit orders are skipped
        with a warning.
    filter : {'none', 'lanczos'}, optional
        ``'lanczos'`` damps the high orders (Gibbs suppression at sharp
        permittivity steps) -- a visualisation aid, not energy-exact.
    dy : float, optional
        Grid pitch in y [m] (default ``dx``).

    Returns
    -------
    JonesField
        The reconstructed field on a centred ``(ny, nx)`` grid.  The
        reconstruction is exact only over one unit cell (the field is
        quasi-periodic); evanescent orders are excluded.
    """
    if normalize not in ("power", "field"):
        raise ValueError(
            f"jones_field_from_orders: normalize must be 'power' or 'field', "
            f"got {normalize!r}.")
    if filter not in ("none", "lanczos"):
        raise ValueError(
            f"jones_field_from_orders: filter must be 'none' or 'lanczos', "
            f"got {filter!r}.")
    for key in ("orders", "Ex", "Ey", "kx", "ky", "kz", "wavelength",
                "kz_inc", "kx0", "ky0"):
        if key not in amps:
            raise ValueError(
                f"jones_field_from_orders: the amplitude dict is missing "
                f"{key!r}; pass a per_order_amplitudes(port) dict from a "
                f"NumPy solve (kz_inc/kx0/ky0 are required for the power "
                f"normalization).")
    oarr = np.asarray(amps["orders"])
    Ex = np.asarray(amps["Ex"])
    Ey = np.asarray(amps["Ey"])
    kx = np.asarray(amps["kx"])
    ky = np.asarray(amps["ky"])
    kz = np.asarray(amps["kz"])
    wavelength = amps["wavelength"]
    kz_inc, kx0, ky0 = amps["kz_inc"], amps["kx0"], amps["ky0"]
    inc = np.asarray(incident, dtype=np.complex128).reshape(2)
    sigma = _order_lanczos_sigma(oarr) if filter == "lanczos" else None
    if orders is None:
        idxs = [i for i in range(kz.shape[0]) if np.real(kz[i]) > 1e-12]
    else:
        idxs = []
        for o in orders:
            i = _order_flat_index(oarr, o)
            if np.real(kz[i]) <= 1e-12:
                warnings.warn(
                    f"jones_field_from_orders: order {o!r} is evanescent "
                    f"(Re(kz) <= 0) and is skipped; it carries no propagating "
                    f"power.", stacklevel=2)
                continue
            idxs.append(i)
    ny_i, nx_i = int(ny), int(nx)
    dx_f = float(dx)
    dy_f = float(dx if dy is None else dy)
    ex = np.zeros((ny_i, nx_i), dtype=np.complex128)
    ey = np.zeros((ny_i, nx_i), dtype=np.complex128)
    for idx in idxs:
        ax = complex(inc @ Ex[:, idx])
        ay = complex(inc @ Ey[:, idx])
        if normalize == "power":
            s = _order_power_scale(ax, ay, complex(kz[idx]), complex(kx[idx]),
                                   complex(ky[idx]), kz_inc, kx0, ky0, inc)
            ax, ay = s * ax, s * ay
        if sigma is not None:
            w = sigma[idx]
            ax, ay = w * ax, w * ay
        carrier = _plane_wave_carrier(kx[idx], ky[idx], wavelength,
                                      nx_i, ny_i, dx_f, dy_f)
        ex += ax * carrier
        ey += ay * carrier
    return JonesField(ex, ey, dx=dx, dy=dy)
