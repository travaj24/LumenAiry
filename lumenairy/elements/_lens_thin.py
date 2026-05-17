"""
lumenairy.elements._lens_thin -- thin-lens / single-element phase screens.

Six small entry points that model an optical element as a single
thin phase mask multiplied onto the input field.  Extracted from
``lenses.py`` in v3.5.5 to reduce that module's bloat.  All names
re-exported from :mod:`lumenairy.elements.lenses` for
backwards-compatible imports.

Functions
---------
apply_thin_lens / apply_spherical_lens / apply_aspheric_lens
    Plano + bi-curved single-element refractive lenses.
apply_cylindrical_lens
    Single-axis focusing element.
apply_axicon
    Conical phase element (Bessel-beam generator).
apply_grin_lens
    Gradient-index rod lens (paraxial).

All functions accept a ``use_gpu=False`` flag and dispatch to CuPy
when CuPy is installed.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

# CuPy is lazy-loaded; this module accesses it via the lenses module's
# lazy slot so a single load is shared across the package.
from . import lenses as _lenses_module
from ..glass import get_glass_index  # 4.10: was missing, broke apply_axicon
from .lenses import (
    surface_sag_general,
    surface_sag_biconic,
    CUPY_AVAILABLE,
)


def _is_cupy_array(x):
    return _lenses_module._is_cupy_array(x)


# Module-level cp alias.  Updated whenever _lenses_module's cp is loaded
# (it points at None until first GPU call, then the actual cupy
# module).  We sync via a property-style accessor below.

def __getattr__(name):
    """PEP 562 module-level __getattr__: route ``cp`` to the lenses
    module's lazy slot.  Triggers when callers do
    ``from ._lens_thin import cp`` -- the in-function references inside
    each apply_* below resolve via this fallback if `cp` isn't yet a
    module global.
    """
    if name == 'cp':
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        return _lenses_module.cp
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# ---------------------------------------------------------------------------
# Thin lens models
# ---------------------------------------------------------------------------

def apply_thin_lens(
    E_in: np.ndarray,
    *,
    f: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
    lens_model: str = 'paraxial',
) -> np.ndarray:
    """
    Apply a thin-lens phase to an optical field.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging, negative = diverging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Center of the lens [m] (for decentered lenses).
    use_gpu : bool
        If True and CuPy is available, run on the GPU.
    lens_model : str
        Phase model.  One of:

        ``'paraxial'``
            Quadratic approximation: phi = -k/(2f) * r**2.
            Valid for r/f < ~0.1 (half-angle < ~6 deg).
        ``'nonparaxial'``
            Exact spherical wavefront: phi = k * (f - sqrt(f**2 + r**2)).
            Accurate up to r/f ~ 0.5 (half-angle ~30 deg).
        ``'aplanatic'``
            Satisfies the Abbe sine condition (sin(theta) = r/f).
            phi = -k * f * (1 - sqrt(1 - r**2/f**2)) for r < |f|.
            Ideal for imaging systems; eliminates coma.
        ``'local_only'``
            Quadratic focusing about the decentered point *without* the
            linear tilt that a decentered paraxial lens would produce.
            Useful for micro-lens arrays where each lenslet should focus
            locally without steering the beam.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    All arguments past ``E_in`` are keyword-only (since 4.7).  This
    makes the call order non-load-bearing and prevents typos that
    silently swap ``wavelength`` and ``dx`` (both ~1e-6).
    """
    # Determine array library.  PEP 562 ``__getattr__`` cannot
    # resolve bare ``cp`` inside a function body (LEGB rules skip
    # module-level __getattr__), so we go through the lenses-module
    # lazy slot explicitly.  Same pattern as apply_cylindrical_lens /
    # apply_grin_lens / apply_axicon.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    if lens_model == 'paraxial':
        lens_phase = xp.exp(-1j * k / (2 * f) * r_sq)

    elif lens_model == 'nonparaxial':
        lens_phase = xp.exp(1j * k * (f - xp.sqrt(f ** 2 + r_sq)))

    elif lens_model == 'aplanatic':
        # 4.10: replacing the outside-aperture region with 0+0j inside
        # the PHASE-MASK array silently clipped the amplitude there.
        # Use 1+0j (unit phase) outside the aplanatic domain so the
        # multiplier leaves the field unchanged in the rim annulus;
        # the lens aperture itself should be enforced via a separate
        # aperture mask, not via the phase mask.
        r_over_f_sq = r_sq / f ** 2
        valid = r_over_f_sq < 1.0
        sqrt_term = xp.sqrt(xp.maximum(1.0 - r_over_f_sq, 0.0))
        phase = k * f * (1.0 - sqrt_term)
        lens_phase = xp.where(valid, xp.exp(-1j * phase), 1.0 + 0.0j)

    elif lens_model == 'local_only':
        # Pure local focusing: the standard decentered quadratic minus the
        # linear tilt k/f * (xc*x + yc*y) that would otherwise steer the beam.
        decentered_phase = -k / (2 * f) * r_sq
        tilt_cancel = -k / f * (xc * X + yc * Y)
        lens_phase = xp.exp(1j * (decentered_phase + tilt_cancel))

    else:
        raise ValueError(
            f"Unknown lens_model: {lens_model!r}. "
            f"Choose from 'paraxial', 'nonparaxial', 'aplanatic', 'local_only'."
        )

    # v4.13.2 (audit C-P1-5): coerce the phase mask to E_in's dtype so
    # a complex64 input stays complex64.  ``xp.exp(1j * <float64
    # phase>)`` produces complex128 regardless of E_in.dtype; without
    # this cast the multiply silently upcasts E to complex128.
    # Mirrors the v4.13.0 L6 apply_mirror dtype guard.
    if lens_phase.dtype != E_in.dtype:
        lens_phase = lens_phase.astype(E_in.dtype)

    return E_in * lens_phase


# ---------------------------------------------------------------------------
# Thick spherical singlet
# ---------------------------------------------------------------------------

def apply_spherical_lens(
    E_in: np.ndarray,
    *,
    R1: float,
    R2: float,
    d: float,
    n_lens: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    aperture_diameter: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply the phase of a thick singlet with spherical surfaces.

    Computes the exact optical-path difference through a glass element with
    two spherical surfaces, naturally including spherical aberration and all
    higher-order monochromatic aberrations.

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1 : float
        Radius of curvature of the front surface [m].
        Positive = center of curvature on the transmission side (convex
        toward input).  ``np.inf`` for a flat surface.
    R2 : float
        Radius of curvature of the back surface [m].
        Negative = center of curvature on the input side (convex toward
        output).  Example: biconvex lens has R1 > 0, R2 < 0.
    d : float
        Center thickness [m].
    n_lens : float
        Refractive index of the lens material.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    aperture_diameter : float or None
        Clear aperture diameter [m].  If None the aperture is set by the
        surface radii of curvature.
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    The thickness profile is ``t(h) = d - sag1(h) - sag2(h)`` where each
    signed sag is ``sag(h) = R - sign(R) * sqrt(R**2 - h**2)``.

    The OPD relative to the center is:

        delta_phi(h) = -k * (n - 1) * (sag1(h) - sag2(h))

    which reduces to ``-k/(2f) * h**2`` in the paraxial limit with
    ``1/f = (n-1) * (1/R1 - 1/R2)`` (lensmaker's equation).
    """
    # See apply_thin_lens for the ``_lenses_module.cp`` rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _surface_sag(h_sq, R):
        """Signed spherical sag: positive for convex (R > 0)."""
        if R is None or np.isinf(R):
            return xp.zeros_like(h_sq)
        h_sq_safe = xp.minimum(h_sq, R ** 2 * 0.9999)
        return R - np.sign(R) * xp.sqrt(R ** 2 - h_sq_safe)

    sag1 = _surface_sag(h_sq, R1)
    sag2 = _surface_sag(h_sq, R2)

    phase = -k * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Clear aperture
    # v4.13.2 (audit C-P1-4): dtype-aware zero so a complex64 E_in
    # stays complex64 (was silently upcasting to complex128 via the
    # ``0.0 + 0.0j`` literal).  Matches the apply_aperture / apply_mirror
    # template added in v4.13.1 P3 #21.
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field,
            xp.zeros((), dtype=lens_field.dtype)
        )
    else:
        max_h_sq = np.inf
        if not np.isinf(R1):
            max_h_sq = min(max_h_sq, R1 ** 2)
        if not np.isinf(R2):
            max_h_sq = min(max_h_sq, R2 ** 2)
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field,
                xp.zeros((), dtype=lens_field.dtype)
            )

    # v4.13.2 (audit C-P1-5): coerce lens_field to E_in.dtype so
    # complex64 inputs stay complex64.
    if lens_field.dtype != E_in.dtype:
        lens_field = lens_field.astype(E_in.dtype)

    return E_in * lens_field


# ---------------------------------------------------------------------------
# Thick aspheric singlet (conic + even polynomial)
# ---------------------------------------------------------------------------

def apply_aspheric_lens(
    E_in: np.ndarray,
    *,
    R1: float,
    R2: float,
    d: float,
    n_lens: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    k1: float = 0,
    k2: float = 0,
    A1: Optional[Dict[int, float]] = None,
    A2: Optional[Dict[int, float]] = None,
    aperture_diameter: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply an aspheric singlet lens phase based on exact OPD through thick glass.

    Each surface follows the standard aspheric sag equation:

        sag(h) = h**2 / (R * (1 + sqrt(1 - (1+k)*h**2/R**2)))
                 + A4*h**4 + A6*h**6 + A8*h**8 + A10*h**10

    Parameters
    ----------
    E_in : ndarray (complex), shape (Ny, Nx)
        Input electric field.
    R1, R2 : float
        Radii of curvature [m] (same sign convention as
        :func:`apply_spherical_lens`).
    d : float
        Center thickness [m].
    n_lens : float
        Refractive index at the operating wavelength.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    k1, k2 : float
        Conic constants for surfaces 1 and 2 (0 = sphere, -1 = paraboloid).
    A1, A2 : dict or None
        Even aspheric polynomial coefficients for each surface.
        Keys are the powers of h: ``{4: A4, 6: A6, 8: A8, 10: A10}``.
    aperture_diameter : float or None
        Clear aperture diameter [m].
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    E_out : ndarray (complex), same shape as *E_in*

    Notes
    -----
    With ``k1=k2=0`` and ``A1=A2=None`` this reduces to
    :func:`apply_spherical_lens`.

    A plano-convex lens with ``k1 = -n_lens**2`` on the curved surface
    eliminates third-order spherical aberration for collimated input.
    """
    # See apply_thin_lens for the ``_lenses_module.cp`` rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx

    kw = 2 * np.pi / wavelength  # wavenumber (avoid shadowing conic k)

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    def _aspheric_sag(h_sq, R, k_conic, A_coeffs):
        """Signed aspheric sag for one surface."""
        if R is None or np.isinf(R):
            sag = xp.zeros_like(h_sq)
            if A_coeffs:
                for power, coeff in A_coeffs.items():
                    sag = sag + coeff * h_sq ** (power // 2)
            return sag

        R_abs = abs(R)
        norm_h_sq = h_sq / R_abs ** 2
        denom_arg = 1 - (1 + k_conic) * norm_h_sq
        # 4.10: clamp invalid (outside-conic-domain) pixels to NaN so a
        # downstream aperture mask explicitly zeros them, rather than
        # silently extrapolating a 1e-12 floor that produced
        # near-singular sag (1e6 m for typical optics) outside the
        # surface domain.
        valid = denom_arg > 0
        denom_arg_safe = xp.where(valid, denom_arg, 1.0)
        sag_unsigned = h_sq / (R_abs * (1 + xp.sqrt(denom_arg_safe)))
        sag = np.sign(R) * sag_unsigned
        sag = xp.where(valid, sag, xp.nan)

        if A_coeffs:
            for power, coeff in A_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

        return sag

    sag1 = _aspheric_sag(h_sq, R1, k1, A1)
    sag2 = _aspheric_sag(h_sq, R2, k2, A2)

    phase = -kw * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Apply aperture
    # v4.13.2 (audit C-P1-4): dtype-aware zero, see apply_spherical_lens
    # above for rationale.
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field,
            xp.zeros((), dtype=lens_field.dtype)
        )
    else:
        max_h_sq = np.inf
        if R1 is not None and not np.isinf(R1):
            if (1 + k1) > 0:
                max_h_sq = min(max_h_sq, R1 ** 2 / (1 + k1))
        if R2 is not None and not np.isinf(R2):
            if (1 + k2) > 0:
                max_h_sq = min(max_h_sq, R2 ** 2 / (1 + k2))
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field,
                xp.zeros((), dtype=lens_field.dtype)
            )

    # v4.13.2 (audit C-P1-5): coerce lens_field to E_in.dtype so
    # complex64 inputs stay complex64.
    if lens_field.dtype != E_in.dtype:
        lens_field = lens_field.astype(E_in.dtype)

    return E_in * lens_field




# ---------------------------------------------------------------------------

def apply_cylindrical_lens(
    E_in: np.ndarray,
    *,
    f: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    axis: str = 'x',
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a cylindrical thin-lens phase (focusing in one axis only).

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    f : float
        Focal length [m].  Positive = converging.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    axis : ``'x'`` or ``'y'``
        Focusing axis.  ``'x'`` applies phi = -k/(2f) * (x - xc)**2;
        ``'y'`` applies phi = -k/(2f) * (y - yc)**2.
    xc, yc : float
        Lens center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) so the module-docstring claim that
        "all functions accept use_gpu=False" is now true.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    Produces a line focus (orthogonal to the focusing axis) instead of a
    point focus.
    """
    # v4.13.2 (audit C-P1-6): dispatch through CuPy when use_gpu=True
    # or E_in is already a CuPy array.  Resolve ``cp`` via the
    # _lenses_module lazy slot rather than a bare global (which is
    # not bound in this module's namespace).  Pre-fix the three
    # sibling functions had no use_gpu path at all.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy

    if axis == 'x':
        phase_1d = -k / (2 * f) * (x - xc) ** 2
        phase = phase_1d[None, :]
    elif axis == 'y':
        phase_1d = -k / (2 * f) * (y - yc) ** 2
        phase = phase_1d[:, None]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # v4.13.2 (audit C-P1-5): cast the phase mask to E_in.dtype so a
    # complex64 input stays complex64 (xp.exp(1j*phase) returns
    # complex128 from float64 phase regardless of E_in.dtype).
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)

    return E_in * phase_exp


# ---------------------------------------------------------------------------
# GRIN lens
# ---------------------------------------------------------------------------

def apply_grin_lens(
    E_in: np.ndarray,
    *,
    n0: float,
    g: float,
    d: float,
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply a gradient-index (GRIN) rod lens phase (thin approximation).

    Models a GRIN rod with parabolic index profile:

        n(r) = n0 * (1 - g**2 / 2 * r**2)

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    n0 : float
        On-axis refractive index.
    g : float
        Gradient constant [1/m] (also called sqrt(A)).
        Pitch P = 2 pi / g.
    d : float
        Rod length (thickness) [m].
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        GRIN lens center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) to honour the module-docstring claim.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The quadratic OPD through the rod gives an effective focal length

        f = 1 / (n0 * g**2 * d)      (thin approximation, g*d << 1)

    For longer rods the exact result is ``f = 1 / (n0 * g * sin(g*d))``.
    Quarter-pitch (g*d = pi/2) collimates a point source at the front face;
    half-pitch (g*d = pi) reimages 1:1 inverted.
    """
    # v4.13.2 (audit C-P1-6): CuPy dispatch (was previously numpy-only).
    # See apply_cylindrical_lens above for the _lenses_module.cp
    # resolution rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    phase = -k * n0 * (g ** 2 / 2) * d * r_sq
    # v4.13.2 (audit C-P1-5): cast phase mask to E_in.dtype so
    # complex64 inputs stay complex64.
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)
    return E_in * phase_exp


# ---------------------------------------------------------------------------
# Axicon
# ---------------------------------------------------------------------------

def apply_axicon(
    E_in: np.ndarray,
    alpha: float,
    n_axicon: Union[float, str],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    xc: float = 0,
    yc: float = 0,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Apply an axicon (conical lens) phase to generate a Bessel-like beam.

    Parameters
    ----------
    E_in : ndarray (complex, N x N)
        Input electric field.
    alpha : float
        Physical half-angle of the cone [radians].
        Typical range: 0.5--5 degrees (0.009--0.087 rad).
    n_axicon : float or str
        Refractive index of the axicon material.  If a string is passed it
        is resolved via :func:`get_glass_index`.
    wavelength : float
        Optical wavelength [m].
    dx : float
        Grid spacing in x [m].
    dy : float or None
        Grid spacing in y [m].  Defaults to *dx*.
    xc, yc : float
        Axicon center [m].
    use_gpu : bool
        If True and CuPy is available, run on the GPU.  Added in
        v4.13.2 (audit C-P1-6) to honour the module-docstring claim.

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    The axicon imparts a phase linear in radial distance:

        phi(r) = -k * (n - 1) * alpha * r

    A collimated input beam produces a non-diffracting Bessel-beam region
    extending over ``z_max ~ w0 / ((n - 1) * alpha)`` where *w0* is the
    input beam radius.
    """
    # v4.13.2 (audit C-P1-6): CuPy dispatch.  See
    # apply_cylindrical_lens above for the _lenses_module.cp
    # resolution rationale.
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        if _lenses_module.cp is None:
            _lenses_module._ensure_cupy_loaded()
        _cp = _lenses_module.cp
        xp = _cp
        if not _is_cupy_array(E_in):
            E_in = _cp.asarray(E_in)
    else:
        xp = np

    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    if isinstance(n_axicon, str):
        n = get_glass_index(n_axicon, wavelength)
    else:
        n = float(n_axicon)

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r = xp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    phase = -k * (n - 1) * alpha * r
    # v4.13.2 (audit C-P1-5): cast phase mask to E_in.dtype so
    # complex64 inputs stay complex64.
    phase_exp = xp.exp(1j * phase)
    if phase_exp.dtype != E_in.dtype:
        phase_exp = phase_exp.astype(E_in.dtype)
    return E_in * phase_exp




__all__ = [
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_cylindrical_lens',
    'apply_axicon',
    'apply_grin_lens',
]
