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

import numpy as np

# CuPy is lazy-loaded; this module accesses it via the lenses module's
# lazy slot so a single load is shared across the package.
from . import lenses as _lenses_module
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

def apply_thin_lens(E_in, f, wavelength, dx, dy=None, xc=0, yc=0,
                    use_gpu=False, lens_model='paraxial'):
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
    """
    # Determine array library
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
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
        r_over_f_sq = r_sq / f ** 2
        valid = r_over_f_sq < 1.0
        sqrt_term = xp.sqrt(xp.maximum(1.0 - r_over_f_sq, 0.0))
        phase = k * f * (1.0 - sqrt_term)
        lens_phase = xp.where(valid, xp.exp(-1j * phase), 0.0 + 0.0j)

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

    return E_in * lens_phase


# ---------------------------------------------------------------------------
# Thick spherical singlet
# ---------------------------------------------------------------------------

def apply_spherical_lens(E_in, R1, R2, d, n_lens, wavelength, dx, dy=None,
                         aperture_diameter=None, xc=0, yc=0, use_gpu=False):
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
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
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
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field, 0.0 + 0.0j
        )
    else:
        max_h_sq = np.inf
        if not np.isinf(R1):
            max_h_sq = min(max_h_sq, R1 ** 2)
        if not np.isinf(R2):
            max_h_sq = min(max_h_sq, R2 ** 2)
        if max_h_sq < np.inf:
            lens_field = xp.where(
                h_sq < max_h_sq * 0.9999, lens_field, 0.0 + 0.0j
            )

    return E_in * lens_field


# ---------------------------------------------------------------------------
# Thick aspheric singlet (conic + even polynomial)
# ---------------------------------------------------------------------------

def apply_aspheric_lens(E_in, R1, R2, d, n_lens, wavelength, dx, dy=None,
                        k1=0, k2=0, A1=None, A2=None,
                        aperture_diameter=None, xc=0, yc=0, use_gpu=False):
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
    if CUPY_AVAILABLE and (use_gpu or _is_cupy_array(E_in)):
        xp = cp
        if not _is_cupy_array(E_in):
            E_in = cp.asarray(E_in)
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
        denom_arg_safe = xp.maximum(denom_arg, 1e-12)
        sag_unsigned = h_sq / (R_abs * (1 + xp.sqrt(denom_arg_safe)))
        sag = np.sign(R) * sag_unsigned

        if A_coeffs:
            for power, coeff in A_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

        return sag

    sag1 = _aspheric_sag(h_sq, R1, k1, A1)
    sag2 = _aspheric_sag(h_sq, R2, k2, A2)

    phase = -kw * (n_lens - 1) * (sag1 - sag2)
    lens_field = xp.exp(1j * phase)

    # Apply aperture
    if aperture_diameter is not None:
        lens_field = xp.where(
            h_sq <= (aperture_diameter / 2) ** 2, lens_field, 0.0 + 0.0j
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
                h_sq < max_h_sq * 0.9999, lens_field, 0.0 + 0.0j
            )

    return E_in * lens_field




# ---------------------------------------------------------------------------

def apply_cylindrical_lens(E_in, f, wavelength, dx, dy=None, axis='x',
                           xc=0, yc=0):
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

    Returns
    -------
    E_out : ndarray (complex, N x N)

    Notes
    -----
    Produces a line focus (orthogonal to the focusing axis) instead of a
    point focus.
    """
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy

    if axis == 'x':
        phase_1d = -k / (2 * f) * (x - xc) ** 2
        phase = phase_1d[np.newaxis, :]
    elif axis == 'y':
        phase_1d = -k / (2 * f) * (y - yc) ** 2
        phase = phase_1d[:, np.newaxis]
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    return E_in * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# GRIN lens
# ---------------------------------------------------------------------------

def apply_grin_lens(E_in, n0, g, d, wavelength, dx, dy=None, xc=0, yc=0):
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
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_sq = (X - xc) ** 2 + (Y - yc) ** 2

    phase = -k * n0 * (g ** 2 / 2) * d * r_sq
    return E_in * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# Axicon
# ---------------------------------------------------------------------------

def apply_axicon(E_in, alpha, n_axicon, wavelength, dx, dy=None, xc=0, yc=0):
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
    Ny, Nx = E_in.shape
    if dy is None:
        dy = dx
    k = 2 * np.pi / wavelength

    if isinstance(n_axicon, str):
        n = get_glass_index(n_axicon, wavelength)
    else:
        n = float(n_axicon)

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    phase = -k * (n - 1) * alpha * r
    return E_in * np.exp(1j * phase)




__all__ = [
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_cylindrical_lens',
    'apply_axicon',
    'apply_grin_lens',
]
