"""
Optical elements, apertures, and wavefront manipulation functions.

This module provides functions for modelling discrete optical elements that
modify an electric field on a 2-D computational grid.  The elements fall
into several categories:

* **Mirrors** -- flat and curved reflectors (including conics/aspheres).
* **Apertures** -- hard-edge amplitude masks (circular, annular, rectangular)
  and soft (Gaussian) apertures.
* **Arbitrary masks** -- generic complex transmission functions for DOEs,
  SLMs, metasurfaces, grey-scale filters, etc.
* **Zernike aberrations** -- phase screens described by Zernike polynomial
  coefficients.
* **Turbulence phase screens** -- random atmospheric phase screens with
  Kolmogorov or von Karman statistics.

Author: Andrew Traverso
"""

from math import factorial

import numpy as np

# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup): this used to be a
# 4-line wrapper duplicated in 5 files; consolidated to the canonical
# backend helper.  The underscore-prefixed alias preserves existing
# in-module references without touching call sites.
from ..backend import array_namespace as _xp_of  # noqa: E402
from .lenses import _surface_sag_general

# =============================================================================
# MIRRORS
# =============================================================================

def apply_mirror(E_in, wavelength, dx, radius=None, conic=0.0,
                 aperture_diameter=None, xc=0, yc=0, dy=None):
    """
    Apply a mirror reflection to an optical field.

    For a flat mirror, this simply reverses the propagation direction
    (the field is unchanged apart from an optional aperture). For a
    curved mirror, the reflection imparts a focusing phase equivalent
    to a thin lens with f = R/2.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.  May be a NumPy, CuPy, or JAX array;
        the namespace is dispatched via :func:`array_namespace`
        (v4.13.0 / L6), so the returned ``E_out`` lives on the same
        backend as ``E_in``.

    wavelength : float
        Free-space wavelength [m].

    dx : float
        Grid spacing in x [m].

    radius : float or None
        Radius of curvature of the mirror [m].
        Positive = concave (focusing), negative = convex (diverging).
        None or inf = flat mirror.

        Note: this is the **wave-side** ``radius`` convention used by
        this user-facing function (R > 0 = concave focusing).  It is
        the OPPOSITE sign convention from the Welford signed-R used
        by :func:`system_abcd` / :func:`seidel_coefficients` (where
        a concave mirror in the incoming-light frame has R < 0).  The
        wave-side convention has been the API for many releases and
        is intentionally preserved here -- see
        ``validation/elements/test_elements.py``::``t_curved_mirror_focus``
        for the reconciliation against the Welford side.

    conic : float, default 0.0
        Conic constant of the mirror surface (0=sphere, -1=paraboloid).

    aperture_diameter : float or None
        Clear aperture [m]. If None, no aperture is applied.

    xc, yc : float, default 0
        Mirror center offset [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid, the
        library's usual case).  Provide explicitly for rectangular
        (non-square) grids so the aperture and sag aren't silently
        stretched along y.  Added in v4.13.0 (L6) to match the rest
        of the ``apply_*`` family.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Reflected field on the same array backend as ``E_in``.  After
        this call, subsequent ASM propagation models the return path
        (caller is responsible for using the correct propagation
        distances and sign conventions).

    Notes
    -----
    A concave mirror with radius R (wave-side convention, R > 0)
    acts like a converging lens with focal length f = R/2. The
    reflected phase is:

        phi(x,y) = -2 * (2*pi/lambda) * sag(x,y)

    where sag is the surface sag and the factor of 2 accounts for the
    double pass (incident + reflected) through the sag height. The sign
    is negative (phase delay) for concave mirrors (positive sag at edges).

    For a flat mirror, there is no phase change -- the field just reverses
    direction. The caller handles the direction reversal via propagation
    distances.

    For parabolic mirrors (conic=-1), the reflection is aberration-free
    for on-axis collimated input.
    """
    if dy is None:
        dy = dx
    xp = _xp_of(E_in)
    # NumPy short-circuit retains the legacy code path that the helper
    # ``_surface_sag_general`` exercises (numba-accelerated aspherics);
    # JAX / CuPy take the xp-native inline path below since the helper
    # silently demotes JAX -> NumPy (round-tripping through the host).
    is_numpy_backend = xp is np

    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    # JAX arrays are immutable so ``.copy()`` doesn't actually copy
    # (it returns a new view); for NumPy / CuPy this ensures we don't
    # mutate the caller's buffer when subsequently masking it.
    E = E_in.copy() if hasattr(E_in, 'copy') else xp.array(E_in)

    # Build the coordinate grid once on the right backend if either
    # the aperture mask or the sag phase is needed.
    need_grid = (aperture_diameter is not None) or (
        radius is not None and not np.isinf(radius)
    )
    if need_grid:
        x = (xp.arange(Nx) - Nx / 2) * dx
        y = (xp.arange(Ny) - Ny / 2) * dy
        X, Y = xp.meshgrid(x, y)
        h_sq = (X - xc) ** 2 + (Y - yc) ** 2

    # Apply aperture: a CIRCULAR clear aperture in physical (x, y)
    # coordinates.  ``h_sq`` above is built as ``(X - xc)^2 + (Y -
    # yc)^2`` directly from the physical coordinate grids (which
    # already incorporate ``dx`` / ``dy`` spacing), so a point (x, y)
    # is inside the aperture when its physical radius from the mirror
    # centre stays within ``aperture_diameter / 2``.  ``dy != dx``
    # (rectangular pixel grid) produces a stretched pixel sampling
    # but does NOT make the aperture itself elliptical -- the
    # aperture is geometrically a circle on the physical grid.  For
    # an elliptical clear aperture supply :func:`apply_aperture` with
    # ``shape='rectangular'`` or wrap with a pre-mask of the desired
    # ellipse.
    if aperture_diameter is not None:
        E = xp.where(h_sq <= (aperture_diameter / 2) ** 2,
                     E, xp.zeros((), dtype=E.dtype))

    # Curved mirror: apply focusing phase
    if radius is not None and not np.isinf(radius):
        if is_numpy_backend:
            # Keep the legacy NumPy / numba-aspheric helper for the
            # CPU path; bit-near-exact identical to v4.12.1.
            sag = _surface_sag_general(h_sq, radius, conic)
        else:
            # Inline xp-native sag (no aspherics in apply_mirror's
            # public signature, so just the conic term).
            #   sag = h_sq / (R * (1 + sqrt(1 - (1+k)*h_sq/R^2)))
            # Outside the conic domain ((1+k)*h_sq/R^2 >= 0.9999) the
            # surface is undefined; return NaN there so a downstream
            # aperture mask can hide those pixels (matching the
            # NumPy helper's v4.10 contract).
            norm = (1 + conic) * h_sq / radius ** 2
            valid = norm < 0.9999
            denom_arg = xp.where(valid, 1 - norm, 0.01)
            sag = xp.where(
                valid,
                h_sq / (radius * (1 + xp.sqrt(denom_arg))),
                xp.nan,
            )

        # Double-pass OPD: ray travels sag down to the surface and sag
        # back up, so total extra path = 2 * sag.
        # Phase delay (negative sign, same convention as apply_real_lens).
        opd = 2.0 * sag
        # v4.13.2 (audit P1-NEW-F): zero the OPD on undefined-surface
        # pixels (NaN sentinel from the conic-domain check above for
        # hyperbolic conics where (1+k)*h_sq/R^2 >= 0.9999, or from
        # _surface_sag_general on the NumPy path with the same
        # contract).  Without this, ``exp(1j * NaN) = NaN`` poisons
        # E and propagates NaN to every pixel during the next ASM
        # step.  Matches the apply_real_lens NaN guard at
        # _lens_real.py:704-705.  The caller's aperture mask should
        # already zero the field on the same pixels, so a 0-OPD
        # phase screen is a safe neutral.
        if bool(xp.any(xp.isnan(opd))):
            opd = xp.where(xp.isnan(opd), 0.0, opd)
        phase = -k * opd
        # Compute exp() in the high-precision dtype first, then cast
        # back to E's dtype so a complex64 input stays complex64.
        phase_exp = xp.exp(1j * phase)
        if phase_exp.dtype != E.dtype:
            phase_exp = phase_exp.astype(E.dtype)
        E = E * phase_exp

    return E


# =============================================================================
# APERTURES AND STOPS
# =============================================================================

def apply_aperture(E_in, dx, shape='circular', params=None, xc=0, yc=0,
                   dy=None):
    """
    Apply a standalone aperture (amplitude mask) to an optical field.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input electric field.

    dx : float
        Grid spacing in the x direction [m].

    shape : str
        Aperture shape:
        - ``'circular'``: disk aperture, params={'diameter': D}
        - ``'annular'``: ring aperture, params={'inner_diameter': Di, 'outer_diameter': Do}
        - ``'rectangular'``: rectangular slit, params={'width_x': Wx, 'width_y': Wy}

    params : dict
        Shape-specific parameters (all in meters). See ``shape`` for keys.

    xc, yc : float, default 0
        Center position of the aperture [m].

    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid, the
        library's usual case).  Supplied explicitly for rectangular
        (non-square) grids so annular / circular / rectangular
        apertures don't get silently stretched along y.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Field with aperture applied (zeroed outside the opening).
    """
    if params is None:
        params = {}
    if dy is None:
        dy = dx
    xp = _xp_of(E_in)

    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    if shape == 'circular':
        D = params.get('diameter', np.inf)
        h_sq = (X - xc)**2 + (Y - yc)**2
        mask = h_sq <= (D / 2)**2

    elif shape == 'annular':
        Di = params.get('inner_diameter', 0)
        Do = params.get('outer_diameter', np.inf)
        h_sq = (X - xc)**2 + (Y - yc)**2
        mask = (h_sq >= (Di / 2)**2) & (h_sq <= (Do / 2)**2)

    elif shape == 'rectangular':
        Wx = params.get('width_x', np.inf)
        Wy = params.get('width_y', np.inf)
        mask = (xp.abs(X - xc) <= Wx / 2) & (xp.abs(Y - yc) <= Wy / 2)

    else:
        raise ValueError(f"Unknown aperture shape: {shape!r}. "
                         f"Use 'circular', 'annular', or 'rectangular'.")

    # v4.14 (audit P3 #21): use a dtype-aware zero so a JAX x32
    # input stays complex64 rather than being silently upcast by the
    # complex128 literal ``0.0 + 0.0j``.
    return xp.where(mask, E_in, xp.zeros((), dtype=E_in.dtype))


# =============================================================================
# GAUSSIAN (SOFT) APERTURE
# =============================================================================

def apply_gaussian_aperture(E_in, dx, sigma, xc=0, yc=0, dy=None):
    """
    Apply a Gaussian (soft) aperture to a field.

    The transmission profile is a real-valued Gaussian:

        T(r) = exp(-r^2 / (2 * sigma^2))

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.
    dx : float
        Grid spacing in x [m].
    sigma : float
        Gaussian width parameter [m]. The transmission is
        exp(-r^2 / (2*sigma^2)).
        The 1/e amplitude radius is sigma*sqrt(2).
        The 1/e^2 intensity radius is also sigma*sqrt(2).
    xc, yc : float, default 0
        Center position [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.  Provide
        explicitly for rectangular (non-square) grids so the
        aperture isn't silently stretched along y.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
    """
    if dy is None:
        dy = dx
    xp = _xp_of(E_in)
    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    r_sq = (X - xc)**2 + (Y - yc)**2

    return E_in * xp.exp(-r_sq / (2 * sigma**2))


# =============================================================================
# ARBITRARY PHASE / AMPLITUDE MASK
# =============================================================================

def apply_mask(E_in, mask):
    """
    Apply an arbitrary complex transmission mask to an optical field.

    This is the most general element -- it can represent any phase-only mask,
    amplitude-only mask, or combined phase+amplitude mask (like a spatial
    light modulator, metasurface, custom DOE, or gray-scale filter).

    Parameters
    ----------
    E_in : ndarray (complex, N×N)
        Input electric field.

    mask : ndarray (complex or real, N×N)
        Transmission function. Must have the same shape as E_in.
        - Phase-only: ``np.exp(1j * phase_array)``
        - Amplitude-only: real array in [0, 1]
        - Combined: complex array with ``|mask| <= 1``

    Returns
    -------
    E_out : ndarray (complex, N×N)
        ``E_in * mask``

    Examples
    --------
    >>> # Apply a custom metasurface phase profile
    >>> phase = load_metasurface_design(...)  # your phase array
    >>> E_out = apply_mask(E_in, np.exp(1j * phase))

    >>> # Apply a neutral density filter (50% transmission)
    >>> E_out = apply_mask(E_in, 0.5 * np.ones_like(E_in))
    """
    if mask.shape != E_in.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match "
                         f"field shape {E_in.shape}")
    return E_in * mask


# =============================================================================
# ZERNIKE POLYNOMIALS
# =============================================================================

def zernike(n, m, rho, theta):
    """
    Compute a single Zernike polynomial Z_n^m on a polar grid.

    Uses **(n, m) indexing** (Born & Wolf) with **unit-variance
    (a.k.a. OSA / ANSI / Wyant) normalisation**:

        Z_n^m has RMS = 1 over the unit disk.
        Normalisation factor N = sqrt(n + 1)         for m == 0
                             N = sqrt(2 * (n + 1))   for m != 0

    This matches :func:`lumenairy.analysis.zernike_polynomial`
    so fits and reconstructions round-trip exactly.  The Noll *single
    index* convention is a different beast: to map j_Noll -> (n, m) use
    :func:`lumenairy.analysis.zernike_index_to_nm`.

    Parameters
    ----------
    n : int
        Radial order (n >= 0).
    m : int
        Azimuthal order (-n <= m <= n, n-|m| must be even).
    rho : ndarray
        Normalized radial coordinate (0 to 1 within the unit circle).
    theta : ndarray
        Azimuthal angle [radians].

    Returns
    -------
    Z : ndarray
        Zernike polynomial values. Zero outside the unit circle (rho > 1).

    Notes
    -----
    Common Zernike terms (this normalisation):
        Z(0,0)  = 1              (piston)
        Z(1,1)  = 2*rho*cos(th)  (tilt x)
        Z(1,-1) = 2*rho*sin(th)  (tilt y)
        Z(2,0)  = sqrt(3)*(2*rho^2 - 1)  (defocus)
        Z(2,2)  = sqrt(6)*rho^2*cos(2th) (astigmatism)
        Z(3,1)  = sqrt(8)*(3*rho^3 - 2*rho)*cos(th) (coma x)
        Z(4,0)  = sqrt(5)*(6*rho^4 - 6*rho^2 + 1)   (spherical)
    """
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n-|m| must be even: n={n}, m={m}")
    if abs(m) > n:
        raise ValueError(f"|m| must be <= n: n={n}, m={m}")

    # v4.16.0 (Agent A xp-dispatch walker): dispatch on rho's backend
    # so a CuPy / JAX rho input stays on its backend.  Pre-v4.16.0
    # the radial polynomial and azimuthal builders hardcoded
    # ``np.zeros_like`` / ``np.cos`` / ``np.where`` which (for the
    # constructor) silently produced a host NumPy array when the
    # caller passed a CuPy / JAX rho via apply_zernike_aberration.
    # ``np.sqrt`` on a scalar ``(n+1)`` remains a host scalar -- not
    # an array-construction call, so left as ``np.sqrt`` for the
    # normalisation factor.
    xp = _xp_of(rho)
    # Radial polynomial R_n^|m|
    m_abs = abs(m)
    R = xp.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1)**s * factorial(n - s)
                 / (factorial(s) * factorial((n + m_abs) // 2 - s)
                    * factorial((n - m_abs) // 2 - s)))
        R = R + coeff * rho**(n - 2 * s)

    # Azimuthal part
    if m > 0:
        Z = R * xp.cos(m * theta)
    elif m < 0:
        Z = R * xp.sin(-m * theta)
    else:
        Z = R

    # Normalization (Born & Wolf convention); scalar np.sqrt on a
    # Python int is a host scalar -- safe.
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))

    Z = norm * Z

    # Zero outside unit circle
    Z = xp.where(rho <= 1.0, Z, 0.0)

    return Z


def apply_zernike_aberration(E_in, dx, coefficients, aperture_radius,
                              dy=None):
    """
    Apply Zernike polynomial aberrations to a field.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input field.
    dx : float
        Grid spacing in x [m].
    coefficients : dict
        Zernike coefficients as ``{(n, m): amplitude_in_waves, ...}``.
        Example: ``{(2,0): 0.5, (4,0): 0.25}`` for 0.5 waves defocus
        + 0.25 waves primary spherical (OSA normalisation, RMS=1 over
        the unit disk, so 0.25 waves is the RMS contribution).

        .. note::
           :func:`apply_zernike_aberration` takes coefficients in
           **waves** (dimensionless fraction of wavelength), but
           :func:`~lumenairy.analysis.zernike_decompose`
           returns coefficients in **metres** (physical OPD).  Round-
           tripping requires a ``/ wavelength`` conversion::

               c_m = zernike_decompose(opd, dx, aperture)[0]
               c_waves = c_m / wavelength
               E_back = apply_zernike_aberration(
                   pupil, dx,
                   coefficients={(n, m): c_waves[j]
                                 for j, (n, m) in enumerate(...)},
                   aperture_radius=aperture / 2)
    aperture_radius : float
        Radius [m] over which the Zernike polynomials are defined.
        rho is normalized to this radius.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid).
        Provide explicitly for rectangular grids so the Zernike
        aperture isn't silently stretched along y.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)

    Examples
    --------
    >>> # Add 1 wave of spherical aberration over a 5mm aperture
    >>> E_out = apply_zernike_aberration(E_in, dx=2e-6,
    ...     coefficients={(4, 0): 1.0}, aperture_radius=5e-3)
    """
    if dy is None:
        dy = dx
    # v4.16.0 (Agent A xp-dispatch walker): route through ``_xp_of``
    # so CuPy / JAX field inputs stay on the dispatched backend.
    # Pre-v4.16.0 the coordinate grid + phase accumulator were
    # ``np.arange`` / ``np.meshgrid`` / ``np.zeros``, which silently
    # demoted any non-NumPy input to host NumPy and back.  The
    # ``zernike`` helper itself remains NumPy-only (it indexes into
    # polynomial coefficient tables that are NumPy-backed); the
    # phase tensor is built on ``xp`` and then the final
    # ``xp.exp(1j * 2 * pi * phase)`` and ``E_in * ...`` keep the
    # output on the dispatched backend.
    xp = _xp_of(E_in)
    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    rho = xp.sqrt(X**2 + Y**2) / aperture_radius
    theta = xp.arctan2(Y, X)

    phase = xp.zeros((Ny, Nx))
    for (n, m), amplitude in coefficients.items():
        phase = phase + amplitude * zernike(n, m, rho, theta)

    # Convert from waves to radians
    return E_in * xp.exp(1j * 2 * np.pi * phase)


# =============================================================================
# CORONAGRAPH TEMPLATES
# =============================================================================
#
# Helpers for the canonical building blocks of high-contrast imaging
# systems: focal-plane occulters (Lyot mask, scalar vortex), downstream
# pupil filters (Lyot stop), and entrance-pupil apodizers.  They follow
# the standard convention used by POPPy / HCIPy / prysm: each is a
# transmission filter applied to an existing complex field; energy
# blocked / phase-shifted by the mask is removed from the field rather
# than renormalised.
#
# Typical pipeline (Lyot coronagraph):
#
#     E_pup  = apply_apodized_pupil(E_pup, dx, diameter=D,
#                                    apodization='cos2')        # entrance
#     E_foc  = la.fraunhofer_propagate_mft(E_pup, f, wl,
#                                          dx, dx_foc, N_foc)   # to focus
#     E_foc  = apply_lyot_focal_plane_mask(E_foc, dx_foc,
#                                          mask_diameter=4*lam_over_D)
#     E_pup2 = la.fraunhofer_propagate_mft(E_foc, f, wl,
#                                          dx_foc, dx, N)        # back to pupil
#     E_pup2 = apply_lyot_stop(E_pup2, dx,
#                              outer_diameter=0.95 * D)
#     # ... propagate to final image plane.
#
# References
# ----------
# [1] Lyot, B. (1939).  "The study of the solar corona without an
#     eclipse."  Monthly Notices RAS 99, 580.
# [2] Mawet, D. et al. (2005).  "Annular Groove Phase Mask Coronagraph."
#     ApJ 633, 1191.
# [3] Soummer, R. (2005).  "Apodized Pupil Lyot Coronagraphs for
#     Arbitrary Telescope Apertures."  ApJ 618, L161.
# [4] Kasdin, N.J. et al. (2003).  "Extrasolar planet finding via
#     optimal apodized-pupil and shaped-pupil coronagraphs."  ApJ
#     582, 1147.

def apply_lyot_focal_plane_mask(E_in, dx, mask_diameter, *,
                                 profile='hard', sigma=None,
                                 xc=0.0, yc=0.0, dy=None):
    """Apply a focal-plane occulter (classical Lyot coronagraph mask).

    A focal-plane mask that blocks the on-axis stellar PSF core,
    leaving the diffracted starlight and any companion light to
    propagate through the downstream Lyot stop.  This is the
    focal-plane element of a Lyot or band-limited coronagraph.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input focal-plane field (typically from
        :func:`fraunhofer_propagate_mft` or :func:`compute_psf`).
    dx : float
        Focal-plane grid spacing [m].
    mask_diameter : float
        Diameter of the occulting mask [m].  For an f/D system at
        wavelength ``lambda`` the diffraction-limited PSF FWHM is
        ``1.028 * lambda * f / D``; coronagraph masks are typically
        sized in units of ``lambda*f/D``, with ``4 lambda*f/D`` a
        common choice for ground-based systems.
    profile : ``'hard'`` (default) / ``'gaussian'`` / ``'sin2'``
        Mask transmission profile:

        * ``'hard'``: ``T = 0`` inside the mask, ``1`` outside.  The
          classical Lyot mask.
        * ``'gaussian'``: ``T(r) = 1 - exp(-r^2 / (2 sigma^2))``;
          smooth-edged for reduced ringing.  Requires ``sigma``.
        * ``'sin2'``: band-limited mask ``T(r) = sin^2(pi r / D)``
          inside the mask diameter, smoothly going to 1 at the edge.
          A simple band-limited approximation in the Kuchner-Traub
          family.
    sigma : float, optional
        Gaussian width parameter [m] for ``profile='gaussian'``.
        Defaults to ``mask_diameter / 6`` (3-sigma at the mask edge).
        Ignored for ``'hard'`` / ``'sin2'``.
    xc, yc : float, default 0
        Mask centre [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Field with the focal-plane mask applied.  Energy blocked by
        the mask is removed (not renormalised).

    Notes
    -----
    The ``'gaussian'`` and ``'sin2'`` smooth-edged profiles avoid the
    Gibbs ringing that a hard-edge mask introduces at the downstream
    Lyot stop, which in turn improves the achievable contrast.  In
    practice they trade off inner-working-angle for contrast --
    smoother masks let starlight closer to the chief through.

    The sign of the field outside the mask is preserved; this routine
    does NOT apply the +1 -> 0 polarity flip that some coronagraph
    architectures use for matched-filter detection.
    """
    if dy is None:
        dy = dx
    # v4.16.0 (Agent A xp-dispatch walker): route through ``_xp_of``
    # so CuPy / JAX focal-plane inputs stay on the dispatched
    # backend.  Pre-v4.16.0 the coordinate grid + transmission
    # tensor were ``np.arange`` / ``np.meshgrid`` / ``np.where``,
    # silently demoting any non-NumPy E_in to host NumPy and back.
    # ``np.pi`` remains as a host scalar.
    xp = _xp_of(E_in)
    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    R = xp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    R_mask = mask_diameter / 2.0

    if profile == 'hard':
        T = xp.where(R <= R_mask, 0.0, 1.0)
    elif profile == 'gaussian':
        sig = sigma if sigma is not None else mask_diameter / 6.0
        T = 1.0 - xp.exp(-R ** 2 / (2.0 * sig ** 2))
    elif profile == 'sin2':
        inside = R <= R_mask
        T = xp.where(inside,
                     xp.sin(np.pi * R / mask_diameter) ** 2,
                     1.0)
    else:
        raise ValueError(
            f"Unknown focal-plane-mask profile: {profile!r}.  "
            f"Use 'hard', 'gaussian', or 'sin2'.")

    return E_in * T


def apply_vortex_phase_mask(E_in, dx, *, charge=2, xc=0.0, yc=0.0,
                             dy=None):
    """Apply a scalar focal-plane vortex phase mask.

    Imparts an azimuthal phase ramp ``exp(1j * l * theta)`` to a
    focal-plane field, where ``l`` is the topological charge
    ("vortex charge").  Combined with a downstream Lyot stop, this is
    the vector- / scalar-vortex coronagraph -- the standard
    architecture for high-contrast imaging close to the chief ray
    (small inner working angle).

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Input focal-plane field.
    dx : float
        Focal-plane grid spacing [m].
    charge : int, default 2
        Vortex topological charge ``l``.  Standard values for
        astronomical coronagraphs are 2, 4, 6, 8 (even charges
        produce nulls at the chief; odd charges leave a residual).
        Charge 2 is the AGPM design [2]; charge 4 trades inner
        working angle for tighter null and is the standard for
        ground-based ELT-class systems.
    xc, yc : float, default 0
        Vortex centre [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Field with the vortex phase applied.  Amplitude is unchanged
        (this is a pure phase mask); the energy redistribution
        happens at the downstream Lyot stop.

    Notes
    -----
    The phase at the exact centre (``r = 0``) is mathematically
    undefined; this routine sets it to zero, which is the standard
    convention for numerical simulations.

    A perfect scalar vortex requires zero-thickness at the singularity,
    so the simulated mask still aliases slightly even at very fine
    grids.  Use an oversampled focal-plane sampling
    (:func:`fraunhofer_propagate_mft` with a small ``dx_out``) to
    minimise the alias.
    """
    if dy is None:
        dy = dx
    # v4.16.0 (Agent A xp-dispatch walker): route through ``_xp_of``
    # so CuPy / JAX inputs stay on the dispatched backend.
    # Pre-v4.16.0 the coordinate grid + transmission tensor were
    # ``np.arange`` / ``np.meshgrid`` / ``np.where``, silently
    # demoting non-NumPy E_in to host NumPy and back.
    xp = _xp_of(E_in)
    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    Xc = X - xc
    Yc = Y - yc

    theta = xp.arctan2(Yc, Xc)
    # Zero out the centre pixel to avoid the arctan2 discontinuity.
    centre = (xp.abs(Xc) < 0.5 * dx) & (xp.abs(Yc) < 0.5 * dy)
    phase = xp.where(centre, 0.0, int(charge) * theta)

    return E_in * xp.exp(1j * phase)


def create_four_quadrant_phase_mask(N, dx, *, phase_step=np.pi, center=None):
    """Construct a four-quadrant phase mask (FQPM).

    The four-quadrant phase mask is a focal-plane coronagraph element
    that imparts a ``phase_step`` retardation on quadrants 1 + 3 and
    zero retardation on quadrants 2 + 4 (or equivalently the opposite
    pairing, depending on the sign convention).  For ``phase_step = pi``
    it is a perfect on-axis nuller for circularly symmetric pupils
    (Rouan et al. 2000).

    Parameters
    ----------
    N : int
        Side length of the (N, N) output grid in pixels.
    dx : float
        Focal-plane grid spacing [m].  Retained for API symmetry with
        the other ``make_*_mask`` builders and for downstream centering
        in physical units; the mask itself is dimensionless.
    phase_step : float, default ``numpy.pi``
        Phase shift [rad] applied on the +/- diagonal quadrants.
        ``pi`` is the canonical FQPM; other values give partial nullers.
    center : tuple[int, int] or None, optional
        ``(row, col)`` pixel index of the mask centre.  ``None``
        defaults to the grid centre ``(N // 2, N // 2)``.

    Returns
    -------
    mask : ndarray (complex, N x N)
        Complex transmission ``exp(1j * phase_step)`` on quadrants
        where ``(x * y) > 0`` and ``1 + 0j`` elsewhere.  Magnitude is
        unity everywhere.

    References
    ----------
    Rouan, D. et al. (2000).  "The Four-Quadrant Phase-Mask
    Coronagraph. I. Principle."  PASP 112, 1479.
    """
    # v5.4 Phase 5: builder for the canonical FQPM coronagraph mask.
    # Returns a complex (N, N) ndarray that callers multiply into the
    # focal-plane field.  Implementation stays on NumPy (no E_in
    # backend to dispatch from); the multiplication site dispatches.
    #
    # Coordinate convention: column axis maps to ``x`` and row axis
    # maps to ``y`` via ``y = (arange - cy) * dx`` -- the same
    # convention used by ``apply_lyot_focal_plane_mask`` etc.  In
    # screen coordinates (row 0 at the top) this means "row index
    # below center" is geometrically the upper half.  FQPM
    # quadrants 1+3 (upper-right + lower-left in screen orientation,
    # i.e. ``mask[row < cy, col > cx]`` and ``mask[row > cy, col <
    # cx]``) carry the ``phase_step`` shift.  These quadrants
    # correspond to opposite signs of ``x`` and ``y``, hence the
    # ``X * Y < 0`` selector below.
    cy, cx = (N // 2, N // 2) if center is None else center
    x = (np.arange(N) - cx) * dx
    y = (np.arange(N) - cy) * dx
    X, Y = np.meshgrid(x, y)
    # ``X * Y < 0`` selects quadrants 1 (upper-right) and 3
    # (lower-left) in screen orientation; the complement (including
    # the axes themselves) gets exp(0) = 1.
    mask = np.where(X * Y < 0,
                    np.exp(1j * phase_step),
                    1.0 + 0.0j).astype(np.complex128)
    return mask


def create_eight_octant_phase_mask(N, dx, *, phase_step=np.pi, center=None):
    """Construct an eight-octant phase mask (8OPM).

    The eight-octant phase mask refines the four-quadrant design by
    splitting each 90-degree quadrant into two 45-degree octants and
    alternating the phase between adjacent octants.  Compared to the
    FQPM it tightens the inner working angle and breaks the residual
    on-axis leakage from non-circular pupils (Murakami et al. 2008).

    Parameters
    ----------
    N : int
        Side length of the (N, N) output grid in pixels.
    dx : float
        Focal-plane grid spacing [m].  Retained for API symmetry with
        the other ``make_*_mask`` builders.
    phase_step : float, default ``numpy.pi``
        Phase shift [rad] applied on alternating octants.  ``pi`` is
        the canonical 8OPM.
    center : tuple[int, int] or None, optional
        ``(row, col)`` pixel index of the mask centre.  ``None``
        defaults to the grid centre ``(N // 2, N // 2)``.

    Returns
    -------
    mask : ndarray (complex, N x N)
        Complex transmission alternating between ``exp(1j * phase_step)``
        and ``1 + 0j`` on the eight angular octants.  Magnitude is
        unity everywhere.

    References
    ----------
    Murakami, N. et al. (2008).  "Eight-Octant Phase-Mask
    Coronagraph."  PASP 120, 1112.
    """
    # v5.4 Phase 5: builder for the canonical 8OPM coronagraph mask.
    # ``theta + pi`` wraps arctan2's [-pi, pi] return into [0, 2*pi];
    # dividing by pi/4 buckets it into eight contiguous octants.
    cy, cx = (N // 2, N // 2) if center is None else center
    x = (np.arange(N) - cx) * dx
    y = (np.arange(N) - cy) * dx
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    octant = np.floor((theta + np.pi) / (np.pi / 4.0)).astype(int)
    # ``octant & 1`` alternates 0/1 across the 8 sectors.
    phase = np.where((octant & 1) == 0, 0.0, phase_step)
    mask = np.exp(1j * phase).astype(np.complex128)
    return mask


def apply_lyot_stop(E_in, dx, *, outer_diameter, inner_diameter=0.0,
                     xc=0.0, yc=0.0, dy=None):
    """Apply a downstream Lyot-stop pupil aperture.

    Hard-edge annular aperture used in the pupil plane downstream of
    a coronagraphic focal-plane mask.  Functionally equivalent to
    ``apply_aperture(..., shape='annular', ...)`` but named to match
    coronagraph literature.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Pupil-plane field (typically the result of a Fraunhofer
        back-propagation from the masked focal plane).
    dx : float
        Pupil-plane grid spacing [m].
    outer_diameter : float
        Outer diameter of the Lyot stop [m].  Typically a fraction
        (0.90 - 0.95) of the entrance-pupil diameter to remove the
        peripheral diffracted starlight.
    inner_diameter : float, default 0
        Inner diameter (central obstruction) [m].  Use a non-zero
        value for systems with a secondary obscuration (e.g. on-axis
        telescopes) or for an aggressive Lyot design that strips out
        the on-axis bright spot.
    xc, yc : float, default 0
        Stop centre [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Field with the annular Lyot stop applied (zeroed outside the
        annulus).

    Notes
    -----
    The optimal Lyot stop is design-specific: for a hard-edge focal
    plane mask the rule of thumb is ``outer = 0.85 * D_entrance`` and
    ``inner = 0`` for circular pupils; vortex coronagraphs typically
    use a tighter outer of ``0.95 * D``.  For arbitrary geometries
    use :func:`apply_aperture` directly to build a custom Lyot stop.
    """
    return apply_aperture(
        E_in, dx,
        shape='annular',
        params={'inner_diameter': inner_diameter,
                'outer_diameter': outer_diameter},
        xc=xc, yc=yc, dy=dy)


def apply_apodized_pupil(E_in, dx, diameter, *,
                         apodization='cos2', exponent=2,
                         sigma=None, xc=0.0, yc=0.0, dy=None):
    """Apply an entrance-pupil apodizer (graded-transmission aperture).

    Replaces the hard-edge pupil with a smooth-edged amplitude
    distribution that suppresses the high-frequency diffraction
    wings of the PSF.  Used at the entrance pupil of apodized-pupil
    Lyot coronagraphs (APLC) [3] and shaped-pupil coronagraphs [4]
    -- a simpler-to-evaluate analog of the Kasdin / Soummer optimal
    apodizations.

    Parameters
    ----------
    E_in : ndarray (complex, Ny x Nx)
        Pupil-plane field.
    dx : float
        Pupil-plane grid spacing [m].
    diameter : float
        Apodizer outer diameter [m] (transmission is zero outside).
    apodization : ``'cos2'`` / ``'cos_power'`` / ``'gaussian'`` / ``'sonine'``
        Radial transmission profile, all functions of ``rho = 2 r / D``
        in [0, 1] (zero outside):

        * ``'cos2'`` (default): ``T(rho) = cos^2(pi/2 * rho)``.
          Classic Hanning-window-style soft edge.  PSF first-null at
          ``~1.5 lambda f / D``, sidelobes ~ 30 dB below peak.
        * ``'cos_power'``: ``T(rho) = cos^n(pi/2 * rho)`` with
          ``n = exponent``.  Higher ``n`` = smoother edge, fainter
          sidelobes, wider core.
        * ``'gaussian'``: ``T(rho) = exp(-(r/sigma)^2 / 2)``.  Requires
          ``sigma`` [m].  Pure Gaussian apodisation (Strehl-optimal for
          a fixed-area aperture).
        * ``'sonine'``: ``T(rho) = (1 - rho^2)^exponent``.  The Sonine /
          Bracewell family.  ``exponent = 1`` is the simple Bartlett
          / cosine taper, higher integer values give faster sidelobe
          rolloff.
    exponent : int or float, default 2
        Exponent for ``cos_power`` and ``sonine`` profiles.  Ignored
        for ``'cos2'`` and ``'gaussian'``.
    sigma : float, optional
        Width parameter [m] for ``'gaussian'``.  Defaults to
        ``diameter / 6`` (3-sigma at the aperture edge).
    xc, yc : float, default 0
        Apodizer centre [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E_out : ndarray (complex, Ny x Nx)
        Apodized field.  Energy removed by the apodization is NOT
        renormalised back; the Strehl ratio (with respect to a unit
        plane wave) drops accordingly.

    Notes
    -----
    These analytic apodisations are useful baselines but not optimal
    in the Kasdin / Soummer / Vanderbei sense -- those require a
    pupil-specific numerical optimisation problem to solve.  For a
    target contrast curve, use these as starting points and refine
    with the prolate-spheroidal or shaped-pupil designs from the
    coronagraph-design literature.

    References
    ----------
    [3] Soummer, R. (2005).  ApJ 618, L161.
    [4] Kasdin, N.J. et al. (2003).  ApJ 582, 1147.
    """
    if dy is None:
        dy = dx
    # v4.16.0 (Agent A xp-dispatch walker): route through ``_xp_of``
    # so CuPy / JAX pupil inputs stay on the dispatched backend.
    # Pre-v4.16.0 the coordinate grid + transmission tensor were
    # ``np.arange`` / ``np.meshgrid`` / ``np.where``, silently
    # demoting non-NumPy E_in to host NumPy and back.
    xp = _xp_of(E_in)
    Ny, Nx = E_in.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)
    R = xp.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    R_max = diameter / 2.0
    rho = R / R_max
    inside = rho <= 1.0

    if apodization == 'cos2':
        T = xp.where(inside, xp.cos(0.5 * np.pi * rho) ** 2, 0.0)
    elif apodization == 'cos_power':
        T = xp.where(inside,
                     xp.cos(0.5 * np.pi * rho) ** float(exponent),
                     0.0)
    elif apodization == 'gaussian':
        sig = sigma if sigma is not None else diameter / 6.0
        T = xp.where(inside,
                     xp.exp(-R ** 2 / (2.0 * sig ** 2)),
                     0.0)
    elif apodization == 'sonine':
        T = xp.where(inside,
                     xp.clip(1.0 - rho ** 2, 0.0, 1.0) ** float(exponent),
                     0.0)
    else:
        raise ValueError(
            f"Unknown apodization profile: {apodization!r}.  "
            f"Use 'cos2', 'cos_power', 'gaussian', or 'sonine'.")

    return E_in * T


def coronagraph_contrast_curve(*args, **kwargs):
    """Backward-compatibility shim; canonical home is
    :func:`lumenairy.analysis.coronagraph.coronagraph_contrast_curve`.

    This function is a post-processing contrast analysis, not an
    element factory, and was relocated in 4.3.0.  Top-level
    ``lumenairy.coronagraph_contrast_curve`` and the
    ``lumenairy.elements`` re-export continue to work; new code
    should import from ``lumenairy.analysis.coronagraph``.
    """
    from ..analysis.coronagraph import (
        coronagraph_contrast_curve as _impl,
    )
    return _impl(*args, **kwargs)


# =============================================================================
# ATMOSPHERIC / TURBULENCE PHASE SCREENS
# =============================================================================

def generate_turbulence_screen(N, dx, r0, L0=np.inf, l0=0.0, seed=None):
    """
    Generate a random atmospheric turbulence phase screen.

    Uses the FFT-based spectral method to generate a phase screen with
    either Kolmogorov (default) or von Karman statistics.

    Parameters
    ----------
    N : int
        Grid size (N×N).

    dx : float
        Grid spacing [m].

    r0 : float
        Fried parameter (coherence diameter) [m]. Smaller r0 = stronger
        turbulence. Typical values: 1-20 cm for ground-level propagation.

    L0 : float, default inf
        Outer scale [m]. Sets the largest turbulence eddy size.
        inf = Kolmogorov (no outer scale cutoff).
        Typical values: 1-100 m.

    l0 : float, default 0
        Inner scale [m]. Sets the smallest turbulence eddy size.
        0 = no inner scale cutoff.
        Typical values: 1-10 mm.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    phase_screen : ndarray (real, N×N)
        Phase screen in radians. Apply as ``E_out = E_in * exp(1j * screen)``.

    Notes
    -----
    The phase power spectral density (PSD) is:

    Kolmogorov:
        PSD(f) = 0.023 * r0^(-5/3) * f^(-11/3)

    von Karman (with outer and inner scale):
        PSD(f) = 0.023 * r0^(-5/3) * (f^2 + 1/L0^2)^(-11/6)
                 * exp(-(2*pi*f*l0/5.92)^2)

    The inner scale cutoff uses kappa_m = 5.92/l0, so in terms of
    spatial frequency f the exponential factor is
    exp(-(2*pi*f*l0/5.92)^2), which suppresses eddies smaller than l0.

    The screen is generated by filtering white noise with the square root
    of the PSD in the frequency domain, then inverse-transforming.

    The structure function of the resulting screen follows:
        D(r) = 6.88 * (r/r0)^(5/3)   for Kolmogorov

    References
    ----------
    [1] Schmidt, J.D. "Numerical Simulation of Optical Wave Propagation"
        (SPIE Press, 2010), Ch. 9.
    [2] Lane, R.G. et al. (1992). "Simulation of a Kolmogorov phase screen."
        Waves in Random Media 2(3): 209-224.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    L = N * dx  # grid extent
    df = 1.0 / L  # frequency spacing

    # Frequency grid (centered)
    fx = (np.arange(N) - N / 2) * df
    FX, FY = np.meshgrid(fx, fx)
    f_sq = FX**2 + FY**2
    f_mag = np.sqrt(f_sq)

    # Avoid division by zero at DC
    f_sq_safe = np.where(f_sq > 0, f_sq, 1.0)

    # von Karman PSD (reduces to Kolmogorov when L0=inf, l0=0)
    psd = 0.023 * r0**(-5.0/3.0) * (f_sq_safe + 1.0 / L0**2)**(-11.0/6.0)

    # Inner scale cutoff
    if l0 > 0:
        # Inner scale cutoff: kappa_m = 5.92/l0, kappa = 2*pi*f
        # exp(-(kappa/kappa_m)^2) = exp(-(2*pi*f*l0/5.92)^2)
        psd *= np.exp(-(f_mag * l0 * 2 * np.pi / 5.92)**2)

    # Zero DC
    psd[N // 2, N // 2] = 0.0

    # Generate random complex coefficients
    noise = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))

    # Filter noise with sqrt(PSD) and transform to spatial domain.
    # The sqrt(2) factor compensates for the halving of variance when
    # taking the real part of the complex IFFT result. Verified against
    # the Kolmogorov structure function D(r=r0) = 6.88.
    amplitude = np.sqrt(2.0 * psd) * df
    phase_fft = noise * amplitude

    # Inverse FFT to get spatial phase screen
    phase_screen = np.real(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phase_fft)))
    ) * N**2

    return phase_screen
