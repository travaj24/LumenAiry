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

import numpy as np
from math import factorial

from .lenses import _surface_sag_general


# =============================================================================
# MIRRORS
# =============================================================================

def apply_mirror(E_in, wavelength, dx, radius=None, conic=0.0,
                 aperture_diameter=None, xc=0, yc=0):
    """
    Apply a mirror reflection to an optical field.

    For a flat mirror, this simply reverses the propagation direction
    (the field is unchanged apart from an optional aperture). For a
    curved mirror, the reflection imparts a focusing phase equivalent
    to a thin lens with f = R/2.

    Parameters
    ----------
    E_in : ndarray (complex, N×N)
        Input electric field.

    wavelength : float
        Free-space wavelength [m].

    dx : float
        Grid spacing [m].

    radius : float or None
        Radius of curvature of the mirror [m].
        Positive = concave (focusing), negative = convex (diverging).
        None or inf = flat mirror.

    conic : float, default 0.0
        Conic constant of the mirror surface (0=sphere, -1=paraboloid).

    aperture_diameter : float or None
        Clear aperture [m]. If None, no aperture is applied.

    xc, yc : float, default 0
        Mirror center offset [m].

    Returns
    -------
    E_out : ndarray (complex, N×N)
        Reflected field. After this call, subsequent ASM propagation
        models the return path (caller is responsible for using the
        correct propagation distances and sign conventions).

    Notes
    -----
    A concave mirror with radius R acts like a converging lens with
    focal length f = R/2. The reflected phase is:

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
    Ny, Nx = E_in.shape
    k = 2 * np.pi / wavelength

    E = E_in.copy()

    # Apply aperture
    if aperture_diameter is not None:
        x = (np.arange(Nx) - Nx / 2) * dx
        y = (np.arange(Ny) - Ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        h_sq = (X - xc)**2 + (Y - yc)**2
        E = np.where(h_sq <= (aperture_diameter / 2)**2, E, 0.0 + 0.0j)

    # Curved mirror: apply focusing phase
    if radius is not None and not np.isinf(radius):
        x = (np.arange(Nx) - Nx / 2) * dx
        y = (np.arange(Ny) - Ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        h_sq = (X - xc)**2 + (Y - yc)**2

        # Compute surface sag
        sag = _surface_sag_general(h_sq, radius, conic)

        # Double-pass OPD: ray travels sag down to the surface and sag
        # back up, so total extra path = 2 * sag.
        # Phase delay (negative sign, same convention as apply_real_lens).
        phase = -2 * k * sag
        # Compute exp() in complex128 for phase precision, then cast
        # back to E's dtype so a complex64 input stays complex64.
        phase_exp = np.exp(1j * phase)
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

    Ny, Nx = E_in.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

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
        mask = (np.abs(X - xc) <= Wx / 2) & (np.abs(Y - yc) <= Wy / 2)

    else:
        raise ValueError(f"Unknown aperture shape: {shape!r}. "
                         f"Use 'circular', 'annular', or 'rectangular'.")

    return np.where(mask, E_in, 0.0 + 0.0j)


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
    Ny, Nx = E_in.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_sq = (X - xc)**2 + (Y - yc)**2

    return E_in * np.exp(-r_sq / (2 * sigma**2))


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

    # Radial polynomial R_n^|m|
    m_abs = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1)**s * factorial(n - s)
                 / (factorial(s) * factorial((n + m_abs) // 2 - s)
                    * factorial((n - m_abs) // 2 - s)))
        R = R + coeff * rho**(n - 2 * s)

    # Azimuthal part
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(-m * theta)
    else:
        Z = R

    # Normalization (Born & Wolf convention)
    if m == 0:
        norm = np.sqrt(n + 1)
    else:
        norm = np.sqrt(2 * (n + 1))

    Z = norm * Z

    # Zero outside unit circle
    Z = np.where(rho <= 1.0, Z, 0.0)

    return Z


def apply_zernike_aberration(E_in, dx, coefficients, aperture_radius):
    """
    Apply Zernike polynomial aberrations to a field.

    Parameters
    ----------
    E_in : ndarray (complex, N×N)
        Input field.
    dx : float
        Grid spacing [m].
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

    Returns
    -------
    E_out : ndarray (complex, N×N)

    Examples
    --------
    >>> # Add 1 wave of spherical aberration over a 5mm aperture
    >>> E_out = apply_zernike_aberration(E_in, dx=2e-6,
    ...     coefficients={(4, 0): 1.0}, aperture_radius=5e-3)
    """
    Ny, Nx = E_in.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dx
    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2) / aperture_radius
    theta = np.arctan2(Y, X)

    phase = np.zeros((Ny, Nx))
    for (n, m), amplitude in coefficients.items():
        phase += amplitude * zernike(n, m, rho, theta)

    # Convert from waves to radians
    return E_in * np.exp(1j * 2 * np.pi * phase)


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
