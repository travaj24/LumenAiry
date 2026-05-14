"""
Beam source and mode generators for optical propagation simulations.

This module provides functions to create common laser beam profiles on a
discrete 2-D grid:

- Fundamental Gaussian beams (with optional GPU acceleration via CuPy)
- Hermite-Gaussian (HG_mn) modes
- Laguerre-Gaussian (LG_pl) modes

All fields are returned at the beam waist (flat phase) and are suitable for
use as input to angular-spectrum or other propagation routines.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import importlib.util as _importlib_util
CUPY_AVAILABLE = _importlib_util.find_spec('cupy') is not None
cp = None  # populated lazily on first use


def _ensure_cupy_loaded():
    global cp
    if cp is None and CUPY_AVAILABLE:
        import cupy as _c
        cp = _c
    return cp is not None


# ---------------------------------------------------------------------------
# Fundamental Gaussian beam
# ---------------------------------------------------------------------------

def create_gaussian_beam(
    N: Union[int, Tuple[int, int]],
    dx: float,
    wavelength: float,
    *,
    sigma: float,
    x0: float = 0,
    y0: float = 0,
    use_gpu: bool = False,
    dy: Optional[float] = None,
    normalize: str = 'peak',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Gaussian beam field.

    Parameters
    ----------
    N : int or tuple
        Grid size. If int, creates an N x N grid. If tuple, interpreted as
        (Ny, Nx).
    dx : float
        Grid spacing in x [m].
    sigma : float
        Gaussian width parameter (field standard deviation) [m].
        The 1/e field amplitude radius is sigma * sqrt(2).
        The 1/e^2 intensity radius (beam waist w0) is also sigma * sqrt(2).
    wavelength : float, optional
        Reserved for future use (e.g. adding a spherical phase for a
        focused beam). Currently unused -- the returned field has flat phase.
    x0, y0 : float, default 0
        Center position of the beam [m].
    use_gpu : bool, default False
        If True and CuPy is available, create the arrays on the GPU.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    normalize : ``'peak'`` (default) / ``'power'`` / ``'none'``
        Output scaling.  ``'peak'`` (the historical default) returns
        a unit-peak amplitude field.  ``'power'`` returns a
        unit-integrated-power field (matching :func:`create_hermite_gauss`
        and :func:`create_laguerre_gauss` -- pass ``normalize='power'``
        whenever you want to chain or compare across the mode-family
        helpers).  ``'none'`` returns the raw ``exp(-r^2/(2 sigma^2))``
        without scaling.

    Returns
    -------
    E : ndarray, complex
        Gaussian beam field (Ny x Nx).
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    Signature is ``(N, dx, wavelength, *, sigma, ...)`` since 4.7.
    Prior to 4.7 the ordering was
    ``(N, dx, sigma, wavelength=None, ...)`` with positional ``sigma``;
    the new style places ``wavelength`` at the third positional
    slot (matching every other source factory) and makes
    ``sigma`` keyword-only.
    """
    if CUPY_AVAILABLE and use_gpu:
        xp = cp
    else:
        xp = np
    if dy is None:
        dy = dx

    if isinstance(N, int):
        Ny, Nx = N, N
    else:
        Ny, Nx = N

    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    # Gaussian amplitude: exp(-r^2 / (2 sigma^2))
    E = xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    E = E.astype(complex)

    if normalize == 'peak':
        pass  # already peak == 1 from exp(0) at the centre
    elif normalize == 'power':
        norm = xp.sqrt(xp.sum(xp.abs(E) ** 2) * dx * dy)
        if float(norm) > 0:
            E = E / norm
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_gaussian_beam: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Hermite-Gaussian modes
# ---------------------------------------------------------------------------

def hermite_physicist(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the physicist's Hermite polynomial H_n(x) via recurrence.

    Uses the three-term recurrence relation:

        H_0(x) = 1
        H_1(x) = 2x
        H_k(x) = 2x H_{k-1}(x) - 2(k-1) H_{k-2}(x)

    Parameters
    ----------
    n : int
        Polynomial order (>= 0).
    x : ndarray
        Points at which to evaluate H_n.

    Returns
    -------
    H_n : ndarray
        Values of the physicist's Hermite polynomial of order *n*.
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev2 = np.ones_like(x)
        H_prev1 = 2 * x
        for k in range(2, n + 1):
            H_curr = 2 * x * H_prev1 - 2 * (k - 1) * H_prev2
            H_prev2 = H_prev1
            H_prev1 = H_curr
        return H_curr


def create_hermite_gauss(
    N: Union[int, Tuple[int, int]],
    dx: float,
    w0: float,
    wavelength: float,
    m: int = 0,
    n: int = 0,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    normalize: str = 'power',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Hermite-Gaussian (HG_mn) beam mode at the waist.

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size.  If a scalar, an ``N x N`` square grid is built.  Pass
        a 2-tuple ``(Ny, Nx)`` for rectangular grids.
    dx : float
        Grid spacing in x [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    m, n : int, default 0
        Transverse mode indices. HG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square pitch).
        Provide explicitly for rectangular-pitch grids so the
        Gaussian envelope isn't silently stretched along y.

    Returns
    -------
    E : ndarray, complex (Ny x Nx)
        Hermite-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_mn(x, y) = H_m(sqrt(2) x / w0) * H_n(sqrt(2) y / w0)
                      * exp(-(x^2 + y^2) / w0^2)

    where H_m is the physicist's Hermite polynomial of order m.

    **Normalisation note.**  When ``normalize='power'``, the returned
    field is power-normalised by **numerical integration over the
    grid** (``sum(|E|^2) * dx * dy = 1``).  This is grid-exact for any
    grid that fully contains the mode (typically ``L >= 4 w0``), but
    differs from the analytical normalisation
    ``N = 1 / sqrt(2^m m! 2^n n! pi w0^2 / 2)``
    used by :func:`lumenairy.propagators.asymptotic.fit_canonical_polynomials`
    and the modal-asymptotic propagators when the grid truncates the
    Gaussian tails.  For typical wave-optics grids the two normalisations
    agree to ~1e-6; for tight grids that clip at 2-3 w0 the discrepancy
    can grow.  Use the analytical asymptotic-module normalisation
    when chaining HG modes through the modal asymptotic propagator.
    """
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    u = np.sqrt(2) * (X - x0) / w0
    v = np.sqrt(2) * (Y - y0) / w0

    Hm = hermite_physicist(m, u)
    Hn = hermite_physicist(n, v)

    gaussian = np.exp(-((X - x0)**2 + (Y - y0)**2) / w0**2)
    E = (Hm * Hn * gaussian).astype(complex)

    if normalize == 'power':
        norm = np.sqrt(np.sum(np.abs(E)**2) * dx * dy)
        if norm > 0:
            E /= norm
    elif normalize == 'peak':
        pk = float(np.abs(E).max())
        if pk > 0:
            E /= pk
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_hermite_gauss: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Laguerre-Gaussian modes
# ---------------------------------------------------------------------------

def laguerre_generalized(p: int, l_abs: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate the generalized Laguerre polynomial L_p^l(x) via recurrence.

    Uses the three-term recurrence relation:

        L_0^l(x) = 1
        L_1^l(x) = 1 + l - x
        L_k^l(x) = ((2k - 1 + l - x) L_{k-1}^l(x)
                     - (k - 1 + l) L_{k-2}^l(x)) / k

    Parameters
    ----------
    p : int
        Polynomial order (radial index, >= 0).
    l_abs : int
        Associated (generalized) index (|l|, >= 0).
    x : ndarray
        Points at which to evaluate L_p^{l_abs}.

    Returns
    -------
    L_p : ndarray
        Values of the generalized Laguerre polynomial.
    """
    if p == 0:
        return np.ones_like(x)
    elif p == 1:
        return 1 + l_abs - x
    else:
        L_prev2 = np.ones_like(x)
        L_prev1 = 1 + l_abs - x
        for k in range(2, p + 1):
            L_curr = ((2 * k - 1 + l_abs - x) * L_prev1
                      - (k - 1 + l_abs) * L_prev2) / k
            L_prev2 = L_prev1
            L_prev1 = L_curr
        return L_curr


def create_laguerre_gauss(
    N: Union[int, Tuple[int, int]],
    dx: float,
    w0: float,
    wavelength: float,
    p: int = 0,
    l: int = 0,
    x0: float = 0,
    y0: float = 0,
    dy: Optional[float] = None,
    normalize: str = 'power',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a Laguerre-Gaussian (LG_pl) beam mode at the waist.

    Parameters
    ----------
    N : int or (Ny, Nx)
        Grid size.  Scalar = square grid; tuple = rectangular.
    dx : float
        Grid spacing in x [m].
    w0 : float
        Beam waist (1/e^2 intensity radius) [m].
    wavelength : float
        Wavelength [m]. Currently unused -- the field is returned at the
        waist with flat phase.
    p : int, default 0
        Radial index (number of radial nodes).
    l : int, default 0
        Azimuthal index (topological charge / orbital angular momentum).
        LG_00 is the fundamental Gaussian.
    x0, y0 : float, default 0
        Beam center [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E : ndarray, complex (Ny x Nx)
        Laguerre-Gaussian mode field, power-normalised.
    x : ndarray
        1-D x-coordinate array [m].
    y : ndarray
        1-D y-coordinate array [m].

    Notes
    -----
    The (un-normalised) field is

        E_pl(r, theta) = (r sqrt(2) / w0)^|l| * L_p^|l|(2 r^2 / w0^2)
                          * exp(-r^2 / w0^2) * exp(i l theta)

    where L_p^|l| is the generalized Laguerre polynomial.

    **Normalisation note.**  ``normalize='power'`` integrates over
    the grid numerically; same caveat as :func:`create_hermite_gauss`
    -- for grids that clip the mode tails the result differs from
    the analytical LG normalisation used in
    :mod:`lumenairy.propagators.asymptotic`.  For modal-asymptotic
    chains, prefer the analytical-normalised modes built inside the
    asymptotic propagator over passing this function's output
    through ``propagate_modal_asymptotic``.
    """
    if dy is None:
        dy = dx
    if isinstance(N, (tuple, list)):
        Ny, Nx = int(N[0]), int(N[1])
    else:
        Ny = Nx = int(N)
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    theta = np.arctan2(Y - y0, X - x0)

    rho = np.sqrt(2) * r / w0
    rho_sq = rho**2

    # Generalized Laguerre polynomial L_p^|l|
    L = laguerre_generalized(p, abs(l), rho_sq)

    gaussian = np.exp(-r**2 / w0**2)
    E = (rho**abs(l) * L * gaussian * np.exp(1j * l * theta))

    if normalize == 'power':
        norm = np.sqrt(np.sum(np.abs(E)**2) * dx * dy)
        if norm > 0:
            E /= norm
    elif normalize == 'peak':
        pk = float(np.abs(E).max())
        if pk > 0:
            E /= pk
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"create_laguerre_gauss: normalize must be one of "
            f"'peak', 'power', 'none'; got {normalize!r}.")

    return E, x, y


# ---------------------------------------------------------------------------
# Off-axis / tilted plane-wave sources
# ---------------------------------------------------------------------------

def create_tilted_plane_wave(
    N: int,
    dx: float,
    wavelength: float,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    amplitude: float = 1.0,
    dy: Optional[float] = None,
    *,
    angle_x_deg: Optional[float] = None,
    angle_y_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a tilted (off-axis) plane wave on an N x N grid.

    A tilted plane wave has a linear phase ramp across the pupil,
    representing a collimated beam arriving from a direction offset
    from the optical axis by ``angle_x`` (horizontal) and ``angle_y``
    (vertical).  This is the standard source for evaluating off-axis
    imaging performance -- pass it through ``apply_real_lens`` and
    compare the exit-pupil OPD or PSF to the on-axis case.

    Parameters
    ----------
    N : int
        Grid dimension (square N x N).
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    angle_x : float, default 0
        Field angle in the x-z plane [rad].  Positive = source
        tilted toward +x.
    angle_y : float, default 0
        Field angle in the y-z plane [rad].  Positive = source
        tilted toward +y.
    angle_x_deg, angle_y_deg : float, optional
        Same as ``angle_x`` / ``angle_y`` but expressed in degrees.
        When provided, these take precedence over the radian forms
        and provide a convenience for human-readable field-angle
        sweeps.  4.7+: the library is converging on ``*_deg`` as the
        canonical user-facing angle unit; the bare-radians
        ``angle_x`` / ``angle_y`` will become deprecated aliases in
        a future release.
    amplitude : float, default 1
        Uniform amplitude.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    E : ndarray, complex, shape (N, N)
        Complex field on the grid.
    x, y : ndarray
        1-D coordinate arrays [m].
    """
    if angle_x_deg is not None:
        angle_x = float(np.radians(angle_x_deg))
    if angle_y_deg is not None:
        angle_y = float(np.radians(angle_y_deg))
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    phase = k0 * (np.sin(angle_x) * X + np.sin(angle_y) * Y)
    E = amplitude * np.exp(1j * phase)
    return E, x, y


def create_point_source(
    N: int,
    dx: float,
    wavelength: float,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    amplitude: float = 1.0,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a diverging spherical wave from a point source at
    ``(x0, y0, z0)`` evaluated at z=0.

    For ``z0 < 0`` the source is *before* the grid (diverging);
    for ``z0 > 0`` it is *after* (converging).  ``z0 = 0`` gives
    a delta at (x0, y0).

    Parameters
    ----------
    N, dx, wavelength : usual
    x0, y0 : float
        Transverse position of the point source [m].
    z0 : float
        Axial position of the point source [m] relative to the
        grid plane at z=0.  Negative = source before grid (diverging).
    amplitude : float
    dy : float, optional

    Returns
    -------
    E : ndarray, complex, shape (N, N)
    x, y : ndarray
    """
    if dy is None:
        dy = dx
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y)
    k0 = 2 * np.pi / wavelength
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + z0 ** 2)
    r = np.maximum(r, 1e-30)
    E = amplitude * np.exp(1j * k0 * r) / r
    return E, x, y


def create_multi_field_sources(
    N: int,
    dx: float,
    wavelength: float,
    field_angles: Sequence[Union[float, Tuple[float, float]]],
    amplitude: float = 1.0,
    dy: Optional[float] = None,
) -> Tuple[List[Tuple[np.ndarray, float, float]], np.ndarray, np.ndarray]:
    """Generate a list of tilted plane waves at the given field angles.

    Convenience wrapper around :func:`create_tilted_plane_wave` for
    setting up multi-field analyses.

    Parameters
    ----------
    N, dx, wavelength : usual
    field_angles : sequence of float or tuple
        Each element is either a scalar (y-tilt only) or a
        ``(angle_x, angle_y)`` tuple.
    amplitude : float
    dy : float, optional

    Returns
    -------
    sources : list of (E, angle_x, angle_y)
        One per field angle.  **Note the return shape differs from
        the scalar ``create_*`` helpers** (which return ``(E, x, y)``
        directly): this is a *list of tilted-plane-wave sources*,
        not a single field.
    x, y : ndarray
        Shared 1-D coordinate arrays.
    """
    sources = []
    x = y = None
    for a in field_angles:
        if isinstance(a, (list, tuple)):
            ax, ay = float(a[0]), float(a[1])
        else:
            ax, ay = 0.0, float(a)
        E, x, y = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=ax, angle_y=ay,
            amplitude=amplitude, dy=dy)
        sources.append((E, ax, ay))
    return sources, x, y


# ---------------------------------------------------------------------------
# Extended source models (LED, fiber, top-hat, annular, Bessel)
# ---------------------------------------------------------------------------

def create_top_hat_beam(
    N: int,
    dx: float,
    wavelength: float,
    *,
    diameter: float,
    x0: float = 0,
    y0: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-intensity circular beam (top-hat / flat-top).

    Parameters
    ----------
    N, dx : int, float
    wavelength : float
        Reserved for future use (overlay of a spherical-phase term);
        currently does not affect the returned field.
    diameter : float
        Beam diameter [m].
    x0, y0 : float
        Center [m].

    Returns
    -------
    E, x, y : ndarray

    Notes
    -----
    Signature is ``(N, dx, wavelength, *, diameter, ...)`` since 4.7.
    Prior to 4.7 the ordering was
    ``(N, dx, diameter, wavelength=None, ...)``.
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where(r <= diameter / 2, 1.0, 0.0).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx ** 2)
    if norm > 0:
        E /= norm
    return E, x, y


def create_annular_beam(
    N: int,
    dx: float,
    wavelength: float,
    *,
    outer_diameter: float,
    inner_diameter: float,
    x0: float = 0,
    y0: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Annular (donut) beam.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float
        Reserved for future use.
    outer_diameter, inner_diameter : float [m]

    Returns
    -------
    E, x, y : ndarray

    Notes
    -----
    Signature is
    ``(N, dx, wavelength, *, outer_diameter, inner_diameter, ...)``
    since 4.7.
    """
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    E = np.where((r <= outer_diameter / 2) & (r >= inner_diameter / 2),
                  1.0, 0.0).astype(np.complex128)
    norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx ** 2)
    if norm > 0:
        E /= norm
    return E, x, y


def create_fiber_mode(
    N: int,
    dx: float,
    wavelength: float,
    *,
    mode_field_diameter: float,
    x0: float = 0,
    y0: float = 0,
    na: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-mode fiber output (Gaussian with NA-defined divergence).

    The mode-field diameter (MFD) is the 1/e^2 intensity diameter.
    The field is a Gaussian with w0 = MFD/2, and the NA is encoded
    in the far-field divergence angle.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float [m]
    mode_field_diameter : float [m]
    x0, y0 : float
    na : float
        Fiber numerical aperture (informational; the near-field
        profile is MFD-determined).

    Notes
    -----
    Signature is
    ``(N, dx, wavelength, *, mode_field_diameter, ...)`` since 4.7.

    Returns
    -------
    E, x, y : ndarray
    """
    w0 = mode_field_diameter / 2.0
    sigma = w0 / np.sqrt(2)
    return create_gaussian_beam(N, dx, wavelength, sigma=sigma,
                                 x0=x0, y0=y0)


def create_led_source(
    N: int,
    dx: float,
    diameter: float,
    divergence_angle: float,
    wavelength: float,
    x0: float = 0,
    y0: float = 0,
) -> Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray, np.ndarray]:
    """Lambertian LED source (incoherent; returns the intensity
    envelope as a complex field for use with partial-coherence
    imaging).

    The spatial extent is a uniform disk of given diameter; the
    angular extent (divergence) determines how many source angles
    to sample when using ``koehler_image`` or
    ``extended_source_image``.

    Parameters
    ----------
    N, dx : int, float
    diameter : float [m]
        Emitting area diameter.
    divergence_angle : float [rad]
        Half-angle of the emission cone.
    wavelength : float [m]

    Returns
    -------
    E : ndarray (complex)
        Amplitude envelope (uniform inside disk, zero outside).
    source_angles : list of (float, float)
        Suggested source angles for partial-coherence integration,
        covering the divergence cone with ~21 samples.
    x, y : ndarray
    """
    E, x, y = create_top_hat_beam(N, dx, wavelength, diameter=diameter, x0=x0, y0=y0)
    # Generate suggested source angles
    n_ring = 3
    angles = [(0.0, 0.0)]
    for ring in range(1, n_ring + 1):
        r = divergence_angle * ring / n_ring
        for k in range(6 * ring):
            theta = 2 * np.pi * k / (6 * ring)
            angles.append((r * np.cos(theta), r * np.sin(theta)))
    return E, angles, x, y


def create_bessel_beam(
    N: int,
    dx: float,
    wavelength: float,
    cone_angle: float,
    x0: float = 0,
    y0: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal Bessel beam (J_0 profile).

    Creates the field proportional to J_0(k_r * r) where
    k_r = k * sin(cone_angle).  This is an idealized non-diffracting
    beam; in practice it's produced by an axicon or annular aperture.

    Parameters
    ----------
    N, dx : int, float
    wavelength : float [m]
    cone_angle : float [rad]
        Half-angle of the Bessel cone.

    Returns
    -------
    E, x, y : ndarray
    """
    from scipy.special import j0

    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    k_r = 2 * np.pi / wavelength * np.sin(cone_angle)
    E = j0(k_r * r).astype(np.complex128)
    return E, x, y


# ---------------------------------------------------------------------------
# Source -- bundles E + dx + wavelength + source_point with chainable
# .propagate(...) and .from_X(...) factories
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as _dc_field
from typing import Optional as _Optional, Tuple as _Tuple


@dataclass
class Source:
    """A complex field on a regular grid with the metadata it needs to
    propagate through the rest of lumenairy.

    Bundles the four pieces of state every propagator wants -- ``E``,
    ``dx``, ``wavelength``, and (for asymptotic / aberration-tensor
    work) ``source_point`` -- so callers don't have to repeat them.

    Attributes
    ----------
    E : ndarray, complex
        Field on a (Ny, Nx) grid.
    dx : float
        Sample spacing [m].
    wavelength : float
        Vacuum wavelength [m].
    source_point : (float, float), default (0, 0)
        Object-plane location used by the asymptotic / LG-tensor
        propagators.  Ignored by ASM / GBD / HFPI / HF.
    name : str, optional
        Human-readable label, propagated to descendants for tracing.
    """
    E: 'object'  # numpy or cupy or jax ndarray
    dx: float
    wavelength: float
    source_point: _Tuple[float, float] = (0.0, 0.0)
    name: _Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.E.shape[-2:])

    def __repr__(self) -> str:
        return (f"Source(shape={self.shape}, dx={self.dx:.3g}m, "
                f"wavelength={self.wavelength*1e9:.1f}nm, "
                f"source_point={self.source_point}, "
                f"name={self.name!r})")

    def propagate(self,
                   *,
                   method: str = 'auto',
                   z: _Optional[float] = None,
                   prescription: _Optional[dict] = None,
                   **kwargs) -> 'Source':
        """Propagate this Source via :func:`la.propagate` and return
        the result as a new Source on the (possibly resampled) output
        plane.

        ``z``, ``prescription``, ``method``, and any propagator-
        specific keyword arguments are forwarded to the dispatcher.
        ``dx`` defaults to the input spacing; pass ``output_dx`` to
        change it.  Wavelength, source_point, and name are inherited
        on the new Source (with ``->{method}`` appended to the name
        for trace-ability).
        """
        from ..propagators.dispatch import propagate
        E_out = propagate(
            self.E,
            wavelength=self.wavelength,
            dx=self.dx,
            z=z,
            prescription=prescription,
            method=method,
            **kwargs,
        )
        out_dx = kwargs.get('output_dx', self.dx) or self.dx
        new_name = (self.name or 'Source')
        new_name = f'{new_name}->{method}'
        return Source(
            E=E_out, dx=out_dx, wavelength=self.wavelength,
            source_point=self.source_point, name=new_name,
        )

    # -- Factories that wrap the existing create_X functions ----------

    @classmethod
    def gaussian(cls, w0: float, N: int, dx: float, wavelength: float,
                  *, x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None,
                  use_gpu: bool = False) -> 'Source':
        """Gaussian beam at the waist.  ``w0`` is the 1/e^2 intensity
        radius."""
        sigma = w0 / np.sqrt(2)
        E, _, _ = create_gaussian_beam(
            N, dx, wavelength, sigma=sigma, x0=x0, y0=y0,
            use_gpu=use_gpu)
        return cls(E=E, dx=dx, wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Gaussian(w0={w0:.2g}m)')

    @classmethod
    def plane_wave(cls, N: int, dx: float, wavelength: float,
                    *, angle_x: float = 0.0, angle_y: float = 0.0,
                    amplitude: float = 1.0,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None) -> 'Source':
        """Tilted plane wave (uses ``create_tilted_plane_wave``)."""
        E, _, _ = create_tilted_plane_wave(
            N, dx, wavelength, angle_x=angle_x, angle_y=angle_y,
            amplitude=amplitude)
        return cls(E=E, dx=dx, wavelength=wavelength,
                   source_point=source_point,
                   name=name or 'PlaneWave')

    @classmethod
    def point_source(cls, N: int, dx: float, wavelength: float,
                      *, x0: float = 0.0, y0: float = 0.0,
                      z0: float = 0.0, amplitude: float = 1.0,
                      name: _Optional[str] = None) -> 'Source':
        """Diverging spherical wave from a point at (x0, y0, z0)."""
        E, _, _ = create_point_source(
            N, dx, wavelength, x0=x0, y0=y0, z0=z0,
            amplitude=amplitude)
        return cls(E=E, dx=dx, wavelength=wavelength,
                   source_point=(float(x0), float(y0)),
                   name=name or 'PointSource')

    @classmethod
    def top_hat(cls, diameter: float, N: int, dx: float, wavelength: float,
                  *, x0: float = 0.0, y0: float = 0.0,
                  source_point: _Tuple[float, float] = (0.0, 0.0),
                  name: _Optional[str] = None) -> 'Source':
        """Uniform circular aperture beam."""
        E, _, _ = create_top_hat_beam(
            N, dx, wavelength, diameter=diameter, x0=x0, y0=y0)
        return cls(E=E, dx=dx, wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'TopHat(D={diameter:.2g}m)')

    @classmethod
    def fiber_mode(cls, mode_field_diameter: float, N: int, dx: float,
                    wavelength: float, *, x0: float = 0.0, y0: float = 0.0,
                    na: float = 0.12,
                    source_point: _Tuple[float, float] = (0.0, 0.0),
                    name: _Optional[str] = None) -> 'Source':
        """Single-mode fiber output."""
        E, _, _ = create_fiber_mode(
            N, dx, wavelength, mode_field_diameter=mode_field_diameter,
            x0=x0, y0=y0, na=na)
        return cls(E=E, dx=dx, wavelength=wavelength,
                   source_point=source_point,
                   name=name or f'Fiber(MFD={mode_field_diameter:.2g}m)')

