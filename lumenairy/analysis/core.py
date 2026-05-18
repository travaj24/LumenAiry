"""
Beam measurement and diagnostic functions for optical propagation analysis.

This module provides utilities for characterizing optical beams after
propagation, including centroid location, second-moment (D4sigma) beam
diameter, integrated power (total or power-in-bucket), Strehl ratio
computation, PSF/MTF analysis, and sampling-condition diagnostics for
the Angular Spectrum Method (ASM).

Backend awareness: the lightweight beam-statistic and PSF/MTF helpers
in this module dispatch through :func:`lumenairy.backend.array_namespace`,
so a CuPy or JAX input field flows through them without an implicit
host transfer.  Functions that pull in SciPy primitives
(``wave_opd_1d``, Zernike decomposition, ``polychromatic_*``) still
operate on the NumPy backend internally and coerce on entry.

Author: Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    # beam statistics
    'beam_centroid', 'beam_d4sigma', 'beam_power', 'radial_power_bands',
    'beam_diameter',
    # Strehl
    'strehl_ratio', 'strehl_marechal', 'strehl_phase_integral',
    # coupling + beam quality
    'coupling_efficiency', 'M2',
    # sampling diagnostics
    'check_sampling_conditions', 'check_opd_sampling',
    # PSF / OTF / MTF
    'compute_psf', 'compute_otf', 'compute_mtf', 'mtf_radial',
    'mtf_cutoff',
    # Encircled energy / spec-sheet metrics (v4.14.0)
    'encircled_energy_curve', 'encircled_energy_radius',
    # Depth of focus (v4.14.0)
    'depth_of_focus',
    # polychromatic
    'chromatic_focal_shift', 'polychromatic_strehl', 'polychromatic_psf',
    # Zernike basis + decomposition
    'zernike_index_to_nm', 'zernike_nm_to_index',
    'zernike_polynomial', 'zernike_basis_matrix',
    'zernike_decompose', 'zernike_reconstruct',
    'clear_zernike_basis_cache',
    # OPD extraction / mode removal
    'remove_wavefront_modes', 'opd_pv_rms',
    'wave_opd_1d', 'wave_opd_2d',
]


def _xp_of(*arrays):
    """Return the array namespace for the inputs (numpy / cupy / jax.numpy)."""
    from ..backend import array_namespace
    return array_namespace(*arrays)


def beam_centroid(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the centroid (center of mass) of the beam intensity.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric-field distribution.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to *dx*.

    Returns
    -------
    cx : float
        Centroid x-position [m].
    cy : float
        Centroid y-position [m].
    """
    if dy is None:
        dy = dx
    xp = _xp_of(E)
    Ny, Nx = E.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    I = xp.abs(E) ** 2
    total = xp.sum(I)
    if float(total) == 0:
        return 0.0, 0.0
    return float(xp.sum(X * I) / total), float(xp.sum(Y * I) / total)


def beam_d4sigma(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute the D4sigma (second-moment) beam diameter in x and y.

    This is the ISO 11146 standard beam-width definition:
    D4sigma = 4 * sqrt(variance of intensity distribution).

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric-field distribution.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to *dx*.

    Returns
    -------
    d4s_x : float
        D4sigma beam diameter in x [m].
    d4s_y : float
        D4sigma beam diameter in y [m].
    """
    if dy is None:
        dy = dx
    xp = _xp_of(E)
    Ny, Nx = E.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    I = xp.abs(E) ** 2
    total = xp.sum(I)
    if float(total) == 0:
        return 0.0, 0.0

    cx = xp.sum(X * I) / total
    cy = xp.sum(Y * I) / total
    var_x = xp.sum((X - cx) ** 2 * I) / total
    var_y = xp.sum((Y - cy) ** 2 * I) / total

    return float(4 * xp.sqrt(var_x)), float(4 * xp.sqrt(var_y))


def beam_power(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    region: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute total power or power-in-bucket for a complex field.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric-field distribution.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to *dx*.
    region : dict or None
        If ``None``, compute total power on the grid.
        If a dict, compute power within a spatial region.  Supported forms:

        - ``{'shape': 'circular', 'diameter': D}``
          Circle of diameter *D* centered at the origin.
        - ``{'shape': 'circular', 'diameter': D, 'xc': x, 'yc': y}``
          Circle centered at *(x, y)*.
        - ``{'shape': 'rectangular', 'width_x': Wx, 'width_y': Wy}``
          Rectangle of width *Wx* x *Wy*, optionally offset with *xc*, *yc*.

    Returns
    -------
    power : float
        Integrated power [arb. units, same as ``sum(|E|^2) * dx * dy``].
    """
    if dy is None:
        dy = dx
    xp = _xp_of(E)
    I = xp.abs(E) ** 2

    if region is None:
        return float(xp.sum(I) * dx * dy)

    Ny, Nx = E.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    shape = region.get('shape', 'circular')
    xc = region.get('xc', 0)
    yc = region.get('yc', 0)

    if shape == 'circular':
        D = region['diameter']
        mask = ((X - xc) ** 2 + (Y - yc) ** 2) <= (D / 2) ** 2
    elif shape == 'rectangular':
        Wx = region['width_x']
        Wy = region['width_y']
        mask = (xp.abs(X - xc) <= Wx / 2) & (xp.abs(Y - yc) <= Wy / 2)
    else:
        raise ValueError(f"Unknown region shape: {shape}")

    return float(xp.sum(I[mask]) * dx * dy)


def radial_power_bands(
    E: np.ndarray,
    dx: float,
    radii: Sequence[float],
    dy: Optional[float] = None,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute cumulative integrated power within concentric circular
    apertures centered on ``center`` (default: grid origin).

    This is a generalisation of ``beam_power(..., region='circular')``
    to a *sequence* of radii, useful for quickly characterising how
    much power a beam packs within successively larger apertures
    (encircled-energy curves, aperture-clipping budgets, focal-spot
    containment checks, diagnostic band splits for Fourier-plane
    simulations, etc.).

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric field.
    dx : float
        Grid spacing in x [m].
    radii : sequence of float
        Radii at which to compute enclosed power [m].  Does not need
        to be sorted -- the returned array preserves the input order.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    center : tuple of (xc, yc) or None, optional
        Center of the circular bands in meters, measured from the
        grid origin (which is at pixel (Nx/2, Ny/2)).  Default is
        ``(0.0, 0.0)`` -- the grid center.

    Returns
    -------
    powers : ndarray, shape (len(radii),)
        Integrated power within radius ``radii[i]`` for each i, in the
        same units as ``beam_power`` (``sum(|E|^2) * dx * dy``).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import radial_power_bands
    >>> # Synthesize a 100 um Gaussian and measure encircled energy
    >>> N, dx = 512, 2e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> w0 = 100e-6
    >>> E = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    >>> radii = [0.5*w0, w0, 2*w0]   # half-waist, 1/e^2, 2x
    >>> P = radial_power_bands(E, dx, radii)
    >>> # For a Gaussian, P(r<w0) should be ~86.5% of total power
    >>> P[1] / P[2]  # doctest: +SKIP
    0.865...
    """
    if dy is None:
        dy = dx
    if center is None:
        xc, yc = 0.0, 0.0
    else:
        xc, yc = center

    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    R2 = (X - xc) ** 2 + (Y - yc) ** 2
    I = np.abs(E) ** 2

    radii_arr = np.asarray(radii, dtype=float)
    powers = np.empty(radii_arr.shape, dtype=float)
    for i, r in enumerate(radii_arr):
        mask = R2 <= r * r
        powers[i] = float(np.sum(I[mask]) * dx * dy)
    return powers


def strehl_ratio(
    E: np.ndarray,
    E_ref: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> float:
    """
    Compute the Strehl ratio of a field relative to a reference field.

    Both fields are normalised to the same total power before comparison
    so that the ratio reflects wavefront quality rather than throughput.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Aberrated field (e.g. at the focal plane).
    E_ref : ndarray, complex, shape (Ny, Nx)
        Reference (diffraction-limited) field at the same plane.
    dx : float
        Grid spacing along x [m].
    dy : float, optional
        Grid spacing along y [m].  Defaults to ``dx`` (square grid).
        v4.13.2 added this kwarg so anamorphic / non-square grids no
        longer mis-scale the per-pixel area in the total-power
        normalisation.  Backward compatible: callers that omit ``dy``
        get identical behaviour to v4.13.1.

    Returns
    -------
    strehl : float
        Strehl ratio (0 to 1).  A value of 1.0 indicates a
        diffraction-limited beam.

    Notes
    -----
    ``Strehl = max(|E|^2) / max(|E_ref|^2)`` after both fields have been
    normalised to equal total power.  The Strehl ratio is dimensionless
    and the ``dx * dy`` factor cancels in the ratio, but using the
    correct pixel area keeps any external comparison consistent.
    """
    xp = _xp_of(E, E_ref)
    I = xp.abs(E) ** 2
    I_ref = xp.abs(E_ref) ** 2

    # 4.13.2 (C-P1-1): use ``dx * dy`` for the pixel area when ``dy``
    # is explicitly provided.  Pre-4.13.2 the v4.13.0 L3 sweep missed
    # this site and any anamorphic / non-square grid produced a wrong
    # total-power normalisation.  When ``dy`` is omitted we keep the
    # historical ``dx ** 2`` form bit-for-bit so callers that did not
    # pass a ``dy`` see exactly identical numerics to v4.13.1 (the
    # Strehl ratio is dimensionless and ``dx ** 2 == dx * dx`` is
    # numerically equal but uses a different IEEE rounding pathway
    # than ``dx * dy``; preserving the form keeps a small floating-
    # point identity that downstream tests rely on).
    if dy is None:
        pixel_area = dx ** 2
    else:
        pixel_area = dx * dy
    P = float(xp.sum(I) * pixel_area)
    P_ref = float(xp.sum(I_ref) * pixel_area)

    if P_ref == 0 or P == 0:
        return 0.0

    # Normalize to same total power
    return float(xp.max(I)) / P * P_ref / float(xp.max(I_ref))


def strehl_marechal(rms_waves: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Marechal-approximation Strehl ratio from wavefront RMS.

    .. math::
        S \\approx \\exp\\bigl(-(2\\pi\\sigma)^2\\bigr)

    where :math:`\\sigma` is the wavefront RMS error in waves.  Valid for
    :math:`\\sigma \\ll 1` (typically :math:`\\sigma < 0.1` waves).  For
    larger aberrations use :func:`strehl_phase_integral` or
    :func:`strehl_ratio`.

    Useful when you have an RMS-WFE estimate but no full PSF, or when
    comparing predictions against analytic small-aberration theory.

    Parameters
    ----------
    rms_waves : float or array
        RMS wavefront error in waves (NOT radians).

    Returns
    -------
    strehl : float or ndarray
        Marechal-approximation Strehl in [0, 1].

    See Also
    --------
    strehl_ratio : peak-ratio Strehl from full PSF.
    strehl_phase_integral : exact small-aberration Strehl from a pupil.

    Examples
    --------
    >>> import lumenairy as la
    >>> # Diffraction-limited rule of thumb: sigma ~ 1/14 wave -> S ~ 0.82
    >>> float(la.strehl_marechal(1.0 / 14.0))
    0.8189...
    """
    sigma = 2.0 * np.pi * np.asarray(rms_waves, dtype=float)
    return np.exp(-(sigma ** 2))


def strehl_phase_integral(pupil: np.ndarray) -> float:
    """Strehl ratio from the pupil-phase integral (Born & Wolf 9.1.10).

    .. math::
        S = \\left| \\frac{\\int A(x, y) \\, e^{i\\phi(x, y)} \\, dA}{\\int A(x, y) \\, dA} \\right|^2

    where :math:`A = |\\mathrm{pupil}|` is the pupil amplitude and
    :math:`\\phi = \\arg(\\mathrm{pupil})` is the pupil phase.  This is
    the exact small-aberration Strehl formula and avoids the
    peak-finding bias of :func:`strehl_ratio` on asymmetric PSFs where
    the diffraction-limited peak does not sit on the geometric chief
    ray.

    Parameters
    ----------
    pupil : ndarray (complex, 2-D)
        Complex pupil function.  Amplitude defines the aperture; phase
        carries the wavefront aberration.  Outside the aperture the
        amplitude should be zero so it does not contribute to the
        integral.

    Returns
    -------
    strehl : float
        Strehl ratio in [0, 1].  Returns 0.0 if the pupil has zero
        net amplitude (degenerate aperture).

    See Also
    --------
    strehl_ratio : peak-ratio Strehl from a full diffraction PSF.
    strehl_marechal : closed-form ``exp(-(2 pi sigma)^2)`` approximation
        from an RMS estimate.

    Examples
    --------
    >>> import numpy as np, lumenairy as la
    >>> N = 128
    >>> x = (np.arange(N) - N/2) / (N/2)
    >>> X, Y = np.meshgrid(x, x)
    >>> aperture = (X**2 + Y**2) <= 1.0
    >>> # Flat-phase pupil -> S = 1
    >>> P = aperture.astype(complex)
    >>> float(la.strehl_phase_integral(P))
    1.0
    """
    A = np.abs(pupil)
    A_sum = float(A.sum())
    if A_sum == 0:
        return 0.0
    num = float(np.abs(np.sum(pupil)) ** 2)
    den = A_sum ** 2
    return num / den


def coupling_efficiency(
    E: np.ndarray,
    mode: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> float:
    r"""Compute the mode-overlap coupling efficiency between a field
    and a target mode.

    Returns ``\eta = |<E | mode>|^2 / (<E|E> * <mode|mode>)``, the
    standard receiver / fiber-coupling efficiency expression.  Both
    fields must be sampled on the SAME grid (same shape, dx, dy);
    centroids may differ if the mode is intentionally offset.

    Parameters
    ----------
    E : ndarray, complex
        Incoming field at the coupling plane (e.g. focal plane after
        the receive lens).
    mode : ndarray, complex
        Target mode (e.g. a fiber LP01 mode generated by
        :func:`create_fiber_mode`, or any other reference complex
        amplitude pattern).
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    eta : float
        Coupling efficiency in [0, 1].  1.0 means ``E`` is a unit
        complex multiple of ``mode``; 0.0 means orthogonal.

    Notes
    -----
    For amplitude-only matching with E and mode both real-positive,
    this reduces to the classical overlap integral.  For complex
    fields the phase structure must also match for full coupling --
    a perfectly-shaped beam with the wrong phase ramp couples to
    zero efficiency.

    The function is :class:`numpy.float`-conservative: if the mode
    or field is identically zero, returns 0.0.
    """
    if E.shape != mode.shape:
        raise ValueError(
            f"coupling_efficiency: shape mismatch -- E is {E.shape}, "
            f"mode is {mode.shape}.  Resample to a common grid first.")
    if dy is None:
        dy = dx
    da = float(dx) * float(dy)
    overlap = np.sum(np.conj(mode) * E) * da
    p_E = np.sum(np.abs(E) ** 2) * da
    p_mode = np.sum(np.abs(mode) ** 2) * da
    denom = p_E * p_mode
    if denom == 0:
        return 0.0
    return float(np.abs(overlap) ** 2 / denom)


def M2(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
) -> Tuple[float, float]:
    r"""Compute the ISO 11146 :math:`M^2` beam-quality factor at a
    single plane.

    Uses the second-moment definition with phase-curvature correction
    via the Wigner cross-term, so a single plane (any z) gives the
    correct invariant :math:`M^2`.  No through-focus sweep required.

    Returns ``(M2_x, M2_y)``: the per-axis quality factors with the
    Heisenberg lower bound :math:`M^2 \geq 1`.  A perfect TEM_{00}
    Gaussian gives 1.0 exactly; super-Gaussians, multi-mode fibers,
    and aberrated beams give larger values.

    Parameters
    ----------
    E : ndarray, complex
        Complex field at the measurement plane.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Optical wavelength [m].  Used only as a sanity-scale anchor;
        the M^2 invariant is dimensionless and the wavelength enters
        through the angular-spectrum k-domain conversion.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    M2_x, M2_y : float
        M^2 values on the x and y axes.  Both >= 1 for any physical
        field.  Numerical floor is ~ 1e-3 below 1.0 from FFT/discrete-
        derivative round-off; clamp to 1.0 if you want strict
        physicality.

    Notes
    -----
    The ISO 11146 definition for a single dimension is
    :math:`M^2 = 2 \sqrt{\sigma_x^2 \sigma_{k_x}^2 - \sigma_{x,k_x}^2}`
    where :math:`\sigma_{x,k_x}` is the Wigner cross-correlation that
    captures phase curvature.  Without the cross-term, a curved
    (non-waist) beam would give an inflated M^2 because it has more
    apparent angular spread than its waist beam.  The cross-term is
    computed here as
    :math:`\sigma_{x,k_x} = \langle x \, \mathrm{Im}(E^* \partial_x E) \rangle / \langle |E|^2 \rangle`.

    For a fundamental Gaussian, the function returns 1.0 to within
    discrete-grid sampling error (a few times 1e-3 at N=128, scaling
    as ~ 1/N for fine grids).
    """
    _ = float(wavelength)  # validation only; wavelength cancels out
    if dy is None:
        dy = dx
    Ny, Nx = E.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    I = np.abs(E) ** 2
    P = float(I.sum())
    if P == 0:
        return float('nan'), float('nan')

    # Spatial centroid + variance about centroid
    cx = float((X * I).sum() / P)
    cy = float((Y * I).sum() / P)
    Xs = X - cx; Ys = Y - cy
    sx2 = float((Xs ** 2 * I).sum() / P)
    sy2 = float((Ys ** 2 * I).sum() / P)

    # Angular-spectrum variances via FFT.  Use 2*pi*fftfreq for
    # angular wavenumber; the centroid of the angular spectrum
    # captures any global tilt in E (which contributes to apparent
    # angular spread but not to the invariant M^2).
    F = np.fft.fftshift(np.fft.fft2(E))
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    Iang = np.abs(F) ** 2
    Pang = float(Iang.sum())
    if Pang == 0:
        return float('nan'), float('nan')
    cx_k = float((KX * Iang).sum() / Pang)
    cy_k = float((KY * Iang).sum() / Pang)
    skx2 = float(((KX - cx_k) ** 2 * Iang).sum() / Pang)
    sky2 = float(((KY - cy_k) ** 2 * Iang).sum() / Pang)

    # Wigner cross-correlation via the imaginary-derivative form.
    # sigma_xk = <x * Im(E* dE/dx)> / <|E|^2>, evaluated about the
    # spatial AND angular centroids.
    Ex = np.gradient(E, dx, axis=1)
    Ey = np.gradient(E, dy, axis=0)
    # 4.10: removed dead `- cx_k * 0.0` / `- cy_k * 0.0` placeholders
    # (the centring is already provided by `Xs = X - cx`).
    cross_x = float((Xs * (np.conj(E) * Ex).imag).sum() / P)
    cross_y = float((Ys * (np.conj(E) * Ey).imag).sum() / P)

    M2x_sq = max(4.0 * (sx2 * skx2 - cross_x ** 2), 0.0)
    M2y_sq = max(4.0 * (sy2 * sky2 - cross_y ** 2), 0.0)
    return float(np.sqrt(M2x_sq)), float(np.sqrt(M2y_sq))


def check_sampling_conditions(
    N: int,
    dx: float,
    z: float,
    wavelength: float,
    feature_size: Optional[float] = None,
    NA: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Check whether grid parameters satisfy ASM sampling conditions.

    Evaluates the Nyquist criterion and the Fresnel aliasing condition
    for a given propagation geometry, and returns actionable diagnostics.

    Parameters
    ----------
    N : int
        Grid size (assumes a square N x N grid).
    dx : float
        Grid spacing [m].
    z : float
        Propagation distance [m].
    wavelength : float
        Optical wavelength [m].
    feature_size : float, optional
        Minimum feature size to resolve [m].  Required for the Fresnel
        aliasing check; if omitted that check is skipped.
    NA : float, optional
        4.10: when provided, the Nyquist criterion is relaxed to
        ``dx < wavelength / (2 * NA)``, which is what's actually needed
        to resolve the propagating cone within the specified NA.
        The strict ``dx < wavelength/2`` criterion (i.e. NA = 1) is
        only required if you also intend to resolve the full
        evanescent spectrum.
    verbose : bool, default True
        If ``True``, print a human-readable diagnostic summary.

    Returns
    -------
    dict
        ``'nyquist_ok'`` : bool
            Whether the Nyquist condition is satisfied (NA-aware if NA
            is supplied).
        ``'fresnel_ok'`` : bool
            Whether the Fresnel aliasing condition is satisfied.
        ``'d_min'`` : float
            Minimum resolvable feature size [m] for the current grid.
        ``'recommendations'`` : list of str
            Suggestions for fixing any violated conditions.  Empty when
            all conditions are met.
    """
    L = N * dx  # Grid extent

    # Condition 1: Nyquist.  Strict form dx < lambda/2 is for the full
    # angular spectrum (including evanescents); for a beam with max
    # NA, dx < lambda/(2*NA) is sufficient.  Default to the strict
    # form (NA = 1) for backward compatibility.
    if NA is None or NA <= 0:
        nyquist_limit = wavelength / 2
    else:
        nyquist_limit = wavelength / (2.0 * float(NA))
    nyquist_ok = dx < nyquist_limit

    # Condition 2: Fresnel aliasing (d_min = 2*lambda*z/L)
    d_min = 2 * wavelength * abs(z) / L

    if feature_size is not None:
        fresnel_ok = d_min < feature_size
    else:
        fresnel_ok = True  # Can't check without feature size

    recommendations = []
    if not nyquist_ok:
        recommendations.append(f"Decrease dx below {nyquist_limit * 1e6:.3f} um")
    if not fresnel_ok:
        required_L = 2 * wavelength * abs(z) / feature_size
        required_N = int(np.ceil(required_L / dx))
        recommendations.append(
            f"Increase grid extent to L > {required_L * 1e3:.2f} mm (N > {required_N})"
        )

    if verbose:
        print("ASM Sampling Conditions Check")
        print("=" * 40)
        print(f"Grid: {N}x{N}, dx = {dx * 1e6:.3f} um")
        print(f"Extent: L = {L * 1e3:.3f} mm")
        print(f"Propagation: z = {z * 1e3:.3f} mm")
        print(f"Wavelength: {wavelength * 1e9:.1f} nm")
        print()
        print(f"Nyquist (dx < \u03bb/2 = {nyquist_limit * 1e6:.3f} um): "
              f"{'OK' if nyquist_ok else 'FAIL'}")
        print(f"Minimum resolvable feature: d_min = {d_min * 1e6:.2f} um")
        if feature_size is not None:
            print(f"Target feature size: {feature_size * 1e6:.2f} um")
            print(f"Fresnel aliasing: "
                  f"{'OK' if fresnel_ok else 'FAIL - increase grid extent'}")
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

    return {
        'nyquist_ok': nyquist_ok,
        'fresnel_ok': fresnel_ok,
        'd_min': d_min,
        'recommendations': recommendations,
    }


# =============================================================================
# PSF / MTF COMPUTATION
# =============================================================================

def compute_psf(
    pupil: np.ndarray,
    wavelength: float,
    f: float,
    dx_pupil: float,
    N_psf: Optional[int] = None,
    oversample: int = 1,
    normalize: str = 'power',
) -> Tuple[np.ndarray, float]:
    """
    Compute the point spread function (PSF) from a pupil function.

    Uses the Fraunhofer relation: the PSF at the focal plane is the squared
    magnitude of the Fourier transform of the complex pupil function.

    Parameters
    ----------
    pupil : ndarray (complex, Np x Np)
        Complex pupil function. Amplitude describes the aperture shape
        (0 outside, 1 inside for a simple aperture), phase describes
        wavefront aberrations.
    wavelength : float
        Operating wavelength [m].
    f : float
        Focal length of the imaging lens [m].
    dx_pupil : float
        Pupil-plane grid spacing [m].
    N_psf : int or None, optional
        Size of the output PSF grid. If None, uses ``pupil.shape[0] * oversample``.
        Larger N gives finer focal-plane sampling.
    oversample : int, default 1
        Zero-pad factor for the FFT. Equivalent to N_psf = N_pupil * oversample.
    normalize : ``'power'`` (default) / ``'peak'`` / ``'none'``
        How the returned PSF is scaled.

        * ``'power'`` (default, v3.1.1+): total integrated intensity
          equals the pupil's total intensity (Parseval).  This is the
          correct choice for **Strehl-ratio comparisons**: under this
          normalisation ``psf_abb.max() / psf_ideal.max()`` is
          directly the Strehl, because the total energy is preserved
          across the pupil-to-focal transform for both fields.
        * ``'peak'``: divides by ``psf.max()`` so the peak is 1.
          Useful only for displaying a PSF *shape*; **do not use it
          for Strehl** -- every PSF (ideal or aberrated) comes out
          peaked at 1, hiding the peak drop caused by aberrations.
        * ``'none'``: raw ``|FFT{pupil}|^2`` with no normalisation
          at all.  Useful for absolute-photon-flux calculations when
          the pupil is normalised to a known input power.

    Returns
    -------
    psf : ndarray (real, N_psf x N_psf)
        Intensity point spread function, scaled according to
        ``normalize``.
    dx_psf : float
        Focal-plane grid spacing [m] = wavelength * f / (N_psf * dx_pupil).

    Notes
    -----
    The PSF is the intensity response of the system to a point source at
    infinity. For an unaberrated circular aperture of diameter D, the PSF
    is the Airy pattern with first zero at r = 1.22 * lambda * f / D.

    To include wavefront aberrations, apply them to the pupil phase before
    calling this function, e.g.::

        pupil = aperture * np.exp(1j * aberration_phase)
        psf, dx_psf = compute_psf(pupil, wavelength, f, dx_pupil)

    Prior to v3.1.1 the default was ``normalize='peak'``, which silently
    broke the canonical Strehl calculation pattern; ``'power'`` is now
    the default and ``'peak'`` is opt-in.
    """
    xp = _xp_of(pupil)
    # 4.11.2: enforce the (long-undocumented) square-pupil assumption.
    # Pre-4.11.2 the function silently used ``pupil.shape[0]`` for
    # both axes and applied an isotropic pad / Fraunhofer-grid scale,
    # so rectangular inputs (Ny != Nx) produced wrong PSF dimensions
    # and an anisotropically-mispadded transform.  Raise here so
    # rectangular-aperture callers get a visible failure instead of
    # silently wrong output; the underlying FFT handles non-square
    # arrays fine, the pad / grid code does not.
    if pupil.ndim != 2:
        raise ValueError(
            f"compute_psf: pupil must be 2-D; got shape {pupil.shape!r}.")
    if pupil.shape[0] != pupil.shape[1]:
        raise ValueError(
            f"compute_psf: only square pupils are supported "
            f"(pupil.shape = {pupil.shape!r}).  For rectangular "
            f"apertures, embed the support in a square grid before "
            f"calling this function.")
    Np = pupil.shape[0]
    if N_psf is None:
        N_psf = Np * oversample

    # Zero-pad pupil if oversampling.  Uses xp.pad so CuPy / JAX
    # arrays don't get coerced through NumPy.
    if N_psf > Np:
        pad_before = (N_psf - Np) // 2
        pad_after = N_psf - Np - pad_before
        pupil_padded = xp.pad(pupil, ((pad_before, pad_after),
                                       (pad_before, pad_after)),
                              mode='constant')
    else:
        pupil_padded = pupil

    # Fraunhofer: PSF amplitude is FFT of pupil
    amp = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(pupil_padded)))
    psf = xp.abs(amp) ** 2

    # Apply the requested normalisation.  Default is 'power' because
    # Strehl-ratio computations rely on the peak-ratio of two PSFs
    # normalised to equal total intensity.
    if normalize == 'peak':
        if float(psf.max()) > 0:
            psf = psf / psf.max()
    elif normalize == 'power':
        # 4.10: Parseval-correct rescaling.  Physical Parseval says
        #   ∫ |E_pupil(x)|^2 dA_pupil  ==  ∫ |E_psf(x)|^2 dA_psf
        # i.e. sum(|E_pupil|^2) * dx_pupil^2 == sum(|E_psf|^2) * dx_psf^2.
        # Pre-4.10 enforced equal pixel-sum (sum(psf) == sum(|pupil|^2))
        # which differs from physical Parseval by (dx_pupil/dx_psf)^2.
        # Strehl ratios cancel the constant so that doesn't notice; but
        # users asking for absolute photon flux (also a documented
        # use-case) were getting the wrong answer.
        dx_psf_local = wavelength * f / (N_psf * dx_pupil)
        pupil_power_area = float(xp.sum(xp.abs(pupil_padded) ** 2)) * (dx_pupil ** 2)
        psf_power_area = float(xp.sum(psf)) * (dx_psf_local ** 2)
        if psf_power_area > 0 and pupil_power_area > 0:
            psf = psf * (pupil_power_area / psf_power_area)
    elif normalize == 'none':
        pass
    else:
        raise ValueError(
            f"normalize must be 'power', 'peak', or 'none'; got {normalize!r}")

    # Focal-plane grid spacing from Fraunhofer relation
    dx_psf = wavelength * f / (N_psf * dx_pupil)

    return psf, dx_psf


def compute_otf(psf: np.ndarray) -> np.ndarray:
    """
    Compute the optical transfer function (OTF) from a PSF.

    The OTF is the Fourier transform of the PSF. Its magnitude is the
    modulation transfer function (MTF), and its phase is the phase
    transfer function (PTF).

    Parameters
    ----------
    psf : ndarray (real, N×N)
        Intensity PSF (typically from :func:`compute_psf`).

    Returns
    -------
    otf : ndarray (complex, N×N)
        Complex OTF, normalized so ``otf[0, 0]`` (DC) = 1.

    Notes
    -----
    By the Wiener-Khinchin theorem, the OTF is also the autocorrelation
    of the pupil function. Both approaches give the same result for
    coherent imaging systems.
    """
    xp = _xp_of(psf)
    otf = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(psf)))
    # Normalize so DC component = 1
    dc = otf[otf.shape[0] // 2, otf.shape[1] // 2]
    if abs(complex(dc)) > 0:
        otf = otf / dc
    return otf


def compute_mtf(psf: np.ndarray) -> np.ndarray:
    """
    Compute the modulation transfer function (MTF) from a PSF.

    The MTF is |OTF| — the magnitude of the optical transfer function.
    It describes the contrast transfer of the imaging system as a
    function of spatial frequency.

    Parameters
    ----------
    psf : ndarray (real, N×N)
        Intensity PSF.

    Returns
    -------
    mtf : ndarray (real, N×N)
        MTF normalized so ``mtf[0, 0]`` = 1 at DC.

    Notes
    -----
    For a diffraction-limited circular aperture, the MTF is the
    autocorrelation of the pupil, cutting off at the diffraction
    cutoff frequency:

        f_cutoff = D / (wavelength * f)

    To get radial MTF profiles (tangential/sagittal or azimuthal
    average), take cuts or radial averages of this 2D array.
    """
    xp = _xp_of(psf)
    return xp.abs(compute_otf(psf))


def mtf_radial(
    mtf: np.ndarray,
    dx_psf: float,
    wavelength: float,
    f: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally-averaged radial MTF profile.

    Parameters
    ----------
    mtf : ndarray (real, N×N)
        2D MTF array from :func:`compute_mtf`.
    dx_psf : float
        PSF-plane grid spacing [m] (from :func:`compute_psf`).
    wavelength : float
        Wavelength [m].
    f : float
        Focal length [m].

    Returns
    -------
    freq : ndarray (real, N/2,)
        Spatial frequencies in cycles per mm at the focal plane.
    mtf_profile : ndarray (real, N/2,)
        Azimuthally-averaged MTF at each frequency.
    """
    N = mtf.shape[0]
    # Frequency grid for the PSF plane (in cycles/m)
    df = 1.0 / (N * dx_psf)

    # Radial bin the MTF
    cx = N // 2
    y, x = np.indices(mtf.shape)
    r = np.sqrt((x - cx)**2 + (y - cx)**2)
    r_int = np.rint(r).astype(int)

    # Azimuthal average via numpy bincount
    tbin = np.bincount(r_int.ravel(), weights=mtf.ravel())
    nbin = np.bincount(r_int.ravel())
    radial_profile = np.where(nbin > 0, tbin / np.maximum(nbin, 1), 0.0)

    # Keep only up to Nyquist
    n_max = N // 2
    freq = np.arange(n_max) * df * 1e-3  # cycles per mm
    return freq, radial_profile[:n_max]


# ============================================================================
# Spec-sheet metrics (v4.14.0): encircled energy, beam diameter,
# MTF cutoff, depth of focus
# ============================================================================

def encircled_energy_curve(
    E: np.ndarray,
    dx: float,
    *,
    dy: Optional[float] = None,
    radii: Optional[np.ndarray] = None,
    centroid: Optional[Tuple[float, float]] = None,
    n_radii: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encircled-energy curve of an intensity distribution.

    Returns ``(radii, ee)`` where ``ee[i]`` is the fraction of total
    power within a circle of radius ``radii[i]`` of the centroid (or
    a user-supplied centre).  This is the standard spec-sheet metric
    used to characterise focal-spot containment (e.g. the canonical
    "84% encircled energy" radius reported on lens spec sheets).

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric-field distribution.  The intensity is
        ``|E|**2``; if ``E`` is real-valued it is treated as an
        amplitude.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    radii : ndarray, optional
        Radii at which to evaluate the curve [m].  If ``None``, a
        linear sweep of ``n_radii`` samples from ``0`` to the maximum
        in-grid radius (corner distance) is used.
    centroid : (cx, cy), optional
        Centre of the encircled-energy circles [m] measured from the
        grid origin (pixel ``(Nx/2, Ny/2)``).  Defaults to the
        intensity centroid via :func:`beam_centroid`.
    n_radii : int, default 64
        Number of radii to sample when ``radii`` is ``None``.

    Returns
    -------
    radii : ndarray, shape (n,)
        Radii at which the curve is evaluated [m] (sorted ascending).
    ee : ndarray, shape (n,)
        Fraction of total in-grid power encircled, monotonically
        non-decreasing and converging to ``~1.0`` (less the power that
        falls outside the grid).

    Notes
    -----
    Implementation sorts the per-pixel intensities by radial distance
    and uses :func:`numpy.cumsum` to build the encircled-power curve,
    then linearly interpolates onto the requested ``radii``.  This is
    far cheaper than the naive ``O(N^2 * R)`` mask-and-sum loop and
    gives the same answer to floating-point precision when the
    requested radii are coarser than the pixel pitch.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import encircled_energy_curve
    >>> N, dx = 256, 1e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> w0 = 30e-6
    >>> E = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    >>> r, ee = encircled_energy_curve(E, dx, n_radii=8)
    >>> bool(np.all(np.diff(ee) >= -1e-12))
    True
    """
    if dy is None:
        dy = dx
    if E.ndim != 2:
        raise ValueError(
            f"encircled_energy_curve: E must be 2-D; got shape "
            f"{E.shape!r}.")
    if n_radii < 2:
        raise ValueError(
            f"encircled_energy_curve: n_radii must be >= 2; got "
            f"{n_radii!r}.")

    Ny, Nx = E.shape

    if centroid is None:
        cx, cy = beam_centroid(E, dx, dy)
    else:
        cx, cy = float(centroid[0]), float(centroid[1])

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    I = np.abs(np.asarray(E)) ** 2
    pixel_area = float(dx) * float(dy)
    total = float(I.sum()) * pixel_area
    if total <= 0:
        # Degenerate input -- emit a zero curve over the requested
        # (or default) radii grid so downstream callers don't have to
        # special-case the empty-field branch.
        if radii is None:
            r_max = float(R.max())
            radii_out = np.linspace(0.0, r_max if r_max > 0 else 1.0,
                                    n_radii)
        else:
            radii_out = np.asarray(radii, dtype=float)
            radii_out = np.sort(radii_out)
        return radii_out, np.zeros_like(radii_out)

    # Sort pixels by radial distance and build the cumulative power
    # curve once.  The same cumulative curve is sampled at every
    # requested radius via np.searchsorted, which keeps the cost at
    # O(N log N) regardless of how many radii are asked for.
    r_flat = R.ravel()
    i_flat = I.ravel() * pixel_area
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    p_cum = np.cumsum(i_flat[order]) / total

    if radii is None:
        r_max = float(r_sorted[-1])
        radii_out = np.linspace(0.0, r_max, n_radii)
    else:
        radii_out = np.asarray(radii, dtype=float)
        # Sort + validate -- the contract says the returned curve is
        # monotonically non-decreasing in radius.
        if not np.all(np.isfinite(radii_out)) or np.any(radii_out < 0):
            raise ValueError(
                "encircled_energy_curve: radii must be finite and "
                "non-negative.")
        radii_out = np.sort(radii_out)

    # Interpolate the cumulative-power curve onto the requested
    # radii.  np.searchsorted gives the insertion index; linear
    # interpolation between adjacent samples removes the pixel-grid
    # staircase.
    idx = np.searchsorted(r_sorted, radii_out, side='right')
    ee = np.empty_like(radii_out)
    for i, (r, j) in enumerate(zip(radii_out, idx)):
        if j == 0:
            ee[i] = 0.0 if r < r_sorted[0] else float(p_cum[0])
        elif j >= r_sorted.size:
            ee[i] = float(p_cum[-1])
        else:
            r_lo, r_hi = r_sorted[j - 1], r_sorted[j]
            p_lo, p_hi = p_cum[j - 1], p_cum[j]
            if r_hi == r_lo:
                ee[i] = float(p_hi)
            else:
                t = (r - r_lo) / (r_hi - r_lo)
                ee[i] = float(p_lo + t * (p_hi - p_lo))

    # Numerical-safety clamp -- np.cumsum can drift very slightly
    # below zero on degenerate inputs but the curve is bounded in
    # [0, 1] by definition.
    np.clip(ee, 0.0, 1.0, out=ee)
    return radii_out, ee


def encircled_energy_radius(
    E: np.ndarray,
    dx: float,
    *,
    dy: Optional[float] = None,
    threshold: float = 0.84,
    centroid: Optional[Tuple[float, float]] = None,
) -> float:
    """Radius within which a given fraction of the total power is
    encircled.

    Returns the radius (in meters) at which the encircled-energy curve
    of :func:`encircled_energy_curve` first crosses ``threshold``.  The
    default ``0.84`` matches the conventional "84% encircled energy"
    radius reported on most lens spec sheets (close to the Airy first-
    null at ``1.22 * lambda * f_number``).

    Parameters
    ----------
    E : ndarray, complex
        Complex electric-field distribution.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    threshold : float, default 0.84
        Encircled-power fraction in ``(0, 1]``.
    centroid : (cx, cy), optional
        Centroid coordinates [m].  Defaults to the intensity centroid.

    Returns
    -------
    radius : float
        Encircled-energy radius [m].  Linearly interpolated between
        the two grid samples that straddle ``threshold``.  Returns the
        maximum in-grid radius if the curve never reaches the
        threshold (e.g. the beam clips the grid).

        The encircled-energy curve sampled by
        :func:`encircled_energy_curve` is NOT guaranteed to start at
        ``ee[0] = 0``.  When the requested radii grid starts at
        ``radii[0] = 0`` and at least one pixel sits exactly at the
        centre (``r_sorted[0] = 0``), the cumulative-power lookup at
        radius 0 picks up that centre-pixel contribution and
        ``ee[0] = p_cum[0]`` (i.e. the centre-pixel's fractional
        intensity).  If ``threshold`` is small enough that the
        centre-pixel contribution alone already exceeds it (the
        "hot-centre" case: a delta-like input concentrated at the
        centre pixel), the short-circuit returns ``radii[0] = 0`` m,
        which is the physically reasonable answer for that input.

    See Also
    --------
    encircled_energy_curve : the underlying curve.
    beam_diameter : intensity-drop diameter (e.g. 1/e^2, FWHM).

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import encircled_energy_radius
    >>> # 2-D Gaussian: 86.5% encircled at r = w0 (the 1/e^2 radius)
    >>> N, dx = 256, 1e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> w0 = 20e-6
    >>> E = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    >>> r84 = encircled_energy_radius(E, dx, threshold=0.865)
    >>> bool(abs(r84 - w0) / w0 < 0.05)
    True
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"encircled_energy_radius: threshold must be in (0, 1]; "
            f"got {threshold!r}.")

    # Dense grid so the threshold crossing has good interpolation
    # support.  256 samples is well below the typical pixel count yet
    # gives sub-percent accuracy on the threshold crossing.
    radii, ee = encircled_energy_curve(
        E, dx, dy=dy, centroid=centroid, n_radii=256)

    # If the curve never reaches the threshold (beam clips the grid
    # or threshold > max(ee)), return the maximum radius.
    if ee[-1] < threshold:
        return float(radii[-1])

    # First index where ee >= threshold.  ``ee[0]`` is NOT always 0
    # -- when ``radii[0] = 0`` collides with a centre-pixel at
    # ``r_sorted[0] = 0`` (delta-like inputs), the cumulative-power
    # lookup at radius 0 picks up the centre-pixel contribution and
    # ``ee[0] = p_cum[0]``.  When ``threshold <= ee[0]`` the
    # short-circuit below returns ``radii[0]`` (= 0 m), the
    # physically-reasonable hot-centre answer.
    idx = int(np.searchsorted(ee, threshold, side='left'))
    if idx <= 0:
        return float(radii[0])
    r_lo, r_hi = radii[idx - 1], radii[idx]
    e_lo, e_hi = ee[idx - 1], ee[idx]
    if e_hi == e_lo:
        return float(r_hi)
    t = (threshold - e_lo) / (e_hi - e_lo)
    return float(r_lo + t * (r_hi - r_lo))


def mtf_cutoff(
    mtf_profile: np.ndarray,
    freq: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """Spatial frequency at which a 1-D MTF profile first drops below a
    threshold.

    The "useful cutoff" reported on most lens spec sheets is the
    frequency at which MTF = 0.5; this function returns that crossing
    (or any other user-supplied threshold) by linearly interpolating
    across the two adjacent samples.

    Parameters
    ----------
    mtf_profile : ndarray, shape (N,)
        1-D MTF values.  Typically the radial / azimuthally-averaged
        profile returned by :func:`mtf_radial`.  Assumed to start at
        DC and be ordered with monotonically increasing frequency.
    freq : ndarray, shape (N,)
        Spatial frequencies corresponding to each MTF sample.  Must be
        the same length as ``mtf_profile`` and strictly increasing.
    threshold : float, default 0.5
        MTF threshold in ``(0, 1]``.  ``0.5`` is the classical "useful
        cutoff" used on lens spec sheets.

    Returns
    -------
    f_cutoff : float
        Spatial frequency at which the MTF first crosses below the
        threshold, in the same units as ``freq``.  Returns
        ``numpy.inf`` if the MTF stays above the threshold for every
        frequency in the supplied array.

    See Also
    --------
    compute_mtf : 2-D MTF from a PSF.
    mtf_radial : azimuthally-averaged 1-D MTF profile.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import mtf_cutoff
    >>> freq = np.linspace(0.0, 100.0, 101)        # cyc/mm
    >>> mtf = np.exp(-freq / 30.0)                 # synthetic falloff
    >>> # MTF = 0.5 at freq = 30 * ln(2) ~ 20.79 cyc/mm
    >>> bool(abs(mtf_cutoff(mtf, freq) - 30.0 * np.log(2)) < 0.5)
    True
    """
    mtf_arr = np.asarray(mtf_profile, dtype=float)
    f_arr = np.asarray(freq, dtype=float)
    if mtf_arr.ndim != 1 or f_arr.ndim != 1:
        raise ValueError(
            f"mtf_cutoff: both mtf_profile and freq must be 1-D; got "
            f"shapes {mtf_arr.shape!r} and {f_arr.shape!r}.")
    if mtf_arr.shape != f_arr.shape:
        raise ValueError(
            f"mtf_cutoff: mtf_profile and freq must be the same length;"
            f" got {mtf_arr.size} and {f_arr.size}.")
    if mtf_arr.size < 2:
        raise ValueError(
            f"mtf_cutoff: need at least 2 samples; got "
            f"{mtf_arr.size}.")
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"mtf_cutoff: threshold must be in (0, 1]; got "
            f"{threshold!r}.")

    # If the MTF starts below the threshold (i.e. DC is already
    # below), interpret that as a zero-cutoff system rather than
    # +inf.  The contract in the docstring says "stays above for
    # ALL frequencies" gives +inf, which is the opposite case.
    if mtf_arr[0] < threshold:
        return float(f_arr[0])
    # If every sample stays above the threshold, the cutoff is
    # outside the supplied range -- return +inf per the docstring.
    if np.all(mtf_arr >= threshold):
        return float(np.inf)

    # First index at which the MTF dips below threshold.
    below = np.where(mtf_arr < threshold)[0]
    j = int(below[0])
    if j == 0:
        return float(f_arr[0])
    m_lo, m_hi = mtf_arr[j - 1], mtf_arr[j]
    f_lo, f_hi = f_arr[j - 1], f_arr[j]
    if m_lo == m_hi:
        return float(f_hi)
    # Linear interp: MTF(f) = m_lo + (f - f_lo) / (f_hi - f_lo) *
    #                          (m_hi - m_lo) = threshold  ->  solve for f.
    t = (threshold - m_lo) / (m_hi - m_lo)
    return float(f_lo + t * (f_hi - f_lo))


def beam_diameter(
    E: np.ndarray,
    dx: float,
    *,
    dy: Optional[float] = None,
    threshold: Union[float, str] = '1/e^2',
    centroid: Optional[Tuple[float, float]] = None,
) -> float:
    """Beam diameter at a specified intensity threshold.

    Returns the diameter (in meters) at which the radially-averaged
    intensity drops below ``threshold * peak``.  The radial average is
    computed from the supplied centroid (or the intensity centroid by
    default) and the first crossing is found by linear interpolation.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric-field distribution.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    threshold : float or {'1/e^2', '1/e', 'FWHM', 'D4sigma'}, default '1/e^2'
        Either a numeric fractional intensity in ``(0, 1]`` relative
        to the peak, or one of the named conventions:

        * ``'1/e^2'`` (default): intensity = 1/e**2 ~ 0.1353.  Gives
          the classical Gaussian-beam diameter ``2 * w_0``.
        * ``'1/e'``: intensity = 1/e ~ 0.3679.
        * ``'FWHM'``: intensity = 0.5 (full-width-half-max).
        * ``'D4sigma'``: forwards to :func:`beam_d4sigma` and returns
          ``sqrt(d4x * d4y)`` (the geometric-mean second-moment
          diameter -- the closest scalar analogue to a radial
          measure).
    centroid : (cx, cy), optional
        Centre of the radial average [m].  Defaults to the intensity
        centroid via :func:`beam_centroid`.

    Returns
    -------
    diameter : float
        Beam diameter [m].

    Notes
    -----
    Algorithm (numeric thresholds):

    1. Compute ``I = |E|**2``.
    2. Build a radial profile by sorting pixels by distance from the
       centroid and averaging within radial bins.
    3. Find the first radius at which the smoothed radial intensity
       crosses ``threshold * peak`` from above; linearly interpolate
       between the two adjacent samples.
    4. Return ``2 * radius``.

    The radial-average step washes out azimuthal asymmetry, so this
    function reports a single scalar diameter even for elliptical
    beams.  For full per-axis widths on an elliptical or anamorphic
    beam, use :func:`beam_d4sigma` directly.

    See Also
    --------
    beam_d4sigma : ISO 11146 second-moment diameter (per-axis).
    encircled_energy_radius : encircled-power radius.

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import beam_diameter
    >>> N, dx = 256, 1e-6
    >>> x = (np.arange(N) - N/2) * dx
    >>> X, Y = np.meshgrid(x, x)
    >>> w0 = 25e-6
    >>> E = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    >>> d = beam_diameter(E, dx, threshold='1/e^2')
    >>> bool(abs(d - 2 * w0) / (2 * w0) < 0.05)
    True
    """
    if dy is None:
        dy = dx
    if E.ndim != 2:
        raise ValueError(
            f"beam_diameter: E must be 2-D; got shape {E.shape!r}.")

    # Named-threshold dispatch.
    named = {
        '1/e^2': float(np.exp(-2.0)),
        '1/e': float(np.exp(-1.0)),
        'FWHM': 0.5,
    }
    if isinstance(threshold, str):
        if threshold == 'D4sigma':
            d4x, d4y = beam_d4sigma(E, dx, dy)
            # Geometric mean of per-axis D4sigma so callers get a
            # single scalar; users who want per-axis widths should
            # call beam_d4sigma directly.
            return float(np.sqrt(max(d4x, 0.0) * max(d4y, 0.0)))
        if threshold not in named:
            raise ValueError(
                f"beam_diameter: unknown threshold name "
                f"{threshold!r}; expected one of "
                f"{sorted(named) + ['D4sigma']} or a numeric value "
                "in (0, 1].")
        thr = named[threshold]
    else:
        thr = float(threshold)
        if not (0.0 < thr <= 1.0):
            raise ValueError(
                f"beam_diameter: numeric threshold must be in (0, 1]; "
                f"got {thr!r}.")

    Ny, Nx = E.shape
    if centroid is None:
        cx, cy = beam_centroid(E, dx, dy)
    else:
        cx, cy = float(centroid[0]), float(centroid[1])

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    I = np.abs(np.asarray(E)) ** 2
    peak = float(I.max())
    if peak <= 0:
        return 0.0

    # Radial-average: bin pixels by integer radial-pixel index and
    # average the intensity within each bin.  Bin pitch follows the
    # smaller of dx / dy so the averaging window matches the finer
    # pixel pitch on anamorphic grids.
    dr = min(float(dx), float(dy))
    r_bin = np.floor(R / dr).astype(np.int64)
    n_bins = int(r_bin.max()) + 1
    tbin = np.bincount(r_bin.ravel(), weights=I.ravel(),
                       minlength=n_bins)
    cbin = np.bincount(r_bin.ravel(), minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        I_radial = np.where(cbin > 0, tbin / np.maximum(cbin, 1), 0.0)
    radial_r = (np.arange(n_bins) + 0.5) * dr

    target = thr * peak
    # Find the first bin at which the radial-average intensity drops
    # below target.  Pre-target peak occurs at small r (well within
    # the beam); if the radial average never dips below target, the
    # beam is wider than the grid and the best estimate is the
    # maximum in-grid radius.
    below = np.where(I_radial < target)[0]
    if below.size == 0:
        return float(2.0 * radial_r[-1])
    j = int(below[0])
    if j == 0:
        # The very first bin (centroid) is already below the
        # threshold -- this is the degenerate "empty beam" case.
        return 0.0
    r_lo, r_hi = radial_r[j - 1], radial_r[j]
    I_lo, I_hi = I_radial[j - 1], I_radial[j]
    if I_lo == I_hi:
        return float(2.0 * r_hi)
    t = (target - I_lo) / (I_hi - I_lo)
    r_cross = r_lo + t * (r_hi - r_lo)
    return float(2.0 * r_cross)


def depth_of_focus(
    wavelength: float,
    f_number: float,
    *,
    formula: str = 'rayleigh',
) -> float:
    """One-sided depth of focus [m] for a diffraction-limited system.

    Two standard formulas are supported:

    * ``'rayleigh'`` (default): ``+/- 4 * f_number**2 * wavelength``.
      The classical Rayleigh quarter-wave (``lambda/4`` OPD) limit at
      the marginal ray.
    * ``'marechal'``: ``+/- wavelength / NA**2`` with
      ``NA = 1 / (2 * f_number)`` (the paraxial NA-from-f# conversion).
      The Marechal-criterion DOF that keeps Strehl > 0.8 -- a tighter
      bound than Rayleigh for high-quality imaging.

    The full depth-of-focus range is ``+/-`` the returned value, so the
    total axial tolerance is ``2 * depth_of_focus(...)``.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength [m].
    f_number : float
        System f-number ``f / D``.  Must be > 0.
    formula : {'rayleigh', 'marechal'}, default 'rayleigh'
        Which DOF expression to evaluate.

    Returns
    -------
    dof : float
        Half-range depth of focus [m].

    Notes
    -----
    With ``NA = 1 / (2 * f#)`` both formulas evaluate to
    ``4 * f#**2 * wavelength`` -- they are mathematically equivalent.
    The two named entries are retained because optical-design
    practice distinguishes them by *derivation* (Rayleigh: OPD margin;
    Marechal: Strehl criterion), and downstream tools may want to
    annotate the choice in reports.

    Examples
    --------
    >>> from lumenairy.analysis import depth_of_focus
    >>> # f/2 at 550 nm, Rayleigh: 4 * 4 * 550e-9 = 8.8 um
    >>> float(depth_of_focus(550e-9, 2.0))
    8.8e-06
    >>> # Same system, Marechal: 550e-9 / (1/4)**2 = 8.8 um
    >>> float(depth_of_focus(550e-9, 2.0, formula='marechal'))
    8.8e-06
    """
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(
            f"depth_of_focus: wavelength must be positive and finite; "
            f"got {wavelength!r}.")
    if not np.isfinite(f_number) or f_number <= 0:
        raise ValueError(
            f"depth_of_focus: f_number must be positive and finite; "
            f"got {f_number!r}.")

    f = float(f_number)
    wl = float(wavelength)
    if formula == 'rayleigh':
        return 4.0 * f * f * wl
    if formula == 'marechal':
        # NA = 1 / (2 * f#) gives DOF = wavelength / NA**2 =
        # 4 * f#**2 * wavelength.  This matches the Rayleigh
        # expression with a factor of 1 instead of 4 because the
        # Marechal criterion is tighter; the standard textbook form
        # is wavelength / NA**2.
        NA = 1.0 / (2.0 * f)
        return wl / (NA * NA)
    raise ValueError(
        f"depth_of_focus: formula must be 'rayleigh' or 'marechal'; "
        f"got {formula!r}.")


# ============================================================================
# Multi-wavelength / chromatic analysis
# ============================================================================

def chromatic_focal_shift(
    prescription: Dict[str, Any],
    wavelengths: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the paraxial focal length at each wavelength and return
    the chromatic focal shift (axial colour).

    Parameters
    ----------
    prescription : dict
    wavelengths : sequence of float
        Wavelengths [m] to evaluate.

    Returns
    -------
    efls : ndarray
        Effective focal length at each wavelength [m].
    bfls : ndarray
        Back focal length at each wavelength [m].
    shift : float
        Peak-valley of BFL across wavelengths [m] (= axial colour).
    """
    from ..raytrace import surfaces_from_prescription, system_abcd

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    efls = np.empty_like(wavelengths)
    bfls = np.empty_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        surfs = surfaces_from_prescription(prescription)
        _, efl, bfl, _ = system_abcd(surfs, float(wl))
        efls[i] = float(efl)
        bfls[i] = float(bfl)
    shift = float(bfls.max() - bfls.min())
    return efls, bfls, shift


def polychromatic_strehl(
    prescription: Dict[str, Any],
    wavelengths: Sequence[float],
    weights: Sequence[float],
    N: int,
    dx: float,
    E_in: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the polychromatic Strehl ratio.

    Propagates a plane wave through the lens at each wavelength,
    finds the best focus for each, and combines the weighted peak
    intensities.

    Parameters
    ----------
    prescription : dict
    wavelengths : sequence of float
    weights : sequence of float
        Relative spectral weights (summed to 1 internally).
    N, dx : int, float
        Wave-grid parameters.
    E_in : ndarray, optional
        Input field (default: unit plane wave).

    Returns
    -------
    strehl_poly : float
        Weighted average Strehl ratio across wavelengths.
    strehls : ndarray
        Per-wavelength Strehl ratios.
    z_bests : ndarray
        Per-wavelength best-focus positions [m].
    """
    from ..elements.lenses import apply_real_lens
    from .through_focus import (through_focus_scan, find_best_focus,
                                diffraction_limited_peak)
    from ..raytrace import surfaces_from_prescription, system_abcd

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    strehls = np.empty(len(wavelengths))
    z_bests = np.empty(len(wavelengths))
    # 4.11.2: honour the global precision context (single vs double) by
    # routing through get_default_complex_dtype() instead of hard-coding
    # complex128.  Pre-4.11.2 this silently coerced single-precision
    # users back to double.
    from ..propagators.propagation import get_default_complex_dtype
    cdtype = get_default_complex_dtype()
    if E_in is None:
        E_in = np.ones((N, N), dtype=cdtype)

    for i, wl in enumerate(wavelengths):
        surfs = surfaces_from_prescription(prescription)
        _, _, bfl, _ = system_abcd(surfs, float(wl))
        E_exit = apply_real_lens(E_in, prescription=prescription, wavelength=float(wl), dx=dx)
        ideal = diffraction_limited_peak(E_exit, float(wl), bfl, dx)
        half = max(abs(bfl) / 20.0, 1e-3)
        z = np.linspace(bfl - half, bfl + half, 21)
        scan = through_focus_scan(E_exit, dx, float(wl), z,
                                   ideal_peak=ideal, verbose=False)
        z_best, s_best = find_best_focus(scan, 'strehl')
        strehls[i] = float(s_best)
        z_bests[i] = float(z_best)

    strehl_poly = float(np.sum(weights * strehls))
    return strehl_poly, strehls, z_bests


def polychromatic_psf(
    prescription: Dict[str, Any],
    wavelengths: Sequence[float],
    weights: Sequence[float],
    N: int,
    dx: float,
    *,
    E_in: Optional[np.ndarray] = None,
    image_distance: Optional[float] = None,
    normalize: str = 'power',
    bandlimit: bool = True,
    return_components: bool = False,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Accumulate a polychromatic PSF on a common image-plane grid.

    For each wavelength in ``wavelengths`` propagates a pupil-plane
    field through the prescription, propagates from the exit pupil
    to a common image plane via the angular-spectrum method, and
    sums the per-wavelength intensities weighted by ``weights``.
    Companion to :func:`polychromatic_strehl`, which only returns
    scalar Strehl ratios -- this routine returns the full integrated
    intensity map at the detector.

    Parameters
    ----------
    prescription : dict
        Lumenairy lens prescription (see [[Function Reference Prescriptions]]).
    wavelengths : sequence of float
        Vacuum wavelengths [m].  Typically 3-10 samples bracketing
        the operating band; more samples give smoother chromatic
        broadening but linearly more work.
    weights : sequence of float
        Per-wavelength spectral weights (e.g. blackbody, LED emission
        curve, AM1.5 solar).  Re-normalised internally so they sum
        to 1.
    N, dx : int, float
        Pupil-plane grid (input field is ``N x N`` on pitch ``dx``).
    E_in : ndarray (complex, N x N), optional
        Pupil-plane input field.  Defaults to a unit plane wave
        (constant amplitude 1, flat phase).
    image_distance : float, optional
        Common image-plane distance measured from the **exit surface
        of the lens** [m].  All wavelengths are propagated to this
        same plane so the per-wavelength PSFs live on a common grid
        and can be summed directly.  Defaults to the paraxial back
        focal length at the centroid wavelength
        ``sum(weights * wavelengths)``.
    normalize : ``'power'`` (default) / ``'peak'`` / ``'none'``
        Output scaling of the accumulated PSF intensity:

        * ``'power'``: ``sum(psf) * dx**2 == 1``.  Correct for
          relative-throughput / encircled-energy work.
        * ``'peak'``: ``psf.max() == 1``.  Useful for PSF-shape
          display.
        * ``'none'``: raw weighted-sum of per-wavelength
          ``|E_image|**2``.
    bandlimit : bool, default True
        Forwarded to the internal :func:`angular_spectrum_propagate`
        calls (Matsushima-Shimobaba band-limit on the ASM transfer
        function).
    return_components : bool, default False
        If True, also return the per-wavelength PSF stack as an
        ``(n_wavelengths, N, N)`` ndarray under
        ``info['per_wavelength_psf']``.  Memory-hungry for large N.

    Returns
    -------
    psf_poly : ndarray (real, N x N)
        Weighted-sum polychromatic PSF on the input grid, scaled
        according to ``normalize``.
    dx_psf : float
        Image-plane grid spacing [m].  Equals ``dx`` -- the ASM
        propagation preserves the grid.
    info : dict
        Diagnostic metrics:

        * ``'wavelengths'``, ``'weights'``, ``'image_distance'``
          -- echoed inputs (weights are the renormalised values).
        * ``'centroid_wavelength'`` [m] -- spectral centroid.
        * ``'per_wavelength_strehl'`` -- peak / diffraction-limited
          peak at the common image plane (NOT each wavelength's
          own best-focus, so this is the chromatic-defocus-included
          Strehl).  Informational only -- on coarse grids small
          deviations above 1.0 can occur because the reference
          uses ASM with band-limit, whereas the aberrated peak
          inherits the lens's own ASM step; for canonical Strehl
          ratios use :func:`polychromatic_strehl` (per-wavelength
          best focus).
        * ``'per_wavelength_peak'`` -- raw peak intensity at each
          wavelength (same units as the input).
        * ``'centroid'`` -- intensity-weighted ``(x, y)`` of the
          accumulated PSF [m].
        * ``'d4sigma'`` -- D4-sigma widths ``(Dx, Dy)`` of the
          accumulated PSF [m].
        * ``'per_wavelength_psf'`` -- per-wavelength stack (only if
          ``return_components=True``).

    Notes
    -----
    The "common image plane" approach means each wavelength experiences
    its own chromatic defocus relative to the paraxial focus at the
    centroid wavelength -- exactly what a real broadband detector
    sees.  This is the right tool for **answering "what does my camera
    record"** rather than "what's the diffraction-limited PSF at this
    wavelength?".

    For the latter (per-wavelength best-focus Strehl), use
    :func:`polychromatic_strehl`.  For polychromatic OTF / MTF, take
    the FFT of the returned ``psf_poly``:

    .. code-block:: python

        otf_poly = lm.compute_otf(psf_poly)
        mtf_poly = np.abs(otf_poly)

    See also
    --------
    polychromatic_strehl : scalar Strehl with per-wavelength best
        focus.
    compute_psf : monochromatic PSF from a pupil function.
    """
    from ..elements.lenses import apply_real_lens
    from ..propagators.propagation import angular_spectrum_propagate
    from ..raytrace import surfaces_from_prescription, system_abcd
    from .through_focus import diffraction_limited_peak

    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if wavelengths.size != weights.size:
        raise ValueError(
            f"wavelengths and weights must have the same length; "
            f"got {wavelengths.size} and {weights.size}.")
    if wavelengths.size == 0:
        raise ValueError("Need at least one wavelength.")
    weights = weights / weights.sum()

    centroid_wl = float(np.sum(weights * wavelengths))

    if image_distance is None:
        surfs = surfaces_from_prescription(prescription)
        _, _, bfl, _ = system_abcd(surfs, centroid_wl)
        image_distance = float(bfl)
    image_distance = float(image_distance)

    # 4.11.2: honour the global precision context (single vs double).
    from ..propagators.propagation import get_default_complex_dtype
    cdtype = get_default_complex_dtype()
    if E_in is None:
        E_in = np.ones((N, N), dtype=cdtype)

    psf_acc = np.zeros((N, N), dtype=np.float64)
    per_peak = np.empty(wavelengths.size, dtype=np.float64)
    per_strehl = np.empty(wavelengths.size, dtype=np.float64)
    components = None
    if return_components:
        components = np.empty((wavelengths.size, N, N), dtype=np.float64)

    for i, wl in enumerate(wavelengths):
        wl_f = float(wl)
        E_exit = apply_real_lens(E_in, prescription=prescription, wavelength=wl_f, dx=dx)
        E_image = angular_spectrum_propagate(
            E_exit, image_distance, wl_f, dx, bandlimit=bandlimit)
        I = np.abs(E_image) ** 2
        per_peak[i] = float(I.max())
        # Strehl reference: amplitude-only-pupil propagated to the
        # SAME common image_distance with a converging phase tuned
        # to that distance.  Diverges from a "per-wavelength best
        # focus" reference -- intentional, because the reported
        # Strehl is the chromatic-defocus-aware peak ratio at the
        # detector plane.
        peak_ref = diffraction_limited_peak(
            E_exit, wl_f, image_distance, dx, bandlimit=bandlimit)
        per_strehl[i] = (per_peak[i] / peak_ref
                         if peak_ref > 0 else 0.0)
        contribution = float(weights[i]) * I
        psf_acc += contribution
        if components is not None:
            components[i] = I

    # 4.13.2 (C-P1-1): use ``dx * dy`` for the pixel-area normalisation
    # when ``dy`` is explicitly provided; fall back to ``dx ** 2`` when
    # the caller omits ``dy`` to preserve bit-for-bit backward
    # compatibility with the v4.13.1 default-square path.  Pre-4.13.2
    # the v4.13.0 L3 sweep missed this site and anamorphic input
    # grids mis-scaled the integrated PSF power.
    if dy is None:
        pixel_area = dx ** 2
    else:
        pixel_area = dx * dy
    if normalize == 'power':
        total = float(psf_acc.sum() * pixel_area)
        if total > 0:
            psf_out = psf_acc / total
        else:
            psf_out = psf_acc
    elif normalize == 'peak':
        peak = float(psf_acc.max())
        psf_out = psf_acc / peak if peak > 0 else psf_acc
    elif normalize == 'none':
        psf_out = psf_acc
    else:
        raise ValueError(
            f"Unknown normalize mode: {normalize!r}.  "
            f"Use 'power', 'peak', or 'none'.")

    # Centroid + D4-sigma diagnostics over the accumulated PSF.
    total = float(psf_out.sum())
    if total > 0:
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        xc = float((psf_out * X).sum() / total)
        yc = float((psf_out * Y).sum() / total)
        var_x = float((psf_out * (X - xc) ** 2).sum() / total)
        var_y = float((psf_out * (Y - yc) ** 2).sum() / total)
        d4x = 4.0 * np.sqrt(max(var_x, 0.0))
        d4y = 4.0 * np.sqrt(max(var_y, 0.0))
    else:
        xc = yc = 0.0
        d4x = d4y = 0.0

    info = {
        'wavelengths': wavelengths,
        'weights': weights,
        'image_distance': image_distance,
        'centroid_wavelength': centroid_wl,
        'per_wavelength_peak': per_peak,
        'per_wavelength_strehl': per_strehl,
        'centroid': (xc, yc),
        'd4sigma': (d4x, d4y),
    }
    if components is not None:
        info['per_wavelength_psf'] = components

    return psf_out, float(dx), info


# ============================================================================
# Zernike polynomial decomposition of OPD / wavefront maps
# ============================================================================
#
# We use the OSA / ANSI single-index ordering
#
#   j  |  (n, m)  |  classical name
#   --|----------|-----------------
#   0  |  (0,  0) |  Piston
#   1  |  (1, -1) |  Tilt  Y
#   2  |  (1,  1) |  Tilt  X
#   3  |  (2, -2) |  Oblique astigmatism
#   4  |  (2,  0) |  Defocus
#   5  |  (2,  2) |  Vertical astigmatism
#   6  |  (3, -3) |  Oblique trefoil
#   7  |  (3, -1) |  Vertical coma
#   8  |  (3,  1) |  Horizontal coma
#   9  |  (3,  3) |  Vertical trefoil
#  10  |  (4, -4) |  Oblique quadrafoil
#  11  |  (4, -2) |  Oblique secondary astigmatism
#  12  |  (4,  0) |  Primary spherical
#  13  |  (4,  2) |  Vertical secondary astigmatism
#  14  |  (4,  4) |  Vertical quadrafoil
#   ...
#
# All Zernikes are normalised so that the rms of each mode over the
# unit disk is 1.  Coefficients returned by :func:`zernike_decompose`
# are therefore directly interpretable as RMS contributions in the
# same units as the input OPD (meters if OPD is in meters).

def zernike_index_to_nm(j: int) -> Tuple[int, int]:
    """Convert OSA single-index ``j`` to (n, m) Zernike indices."""
    j = int(j)
    if j < 0:
        raise ValueError(f"j must be >= 0, got {j}")
    # n = floor( (-1 + sqrt(1 + 8*j)) / 2 )
    n = int((-1 + np.sqrt(1 + 8 * j)) // 2)
    # Ensure n is large enough; guard floating-point edge
    while (n * (n + 2) - (j - n * (n + 1) // 2) * 2) < 0:
        n += 1
    m = 2 * j - n * (n + 2)
    return n, m


def zernike_nm_to_index(n: int, m: int) -> int:
    """Convert Zernike (n, m) to OSA single-index ``j``."""
    return (n * (n + 2) + m) // 2


def _zernike_radial(n, m, rho):
    """Radial polynomial R_n^m(rho) for rho in [0, 1].

    Computed via the explicit closed-form sum; stable and fast for
    ``n <= 20``.  Returns zero outside the unit disk.
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    import math as _math
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * _math.factorial(n - s)
        den = (_math.factorial(s)
               * _math.factorial((n + m) // 2 - s)
               * _math.factorial((n - m) // 2 - s))
        R = R + (num / den) * rho ** (n - 2 * s)
    return R


def zernike_polynomial(
    n: int,
    m: int,
    rho: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Evaluate the OSA-normalised Zernike Z_n^m on pupil polar
    coordinates.

    Parameters
    ----------
    n : int
        Radial order (``n >= 0``).
    m : int
        Azimuthal order (``|m| <= n``, ``(n-m)`` even).
    rho : ndarray
        Normalised radius (0 outside pupil, 1 at pupil edge).
    theta : ndarray
        Azimuthal angle [rad], same shape as ``rho``.

    Returns
    -------
    Z : ndarray
        Zernike polynomial evaluated at (rho, theta), normalised so
        that the rms of ``Z`` over the unit disk equals 1.
    """
    rho = np.asarray(rho)
    theta = np.asarray(theta)
    if (n - abs(m)) % 2 != 0 or abs(m) > n:
        raise ValueError(f"Invalid Zernike indices (n, m) = ({n}, {m})")
    # Normalisation constant (Noll 1976)
    if m == 0:
        N = np.sqrt(n + 1)
    else:
        N = np.sqrt(2 * (n + 1))
    R = _zernike_radial(n, m, rho)
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        angular = np.sin(-m * theta)
    Z = N * R * angular
    # Zero outside pupil
    Z = np.where(rho <= 1.0, Z, 0.0)
    return Z


# ---------------------------------------------------------------------
# Zernike basis-matrix cache (audit perf #7, v4.12.0)
# ---------------------------------------------------------------------
#
# ``zernike_basis_matrix`` is hot during ``design_optimize`` runs:
# every Zernike-using merit term (RMSWavefrontMerit,
# MatchIdealThinLensMerit, MatchTargetOPDMerit, ZernikeCoefficientMerit,
# EvaluationContext.rms_wavefront_waves) rebuilds it on every call, and
# ``CompositeMerit`` evaluates several of these per ``evaluate(x)``.
# Across a finite-difference Jacobian sweep that's 60-100x rebuild of
# the same basis on the same grid -- ~20 ms each at 256x256, 21 modes.
#
# Cache strategy: content-fingerprint key.  The pupil grids X, Y are
# deterministic functions of (N, dx) -- so two distinct arrays with
# matching shape, dtype, and corner values are guaranteed (to within
# floating-point noise of the meshgrid arithmetic) to describe the
# same pupil.  Keying on the fingerprint (NOT on ``id()``) lets fresh
# ``np.meshgrid`` outputs across successive ``zernike_decompose`` /
# ``EvaluationContext.rms_wavefront_waves`` calls all hit the same
# cached basis.
#
# Trade-off: if a caller mutates an input array in place but does not
# change shape, dtype, or the first/last entries, the cache will
# still hit and return the stale basis.  This is highly unlikely in
# practice (pupil grids are rebuilt fresh each call) but documented
# here.  Use ``clear_zernike_basis_cache()`` to force a rebuild.
# ---------------------------------------------------------------------

_ZERNIKE_BASIS_CACHE: "OrderedDict[Any, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_ZERNIKE_BASIS_CACHE_MAXSIZE = 32
# v4.14.2 (P1-NEW-2 / Agent C): thread-safety lock for
# ``_ZERNIKE_BASIS_CACHE``.  Without this two threads racing through
# :func:`zernike_basis_matrix` could see a torn OrderedDict (``get`` ->
# ``__setitem__`` -> ``popitem`` is a read-modify-write sequence).
# Follows the ``_ASM_CACHE_LOCK`` precedent in
# :mod:`propagators.propagation`; the build itself (``_zernike_basis_
# matrix_build``) is pure-CPU numpy and re-entrant so it runs OUTSIDE
# the lock -- only the OrderedDict ops need guarding.
_ZERNIKE_BASIS_CACHE_LOCK = threading.Lock()


def _zernike_basis_cache_key(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[Any, ...]:
    """Build a (cheap) cache key for ``zernike_basis_matrix``.

    Uses a small content fingerprint -- shape, dtype, and the first +
    last entries of the grid -- as the cache key.  This intentionally
    *omits* object identity so that two distinct arrays produced by
    independent ``np.meshgrid`` calls (the common case in
    ``zernike_decompose`` and ``EvaluationContext.rms_wavefront_waves``)
    hit the same cache entry as long as their structure agrees.

    Trade-off: if a caller mutates the input arrays in place but does
    not change shape, dtype, or the corner values, the cache will
    still hit and return a stale basis.  This is unlikely in practice
    (pupil grids are rebuilt fresh each call) but documented for
    completeness.  Call ``clear_zernike_basis_cache()`` to force a
    rebuild.
    """
    # Coerce to ndarray for shape/dtype/corner access without copying.
    Xa = np.asarray(X)
    Ya = np.asarray(Y)
    return (
        int(n_modes),
        Xa.shape, Xa.dtype.str,
        Ya.shape, Ya.dtype.str,
        float(Xa.flat[0]), float(Xa.flat[-1]),
        float(Ya.flat[0]), float(Ya.flat[-1]),
        float(pupil_radius),
    )


def clear_zernike_basis_cache() -> None:
    """Drop every cached Zernike basis matrix.

    Useful when the caller has mutated a coordinate grid in place and
    wants the next ``zernike_basis_matrix`` call to recompute from
    scratch.  Also handy in unit tests that pin cache behaviour.
    """
    with _ZERNIKE_BASIS_CACHE_LOCK:
        _ZERNIKE_BASIS_CACHE.clear()


def _zernike_basis_matrix_build(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uncached build of the Zernike basis matrix.  See
    :func:`zernike_basis_matrix` for the public, cached entry point.
    """
    r_sq = (X ** 2 + Y ** 2) / (pupil_radius ** 2)
    pupil_mask = r_sq <= 1.0
    rho = np.sqrt(r_sq[pupil_mask])
    theta = np.arctan2(Y[pupil_mask], X[pupil_mask])

    n_pixels = rho.size
    basis = np.empty((n_pixels, n_modes), dtype=np.float64)
    for j in range(n_modes):
        n, m = zernike_index_to_nm(j)
        basis[:, j] = zernike_polynomial(n, m, rho, theta)
    return basis, pupil_mask


def zernike_basis_matrix(
    n_modes: int,
    X: np.ndarray,
    Y: np.ndarray,
    pupil_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a design matrix of the first ``n_modes`` Zernike
    polynomials evaluated on the grid ``(X, Y)``.

    Returns a ``(N_pupil_pixels, n_modes)`` 2-D array where each row
    is one pupil pixel and each column is one Zernike mode.  Only
    pixels inside the pupil (``x^2 + y^2 <= pupil_radius^2``) are
    included in the returned matrix.

    Parameters
    ----------
    n_modes : int
    X, Y : ndarray
        Pupil-plane coordinate grids [m] (same shape).
    pupil_radius : float
        Pupil radius [m].

    Returns
    -------
    basis : ndarray, shape (N_pixels, n_modes)
    pupil_mask : ndarray of bool, same shape as X
        True where pixels are inside the pupil (= rows of ``basis``).

    Notes
    -----
    Results are cached (LRU, ``maxsize=32``) keyed on a small content
    fingerprint of the input grids (shape, dtype, first + last
    entries) plus ``n_modes`` and ``pupil_radius``.  Two structurally
    identical grids hit the same cache entry even if they are distinct
    array objects -- this is the common case under
    ``design_optimize``.  The same underlying arrays are returned on a
    cache hit -- treat the returned ``basis`` and ``pupil_mask`` as
    immutable; copy first if you need to modify them.  Call
    :func:`clear_zernike_basis_cache` to drop the cache (e.g. after
    mutating a grid in place).
    """
    key = _zernike_basis_cache_key(n_modes, X, Y, pupil_radius)
    with _ZERNIKE_BASIS_CACHE_LOCK:
        cached = _ZERNIKE_BASIS_CACHE.get(key)
        if cached is not None:
            # LRU bump: mark as most recently used.
            _ZERNIKE_BASIS_CACHE.move_to_end(key)
            return cached

    # Cache miss: build the basis OUTSIDE the lock (pure-CPU numpy,
    # re-entrant; lock-scope discipline keeps expensive work off the
    # critical section).  Two threads may double-build on a cold
    # cache for the same key -- benign waste, the second insert just
    # overwrites the first.
    basis, pupil_mask = _zernike_basis_matrix_build(
        n_modes, X, Y, pupil_radius)

    with _ZERNIKE_BASIS_CACHE_LOCK:
        _ZERNIKE_BASIS_CACHE[key] = (basis, pupil_mask)
        # Evict LRU entries until we're at or below the cap.
        while len(_ZERNIKE_BASIS_CACHE) > _ZERNIKE_BASIS_CACHE_MAXSIZE:
            _ZERNIKE_BASIS_CACHE.popitem(last=False)
    return basis, pupil_mask


def zernike_decompose(
    opd_map: np.ndarray,
    dx: float,
    aperture: float,
    n_modes: int = 21,
    dy: Optional[float] = None,
    return_residual: bool = False,
) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str], np.ndarray, float]]:
    """Decompose a 2-D OPD map into Zernike coefficients using a
    numerically-stable Householder QR least-squares solve.

    Parameters
    ----------
    opd_map : ndarray (2-D, real)
        Optical path difference [m] over a grid.  Values outside the
        pupil may be ``NaN`` or 0; they are masked out before fitting.
    dx : float
        Grid spacing in x [m].
    aperture : float
        Clear aperture diameter [m].  Defines the pupil radius as
        ``aperture / 2``.
    n_modes : int, default 21
        Number of OSA-indexed Zernike modes to fit.  21 covers up
        through 5th-order spherical.  Higher = finer detail at the
        cost of ill-conditioning for sparsely-illuminated pupils.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    return_residual : bool, default False
        Also return the 2-D residual ``opd_map - reconstruction`` and
        its RMS.

    Returns
    -------
    coeffs : ndarray, shape (n_modes,)
        Fitted Zernike coefficients, units of ``opd_map`` (typically
        meters).  With OSA normalisation, the RMS wavefront error
        contributed by mode ``j`` is ``|coeffs[j]|``.
    names : list of str
        Human-readable name for each mode (e.g. ``'Defocus'``,
        ``'Primary spherical'``).
    residual : ndarray, optional
        ``opd_map - reconstruction``, 2-D.  Only returned when
        ``return_residual=True``.
    rms_residual : float, optional
        RMS of ``residual`` over the pupil, same units as
        ``opd_map``.  Only returned when ``return_residual=True``.

    Notes
    -----
    Uses ``scipy.linalg.lstsq(..., lapack_driver='gelsy')`` under the
    hood, which is a column-pivoted Householder QR with rank
    revelation.  This is more stable than the default SVD driver for
    ill-conditioned Zernike bases (common when the pupil is partially
    illuminated or when many modes are requested).
    """
    if dy is None:
        dy = dx
    Ny, Nx = opd_map.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_pupil = 0.5 * aperture

    basis, pupil_mask = zernike_basis_matrix(n_modes, X, Y, r_pupil)

    # Flatten the OPD to match basis rows
    opd_flat = opd_map[pupil_mask]
    # Drop NaN/inf rows from both sides
    finite = np.isfinite(opd_flat)
    if not finite.all():
        basis = basis[finite]
        opd_flat = opd_flat[finite]
    if opd_flat.size < n_modes:
        raise ValueError(
            f'Not enough valid pupil samples ({opd_flat.size}) to fit '
            f'{n_modes} modes.  Check aperture/grid alignment or '
            f'reduce n_modes.')

    # Householder QR with column pivoting (gelsy driver)
    try:
        from scipy.linalg import lstsq as _slstsq
        coeffs, _residuals_sq, _rank, _sv = _slstsq(
            basis, opd_flat, lapack_driver='gelsy')
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # Fallback to numpy if scipy lstsq is unavailable
        # (ImportError) or rejects the inputs / diverges
        # (ValueError, LinAlgError).
        coeffs, *_ = np.linalg.lstsq(basis, opd_flat, rcond=None)

    names = [_zernike_classical_name(*zernike_index_to_nm(j))
             for j in range(n_modes)]

    if return_residual:
        # Reconstruct over the full pupil, not just the finite subset
        basis_full, _ = zernike_basis_matrix(
            n_modes, X, Y, r_pupil)
        recon_flat = basis_full @ coeffs
        reconstruction = np.zeros_like(opd_map)
        reconstruction[pupil_mask] = recon_flat
        residual = opd_map - reconstruction
        residual = np.where(pupil_mask, residual, np.nan)
        rms = float(np.sqrt(np.nanmean(residual ** 2)))
        return coeffs, names, residual, rms
    return coeffs, names


def zernike_reconstruct(
    coeffs: np.ndarray,
    dx: float,
    shape: Tuple[int, int],
    aperture: float,
    dy: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct a 2-D OPD map from Zernike coefficients.

    Inverse of :func:`zernike_decompose`: ``opd_map ≈ sum_j coeffs[j]
    * Z_j(x, y)`` inside the pupil, zero outside.

    Parameters
    ----------
    coeffs : ndarray, shape (n_modes,)
        Coefficients in OSA order.
    dx : float
        Grid spacing [m].
    shape : tuple (Ny, Nx)
        Output grid shape.
    aperture : float
        Pupil diameter [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    opd_map : ndarray, shape ``shape``
        Reconstructed OPD, zero outside the pupil.
    """
    if dy is None:
        dy = dx
    Ny, Nx = shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    r_pupil = 0.5 * aperture
    n_modes = len(coeffs)

    basis, pupil_mask = zernike_basis_matrix(n_modes, X, Y, r_pupil)
    recon_flat = basis @ np.asarray(coeffs, dtype=np.float64)
    opd_map = np.zeros(shape, dtype=np.float64)
    opd_map[pupil_mask] = recon_flat
    return opd_map


def _zernike_classical_name(n, m):
    """Return the classical name for a Zernike mode (n, m).

    Covers common aberration names; falls back to ``Z(n, m)`` for
    uncommon modes.
    """
    table = {
        (0, 0): 'Piston',
        (1, -1): 'Tilt Y',
        (1, 1): 'Tilt X',
        (2, -2): 'Oblique astigmatism',
        (2, 0): 'Defocus',
        (2, 2): 'Vertical astigmatism',
        (3, -3): 'Oblique trefoil',
        (3, -1): 'Vertical coma',
        (3, 1): 'Horizontal coma',
        (3, 3): 'Vertical trefoil',
        (4, -4): 'Oblique quadrafoil',
        (4, -2): 'Oblique secondary astigmatism',
        (4, 0): 'Primary spherical',
        (4, 2): 'Vertical secondary astigmatism',
        (4, 4): 'Vertical quadrafoil',
        (5, -1): 'Secondary vertical coma',
        (5, 1): 'Secondary horizontal coma',
        (6, 0): 'Secondary spherical',
        (8, 0): 'Tertiary spherical',
    }
    return table.get((n, m), f'Z({n}, {m})')


# ============================================================================
# Wavefront / OPD analysis
# ============================================================================

def check_opd_sampling(
    dx: float,
    wavelength: float,
    aperture: float,
    focal_length: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check whether grid sampling is adequate for clean OPD extraction
    from a converging wavefront.

    A converging wavefront of focal length ``f`` has a radial phase
    gradient ``k * r / f`` at pupil height ``r``.  At the pupil edge
    ``r = aperture / 2`` this gradient is maximal, so the phase change
    per grid sample is

        dphi = k * (aperture / 2) / f * dx
             = pi * aperture * dx / (wavelength * f)

    ``np.unwrap`` correctly tracks cycles as long as ``|dphi| < pi``
    at every sample, giving the Nyquist sampling rule

        dx <= lambda * f / aperture

    Violating this rule causes ``np.unwrap`` to skip cycles near the
    pupil edge, producing catastrophically wrong OPD values there (the
    classic symptom is a quadratic residual that blows up beyond some
    radius while the inner pupil looks clean).  See
    ``validation/real_lens_opd`` for an empirical illustration.

    Parameters
    ----------
    dx : float
        Grid spacing [m].
    wavelength : float
        Vacuum wavelength [m].
    aperture : float
        Clear aperture diameter [m].
    focal_length : float
        Effective focal length [m] of the optic producing the
        converging wavefront.  For a lens prescription, use the
        paraxial back focal length (BFL) from
        :func:`lumenairy.raytrace.system_abcd`.
    verbose : bool, default True
        Print a human-readable diagnostic.

    Returns
    -------
    result : dict
        ``'ok'`` : bool -- whether sampling is safely above Nyquist.
        ``'margin'`` : float -- ``dx_max / dx`` where dx_max is the
            Nyquist sampling limit.  Margin >= 2 is safe, 1 < margin
            < 2 is marginal, < 1 is failing.
        ``'dx_max'`` : float -- Nyquist-limited maximum dx [m].
        ``'phase_per_sample'`` : float -- radians of phase change per
            sample at the pupil edge (Nyquist limit is pi).
        ``'recommendations'`` : list of str -- suggestions to fix
            marginal or failing sampling.
    """
    f = float(abs(focal_length))
    ap = float(aperture)
    # Phase gradient at pupil edge = k * (ap/2) / f
    # Phase change per sample = gradient * dx
    phase_per_sample = (2 * np.pi / wavelength) * (ap / 2.0) / f * dx

    # Nyquist limit: max dx such that phase_per_sample <= pi
    dx_max = wavelength * f / ap
    margin = dx_max / dx
    ok = margin >= 2.0

    recommendations = []
    if not ok:
        required_dx = 0.5 * dx_max  # 2x safety margin
        recommendations.append(
            f'Reduce dx to <= {required_dx*1e6:.3f} um '
            f'(currently {dx*1e6:.3f} um).')
        recommendations.append(
            f'Or reduce aperture below '
            f'{(wavelength * f / (2 * dx)) * 1e3:.3f} mm at current dx.')
        recommendations.append(
            f'Or use f_ref in wave_opd_1d/2d to subtract the reference '
            f'sphere before unwrapping.')

    if verbose:
        print('--- OPD sampling check ---')
        print(f'  dx                          = {dx*1e6:.3f} um')
        print(f'  wavelength                  = {wavelength*1e9:.1f} nm')
        print(f'  aperture                    = {ap*1e3:.3f} mm')
        print(f'  focal length                = {f*1e3:.3f} mm')
        print(f'  phase change per sample     = {phase_per_sample:.3f} rad '
              f'(Nyquist limit = pi = {np.pi:.3f})')
        print(f'  Nyquist dx_max              = {dx_max*1e6:.3f} um')
        print(f'  margin (dx_max/dx)          = {margin:.2f} '
              f'({"SAFE" if margin >= 2 else ("MARGINAL" if margin >= 1 else "FAIL")})')
        if recommendations:
            print('  Recommendations:')
            for rec in recommendations:
                print(f'    - {rec}')

    return {
        'ok': ok,
        'margin': float(margin),
        'dx_max': float(dx_max),
        'phase_per_sample': float(phase_per_sample),
        'recommendations': recommendations,
    }


def remove_wavefront_modes(
    x: np.ndarray,
    opd: np.ndarray,
    modes: str = 'piston,tilt,defocus',
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Least-squares subtract low-order 1-D wavefront modes from an OPD
    profile.

    Useful for isolating high-order aberrations from an OPD cut.
    Operates on a 1-D OPD profile ``opd(x)`` where ``x`` is a pupil
    coordinate.

    Parameters
    ----------
    x : ndarray
        Pupil coordinate [m], 1-D.
    opd : ndarray
        Optical-path-difference values at ``x``, same length.  May contain
        ``NaN`` for out-of-aperture samples; those are ignored in the fit.
    modes : str
        Comma-separated subset of ``'piston'``, ``'tilt'``, ``'defocus'``.
        Pass ``''`` or ``None`` to fit nothing (returns input unchanged).
    weights : ndarray, optional
        Per-sample non-negative weights (e.g. pupil intensity ``|E|^2``).
        When supplied, the fit minimises ``sum(w_i * (opd_i - fit_i)^2)``
        so that the piston / tilt / defocus split honours where the
        light actually is rather than treating every grid point equally.
        Critical for vignetted, annular, or sparsely-illuminated pupils
        where unweighted fits leak high-order content into the low-order
        coefficients.  Default ``None`` reproduces the legacy uniform
        behaviour bit-for-bit.

    Returns
    -------
    opd_residual : ndarray
        ``opd`` minus the fitted modes.
    coeffs : dict
        Fit coefficients for each included mode.  Keys match the names
        passed in ``modes``.  Units: piston [m]; tilt [dimensionless
        slope]; defocus [1/m] (coefficient of x**2).

    Notes
    -----
    "Piston" is a constant phase offset -- physically irrelevant because
    detectors only see intensity.  "Tilt" is a linear phase ramp -- it
    just shifts the image laterally.  "Defocus" is a quadratic ``x**2``
    term -- it moves the focal plane axially.  Remove one, several, or
    all of these to isolate the "interesting" aberration content.
    """
    x = np.asarray(x)
    opd = np.asarray(opd)

    if not modes:
        return opd.copy(), {}
    mode_set = set(m.strip() for m in modes.split(',') if m.strip())

    cols, names = [], []
    if 'piston' in mode_set:
        cols.append(np.ones_like(x))
        names.append('piston')
    if 'tilt' in mode_set:
        cols.append(x)
        names.append('tilt')
    if 'defocus' in mode_set:
        cols.append(x ** 2)
        names.append('defocus')

    if not cols:
        return opd.copy(), {}

    A = np.column_stack(cols)
    mask = np.isfinite(opd)
    if not mask.any():
        return opd.copy(), {}

    if weights is None:
        coeffs, *_ = np.linalg.lstsq(A[mask], opd[mask], rcond=None)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != opd.shape:
            raise ValueError(
                f"weights shape {w.shape} != opd shape {opd.shape}")
        # Drop non-finite / non-positive weights from the fit.
        wmask = mask & np.isfinite(w) & (w > 0)
        if not wmask.any():
            return opd.copy(), {}
        sw = np.sqrt(w[wmask])
        coeffs, *_ = np.linalg.lstsq(
            A[wmask] * sw[:, None], opd[wmask] * sw, rcond=None)
    fit = A @ coeffs
    return opd - fit, dict(zip(names, coeffs.tolist()))


def opd_pv_rms(opd: np.ndarray) -> Tuple[float, float]:
    """Peak-valley and RMS of a 1-D or 2-D OPD array.

    Parameters
    ----------
    opd : ndarray
        OPD values.  ``NaN`` entries are ignored.

    Returns
    -------
    pv : float
        Peak-valley (max - min), in the same units as ``opd``.
    rms : float
        RMS deviation from the mean, in the same units as ``opd``.
    """
    arr = np.asarray(opd)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan'), float('nan')
    pv = float(finite.max() - finite.min())
    rms = float(np.sqrt(np.mean((finite - finite.mean()) ** 2)))
    return pv, rms


def wave_opd_1d(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    axis: str = 'x',
    aperture: Optional[float] = None,
    dy: Optional[float] = None,
    focal_length: Optional[float] = None,
    f_ref: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a 1-D OPD profile along the central row or column of a
    complex field.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric field on a regular grid.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].  Used to convert unwrapped phase to OPL.
    axis : ``'x'`` or ``'y'``
        Which pupil cut to extract.  ``'x'`` takes the row ``y = 0``;
        ``'y'`` takes the column ``x = 0``.
    aperture : float, optional
        Clear-aperture diameter [m].  If given, the returned profile is
        cropped to |pupil coordinate| <= 0.5 * aperture and any
        out-of-aperture zero-amplitude samples are excluded from
        unwrapping.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.

    Returns
    -------
    coord : ndarray
        Pupil coordinate [m] for each returned sample.
    opd : ndarray
        Optical path length [m], ``+phase / k0`` with ``np.unwrap``
        applied along the cut.

    Notes
    -----
    * The sign convention assumes a forward-propagating wave, for which
      the phase at a given height equals ``+k * OPL``.
    * Unwrapping along a single row requires ``dx`` fine enough that
      the phase change between adjacent samples is below ``pi``.  For a
      lens of focal length ``f``, the worst case is at the pupil edge:
      ``dx < lambda * f / pupil_diameter``.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E.shape
    k0 = 2 * np.pi / wavelength

    # Emit a Nyquist sampling warning if focal_length is known and
    # sampling is marginal / failing.
    if focal_length is not None and aperture is not None:
        samp = check_opd_sampling(
            dx, wavelength, aperture, focal_length, verbose=False)
        if not samp['ok']:
            import warnings as _w
            _w.warn(
                f'wave_opd_1d: Nyquist sampling is '
                f'{"failing" if samp["margin"] < 1 else "marginal"} '
                f'(margin = {samp["margin"]:.2f}).  Phase unwrap may '
                f'lose cycles near the pupil edge, producing '
                f'catastrophically wrong OPD values there.  '
                f'Recommended: {samp["recommendations"][0] if samp["recommendations"] else "see check_opd_sampling"}',
                RuntimeWarning, stacklevel=2)

    if axis == 'x':
        row = E[Ny // 2, :]
        coord = (np.arange(Nx) - Nx / 2) * dx
    elif axis == 'y':
        row = E[:, Nx // 2]
        coord = (np.arange(Ny) - Ny / 2) * dy
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    # Optional reference-sphere subtraction: for strongly-converging
    # wavefronts we can divide out ``exp(-i*k0*coord**2 / (2*f_ref))``
    # before unwrap so the residual phase is small and unwrap is
    # robust regardless of sampling.  Caller must add the reference
    # phase back to the returned OPD.
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        ref_phase = -k0 * coord ** 2 / (2.0 * f_ref)
        row = row * np.exp(-1j * ref_phase)  # conjugate ref sphere

    valid = np.abs(row) > 0
    if aperture is not None:
        valid = valid & (np.abs(coord) <= 0.5 * aperture)

    if not valid.any():
        raise ValueError("No valid samples along the selected cut.")

    idx = np.where(valid)[0]
    i0, i1 = idx[0], idx[-1]
    row_crop = row[i0:i1 + 1]
    coord_crop = coord[i0:i1 + 1]

    phase = np.unwrap(np.angle(row_crop))
    opd = phase / k0

    # Add back the reference sphere so the returned OPD is absolute
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        opd = opd + (-coord_crop ** 2 / (2.0 * f_ref))
    return coord_crop, opd


def wave_opd_2d(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    aperture: Optional[float] = None,
    dy: Optional[float] = None,
    f_ref: Optional[float] = None,
    focal_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2-D OPD map from a complex field over its pupil.

    For converging wavefronts with many fringes, a reference spherical
    wave of focal length ``f_ref`` can be divided out before unwrapping
    so that the remaining phase is small enough for robust 2-D unwrap
    (currently a simple Itoh-style unwrap along rows followed by
    columns).  Pass ``f_ref=None`` for nearly-flat wavefronts only.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Complex electric field on a regular grid.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].
    aperture : float, optional
        Clear-aperture diameter [m].  Samples outside the aperture
        (and any with |E| == 0) are set to ``NaN`` in the returned map.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    f_ref : float, optional
        If given, divide ``E`` by ``exp(-1j * k0 * r**2 / (2 * f_ref))``
        before unwrap.  The returned map is then the OPD *deviation* from
        that reference sphere.  Supply the paraxial focal length to
        flatten the converging wavefront before unwrap.

    Returns
    -------
    X, Y : ndarray
        Pupil coordinate grids [m], same shape as ``opd_map``.
    opd_map : ndarray
        2-D OPD in meters.  ``NaN`` outside the aperture.

    Notes
    -----
    Quality of the 2-D unwrap depends on the residual phase after
    reference-sphere subtraction being well under ``pi`` per sample.
    For diagnostic OPD maps over small apertures a simple row-then-
    column unwrap is adequate; for large, noisy, or vortex-containing
    wavefronts use a dedicated 2-D unwrap library.
    """
    if dy is None:
        dy = dx

    Ny, Nx = E.shape
    k0 = 2 * np.pi / wavelength

    # Emit a Nyquist sampling warning if focal_length is known and
    # sampling is marginal / failing (see wave_opd_1d for rationale).
    if focal_length is not None and aperture is not None and f_ref is None:
        samp = check_opd_sampling(
            dx, wavelength, aperture, focal_length, verbose=False)
        if not samp['ok']:
            import warnings as _w
            _w.warn(
                f'wave_opd_2d: Nyquist sampling is '
                f'{"failing" if samp["margin"] < 1 else "marginal"} '
                f'(margin = {samp["margin"]:.2f}).  2-D unwrap may '
                f'lose cycles near the pupil edge.  '
                f'Recommended: pass f_ref={focal_length:.4g} to divide '
                f'out the reference sphere before unwrap, or {samp["recommendations"][0] if samp["recommendations"] else "reduce aperture / dx"}',
                RuntimeWarning, stacklevel=2)

    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    field = E.copy()
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        # Remove ideal converging reference sphere.  A lens of focal
        # length f imparts phase exp(-i k0 r^2 / (2 f)); dividing by
        # that is the same as multiplying by the conjugate.
        field = field * np.exp(+1j * k0 * (X ** 2 + Y ** 2) / (2.0 * f_ref))

    valid = np.abs(field) > 0
    if aperture is not None:
        valid = valid & (X ** 2 + Y ** 2 <= (0.5 * aperture) ** 2)

    phase = np.angle(field)

    # Row-then-column unwrap.  Crude but adequate when the residual
    # phase is smooth and the aperture is simply connected.
    # v4.13.0 perf: np.unwrap accepts axis=, so the Python row-and-
    # column double-loop collapses into two compiled C calls.  Same
    # 2-D path-integral unwrap, ~5-10x faster on N>=512.
    phase_unwrapped = np.unwrap(phase, axis=1)
    phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)

    opd = phase_unwrapped / k0
    if f_ref is not None and np.isfinite(f_ref) and f_ref != 0.0:
        # Add the reference sphere back so the returned OPD is
        # ABSOLUTE (matching wave_opd_1d's convention), not a
        # deviation.  This makes f_ref purely a numerical
        # conditioning knob, not a physical reinterpretation.
        opd = opd + (-(X ** 2 + Y ** 2) / (2.0 * f_ref))

    opd = np.where(valid, opd, np.nan)
    return X, Y, opd
