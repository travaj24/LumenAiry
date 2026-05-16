"""
Detector and wavefront-sensor simulation.

Models the final stage of an optical chain: converting a coherent
complex field into a measured intensity image with realistic noise
and pixel response.

Provides:

* :func:`apply_detector` — pixel-integrate a field onto a detector
  grid with optional Poisson shot noise and Gaussian read noise.
* :func:`shack_hartmann` — simulate a Shack-Hartmann wavefront sensor
  (microlens array + detector) and reconstruct the wavefront.

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..propagators.propagation import angular_spectrum_propagate


__all__ = [
    'apply_detector',
    'shack_hartmann',
]


def apply_detector(
    E: np.ndarray,
    dx_field: float,
    pixel_pitch: float,
    n_pixels: Optional[int] = None,
    exposure_time: float = 1.0,
    quantum_efficiency: float = 1.0,
    read_noise_e: float = 0.0,
    dark_current_e_per_s: float = 0.0,
    full_well: float = np.inf,
    seed: Optional[int] = None,
    hot_pixel_map: Optional[np.ndarray] = None,
    cosmic_ray_rate: float = 0.0,
    cosmic_ray_amp_e: float = 5e4,
    cosmic_ray_rate_per_m2_per_s: Optional[float] = None,
    bayer_pattern: Optional[str] = None,
    bayer_qe: Tuple[float, float, float] = (0.40, 0.55, 0.20),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate detection of a coherent field on a pixel array.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Input field (amplitude units: sqrt(photons / m^2 / s) if you
        want absolute photon counts, or arbitrary if you just want
        relative noise modelling).
    dx_field : float
        Field grid spacing [m].
    pixel_pitch : float
        Detector pixel pitch [m].  Determines the detector resolution.
    n_pixels : int, optional
        Number of detector pixels across.  Default: covers the field
        extent.
    exposure_time : float, default 1.0
        Integration time [s].
    quantum_efficiency : float, default 1.0
        Fraction of incident photons detected (0 to 1).  When
        ``bayer_pattern`` is set this is the *global* QE before the
        Bayer-cell weight is applied.
    read_noise_e : float, default 0
        Gaussian read noise [electrons RMS] per pixel.
    dark_current_e_per_s : float, default 0
        Dark-current rate [electrons/pixel/s].
    full_well : float, default inf
        Saturation level [electrons].  Pixels above this are clipped.
    seed : int, optional
        Random seed for reproducible noise.
    hot_pixel_map : ndarray (bool, n_pixels x n_pixels), optional *(4.0+)*
        Boolean map of hot pixels: ``True`` pixels are saturated to
        ``full_well`` regardless of incident signal.  Useful for
        modelling a known defect map from detector characterisation.
    cosmic_ray_rate : float, default 0  *(4.0+; deprecated 4.9)*
        Expected total cosmic-ray strikes for the whole exposure
        on the whole array, ignoring detector size and exposure
        time.  This is the historical (and physically wrong) form:
        a 4096 × 4096 sensor at 10 s should see ~160× the strikes
        of a 1024 × 1024 sensor at 1 s for the same camera, but
        this parameter doesn't scale.  Prefer
        ``cosmic_ray_rate_per_m2_per_s`` for physically-correct
        scaling.  Retained for back-compat.
    cosmic_ray_rate_per_m2_per_s : float, optional  *(4.9+)*
        Physically-correct cosmic-ray rate density [strikes per m²
        per second].  When provided, the expected number of strikes
        per exposure is computed as
        ``rate · (n_pixels · pixel_pitch)² · exposure_time``.  At
        sea level the typical secondary-cosmic-ray flux is ~1 /m²/s;
        at altitude / in space it scales upward (LEO ~ 10¹/m²/s,
        deep space ~ 10²/m²/s) -- pick the value appropriate to
        your detector environment.  Overrides ``cosmic_ray_rate``
        when both are passed.
    cosmic_ray_amp_e : float, default 5e4  *(4.0+)*
        Charge per cosmic-ray strike [electrons].
    bayer_pattern : ``None`` (default) or ``'RGGB'`` / ``'BGGR'`` / ``'GRBG'`` / ``'GBRG'``  *(4.0+)*
        If set, applies a 2 x 2 Bayer colour-filter array to the
        per-pixel QE.  The default ``None`` produces a monochromatic
        detector (matches pre-4.0 behaviour).  The Bayer cell uses
        ``bayer_qe = (qe_R, qe_G, qe_B)``.
    bayer_qe : (float, float, float), default ``(0.40, 0.55, 0.20)``
        Per-channel QE multipliers when ``bayer_pattern`` is set.
        Defaults are representative of a generic visible-light CMOS
        sensor; tune for specific hardware.

    Returns
    -------
    image : ndarray, float, shape (n_pixels, n_pixels)
        Detected image in electrons (or photon-equivalent if input
        field is normalised to photons).
    x_det, y_det : ndarray
        Detector pixel center coordinates [m].
    """
    Ny, Nx = E.shape
    I_field = np.abs(E) ** 2  # intensity [per m^2 if field is normalised]

    # Determine detector grid
    if n_pixels is None:
        n_pixels = max(1, int(Nx * dx_field / pixel_pitch))

    x_det = (np.arange(n_pixels) - n_pixels / 2) * pixel_pitch
    y_det = (np.arange(n_pixels) - n_pixels / 2) * pixel_pitch

    # ---- Area-weighted integration onto the detector grid --------------
    # The old approach used integer truncation of the per-field-sample
    # index into the detector pixel grid, which gave non-uniform per-
    # pixel sample counts when (pixel_pitch / dx_field) wasn't an exact
    # integer aligned with the grid.  That imbalance dominated the
    # Poisson statistics (std was 20x sqrt(mean)).
    #
    # Here we use scipy.ndimage.zoom to resample to the detector pitch
    # with proper anti-aliased integration, then multiply by dx_field^2
    # to turn the re-sampled intensity (per unit area) into a per-pixel
    # integrated signal.  For integer ratios this agrees with block-sum
    # reshape to machine precision; for non-integer ratios it
    # interpolates cleanly.
    from scipy.ndimage import zoom as _zoom
    # zoom_factor = (output_size / input_size) along each axis; we want
    # output_size = n_pixels along both axes.
    zoom_y = n_pixels / Ny
    zoom_x = n_pixels / Nx
    # order=1 (linear) is area-preserving when combined with the dx_field^2
    # weighting below; higher orders can ring and produce negatives.
    # grid_mode=True anchors cells to edges (matches the physical binning
    # contract), and prefilter=False avoids a spline prefilter that would
    # introduce negative-going lobes.
    try:
        resampled = _zoom(I_field, (zoom_y, zoom_x), order=1,
                          mode='constant', cval=0.0,
                          grid_mode=True, prefilter=False)
    except TypeError:
        # Older scipy without grid_mode; fall back to plain zoom which
        # is still much better than the integer-truncation approach.
        resampled = _zoom(I_field, (zoom_y, zoom_x), order=1,
                          mode='constant', cval=0.0, prefilter=False)
    # Guarantee exact output shape (zoom can be off-by-one on some
    # scipy versions when the zoom factor isn't a clean integer ratio).
    if resampled.shape != (n_pixels, n_pixels):
        out = np.zeros((n_pixels, n_pixels), dtype=np.float64)
        ny_c = min(n_pixels, resampled.shape[0])
        nx_c = min(n_pixels, resampled.shape[1])
        out[:ny_c, :nx_c] = resampled[:ny_c, :nx_c]
        resampled = out
    # The resampled array is intensity (per unit area on the detector
    # grid); multiplying by the detector pixel area converts to a per-
    # pixel integrated signal in the same (photons/m^2/s * m^2 * s) units
    # the old code produced, i.e. photons.  Note: pixel_pitch^2 is used
    # (not dx_field^2) because we integrate over the detector pixel, not
    # the field sample.
    image = resampled * pixel_pitch ** 2

    # Per-pixel QE map.  For a Bayer detector, the QE varies per-cell
    # on a 2x2 mosaic; otherwise QE is uniform.
    if bayer_pattern is None:
        qe_map = float(quantum_efficiency)
    else:
        valid_patterns = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        if bayer_pattern not in valid_patterns:
            raise ValueError(
                f"bayer_pattern must be one of {valid_patterns} or None; "
                f"got {bayer_pattern!r}.")
        qe_r, qe_g, qe_b = bayer_qe
        # 2x2 base cell.  Within each cell index (i, j):
        #   RGGB: [[R, G], [G, B]]
        cells = {
            'RGGB': np.array([[qe_r, qe_g], [qe_g, qe_b]], dtype=float),
            'BGGR': np.array([[qe_b, qe_g], [qe_g, qe_r]], dtype=float),
            'GRBG': np.array([[qe_g, qe_r], [qe_b, qe_g]], dtype=float),
            'GBRG': np.array([[qe_g, qe_b], [qe_r, qe_g]], dtype=float),
        }
        cell = cells[bayer_pattern] * float(quantum_efficiency)
        # Tile to full detector size.
        qe_map = np.tile(cell, (n_pixels // 2 + 1, n_pixels // 2 + 1))
        qe_map = qe_map[:n_pixels, :n_pixels]

    # Convert to photon counts
    signal_e = image * qe_map * exposure_time
    signal_e = signal_e + dark_current_e_per_s * exposure_time

    # Noise
    rng = np.random.default_rng(seed)
    if float(np.asarray(signal_e).max()) > 0:
        # Poisson shot noise
        signal_e = rng.poisson(np.maximum(signal_e, 0).astype(np.float64))
        signal_e = signal_e.astype(np.float64)
    if read_noise_e > 0:
        signal_e = signal_e + rng.normal(0, read_noise_e, signal_e.shape)

    # Cosmic-ray strikes: Poisson count of pixel-localised events.
    # 4.9 fix (audit #4.5): when ``cosmic_ray_rate_per_m2_per_s`` is
    # given, scale by detector area · exposure time -- the physically
    # correct way.  The legacy ``cosmic_ray_rate`` (which the audit
    # called out as not scaling with detector size or exposure) is
    # retained for back-compat but emits a deprecation warning when
    # used.
    effective_mean_strikes = 0.0
    if cosmic_ray_rate_per_m2_per_s is not None:
        area_m2 = (n_pixels * pixel_pitch) ** 2
        effective_mean_strikes = (
            float(cosmic_ray_rate_per_m2_per_s)
            * area_m2 * float(exposure_time)
        )
    elif cosmic_ray_rate > 0:
        import warnings
        warnings.warn(
            "simulate_detector_image: ``cosmic_ray_rate`` does not scale "
            "with detector size or exposure time (the audit's finding "
            "#4.5).  For physically-correct scaling pass "
            "``cosmic_ray_rate_per_m2_per_s`` instead (typical sea-level "
            "value ~ 1 /m²/s).  Legacy behaviour retained.",
            DeprecationWarning, stacklevel=2,
        )
        effective_mean_strikes = float(cosmic_ray_rate)
    if effective_mean_strikes > 0:
        n_strikes = rng.poisson(effective_mean_strikes)
        if n_strikes > 0:
            ys = rng.integers(0, n_pixels, size=n_strikes)
            xs = rng.integers(0, n_pixels, size=n_strikes)
            for yy, xx in zip(ys, xs):
                signal_e[yy, xx] += float(cosmic_ray_amp_e)

    # Hot pixels: saturated to full_well regardless of incident signal.
    if hot_pixel_map is not None:
        hp = np.asarray(hot_pixel_map, dtype=bool)
        if hp.shape != signal_e.shape:
            raise ValueError(
                f"hot_pixel_map shape {hp.shape} does not match "
                f"detector shape {signal_e.shape}.")
        if np.isfinite(full_well):
            signal_e = np.where(hp, full_well, signal_e)
        else:
            # full_well == inf and hot map specified: still flag them
            # with a finite spike rather than leaving as detected signal.
            signal_e = np.where(hp, 1e9, signal_e)

    # Full-well clipping
    signal_e = np.clip(signal_e, 0, full_well)

    return signal_e, x_det, y_det


def shack_hartmann(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    lenslet_pitch: float,
    lenslet_focal: float,
    n_lenslets: Optional[int] = None,
    detector_pixels_per_lenslet: int = 16,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Shack-Hartmann wavefront sensor.

    Divides the pupil into sub-apertures, propagates each to the
    lenslet focal plane, finds each sub-aperture's centroid, and
    reconstructs the wavefront slope map + integrated wavefront.

    Parameters
    ----------
    E : ndarray, complex, shape (N, N)
        Input field at the lenslet array plane.
    dx : float
        Field grid spacing [m].
    wavelength : float
    lenslet_pitch : float
        Sub-aperture pitch [m].
    lenslet_focal : float
        Lenslet focal length [m].
    n_lenslets : int, optional
        Number of lenslets across.  Default: auto from field extent.
    detector_pixels_per_lenslet : int, default 16
        Pixel density per sub-aperture on the detector (determines
        centroiding accuracy).
    seed : int, optional
        Random seed for noise (currently deterministic; reserved).

    Returns
    -------
    slopes_x, slopes_y : ndarray, shape (n_lenslets, n_lenslets)
        Measured wavefront slopes [rad] at each sub-aperture.
    wavefront : ndarray, shape (n_lenslets, n_lenslets)
        Reconstructed wavefront [m] via cumulative trapezoidal
        integration of the slopes.
    centroids_x, centroids_y : ndarray
        Raw centroid positions [m] at each sub-aperture.
    """
    N = E.shape[0]
    extent = N * dx
    if n_lenslets is None:
        n_lenslets = max(1, int(extent / lenslet_pitch))

    k0 = 2 * np.pi / wavelength
    slopes_x = np.zeros((n_lenslets, n_lenslets))
    slopes_y = np.zeros((n_lenslets, n_lenslets))
    centroids_x = np.zeros((n_lenslets, n_lenslets))
    centroids_y = np.zeros((n_lenslets, n_lenslets))

    # Sub-aperture size in pixels
    sa_pixels = int(round(lenslet_pitch / dx))
    if sa_pixels < 2:
        raise ValueError(
            f'lenslet_pitch ({lenslet_pitch*1e6:.1f} um) < 2*dx '
            f'({2*dx*1e6:.1f} um); increase grid resolution.')

    x0 = N // 2 - (n_lenslets * sa_pixels) // 2

    for iy in range(n_lenslets):
        for ix in range(n_lenslets):
            # Extract sub-aperture
            r0 = x0 + iy * sa_pixels
            c0 = x0 + ix * sa_pixels
            if r0 < 0 or r0 + sa_pixels > N or c0 < 0 or c0 + sa_pixels > N:
                continue
            E_sub = E[r0:r0 + sa_pixels, c0:c0 + sa_pixels].copy()
            # Apply lenslet focusing phase
            xsa = (np.arange(sa_pixels) - sa_pixels / 2) * dx
            Xsa, Ysa = np.meshgrid(xsa, xsa)
            E_sub = E_sub * np.exp(-1j * k0 * (Xsa ** 2 + Ysa ** 2)
                                     / (2 * lenslet_focal))
            # Propagate to focal plane
            E_focus = angular_spectrum_propagate(
                E_sub, lenslet_focal, wavelength, dx, bandlimit=True)
            I_focus = np.abs(E_focus) ** 2
            total = I_focus.sum()
            if total < 1e-30:
                continue
            # Centroid
            cx = float(np.sum(Xsa * I_focus) / total)
            cy = float(np.sum(Ysa * I_focus) / total)
            centroids_x[iy, ix] = cx
            centroids_y[iy, ix] = cy
            # Slope = centroid / focal_length [rad]
            slopes_x[iy, ix] = cx / lenslet_focal
            slopes_y[iy, ix] = cy / lenslet_focal

    # 4.10: Wavefront reconstruction
    # slopes_x / slopes_y are OPD gradients in radians-of-tilt (m / m).
    # cumsum(slopes) * lenslet_pitch is the cumulative OPD in METERS.
    # Pre-4.10 multiplied by wavelength/(2 pi) (a radians-to-meters
    # conversion) AFTER cumsum, producing units of m^2 (off by ~1e6 at
    # visible wavelengths).  Drop that conversion.
    #
    # Also: averaging two cumulative-row and cumulative-column integrals
    # is not a valid 2-D reconstruction (Southwell/Hudgin/Fried require
    # an actual least-squares solve).  Cross-coupled aberrations like
    # astigmatism mis-reconstruct.  Anchor both halves to the (0, 0)
    # corner so they share an origin, then average.  Documented as an
    # approximation; users wanting full 2-D recon should call
    # `slope_to_modal()` directly on the (slopes_x, slopes_y) pair.
    wf_x = np.cumsum(slopes_x, axis=1) * lenslet_pitch
    wf_y = np.cumsum(slopes_y, axis=0) * lenslet_pitch
    # Anchor to (0, 0) corner
    wf_x = wf_x - wf_x[0, 0]
    wf_y = wf_y - wf_y[0, 0]
    wavefront = 0.5 * (wf_x + wf_y)

    return slopes_x, slopes_y, wavefront, centroids_x, centroids_y
