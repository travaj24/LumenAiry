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
    # 4.10: proper area integration of the intensity field onto the
    # detector grid.  Pre-4.10 used scipy.ndimage.zoom(order=1), which
    # is BILINEAR INTERPOLATION (point-sample at the new pixel
    # centers), NOT area integration.  Multiplying that by pixel_pitch^2
    # is dimensionally pixel_pitch^2 * intensity, NOT
    # integral_over_pixel(intensity) * dx_field^2 -- so photon
    # conservation fails for non-integer pixel_pitch/dx_field ratios
    # and shot-noise calibration loses meaning.
    #
    # For integer ratios use block-sum via np.add.reduceat.  For
    # non-integer ratios first uniform-filter to anti-alias, then
    # sample at the new pixel centers, scaled by pixel_pitch^2 so the
    # integral over each detector pixel is correctly represented.
    px_per_pixel_y = float(pixel_pitch / (Ny / n_pixels * dx_field)) if Ny > 0 else 0
    # Simpler: per-detector-pixel area in field samples.
    samples_per_pix_y = (Ny / n_pixels) if n_pixels > 0 else 1.0
    samples_per_pix_x = (Nx / n_pixels) if n_pixels > 0 else 1.0
    if (abs(samples_per_pix_y - round(samples_per_pix_y)) < 1e-9
            and abs(samples_per_pix_x - round(samples_per_pix_x)) < 1e-9):
        # Integer ratio: block-sum is exact area integration.
        spy = int(round(samples_per_pix_y))
        spx = int(round(samples_per_pix_x))
        # Use np.add.reduceat for the 2-D block-sum.  Trim trailing
        # samples that don't fit a full block, then reduceat with the
        # block-start indices.
        Ny_trim = (Ny // spy) * spy
        Nx_trim = (Nx // spx) * spx
        I_trim = I_field[:Ny_trim, :Nx_trim]
        starts_y = np.arange(0, Ny_trim, spy)
        starts_x = np.arange(0, Nx_trim, spx)
        # add.reduceat sums each segment.
        block = np.add.reduceat(
            np.add.reduceat(I_trim, starts_y, axis=0),
            starts_x, axis=1)
        image = np.zeros((n_pixels, n_pixels), dtype=np.float64)
        block_h = min(block.shape[0], n_pixels)
        block_w = min(block.shape[1], n_pixels)
        image[:block_h, :block_w] = block[:block_h, :block_w]
        image = image * (dx_field ** 2)
    else:
        # Non-integer ratio: uniform-filter then point-sample at the
        # detector pixel centers.  uniform_filter computes the mean
        # over a (2*radius+1) window; multiply by window area to get
        # the integral over the detector-pixel-sized neighbourhood,
        # which approximates the area integral up to the
        # uniform-filter's box-filter response.
        from scipy.ndimage import uniform_filter as _ufilt
        win_y = max(1, int(round(samples_per_pix_y)))
        win_x = max(1, int(round(samples_per_pix_x)))
        avg = _ufilt(I_field, size=(win_y, win_x), mode='constant', cval=0.0)
        # Sample at detector-pixel centres in field-grid coords.
        # Field-grid pixel j corresponds to physical x = (j - Nx/2) * dx_field.
        # Detector pixel k centre is at  (k - n_pixels/2 + 0.5) * pixel_pitch.
        # Convert detector centre to a field-grid float index:
        k = np.arange(n_pixels)
        det_centres_x = (k - n_pixels / 2 + 0.5) * pixel_pitch
        det_centres_y = (k - n_pixels / 2 + 0.5) * pixel_pitch
        idx_fx = det_centres_x / dx_field + Nx / 2 - 0.5
        idx_fy = det_centres_y / dx_field + Ny / 2 - 0.5
        ix0 = np.clip(np.floor(idx_fx).astype(int), 0, Nx - 1)
        iy0 = np.clip(np.floor(idx_fy).astype(int), 0, Ny - 1)
        IX, IY = np.meshgrid(ix0, iy0)
        # 4.11.2: scale by the actual fractional samples-per-pixel
        # rather than the integer-rounded window size.  ``uniform_filter``
        # with ``size=(win_y, win_x)`` returns the box-mean over a
        # (win_y * dx_field) x (win_x * dx_field) field-grid area.
        # Multiplying that by ``pixel_pitch**2`` gives the right
        # answer when win_{y,x} == samples_per_pix_{y,x} exactly --
        # but for a ratio like 2.5 the rounded window (3) over-samples
        # the area by 20% and the result over-/undercounts photons
        # proportionally.  The correct integration scale for "intensity
        # per m² integrated over one detector pixel" is the actual
        # detector-pixel area (``pixel_pitch**2``), not the window area,
        # so the dominant correction is to keep the pixel_pitch**2
        # factor but re-normalise the box mean back to the true
        # samples-per-pixel.  Algebraically: avg * (samples/win) =
        # mean over the true detector-pixel-sized region.  Note this is
        # still a box-filter approximation -- for sub-pixel-accurate
        # cases the caller should resample on a grid with an integer
        # samples-per-pixel ratio.
        scale_y = float(samples_per_pix_y) / float(win_y)
        scale_x = float(samples_per_pix_x) / float(win_x)
        image = avg[IY, IX] * (pixel_pitch ** 2) * (scale_y * scale_x)

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
    # 4.10: initialise slopes / centroids to NaN.  Pre-4.10 left
    # out-of-bounds sub-apertures at 0, which propagated through the
    # cumulative-sum wavefront reconstruction as if real measurements.
    # NaN sentinels make OOB lenslets visible and inert in downstream
    # least-squares solvers (Hudgin / Southwell) via np.isfinite() masks.
    slopes_x = np.full((n_lenslets, n_lenslets), np.nan)
    slopes_y = np.full((n_lenslets, n_lenslets), np.nan)
    centroids_x = np.full((n_lenslets, n_lenslets), np.nan)
    centroids_y = np.full((n_lenslets, n_lenslets), np.nan)

    # Sub-aperture size in pixels
    sa_pixels = int(round(lenslet_pitch / dx))
    if sa_pixels < 2:
        raise ValueError(
            f'lenslet_pitch ({lenslet_pitch*1e6:.1f} um) < 2*dx '
            f'({2*dx*1e6:.1f} um); increase grid resolution.')

    x0 = N // 2 - (n_lenslets * sa_pixels) // 2

    # 4.10: reference-centroid pass on a flat-wavefront calibration field.
    # Pre-4.10 reported raw centroid / lenslet_focal as the slope,
    # baking in any per-lenslet centring bias from sa_pixels rounding
    # / x0 offset as a fake tilt in EVERY measurement.  Compute the
    # zero-slope reference centroids once from a unit-amplitude flat
    # field and subtract.
    E_flat = np.ones_like(E)
    ref_centroids_x = np.zeros((n_lenslets, n_lenslets))
    ref_centroids_y = np.zeros((n_lenslets, n_lenslets))
    for iy_r in range(n_lenslets):
        for ix_r in range(n_lenslets):
            r0r = x0 + iy_r * sa_pixels
            c0r = x0 + ix_r * sa_pixels
            if r0r < 0 or r0r + sa_pixels > N or c0r < 0 or c0r + sa_pixels > N:
                continue
            E_sub_r = E_flat[r0r:r0r + sa_pixels, c0r:c0r + sa_pixels].copy()
            xsa_r = (np.arange(sa_pixels) - sa_pixels / 2) * dx
            Xsa_r, Ysa_r = np.meshgrid(xsa_r, xsa_r)
            E_sub_r = E_sub_r * np.exp(-1j * k0 * (Xsa_r ** 2 + Ysa_r ** 2)
                                       / (2 * lenslet_focal))
            E_focus_r = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_sub_r)))
            I_focus_r = np.abs(E_focus_r) ** 2
            total_r = float(np.sum(I_focus_r))
            if total_r > 0:
                ref_centroids_x[iy_r, ix_r] = float(
                    np.sum(Xsa_r * I_focus_r) / total_r)
                ref_centroids_y[iy_r, ix_r] = float(
                    np.sum(Ysa_r * I_focus_r) / total_r)

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
            # Centroid (calibrated against the flat-wavefront reference
            # so per-lenslet centring biases from sa_pixels rounding /
            # x0 offset don't masquerade as tilts).
            cx = float(np.sum(Xsa * I_focus) / total) - ref_centroids_x[iy, ix]
            cy = float(np.sum(Ysa * I_focus) / total) - ref_centroids_y[iy, ix]
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
    # 4.10: NaN-mask OOB lenslets before cumsum so they zero-out
    # rather than NaN-poison the entire row / column of the integrator.
    sx_safe = np.where(np.isfinite(slopes_x), slopes_x, 0.0)
    sy_safe = np.where(np.isfinite(slopes_y), slopes_y, 0.0)
    wf_x = np.cumsum(sx_safe, axis=1) * lenslet_pitch
    wf_y = np.cumsum(sy_safe, axis=0) * lenslet_pitch
    # Anchor to (0, 0) corner
    wf_x = wf_x - wf_x[0, 0]
    wf_y = wf_y - wf_y[0, 0]
    wavefront = 0.5 * (wf_x + wf_y)

    return slopes_x, slopes_y, wavefront, centroids_x, centroids_y
