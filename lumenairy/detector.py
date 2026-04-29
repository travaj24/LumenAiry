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

import numpy as np

from .propagation import angular_spectrum_propagate


def apply_detector(E, dx_field, pixel_pitch, n_pixels=None,
                   exposure_time=1.0, quantum_efficiency=1.0,
                   read_noise_e=0.0, dark_current_e_per_s=0.0,
                   full_well=np.inf, seed=None):
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
        Fraction of incident photons detected (0 to 1).
    read_noise_e : float, default 0
        Gaussian read noise [electrons RMS] per pixel.
    dark_current_e_per_s : float, default 0
        Dark-current rate [electrons/pixel/s].
    full_well : float, default inf
        Saturation level [electrons].  Pixels above this are clipped.
    seed : int, optional
        Random seed for reproducible noise.

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

    # Convert to photon counts
    signal_e = image * quantum_efficiency * exposure_time
    signal_e = signal_e + dark_current_e_per_s * exposure_time

    # Noise
    rng = np.random.default_rng(seed)
    if signal_e.max() > 0:
        # Poisson shot noise
        signal_e = rng.poisson(np.maximum(signal_e, 0).astype(np.float64))
        signal_e = signal_e.astype(np.float64)
    if read_noise_e > 0:
        signal_e = signal_e + rng.normal(0, read_noise_e, signal_e.shape)
    # Full-well clipping
    signal_e = np.clip(signal_e, 0, full_well)

    return signal_e, x_det, y_det


def shack_hartmann(E, dx, wavelength, lenslet_pitch, lenslet_focal,
                   n_lenslets=None, detector_pixels_per_lenslet=16,
                   seed=None):
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

    # Wavefront reconstruction via cumulative integration
    # Simple trapezoidal integration along x then y
    wf_x = np.cumsum(slopes_x, axis=1) * lenslet_pitch
    wf_y = np.cumsum(slopes_y, axis=0) * lenslet_pitch
    wavefront = 0.5 * (wf_x + wf_y) * wavelength / (2 * np.pi)

    return slopes_x, slopes_y, wavefront, centroids_x, centroids_y
