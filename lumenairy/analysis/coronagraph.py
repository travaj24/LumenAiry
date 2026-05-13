"""
lumenairy.analysis.coronagraph -- coronagraph performance analysis.

Post-processing analysis for coronagraphic systems: the
``contrast_curve`` style radial reduction comparing a coronagraphed
PSF to its un-coronagraphed reference.

The element-side coronagraph builders (Lyot focal-plane mask, vortex
phase mask, Lyot stop, apodized pupil) live in
:mod:`lumenairy.elements`.  This module is the analysis counterpart.

Moved here in 4.3.0 from ``lumenairy.elements.elements``;
``lumenairy.coronagraph_contrast_curve`` and the
``elements`` re-export path continue to work.
"""

from __future__ import annotations

import numpy as np


def coronagraph_contrast_curve(psf_coro, psf_ref, dx_focal, wavelength,
                                  f_eff, *, n_radii=64, max_lam_over_D=20.0,
                                  center=None, azimuthal='mean'):
    """Compute the post-coronagraph contrast curve vs angular separation.

    Given a coronagraphed PSF (e.g. the output of the
    apply_aperture -> Fraunhofer-MFT -> vortex / Lyot-FPM -> back-MFT
    -> Lyot stop -> compute_psf pipeline) and the un-coronagraphed
    reference PSF of the same pupil, returns the **radial contrast**
    -- the local mean coronagraphed intensity divided by the
    reference peak intensity -- as a function of angular separation
    in units of ``lambda * f / D``.

    Parameters
    ----------
    psf_coro : ndarray (real, N x N)
        Intensity PSF of the coronagraphed system at the final
        detector plane.
    psf_ref : ndarray (real, N x N)
        Intensity PSF of the same pupil WITHOUT the coronagraph
        masks (used to set the contrast denominator at the on-axis
        peak).  Both PSFs should be computed with the same
        normalisation (typically ``normalize='none'`` so the
        intensities are directly comparable).
    dx_focal : float
        Focal-plane grid spacing [m].
    wavelength : float
        Operating wavelength [m].
    f_eff : float
        Effective focal length [m].  Together with the pupil
        diameter (implicit in ``psf_ref``) this defines the
        lambda*f/D scale.  The function does NOT depend on the
        pupil diameter directly -- it normalises by the chief peak
        of ``psf_ref`` so the angular scale comes from
        ``lambda * f_eff / D = dx_focal * (N * dx_pupil / f_eff)``
        through the Fraunhofer relation; pass ``f_eff`` to set the
        x-axis units.
    n_radii : int, default 64
        Number of radial bins.
    max_lam_over_D : float, default 20.0
        Outer radius of the contrast curve in units of
        ``lambda * f_eff / D``.  Past this point the bins overflow
        the grid for typical setups.
    center : tuple ``(xc_pix, yc_pix)``, optional
        Pixel coordinates of the coronagraphic chief.  Defaults to
        the brightest pixel of ``psf_ref`` (the un-blocked chief).
    azimuthal : ``'mean'`` (default) / ``'median'`` / ``'rms'``
        Per-radius reduction over the azimuth.  ``'mean'`` is the
        textbook "average contrast curve"; ``'median'`` is robust
        against bright residual speckles; ``'rms'`` reports
        ``sqrt(mean(I^2))`` which is the standard 1-sigma
        speckle-noise floor metric.

    Returns
    -------
    result : dict with keys
        * ``'r_lam_over_D'`` -- ``(n_radii,)`` array of radial bin
          centres in units of ``lambda * f_eff / D``.
        * ``'contrast'`` -- ``(n_radii,)`` array of radial contrast
          values.
        * ``'r_pixels'`` -- raw radial bin centres in pixels.
        * ``'peak_ref'`` -- the reference peak intensity used as
          denominator.
        * ``'center'`` -- pixel coordinates of the chief.

    Notes
    -----
    For a vortex coronagraph with a 0.85 * D Lyot stop the curve
    should drop by 3-4 orders of magnitude inside ``~3 lambda/D``
    and asymptote to the residual-speckle floor outside that.

    Examples
    --------
    >>> import lumenairy as la
    >>> # Build coronagraphed and reference PSFs (see
    >>> # Function-Reference-Coronagraphs for the pipeline)
    >>> psf_coro, dx = la.compute_psf(E_after_coro, wl, f, dx_pupil,
    ...                                  normalize='none')
    >>> psf_ref, _   = la.compute_psf(E_pupil_clean, wl, f, dx_pupil,
    ...                                  normalize='none')
    >>> curve = la.coronagraph_contrast_curve(
    ...     psf_coro, psf_ref, dx, wl, f)
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(curve['r_lam_over_D'], curve['contrast'])
    """
    psf_coro = np.asarray(psf_coro, dtype=float)
    psf_ref = np.asarray(psf_ref, dtype=float)
    if psf_coro.shape != psf_ref.shape:
        raise ValueError(
            f"coronagraph_contrast_curve: coro {psf_coro.shape} "
            f"and ref {psf_ref.shape} must have the same shape.")

    Ny, Nx = psf_ref.shape
    if center is None:
        idx = int(np.argmax(psf_ref))
        cy_pix = idx // Nx
        cx_pix = idx % Nx
    else:
        cx_pix, cy_pix = center

    peak_ref = float(psf_ref.max())
    if peak_ref <= 0:
        raise ValueError("psf_ref has no positive intensity.")

    # For the natural-FFT-pitch grid 1 lambda/D = N pixels.  Expose
    # r_pixels too so users with a non-FFT pupil geometry can rescale.
    pix_per_lam_over_D = float(Nx)

    y_grid = np.arange(Ny) - cy_pix
    x_grid = np.arange(Nx) - cx_pix
    YY, XX = np.meshgrid(y_grid, x_grid, indexing='ij')
    r_pix = np.sqrt(XX ** 2 + YY ** 2)

    r_max_pix = max_lam_over_D * pix_per_lam_over_D
    r_max_pix = min(r_max_pix, min(Nx, Ny) / 2 - 1)

    edges = np.linspace(0.0, r_max_pix, n_radii + 1)
    centres_pix = 0.5 * (edges[:-1] + edges[1:])
    contrast = np.full(n_radii, np.nan, dtype=float)
    for i in range(n_radii):
        m = (r_pix >= edges[i]) & (r_pix < edges[i + 1])
        if not m.any():
            continue
        vals = psf_coro[m]
        if azimuthal == 'mean':
            agg = float(np.mean(vals))
        elif azimuthal == 'median':
            agg = float(np.median(vals))
        elif azimuthal == 'rms':
            agg = float(np.sqrt(np.mean(vals ** 2)))
        else:
            raise ValueError(
                f"azimuthal must be 'mean'/'median'/'rms'; got {azimuthal!r}")
        contrast[i] = agg / peak_ref

    return {
        'r_lam_over_D': centres_pix / pix_per_lam_over_D,
        'contrast': contrast,
        'r_pixels': centres_pix,
        'peak_ref': peak_ref,
        'center': (cx_pix, cy_pix),
    }


__all__ = ['coronagraph_contrast_curve']
