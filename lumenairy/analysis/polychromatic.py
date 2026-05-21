"""
Multi-wavelength / chromatic analysis wrappers.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

Contents:

* :func:`chromatic_focal_shift` -- per-wavelength EFL / BFL and axial
  colour.
* :func:`polychromatic_strehl` -- weighted Strehl across the band.
* :func:`polychromatic_psf` -- weighted-sum PSF on a common image
  plane.
* :func:`ee_polychromatic` -- encircled energy from the polychromatic
  PSF.
* :func:`radial_power_bands` -- cumulative-power sampler used by the
  through-focus / encircled-energy stack.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'chromatic_focal_shift',
    'polychromatic_strehl',
    'polychromatic_psf',
    'ee_polychromatic',
    'radial_power_bands',
]


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
    from ..raytrace import surfaces_from_prescription, system_abcd
    from .through_focus import diffraction_limited_peak, find_best_focus, through_focus_scan

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
# v4.15.0 (C.1) -- Polychromatic encircled-energy convenience wrapper
# ============================================================================

def ee_polychromatic(
    prescription: Dict[str, Any],
    wavelengths: Sequence[float],
    weights: Sequence[float],
    radii: Sequence[float],
    *,
    source: Optional[np.ndarray] = None,
    output_grid: Optional[int] = None,
    output_dx: Optional[float] = None,
    N: Optional[int] = None,
    dx: Optional[float] = None,
    E_in: Optional[np.ndarray] = None,
    image_distance: Optional[float] = None,
    bandlimit: bool = True,
    dy: Optional[float] = None,
    centroid: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Polychromatic encircled-energy curve at user-supplied radii.

    Convenience wrapper around :func:`polychromatic_psf` +
    :func:`encircled_energy_radius`.  Computes the weighted-sum
    polychromatic PSF on a common image plane, then evaluates the
    encircled-energy curve at each requested radius.

    Parameters
    ----------
    prescription : dict
        Lumenairy lens prescription.
    wavelengths : sequence of float
        Vacuum wavelengths [m].
    weights : sequence of float
        Per-wavelength spectral weights.  Re-normalised internally so
        they sum to 1.  Must satisfy ``sum(weights) > 0``.
    radii : sequence of float
        Strictly-increasing positive radii [m] at which to evaluate
        the encircled-energy curve.
    source : ndarray (complex, N x N), optional
        Pupil-plane input field.  Alias for ``E_in``; either may be
        supplied (but not both).  Defaults to a unit plane wave.
    output_grid : int, optional
        Output grid size.  Alias for ``N``; either may be supplied
        (but not both).  If ``source`` is provided, defaults to
        ``source.shape[0]``.
    output_dx : float, optional
        Output grid spacing [m].  Alias for ``dx``; either may be
        supplied (but not both).
    N : int, optional
        Output grid size.  See ``output_grid``.
    dx : float, optional
        Output grid spacing [m].  See ``output_dx``.
    E_in : ndarray (complex, N x N), optional
        Pupil-plane input field.  See ``source``.
    image_distance : float, optional
        Forwarded to :func:`polychromatic_psf`.  Defaults to the
        centroid-wavelength paraxial BFL.
    bandlimit : bool, default True
        Forwarded to :func:`polychromatic_psf`.
    dy : float, optional
        Forwarded to :func:`polychromatic_psf` / encircled-energy
        evaluation.  Defaults to ``dx``.
    centroid : (cx, cy), optional
        Centre of the encircled-energy circles [m].  Defaults to the
        intensity centroid of the accumulated PSF.

    Returns
    -------
    radii : ndarray, shape (n,)
        The (validated) radii in ascending order [m].
    ee : ndarray, shape (n,)
        Encircled-energy fraction at each radius in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``wavelengths`` and ``weights`` have different lengths, if
        the sum of ``weights`` is not strictly positive, if ``radii``
        is empty / non-monotonic / contains non-positive entries, or
        if neither ``source`` nor ``output_grid``/``output_dx`` is
        sufficient to specify the output grid.

    See Also
    --------
    polychromatic_psf : underlying polychromatic PSF accumulator.
    encircled_energy_radius : single-threshold inverse query.
    encircled_energy_curve : monochromatic encircled-energy curve.

    Examples
    --------
    >>> import numpy as np, lumenairy as la
    >>> rx = la.make_singlet(R1=50e-3, R2=float('inf'), d=2e-3,
    ...                      glass='N-BK7', aperture=10e-3)
    >>> wls = [1.30e-6, 1.55e-6]
    >>> wts = [0.5, 0.5]
    >>> radii = [5e-6, 10e-6, 20e-6, 50e-6]
    >>> r, ee = la.ee_polychromatic(rx, wls, wts, radii,
    ...                              output_grid=32, output_dx=5e-6)
    >>> bool(np.all(np.diff(ee) >= -1e-12))
    True
    """
    from .psf_mtf_otf import encircled_energy_curve

    # ---- Alias plumbing.  Accept either canonical or alias kwarg
    # (output_grid <-> N, output_dx <-> dx, source <-> E_in) but not
    # both.  Raise on collision so the caller gets a clear error
    # rather than a silent precedence rule.
    if N is not None and output_grid is not None:
        raise ValueError(
            "ee_polychromatic: pass either N or output_grid, not both.")
    if dx is not None and output_dx is not None:
        raise ValueError(
            "ee_polychromatic: pass either dx or output_dx, not both.")
    if E_in is not None and source is not None:
        raise ValueError(
            "ee_polychromatic: pass either E_in or source, not both.")
    if output_grid is not None:
        N = int(output_grid)
    if output_dx is not None:
        dx = float(output_dx)
    if source is not None:
        E_in = source

    # ---- Derive N from the source if neither N nor output_grid was
    # supplied.  dx has no source-derived fallback (the pixel pitch is
    # a physical scale, not encoded in the array shape).
    if N is None and E_in is not None:
        N = int(E_in.shape[0])
    if N is None:
        raise ValueError(
            "ee_polychromatic: must supply N (or output_grid), either "
            "directly or implicitly via source.shape[0].")
    if dx is None:
        raise ValueError(
            "ee_polychromatic: must supply dx (or output_dx); the pixel "
            "pitch is a physical scale and has no implicit fallback.")

    # ---- Validate wavelengths / weights.  Match polychromatic_psf's
    # error class (ValueError) so callers can catch both with one
    # except clause.
    wavelengths_arr = np.asarray(wavelengths, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    if wavelengths_arr.size != weights_arr.size:
        raise ValueError(
            f"ee_polychromatic: wavelengths and weights must have the "
            f"same length; got {wavelengths_arr.size} and "
            f"{weights_arr.size}.")
    if wavelengths_arr.size == 0:
        raise ValueError(
            "ee_polychromatic: need at least one wavelength.")
    w_sum = float(weights_arr.sum())
    if not (w_sum > 0.0):
        raise ValueError(
            f"ee_polychromatic: weights must sum to a strictly "
            f"positive value; got sum={w_sum!r}.")
    # polychromatic_psf re-normalises internally, but we re-normalise
    # here too so the user-visible weight error is raised here rather
    # than deep inside the propagator.
    weights_arr = weights_arr / w_sum

    # ---- Validate radii: non-empty, strictly increasing, > 0.
    radii_arr = np.asarray(radii, dtype=np.float64)
    if radii_arr.ndim != 1 or radii_arr.size == 0:
        raise ValueError(
            f"ee_polychromatic: radii must be a non-empty 1-D "
            f"sequence; got shape {radii_arr.shape!r}.")
    if not np.all(np.isfinite(radii_arr)):
        raise ValueError(
            "ee_polychromatic: radii must be finite.")
    if not np.all(radii_arr > 0.0):
        raise ValueError(
            f"ee_polychromatic: radii must be strictly positive; got "
            f"min={float(radii_arr.min())!r}.")
    if radii_arr.size >= 2 and not np.all(np.diff(radii_arr) > 0.0):
        raise ValueError(
            "ee_polychromatic: radii must be strictly increasing.")

    # ---- Compute the polychromatic PSF and the encircled-energy
    # curve at the requested radii.  The PSF is normalised to unit
    # power ('power' mode); the encircled-energy curve is in [0, 1]
    # so the absolute normalisation cancels.
    psf, dx_psf, _info = polychromatic_psf(
        prescription, wavelengths_arr, weights_arr,
        N=int(N), dx=float(dx),
        E_in=E_in,
        image_distance=image_distance,
        normalize='power',
        bandlimit=bandlimit,
        dy=dy,
    )

    # Pass the validated radii through.  The PSF is a real-valued
    # intensity array; encircled_energy_curve treats it as an
    # amplitude (which it then squares), so wrap as a complex with
    # zero phase by taking sqrt.  Equivalently, treat the PSF as
    # ``|E|^2`` directly: encircled_energy_curve does ``np.abs(E)**2``
    # internally, so feeding ``np.sqrt(psf)`` recovers the intended
    # cumulative-intensity curve without sign ambiguity.
    psf_amp = np.sqrt(np.asarray(psf, dtype=np.float64))
    r_out, ee = encircled_energy_curve(
        psf_amp.astype(np.complex128), dx_psf,
        dy=dy, radii=radii_arr, centroid=centroid)
    return r_out, ee
