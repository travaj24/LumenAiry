"""
PSF / OTF / MTF computation + spec-sheet metrics + resolution criteria.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

Contents:

* Fraunhofer PSF + OTF + MTF (:func:`compute_psf`, :func:`compute_otf`,
  :func:`compute_mtf`, :func:`mtf_radial`).
* Spec-sheet metrics (v4.14.0): encircled-energy curve / radius,
  :func:`mtf_cutoff`.
* Optical resolution metrics (v4.15.0): Rayleigh, Sparrow, FWHM.

See Also
--------
lumenairy.analysis.beam_stats : beam-shape statistics.
lumenairy.analysis.polychromatic : multi-wavelength wrappers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    'compute_psf',
    'compute_otf',
    'compute_mtf',
    'mtf_radial',
    'mtf_cutoff',
    'encircled_energy_curve',
    'encircled_energy_radius',
    'rayleigh_resolution',
    'sparrow_resolution',
    'fwhm_resolution',
]


# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup): see beam_stats.py.
from ..backend import array_namespace as _xp_of  # noqa: E402

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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble pupil failed downstream at ``pupil.ndim`` /
    # ``pupil.shape`` and produced an unhelpful TypeError /
    # ValueError instead of the canonical v4.16 message.  Input
    # kind: 'pupil' (the function consumes a 2-D pupil amplitude *
    # phase product and does a single Fraunhofer FT to the PSF
    # plane).  Note: ``input_kind='pupil'`` would be ideal once
    # Agent B's parameterised ``_check_2d_scalar_field`` lands;
    # the default form here is correct in the interim.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(pupil, 'compute_psf')
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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble psf failed downstream at ``xp.fft.fft2`` (which would
    # FFT along the last two axes of a 3-D stack -- silently wrong
    # output shape).  Input kind: 'psf' (a real-valued intensity
    # PSF; the helper still accepts it because the only invariant
    # checked is ``.ndim == 2`` plus the MCF rejection).  Routes
    # both failure modes to the canonical v4.16 message via the V6
    # walker.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(psf, 'compute_otf')
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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard at the entry point
    # so the canonical message names ``compute_mtf`` rather than
    # the inner ``compute_otf``.  The downstream ``compute_otf``
    # call would catch the same failure mode but with a less
    # informative call-site name.  Input kind: 'psf'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(psf, 'compute_mtf')
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
# Spec-sheet metrics (v4.14.0): encircled energy, MTF cutoff
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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-existing inline
    # ``if E.ndim != 2`` check (below) caught the 3-D / 1-D ensemble
    # case but did NOT catch a ``PartialCoherenceMCF`` input (no
    # ``.ndim`` attribute), which failed at the bare ``E.ndim``
    # access with ``AttributeError``.  Routes both MCF and 3-D /
    # 1-D ensembles to the canonical v4.16 message via the V6
    # walker.  Input kind: 'field' (or 'psf' if user passed an
    # intensity PSF -- the function detects both).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'encircled_energy_curve')
    from .beam_stats import beam_centroid
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


# ============================================================================
# v4.15.0 (C.3) -- Optical resolution metrics (Rayleigh / Sparrow / FWHM)
# ============================================================================

def _psf_1d_profile(
    psf: np.ndarray,
    dx: float,
    *,
    axis: str,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal: extract a 1-D profile through the peak of a 2-D PSF.

    Returns ``(r, profile)`` with ``r`` an offset axis in metres
    centred on the PSF peak.  ``axis='x'`` returns a row cut,
    ``axis='y'`` a column cut, ``axis='radial'`` an azimuthally
    averaged radial profile.

    When ``dy is None`` (default) the y-spacing is taken equal to
    ``dx`` (square pixels).  Passing ``dy != dx`` selects an
    anamorphic grid: the row / column cuts scale by their own axis
    spacing, and the radial profile bins by true Euclidean distance
    ``sqrt((dx*Δi)^2 + (dy*Δj)^2)`` rather than pixel index.
    """
    if not isinstance(psf, np.ndarray):
        psf_arr = np.asarray(psf)
    else:
        psf_arr = psf
    if psf_arr.ndim != 2:
        raise ValueError(
            f"PSF profile: expected 2-D PSF; got shape "
            f"{psf_arr.shape!r} (ndim={psf_arr.ndim}).")

    dx_f = float(dx)
    dy_f = float(dy) if dy is not None else dx_f

    Ny, Nx = psf_arr.shape
    # Locate the peak.  For radially-symmetric Airy-like PSFs the
    # peak lives near the grid centre; we locate it explicitly to
    # accommodate off-axis or shifted PSFs.
    peak_idx = int(np.argmax(psf_arr))
    py, px = divmod(peak_idx, Nx)

    if axis == 'x':
        r = (np.arange(Nx) - px) * dx_f
        profile = psf_arr[py, :].astype(np.float64)
        return r, profile
    if axis == 'y':
        r = (np.arange(Ny) - py) * dy_f
        profile = psf_arr[:, px].astype(np.float64)
        return r, profile
    if axis == 'radial':
        # Radial profile by binning true Euclidean distance
        # ``sqrt((dx*(i-px))^2 + (dy*(j-py))^2)`` so anamorphic grids
        # produce a metric-correct azimuthal average.  The bin grid is
        # chosen so the radial step matches the geometric mean of dx
        # and dy (so square grids reduce to the classical pixel-step
        # binning).
        y_idx, x_idx = np.indices(psf_arr.shape)
        rr = np.sqrt(((x_idx - px) * dx_f) ** 2 +
                     ((y_idx - py) * dy_f) ** 2)
        d_bin = float(np.sqrt(dx_f * dy_f))
        r_int = np.rint(rr / d_bin).astype(int)
        tbin = np.bincount(r_int.ravel(),
                            weights=psf_arr.ravel().astype(np.float64))
        nbin = np.bincount(r_int.ravel())
        # Trim trailing all-empty bins
        radial_profile = np.where(
            nbin > 0, tbin / np.maximum(nbin, 1), 0.0)
        r = np.arange(radial_profile.size, dtype=np.float64) * d_bin
        return r, radial_profile

    raise ValueError(
        f"PSF profile: axis must be 'x', 'y', or 'radial'; got "
        f"{axis!r}.")


def _to_numpy_host(arr) -> np.ndarray:
    """Internal: coerce a backend-array (numpy / cupy / jax / etc.)
    to a host numpy array via an explicit dispatch.

    Used by the resolution-metric functions (``rayleigh_resolution``,
    ``sparrow_resolution``, ``fwhm_resolution``) which take a host-
    side scalar exit so a single host transfer is acceptable.  CuPy
    arrays raise on the implicit ``np.asarray`` path and must be
    pulled with ``.get()``; JAX arrays support the ``__array__``
    protocol.
    """
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, 'get') and callable(getattr(arr, 'get')):
        return arr.get()
    return np.asarray(arr)


def _radial_profile_subpixel(
    psf_arr: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Internal: high-fidelity azimuthally-averaged radial profile.

    Uses ``scipy.ndimage.map_coordinates`` to sample the 2-D PSF on a
    fine polar grid, then azimuthally averages.  This avoids the bias
    inherent in integer-pixel-bin radial averaging at small r and
    delivers a smooth profile suitable for spline / curvature root-
    finding.  Used by ``sparrow_resolution(axis='radial')``.

    Returns ``(r, profile)`` with ``r >= 0`` and a fine radial step.
    """
    from scipy.ndimage import map_coordinates

    Ny, Nx = psf_arr.shape
    peak_idx = int(np.argmax(psf_arr))
    py, px = divmod(peak_idx, Nx)
    # Maximum integer-pixel radius that stays inside the array.
    r_max_pixels = float(min(py, Ny - 1 - py, px, Nx - 1 - px))
    if r_max_pixels < 4.0:
        # Insufficient grid for radial averaging.
        return (np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))
    # Sub-pixel radial step (4 samples per pixel) and 64 azimuthal
    # samples; both balance accuracy vs cost.
    n_r = int(r_max_pixels * 4)
    r_pixels = np.linspace(0.0, r_max_pixels, n_r + 1)
    n_phi = 64
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    # Anamorphic conversion: the sample radius r is in metres on the
    # canonical isotropic axis ``r_metric = pixel * sqrt(dx*dy)``,
    # while map_coordinates uses pixel-index coordinates.
    xs_all = px + r_pixels[:, None] * np.cos(phi)[None, :]
    ys_all = py + r_pixels[:, None] * np.sin(phi)[None, :]
    samples_all = map_coordinates(
        psf_arr.astype(np.float64),
        [ys_all.ravel(), xs_all.ravel()],
        order=3, mode='constant', cval=0.0).reshape(n_r + 1, n_phi)
    radial = samples_all.mean(axis=1)
    # Convert the integer-pixel radii to a metric distance using the
    # isotropic-average pixel size (sqrt(dx*dy)).  For square grids
    # this reduces to ``r = i * dx``.
    d_bin = float(np.sqrt(float(dx) * float(dy)))
    r = r_pixels * d_bin
    return r.astype(np.float64), radial.astype(np.float64)


def rayleigh_resolution(
    psf: np.ndarray,
    dx: float,
    wavelength: float,
    *,
    axis: str = 'radial',
    dy: Optional[float] = None,
) -> float:
    """Rayleigh diffraction-limit resolution from a 2-D PSF.

    The Rayleigh criterion places two point sources at the separation
    where one source's principal maximum coincides with the other's
    first dark ring.  For an Airy pattern from a circular aperture of
    focal ratio ``f/#``, this is the canonical
    ``1.22 * wavelength * f_number`` separation.

    This implementation computes the first zero of the PSF profile
    (along ``axis``) past the peak and returns that distance.  For a
    perfect Airy pattern the radial first zero is the Rayleigh
    separation directly; for asymmetric / aberrated PSFs the axis cut
    captures the criterion along that line.

    Parameters
    ----------
    psf : ndarray (real, 2-D)
        Intensity PSF.
    dx : float
        PSF-plane grid spacing in x [m].
    wavelength : float
        Wavelength [m].  Currently used only as a numerical anchor
        for the small-separation tolerance; the first-zero search
        does not require it explicitly.
    axis : ``'radial'`` (default) | ``'x'`` | ``'y'``
        Profile axis to scan for the first zero.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid).
        For anamorphic grids (``dy != dx``) the radial profile uses
        true Euclidean distance ``sqrt((dx*Δi)^2 + (dy*Δj)^2)``.

    Returns
    -------
    d_rayleigh : float
        Rayleigh resolution [m].  ``NaN`` if no zero can be located
        (e.g. flat / zero-intensity input, or a Gaussian-like PSF
        with no true first-ring minimum -- a ``RuntimeWarning`` is
        emitted in the latter case directing the user to
        :func:`fwhm_resolution` or :func:`sparrow_resolution`).

    See Also
    --------
    sparrow_resolution : Sparrow dip-vanishing criterion.
    fwhm_resolution : FWHM-doubled resolution.

    Notes
    -----
    Convention: the returned value is the first-zero radius, which
    equals the *separation* between two adjacent diffraction-limited
    sources at the Rayleigh limit (the two PSFs touch at the first
    zero).  Some texts report the diameter (twice this value); we
    pin the radius form because it matches the standard
    ``1.22 lambda f/#`` formula directly.

    The first-zero search requires a *true* local minimum (strict
    inequality on at least one side).  Gaussian-like PSFs whose
    radial profile decreases monotonically into the noise / underflow
    floor without crossing a true minimum return ``NaN`` and emit a
    ``RuntimeWarning`` -- the Rayleigh criterion is not defined for
    PSFs without a first-ring zero.
    """
    import warnings
    xp = _xp_of(psf)
    # Coerce backend arrays to numpy for the host-side scan; the
    # resolution metric is a scalar so a single host transfer is fine.
    psf_np = _to_numpy_host(psf)
    del xp  # consumed for dispatch-trace; no array math beyond here
    if not np.isfinite(float(wavelength)) or float(wavelength) <= 0.0:
        raise ValueError(
            f"rayleigh_resolution: wavelength must be positive and "
            f"finite; got {wavelength!r}.")
    r, profile = _psf_1d_profile(psf_np, dx, axis=axis, dy=dy)

    # Sanity check the profile -- a flat / zero PSF has no resolvable
    # first zero.
    peak = float(profile.max())
    if peak <= 0.0 or not np.all(np.isfinite(profile)):
        return float('nan')

    # Locate the peak index on the profile (radial profile starts at
    # r=0; x/y profiles centre the peak at r=0 by construction in
    # _psf_1d_profile).  Walk outward until the profile reaches a
    # threshold-defined "zero".  We use a small fraction of the peak
    # so noise / discretisation does not spawn spurious zeros.
    profile_norm = profile / peak
    # Use a threshold proportional to peak; first-zero of a clean
    # Airy is identically zero, but numerical PSFs from compute_psf
    # have ~1e-3..1e-4 floor at the first zero.  Hunt for the first
    # true local minimum below 5% of peak: strict inequality is
    # required on at least one side so a monotonically-decreasing-to-
    # underflow Gaussian profile does NOT spawn a false minimum at
    # the floor-of-zeros plateau (audit V4.15.0 P1-F1-4).
    if axis == 'radial':
        peak_idx = 0
        scan_range = range(1, profile.size - 1)
    else:
        # x / y profile is centred on the peak; scan to the right.
        peak_idx = int(np.argmax(profile))
        scan_range = range(peak_idx + 1, profile.size - 1)

    # First-zero must be a true minimum that is followed by a
    # secondary maximum (the first Airy ring).  A Gaussian PSF
    # decreases monotonically into the float-underflow floor without
    # ever turning back up; on its radial profile the "first sample
    # below 0.05" satisfies the trivial three-point inequality
    # ``a >= b <= c`` only because c == 0 == b on the underflow
    # plateau.  We REQUIRE a strict subsequent rise: somewhere in
    # the lookahead window the profile must exceed the candidate
    # minimum by an absolute margin scaled to the Airy first-ring
    # height (~1.75% of peak).  This rejects Gaussian-style monotone
    # decay (post_max == 0 for an underflow tail) and accepts true
    # Airy first rings.
    first_zero_idx: Optional[int] = None
    # A clean Airy first ring is ~1.75% of peak; require the post-
    # min lookahead to exceed the candidate by >= 0.5% of peak to
    # confirm an actual ring.  A monotonically-decreasing-to-zero
    # Gaussian profile fails this because the post-min lookahead is
    # all zeros (or float-denormal values orders of magnitude below
    # the candidate threshold).
    ring_margin = 5.0e-3  # 0.5% of normalised peak
    scan_list = list(scan_range)
    if scan_list:
        scan_start = scan_list[0]
    else:
        scan_start = 1
    for i in scan_list:
        if not (profile_norm[i] < 0.05
                 and profile_norm[i] <= profile_norm[i - 1]
                 and profile_norm[i] <= profile_norm[i + 1]):
            continue
        # Strict inequality on at least one side: equal-zero plateaus
        # have BOTH neighbours equal; a real first ring has the
        # incoming side strictly greater (curve was dropping).
        if not (profile_norm[i] < profile_norm[i - 1]
                 or profile_norm[i] < profile_norm[i + 1]):
            continue
        # Look ahead for a secondary rise: window is at least 6
        # samples or twice the distance from the scan start.
        window = max(int(2 * (i - scan_start) + 4), 6)
        end_look = min(i + window, profile.size)
        if end_look <= i + 1:
            continue
        post_max = float(np.max(profile_norm[i + 1:end_look]))
        if post_max > profile_norm[i] + ring_margin:
            first_zero_idx = i
            break

    if first_zero_idx is None:
        warnings.warn(
            "rayleigh_resolution: no true first-ring minimum located "
            "in the radial profile (the criterion is not defined for "
            "Gaussian-like PSFs without a true first zero).  Use "
            "fwhm_resolution or sparrow_resolution for PSFs without "
            "a Rayleigh first ring.",
            RuntimeWarning, stacklevel=2)
        return float('nan')

    # Sub-pixel refinement via parabolic interpolation around the
    # local minimum (3-point form).  Falls back to the integer index
    # if the parabola is degenerate.  The horizontal step is the
    # radial bin step (sqrt(dx*dy) for the radial profile, dx / dy
    # for the row / column cut).
    j = first_zero_idx
    a = profile[j - 1]
    b = profile[j]
    c = profile[j + 1]
    denom = (a - 2.0 * b + c)
    if denom == 0.0:
        # Linear fallback
        sub = 0.0
    else:
        sub = 0.5 * (a - c) / denom
    sub = max(-1.0, min(1.0, float(sub)))
    # Step between adjacent samples of ``r`` (constant by
    # construction in _psf_1d_profile).
    r_step = float(r[1] - r[0]) if r.size > 1 else float(dx)
    return float(abs(r[j] - r[peak_idx]) +
                 sub * r_step * (1.0 if r[j] >= r[peak_idx] else -1.0))


def sparrow_resolution(
    psf: np.ndarray,
    dx: float,
    *,
    axis: str = 'radial',
    dy: Optional[float] = None,
) -> float:
    r"""Canonical Sparrow resolution criterion from a 2-D PSF.

    The Sparrow criterion defines the two-point separation ``d`` at
    which the dip between two overlapping point-source PSFs just
    vanishes:

    .. math::
        \left.\frac{d^2}{dr^2}\left[I(r - d/2) + I(r + d/2)\right]
        \right|_{r=0} = 0

    For a radially symmetric :math:`I(r)`, even-symmetry of
    :math:`I''` reduces this to :math:`I''(d/2) = 0`, i.e. ``d/2`` is
    the first inflection point of the single-source intensity
    profile.  For an Airy pattern this evaluates to
    ``d_sparrow ~= 0.947 * lambda * f/#`` -- slightly smaller than the
    Rayleigh separation (``1.22 * lambda * f/#``).

    Accuracy (v4.15.2): on a properly-sampled analytical Airy PSF
    (N=256, dx well below the first-zero radius) the canonical
    constant is recovered to **<1%** relative error (measured 0.02%
    on the canonical lambda=600 nm, f/#=4 fixture in
    ``test_v4_15_1_agent_c::test_sparrow_resolution_airy_analytical``;
    the test pin uses 1% as the tolerance with comfortable headroom).
    Undersampled or aberrated PSFs degrade this accuracy; consider
    :func:`fwhm_resolution` for noisy / Gaussian-tail PSFs.

    Implementation
    --------------
    The radial profile is built either from a row / column cut
    (``axis='x'`` / ``'y'``) or from a sub-pixel azimuthally-averaged
    polar resample (``axis='radial'``, via
    :func:`scipy.ndimage.map_coordinates`).  A natural-boundary cubic
    spline is fit to the profile, its analytical second derivative is
    evaluated, and :func:`scipy.optimize.brentq` brackets the first
    sign change of :math:`I''` in :math:`r \in (dx/2,\, N\,dx/2)`.
    The returned value is twice that root.

    Parameters
    ----------
    psf : ndarray (real, 2-D)
        Intensity PSF.
    dx : float
        PSF-plane grid spacing in x [m].
    axis : ``'radial'`` (default) | ``'x'`` | ``'y'``
        Profile axis.  ``'radial'`` uses the sub-pixel azimuthal
        average; ``'x'`` / ``'y'`` use pixel-aligned cuts through the
        peak.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid).
        Anamorphic grids (``dy != dx``) use the metric average
        ``sqrt(dx*dy)`` for the radial step.

    Returns
    -------
    d_sparrow : float
        Sparrow resolution [m].  ``NaN`` if no valid first inflection
        can be located (e.g. flat / zero-intensity input or a
        severely undersampled PSF).
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq

    xp = _xp_of(psf)
    psf_np = _to_numpy_host(psf)
    del xp  # backend probe; metric is host-side

    if psf_np.ndim != 2:
        raise ValueError(
            f"sparrow_resolution: expected 2-D PSF; got shape "
            f"{psf_np.shape!r} (ndim={psf_np.ndim}).")
    dx_f = float(dx)
    dy_f = float(dy) if dy is not None else dx_f

    if axis == 'radial':
        r_arr, prof = _radial_profile_subpixel(psf_np, dx_f, dy_f)
        if r_arr.size == 0:
            return float('nan')
    elif axis in ('x', 'y'):
        r_full, prof_full = _psf_1d_profile(
            psf_np, dx_f, axis=axis, dy=dy_f)
        peak_idx = int(np.argmax(prof_full))
        # Right-of-peak half (r >= 0).
        sel = np.arange(prof_full.size) >= peak_idx
        r_arr = (r_full[sel] - r_full[peak_idx]).astype(np.float64)
        prof = prof_full[sel].astype(np.float64)
    else:
        raise ValueError(
            f"sparrow_resolution: axis must be 'x', 'y', or "
            f"'radial'; got {axis!r}.")

    peak = float(prof.max())
    if peak <= 0.0 or not np.all(np.isfinite(prof)):
        return float('nan')
    prof_n = (prof / peak).astype(np.float64)

    if r_arr.size < 4:
        return float('nan')

    # Cubic-spline interpolant with analytical second derivative.
    try:
        cs = CubicSpline(r_arr, prof_n, bc_type='natural',
                          extrapolate=False)
    except Exception:
        return float('nan')

    def _ipp(rr: float) -> float:
        return float(cs(rr, 2))

    # Bracket: just past the peak (dx/2) out to half the array span
    # so the brentq search stays inside the well-sampled region.
    r_min = max(0.5 * dx_f, 0.5 * float(r_arr[1] - r_arr[0]))
    r_max = float(r_arr[-1]) * 0.5
    if not (r_max > r_min):
        return float('nan')
    n_scan = 200
    rs = np.linspace(r_min, r_max, n_scan)
    vals = np.array([_ipp(rr) for rr in rs], dtype=np.float64)

    half_d: Optional[float] = None
    for i in range(len(rs) - 1):
        if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        # First crossing from concave-down (I''<0 near peak) to
        # concave-up (I''>=0 in the wings).
        if vals[i] < 0.0 <= vals[i + 1]:
            try:
                half_d = float(brentq(_ipp, rs[i], rs[i + 1]))
                break
            except Exception:
                continue

    if half_d is None or not np.isfinite(half_d):
        return float('nan')
    return float(2.0 * half_d)


def fwhm_resolution(
    psf: np.ndarray,
    dx: float,
    *,
    axis: str = 'radial',
    dy: Optional[float] = None,
) -> float:
    """Twice the full-width-at-half-maximum half-radius of the central
    peak.

    The FWHM measurement is the standard rule-of-thumb resolution for
    PSFs whose first zero is poorly defined (Gaussian beams, heavily
    aberrated PSFs).  We return ``2 * r_half`` where ``r_half`` is
    the distance from the peak to where the profile crosses half-max
    on the outward side.

    Parameters
    ----------
    psf : ndarray (real, 2-D)
        Intensity PSF.
    dx : float
        PSF-plane grid spacing in x [m].
    axis : ``'radial'`` (default) | ``'x'`` | ``'y'``
        Profile axis to scan.
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx`` (square grid).

    Returns
    -------
    d_fwhm : float
        Twice the half-radius at half-max [m].  ``NaN`` if the
        half-max crossing cannot be located.

    Notes
    -----
    For a radial profile the half-radius is the smallest r > 0 with
    ``profile(r) <= 0.5 * profile.max()``.  Linear interpolation
    across the crossing gives sub-pixel accuracy.
    """
    xp = _xp_of(psf)
    psf_np = _to_numpy_host(psf)
    del xp
    r, profile = _psf_1d_profile(psf_np, dx, axis=axis, dy=dy)

    peak = float(profile.max())
    if peak <= 0.0 or not np.all(np.isfinite(profile)):
        return float('nan')

    half = 0.5 * peak

    if axis == 'radial':
        peak_idx = 0
        scan = range(1, profile.size)
    else:
        peak_idx = int(np.argmax(profile))
        scan = range(peak_idx + 1, profile.size)

    crossing_idx: Optional[int] = None
    for i in scan:
        if profile[i] <= half:
            crossing_idx = i
            break
    if crossing_idx is None:
        return float('nan')

    # Linear interpolation between the two samples that straddle the
    # half-max.
    j = crossing_idx
    if j == 0:
        return float('nan')
    y_lo = profile[j - 1]
    y_hi = profile[j]
    if y_lo == y_hi:
        t = 0.0
    else:
        t = (half - y_lo) / (y_hi - y_lo)
    r_step = float(r[1] - r[0]) if r.size > 1 else float(dx)
    r_half = abs(r[j - 1] - r[peak_idx]) + t * r_step
    return float(2.0 * r_half)
