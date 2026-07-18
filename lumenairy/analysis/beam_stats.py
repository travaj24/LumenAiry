"""
Beam statistics: centroid, second-moment widths, integrated power, M^2.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

Backend awareness: dispatch through :func:`lumenairy.backend.array_namespace`
so CuPy / JAX inputs flow through without an implicit host transfer
(``M2``, which uses ``np.gradient`` / ``np.fft``, still coerces on
entry).

See Also
--------
lumenairy.analysis.strehl : Strehl / coupling-efficiency metrics.
lumenairy.analysis.psf_mtf_otf : PSF / MTF / OTF + spec-sheet metrics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

__all__ = [
    'beam_centroid',
    'beam_d4sigma',
    'beam_power',
    'beam_diameter',
    'M2',
]


# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup):  this used to be a
# 4-line wrapper duplicated in 5 files (elements/elements.py,
# elements/freeform.py, analysis/beam_stats.py, analysis/strehl.py,
# analysis/psf_mtf_otf.py).  Consolidated to the canonical backend
# helper; the underscore-prefixed alias preserves the existing
# in-module references without touching call sites.
from ..backend import array_namespace as _xp_of  # noqa: E402

# ISO 11146-1:2005 clause 9 recommends the second-moment integration
# aperture be about three times the beam width (D4sigma).  Exposed as the
# default multiple used by ``beam_d4sigma(..., aperture=True)``.
_ISO_APERTURE_WIDTHS = 3.0


def _subtract_background(xp, I, background):
    """ISO 11146 sec. 3.4 pedestal subtraction on an intensity map.

    ``background`` is either a float (subtract that constant level),
    the string ``'corner'`` (estimate the pedestal as the mean of the
    outer one-pixel border ring, assumed beam-free), or ``None``
    (no-op -- returned unchanged).  Negative residuals are clipped to
    zero so leftover noise cannot bias the moments downward.
    """
    if background is None:
        return I
    if isinstance(background, str):
        if background == 'corner':
            border = xp.concatenate([
                I[0, :], I[-1, :], I[1:-1, 0], I[1:-1, -1],
            ])
            level = xp.mean(border)
        else:
            raise ValueError(
                "background must be a float, 'corner', or None; got "
                f"{background!r}.")
    else:
        level = float(background)
    I = I - level
    return xp.where(I > 0, I, xp.zeros_like(I))


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
    *,
    background: Optional[Union[float, str]] = None,
    aperture: Optional[Union[bool, float]] = None,
    max_iter: int = 5,
    tol: float = 1e-3,
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
    background : float or {'corner'}, optional
        ISO 11146 sec. 3.4 pedestal subtraction (opt-in).  A float
        subtracts that constant intensity level; ``'corner'`` estimates
        the pedestal from the outer one-pixel border ring (assumed
        beam-free).  Negative residuals are clipped to zero.  Default
        ``None`` leaves the intensity untouched (historical behaviour).
    aperture : bool or float, optional
        ISO 11146 clause 9 iterative integration aperture (opt-in).
        ``True`` restricts the moment integral to a rectangle of full
        width ``3 * D4sigma`` centred on the running centroid; a float
        overrides the multiple (aperture full width = ``aperture *
        D4sigma``).  The centroid and widths are recomputed inside the
        aperture and the aperture re-sized until ``D4sigma`` converges
        (relative change < *tol*) or *max_iter* passes elapse.  Default
        ``None`` integrates over the whole grid (historical behaviour).
        The iteration is seeded from the whole-grid width, so on a
        *heavily* pedestal-inflated field the first aperture can already
        exceed the grid and fail to engage -- combine ``aperture`` with
        ``background`` (the ISO-recommended order) for such inputs.
    max_iter : int, default 5
        Maximum aperture-refinement passes (only used when
        ``aperture`` is set).
    tol : float, default 1e-3
        Relative-convergence tolerance on ``D4sigma`` for the aperture
        iteration.

    Returns
    -------
    d4s_x : float
        D4sigma beam diameter in x [m].
    d4s_y : float
        D4sigma beam diameter in y [m].

    Notes
    -----
    **Pedestal sensitivity.**  With no background subtraction and no
    integration aperture (the defaults), a uniform intensity pedestal
    biases the second moment *strongly*.  A flat pedestal of fractional
    peak level :math:`b` over an :math:`L \\times L` grid contributes a
    variance :math:`\\approx L^2 / 12` weighted by its total power, so
    the reported D4sigma inflates roughly as

    .. math::
        \\left(\\frac{D_{4\\sigma}^{\\text{meas}}}{D_{4\\sigma}^{\\text{true}}}
        \\right)^2 \\sim 1 + f_{\\text{ped}}
        \\left(\\frac{L}{D_{4\\sigma}^{\\text{true}}}\\right)^2 ,

    where :math:`f_{\\text{ped}}` is the pedestal's power fraction.  The
    inflation is *unbounded in the grid size* -- a 1e-4-of-peak floor on
    a grid many beam-widths wide can multiply D4sigma several-fold.
    ``single_plane_metrics`` and ``find_best_focus(metric='spot'|'rms')``
    inherit this, so a noisy field can report a wrong best-focus plane.
    Enable ``background`` and/or ``aperture`` to suppress it.  Defaults
    are byte-identical to the historical whole-grid moment.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously a
    # ``PartialCoherenceMCF`` input failed downstream at
    # ``np.abs(E)`` with ``TypeError: bad operand type for abs()``;
    # a 3-D ensemble would have produced a wrong (3-D) variance
    # estimate via NumPy broadcasting.  The V6 walker now discovers
    # this entry via first-positional-name ``E``; the inline guard
    # routes both failure modes to the canonical v4.16 message.
    # Input kind: 'field' (2-D scalar complex amplitude).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'beam_d4sigma')
    if dy is None:
        dy = dx
    xp = _xp_of(E)
    Ny, Nx = E.shape
    x = (xp.arange(Nx) - Nx / 2) * dx
    y = (xp.arange(Ny) - Ny / 2) * dy
    X, Y = xp.meshgrid(x, y)

    I = xp.abs(E) ** 2
    # S3-5: opt-in ISO 11146 sec. 3.4 pedestal subtraction (no-op by
    # default, so clean fields are byte-identical to prior releases).
    I = _subtract_background(xp, I, background)

    total = xp.sum(I)
    if float(total) == 0:
        return 0.0, 0.0

    cx = xp.sum(X * I) / total
    cy = xp.sum(Y * I) / total
    var_x = xp.sum((X - cx) ** 2 * I) / total
    var_y = xp.sum((Y - cy) ** 2 * I) / total
    d4x = 4 * xp.sqrt(var_x)
    d4y = 4 * xp.sqrt(var_y)

    # S3-5: opt-in ISO 11146 clause 9 iterative integration aperture.
    # The default (aperture is None) integrates over the whole grid and
    # never enters this loop, so it is byte-identical to prior releases.
    if aperture is not None:
        widths = (_ISO_APERTURE_WIDTHS if aperture is True
                  else float(aperture))
        half_w = 0.5 * widths
        for _ in range(int(max_iter)):
            ax = half_w * float(d4x)
            ay = half_w * float(d4y)
            # A degenerate zero-width beam collapses the aperture to a
            # point; stop before the mask empties the integral.
            if ax <= 0.0 or ay <= 0.0:
                break
            mask = (xp.abs(X - cx) <= ax) & (xp.abs(Y - cy) <= ay)
            Im = xp.where(mask, I, xp.zeros_like(I))
            tot = xp.sum(Im)
            if float(tot) == 0:
                break
            cx = xp.sum(X * Im) / tot
            cy = xp.sum(Y * Im) / tot
            var_x = xp.sum((X - cx) ** 2 * Im) / tot
            var_y = xp.sum((Y - cy) ** 2 * Im) / tot
            new_d4x = 4 * xp.sqrt(var_x)
            new_d4y = 4 * xp.sqrt(var_y)
            fx, fy = float(new_d4x), float(new_d4y)
            converged = (
                fx > 0.0 and fy > 0.0
                and abs(fx - float(d4x)) <= tol * fx
                and abs(fy - float(d4y)) <= tol * fy)
            d4x, d4y = new_d4x, new_d4y
            if converged:
                break

    return float(d4x), float(d4y)


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

    Like :func:`beam_d4sigma`, the spatial and angular second moments
    integrate over the whole grid, so an intensity pedestal inflates
    both variances (see the pedestal-sensitivity note there).  ``M2``
    does *not* expose ``background`` / ``aperture`` opt-ins: the
    angular moment is taken from the field's FFT, so an ISO integration
    aperture would have to window the complex field (injecting edge
    diffraction into the angular spectrum) rather than mask an
    intensity map, and near-/far-field pedestals are independent.
    Clean a noisy input before calling ``M2``; use ``beam_d4sigma`` with
    ``background``/``aperture`` when a plain second-moment width suffices.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble input failed at ``E.shape`` unpacking with
    # ``ValueError: too many values to unpack`` (for 3-D) or at the
    # ``.shape`` attribute access (for MCF).  Routes both failure
    # modes to the canonical v4.16 message via the V6 walker.
    # Input kind: 'field'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'M2')
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
    Xs = X - cx
    Ys = Y - cy
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

    Note (P4): the threshold is taken against the 2-D ``peak`` but applied
    to the RADIALLY-AVERAGED profile.  For an asymmetric beam the azimuthal
    average is below the 2-D peak, so the reported diameter is biased
    slightly LOW; use ``M2`` / second-moment widths for anisotropic beams.

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
