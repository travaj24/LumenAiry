"""
Wavefront / OPD analysis + sampling diagnostics + depth-of-focus.

This submodule was carved out of ``lumenairy.analysis.core`` in v5.1.0
as part of the mechanical 6-file split (see ``ROADMAP.md`` v5.1
"Architecture / housekeeping").  All functions, signatures, and numerics
are unchanged -- the historical public API is preserved by a thin
re-export shell in ``lumenairy.analysis.core``.

Contents:

* Sampling diagnostics: :func:`check_sampling_conditions`,
  :func:`check_opd_sampling`.
* Mode subtraction: :func:`remove_wavefront_modes`.
* OPD statistics: :func:`opd_pv_rms`, :func:`wave_opd_1d`,
  :func:`wave_opd_2d`.
* Depth of focus: :func:`depth_of_focus`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


__all__ = [
    'check_sampling_conditions',
    'check_opd_sampling',
    'remove_wavefront_modes',
    'opd_pv_rms',
    'wave_opd_1d',
    'wave_opd_2d',
    'depth_of_focus',
]


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
        print(f"Nyquist (dx < λ/2 = {nyquist_limit * 1e6:.3f} um): "
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
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble input failed at ``E.shape`` unpacking with
    # ``ValueError: too many values to unpack`` (3-D) or
    # ``AttributeError`` (MCF) -- routes both to the canonical
    # v4.16 message via the V6 walker.  Input kind: 'field'.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E, 'wave_opd_2d')
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
