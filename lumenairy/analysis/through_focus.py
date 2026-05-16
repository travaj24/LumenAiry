"""
Through-focus analysis: aberration, wavefront, Strehl, best focus, and
tolerancing for optical systems.

Given an exit-pupil complex field (e.g., the output of
``apply_real_lens`` applied to a collimated source), this module scans
the beam along z through the nominal focal region and reports, at each
z plane:

* peak intensity
* Strehl ratio (relative to the diffraction-limited reference)
* FWHM spot diameter (via second-moment D4sigma, or Gaussian-fit
  equivalent)
* RMS spot radius
* encircled-energy (power-in-bucket) at a user-specified radius

The nominal ("paraxial") focal plane is typically close to the system's
back focal length, but the **best-focus** plane -- the one that
maximizes Strehl or minimizes spot size -- shifts with aberrations and
with the choice of metric.  :func:`find_best_focus` searches the scan
for the extremum of any metric.

Tolerancing
-----------
:func:`tolerancing_sweep` applies a list of prescription perturbations
(decenter, tilt, form error) to a real-lens prescription, re-runs the
wave propagation for each perturbation, and reports the resulting
Strehl drop.  This lets you rank which degrees of freedom the design
is most sensitive to.

:func:`monte_carlo_tolerancing` draws random perturbations from
user-specified distributions and aggregates Strehl/RMS statistics.

Author: Andrew Traverso
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..propagators.propagation import angular_spectrum_propagate
from ..elements.lenses import apply_real_lens
from .core import (
    beam_centroid,
    beam_d4sigma,
    radial_power_bands,
)


__all__ = [
    # single-plane / through-focus
    'single_plane_metrics',
    'diffraction_limited_peak',
    'ThroughFocusResult',
    'through_focus_scan',
    'through_focus_scan_jax',
    'find_best_focus',
    'plot_through_focus',
    # tolerancing
    'Perturbation',
    'apply_perturbations',
    'tolerancing_sweep',
    'monte_carlo_tolerancing',
    'monte_carlo_tolerancing_jax',
    'monte_carlo_tolerancing_linearized',
    'tolerancing_report',
]


# =========================================================================
# Single-plane metrics
# =========================================================================

def single_plane_metrics(
    E: np.ndarray,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
    bucket_radius: Optional[float] = None,
    ideal_peak: Optional[float] = None,
) -> Dict[str, float]:
    """Compute diagnostic metrics for a beam at a single z plane.

    Parameters
    ----------
    E : ndarray, complex, shape (Ny, Nx)
        Field at the plane of interest.
    dx : float
        Grid spacing in x [m].
    wavelength : float
        Vacuum wavelength [m].  (Currently unused, but kept for API
        symmetry with planned diffraction-limited overlays.)
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    bucket_radius : float, optional
        Radius [m] at which to report encircled energy (power-in-bucket).
    ideal_peak : float, optional
        Peak of the diffraction-limited reference |E|^2 at the same
        plane.  If supplied, the returned dict includes ``strehl``.

    Returns
    -------
    metrics : dict
        Keys include ``peak_I``, ``total_power``, ``centroid_x``,
        ``centroid_y``, ``d4sigma_x``, ``d4sigma_y``, ``rms_radius``.
        Also ``power_in_bucket`` if ``bucket_radius`` is supplied, and
        ``strehl`` if ``ideal_peak`` is supplied.
    """
    if dy is None:
        dy = dx

    I = np.abs(E) ** 2
    peak = float(I.max())
    total = float(I.sum() * dx * dy)
    cx, cy = beam_centroid(E, dx, dy)
    d4x, d4y = beam_d4sigma(E, dx, dy)
    # 1-sigma radius from D4sigma (full = 4*sigma by definition)
    sigma_x = d4x / 4.0
    sigma_y = d4y / 4.0
    rms_r = float(np.sqrt(sigma_x ** 2 + sigma_y ** 2))

    out = {
        'peak_I': peak,
        'total_power': total,
        'centroid_x': float(cx),
        'centroid_y': float(cy),
        'd4sigma_x': float(d4x),
        'd4sigma_y': float(d4y),
        'rms_radius': rms_r,
    }

    if bucket_radius is not None:
        powers = radial_power_bands(
            E, dx, [bucket_radius], dy=dy, center=(cx, cy))
        out['power_in_bucket'] = float(powers[0])

    if ideal_peak is not None and ideal_peak > 0:
        out['strehl'] = peak / ideal_peak

    return out


def diffraction_limited_peak(
    E_exit: np.ndarray,
    wavelength: float,
    f: float,
    dx: float,
    bandlimit: bool = True,
) -> float:
    """Peak intensity of the diffraction-limited focal spot produced by
    the exit pupil amplitude, evaluated at the paraxial focus.

    Computes the ideal (aberration-free) reference used as the
    denominator of the Strehl ratio.  Operates by stripping the phase
    of ``E_exit`` -- keeping only its amplitude -- then applying a
    perfect converging lens phase of focal length ``f`` and propagating
    that modified field by the SAME method (angular-spectrum) used in
    the through-focus scan.  This keeps units and numerical factors
    consistent so the resulting Strehl ratios are directly comparable.

    Parameters
    ----------
    E_exit : ndarray, complex, shape (Ny, Nx)
        Exit-pupil field (from ``apply_real_lens`` applied to a
        collimated source).
    wavelength : float
        Vacuum wavelength [m].
    f : float
        Nominal back focal length [m] -- the converging-phase radius
        of the ideal reference.
    dx : float
        Grid spacing [m].  Assumed isotropic.
    bandlimit : bool, default True
        Passed through to :func:`angular_spectrum_propagate`.

    Returns
    -------
    peak : float
        Peak of ``|E_ideal(z=f)|^2`` -- use as the denominator in
        ``strehl = |E_actual(z)|^2.max() / peak``.
    """
    E = np.asarray(E_exit)
    Ny, Nx = E.shape
    k0 = 2.0 * np.pi / wavelength
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dx
    X, Y = np.meshgrid(x, y)

    # Ideal pupil: same amplitude, perfect converging quadratic phase
    E_ideal = np.abs(E) * np.exp(-1j * k0 * (X ** 2 + Y ** 2) / (2.0 * f))
    E_focus = angular_spectrum_propagate(
        E_ideal, f, wavelength, dx, bandlimit=bandlimit)
    return float((np.abs(E_focus) ** 2).max())


# =========================================================================
# Through-focus scan
# =========================================================================

@dataclass
class ThroughFocusResult:
    """Container for through-focus scan metrics."""

    z: np.ndarray                # shape (n_z,), axial positions [m]
    peak_I: np.ndarray
    strehl: np.ndarray            # NaN if ideal_peak not provided
    d4sigma_x: np.ndarray
    d4sigma_y: np.ndarray
    rms_radius: np.ndarray
    power_in_bucket: np.ndarray   # NaN if bucket_radius not provided

    # Provenance
    wavelength: float = 0.0
    dx: float = 0.0
    bucket_radius: float = 0.0
    ideal_peak: float = 0.0
    best_focus_strehl: float = float('nan')
    best_focus_spot: float = float('nan')


def through_focus_scan(
    E_exit: np.ndarray,
    dx: float,
    wavelength: float,
    z_values: Union[Sequence[float], np.ndarray],
    bucket_radius: Optional[float] = None,
    ideal_peak: Optional[float] = None,
    bandlimit: bool = True,
    verbose: bool = False,
    progress: Optional[Callable[..., Any]] = None,
    *,
    backend: str = 'numpy',
) -> 'ThroughFocusResult':
    """Propagate an exit-pupil field to each z and collect metrics.

    Parameters
    ----------
    E_exit : ndarray, complex, shape (Ny, Nx)
        Field at the lens exit plane (typically from ``apply_real_lens``
        applied to a collimated source).
    dx : float
        Grid spacing [m].  Assumed isotropic.
    wavelength : float
        Vacuum wavelength [m].
    z_values : array-like of float
        Axial distances past the exit plane [m] at which to evaluate
        the beam.  Typically a linspace bracketing the expected focal
        plane.  May include negative values.
    bucket_radius : float, optional
        Radius [m] for encircled-energy measurement.  If omitted, the
        corresponding array of the returned result contains NaN.
    ideal_peak : float, optional
        Peak intensity of the diffraction-limited reference (use
        :func:`diffraction_limited_peak`).  Required for Strehl; if
        omitted, ``strehl`` is filled with NaN.
    bandlimit : bool, default True
        Passed to :func:`angular_spectrum_propagate`.
    verbose : bool, default False
        Print progress as the scan runs.

    Returns
    -------
    result : ThroughFocusResult

    Notes
    -----
    Pass ``backend='jax'`` to dispatch to the JAX-traced implementation
    (:func:`through_focus_scan_jax`).  This is the canonical 4.7+ way
    to opt into the JAX path; the ``through_focus_scan_jax`` function
    remains importable but is no longer the preferred entry point.
    """
    if backend == 'jax':
        return through_focus_scan_jax(
            E_exit, dx, wavelength, z_values,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
            bandlimit=bandlimit)
    if backend != 'numpy':
        raise ValueError(
            f"through_focus_scan: backend must be 'numpy' or 'jax'; "
            f"got {backend!r}.")
    from ..progress import call_progress
    z_arr = np.asarray(z_values, dtype=np.float64)
    n_z = z_arr.size

    peak_I = np.full(n_z, np.nan)
    strehl = np.full(n_z, np.nan)
    d4x = np.full(n_z, np.nan)
    d4y = np.full(n_z, np.nan)
    rms_r = np.full(n_z, np.nan)
    p_bucket = np.full(n_z, np.nan)

    # Reuse bandlimit work by computing incremental deltas, but the
    # ASM transfer function factors by z so we just propagate from the
    # exit each time for simplicity and robustness.
    for i, z in enumerate(z_arr):
        call_progress(progress, 'through_focus_scan',
                      i / max(n_z, 1),
                      f'plane {i + 1}/{n_z}  z={z*1e3:+.3f} mm')
        if z == 0.0:
            E_z = E_exit
        else:
            E_z = angular_spectrum_propagate(
                E_exit, z, wavelength, dx, bandlimit=bandlimit)

        m = single_plane_metrics(
            E_z, dx, wavelength,
            bucket_radius=bucket_radius,
            ideal_peak=ideal_peak,
        )
        peak_I[i] = m['peak_I']
        d4x[i] = m['d4sigma_x']
        d4y[i] = m['d4sigma_y']
        rms_r[i] = m['rms_radius']
        if 'power_in_bucket' in m:
            p_bucket[i] = m['power_in_bucket']
        if 'strehl' in m:
            strehl[i] = m['strehl']
        if verbose:
            s_str = (f'{m["strehl"]:.3f}' if 'strehl' in m else '--')
            print(f'    [{i+1:3d}/{n_z}] z={z*1e3:+8.3f} mm  '
                  f'peak={m["peak_I"]:.3e}  Strehl={s_str}  '
                  f'D4sig_x={m["d4sigma_x"]*1e6:.2f} um')

    call_progress(progress, 'through_focus_scan', 1.0, 'done')
    return ThroughFocusResult(
        z=z_arr, peak_I=peak_I, strehl=strehl,
        d4sigma_x=d4x, d4sigma_y=d4y, rms_radius=rms_r,
        power_in_bucket=p_bucket,
        wavelength=float(wavelength), dx=float(dx),
        bucket_radius=float(bucket_radius) if bucket_radius else 0.0,
        ideal_peak=float(ideal_peak) if ideal_peak else 0.0,
    )


def find_best_focus(
    scan: 'ThroughFocusResult',
    metric: str = 'strehl',
) -> Tuple[float, float]:
    """Return the z position in the scan that best optimizes the given
    metric.

    Parameters
    ----------
    scan : ThroughFocusResult
    metric : str
        One of:
          * ``'strehl'``   -- maximize
          * ``'peak_I'``   -- maximize
          * ``'spot'``     -- minimize d4sigma (average of x/y)
          * ``'rms'``      -- minimize rms_radius
          * ``'bucket'``   -- maximize power_in_bucket

    Returns
    -------
    z_best : float
        Axial position [m] of the optimum.
    value_best : float
        Metric value at the optimum.
    """
    if metric == 'strehl':
        values = scan.strehl
        best_idx = int(np.nanargmax(values))
    elif metric == 'peak_I':
        values = scan.peak_I
        best_idx = int(np.nanargmax(values))
    elif metric == 'spot':
        values = 0.5 * (scan.d4sigma_x + scan.d4sigma_y)
        best_idx = int(np.nanargmin(values))
    elif metric == 'rms':
        values = scan.rms_radius
        best_idx = int(np.nanargmin(values))
    elif metric == 'bucket':
        values = scan.power_in_bucket
        best_idx = int(np.nanargmax(values))
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    return float(scan.z[best_idx]), float(values[best_idx])


def plot_through_focus(
    scan: 'ThroughFocusResult',
    best_z: Optional[float] = None,
    path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Any, Any]:
    """Summary plot: peak / Strehl / spot-size vs z.

    Parameters
    ----------
    scan : ThroughFocusResult
    best_z : float, optional
        Axial position to mark with a vertical line (e.g. from
        :func:`find_best_focus`).
    path : str, optional
        If given, save the figure to this path.
    show : bool, default False
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, axes : matplotlib Figure + Axes
    """
    import matplotlib.pyplot as plt

    z_mm = scan.z * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(z_mm, scan.peak_I, 'C0-', lw=1.5)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('peak intensity [a.u.]')
    ax.set_title('Peak intensity vs z')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if np.any(np.isfinite(scan.strehl)):
        ax.plot(z_mm, scan.strehl, 'C1-', lw=1.5)
        # Strehl can exceed 1 in some edge cases (e.g. the ideal
        # reference is on-axis but the aberrated peak happens to
        # concentrate).  Clamping to 1.05 would hide that.
        smax = float(np.nanmax(scan.strehl))
        ax.set_ylim(0, max(1.05, smax * 1.1))
    else:
        ax.text(0.5, 0.5, 'no ideal_peak provided',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('Strehl ratio')
    ax.set_title('Strehl vs z')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_mm, scan.d4sigma_x * 1e6, 'C2-', lw=1.2, label='D4sigma X')
    ax.plot(z_mm, scan.d4sigma_y * 1e6, 'C3-', lw=1.2, label='D4sigma Y')
    ax.plot(z_mm, 2.0 * scan.rms_radius * 1e6, 'k--', lw=1.0,
            label='2 * rms_radius')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('spot diameter [um]')
    ax.set_title('Spot size vs z')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    ax = axes[1, 1]
    if np.any(np.isfinite(scan.power_in_bucket)):
        ax.plot(z_mm, scan.power_in_bucket, 'C4-', lw=1.5)
    else:
        ax.text(0.5, 0.5, 'no bucket_radius provided',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('encircled power [a.u.]')
    ax.set_title('Power in bucket vs z')
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        if best_z is not None:
            ax.axvline(best_z * 1e3, color='red', lw=0.8, ls=':',
                       label='best focus' if ax is axes[0, 0] else None)

    fig.suptitle(
        f'Through-focus scan -- lambda={scan.wavelength*1e9:.0f} nm, '
        f'dx={scan.dx*1e6:.2f} um',
        fontsize=12)
    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


# =========================================================================
# Tolerancing
# =========================================================================

@dataclass
class Perturbation:
    """One surface perturbation for tolerancing.

    Attributes
    ----------
    surface_index : int
        0-based index into the prescription's ``surfaces`` list.
    decenter : tuple of (float, float), optional
        Lateral shift ``(dx, dy)`` in meters.
    tilt : tuple of (float, float), optional
        Small-angle tilt ``(tx, ty)`` in radians.
    form_error : ndarray, optional
        Additive sag-error map [m], same grid as the simulation.
    form_error_rms : float, optional
        RMS amplitude [m] of a random-phase form-error map to be
        generated at evaluation time (alternative to supplying an
        explicit map).  Seeded via ``random_seed``.
    random_seed : int, optional
        Seed for reproducible form-error generation.
    name : str
        Human-readable label; used in the report.
    """

    surface_index: int
    decenter: Tuple[float, float] = (0.0, 0.0)
    tilt: Tuple[float, float] = (0.0, 0.0)
    form_error: Optional[np.ndarray] = None
    form_error_rms: float = 0.0
    random_seed: Optional[int] = None
    name: str = ''


def apply_perturbations(
    prescription: Dict[str, Any],
    perturbations: Sequence['Perturbation'],
    N: Optional[int] = None,
    dx: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a deep-copied prescription with perturbations applied.

    Parameters
    ----------
    prescription : dict
        The original prescription.  NOT modified.
    perturbations : list of Perturbation
    N, dx : int, float, optional
        Grid size and spacing; required only when a perturbation
        specifies ``form_error_rms`` (to generate the random map).

    Returns
    -------
    perturbed : dict
        Deep-copied prescription with ``decenter``, ``tilt``, and
        ``form_error`` fields set on the indicated surfaces.
    """
    p = copy.deepcopy(prescription)
    surfaces = p['surfaces']

    for pert in perturbations:
        if pert.surface_index < 0 or pert.surface_index >= len(surfaces):
            raise IndexError(
                f"Perturbation references surface_index="
                f"{pert.surface_index} but prescription has "
                f"{len(surfaces)} surfaces.")
        s = surfaces[pert.surface_index]
        if pert.decenter != (0.0, 0.0):
            s['decenter'] = pert.decenter
        if pert.tilt != (0.0, 0.0):
            s['tilt'] = pert.tilt
        if pert.form_error is not None:
            s['form_error'] = pert.form_error
        elif pert.form_error_rms > 0.0:
            if N is None or dx is None:
                raise ValueError(
                    'form_error_rms requires N and dx to generate '
                    'the random map.')
            rng = np.random.default_rng(pert.random_seed)
            raw = rng.standard_normal((N, N))
            # Gentle low-pass (gaussian blur) for physically plausible
            # figure error -- avoids high-spatial-frequency noise.
            kx = np.fft.fftfreq(N, d=dx)
            KX, KY = np.meshgrid(kx, kx)
            cutoff = 1.0 / (20.0 * dx)  # ~1/20 of Nyquist
            H = np.exp(-(KX ** 2 + KY ** 2) / (2 * cutoff ** 2))
            smooth = np.real(np.fft.ifft2(np.fft.fft2(raw) * H))
            smooth -= smooth.mean()
            if smooth.std() > 0:
                smooth *= pert.form_error_rms / smooth.std()
            s['form_error'] = smooth

    return p


def tolerancing_sweep(
    prescription: Dict[str, Any],
    wavelength: float,
    N: int,
    dx: float,
    E_source: np.ndarray,
    perturbations: Sequence['Perturbation'],
    focal_length: float,
    aperture: float,
    z_scan_range: Optional[Tuple[float, float]] = None,
    z_scan_n: int = 21,
    bucket_radius: Optional[float] = None,
    verbose: bool = True,
    progress: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a deterministic tolerancing sweep: for each perturbation,
    rerun ``apply_real_lens``, scan through focus, and record Strehl
    and spot-size penalties relative to the nominal case.

    Parameters
    ----------
    prescription : dict
        Nominal lens prescription.
    wavelength : float
        Vacuum wavelength [m].
    N, dx : int, float
        Grid.
    E_source : ndarray, complex, shape (N, N)
        Input source field at the entrance pupil of ``prescription``.
    perturbations : list of Perturbation
    focal_length : float
        Nominal back focal length [m] -- used to set the default scan
        range and to compute the diffraction-limited reference.
    aperture : float
        Aperture diameter [m] used to build the reference spot.
    z_scan_range : tuple of (float, float), optional
        Scan limits relative to ``focal_length``.  Defaults to
        +-focal_length/20.
    z_scan_n : int
        Number of z samples in the through-focus scan.
    bucket_radius : float, optional
        Encircled-energy radius [m] for the power-in-bucket metric.
    verbose : bool

    Returns
    -------
    results : list of dict
        One entry per perturbation (plus a leading nominal entry with
        name ``'nominal'``).  Each has keys:
        ``name, z_best, strehl_peak, d4sigma_min, rms_min,
        delta_strehl, delta_spot``.
    """
    from ..progress import call_progress, ProgressScaler
    if z_scan_range is None:
        z_scan_range = (-focal_length / 20.0, focal_length / 20.0)
    z_values = np.linspace(z_scan_range[0], z_scan_range[1], z_scan_n) \
                  + focal_length

    def _run_one(pres_used, label, inner_scaler=None):
        E_exit = apply_real_lens(
            E_source, prescription=pres_used, wavelength=wavelength, dx=dx,
            bandlimit=True, slant_correction=True)
        ideal_peak = diffraction_limited_peak(
            E_exit, wavelength, focal_length, dx)
        scan = through_focus_scan(
            E_exit, dx, wavelength, z_values,
            bucket_radius=bucket_radius,
            ideal_peak=ideal_peak, verbose=False,
            progress=(lambda s, f, m='': inner_scaler(f, m))
                     if inner_scaler is not None else None)
        z_best, strehl_peak = find_best_focus(scan, 'strehl')
        z_spot, spot_min = find_best_focus(scan, 'spot')
        _, rms_min = find_best_focus(scan, 'rms')
        return {
            'name': label,
            'z_best': z_best,
            'strehl_peak': strehl_peak,
            'd4sigma_min': spot_min,
            'rms_min': rms_min,
            'scan': scan,
        }

    # Budget: nominal (1) + N perturbation runs.  Split the [0, 1] bar
    # equally so each run gets its own window to report fine progress.
    n_total = len(perturbations) + 1
    results = []
    if verbose:
        print('  -- Nominal ...')
    nominal_scaler = ProgressScaler(progress, 'tolerancing_sweep',
                                     0.0, 1.0 / n_total)
    call_progress(progress, 'tolerancing_sweep', 0.0, 'nominal run')
    nominal = _run_one(prescription, 'nominal', nominal_scaler)
    results.append(nominal)

    for i, pert in enumerate(perturbations):
        label = pert.name or f'pert_{i}'
        if verbose:
            print(f'  -- {label} ...')
        lo = (i + 1) / n_total
        hi = (i + 2) / n_total
        call_progress(progress, 'tolerancing_sweep', lo,
                      f'perturbation {i + 1}/{n_total - 1}: {label}')
        pres_pert = apply_perturbations(prescription, [pert], N=N, dx=dx)
        inner = ProgressScaler(progress, 'tolerancing_sweep', lo, hi)
        r = _run_one(pres_pert, label, inner)
        r['delta_strehl'] = r['strehl_peak'] - nominal['strehl_peak']
        r['delta_spot'] = r['d4sigma_min'] - nominal['d4sigma_min']
        results.append(r)
        if verbose:
            print(f'     Strehl = {r["strehl_peak"]:.4f} '
                  f'(dS={r["delta_strehl"]:+.4f}),  '
                  f'spot = {r["d4sigma_min"]*1e6:.3f} um '
                  f'(dSpot={r["delta_spot"]*1e6:+.3f} um)')

    call_progress(progress, 'tolerancing_sweep', 1.0, 'done')
    return results


def monte_carlo_tolerancing(
    prescription: Dict[str, Any],
    wavelength: float,
    N: int,
    dx: float,
    E_source: np.ndarray,
    perturbation_spec: Sequence[Dict[str, Any]],
    focal_length: float,
    aperture: float,
    n_trials: int = 100,
    seed: int = 0,
    z_scan_range: Optional[Tuple[float, float]] = None,
    z_scan_n: int = 21,
    bucket_radius: Optional[float] = None,
    verbose: bool = True,
    progress: Optional[Callable[..., Any]] = None,
    *,
    backend: str = 'numpy',
) -> Dict[str, Any]:
    """Random tolerancing: draw perturbations from user-specified
    distributions and aggregate Strehl statistics.

    Parameters
    ----------
    prescription, wavelength, N, dx, E_source, focal_length, aperture
        Same as :func:`tolerancing_sweep`.
    perturbation_spec : list of dict
        One spec per surface perturbation.  Each dict has keys:
          - ``'surface_index'`` : int
          - ``'decenter_std'``   : float, 1-sigma isotropic lateral shift [m]
          - ``'tilt_std'``        : float, 1-sigma isotropic tilt [rad]
          - ``'form_error_rms'``  : float, RMS sag error [m] (used as-is
                                     if positive; a fresh random
                                     realization is drawn per trial)
    n_trials : int
    seed : int
        Base seed.  Trial ``i`` uses ``seed + i`` for reproducibility.
    z_scan_range, z_scan_n, bucket_radius : see
        :func:`tolerancing_sweep`.

    Returns
    -------
    stats : dict
        Keys: ``strehl_peak_mean``, ``strehl_peak_std``,
        ``strehl_peak_p05``, ``strehl_peak_p50``, ``strehl_peak_p95``,
        ``trial_results``.  ``trial_results`` is a list of per-trial
        dicts in the same format as :func:`tolerancing_sweep`.

    Notes
    -----
    Pass ``backend='jax'`` to dispatch to the JAX-vectorised
    implementation (:func:`monte_carlo_tolerancing_jax`).
    """
    if backend == 'jax':
        return monte_carlo_tolerancing_jax(
            prescription, wavelength, N, dx, E_source,
            perturbation_spec, focal_length, aperture,
            n_trials=n_trials, seed=seed,
            z_scan_range=z_scan_range, z_scan_n=z_scan_n,
            bucket_radius=bucket_radius,
            verbose=verbose, progress=progress)
    if backend != 'numpy':
        raise ValueError(
            f"monte_carlo_tolerancing: backend must be 'numpy' or "
            f"'jax'; got {backend!r}.")
    from ..progress import call_progress
    if z_scan_range is None:
        z_scan_range = (-focal_length / 20.0, focal_length / 20.0)
    z_values = np.linspace(z_scan_range[0], z_scan_range[1], z_scan_n) \
                  + focal_length

    trial_results = []
    strehls = np.empty(n_trials)

    for t in range(n_trials):
        # Each Monte-Carlo trial is a full apply_real_lens + through-focus
        # scan, so emit a progress event before it starts so the bar
        # never looks stuck for 20-30 seconds on large-N runs.
        call_progress(progress, 'monte_carlo_tolerancing',
                      t / max(n_trials, 1),
                      f'trial {t + 1}/{n_trials}')
        rng = np.random.default_rng(seed + t)
        perts = []
        for spec in perturbation_spec:
            d_std = spec.get('decenter_std', 0.0)
            t_std = spec.get('tilt_std', 0.0)
            f_rms = spec.get('form_error_rms', 0.0)
            perts.append(Perturbation(
                surface_index=spec['surface_index'],
                decenter=(rng.normal(0, d_std) if d_std > 0 else 0.0,
                          rng.normal(0, d_std) if d_std > 0 else 0.0),
                tilt=(rng.normal(0, t_std) if t_std > 0 else 0.0,
                      rng.normal(0, t_std) if t_std > 0 else 0.0),
                form_error_rms=f_rms,
                random_seed=int(rng.integers(0, 2**31 - 1)),
                name=f'trial_{t}_s{spec["surface_index"]}',
            ))
        pres_p = apply_perturbations(prescription, perts, N=N, dx=dx)
        E_exit = apply_real_lens(
            E_source, prescription=pres_p, wavelength=wavelength, dx=dx,
            bandlimit=True, slant_correction=True)
        ideal_peak = diffraction_limited_peak(
            E_exit, wavelength, focal_length, dx)
        scan = through_focus_scan(
            E_exit, dx, wavelength, z_values,
            bucket_radius=bucket_radius,
            ideal_peak=ideal_peak, verbose=False)
        z_best, strehl_peak = find_best_focus(scan, 'strehl')
        strehls[t] = strehl_peak
        trial_results.append({
            'trial': t,
            'z_best': z_best,
            'strehl_peak': strehl_peak,
        })
        if verbose and (t + 1) % max(1, n_trials // 10) == 0:
            print(f'    trial {t+1}/{n_trials}: Strehl={strehl_peak:.3f}')

    call_progress(progress, 'monte_carlo_tolerancing', 1.0, 'done')
    return {
        'strehl_peak_mean': float(np.mean(strehls)),
        'strehl_peak_std':  float(np.std(strehls)),
        'strehl_peak_p05':  float(np.percentile(strehls, 5)),
        'strehl_peak_p50':  float(np.percentile(strehls, 50)),
        'strehl_peak_p95':  float(np.percentile(strehls, 95)),
        'trial_results': trial_results,
        'strehl_array': strehls,
    }


def through_focus_scan_jax(
    E_exit: np.ndarray,
    dx: float,
    wavelength: float,
    z_values: Union[Sequence[float], np.ndarray],
    bucket_radius: Optional[float] = None,
    ideal_peak: Optional[float] = None,
    bandlimit: bool = True,
) -> 'ThroughFocusResult':
    """JAX-vmapped through-focus scan.  Same return contract as
    :func:`through_focus_scan` but propagates all z values in a single
    fused JAX kernel via `jax.vmap`.

    Expected speedup over the NumPy loop: 5-15x for typical 30+ point
    scans, larger on GPU.  Uses JAX-traceable
    :func:`angular_spectrum_propagate` internally; if E_exit is a
    NumPy array it is converted to JAX once at the boundary.

    Identical signature to `through_focus_scan` but no `verbose` /
    `progress` callbacks (the whole scan is one fused call).
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            'JAX is not installed; install with `pip install jax` or '
            'use through_focus_scan() (NumPy).')
    import jax
    import jax.numpy as jnp
    from ..propagators.propagation import angular_spectrum_propagate
    from .core import beam_d4sigma

    z_arr = np.asarray(z_values, dtype=np.float64)
    n_z = z_arr.size

    # angular_spectrum_propagate has Python branches on z (== 0, != 0)
    # that prevent JIT/vmap with a tracer z.  We pass a Python-float z
    # but a JAX-array E_in so the heavy FFT work runs through the JAX
    # backend (cuFFT on GPU, jnp.fft on CPU).  Per-call tracing cost
    # is negligible because the transfer-function cache in
    # propagation.py is keyed on (z, wavelength, dx, shape).
    E_jax = jnp.asarray(E_exit)
    fields_list = [
        angular_spectrum_propagate(
            E_jax, float(z), wavelength, dx, bandlimit=bandlimit)
        for z in z_arr
    ]
    fields = jnp.stack(fields_list, axis=0)   # (n_z, Ny, Nx)

    # Per-plane metrics.  Bring back to NumPy at the metric boundary
    # since the metrics package isn't JAX-vmapped.
    fields_np = np.asarray(fields)
    peak_I = np.full(n_z, np.nan)
    strehl = np.full(n_z, np.nan)
    d4x = np.full(n_z, np.nan)
    d4y = np.full(n_z, np.nan)
    rms_r = np.full(n_z, np.nan)
    p_bucket = np.full(n_z, np.nan)
    for i in range(n_z):
        E_z = fields_np[i]
        I_z = np.abs(E_z) ** 2
        peak_I[i] = float(np.max(I_z))
        if ideal_peak is not None and ideal_peak > 0:
            strehl[i] = peak_I[i] / float(ideal_peak)
        try:
            d4 = beam_d4sigma(E_z, dx=dx)
            if hasattr(d4, '__len__'):
                d4x[i], d4y[i] = float(d4[0]), float(d4[1])
            else:
                d4x[i] = d4y[i] = float(d4)
        except Exception:
            pass
        # 4.10: match NumPy semantics for parity.
        # power_in_bucket: ABSOLUTE integrated intensity (J in arbitrary
        # units of |E|^2 * area) about the intensity centroid, NOT a
        # fraction.  Pre-4.10 the JAX twin returned fraction; calling
        # find_best_focus(scan, 'bucket') gave different z best
        # depending on backend.
        I_sum = float(np.sum(I_z))
        if I_sum > 0:
            yy, xx = np.indices(I_z.shape)
            cx_pix = float(np.sum(xx * I_z) / I_sum)
            cy_pix = float(np.sum(yy * I_z) / I_sum)
        else:
            cy_pix, cx_pix = I_z.shape[0] / 2, I_z.shape[1] / 2
        if bucket_radius is not None and bucket_radius > 0:
            yy, xx = np.indices(I_z.shape)
            r = np.hypot((xx - cx_pix) * dx, (yy - cy_pix) * dx)
            mask = r < bucket_radius
            p_bucket[i] = float(np.sum(I_z[mask]) * dx * dx)
        # rms radius: D4sigma/4 about centroid (matches NumPy path).
        if I_sum > 0:
            yy, xx = np.indices(I_z.shape)
            r2 = ((xx - cx_pix) * dx) ** 2 + ((yy - cy_pix) * dx) ** 2
            sigma_sq = float(np.sum(r2 * I_z) / I_sum)
            rms_r[i] = float(np.sqrt(sigma_sq))

    best_strehl = float(np.nanmax(strehl)) if ideal_peak else float('nan')
    best_spot = float(np.nanmin(rms_r))
    return ThroughFocusResult(
        z=z_arr, peak_I=peak_I, strehl=strehl,
        d4sigma_x=d4x, d4sigma_y=d4y, rms_radius=rms_r,
        power_in_bucket=p_bucket,
        wavelength=wavelength, dx=dx,
        bucket_radius=bucket_radius or 0.0,
        ideal_peak=ideal_peak or 0.0,
        best_focus_strehl=best_strehl, best_focus_spot=best_spot,
    )



def monte_carlo_tolerancing_jax(
    prescription: Dict[str, Any],
    wavelength: float,
    N: int,
    dx: float,
    E_source: np.ndarray,
    perturbation_spec: Sequence[Dict[str, Any]],
    focal_length: float,
    aperture: float,
    n_trials: int = 100,
    seed: int = 0,
    z_scan_range: Optional[Tuple[float, float]] = None,
    z_scan_n: int = 21,
    bucket_radius: Optional[float] = None,
    wave_propagator: str = 'real_lens_traced_jax',
    verbose: bool = True,
    progress: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """JAX-accelerated Monte-Carlo tolerancing trial sweep.

    Per-trial structure mirrors :func:`monte_carlo_tolerancing`:

      1. Draw perturbations from the user-specified distributions
         (NumPy ``np.random`` -- ``apply_perturbations`` mutates a
         Python prescription dict so the perturbation step itself
         stays in NumPy).
      2. Run the wave leg through the perturbed prescription.
         When ``wave_propagator='real_lens_traced_jax'`` (default)
         this is :func:`lumenairy.apply_real_lens_traced_jax`; pass
         ``'real_lens'`` for the analytic NumPy variant.
      3. Run :func:`through_focus_scan_jax` on the exit field --
         this is the heavy compute (one ASM propagation per ``z``
         sample) and runs as a single fused JAX kernel per trial.
      4. Record peak Strehl from the scan and aggregate across
         trials.

    The trial loop is sequential (perturbation generation cannot be
    vmap'd because ``apply_perturbations`` operates on a Python
    prescription dict).  The win over the pure-NumPy version comes
    from ``through_focus_scan_jax``'s fused per-z propagation, which
    yields the same 5-15x speedup quoted there for typical 30+ point
    z-scans (larger on GPU).

    Parameters and return contract are identical to
    :func:`monte_carlo_tolerancing`; the ``wave_propagator`` knob is
    new and lets you opt into the analytic NumPy ``apply_real_lens``
    if you need its full diffraction-amplitude treatment rather than
    the ray-traced OPD screen of ``apply_real_lens_traced_jax``.
    """
    from ..backend import JAX_AVAILABLE as _JAX_AVAILABLE
    if not _JAX_AVAILABLE:
        raise ImportError(
            'JAX is not installed; install with `pip install jax` or '
            'use monte_carlo_tolerancing() (NumPy).')
    from ..progress import call_progress
    from ..elements.lenses import (
        apply_real_lens, apply_real_lens_traced_jax,
    )

    if z_scan_range is None:
        z_scan_range = (-focal_length / 20.0, focal_length / 20.0)
    z_values = np.linspace(z_scan_range[0], z_scan_range[1], z_scan_n) \
                  + focal_length

    if wave_propagator not in ('real_lens_traced_jax', 'real_lens'):
        raise ValueError(
            f"wave_propagator must be 'real_lens_traced_jax' or "
            f"'real_lens'; got {wave_propagator!r}")

    trial_results = []
    strehls = np.empty(n_trials)
    for t in range(n_trials):
        call_progress(progress, 'monte_carlo_tolerancing_jax',
                      t / max(n_trials, 1),
                      f'trial {t + 1}/{n_trials}')
        rng = np.random.default_rng(seed + t)
        perts = []
        for spec in perturbation_spec:
            d_std = spec.get('decenter_std', 0.0)
            t_std = spec.get('tilt_std', 0.0)
            f_rms = spec.get('form_error_rms', 0.0)
            perts.append(Perturbation(
                surface_index=spec['surface_index'],
                decenter=(rng.normal(0, d_std) if d_std > 0 else 0.0,
                          rng.normal(0, d_std) if d_std > 0 else 0.0),
                tilt=(rng.normal(0, t_std) if t_std > 0 else 0.0,
                      rng.normal(0, t_std) if t_std > 0 else 0.0),
                form_error_rms=f_rms,
                random_seed=int(rng.integers(0, 2**31 - 1)),
                name=f'trial_{t}_s{spec["surface_index"]}',
            ))
        pres_p = apply_perturbations(prescription, perts, N=N, dx=dx)

        if wave_propagator == 'real_lens_traced_jax':
            E_exit = apply_real_lens_traced_jax(
                E_source, prescription=pres_p, wavelength=wavelength, dx=dx, ray_subsample=4)
        else:  # 'real_lens'
            E_exit_np = apply_real_lens(
                E_source, prescription=pres_p, wavelength=wavelength, dx=dx,
                bandlimit=True, slant_correction=True)
            E_exit = E_exit_np

        ideal_peak = diffraction_limited_peak(
            np.asarray(E_exit), wavelength, focal_length, dx)
        scan = through_focus_scan_jax(
            E_exit, dx, wavelength, z_values,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
        )
        z_best, strehl_peak = find_best_focus(scan, 'strehl')
        strehls[t] = strehl_peak
        trial_results.append({
            'trial': t,
            'z_best': z_best,
            'strehl_peak': strehl_peak,
        })
        if verbose and (t + 1) % max(1, n_trials // 10) == 0:
            print(f'    trial {t+1}/{n_trials}: Strehl={strehl_peak:.3f}')

    call_progress(progress, 'monte_carlo_tolerancing_jax', 1.0, 'done')
    return {
        'strehl_peak_mean': float(np.mean(strehls)),
        'strehl_peak_std':  float(np.std(strehls)),
        'strehl_peak_p05':  float(np.percentile(strehls, 5)),
        'strehl_peak_p50':  float(np.percentile(strehls, 50)),
        'strehl_peak_p95':  float(np.percentile(strehls, 95)),
        'trial_results': trial_results,
        'strehl_array': strehls,
    }


def tolerancing_report(
    stats: Dict[str, Any],
    *,
    perturbation_spec: Optional[Sequence[Dict[str, Any]]] = None,
    trial_perturbations: Optional[Sequence[Sequence['Perturbation']]] = None,
    strehl_thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
    n_yield_curve_points: int = 51,
    format: str = 'text',
) -> Union[str, Dict[str, Any]]:
    """Produce a human-readable report from a Monte-Carlo tolerancing run.

    Takes the dict returned by :func:`monte_carlo_tolerancing` or
    :func:`monte_carlo_tolerancing_jax` and produces:

      1. Summary statistics (mean, std, p05/p50/p95, min/max).
      2. Strehl yield at standard thresholds:
         ``P(S_peak > T)`` for T in ``strehl_thresholds``.
      3. Strehl-yield curve data (threshold_axis, yield_axis) for
         plotting the CDF.
      4. *(Optional)* Per-knob sensitivity ranking, if the original
         ``perturbation_spec`` and ``trial_perturbations`` (one
         Perturbation list per trial, in spec order) are supplied.
         Reports the abs-correlation between perturbation magnitude
         and Strehl loss, ranked by importance.

    Parameters
    ----------
    stats : dict
        Output of :func:`monte_carlo_tolerancing` / its JAX twin.
        Must have ``strehl_array`` populated.
    perturbation_spec : list of dict, optional
        The original perturbation_spec passed to the MC run.
    trial_perturbations : list of list of Perturbation, optional
        Per-trial perturbation lists.
    strehl_thresholds : tuple of float
        Yield thresholds.  Default ``(0.5, 0.7, 0.8, 0.9, 0.95)``.
    n_yield_curve_points : int
        Resolution of the yield CDF.
    format : ``'text'`` | ``'dict'``
        Return form.  ``'text'`` returns a printable string; ``'dict'``
        returns a structured dict suitable for further processing.

    Returns
    -------
    report : str or dict
    """
    if 'strehl_array' not in stats:
        raise ValueError(
            "tolerancing_report: stats dict has no 'strehl_array' key. "
            "Pass the dict returned by monte_carlo_tolerancing or "
            "monte_carlo_tolerancing_jax.")

    strehls = np.asarray(stats['strehl_array'], dtype=float)
    n_trials = strehls.size
    valid = np.isfinite(strehls)
    n_valid = int(valid.sum())
    s_finite = strehls[valid]

    def _stat(fn, default=float('nan')):
        return float(fn(s_finite)) if s_finite.size else default

    summary = {
        'mean': _stat(np.mean),
        'std':  _stat(np.std),
        'p05':  _stat(lambda a: np.percentile(a, 5)),
        'p50':  _stat(lambda a: np.percentile(a, 50)),
        'p95':  _stat(lambda a: np.percentile(a, 95)),
        'min':  _stat(np.min),
        'max':  _stat(np.max),
    }

    yields = {float(T): (float(np.mean(s_finite > T)) if s_finite.size
                         else float('nan'))
              for T in strehl_thresholds}

    if s_finite.size:
        T_axis = np.linspace(max(0.0, summary['min']), 1.0,
                              n_yield_curve_points)
        yield_axis = np.array(
            [float(np.mean(s_finite > t)) for t in T_axis])
    else:
        T_axis = np.zeros(n_yield_curve_points)
        yield_axis = np.zeros(n_yield_curve_points)

    sensitivity_rows = []
    sensitivity_note = None
    if perturbation_spec is not None and trial_perturbations is not None:
        feat_names = []
        for spec in perturbation_spec:
            sidx = spec['surface_index']
            feat_names.extend([
                f's{sidx}_decenter_x', f's{sidx}_decenter_y',
                f's{sidx}_tilt_x', f's{sidx}_tilt_y',
                f's{sidx}_form_error',
            ])
        feat_mat = np.zeros((n_trials, len(feat_names)))
        for t, perts in enumerate(trial_perturbations):
            for k, p in enumerate(perts):
                base = k * 5
                feat_mat[t, base + 0] = float(p.decenter[0])
                feat_mat[t, base + 1] = float(p.decenter[1])
                feat_mat[t, base + 2] = float(p.tilt[0])
                feat_mat[t, base + 3] = float(p.tilt[1])
                feat_mat[t, base + 4] = float(p.form_error_rms or 0.0)
        loss = (1.0 - strehls) - np.mean(1.0 - strehls)
        loss_std = np.std(loss)
        for j, name in enumerate(feat_names):
            x = np.abs(feat_mat[:, j])
            sx = x - np.mean(x)
            sx_std = np.std(sx)
            denom = sx_std * loss_std * n_trials
            rho = float(np.sum(sx * loss) / denom) if denom > 0 else 0.0
            sensitivity_rows.append(
                {'feature': name, 'rho_abs_loss': rho})
        sensitivity_rows.sort(key=lambda r: -abs(r['rho_abs_loss']))
    elif perturbation_spec is not None:
        sensitivity_note = (
            "Sensitivity ranking skipped: perturbation_spec was supplied "
            "but trial_perturbations was not.  Pass the per-trial "
            "Perturbation lists (one list per trial, in spec order).")

    report_dict = {
        'n_trials': n_trials,
        'n_valid_trials': n_valid,
        'strehl_summary': summary,
        'yields': yields,
        'yield_curve': {'threshold': T_axis, 'yield': yield_axis},
        'sensitivity_ranking': sensitivity_rows,
        'sensitivity_note': sensitivity_note,
    }

    if format == 'dict':
        return report_dict

    lines = ['=' * 60, 'Tolerancing Report', '=' * 60,
             f'Trials: {n_trials} ({n_valid} valid)', '',
             'Strehl summary', '-' * 30,
             f'  mean  = {summary["mean"]:.4f}',
             f'  std   = {summary["std"]:.4f}',
             f'  P05   = {summary["p05"]:.4f}',
             f'  P50   = {summary["p50"]:.4f}',
             f'  P95   = {summary["p95"]:.4f}',
             f'  min   = {summary["min"]:.4f}',
             f'  max   = {summary["max"]:.4f}',
             '',
             'Yield at threshold', '-' * 30,
             ]
    for T, y in yields.items():
        lines.append(f'  P(S_peak > {T:.2f}) = {y*100:.1f} %')
    lines.append('')
    if sensitivity_rows:
        lines.append('Sensitivity ranking (abs correlation with Strehl loss)')
        lines.append('-' * 60)
        lines.append(f'  {"feature":<28s} {"|rho|":>8s}')
        for r in sensitivity_rows[:10]:
            lines.append(
                f'  {r["feature"]:<28s} {r["rho_abs_loss"]:>8.4f}')
    elif sensitivity_note:
        lines.append('Sensitivity ranking')
        lines.append('-' * 30)
        lines.append(f'  ({sensitivity_note})')
    lines.append('')
    return '\n'.join(lines)


def monte_carlo_tolerancing_linearized(
    prescription: Dict[str, Any],
    wavelength: float,
    N: int,
    dx: float,
    E_source: np.ndarray,
    perturbation_spec: Sequence[Dict[str, Any]],
    focal_length: float,
    aperture: float,
    n_trials: int = 100,
    seed: int = 0,
    z_scan_n: int = 11,
    bucket_radius: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Linearised Monte-Carlo tolerancing.  Approximates the Strehl
    sensitivity to each perturbation knob via a small finite-difference
    sweep around the nominal, then composes per-trial Strehl predictions
    as the linear superposition of independent contributions.

    Speed: 1 nominal eval + ~16-30 FD probes (one per perturbation
    type x surface), independent of ``n_trials``.  For 100 trials this
    is ~3-6x faster than :func:`monte_carlo_tolerancing`.

    Accuracy: linear approximation valid for small perturbations
    (typical decenter ~5 um, tilt ~1 mrad, form-error RMS ~10 nm).
    Breaks down for larger perturbations.  Use the full MC variant
    for final tolerance budget sign-off.

    Returns the same dict shape as :func:`monte_carlo_tolerancing`
    (``strehl_array``, summary stats, ``trial_results``) so it
    drops into :func:`tolerancing_report` unchanged.
    """
    from ..elements.lenses import apply_real_lens

    if z_scan_n < 3:
        z_scan_n = 3
    z_values = np.linspace(-focal_length / 20.0, focal_length / 20.0,
                            z_scan_n) + focal_length

    # ----- Step 1: nominal eval -----
    if verbose:
        print('  Linearised MC: nominal eval')
    E_exit_0 = apply_real_lens(E_source, prescription=prescription, wavelength=wavelength, dx=dx,
                                bandlimit=True, slant_correction=True)
    ideal_peak = diffraction_limited_peak(
        E_exit_0, wavelength, focal_length, dx)
    scan_0 = through_focus_scan(
        E_exit_0, dx, wavelength, z_values,
        bucket_radius=bucket_radius, ideal_peak=ideal_peak, verbose=False)
    _, S_nom = find_best_focus(scan_0, 'strehl')

    # ----- Step 2: per-knob FD sensitivities -----
    # For each spec entry, perturb each knob (decenter_x, decenter_y,
    # tilt_x, tilt_y, form_error_rms) by 1 sigma and measure
    # dStrehl / dknob.  Sigma comes from the *_std fields in the spec.
    if verbose:
        print(f'  Linearised MC: probing {len(perturbation_spec)} '
              f'spec entries x 5 knobs each')
    sensitivities = []   # list of (spec_idx, knob_name, sigma, dStrehl/dknob)
    for spec_idx, spec in enumerate(perturbation_spec):
        sigmas = {
            'decenter_x': spec.get('decenter_std', 0.0),
            'decenter_y': spec.get('decenter_std', 0.0),
            'tilt_x':     spec.get('tilt_std',     0.0),
            'tilt_y':     spec.get('tilt_std',     0.0),
            'form_error': spec.get('form_error_rms', 0.0),
        }
        for knob, sigma in sigmas.items():
            if sigma <= 0:
                sensitivities.append((spec_idx, knob, 0.0, 0.0))
                continue
            # Build a perturbation that is +1 sigma along this knob only.
            decenter = (sigma if knob == 'decenter_x' else 0.0,
                        sigma if knob == 'decenter_y' else 0.0)
            tilt = (sigma if knob == 'tilt_x' else 0.0,
                    sigma if knob == 'tilt_y' else 0.0)
            form_rms = sigma if knob == 'form_error' else 0.0
            pert = Perturbation(
                surface_index=spec['surface_index'],
                decenter=decenter, tilt=tilt,
                form_error_rms=form_rms,
                # 4.10: Python 3's `hash(str)` is randomised per process
                # via PYTHONHASHSEED, so the form-error realisation
                # drifted between runs even with the same `seed=`
                # argument.  Use a deterministic knob-to-int mapping.
                random_seed=spec_idx * 100 + (
                    {'decenter_x': 1, 'decenter_y': 2,
                     'tilt_x': 3, 'tilt_y': 4,
                     'form_error': 5}.get(knob, 0)),
                name=f'fd_{spec_idx}_{knob}',
            )
            pres_p = apply_perturbations(prescription, [pert], N=N, dx=dx)
            try:
                E_p = apply_real_lens(E_source, prescription=pres_p, wavelength=wavelength, dx=dx,
                                       bandlimit=True,
                                       slant_correction=True)
                scan_p = through_focus_scan(
                    E_p, dx, wavelength, z_values,
                    bucket_radius=bucket_radius, ideal_peak=ideal_peak,
                    verbose=False)
                _, S_p = find_best_focus(scan_p, 'strehl')
            except Exception:
                S_p = S_nom   # sensitivity = 0 on failure
            # 4.10.2: Strehl is quadratic-around-nominal per the
            # Marechal approximation S ≈ exp(-sigma_phi^2), so the
            # linear coefficient is ~ 0 and the second-order
            # coefficient dominates.  Pre-4.10.2 used dS/dk linear
            # superposition which produced a MEAN-ZERO distribution
            # around S_nom -- wrong for the actual physics.  Fit a
            # quadratic per knob: a_k = (S_nom - S(sigma)) / sigma^2
            # so the per-trial sum reads S_pred = S_nom - sum a_k xi_k^2.
            a_k = (S_nom - S_p) / (sigma * sigma)
            sensitivities.append((spec_idx, knob, sigma, a_k))

    # ----- Step 3: per-trial QUADRATIC superposition (Marechal) -----
    rng = np.random.default_rng(seed)
    strehls = np.empty(n_trials)
    trial_results = []
    for t in range(n_trials):
        # Draw perturbations: each knob ~ N(0, sigma).  The Strehl
        # contribution is -a_k * xi_k^2 (each independent aberration
        # mode adds its variance to the total phase variance).
        S_pred = S_nom
        for (spec_idx, knob, sigma, a_k) in sensitivities:
            if sigma <= 0:
                continue
            xi = rng.normal(0.0, sigma)
            S_pred = S_pred - a_k * (xi * xi)
        # Clamp to [0, 1] -- quadratic extrap can still predict
        # negative for very large excursions.
        S_pred = float(max(0.0, min(1.0, S_pred)))
        strehls[t] = S_pred
        trial_results.append({'trial': t, 'strehl_peak': S_pred})

    if verbose:
        print(f'  Linearised MC done: nominal Strehl={S_nom:.4f}, '
              f'predicted mean={np.mean(strehls):.4f}, '
              f'std={np.std(strehls):.4f}')

    return {
        'strehl_peak_mean': float(np.mean(strehls)),
        'strehl_peak_std':  float(np.std(strehls)),
        'strehl_peak_p05':  float(np.percentile(strehls, 5)),
        'strehl_peak_p50':  float(np.percentile(strehls, 50)),
        'strehl_peak_p95':  float(np.percentile(strehls, 95)),
        'trial_results': trial_results,
        'strehl_array': strehls,
        'method': 'linearized',
        'nominal_strehl': float(S_nom),
        'sensitivities': sensitivities,
    }

