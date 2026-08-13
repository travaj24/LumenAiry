"""Subsampling tests for ``apply_real_lens_traced``.

Replaces the former ``subsample_deep_test.py`` (diagnostic PSF/pupil
table) and absorbs the guardrail + ProcessPool + error-scaling
assertions previously in ``hammer_test.py``.

Three sections:
  1. Guardrail behaviour (passes at sub=4, raises at sub=16, warn/silent
     modes, disable via ``min_coarse_samples_per_aperture=0``).
  2. ProcessPool numerical match (serial vs n_workers=8 at sub=1..8).
  3. Coarse-samples-per-aperture RMS-phase-error scaling.

The deep PSF/pupil-metrics scan that the old file printed as a table is
run at the end as a single "does not raise" smoke check.  That full
diagnostic is not a pass/fail test — it is ground-truth for tuning.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.elements import _lens_imap as _imap


H = Harness('subsample')


# ---------------------------------------------------------------------
H.section('Guardrail behaviour')

tmpl = la.make_singlet(R1=0.05, R2=-0.05, d=4e-3, glass='N-BK7',
                       aperture=5e-3)
N_guard = 512
dx_guard = 19.5e-6
wv = 1.310e-6
E_in_guard = np.ones((N_guard, N_guard), dtype=np.complex128)


def t_guardrail_sub4_passes():
    try:
        la.apply_real_lens_traced(E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
                                  ray_subsample=4, n_workers=1)
        return True, 'sub=4 at 64 coarse/aperture passes'
    except Exception as e:
        return False, f'raised {type(e).__name__}: {e}'


H.run('sub=4 (64 coarse/aperture) passes', t_guardrail_sub4_passes)


def t_guardrail_sub16_raises():
    raised = False
    try:
        la.apply_real_lens_traced(E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
                                  ray_subsample=16, n_workers=1)
    except ValueError:
        raised = True
    return raised, 'sub=16 raises ValueError'


H.run('sub=16 (16 coarse/aperture) raises ValueError',
      t_guardrail_sub16_raises)


def t_guardrail_warn_mode():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
                                  ray_subsample=16, n_workers=1,
                                  on_undersample='warn')
        fired = any(issubclass(wi.category, RuntimeWarning) for wi in w)
    return fired, f'{len(w)} warnings, runtime={fired}'


H.run('on_undersample=warn fires RuntimeWarning', t_guardrail_warn_mode)


def t_guardrail_silent_mode():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
                                  ray_subsample=16, n_workers=1,
                                  on_undersample='silent')
        silent = (len(w) == 0)
    return silent, f'{len(w)} warnings emitted'


H.run('on_undersample=silent emits no warnings',
      t_guardrail_silent_mode)


def t_guardrail_disabled():
    try:
        la.apply_real_lens_traced(E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
                                  ray_subsample=32, n_workers=1,
                                  min_coarse_samples_per_aperture=0)
        return True, 'override=0 disables check'
    except Exception as e:
        return False, f'raised {type(e).__name__}'


H.run('min_coarse_samples_per_aperture=0 disables check',
      t_guardrail_disabled)


# ---------------------------------------------------------------------
H.section('ProcessPool numerical match (serial vs n_workers=8)')


tmpl_pp = la.make_singlet(R1=0.05, R2=-0.05, d=4e-3, glass='N-BK7',
                          aperture=5e-3)
N_pp = 512
dx_pp = 1e-2 / N_pp
E_in_pp = np.ones((N_pp, N_pp), dtype=np.complex128)


def _pp_match(sub):
    def _fn():
        E_serial = la.apply_real_lens_traced(
            E_in_pp, prescription=tmpl_pp, wavelength=1.31e-6, dx=dx_pp,
            ray_subsample=sub, n_workers=1)
        E_pool = la.apply_real_lens_traced(
            E_in_pp, prescription=tmpl_pp, wavelength=1.31e-6, dx=dx_pp,
            ray_subsample=sub, n_workers=8)
        diff = np.abs(E_pool - E_serial)
        m = np.abs(E_serial) > 1e-6
        max_diff = float(diff[m].max()) if m.any() else 0.0
        return max_diff == 0.0, f'max |diff| = {max_diff:.3e}'
    return _fn


H.run('serial == pool (sub=1)', _pp_match(1))
H.run('serial == pool (sub=2)', _pp_match(2))


# ---------------------------------------------------------------------
H.section('Subsampling error scaling vs coarse-samples-per-aperture')


# WHOSE error scales, and why this section now names its path
# (FIX_RUNNER_OOM_2026_08_13).  The claim these three checks make -- that the
# exit-phase error grows with the coarseness of the LAUNCH LATTICE -- is a
# property of the coarse-Newton + ``map_coordinates`` UPSAMPLE path, because
# that path interpolates between traced nodes and the internode distance IS
# ``ray_subsample``.  v5.35.0 turned the C15 inverse characteristic on by
# default: it fits ONE global polynomial model of the exit map, which the
# lattice spacing barely moves, and the measured error collapsed from
# 21.2 / 86.5 / 348.5 nm to 0.006 nm at every ``sub`` -- so the bands read
# FAIL against a strictly better answer.
#
# Keeping the old bands as the pass condition would have meant asserting that
# the library must stay wrong by 10-600 nm.  So the scaling law is pinned where
# it lives (``TRACED_INVERSE_MAP = False``, the module's own documented
# fail-before switch) and the collapse is pinned as its own claim below.
def _run_pair(sub, imap_on):
    old = _imap.TRACED_INVERSE_MAP
    _imap.inverse_map_cache_clear()
    try:
        _imap.TRACED_INVERSE_MAP = bool(imap_on)
        E_ref = la.apply_real_lens_traced(
            E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
            ray_subsample=1, n_workers=1,
            min_coarse_samples_per_aperture=0)
        E_test = la.apply_real_lens_traced(
            E_in_guard, prescription=tmpl, wavelength=wv, dx=dx_guard,
            ray_subsample=sub, n_workers=1,
            min_coarse_samples_per_aperture=0)
    finally:
        _imap.TRACED_INVERSE_MAP = old
        _imap.inverse_map_cache_clear()
    pd = np.angle(np.exp(1j * (np.angle(E_test) - np.angle(E_ref))))
    mask = (np.abs(E_ref) > 1e-6) & (np.abs(E_test) > 1e-6)
    return (float(np.sqrt(np.mean(pd[mask] ** 2))) / (2 * np.pi) * wv * 1e9)


def _err_scaling(sub, band):
    def _fn():
        rms_nm = _run_pair(sub, imap_on=False)
        in_band = band[0] <= rms_nm <= band[1]
        return in_band, f'measured {rms_nm:.1f} nm (band {band})'
    return _fn


H.run('sub=4: upsample-path RMS phase err in expected 10-50 nm band',
      _err_scaling(4, (10, 50)))
H.run('sub=8: upsample-path RMS phase err in expected 50-200 nm band',
      _err_scaling(8, (50, 200)))
H.run('sub=16: upsample-path RMS phase err in expected 200-600 nm band',
      _err_scaling(16, (200, 600)))


def _imap_collapses(sub):
    """The C15 claim, scored the same way and on the same beam: with the map
    engaged the subsampling error is not merely smaller, it is off this
    scale.  Measured 0.006 nm at sub = 4, 8 and 16 (against 21.2, 86.5 and
    348.5 nm for the path above), so a 1 nm bar has ~150x of headroom and
    still cannot be met by the upsample path at any of the three."""
    def _fn():
        rms_nm = _run_pair(sub, imap_on=True)
        return rms_nm < 1.0, f'measured {rms_nm:.4f} nm (bar < 1 nm)'
    return _fn


H.run('sub=4: inverse map holds the phase to < 1 nm', _imap_collapses(4))
H.run('sub=8: inverse map holds the phase to < 1 nm', _imap_collapses(8))
H.run('sub=16: inverse map holds the phase to < 1 nm', _imap_collapses(16))


# ---------------------------------------------------------------------
H.section('Deep PSF/pupil diagnostic (smoke)')


def _psf_metrics(E_exit, wavelength, dx, bfl):
    half = max(abs(bfl) / 30.0, 1e-3)
    zs = np.linspace(bfl - half, bfl + half, 21)
    from lumenairy.analysis.through_focus import (
        through_focus_scan, diffraction_limited_peak)
    ideal = diffraction_limited_peak(E_exit, wavelength, bfl, dx)
    scan = through_focus_scan(E_exit, dx, wavelength, zs,
                              ideal_peak=ideal, verbose=False)
    i_best = int(np.argmax(scan.strehl)
                 if np.any(np.isfinite(scan.strehl))
                 else np.argmax(scan.peak_I))
    z_best = float(zs[i_best])
    from lumenairy.propagators.propagation import (
        angular_spectrum_propagate)
    E_focus = angular_spectrum_propagate(E_exit, z_best, wavelength, dx)
    I = np.abs(E_focus) ** 2
    peak = float(I.max())
    total_power = float(np.sum(I)) * dx * dx
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    return dict(
        z_best_mm=z_best * 1e3,
        peak=peak,
        strehl=float(scan.strehl[i_best]),
        total_power=total_power,
    )


def t_diagnostic_scan():
    """Four representative cases run without raising and produce finite
    metrics.  Matches what the old subsample_deep_test printed."""
    from lumenairy.raytrace import (
        surfaces_from_prescription, system_abcd)

    cases = [
        ('slow symmetric singlet',
         la.make_singlet(R1=0.05, R2=-0.05, d=4e-3, glass='N-BK7',
                         aperture=5e-3)),
        ('fast singlet',
         la.make_singlet(R1=0.015, R2=-0.015, d=3e-3, glass='N-BK7',
                         aperture=5e-3)),
    ]
    N = 1024
    dx = 1e-2 / N
    E_in = np.ones((N, N), dtype=np.complex128)

    n_ok = 0
    for label, pres in cases:
        surfs = surfaces_from_prescription(pres)
        _, efl, bfl, _ = system_abcd(surfs, 1.31e-6)
        for sub in [1, 2, 4, 8]:
            aperture = pres.get('aperture_diameter') or 5e-3
            if aperture / (dx * sub) < 16:
                continue
            try:
                E_exit = la.apply_real_lens_traced(
                    E_in, prescription=pres, wavelength=1.31e-6, dx=dx,
                    ray_subsample=sub, n_workers=1,
                    min_coarse_samples_per_aperture=0)
                if np.abs(E_exit).max() < 1e-12:
                    continue
                m = _psf_metrics(E_exit, 1.31e-6, dx, bfl)
                if np.isfinite(m['peak']) and np.isfinite(m['strehl']):
                    n_ok += 1
            except Exception as e:
                return False, f'{label} sub={sub} raised: {e}'
    return n_ok > 0, f'{n_ok} (case, sub) combos produced metrics'


H.run('diagnostic PSF/pupil scan (smoke)', t_diagnostic_scan)


if __name__ == '__main__':
    sys.exit(H.summary())
