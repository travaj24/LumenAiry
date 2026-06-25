"""Tests for the v5.16.1 system-level memory estimator + autodetect guardrail
(lumenairy.memory: estimate_lens_memory / estimate_asm_memory /
estimate_sim_memory / check_sim_memory / set_low_memory).

The estimator is calibrated to MEASURED peak RSS (system-wide, incl. the Newton
worker pool) on a 137 GB / 24-core box for design-119's biggest relay lens
(apply_real_lens_traced, parallel_amp=False):

    N=16384 c64  sub=8  -> 44.5 GB    N=16384 c64  sub=16 -> 37.2 GB
    N=16384 c128 sub=8  -> 57.3 GB    N=24576 c64  sub=8  -> 83.6 GB

and the verdict is validated against a guard-killed N=32768/sub=16 run that hit
135.8 GB on the same box (i.e. did NOT fit).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy import memory as m

GB = 1e9

# (N, complex_dtype, ray_subsample, measured_GB)
_ANCHORS = [
    (16384, 'complex64', 8, 44.5),
    (16384, 'complex64', 16, 37.2),
    (16384, 'complex128', 8, 57.3),
    (24576, 'complex64', 8, 83.6),
]


@pytest.mark.parametrize("N,dt,sub,meas", _ANCHORS)
def test_estimate_lens_matches_measured_anchors(N, dt, sub, meas):
    """The traced-lens estimate is within a conservative band of the measured
    peak: exact at the 16384 calibration points, a conservative UPPER bound at
    larger N (the N^2 form over-predicts there -- the fail-safe direction)."""
    est_gb = m.estimate_lens_memory(N, dt, ray_subsample=sub,
                                    parallel_amp=False) / GB
    ratio = est_gb / meas
    # 16384 anchors: tight (+-12%). 24576: allowed to over-predict up to +30%.
    hi = 1.30 if N >= 24576 else 1.12
    assert 0.88 <= ratio <= hi, f"N={N} {dt} sub={sub}: est {est_gb:.1f} GB vs measured {meas} GB (ratio {ratio:.2f})"


def test_complex64_only_saves_the_complex_part():
    """complex64 must NOT shrink the float64 geometric core -- only the
    complex fields + the complex128 phase_exp transient. So the c64->c128
    delta equals the complex terms, and the float64 core is identical."""
    i64 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                 parallel_amp=False, itemized=True)
    i128 = m.estimate_lens_memory(16384, 'complex128', ray_subsample=8,
                                  parallel_amp=False, itemized=True)
    assert i64['items']['float64_geometric_core'] == i128['items']['float64_geometric_core']
    assert i64['items']['newton_coarse_solve'] == i128['items']['newton_coarse_solve']
    # complex fields double; the phase_exp transient is complex128 in BOTH.
    assert i128['items']['complex_fields'] == 2 * i64['items']['complex_fields']
    assert i64['items']['phase_exp_c128_transient'] == i128['items']['phase_exp_c128_transient']


def test_subsample_reduces_only_newton_term():
    """Coarser subsample shrinks the Newton coarse solve (quadratically) and
    nothing else."""
    s8 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                parallel_amp=False, itemized=True)
    s16 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                 parallel_amp=False, itemized=True)
    assert s8['items']['float64_geometric_core'] == s16['items']['float64_geometric_core']
    # newton ~ (N/sub)^2 -> sub doubling quarters it
    assert s16['items']['newton_coarse_solve'] == pytest.approx(
        s8['items']['newton_coarse_solve'] / 4.0, rel=1e-9)


def test_parallel_amp_doubles_working_set():
    off = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8, parallel_amp=False)
    on = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8, parallel_amp=True)
    assert on > 1.5 * off  # ~2x the f64-core + complex; newton/trap once


def test_estimate_sim_memory_itemized_structure():
    d = m.estimate_sim_memory(16384, 'complex64', ray_subsample=8,
                              parallel_amp=False, itemized=True)
    assert set(['peak_bytes', 'raw_bytes', 'driving_step', 'lens_step_bytes',
                'asm_step_bytes', 'lens_items']).issubset(d.keys())
    assert d['driving_step'] == 'lens'         # lens dominates the bare ASM step
    assert d['peak_bytes'] == pytest.approx(d['raw_bytes'] * d['safety_factor'], rel=1e-9)


def test_guardrail_predicts_32768_oom_and_recommends_24576():
    """The guard-killed N=32768/sub=16 run (135.8 GB, did not fit ~117 GB
    available) must be predicted as INSUFFICIENT, with a claw-back that fits."""
    avail = 117 * GB
    v = m.check_sim_memory(32768, 'complex64', ray_subsample=16,
                           parallel_amp=False, available=avail,
                           mode='silent', verbose=False)
    assert v['fits'] is False
    assert any('24576' in r['change'] for r in v['recommendations'])
    # and every recommendation it offers genuinely fits
    assert all(r['peak_bytes'] <= avail for r in v['recommendations'])


def test_guardrail_24576_fits():
    avail = 117 * GB
    v = m.check_sim_memory(24576, 'complex64', ray_subsample=8,
                           parallel_amp=False, available=avail,
                           mode='silent', verbose=False)
    assert v['fits'] is True
    assert v['recommendations'] == []


def test_guardrail_raise_mode():
    with pytest.raises(MemoryError):
        m.check_sim_memory(32768, 'complex128', ray_subsample=8,
                           parallel_amp=True, available=80 * GB,
                           mode='raise', verbose=False)


def test_estimate_op_memory_backward_compat():
    """The legacy estimator keeps its bare-int contract."""
    val = la.estimate_op_memory((4096, 4096), 'complex128', n_work_arrays=3)
    assert isinstance(val, int)
    assert val == 3 * 4096 * 4096 * 16


def test_set_low_memory_roundtrip_and_byte_safe_defaults():
    """Shipped defaults are unchanged; set_low_memory flips the lean knobs and
    set_low_memory(False) restores them exactly."""
    assert la.get_fft_plan_cache_size() == 8
    assert la.get_lens_parallel_amp() is True
    prior = la.set_low_memory(True)
    try:
        assert la.get_fft_plan_cache_size() == 2
        assert la.get_lens_parallel_amp() is False
        assert prior['plan_cache_size'] == 8
        assert prior['lens_parallel_amp'] is True
    finally:
        la.set_low_memory(False)
    assert la.get_fft_plan_cache_size() == 8
    assert la.get_lens_parallel_amp() is True


def test_parallel_amp_default_resolution():
    """parallel_amp=None resolves to the module global, so set_lens_parallel_amp
    flips the default for callers that don't pass the kwarg -- the mechanism the
    engine/consumer relies on, without changing the shipped default."""
    from lumenairy.elements import _lens_traced as lt
    assert lt._LENS_PARALLEL_AMP_DEFAULT is True   # shipped default
    la.set_lens_parallel_amp(False)
    try:
        assert lt._LENS_PARALLEL_AMP_DEFAULT is False
        assert la.get_lens_parallel_amp() is False
    finally:
        la.set_lens_parallel_amp(True)
    assert la.get_lens_parallel_amp() is True
