"""Tests for the v5.16.1 system-level memory estimator + autodetect guardrail
(lumenairy.memory: estimate_lens_memory / estimate_asm_memory /
estimate_sim_memory / check_sim_memory / set_low_memory).

The estimator is calibrated to MEASURED peak RSS (system-wide, incl. the Newton
worker pool) on a 137 GB / 24-core box for design-119's biggest relay lens
(apply_real_lens_traced, parallel_amp=False).  The v5.17.1 recalibration (see
the anchor table in ``lumenairy/memory.py``) fits the 3-parameter model EXACTLY
to the whole-grid deltas measured AFTER the v4.10 tilt-check leak was fixed:

    N=16384 c64  sub=8  -> 29.69 GB   N=16384 c64  sub=16 -> 21.87 GB
    N=16384 c128 sub=8  -> 31.39 GB   (chunked c128 sub=16 -> 26.30 GB)

(The pre-v5.17.0 anchors 44.5/37.2/57.3 GB reflected that since-fixed leak and
survive only in memory.py's history comment.)
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy import memory as m

GB = 1e9

# (N, complex_dtype, ray_subsample, measured_GB)
_ANCHORS = [
    (16384, 'complex64', 8, 29.69),
    (16384, 'complex64', 16, 21.87),
    (16384, 'complex128', 8, 31.39),
]


@pytest.mark.parametrize("N,dt,sub,meas", _ANCHORS)
def test_estimate_lens_matches_measured_anchors(N, dt, sub, meas):
    """The traced-lens estimate is within a conservative band of the measured
    peak: exact at the 16384 calibration points, a conservative UPPER bound at
    larger N (the N^2 form over-predicts there -- the fail-safe direction)."""
    est_gb = m.estimate_lens_memory(N, dt, ray_subsample=sub,
                                    parallel_amp=False,
                                    sag_chunk_rows=0) / GB
    ratio = est_gb / meas
    # 16384 anchors: tight (+-12%). 24576: allowed to over-predict up to +30%.
    hi = 1.30 if N >= 24576 else 1.12
    assert 0.88 <= ratio <= hi, f"N={N} {dt} sub={sub}: est {est_gb:.1f} GB vs measured {meas} GB (ratio {ratio:.2f})"


def test_complex64_only_saves_the_complex_part():
    """complex64 must NOT shrink the float64 geometric core -- only the
    complex fields + the complex128 phase_exp transient. So the c64->c128
    delta equals the complex terms, and the float64 core is identical."""
    i64 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                 parallel_amp=False, sag_chunk_rows=0,
                                 itemized=True)
    i128 = m.estimate_lens_memory(16384, 'complex128', ray_subsample=8,
                                  parallel_amp=False, sag_chunk_rows=0,
                                  itemized=True)
    assert i64['items']['float64_geometric_core'] == i128['items']['float64_geometric_core']
    assert i64['items']['newton_coarse_solve'] == i128['items']['newton_coarse_solve']
    # complex fields double; the phase_exp transient is complex128 in BOTH.
    assert i128['items']['complex_fields'] == 2 * i64['items']['complex_fields']
    assert i64['items']['phase_exp_c128_transient'] == i128['items']['phase_exp_c128_transient']


def test_subsample_reduces_only_newton_term():
    """Coarser subsample shrinks the Newton coarse solve (quadratically) and
    nothing else."""
    s8 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                parallel_amp=False, sag_chunk_rows=0,
                                itemized=True)
    s16 = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                 parallel_amp=False, sag_chunk_rows=0,
                                 itemized=True)
    assert s8['items']['float64_geometric_core'] == s16['items']['float64_geometric_core']
    # newton ~ (N/sub)^2 -> sub doubling quarters it
    assert s16['items']['newton_coarse_solve'] == pytest.approx(
        s8['items']['newton_coarse_solve'] / 4.0, rel=1e-9)


def test_parallel_amp_doubles_working_set():
    off = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                 parallel_amp=False, sag_chunk_rows=0)
    on = m.estimate_lens_memory(16384, 'complex64', ray_subsample=8,
                                parallel_amp=True, sag_chunk_rows=0)
    assert on > 1.5 * off  # ~2x the f64-core + complex; newton/trap once


def test_estimate_sim_memory_itemized_structure():
    d = m.estimate_sim_memory(16384, 'complex64', ray_subsample=8,
                              parallel_amp=False, itemized=True)
    assert set(['peak_bytes', 'raw_bytes', 'driving_step', 'lens_step_bytes',
                'asm_step_bytes', 'lens_items']).issubset(d.keys())
    assert d['driving_step'] == 'lens'         # lens dominates the bare ASM step
    assert d['peak_bytes'] == pytest.approx(d['raw_bytes'] * d['safety_factor'], rel=1e-9)


def test_guardrail_predicts_32768_oom_and_recommends_24576():
    """A whole-grid (chunking explicitly disabled) N=32768/sub=16 c64 run
    must be predicted INSUFFICIENT on an 80 GB budget (post-fix whole-grid
    estimate ~100 GB), with claw-backs that fit -- led by re-enabling the
    byte-identical row-band mode."""
    avail = 80 * GB
    v = m.check_sim_memory(32768, 'complex64', ray_subsample=16,
                           parallel_amp=False, sag_chunk_rows=0,
                           available=avail, mode='silent', verbose=False)
    assert v['fits'] is False
    # v5.17.0: the explicit whole-grid refusal must recommend re-enabling
    # the byte-identical row-band mode first
    assert any('sag_chunk_rows' in r['change'] for r in v['recommendations'])
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


# ---------------------------------------------------------------------------
# Wave-2 audit fixes (AUDIT_V5_17_0_2026_07_01_DEEP.md, memory-knobs cluster)
# ---------------------------------------------------------------------------

def test_set_max_ram_governs_pick_batch_and_should_split():
    """audit P2-21: pick_batch_size / should_split must honour a pinned
    set_max_ram() budget through get_ram_budget() (pre-fix they read psutil
    directly, so a 1 GB pin was silently ignored)."""
    la.set_max_ram(1)   # 1 GB budget
    try:
        assert la.get_ram_budget() == 1024**3
        # 0.9 GB > safety(0.5) * 1 GB -> must split
        assert m.should_split(int(0.9 * 1024**3)) is True
        # budget 0.5 GB / 0.2 GB per item -> batch of 2, not all 100
        assert m.pick_batch_size(100, int(0.2 * 1024**3)) == 2
    finally:
        la.set_max_ram(None)
    # explicit `available` still wins over the budget
    assert m.should_split(int(0.9 * 1024**3), available=4 * 1024**3) is False


def test_parallel_amp_guard_honors_set_max_ram(monkeypatch):
    """audit P2-21 (runtime half): the apply_real_lens_traced parallel-amp
    RAM guard must fold in the pinned budget (min(psutil-free, budget)) --
    pre-fix a 1 GB pin still ran the doubled parallel working set."""
    import concurrent.futures as cf

    from lumenairy.elements._lens_traced import apply_real_lens_traced

    prefixes = []
    real_tpe = cf.ThreadPoolExecutor

    class Spy(real_tpe):
        def __init__(self, *a, **kw):
            prefixes.append(kw.get('thread_name_prefix', ''))
            super().__init__(*a, **kw)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', Spy)

    presc = {'name': 't', 'aperture_diameter': 3e-3,
             'surfaces': [
                 {'radius': 12e-3, 'conic': -0.4,
                  'glass_before': 'air', 'glass_after': 'N-BK7'},
                 {'radius': -15e-3, 'conic': 0.0,
                  'glass_before': 'N-BK7', 'glass_after': 'air'}],
             'thicknesses': [2.5e-3]}
    N, dx = 256, 12e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X**2 + Y**2) / (1.2e-3)**2).astype(np.complex128)
    kw = dict(prescription=presc, wavelength=1.31e-6, dx=dx,
              parallel_amp=True, ray_subsample=2)

    # control: tiny threshold, no pin -> parallel path engages
    apply_real_lens_traced(E.copy(), parallel_amp_min_free_gb=0.001, **kw)
    assert 'rlt_amp' in prefixes

    # pinned 1 GB budget < 2 GB threshold -> sequential, regardless of
    # how much physical RAM the box has free
    prefixes.clear()
    la.set_max_ram(1)
    try:
        apply_real_lens_traced(E.copy(), parallel_amp_min_free_gb=2.0, **kw)
    finally:
        la.set_max_ram(None)
    assert 'rlt_amp' not in prefixes


def test_chunked_estimate_accounts_for_parallel_amp_and_slant():
    """audit P2-22: the row-band estimate must model the runtime parallel
    doubling of the leg-local working set (resident complex + band
    transients) and the slant full-grid fall-through stack, while the
    calibrated parallel_amp=False anchor stays untouched."""
    # calibrated chunked anchor (c128 sub=16 N=16384, parallel off): 26.30 GB
    anchor = m.estimate_lens_memory(16384, 'complex128', ray_subsample=16,
                                    parallel_amp=False) / GB
    assert 0.88 <= anchor / 26.30 <= 1.12

    off = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                 parallel_amp=False, itemized=True)
    on = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                parallel_amp=True, itemized=True)
    # leg-local terms double; the post-legs Newton solve stays single
    assert on['items']['chunked_resident_complex'] == \
        2 * off['items']['chunked_resident_complex']
    assert on['items']['band_transients'] == 2 * off['items']['band_transients']
    assert on['items']['newton_coarse_solve'] == off['items']['newton_coarse_solve']
    assert on['total'] > 1.5 * off['total']

    # slant_correction disables the narrow per-surface chunking -> the
    # full-grid angle stack must appear in the row-band estimate too
    slant = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                   parallel_amp=False, slant_correction=True,
                                   itemized=True)
    assert slant['items']['slant_fullgrid_stack'] == \
        int(m._LENS_SLANT_F64_ARRAYS * 8 * 16384**2)
    assert slant['total'] > off['total']


def test_chunked_parallel_off_clawback_revived():
    """audit P2-22 (dead rung): with the chunked estimate now depending on
    parallel_amp, the 'set parallel_amp=False (byte-identical)' claw-back
    can genuinely surface for a default (auto-chunked, parallel) config."""
    v = m.check_sim_memory(16384, 'complex64', ray_subsample=16,
                           parallel_amp=True, available=int(25 * GB),
                           mode='silent', verbose=False)
    assert v['fits'] is False   # pre-fix: 17.5 GB estimate 'fit' 25 GB
    labels = [r['change'] for r in v['recommendations']]
    assert any('parallel_amp=False' in lbl for lbl in labels)
    assert all(r['peak_bytes'] <= 25 * GB for r in v['recommendations'])


def test_set_low_memory_aggressive_dtype_roundtrip():
    """audit P2-23: set_low_memory(False) must revert the aggressive
    complex64 default-dtype flip (pre-fix it silently persisted)."""
    assert la.get_default_complex_dtype() == np.complex128
    try:
        with pytest.warns(RuntimeWarning):
            prior = la.set_low_memory(True, aggressive=True)
        assert prior['complex_dtype'] == np.complex128
        assert la.get_default_complex_dtype() == np.complex64
        la.set_low_memory(False)
        assert la.get_default_complex_dtype() == np.complex128
    finally:
        la.set_default_complex_dtype(np.complex128)
        la.set_low_memory(False)   # idempotent: no stash -> shipped defaults


def test_set_low_memory_restores_user_customizations():
    """audit P3-46: priors are captured from the LIVE getters (auto-promote
    was hardcoded True) and set_low_memory(False) restores the captured
    values, not shipped defaults."""
    la.set_fft_auto_promote(False)      # e.g. byte-reproducibility pin
    la.set_fft_plan_cache_size(16)
    try:
        prior = la.set_low_memory(True)
        assert prior['fft_auto_promote'] is False
        assert prior['plan_cache_size'] == 16
        assert la.get_fft_auto_promote() is False   # low-mem sets it False too
        assert la.get_fft_plan_cache_size() == 2
        la.set_low_memory(False)
        assert la.get_fft_auto_promote() is False   # user pin survives
        assert la.get_fft_plan_cache_size() == 16   # user size survives
    finally:
        # audit W9: the shipped default is False since v5.30.1 (auto-promote
        # is opt-in), so restoring True here would leak the non-reproducible
        # planner into the rest of the worker.
        la.set_fft_auto_promote(False)
        la.set_fft_plan_cache_size(8)


def test_set_low_memory_repeated_enable_keeps_first_prior():
    """A second set_low_memory(True) must not overwrite the true prior with
    low-memory values; disable still restores the original settings."""
    assert la.get_fft_plan_cache_size() == 8
    try:
        la.set_low_memory(True)
        la.set_low_memory(True)     # would capture plan_cache_size == 2
        la.set_low_memory(False)
    finally:
        pass
    assert la.get_fft_plan_cache_size() == 8
    assert la.get_lens_parallel_amp() is True


def test_complex64_clawback_label_discloses_parallel_amp():
    """audit P3-45: the complex64 rung's re-estimate assumes
    parallel_amp=False, so its label must say so (like its neighbours)."""
    v = m.check_sim_memory(16384, 'complex128', ray_subsample=8,
                           parallel_amp=True, sag_chunk_rows=0,
                           available=int(40 * GB),
                           mode='silent', verbose=False)
    assert v['fits'] is False
    labels = [r['change'] for r in v['recommendations']]
    c64 = [lbl for lbl in labels if 'complex64' in lbl]
    assert c64, f"complex64 rung missing from {labels}"
    assert all('parallel_amp=False' in lbl for lbl in c64)


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


def test_v5_17_0_chunked_is_the_auto_default():
    """sag_chunk_rows=None resolves to the row-band mode at N >= 4096: the
    default estimate equals the explicit-chunked one and is far below the
    forced whole-grid estimate; the default 32768 config now FITS."""
    dflt = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                  parallel_amp=False)
    chunked = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                     parallel_amp=False,
                                     sag_chunk_rows=max(256, 16384 // 16))
    whole = m.estimate_lens_memory(16384, 'complex64', ray_subsample=16,
                                   parallel_amp=False, sag_chunk_rows=0)
    assert dflt == chunked
    assert dflt < 0.8 * whole
    # below the auto threshold the default stays whole-grid
    small_dflt = m.estimate_lens_memory(2048, 'complex64', ray_subsample=8,
                                        parallel_amp=False)
    small_whole = m.estimate_lens_memory(2048, 'complex64', ray_subsample=8,
                                         parallel_amp=False, sag_chunk_rows=0)
    assert small_dflt == small_whole
    v = m.check_sim_memory(32768, 'complex64', ray_subsample=16,
                           parallel_amp=False, available=117 * GB,
                           mode='silent', verbose=False)
    assert v['fits'] is True
