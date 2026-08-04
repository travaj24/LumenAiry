"""Niche D8 -- congruence-level process parallelism in
:func:`propagate_traced_carrier_chain_multi`.

WHAT THIS PINS.  The K congruence chains are independent until recombination,
so distributing them must not change the answer.  The contract is stronger
than "close": the guards and the accumulation stay in the parent and run in
ASCENDING ``k``, so the complex sum is formed in the same order as the serial
path and the recombined field is **FP-identical**.

WHY PROCESSES.  A single congruence is serial by design on the shipped path
(``apply_real_lens_traced``'s ``n_workers`` is a documented no-op for
``newton_fit='polynomial'``).  MEASURED on design 121's post-DOE chain, 2
congruences at N=1024, paraxial leg: serial 318.8 s vs ``ThreadPoolExecutor(2)``
254.3 s = 1.25x -- GIL-bound.  Hence a process pool.

The fixtures here are deliberately tiny (a single thin-ish group, N=128) so the
file runs in seconds: the property under test is scheduling equivalence, which
is grid-size independent.  The physics is pinned elsewhere (niche D2/D6).
"""
import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.carrier import (
    _multi_apply_worker_state,
    _multi_capture_worker_state,
    _multi_looks_like_spawn_bootstrap,
    _multi_resolve_workers,
    propagate_traced_carrier_chain_multi,
)

_WL = 1.31e-6
_N = 128
_DX = 4.0e-6


def _presc():
    """One weak biconvex group -- enough to exercise the traced path."""
    return {
        'name': 'D8 test group',
        'surfaces': [
            {'radius': 60.0e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
             'conic': 0.0, 'semi_diameter': 6.0e-3},
            {'radius': -60.0e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
             'conic': 0.0, 'semi_diameter': 6.0e-3},
        ],
        'thicknesses': [2.0e-3],
    }


def _field():
    x = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(x, x, indexing='xy')
    w = 12.0 * _DX
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def _congruences(k=3):
    """K congruences sharing ONE field object (the fan case), tilted apart."""
    fld = _field()
    out = []
    for i in range(k):
        L = (i - (k - 1) / 2.0) * 2.0e-3
        out.append({'field': fld, 'name': f'c{i}', 'weight': 1.0 + 0.0j,
                    'carrier': la.TiltedCarrier(np.inf, L, 0.0)})
    return out


def _run(workers):
    return propagate_traced_carrier_chain_multi(
        _congruences(), [{'prescription': _presc(), 'gap_before': 5.0e-3}],
        _WL, _DX,
        output_grid=dict(dx_out=2.0e-6, N_out=128),
        readout_tile=64, on_replica='ignore', on_readout_clip='ignore',
        final_distance=30.0e-3, ray_subsample=4, final_leg='paraxial',
        on_multi_congruence='ignore', on_na_proximity='ignore',
        on_decentred_fit='ignore', on_gap_paraxial='ignore',
        on_tilt_exact_grid='ignore',
        congruence_workers=workers)


# --------------------------------------------------------------------------
# The headline: parallel == serial, exactly.
# --------------------------------------------------------------------------
def test_parallel_recombined_field_is_fp_identical_to_serial():
    a = _run(None)
    b = _run(2)
    assert a.field.shape == b.field.shape
    assert np.array_equal(a.field, b.field), (
        "congruence_workers>1 changed the recombined field; the accumulation "
        "is supposed to run in ascending k in the parent either way "
        f"(max|d| = {np.abs(a.field - b.field).max():.3e})")


def test_parallel_preserves_per_congruence_records_in_order():
    a = _run(None)
    b = _run(2)
    assert [c['name'] for c in a.congruences] == \
           [c['name'] for c in b.congruences], \
        "worker completion order leaked into the congruence record order"
    for ca, cb in zip(a.congruences, b.congruences):
        for key in ('throughput', 'capture', 'power_out', 'power_exit'):
            if key in ca and np.isfinite(ca[key]):
                assert ca[key] == pytest.approx(cb[key], rel=0, abs=0), \
                    f"{key} moved for congruence {ca['name']}"


def test_workers_equal_one_is_the_serial_path():
    a = _run(None)
    b = _run(1)
    assert np.array_equal(a.field, b.field)


# --------------------------------------------------------------------------
# The clamp: over-subscription must be refused, not OOM the box.
# --------------------------------------------------------------------------
def test_worker_count_is_clamped_by_available_ram_and_reports_it():
    # 8192^2 complex128 -> ~1.07 GB/grid, ~23.6 GB per worker at the measured
    # factor, so a 64-worker request cannot be honoured on any real box.
    with pytest.warns(RuntimeWarning, match='congruence_workers'):
        n = _multi_resolve_workers(64, 64, (8192, 8192), 8.0, 'fn')
    assert n >= 1
    assert n < 64, "the RAM clamp did not reduce an impossible request"


def test_worker_count_never_exceeds_the_congruence_count():
    assert _multi_resolve_workers(16, 3, (128, 128), 0.0, 'fn') <= 3


def test_none_and_one_resolve_to_serial():
    assert _multi_resolve_workers(None, 8, (128, 128), 0.0, 'fn') == 1
    assert _multi_resolve_workers(1, 8, (128, 128), 0.0, 'fn') == 1


def test_zero_or_negative_workers_is_refused():
    with pytest.raises(ValueError, match='congruence_workers'):
        _multi_resolve_workers(0, 4, (128, 128), 0.0, 'fn')
    with pytest.raises(ValueError, match='congruence_workers'):
        _multi_resolve_workers(-2, 4, (128, 128), 0.0, 'fn')


# --------------------------------------------------------------------------
# The spawn trap: a driver whose __main__ is not import-safe.  pytest's own
# __main__ IS import-safe, so this class of failure cannot be reproduced by
# calling the orchestrator here -- it is pinned on the PREDICATE that selects
# the actionable message, which is the part that regressed in review.
# --------------------------------------------------------------------------
def test_spawn_bootstrap_is_recognised_from_the_multiprocessing_message():
    exc = RuntimeError(
        "\n        An attempt has been made to start a new process before the\n"
        "        current process has finished its bootstrapping phase.\n")
    assert _multi_looks_like_spawn_bootstrap(exc)


def test_spawn_bootstrap_is_recognised_through_a_cause_chain():
    inner = RuntimeError('... Safe importing of main module ...')
    outer = OSError('worker died')
    outer.__cause__ = inner
    assert _multi_looks_like_spawn_bootstrap(outer)


def test_freeze_support_wording_is_recognised():
    assert _multi_looks_like_spawn_bootstrap(
        RuntimeError('you have forgotten to use freeze_support()'))


def test_an_ordinary_worker_error_is_not_mistaken_for_the_spawn_trap():
    assert not _multi_looks_like_spawn_bootstrap(
        ValueError('prescription surfaces[0] radius must be finite'))


def test_a_self_referential_cause_chain_terminates():
    a = RuntimeError('a')
    b = RuntimeError('b')
    a.__cause__ = b
    b.__cause__ = a
    assert not _multi_looks_like_spawn_bootstrap(a)


# --------------------------------------------------------------------------
# Worker state inheritance.  A spawned worker imports lumenairy fresh, so
# runtime-registered materials and runtime-set behaviour flags do not exist
# there unless carried across.  The glass case CRASHES (loud); the flag case
# would silently compute different physics, which is why both are pinned.
# --------------------------------------------------------------------------
def test_snapshot_captures_runtime_registered_glass():
    from lumenairy import glass as g
    name = '__d8_probe_glass__'
    g.SELLMEIER_COEFFICIENTS[name] = ((1.0, 0.01), (1.0, 0.01), (1.0, 100.0))
    g.GLASS_REGISTRY[name] = '__sellmeier__'
    try:
        snap = _multi_capture_worker_state()
        assert name in snap['glass']['SELLMEIER_COEFFICIENTS']
        assert name in snap['glass']['GLASS_REGISTRY']
    finally:
        g.SELLMEIER_COEFFICIENTS.pop(name, None)
        g.GLASS_REGISTRY.pop(name, None)


def test_apply_restores_glass_into_the_SAME_dict_object():
    from lumenairy import glass as g
    name = '__d8_probe_glass2__'
    ident = id(g.GLASS_REGISTRY)
    _multi_apply_worker_state(
        {'glass': {'GLASS_REGISTRY': {name: '__sellmeier__'}}})
    try:
        assert g.GLASS_REGISTRY.get(name) == '__sellmeier__'
        # other modules hold this dict BY REFERENCE; rebinding would strand them
        assert id(g.GLASS_REGISTRY) == ident
    finally:
        g.GLASS_REGISTRY.pop(name, None)


def test_snapshot_captures_traced_behaviour_flags():
    from lumenairy.elements import _lens_traced as lt
    snap = _multi_capture_worker_state()
    key = 'lumenairy.elements._lens_traced:DECENTRED_FIT_ARBITER'
    assert key in snap['flags']
    assert snap['flags'][key] == lt.DECENTRED_FIT_ARBITER


def test_apply_round_trips_a_flipped_flag():
    from lumenairy.elements import _lens_traced as lt
    key = 'lumenairy.elements._lens_traced:DECENTRED_FIT_ARBITER'
    original = lt.DECENTRED_FIT_ARBITER
    try:
        lt.DECENTRED_FIT_ARBITER = not original
        snap = _multi_capture_worker_state()
        assert snap['flags'][key] == (not original)
        lt.DECENTRED_FIT_ARBITER = original          # worker starts "fresh"
        _multi_apply_worker_state(snap)
        assert lt.DECENTRED_FIT_ARBITER == (not original), \
            "a worker would have run different physics from the parent"
    finally:
        lt.DECENTRED_FIT_ARBITER = original


def test_snapshot_is_picklable():
    import pickle
    snap = _multi_capture_worker_state()
    assert pickle.loads(pickle.dumps(snap))['flags'] == snap['flags']


def test_apply_tolerates_an_empty_or_missing_snapshot():
    _multi_apply_worker_state(None)
    _multi_apply_worker_state({})


def test_ram_budget_is_divided_among_workers_not_copied():
    # get_ram_budget() falls back to GLOBAL free memory and feeds both the
    # readout's fine-grid sizing and the parallel_amp gate, so handing each
    # worker the whole-box figure is K-fold oversubscription.
    one = _multi_capture_worker_state(1)['ram_budget']
    four = _multi_capture_worker_state(4)['ram_budget']
    assert one is not None and four is not None
    assert four <= one // 4 + 1, \
        f"worker budget not divided: {one} -> {four} at 4 workers"


def test_ram_budget_division_is_safe_at_degenerate_worker_counts():
    for n in (0, -3, None):
        try:
            v = _multi_capture_worker_state(n if n is not None else 1)
        except Exception as exc:                   # pragma: no cover
            raise AssertionError(f"n_workers={n!r} raised {exc!r}")
        assert v['ram_budget'] is None or v['ram_budget'] > 0


def test_clamp_prices_the_exact_legs_fine_grid_not_just_the_chain():
    # The fine grid is a SECOND per-worker peak, live at the same time as the
    # chain working set.  Pricing the chain alone let 3 workers each correctly
    # afford a 16384^2 grid and collectively ask for 123 GB of a 127 GB box.
    chain_only = _multi_resolve_workers(32, 32, (8192, 8192), 8.0, 'fn')
    with_readout = _multi_resolve_workers(32, 32, (8192, 8192), 8.0, 'fn',
                                          n_fine_cap=16384)
    assert with_readout <= chain_only, (
        "adding the readout's 17 GB fine grid must not INCREASE the worker "
        f"count ({chain_only} -> {with_readout})")


def test_a_paraxial_readout_is_not_charged_for_a_fine_grid():
    # final_leg='paraxial' builds no fine grid, so n_fine_cap=0 must price
    # the same as omitting it entirely.
    assert (_multi_resolve_workers(8, 8, (4096, 4096), 0.0, 'fn', n_fine_cap=0)
            == _multi_resolve_workers(8, 8, (4096, 4096), 0.0, 'fn'))


def test_a_bigger_fine_grid_cap_costs_workers():
    small = _multi_resolve_workers(32, 32, (4096, 4096), 0.0, 'fn',
                                   n_fine_cap=4096)
    big = _multi_resolve_workers(32, 32, (4096, 4096), 0.0, 'fn',
                                 n_fine_cap=16384)
    assert big <= small
