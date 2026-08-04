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
