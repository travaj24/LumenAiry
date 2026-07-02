"""Audit v5.17.0 Wave 2 regression pins: FFT state snapshot/restore (P3-54).

_FFT_STATE_KEYS was frozen at v5.4.6 with 9 entries, so setter-backed
globals added later (set_fft_double_buffer / set_fft_fallback /
set_fft_plan_cache_size / set_asm_cache_size, plus the raw USE_SCIPY_FFT
toggle) were silently reset to library defaults in spawn workers using
the documented ``initializer=restore_fft_state`` recipe.  These tests pin:

- the snapshot now carries every setter-backed knob;
- a worker-shaped round-trip (parent snapshot -> fresh-defaults process
  -> restore) reinstates all of them;
- restore goes THROUGH the trimming/clearing setters where the setter
  side effects matter (plan-cache trim; ping-pong flip clears plans,
  identical value keeps them);
- snapshots from OLDER library versions (missing the new keys) are
  tolerated: no raise, and this process's values are left alone
  (mixed-version worker pools).
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

import lumenairy.propagators.fft_infra as fi

# The setter-backed knobs added after v5.4.6: global name -> non-default
# value used by these tests.  Defaults are captured per-test from the
# live module so the tests don't hardcode library defaults.
_NEW_KNOBS = {
    '_PYFFTW_DOUBLE_BUFFER': False,       # set_fft_double_buffer (v5.16.2)
    'PYFFTW_FALLBACK_ON_ERROR': False,    # set_fft_fallback
    'USE_SCIPY_FFT': False,               # raw toggle (peer of USE_PYFFTW)
    '_PYFFTW_PLAN_CACHE_SIZE': 3,         # set_fft_plan_cache_size
    '_H_CACHE_SIZE': 5,                   # set_asm_cache_size ...
    '_FREQ_GRID_CACHE_SIZE': 7,
    '_BANDLIMIT_CACHE_SIZE': 9,
    '_H_CACHE_MAX_BYTES_PER_ENTRY': 123456789,
    '_H_CACHE_MAX_TOTAL_BYTES': 1024**3,
}

_V546_KEYS = (
    'DEFAULT_COMPLEX_DTYPE', 'DEFAULT_REAL_DTYPE', 'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_DY', 'FFTW_THREADS', 'SCIPY_FFT_WORKERS', 'USE_PYFFTW',
    '_PYFFTW_PLAN_FLAGS', '_PYFFTW_AUTO_PROMOTE',
)


@pytest.fixture
def _pristine_fft_state():
    """Save/restore every snapshotted global (plus the plan cache) so the
    mutations below cannot leak into other tests."""
    saved = {k: getattr(fi, k) for k in fi._FFT_STATE_KEYS}
    saved_plans = OrderedDict(fi._PYFFTW_PLAN_CACHE)
    yield saved
    for k, v in saved.items():
        setattr(fi, k, v)
    with fi._PYFFTW_PLAN_LOCK:
        fi._PYFFTW_PLAN_CACHE.clear()
        fi._PYFFTW_PLAN_CACHE.update(saved_plans)


def test_snapshot_includes_later_added_setter_backed_keys(_pristine_fft_state):
    state = fi.snapshot_fft_state()
    missing = [k for k in _NEW_KNOBS if k not in state]
    assert not missing, (
        f"snapshot_fft_state omits setter-backed globals {missing}; spawn "
        f"workers would silently revert them to library defaults (P3-54)")


def test_worker_shaped_round_trip_restores_new_knobs(_pristine_fft_state):
    defaults = _pristine_fft_state
    # Parent: set every later-added knob via its public setter.
    fi.set_fft_double_buffer(False)
    fi.set_fft_fallback(False)
    fi.set_fft_plan_cache_size(3)
    fi.set_asm_cache_size(h_cache=5, freq_cache=7, bandlimit_cache=9,
                          h_max_bytes_per_entry=123456789,
                          h_max_total_bytes=1024**3)
    fi.USE_SCIPY_FFT = False
    state = fi.snapshot_fft_state()
    # Fresh spawn worker: module re-imports at library defaults ...
    for k, v in defaults.items():
        setattr(fi, k, v)
    # ... then runs the documented initializer.
    fi.restore_fft_state(state)
    lost = {k: getattr(fi, k) for k, want in _NEW_KNOBS.items()
            if getattr(fi, k) != want}
    assert not lost, f"knobs reverted to defaults after restore: {lost}"


def test_restore_plan_cache_size_trims_via_setter(_pristine_fft_state):
    """Restoring a tighter plan-cache bound in a WARM process must trim
    resident plans (i.e. go through set_fft_plan_cache_size, not a raw
    global write)."""
    fi.set_fft_plan_cache_size(2)
    state = fi.snapshot_fft_state()
    fi.set_fft_plan_cache_size(8)
    with fi._PYFFTW_PLAN_LOCK:
        fi._PYFFTW_PLAN_CACHE.clear()
        for i in range(6):  # dummy resident plans
            fi._PYFFTW_PLAN_CACHE[('fwd', (64 + i, 64 + i), 'c16', 1)] = {}
    fi.restore_fft_state(state)
    assert fi._PYFFTW_PLAN_CACHE_SIZE == 2
    assert len(fi._PYFFTW_PLAN_CACHE) <= 2, (
        "restore must trim the plan cache to the restored bound")


def test_restore_double_buffer_semantics(_pristine_fft_state):
    """Flipping the ping-pong mode on restore must clear plans built in
    the other mode; restoring an IDENTICAL snapshot must keep them."""
    fi.set_fft_double_buffer(True)
    same_state = fi.snapshot_fft_state()
    dummy_key = ('fwd', (64, 64), 'c16', 1)
    with fi._PYFFTW_PLAN_LOCK:
        fi._PYFFTW_PLAN_CACHE.clear()
        fi._PYFFTW_PLAN_CACHE[dummy_key] = {}
    fi.restore_fft_state(same_state)          # no mode change
    assert dummy_key in fi._PYFFTW_PLAN_CACHE, (
        "identical-snapshot restore must not evict warm plans")
    flip_state = dict(same_state)
    flip_state['_PYFFTW_DOUBLE_BUFFER'] = False
    fi.restore_fft_state(flip_state)          # mode change
    assert fi._PYFFTW_DOUBLE_BUFFER is False
    assert dummy_key not in fi._PYFFTW_PLAN_CACHE, (
        "mode flip must clear plans built in the other mode")


def test_old_snapshot_missing_new_keys_tolerated(_pristine_fft_state):
    """A v5.4.6-shaped snapshot (9 keys) from an older process must not
    raise and must leave this process's newer knobs untouched."""
    full = fi.snapshot_fft_state()
    old_state = {k: full[k] for k in _V546_KEYS}
    fi.set_fft_double_buffer(False)
    fi.set_fft_fallback(False)
    fi.set_fft_plan_cache_size(3)
    fi.set_asm_cache_size(h_max_total_bytes=1024**3)
    before = {k: getattr(fi, k) for k in _NEW_KNOBS}
    fi.restore_fft_state(old_state)           # must not raise
    after = {k: getattr(fi, k) for k in _NEW_KNOBS}
    assert before == after, (
        "restore of an old (missing-key) snapshot must leave the newer "
        "knobs alone, not reset them")
    # unknown / empty snapshots stay forward-compatible (v5.4.6 contract)
    fi.restore_fft_state({'BOGUS_KEY': 1})
    fi.restore_fft_state({})
