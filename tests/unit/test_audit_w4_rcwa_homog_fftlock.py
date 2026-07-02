"""Wave-4 audit fixes: cache lifecycle in the rcwa-homog-fftlock cluster.

Pins the v5.17.1 fixes for:

* **P2-16 / P2-17** -- ``_HOMOG_CACHE`` (``lumenairy/elements/rcwa/_core.py``)
  was a plain unbounded module dict keyed on (wl, theta, phi, ...):
  ``RCWAStack.solve_vs_wavelength`` retained 2 dense (2N, 2N) eigenmode
  entries per sweep point forever (measured pre-fix: 25-wl sweep -> 50
  entries / 42.1 MB at nox=noy=4, doubling again per new theta).  Now a
  bounded LRU ``OrderedDict`` (``_HOMOG_CACHE_SIZE = 32``, the
  ``_H_CACHE`` move_to_end/popitem pattern) under the existing
  ``_HOMOG_LOCK``, still drained by the ``'rcwa_homogeneous_modes'``
  registry clearer.

* **P3-55** -- ``_clear_local_asm_caches``
  (``lumenairy/propagators/fft_infra.py``) cleared ``_PYFFTW_PLAN_CACHE``
  and ``_PYFFTW_BAD_SHAPES`` under ``_ASM_CACHE_LOCK`` instead of
  ``_PYFFTW_PLAN_LOCK``, so a concurrent clear could empty the plan cache
  between ``_get_or_make_plan``'s membership check and its indexing (both
  performed while holding the plan lock), raising an uncaught ``KeyError``
  out of ``_fft2``.  Now both pyFFTW structures are cleared under the plan
  lock (sequential acquisition, never nested).

Author: Wave-4 audit implementer -- v5.17.1
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np
import pytest

import lumenairy  # noqa: F401 -- import side-effects: register cache clearers
from lumenairy._cache_registry import (
    _CACHE_CLEARERS,
    list_registered_cache_clearers,
)
from lumenairy.elements.rcwa import _core
from lumenairy.elements.rcwa.stack import RCWAStack
from lumenairy.propagators import fft_infra as fi

WL = 0.6e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stack():
    """Small 2-D patterned stack (n_orders=2x2 -> tiny, fast solves)."""
    cell = np.full((12, 12), 1.0)
    cell[3:9, 3:9] = 2.25
    st = RCWAStack(0.8e-6, period_y=0.8e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=2, n_orders_y=2)
    st.add_layer(0.3e-6, eps_cell=cell)
    st.set_source(WL, theta=np.deg2rad(5))
    return st


def _rt(res):
    _, R, T = res.efficiencies()
    return np.array(R, copy=True), np.array(T, copy=True)


@pytest.fixture(autouse=True)
def _fresh_homog_cache():
    _core._clear_rcwa_caches()
    yield
    _core._clear_rcwa_caches()


# ---------------------------------------------------------------------------
# P2-16 / P2-17: _HOMOG_CACHE bounded LRU
# ---------------------------------------------------------------------------

def test_homog_cache_is_bounded_lru_container():
    """The cache is an OrderedDict with a positive module-level bound."""
    assert isinstance(_core._HOMOG_CACHE, OrderedDict)
    assert int(_core._HOMOG_CACHE_SIZE) >= 2  # one solve needs sup + sub
    assert int(_core._HOMOG_CACHE_SIZE) <= 256  # stays a genuine bound


def test_homog_cache_bounded_across_wavelength_sweep():
    """Sweeping 1.5x the bound in distinct wavelengths (= 3x-bound entries
    at 2 entries/wl) leaves exactly ``_HOMOG_CACHE_SIZE`` entries -- the
    pre-fix plain dict retained 2 entries per sweep point forever."""
    bound = int(_core._HOMOG_CACHE_SIZE)
    st = _make_stack()
    n_wl = (3 * bound) // 2  # 2 fresh entries (sup+sub) per wavelength
    wls = np.linspace(0.45e-6, 0.75e-6, n_wl)
    st.solve_vs_wavelength(wls)
    assert len(_core._HOMOG_CACHE) == bound


def test_homog_cache_lru_direct_eviction_order():
    """Direct-call LRU mechanics: oldest key evicted first, a re-touched
    key survives (move_to_end on hit)."""
    bound = int(_core._HOMOG_CACHE_SIZE)
    n = 2  # 1 order -> tiny matrices
    Kx = np.diag(np.zeros(n, dtype=np.complex128))
    Ky = np.diag(np.zeros(n, dtype=np.complex128))
    for i in range(bound):
        _core._cached_homogeneous_eigenmodes(1.0 + 0.001 * i, Kx, Ky,
                                             ("probe", i))
    assert len(_core._HOMOG_CACHE) == bound
    # touch the oldest so it becomes most-recent
    _core._cached_homogeneous_eigenmodes(1.0, Kx, Ky, ("probe", 0))
    # insert one more -> evicts ("probe", 1), NOT the re-touched ("probe", 0)
    _core._cached_homogeneous_eigenmodes(2.0, Kx, Ky, ("probe", bound))
    assert len(_core._HOMOG_CACHE) == bound
    assert ("probe", 0) in _core._HOMOG_CACHE
    assert ("probe", 1) not in _core._HOMOG_CACHE
    assert ("probe", bound) in _core._HOMOG_CACHE


def test_homog_hit_miss_and_postevict_recompute_byte_identical():
    """Cache hit, cold miss, and post-eviction recompute all produce
    byte-identical solve results; eviction never mutates values a caller
    already holds by reference."""
    st = _make_stack()
    R_miss, T_miss = _rt(st.solve())  # cold: miss path fills sup+sub
    held_refs = dict(_core._HOMOG_CACHE)  # live references to cached tuples
    held_copies = {k: tuple(np.array(a, copy=True) for a in v)
                   for k, v in held_refs.items()}
    assert len(held_refs) == 2  # sup + sub

    R_hit, T_hit = _rt(st.solve())  # warm: hit path
    assert np.array_equal(R_miss, R_hit)
    assert np.array_equal(T_miss, T_hit)

    # Flood with distinct wavelengths until the original entries evict.
    bound = int(_core._HOMOG_CACHE_SIZE)
    wls = np.linspace(0.46e-6, 0.74e-6, (3 * bound) // 2)
    st.solve_vs_wavelength(wls)
    assert all(k not in _core._HOMOG_CACHE for k in held_refs)

    # Held references must be unmutated by the eviction.
    for k, ref in held_refs.items():
        for a, b in zip(ref, held_copies[k]):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    # Post-eviction recompute is byte-identical to the original miss.
    R_re, T_re = _rt(st.solve())
    assert np.array_equal(R_miss, R_re)
    assert np.array_equal(T_miss, T_re)


def test_homog_cache_registry_clearer_drains():
    """'rcwa_homogeneous_modes' stays enrolled and its registered entry
    drains the (now-LRU) cache.

    The drain is exercised through the registry's OWN stored callable
    (what ``clear_all_registered_caches`` / ``clear_asm_caches`` invoke
    for this name) rather than through the full walk, so this test
    discriminates on the rcwa cache alone -- a defect in any OTHER
    module's registered clearer cannot mask (or spuriously fail) this
    cache's lifecycle.  The full-walk integration path is pinned by
    ``test_v4_16_0_agent_d_cache_registry.py``."""
    assert 'rcwa_homogeneous_modes' in list_registered_cache_clearers()
    st = _make_stack()
    st.solve()
    assert len(_core._HOMOG_CACHE) > 0
    _CACHE_CLEARERS['rcwa_homogeneous_modes']()
    assert len(_core._HOMOG_CACHE) == 0


# ---------------------------------------------------------------------------
# P3-55: _clear_local_asm_caches lock discipline
# ---------------------------------------------------------------------------

class _SlowContains(OrderedDict):
    """Plan-cache stand-in whose positive ``__contains__`` holds open the
    check-then-index window inside ``_get_or_make_plan`` (which runs while
    HOLDING ``_PYFFTW_PLAN_LOCK``), signalling the main thread via
    ``window`` so a concurrent clear can be issued deterministically."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.window = threading.Event()

    def __contains__(self, key):
        hit = super().__contains__(key)
        if hit:
            self.window.set()
            time.sleep(0.15)  # keep the window open for the clearer
        return hit


@pytest.mark.skipif(not fi.PYFFTW_AVAILABLE, reason="pyFFTW not installed")
def test_clear_local_asm_caches_serialises_with_plan_lookup():
    """Deterministic P3-55 regression: a clear issued INSIDE the plan-lookup
    window (FFT thread holding _PYFFTW_PLAN_LOCK between the membership
    check and the indexing) must block on the plan lock instead of emptying
    the plan cache underfoot.  Pre-fix this raised an uncaught
    ``KeyError: ('fwd', (256, 256), '<c16', ...)`` out of ``_fft2``."""
    if not fi.USE_PYFFTW:
        pytest.skip("pyFFTW dispatch disabled")
    n = int(fi.FFTW_MIN_SIZE)
    x = (np.ones((n, n)) + 0.5j).astype(np.complex128)
    ref = np.fft.fft2(x)

    fi._clear_local_asm_caches()
    warm = np.array(fi._fft2(x.copy()), copy=True)  # plan now cached
    np.testing.assert_allclose(warm, ref, rtol=0, atol=1e-9)

    orig = fi._PYFFTW_PLAN_CACHE
    slow = _SlowContains(orig)
    fi._PYFFTW_PLAN_CACHE = slow
    result: dict = {}

    def fft_thread():
        try:
            result['out'] = np.array(fi._fft2(x.copy()), copy=True)
        except Exception as e:  # noqa: BLE001 -- pre-fix: uncaught KeyError
            result['exc'] = e

    try:
        t = threading.Thread(target=fft_thread)
        t.start()
        assert slow.window.wait(5.0), "FFT thread never reached the hit path"
        fi._clear_local_asm_caches()  # must serialise on _PYFFTW_PLAN_LOCK
        t.join(10.0)
        assert not t.is_alive()
    finally:
        fi._PYFFTW_PLAN_CACHE = orig
        fi._clear_local_asm_caches()

    assert 'exc' not in result, (
        f"clear raced the plan lookup: {result.get('exc')!r}")
    assert np.array_equal(result['out'], warm)  # byte-identical to warm hit


@pytest.mark.skipif(not fi.PYFFTW_AVAILABLE, reason="pyFFTW not installed")
def test_clear_local_asm_caches_still_drains_everything():
    """The lock-discipline fix must not change WHAT is drained: all three
    ASM caches plus both pyFFTW structures empty after one call."""
    if not fi.USE_PYFFTW:
        pytest.skip("pyFFTW dispatch disabled")
    n = int(fi.FFTW_MIN_SIZE)
    x = (np.ones((n, n)) + 0.25j).astype(np.complex128)
    fi._fft2(x.copy())
    assert len(fi._PYFFTW_PLAN_CACHE) > 0
    with fi._PYFFTW_PLAN_LOCK:
        fi._PYFFTW_BAD_SHAPES.add((7, 7))  # synthetic blacklist entry
    fi._clear_local_asm_caches()
    assert len(fi._PYFFTW_PLAN_CACHE) == 0
    assert len(fi._PYFFTW_BAD_SHAPES) == 0
    assert len(fi._H_CACHE) == 0
    assert len(fi._FREQ_GRID_CACHE) == 0
    assert len(fi._BANDLIMIT_CACHE) == 0


def test_restore_fft_state_path_deadlock_free():
    """Wave-2 made ``restore_fft_state`` call ``set_fft_plan_cache_size``
    (which takes ``_PYFFTW_PLAN_LOCK``); the P3-55 fix must keep the
    snapshot/restore/clear round-trip free of lock nesting.  A watchdog
    bounds the whole round-trip so a deadlock fails fast instead of
    hanging the suite."""
    done = threading.Event()

    def roundtrip():
        snap = fi.snapshot_fft_state()
        fi.restore_fft_state(snap)
        fi._clear_local_asm_caches()
        fi.reset_fft_backend()  # plan lock, then clear_asm_caches (released)
        done.set()

    t = threading.Thread(target=roundtrip, daemon=True)
    t.start()
    assert done.wait(30.0), "FFT state round-trip deadlocked"
