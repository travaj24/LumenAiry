"""Roadmap P0 -- shared byte-budgeted LRU cache infrastructure.

Validates the memory-safety FOUNDATION (roadmap Section 0) that unblocks
every N^2-scale perf cache in Part B:

* byte-budgeted (not count-budgeted) -- retained bytes never exceed the
  budget across a stress insert of oversized entries;
* LRU eviction (not FIFO) within a cache AND globally across caches;
* a single collective global ceiling honoured by all instances;
* env / set_cache_budget / get_cache_budget budget control;
* registry enrollment so ``clear_asm_caches()`` drains every instance
  (the v4.16.1 meta-pin walker stays green);
* releasable (.release()/.clear()) + introspectable (cache_report()).

Arrays here are deliberately tiny (1 MB = a 256x256 complex128) and the
budgets are a few MB, so the suite is fast and light; a defensive RAM
guard still fronts the concurrency stress case.

Author: Andrew Traverso -- roadmap P0
"""
from __future__ import annotations

import gc
import os
import threading

import numpy as np
import pytest

import lumenairy as la
from lumenairy import cache as cache_mod
from lumenairy.cache import (
    ByteBudgetedLRU,
    _auto_cache_budget_bytes,
    _parse_env_budget,
    cache_report,
    clear_all_byte_budgeted_caches,
    deep_nbytes,
    get_cache_budget,
    get_cache_budget_bytes,
    set_cache_budget,
)

_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# INJECTED memory budget -- 2026-08-15,
# docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md
# ---------------------------------------------------------------------------
# Every auto-budget number in this file used to come from a LIVE psutil read
# (``lumenairy.cache.available_memory_bytes`` -> ``psutil``), and the auto
# budget is ``min(512 MB, 10% of available)``.  On a big box the 512 MB cap
# binds, so the value looks frozen; on a runner with < 5.12 GB available the
# 10%-FRACTION branch binds and the value tracks free RAM, which drifts
# between two consecutive reads.  Measured 2026-08-15: consecutive live reads
# on an IDLE Windows dev box differ by up to 139264 B, and forcing a 10 MB
# drop between the two reads turns
# ``get_cache_budget_bytes() == _auto_cache_budget_bytes()`` into
# ``400000000 != 398951424`` -- a per-build coin flip, not a contract.
#
# Fix: INJECT the memory reading (monkeypatch the module's
# ``available_memory_bytes``) so every budget number is exact arithmetic on a
# pinned input, and exercise BOTH branches explicitly.  The fraction branch
# was previously unreachable on any dev box (93 GB available -> cap binds), so
# this is strictly MORE coverage, not less.
_PINNED_AVAILABLE_BYTES = 8 * 1024 * _MB     # 8 GiB -> 10% = 819 MB > cap
_PINNED_AVAILABLE_SMALL = 4 * 1024 * _MB     # 4 GiB -> 10% = 409.6 MB < cap
_EXPECTED_CAP_BYTES = 512 * _MB              # the library's hard cap


def _pin_available_ram(monkeypatch, free_b: int) -> None:
    """Freeze the available-RAM reading the cache budget is derived from.

    Pinned-snapshot idiom, as in
    ``test_hammer_h3_traced_nyquist_guard.py::_pin_available_ram`` /
    ``test_fix_newton_pool_memory.py``.  ``lumenairy.cache`` binds
    ``available_memory_bytes`` at import, so the cache module's own name is
    the one that must be patched; ``lumenairy.memory``'s is patched too so a
    later re-import or a late-bound call site cannot quietly restore the live
    psutil read without this helper noticing.
    """
    from lumenairy import memory as _mem
    monkeypatch.setattr(cache_mod, 'available_memory_bytes',
                        lambda: int(free_b))
    monkeypatch.setattr(_mem, 'available_memory_bytes', lambda: int(free_b))


def _mb_array(mb: float = 1.0) -> np.ndarray:
    """A complex128 array of exactly ``mb`` megabytes (mb must give an
    integral element count)."""
    n_elems = int(mb * _MB // 16)
    return np.zeros(n_elems, dtype=np.complex128)


@pytest.fixture(autouse=True)
def _isolate_cache_state(monkeypatch):
    """Give each test a clean slate: drop dead caches, drain live ones,
    revert the budget override.  Runs before and after every test so no
    cross-test contamination of the collective budget / live registry.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): also pins the
    available-RAM reading the auto budget is derived from, so no assertion in
    this file depends on how much RAM the runner happens to have free at the
    instant it executes.  ``put()`` itself consults
    ``get_cache_budget_bytes()`` for its hard cap, so the pin makes the whole
    file's eviction arithmetic deterministic, not just the budget getters.
    Tests that care about a specific branch re-pin locally.
    """
    _pin_available_ram(monkeypatch, _PINNED_AVAILABLE_BYTES)
    gc.collect()
    clear_all_byte_budgeted_caches()
    set_cache_budget(None)
    yield
    clear_all_byte_budgeted_caches()
    set_cache_budget(None)
    gc.collect()


# ===========================================================================
# Public surface + registry enrollment
# ===========================================================================

def test_public_surface_reexported():
    """The four public names are top-level (the __all__-symmetry walker's
    contract) and resolvable."""
    for name in ('ByteBudgetedLRU', 'cache_report',
                 'get_cache_budget', 'set_cache_budget'):
        assert name in la.__all__, f"{name} missing from lumenairy.__all__"
        assert hasattr(la, name), f"lumenairy.{name} not resolvable"
    assert la.ByteBudgetedLRU is ByteBudgetedLRU


def test_registry_enrollment_present():
    """A single collective clearer is enrolled so the v4.16.1 meta-pin
    walker stays green and clear_asm_caches drains us."""
    assert 'byte_budgeted_lru' in la.list_registered_cache_clearers()


# ===========================================================================
# Byte-budgeted, not count-budgeted: the core contract
# ===========================================================================

def test_byte_budget_never_exceeded_under_stress():
    """A 10 MB cache fed 4 MB arrays evicts LRU and NEVER exceeds the byte
    budget -- the headline memory-safety guarantee."""
    budget = 10 * _MB
    c = ByteBudgetedLRU('stress', max_bytes=budget)
    for i in range(25):
        c.put(i, _mb_array(4.0))
        assert c.retained_bytes <= budget, (
            f"retained {c.retained_bytes} > budget {budget} after insert {i}")
    # 4 MB entries under a 10 MB cap -> at most 2 retained (8 MB).
    assert c.retained_bytes <= budget
    assert len(c) <= 2
    assert c.evictions > 0


def test_bound_by_bytes_not_entry_count():
    """Many tiny entries fit; few large entries are capped -- proving the
    bound is bytes, not a fixed entry count."""
    c = ByteBudgetedLRU('bytes_not_count', max_bytes=8 * _MB)
    # 32 x 0.25 MB = 8 MB: all fit under the byte budget.
    for i in range(32):
        c.put(('tiny', i), _mb_array(0.25))
    assert len(c) == 32
    assert c.retained_bytes <= 8 * _MB
    # Now 4 x 4 MB into the same-size budget -> only 2 survive (byte bound).
    c2 = ByteBudgetedLRU('bytes_not_count2', max_bytes=8 * _MB)
    for i in range(4):
        c2.put(('big', i), _mb_array(4.0))
    assert len(c2) == 2
    assert c2.retained_bytes <= 8 * _MB


def test_per_entry_larger_than_budget_is_skipped():
    """A single value bigger than the hard ceiling is skipped (never
    stored), and the caller-visible state stays empty -- not OOM."""
    c = ByteBudgetedLRU('too_big', max_bytes=2 * _MB)
    stored = c.put('huge', _mb_array(8.0))   # 8 MB into a 2 MB cache
    assert stored is False
    assert len(c) == 0
    assert c.retained_bytes == 0
    assert c.get('huge', 'MISS') == 'MISS'


# ===========================================================================
# LRU (not FIFO) eviction
# ===========================================================================

def test_lru_keeps_most_recently_used():
    """Touching an entry protects it from eviction -- the design-loop
    hot-entry guarantee that FIFO would violate."""
    c = ByteBudgetedLRU('lru', max_bytes=2 * _MB)  # holds 2 x 1 MB
    c.put('a', _mb_array(1.0))
    c.put('b', _mb_array(1.0))
    c.get('a')                       # 'a' now most-recently-used
    c.put('c', _mb_array(1.0))       # evict LRU -> 'b'
    assert 'a' in c, "MRU-touched entry was wrongly evicted (FIFO behaviour)"
    assert 'c' in c
    assert 'b' not in c


def test_lru_evicts_oldest_when_untouched():
    """With no touches, the least-recently-INSERTED goes first."""
    c = ByteBudgetedLRU('lru_fifo_like', max_bytes=2 * _MB)
    c.put('a', _mb_array(1.0))
    c.put('b', _mb_array(1.0))
    c.put('c', _mb_array(1.0))       # evict 'a'
    assert 'a' not in c
    assert 'b' in c and 'c' in c


# ===========================================================================
# Global (collective) budget shared across caches
# ===========================================================================

def test_global_budget_shared_across_two_instances():
    """Two caches, each bounded only by the global budget, collectively
    never exceed it -- global eviction reaches across caches."""
    set_cache_budget(6.0)            # 6 MB collective
    clear_all_byte_budgeted_caches()
    a = ByteBudgetedLRU('gA', max_bytes=None)
    b = ByteBudgetedLRU('gB', max_bytes=None)
    for i in range(20):
        (a if i % 2 == 0 else b).put(i, _mb_array(1.0))
        total = a.retained_bytes + b.retained_bytes
        assert total <= 6 * _MB, (
            f"collective retained {total} > global budget {6 * _MB} "
            f"after insert {i}")
    rep = cache_report()
    assert rep['total_retained_bytes'] <= rep['budget_bytes']


def test_global_lru_evicts_globally_oldest():
    """The globally least-recently-used entry (across caches) is the one
    evicted when the collective ceiling is crossed."""
    set_cache_budget(3.0)            # room for 3 x 1 MB
    clear_all_byte_budgeted_caches()
    a = ByteBudgetedLRU('glA', max_bytes=None)
    b = ByteBudgetedLRU('glB', max_bytes=None)
    a.put('a0', _mb_array(1.0))      # oldest by insert
    b.put('b0', _mb_array(1.0))
    a.put('a1', _mb_array(1.0))      # total = 3 MB (== budget)
    a.get('a0')                      # touch a0 -> now b0 is globally oldest
    b.put('b1', _mb_array(1.0))      # 4 MB > 3 -> evict global-LRU (b0)
    assert 'b0' not in b, "global eviction did not remove the oldest entry"
    assert 'a0' in a, "touched entry wrongly evicted globally"
    assert 'a1' in a
    assert 'b1' in b
    assert (a.retained_bytes + b.retained_bytes) <= 3 * _MB


def test_lowering_budget_evicts_immediately():
    """set_cache_budget() to a smaller value evicts NOW, not lazily on the
    next insert."""
    set_cache_budget(10.0)
    clear_all_byte_budgeted_caches()
    c = ByteBudgetedLRU('shrink', max_bytes=None)
    for i in range(8):
        c.put(i, _mb_array(1.0))     # 8 MB retained under 10 MB budget
    assert c.retained_bytes == 8 * _MB
    set_cache_budget(3.0)            # tighten -> must evict to <= 3 MB
    assert c.retained_bytes <= 3 * _MB


# ===========================================================================
# Registry drain / release / clear
# ===========================================================================

def test_clear_asm_caches_drains_budgeted_cache():
    """la.clear_asm_caches() walks the registry -> our collective clearer
    -> every live budgeted cache is drained."""
    c = ByteBudgetedLRU('drainme', max_bytes=8 * _MB)
    for i in range(4):
        c.put(i, _mb_array(1.0))
    assert c.retained_bytes > 0 and len(c) > 0
    la.clear_asm_caches()
    assert c.retained_bytes == 0
    assert len(c) == 0


def test_release_and_clear_free_memory():
    c = ByteBudgetedLRU('rel', max_bytes=8 * _MB)
    c.put('x', _mb_array(2.0))
    assert c.retained_bytes == 2 * _MB
    c.release()
    assert c.retained_bytes == 0 and len(c) == 0
    c.put('y', _mb_array(2.0))
    c.clear()
    assert c.retained_bytes == 0 and len(c) == 0


# ===========================================================================
# cache_report introspection
# ===========================================================================

def test_cache_report_sums_correctly():
    clear_all_byte_budgeted_caches()
    a = ByteBudgetedLRU('rptA', max_bytes=8 * _MB)
    b = ByteBudgetedLRU('rptB', max_bytes=8 * _MB)
    a.put(1, _mb_array(1.0))
    a.put(2, _mb_array(1.0))
    b.put(1, _mb_array(2.0))
    rep = cache_report()
    per_sum = sum(v['retained_bytes'] for v in rep['caches'].values())
    assert rep['total_retained_bytes'] == per_sum
    assert rep['total_retained_bytes'] == a.retained_bytes + b.retained_bytes
    assert rep['caches']['rptA']['entries'] == 2
    assert rep['caches']['rptB']['retained_bytes'] == 2 * _MB
    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): this used to
    # compare TWO INDEPENDENT LIVE psutil reads -- ``cache_report()`` does its
    # own ``get_cache_budget_bytes()`` internally.  Under the injected pin
    # (autouse fixture) both reads are the same arithmetic on the same input,
    # so the equality is a real report-vs-getter consistency check instead of
    # a RAM-drift coin flip.  The exact value is asserted too, so a silent
    # change of the reported budget cannot hide behind "both sides moved".
    assert rep['budget_bytes'] == get_cache_budget_bytes()
    assert rep['budget_bytes'] == _EXPECTED_CAP_BYTES


def test_cache_report_duplicate_names_not_dropped():
    clear_all_byte_budgeted_caches()
    a = ByteBudgetedLRU('dup', max_bytes=8 * _MB)
    b = ByteBudgetedLRU('dup', max_bytes=8 * _MB)
    a.put(1, _mb_array(1.0))
    b.put(1, _mb_array(1.0))
    rep = cache_report()
    # Both live caches represented (second keyed 'dup#2').  ``n_caches`` is
    # ``>= 2`` rather than ``== 2`` because the library now ships persistent
    # module-level budgeted caches (e.g. R1's opt-in 'displaced_cos_grid'),
    # which are always live in the registry; the ``_isolate_cache_state``
    # fixture drains their ENTRIES (so they contribute 0 bytes here) but does
    # not unregister them.
    assert rep['n_caches'] >= 2
    assert 'dup' in rep['caches'] and 'dup#2' in rep['caches']
    assert rep['total_retained_bytes'] == a.retained_bytes + b.retained_bytes


def test_hit_miss_eviction_counters():
    c = ByteBudgetedLRU('counters', max_bytes=2 * _MB)
    c.put('a', _mb_array(1.0))
    assert c.get('a') is not None       # hit
    assert c.get('missing', None) is None  # miss
    c.put('b', _mb_array(1.0))
    c.put('c', _mb_array(1.0))          # evicts one
    r = c.report()
    assert r['hits'] == 1
    assert r['misses'] == 1
    assert r['evictions'] >= 1
    assert r['name'] == 'counters'
    assert r['max_bytes'] == 2 * _MB


# ===========================================================================
# Budget getters / setters / env
# ===========================================================================

def test_set_get_cache_budget_roundtrip():
    set_cache_budget(128.0)
    assert abs(get_cache_budget() - 128.0) < 1e-9
    assert get_cache_budget_bytes() == 128 * _MB
    set_cache_budget(None)  # revert to auto default
    # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): reverting to
    # auto used to be pinned by comparing two INDEPENDENT live psutil reads
    # (``get_cache_budget_bytes()`` vs ``_auto_cache_budget_bytes()``).  With
    # the injected pin both sides evaluate the same arithmetic on the same
    # input, so the equality now says what it always meant: "clearing the
    # override restores the auto default", with the exact expected value
    # spelled out so the assertion cannot pass by both sides drifting alike.
    assert get_cache_budget_bytes() == _auto_cache_budget_bytes()
    assert get_cache_budget_bytes() == _EXPECTED_CAP_BYTES


@pytest.mark.parametrize(
    'pinned_available, expect_cap_binds',
    [
        pytest.param(_PINNED_AVAILABLE_SMALL, False, id='fraction_branch'),
        pytest.param(_PINNED_AVAILABLE_BYTES, True, id='cap_branch'),
    ],
)
def test_default_budget_is_conservative(monkeypatch, pinned_available,
                                        expect_cap_binds):
    """The auto default is ``min(512 MB, 10% of available)`` -- BOTH
    branches, on an injected available-RAM value.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md).  This used to
    read ``auto <= int(0.10 * available_memory_bytes()) + 1`` against a
    SECOND live psutil read; the ``+ 1`` covered ten bytes of drift while a
    10 MB change in free RAM between the two calls (ordinary on a shared
    runner, measured to break the equality: 400000000 vs 398951424) burned
    the build.  Worse, on any box with > 5.12 GB free the 512 MB cap binds,
    so the fraction branch -- the only one that can move -- was never
    executed at all.  Injecting the reading makes both branches exact:
    4 GiB -> int(0.10 * 4294967296) = 429496729 B (fraction binds),
    8 GiB -> 536870912 B (cap binds).  Two-sided: no runner state can move
    either number, and a regression in either the fraction, the cap, or the
    min() that combines them still fails on the exact-value assertions.
    """
    _pin_available_ram(monkeypatch, pinned_available)
    set_cache_budget(None)
    auto = get_cache_budget_bytes()
    frac = int(0.10 * pinned_available)
    # The branch this parametrisation claims to exercise really is the live
    # one (otherwise the "both branches covered" claim would be vacuous).
    assert (frac > _EXPECTED_CAP_BYTES) is expect_cap_binds
    assert auto == min(_EXPECTED_CAP_BYTES, frac)
    assert auto == _auto_cache_budget_bytes()
    # The headline safety property, on both branches.
    assert auto <= _EXPECTED_CAP_BYTES
    assert auto <= frac


def test_set_cache_budget_rejects_nonpositive():
    with pytest.raises(ValueError):
        set_cache_budget(0)
    with pytest.raises(ValueError):
        set_cache_budget(-5)


def test_parse_env_budget(monkeypatch):
    monkeypatch.setenv('LUMENAIRY_CACHE_BUDGET_MB', '64')
    assert _parse_env_budget() == 64 * _MB
    monkeypatch.setenv('LUMENAIRY_CACHE_BUDGET_MB', 'not-a-number')
    assert _parse_env_budget() is None
    monkeypatch.setenv('LUMENAIRY_CACHE_BUDGET_MB', '-1')
    assert _parse_env_budget() is None
    monkeypatch.delenv('LUMENAIRY_CACHE_BUDGET_MB', raising=False)
    assert _parse_env_budget() is None


# ===========================================================================
# Opt-in / disabled behaviour
# ===========================================================================

def test_disabled_cache_stores_nothing():
    """max_bytes=0 => the N^2-scale-cache 'off by default' state: put is a
    no-op, get always misses."""
    c = ByteBudgetedLRU('disabled', max_bytes=0)
    assert c.put('a', _mb_array(1.0)) is False
    assert len(c) == 0
    assert c.retained_bytes == 0
    assert c.get('a', 'MISS') == 'MISS'


def test_opt_in_via_set_budget():
    """A cache ships disabled then the caller opts in with set_budget()."""
    c = ByteBudgetedLRU('optin', max_bytes=0)
    c.put('a', _mb_array(1.0))
    assert len(c) == 0               # still off
    c.set_budget(4 * _MB)            # user enables the design loop
    assert c.put('a', _mb_array(1.0)) is True
    assert 'a' in c
    c.set_budget(0)                  # disable again -> clears
    assert len(c) == 0 and c.retained_bytes == 0


# ===========================================================================
# Complete keys / no stale reuse
# ===========================================================================

def test_distinct_keys_distinct_values_no_stale_reuse():
    c = ByteBudgetedLRU('keys', max_bytes=64 * _MB)
    v1 = np.arange(10)
    v2 = np.arange(10) + 100
    c.put(('presc', 1.55e-6, 1024), v1)
    c.put(('presc', 1.31e-6, 1024), v2)   # different wavelength -> distinct
    assert np.array_equal(c.get(('presc', 1.55e-6, 1024)), v1)
    assert np.array_equal(c.get(('presc', 1.31e-6, 1024)), v2)


def test_replace_updates_retained_bytes():
    c = ByteBudgetedLRU('replace', max_bytes=64 * _MB)
    c.put('k', _mb_array(4.0))
    assert c.retained_bytes == 4 * _MB
    c.put('k', _mb_array(1.0))       # replace same key with smaller value
    assert c.retained_bytes == 1 * _MB
    assert len(c) == 1


# ===========================================================================
# deep_nbytes accounting
# ===========================================================================

def test_deep_nbytes_ndarray_tuple_dict():
    a = np.zeros(1000, dtype=np.float64)   # 8000 bytes
    assert deep_nbytes(a) == a.nbytes
    t = (np.zeros(1000), np.zeros(2000))   # 8000 + 16000 = 24000 + overhead
    assert deep_nbytes(t) >= 24000
    d = {'x': np.zeros(500), 'y': np.zeros(500)}  # 4000 + 4000
    assert deep_nbytes(d) >= 8000
    assert deep_nbytes(None) == 0


def test_deep_nbytes_handles_cycles():
    lst = []
    lst.append(lst)                  # self-referential
    # Must terminate (id-based seen set) and return a finite size.
    assert deep_nbytes(lst) >= 0


def test_get_or_compute_memoizes():
    c = ByteBudgetedLRU('memo', max_bytes=8 * _MB)
    calls = {'n': 0}

    def build():
        calls['n'] += 1
        return _mb_array(1.0)

    v1 = c.get_or_compute('k', build)
    v2 = c.get_or_compute('k', build)   # served from cache
    assert v1 is v2
    assert calls['n'] == 1


# ===========================================================================
# Thread safety
# ===========================================================================

def test_thread_safety_stress_invariant_holds():
    """Concurrent inserts from several threads never breach the byte budget
    and never raise -- the shared-mutex correctness check.

    2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md): the defensive
    ``available_memory_bytes() < 256 MB -> skip`` guard was removed.  It made
    the shared-mutex contract silently OPTIONAL on exactly the constrained
    runner where a locking bug is most likely to show, and it was sized ~35x
    too large: the case never holds 256 MB.  The cache is capped at 4 MB and
    each of the 6 threads holds one 0.5 MB array at a time, so the measured
    peak RSS growth for the whole stress is 5.78 MB (Windows py3.14 /
    np2.4.4) and 5.89 MB (WSL py3.12 / np2.5.1) -- an allocation any runner
    that can import numpy already satisfies, so the case now runs
    unconditionally.
    """
    budget = 4 * _MB
    c = ByteBudgetedLRU('threads', max_bytes=budget)
    errors = []
    n_threads = 6
    per_thread = 40

    def worker(tid):
        try:
            for j in range(per_thread):
                c.put((tid, j), _mb_array(0.5))
                rb = c.retained_bytes
                if rb > budget:
                    errors.append((tid, j, rb))
        except Exception as exc:       # pragma: no cover - failure path
            errors.append((tid, 'exc', repr(exc)))

    threads = [threading.Thread(target=worker, args=(t,))
               for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"thread-safety invariant breached: {errors[:5]}"
    assert c.retained_bytes <= budget


# ===========================================================================
# Weak-registry: a dropped cache stops counting against the global budget
# ===========================================================================

def test_dropped_cache_leaves_global_accounting():
    set_cache_budget(8.0)
    clear_all_byte_budgeted_caches()
    keep = ByteBudgetedLRU('keep', max_bytes=None)
    keep.put('k', _mb_array(1.0))

    transient = ByteBudgetedLRU('transient', max_bytes=None)
    transient.put('t', _mb_array(2.0))
    assert cache_report()['total_retained_bytes'] == 3 * _MB

    del transient
    gc.collect()
    # The transient cache is gone; only 'keep' still counts.
    rep = cache_report()
    assert rep['total_retained_bytes'] == 1 * _MB
    names = set(rep['caches'].keys())
    assert 'transient' not in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
