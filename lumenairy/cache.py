"""Shared byte-budgeted LRU cache infrastructure (roadmap P0).

This module is the memory-safety FOUNDATION for every N^2-scale cache on
the deferred perf/memory roadmap (Part B).  The stated user concern --
*caches blowing up the memory footprint* -- is answered here, once, so
that every subsequent cache is memory-safe by construction rather than by
each author re-deriving the discipline.

The contract (roadmap Section 0)
--------------------------------

Every cache built on :class:`ByteBudgetedLRU` is:

1. **Byte-budgeted, not count-budgeted.**  Bound by TOTAL RETAINED BYTES
   (``.nbytes`` for ndarrays, a deep sizeof for structures), not by entry
   count.  A single global ceiling -- ``LUMENAIRY_CACHE_BUDGET_MB`` (env)
   + :func:`set_cache_budget` / :func:`get_cache_budget` -- that ALL
   budgeted caches respect *collectively*.  Default conservative:
   ``min(512 MB, 10 %% of available_memory_bytes())`` so enabling caching
   never surprises a memory-constrained host.
2. **LRU eviction** (not FIFO): a design-iteration loop keeps its hot
   entry instead of evicting the one it is re-using.  Eviction is
   least-recently-used both *within* a cache and *globally* across every
   live budgeted cache when the collective ceiling is exceeded.
3. **Opt-in for large caches.**  A cache that can hold N^2-scale arrays is
   the CALLER's opt-in: construct with ``max_bytes=0`` (disabled -- stores
   nothing) and flip to a real budget via :meth:`ByteBudgetedLRU.set_budget`
   when the user enables a design loop.  Small, cheap caches (scalars,
   1-D LUTs) may pass a tiny fixed ``max_bytes`` and stay on.
4. **Registry-enrolled + releasable.**  All instances are drained by
   :func:`lumenairy.clear_asm_caches` (a single ``'byte_budgeted_lru'``
   registry clearer walks every live instance); each instance exposes
   :meth:`~ByteBudgetedLRU.release` / :meth:`~ByteBudgetedLRU.clear`.
5. **Complete keys -- no stale reuse.**  The KEY is the caller's
   responsibility; the roadmap mandates it captures the full determinant
   of the cached value (prescription + conjugate + wavelength + grid
   ``dx, N``).  This module keys on whatever hashable the caller supplies
   and never reuses across distinct keys.
6. **Measured + introspectable.**  :func:`cache_report` returns per-cache
   ``{retained_bytes, entries, hits, misses, ...}`` plus the collective
   total and the live budget (mirroring ``fga_memory_estimate``'s spirit),
   so the footprint is always visible, never a black box.

Locking
-------

All budgeted caches serialise on a single module-level reentrant mutex
(:data:`_BUDGET_MUTEX`).  A single shared lock is the correct choice here
because global eviction reaches *across* caches (it must inspect and pop
from a sibling cache), so a per-instance lock would invite lock-order
inversion; the guarded work (dict bookkeeping) is microseconds, while the
expensive compute a cache exists to avoid (e.g. the ~3.9 s Delaunay build
in the cos-grid path) always runs OUTSIDE the lock in the caller.  The
mutex is a ``threading.RLock`` so :meth:`ByteBudgetedLRU.put` can invoke
:func:`_enforce_global_budget_locked` re-entrantly.

Author: Andrew Traverso -- roadmap P0
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import numpy as np

from .memory import available_memory_bytes

# ---------------------------------------------------------------------------
# Global budget configuration
# ---------------------------------------------------------------------------
# Conservative default: a small fraction of currently-available RAM, hard-
# capped so enabling caching on a big box still never claims more than
# 512 MB by default.  The env var + set_cache_budget() override this.
_DEFAULT_CACHE_BUDGET_CAP_BYTES = 512 * 1024 * 1024      # 512 MB hard cap
_DEFAULT_CACHE_BUDGET_RAM_FRACTION = 0.10                # 10% of available RAM
_FALLBACK_AVAILABLE_BYTES = 4 * 1024 * 1024 * 1024       # 4 GB if query fails

# Shared reentrant mutex guarding every budgeted cache + the global
# bookkeeping below.  Named to avoid the ``_*_LOCK`` / ``_*_CACHE``
# dispatcher-pin patterns (test_v4_14_2_dispatcher_pin_cache_locks): this
# is the one mutex for the whole subsystem, not a per-cache companion lock,
# and an RLock (needed for the reentrant put -> global-evict call) is not
# even an instance of ``type(threading.Lock())`` so the pin would not pair
# it regardless.
_BUDGET_MUTEX = threading.RLock()

# Live registry of ByteBudgetedLRU instances.  WEAK so a discarded cache
# (e.g. a test-local instance) stops counting against the global budget
# once it is garbage-collected, rather than leaking into the collective
# ceiling forever.
_LIVE_CACHES: "weakref.WeakSet" = weakref.WeakSet()

# Global monotonic access counter.  Every put and every get-hit stamps its
# entry with the next value, so the front entry of each cache's OrderedDict
# carries that cache's smallest stamp and the global minimum-stamp entry is
# found by comparing per-cache front entries (see the global evictor).
_GLOBAL_SEQ = 0

# Global budget override in bytes (None => auto = min(cap, fraction * RAM)).
_CACHE_BUDGET_OVERRIDE_BYTES: Optional[int] = None


def _parse_env_budget() -> Optional[int]:
    """Parse ``LUMENAIRY_CACHE_BUDGET_MB`` into a byte override, or None.

    A malformed / non-positive value is ignored (falls back to the auto
    default) rather than raising at import time.
    """
    raw = os.environ.get('LUMENAIRY_CACHE_BUDGET_MB')
    if raw is None:
        return None
    try:
        mb = float(raw)
    except (TypeError, ValueError):
        return None
    if mb <= 0:
        return None
    return int(mb * 1024 * 1024)


_CACHE_BUDGET_OVERRIDE_BYTES = _parse_env_budget()


# ---------------------------------------------------------------------------
# Budget getters / setters
# ---------------------------------------------------------------------------

def _auto_cache_budget_bytes() -> int:
    """The conservative auto budget: ``min(512 MB, 10% of available RAM)``."""
    try:
        avail = int(available_memory_bytes())
    except Exception:
        avail = _FALLBACK_AVAILABLE_BYTES
    frac = int(_DEFAULT_CACHE_BUDGET_RAM_FRACTION * avail)
    return int(max(0, min(_DEFAULT_CACHE_BUDGET_CAP_BYTES, frac)))


def get_cache_budget_bytes() -> int:
    """Return the effective collective cache budget in BYTES.

    The :func:`set_cache_budget` / ``LUMENAIRY_CACHE_BUDGET_MB`` override
    when set, else the auto default ``min(512 MB, 10% of
    available_memory_bytes())``.
    """
    with _BUDGET_MUTEX:
        if _CACHE_BUDGET_OVERRIDE_BYTES is not None:
            return int(_CACHE_BUDGET_OVERRIDE_BYTES)
        return _auto_cache_budget_bytes()


def get_cache_budget() -> float:
    """Return the effective collective cache budget in MEGABYTES (float).

    Mirrors :func:`set_cache_budget` exactly: a round-trip through
    ``set_cache_budget(get_cache_budget())`` re-pins the current effective
    budget.  Use :func:`get_cache_budget_bytes` for the byte value.
    """
    return get_cache_budget_bytes() / (1024.0 * 1024.0)


def set_cache_budget(mb: Optional[float]) -> None:
    """Set the collective byte ceiling that ALL budgeted caches respect.

    Parameters
    ----------
    mb : float or None
        Budget in megabytes.  ``None`` reverts to the conservative auto
        default (``min(512 MB, 10% of available_memory_bytes())``).  A
        non-positive value is rejected.

    Notes
    -----
    Lowering the budget evicts globally (least-recently-used first, across
    every live cache) immediately so the new ceiling is respected on
    return -- not lazily on the next insert.
    """
    global _CACHE_BUDGET_OVERRIDE_BYTES
    with _BUDGET_MUTEX:
        if mb is None:
            _CACHE_BUDGET_OVERRIDE_BYTES = None
        else:
            mb = float(mb)
            if mb <= 0:
                raise ValueError(
                    f"set_cache_budget: mb must be positive (got {mb!r}); "
                    f"pass None to revert to the auto default.")
            _CACHE_BUDGET_OVERRIDE_BYTES = int(mb * 1024 * 1024)
        _enforce_global_budget_locked()


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------

def deep_nbytes(obj: Any, _seen: Optional[set] = None) -> int:
    """Best-effort retained-byte size of a cache value.

    * ndarray (anything exposing an integer ``.nbytes``): the buffer size.
    * tuple / list / set / frozenset: container overhead + recursive sum
      of the elements (the common ``(cos_in, cos_out)`` array-pair case).
    * dict: container overhead + recursive sum of keys and values.
    * anything else: ``sys.getsizeof``.

    Cycles are handled with an id-based ``_seen`` set (each object counted
    once).  Under-counting is the fail-safe direction: it can only let a
    cache hold a little more than its budget, never corrupt behaviour.
    """
    if obj is None:
        return 0
    nb = getattr(obj, 'nbytes', None)
    if isinstance(nb, (int, np.integer)):
        return int(nb)
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    try:
        total = sys.getsizeof(obj)
    except TypeError:
        total = 0
    if isinstance(obj, (tuple, list, set, frozenset)):
        for it in obj:
            total += deep_nbytes(it, _seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            total += deep_nbytes(k, _seen)
            total += deep_nbytes(v, _seen)
    return int(total)


class _Entry:
    """One cached value plus its accounted byte size and recency stamp."""

    __slots__ = ('value', 'nbytes', 'stamp')

    def __init__(self, value: Any, nbytes: int, stamp: int) -> None:
        self.value = value
        self.nbytes = int(nbytes)
        self.stamp = int(stamp)


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

class ByteBudgetedLRU:
    """A thread-safe, byte-budgeted, LRU-evicting memoization cache.

    Bounds retained bytes both locally (this cache's ``max_bytes``) and
    collectively (the shared :func:`get_cache_budget` ceiling).  All
    instances are drained by :func:`lumenairy.clear_asm_caches`.

    Parameters
    ----------
    name : str
        Human-readable cache name, surfaced by :func:`cache_report`.
    max_bytes : int or None, default None
        This cache's LOCAL retained-byte ceiling.

        * ``None`` -- bounded only by the collective global budget (a
          sensible default for small caches).
        * ``0`` -- DISABLED: :meth:`put` stores nothing, :meth:`get`
          always misses.  This is how an N^2-scale cache ships "off by
          default"; the caller flips it on with :meth:`set_budget`.
        * ``> 0`` -- a fixed local ceiling in bytes.
    size_func : callable, optional
        ``value -> int`` byte sizer.  Defaults to :func:`deep_nbytes`.
    register : bool, default True
        Enroll in the live registry (so ``clear_asm_caches`` drains it and
        it counts against the global budget).  Off only for throwaway
        instances in isolated tests.
    """

    def __init__(self, name: str,
                 max_bytes: Optional[int] = None,
                 *,
                 size_func: Optional[Callable[[Any], int]] = None,
                 register: bool = True) -> None:
        self._name = str(name)
        self._max_bytes = self._coerce_max_bytes(max_bytes)
        self._size_func = size_func if size_func is not None else deep_nbytes
        # key -> _Entry, ordered oldest (front) -> newest (back).
        self._store: "OrderedDict[Any, _Entry]" = OrderedDict()
        self._retained = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        # Shared subsystem mutex (see module docstring): a single lock so
        # cross-cache global eviction cannot dead-lock.
        self._lock = _BUDGET_MUTEX
        if register:
            with _BUDGET_MUTEX:
                _LIVE_CACHES.add(self)

    # -- config ------------------------------------------------------------

    @staticmethod
    def _coerce_max_bytes(max_bytes: Optional[int]) -> Optional[int]:
        if max_bytes is None:
            return None
        max_bytes = int(max_bytes)
        if max_bytes < 0:
            raise ValueError(
                f"max_bytes must be >= 0 or None (got {max_bytes!r}).")
        return max_bytes

    def _effective_max_bytes(self) -> int:
        """The live local ceiling in bytes (global budget when ``None``)."""
        if self._max_bytes is None:
            return get_cache_budget_bytes()
        return self._max_bytes

    def set_budget(self, max_bytes: Optional[int]) -> None:
        """Change this cache's local byte ceiling, evicting to fit at once.

        ``0`` disables + clears the cache; ``None`` reverts to
        global-budget-only; ``> 0`` sets a fixed local ceiling.  Global
        eviction is re-run so the collective ceiling is also respected.
        """
        with self._lock:
            self._max_bytes = self._coerce_max_bytes(max_bytes)
            local_cap = self._effective_max_bytes()
            if local_cap <= 0:
                self._store.clear()
                self._retained = 0
            else:
                while self._retained > local_cap and self._store:
                    _k, ev = self._store.popitem(last=False)
                    self._retained -= ev.nbytes
                    self._evictions += 1
            _enforce_global_budget_locked()

    # -- core cache ops ----------------------------------------------------

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the cached value for ``key`` (marking it most-recently-
        used), or ``default`` on a miss.  Updates hit/miss counters."""
        global _GLOBAL_SEQ
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return default
            _GLOBAL_SEQ += 1
            entry.stamp = _GLOBAL_SEQ
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: Any, value: Any, *,
            nbytes: Optional[int] = None) -> bool:
        """Insert / replace ``key -> value``; evict LRU to stay under both
        the local and global budgets.

        Parameters
        ----------
        nbytes : int, optional
            Pre-computed byte size of ``value`` (skips the ``size_func``
            call -- the caller often already knows it).

        Returns
        -------
        bool
            True if the value was stored, False if it was skipped (cache
            disabled, or a single value larger than the hard ceiling that
            could never be retained).  A skip is not an error: a
            subsequent :meth:`get` simply misses and the caller recomputes.
        """
        global _GLOBAL_SEQ
        with self._lock:
            old = self._store.pop(key, None)
            if old is not None:
                self._retained -= old.nbytes
            local_cap = self._effective_max_bytes()
            if local_cap <= 0:
                return False  # disabled: store nothing
            nb = int(nbytes) if nbytes is not None else int(
                self._size_func(value))
            hard_cap = min(local_cap, get_cache_budget_bytes())
            if nb > hard_cap:
                # Single entry can never be retained under the ceiling;
                # skip so we do not thrash inserting-then-evicting it.
                return False
            _GLOBAL_SEQ += 1
            self._store[key] = _Entry(value, nb, _GLOBAL_SEQ)
            self._retained += nb
            # Local LRU eviction: drop oldest until under the local cap.
            # The just-inserted entry (nb <= local_cap) is at the back and
            # is never evicted by this loop (len > 1 guard).
            while self._retained > local_cap and len(self._store) > 1:
                _k, ev = self._store.popitem(last=False)
                self._retained -= ev.nbytes
                self._evictions += 1
            # Global LRU eviction across every live budgeted cache.
            _enforce_global_budget_locked()
            return True

    def get_or_compute(self, key: Any, compute: Callable[[], Any], *,
                       nbytes: Optional[int] = None) -> Any:
        """Memoization convenience: return the cached value for ``key`` or
        compute it via ``compute()`` (OUTSIDE the lock), store, and return
        it.  ``compute`` runs unlocked so a slow build never blocks other
        caches' bookkeeping."""
        sentinel = object()
        hit = self.get(key, sentinel)
        if hit is not sentinel:
            return hit
        value = compute()
        self.put(key, value, nbytes=nbytes)
        return value

    # -- lifecycle ---------------------------------------------------------

    def clear(self) -> None:
        """Drop every entry, releasing all retained bytes.  Hit/miss
        counters are preserved (they are lifetime stats)."""
        with self._lock:
            self._store.clear()
            self._retained = 0

    def release(self) -> None:
        """Alias for :meth:`clear` -- release all retained memory.  Named
        for symmetry with other releasable library resources."""
        self.clear()

    # -- introspection -----------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Return this cache's footprint + stats:
        ``{name, retained_bytes, entries, hits, misses, evictions,
        max_bytes}``."""
        with self._lock:
            return {
                'name': self._name,
                'retained_bytes': int(self._retained),
                'entries': len(self._store),
                'hits': int(self._hits),
                'misses': int(self._misses),
                'evictions': int(self._evictions),
                'max_bytes': (None if self._max_bytes is None
                              else int(self._max_bytes)),
            }

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_bytes(self) -> Optional[int]:
        return self._max_bytes

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return int(self._retained)

    @property
    def hits(self) -> int:
        return int(self._hits)

    @property
    def misses(self) -> int:
        return int(self._misses)

    @property
    def evictions(self) -> int:
        return int(self._evictions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: Any) -> bool:
        # Pure membership: does NOT touch LRU order or hit/miss counters.
        with self._lock:
            return key in self._store

    def __repr__(self) -> str:
        return (f"ByteBudgetedLRU(name={self._name!r}, "
                f"entries={len(self)}, "
                f"retained_bytes={self.retained_bytes}, "
                f"max_bytes={self._max_bytes})")


# ---------------------------------------------------------------------------
# Global (cross-cache) LRU eviction
# ---------------------------------------------------------------------------

def _enforce_global_budget_locked() -> None:
    """Evict globally-least-recently-used entries until the collective
    retained bytes fit the global budget.  MUST be called while holding
    :data:`_BUDGET_MUTEX`.

    The globally-oldest entry is the minimum-stamp front entry across all
    live caches: within any cache the OrderedDict front carries that
    cache's smallest stamp (every access moves-to-end + re-stamps), so
    comparing per-cache front stamps yields the global minimum.
    """
    budget = get_cache_budget_bytes()
    caches = list(_LIVE_CACHES)
    total = sum(c._retained for c in caches)
    # Bounded by the number of live entries; each iteration pops exactly
    # one, so this terminates.
    while total > budget:
        victim_cache = None
        victim_key = None
        victim_entry = None
        for c in caches:
            store = c._store
            if not store:
                continue
            k = next(iter(store))          # oldest (front) key in this cache
            e = store[k]
            if victim_entry is None or e.stamp < victim_entry.stamp:
                victim_cache = c
                victim_key = k
                victim_entry = e
        if victim_cache is None:
            break  # nothing left to evict (all caches empty)
        del victim_cache._store[victim_key]
        victim_cache._retained -= victim_entry.nbytes
        victim_cache._evictions += 1
        total -= victim_entry.nbytes


# ---------------------------------------------------------------------------
# Module-level introspection + registry drain
# ---------------------------------------------------------------------------

def cache_report() -> Dict[str, Any]:
    """Introspect every live byte-budgeted cache.

    Returns
    -------
    dict
        ``{'caches': {name: per_cache_report}, 'total_retained_bytes',
        'budget_bytes', 'budget_mb', 'n_caches'}``.  ``total_retained_bytes``
        is the exact sum of the per-cache ``retained_bytes``; when two
        caches share a name the second is keyed ``name#2`` so no cache is
        dropped from the map.
    """
    with _BUDGET_MUTEX:
        caches = list(_LIVE_CACHES)
        per: Dict[str, Any] = {}
        total = 0
        for c in caches:
            r = c.report()
            total += r['retained_bytes']
            key = r['name']
            if key in per:
                i = 2
                while f'{key}#{i}' in per:
                    i += 1
                key = f'{key}#{i}'
            per[key] = r
        budget = get_cache_budget_bytes()
        return {
            'caches': per,
            'total_retained_bytes': int(total),
            'budget_bytes': int(budget),
            'budget_mb': budget / (1024.0 * 1024.0),
            'n_caches': len(caches),
        }


def clear_all_byte_budgeted_caches() -> None:
    """Drain every live byte-budgeted cache.  Enrolled with the central
    cache-clearer registry so :func:`lumenairy.clear_asm_caches` (and the
    ``lumenairy_context(clear_caches_on_exit=True)`` manager) release all
    retained cache memory in one operation."""
    with _BUDGET_MUTEX:
        for c in list(_LIVE_CACHES):
            c.clear()


# Enroll the single collective clearer with the central registry so the
# v4.16.1 meta-pin walker stays green and clear_asm_caches drains us.
try:
    from ._cache_registry import (
        register_cache_clearer as _register_cache_clearer,
    )

    _register_cache_clearer('byte_budgeted_lru', clear_all_byte_budgeted_caches)
except ImportError:
    pass


__all__ = [
    'ByteBudgetedLRU',
    'cache_report',
    'get_cache_budget',
    'set_cache_budget',
]
