"""
lumenairy.elements.pmm._stack2d_cache -- the priced, bounded, thread-safe
per-layer cache behind :class:`~lumenairy.elements.pmm.stack2d.PMM2DStackHybrid`.
=========================================================================

Why this module exists (measured, 2026-08-16, tesla-ryzen 24-core,
OpenBLAS 0.3.31, ``OPENBLAS_NUM_THREADS=1``, numpy 2.4.4):

* The stack's previous ``_geom_cache`` was a bare ``dict`` that grew WITHOUT
  BOUND.  A dispersive ``solve_vs_wavelength`` mints one entry per wavelength
  (a dispersive tile's bytes are part of the key by design), so the footprint
  is LINEAR in sweep length: measured 12 points -> 8.41 MB, 24 points ->
  16.82 MB at ``n_orders=4``; 0.701 MB per entry at ``n_orders=4``, 1.523 MB at
  ``n_orders=5``, 2.859 MB at ``n_orders=6``.  Extrapolating the measured
  per-entry cost, a 500-point dispersive sweep would retain 0.35 / 0.76 /
  1.43 GB respectively -- the exact unbounded-retention shape audit P3-32
  already fixed on the 1-D :class:`_PreparedPMMStack`.
* ``solve_vs_wavelength`` hands each worker a ``copy.copy(self)`` SHALLOW
  clone, so every worker shared the one ``_geom_cache`` dict object (measured:
  ``sub._geom_cache is st._geom_cache`` -> ``True``) with no lock.

So the cache is given a price and a lock here rather than in the solver.

Pricing (the refuse-never-degrade contract)
-------------------------------------------
The budget is read from :func:`lumenairy.memory.get_ram_budget` AT QUERY TIME
(so a :func:`lumenairy.set_max_ram` override applies immediately, and the
budget tracks a box's actual free memory rather than a number frozen at
construction), scaled by ``budget_fraction`` and floored at ``min_budget``.

Eviction never touches an entry minted by the CURRENT solve: entries carry a
generation stamp, :meth:`new_generation` bumps it at the top of every solve,
and eviction drains strictly older generations in LRU order.  If the current
solve's own working set alone exceeds the budget, the cache REFUSES to store
further entries (the value is still returned to the caller and used) instead of
evicting a live entry it is about to need again -- the GRAM refuse-never-degrade
precedent.  A refusal costs recomputation, never a wrong answer: a recomputed
entry is byte-identical (the same LAPACK on the same input bytes).

Stored values are frozen non-writeable by the caller (``_freeze_cached``, W7
A13), so handing out the stored object by identity is safe.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

__all__ = ["LayerCache", "cached_nbytes"]

#: Default share of the library RAM budget this cache may occupy.
_DEFAULT_BUDGET_FRACTION = 0.05
#: Floor so a modest box still caches a normal stack.  Sized from the measured
#: per-entry cost above: 256 MB holds ~365 entries at ``n_orders=4``
#: (0.701 MB), ~168 at ``n_orders=5`` (1.523 MB), ~89 at ``n_orders=6``
#: (2.859 MB) -- comfortably more than the ``n_layers`` distinct entries a
#: single solve can hold live at once for any stack this solver can afford to
#: build (``_validate_cell_cost`` caps the nodal DOF well below that).
_DEFAULT_MIN_BUDGET = 256 * 1024 ** 2


def cached_nbytes(obj):
    """Total RETAINED bytes reachable from a cached value.

    Walks tuples / lists / dicts and plain objects' ``__dict__`` (the
    ``_build_axis`` axis records are small objects holding arrays) -- the same
    traversal :func:`~lumenairy.elements.pmm._core._freeze_cached` performs.

    Every array is charged against the ROOT buffer it looks through, counted
    ONCE.  A cached value that holds a slice as well as its parent retains one
    allocation, not two, and charging both would price the cache out early on
    exactly the values (the projected-operator dicts) that share buffers.
    Conversely a lone view keeps its whole parent buffer alive, so the ROOT's
    size -- not the view's -- is what retention actually costs.
    """
    seen_containers = set()
    seen_buffers = set()
    total = 0
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, np.ndarray):
            root = o
            while isinstance(getattr(root, "base", None), np.ndarray):
                root = root.base
            if id(root) in seen_buffers:
                continue
            seen_buffers.add(id(root))
            total += int(root.nbytes)
            continue
        if id(o) in seen_containers:
            continue
        seen_containers.add(id(o))
        if isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif hasattr(o, "__dict__"):
            stack.extend(vars(o).values())
    return total


class LayerCache:
    """Content-keyed LRU cache with a RAM-priced byte budget and a lock.

    Parameters
    ----------
    budget_fraction : float, optional
        Share of :func:`lumenairy.memory.get_ram_budget` this cache may hold.
    min_budget : int, optional
        Byte floor for the budget, applied when the scaled RAM budget is
        smaller (a small box still caches a normal stack).
    max_bytes : int, optional
        Hard override; when given, ``budget_fraction`` / ``min_budget`` are
        ignored and the budget is exactly this many bytes.  The test hook for
        forcing the priced-out arm (standards rule 4: force the precondition,
        assert the gate separately) and the caller-facing knob for pinning a
        cache size.
    """

    def __init__(self, *, budget_fraction=_DEFAULT_BUDGET_FRACTION,
                 min_budget=_DEFAULT_MIN_BUDGET, max_bytes=None):
        self._lock = threading.Lock()
        self._d = OrderedDict()          # key -> (value, nbytes, generation)
        self._bytes = 0
        self._gen = 0
        self.budget_fraction = float(budget_fraction)
        self.min_budget = int(min_budget)
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        # Counters: the two-sided pricing claim reads these (engaged vs
        # refused), so they are part of the contract, not debug scaffolding.
        self.n_hits = 0
        self.n_misses = 0
        self.n_refused = 0
        self.n_evicted = 0

    # ------------------------------------------------------------------ #
    # pricing
    # ------------------------------------------------------------------ #
    def budget(self):
        """Current byte budget, priced from the library RAM budget NOW."""
        if self.max_bytes is not None:
            return self.max_bytes
        from ...memory import get_ram_budget
        try:
            ram = int(get_ram_budget())
        except Exception:                      # pragma: no cover - psutil-less
            ram = 0
        return max(self.min_budget, int(self.budget_fraction * ram))

    @property
    def nbytes(self):
        """Bytes currently retained (the measured footprint, not an estimate)."""
        with self._lock:
            return self._bytes

    def __len__(self):
        with self._lock:
            return len(self._d)

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def values(self):
        """Snapshot list of the cached VALUES (the ``dict.values`` the bare
        dict this class replaced offered; a list, so iteration cannot race a
        concurrent eviction)."""
        with self._lock:
            return [v[0] for v in self._d.values()]

    def keys(self):
        """Snapshot list of the cached keys."""
        with self._lock:
            return list(self._d)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def new_generation(self):
        """Stamp a new generation (called at the top of every solve).

        Entries minted from here on are exempt from eviction until the NEXT
        generation starts, which is what makes "a single solve never evicts an
        entry it is about to reuse" true by construction rather than by sizing
        argument.
        """
        with self._lock:
            self._gen += 1
            return self._gen

    def clear(self):
        """Drop everything.  The next solve recomputes byte-identically."""
        with self._lock:
            self._d.clear()
            self._bytes = 0

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #
    def get(self, key):
        """Return the cached value for ``key`` (LRU-refreshed) or ``None``."""
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                self.n_misses += 1
                return None
            self._d.move_to_end(key)
            self.n_hits += 1
            return hit[0]

    def put(self, key, value, nbytes=None):
        """Store ``value`` under ``key`` and return ``value`` unchanged.

        Storage may be REFUSED (footprint would exceed the budget with nothing
        older left to evict); the value is returned either way, so a refusal
        costs recomputation on the next miss and nothing else.
        """
        nb = cached_nbytes(value) if nbytes is None else int(nbytes)
        budget = self.budget()
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            # Evict strictly OLDER generations, LRU-first, to make room.
            if self._bytes + nb > budget:
                for k in list(self._d):
                    if self._bytes + nb <= budget:
                        break
                    if self._d[k][2] == self._gen:
                        continue           # current solve: never evict
                    self._bytes -= self._d.pop(k)[1]
                    self.n_evicted += 1
            if self._bytes + nb > budget:
                # Refuse rather than degrade: the live working set owns the
                # budget, and the caller already holds the value.
                self.n_refused += 1
                return value
            self._d[key] = (value, nb, self._gen)
            self._bytes += nb
            return value

    def stats(self):
        """Snapshot dict -- entries / bytes / budget / hit / miss / refused /
        evicted.  Read by the pricing tests and by callers sizing a sweep."""
        with self._lock:
            return dict(entries=len(self._d), nbytes=self._bytes,
                        generation=self._gen, hits=self.n_hits,
                        misses=self.n_misses, refused=self.n_refused,
                        evicted=self.n_evicted)
