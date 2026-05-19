"""Central cache-clearer registry.

v4.16.0 retires the lazy-import fan-out in ``clear_asm_caches``.
Cache authors register their clear-function once via
:func:`register_cache_clearer` at module-import time;
``clear_asm_caches`` walks the registry rather than enumerating clear
calls by hand.

This is the counter-measure to the recurring "fix N, miss N+1" meta-
pattern in the cache-clear domain.  Pre-v4.16 every new cache had to
remember to thread a new lazy-import + try/except block into
``clear_asm_caches``; v4.14.3 added the 8th cache
(``_lg_polynomial_items``) and the v4.14.2 audit found the meta-
pattern had recurred 5 ways inside a single audit cycle.  Future
cache additions need only register; ``clear_asm_caches`` picks them
up automatically.

The registry intentionally lives in the package root (alongside
``_context.py`` and ``_deprecation.py``) rather than under
``propagators/`` so a module on any branch of the import graph can
register without dragging in the propagation layer at module-load
time.

Public API
----------
:func:`register_cache_clearer` -- module-import-time registration.
:func:`list_registered_cache_clearers` -- introspection helper.
:func:`clear_all_registered_caches` -- internal walker used by
    :func:`lumenairy.propagators.propagation.clear_asm_caches`.

Author: Andrew Traverso -- v4.16.0 / Agent D
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, List

# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

# Lock guarding the registry itself.  Registration happens at module-
# import time on the main thread, but ``clear_all_registered_caches``
# can be called from worker threads in long-running pipelines.  The
# lock keeps the dict iteration consistent against a concurrent
# (rare, but possible) ``register_cache_clearer`` call.
_REGISTRY_LOCK = threading.Lock()

_CACHE_CLEARERS: Dict[str, Callable[[], None]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_cache_clearer(name: str,
                           clear_fn: Callable[[], None]) -> None:
    """Register a cache-clear function.

    Called at module-import time by every module owning an LRU /
    OrderedDict cache.  Replaces the per-cache lazy-import + try/except
    block that pre-v4.16 ``clear_asm_caches`` accumulated as new
    caches were added.

    Parameters
    ----------
    name : str
        Canonical cache name.  Stable across releases so external
        introspection tools (and the meta-pin
        ``test_all_known_caches_are_registered``) can pin known names.
        Examples: ``'asm_kernel'``, ``'lg_mode_stack'``,
        ``'zernike_basis'``, ``'lg_polynomial_items'``,
        ``'through_focus_scan_jax'``, ``'propagate_system_jax'``,
        ``'phase_retrieval_kernels'``, ``'trace_jax'``,
        ``'wrapper_merit_meshgrid'``.
    clear_fn : callable
        Zero-arg function that clears the cache.  Called by
        :func:`clear_all_registered_caches` (and transitively by
        :func:`lumenairy.propagators.propagation.clear_asm_caches` and
        :func:`lumenairy.lumenairy_context` with
        ``clear_caches_on_exit=True``).

    Notes
    -----
    Multiple imports of the same module (e.g. via
    :func:`importlib.reload`) trigger re-registration.  The registry
    treats a duplicate name as a no-op rather than warning -- warnings
    churn during interactive development sessions.  The "name" key is
    therefore idempotent and stable across reloads.
    """
    with _REGISTRY_LOCK:
        if name in _CACHE_CLEARERS:
            # Idempotent: ignore re-registration of the same name.
            # Re-importing the owning module (e.g. during a
            # ``importlib.reload`` cycle, or during a test that
            # imports the module twice) re-triggers the
            # ``register_cache_clearer`` call -- accept silently.
            return
        _CACHE_CLEARERS[name] = clear_fn


def list_registered_cache_clearers() -> List[str]:
    """Return the sorted list of registered cache-clearer names.

    Useful for the ``test_all_known_caches_are_registered`` meta-pin
    and for users who want to introspect which caches the library is
    holding before deciding whether to call
    :func:`lumenairy.clear_asm_caches`.

    Returns
    -------
    list of str
        Sorted (alphabetical) cache names.
    """
    with _REGISTRY_LOCK:
        return sorted(_CACHE_CLEARERS.keys())


def clear_all_registered_caches() -> None:
    """Call every registered cache-clear function.

    Used internally by
    :func:`lumenairy.propagators.propagation.clear_asm_caches`; user
    code should call ``clear_asm_caches`` instead so the local
    propagation-layer caches (H, freq-grid, bandlimit, pyFFTW plans)
    are drained in the same operation.

    Each clear function is invoked inside its own try/except so a
    single failing clearer (rare; would indicate a partial install or
    a corrupted optional-dep state) doesn't strand the rest of the
    chain.  The narrowed-except tuple matches the pre-v4.16
    ``clear_asm_caches`` fan-out (``ImportError``, ``RuntimeError``,
    ``AttributeError``) to preserve back-compat behaviour: a registry
    walk should leave the cache state in exactly the same shape as
    the v4.15 fan-out did on the same failure mode.
    """
    # Snapshot under the lock so a concurrent registration during the
    # walk doesn't produce a "dictionary changed size during
    # iteration" RuntimeError.  The clear functions themselves take
    # their own per-cache locks; we explicitly release the registry
    # lock before calling them to avoid lock-order inversion with
    # any cache-internal lock.
    with _REGISTRY_LOCK:
        items = list(_CACHE_CLEARERS.items())
    for name, fn in items:
        try:
            fn()
        except (ImportError, RuntimeError, AttributeError):
            # Same narrowed-except as the v4.15 fan-out.  A single
            # clearer failure must not strand the rest of the chain.
            # We swallow rather than re-raise to preserve the v4.15
            # contract that ``clear_asm_caches`` is best-effort.
            pass


def _unregister_for_test(name: str) -> bool:
    """Test-only helper: remove a registered clearer by name.

    Returns True if the name was present and removed, False otherwise.
    NOT part of the public API -- exposed only for the
    ``test_double_registration_is_no_op`` and
    ``test_clear_asm_caches_now_walks_registry`` unit tests, which
    need to install/teardown a synthetic clearer without mutating
    the production registry state seen by sibling tests.
    """
    with _REGISTRY_LOCK:
        if name in _CACHE_CLEARERS:
            del _CACHE_CLEARERS[name]
            return True
        return False


__all__ = [
    'register_cache_clearer',
    'list_registered_cache_clearers',
    'clear_all_registered_caches',
]
