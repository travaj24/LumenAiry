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

import functools
import threading
import warnings
from typing import Any, Callable, Dict, List, Tuple

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


def _clearer_identity(fn: Any) -> Tuple[Any, ...]:
    """A key that is EQUAL across ``importlib.reload`` and DIFFERENT for two
    genuinely distinct callables.

    ``importlib.reload`` re-executes the module body, producing a new function
    object compiled from the same source line of the same file -- so
    ``(module, qualname, co_filename, co_firstlineno)`` is unchanged.  Two
    different functions (including two lambdas written on different lines of
    the same module, which share ``__qualname__ == '<lambda>'``) differ in
    ``co_firstlineno``.

    Callables with no code object of their own are resolved to one that does,
    so that "two different clearers" stays audible for them too: a
    ``functools.partial`` keys off the function it wraps, and an instance with
    ``__call__`` off its class's ``__call__``.  (Measured: keying both on
    ``type(fn).__name__`` -- the v5.46 form -- made ALL partials compare equal,
    so a partial of ``clear_a`` and a partial of ``clear_b`` collided
    SILENTLY, which is the very defect this helper exists to make audible.)

    Two *instances* of the same callable class, and two bound methods of the
    same class, still compare equal.  That is deliberate and is the price of
    reload-idempotence: nothing about an instance survives a reload, so any
    per-instance discriminator (``id``, ``repr``) would make every reload warn.
    Register per-instance clearers under distinct names.
    """
    seen = 0
    while isinstance(fn, functools.partial) and seen < 8:
        fn = fn.func                     # key off the wrapped callable
        seen += 1
    code = getattr(fn, '__code__', None)
    module = getattr(fn, '__module__', None)
    qualname = getattr(fn, '__qualname__', None)
    if code is not None:
        return (module, qualname,
                getattr(code, 'co_filename', None),
                getattr(code, 'co_firstlineno', None))
    call_code = getattr(getattr(type(fn), '__call__', None), '__code__', None)
    if call_code is not None:
        return (getattr(type(fn), '__module__', None),
                f"{getattr(type(fn), '__qualname__', None)}.__call__",
                getattr(call_code, 'co_filename', None),
                getattr(call_code, 'co_firstlineno', None))
    return (module, qualname, type(fn).__name__, None)


def _describe_clearer(fn: Any) -> str:
    """``module.qualname (file:line)`` for a collision warning; falls back to
    ``repr`` for a callable with no code object."""
    module = getattr(fn, '__module__', None)
    qualname = getattr(fn, '__qualname__', None)
    if qualname is None:
        return repr(fn)
    code = getattr(fn, '__code__', None)
    where = ''
    if code is not None:
        where = (f" ({getattr(code, 'co_filename', '?')}"
                 f":{getattr(code, 'co_firstlineno', '?')})")
    return f"{module}.{qualname}{where}"


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

    Warns
    -----
    RuntimeWarning
        If ``name`` is already registered to a DIFFERENT callable.  The
        first registration wins (behaviour is unchanged), so the second
        cache would be left permanently unclearable -- in the module whose
        whole purpose is to retire the "fix N, miss N+1" cache-clear
        pattern.  v5.46 (audit Z4): pre-v5.46 the collision was silent.

    Notes
    -----
    Multiple imports of the same module (e.g. via
    :func:`importlib.reload`) trigger re-registration.  The registry
    treats a duplicate name from the SAME call site as a no-op rather
    than warning -- warnings churn during interactive development
    sessions.  "Same call site" is decided by
    :func:`_clearer_identity` (module + qualname + source file + first
    line), which a reload preserves and a genuinely different function
    does not.  The "name" key is therefore idempotent and stable across
    reloads while a real collision is audible.
    """
    with _REGISTRY_LOCK:
        existing = _CACHE_CLEARERS.get(name)
        if existing is not None:
            # Idempotent: ignore re-registration of the same name FROM THE
            # SAME SITE.  Re-importing the owning module (e.g. during an
            # ``importlib.reload`` cycle, or during a test that imports the
            # module twice) re-triggers the ``register_cache_clearer`` call
            # -- accept silently.
            if (existing is clear_fn
                    or _clearer_identity(existing) == _clearer_identity(clear_fn)):
                return
            old_id = _describe_clearer(existing)
            new_id = _describe_clearer(clear_fn)
            warnings.warn(
                f"register_cache_clearer: the name {name!r} is already "
                f"registered to a different clearer ({old_id}); the new one "
                f"({new_id}) is IGNORED, so its cache will never be cleared "
                f"by clear_asm_caches() / "
                f"lumenairy_context(clear_caches_on_exit=True).  Register it "
                f"under a unique name.  Already registered: "
                f"{sorted(_CACHE_CLEARERS)}.",
                RuntimeWarning, stacklevel=2)
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

    Warns
    -----
    RuntimeWarning
        Once, at the end of the walk, naming every clearer that raised.
        v5.46 (audit Z4): the failures were previously swallowed with no
        signal at all, so a caller who ran ``clear_asm_caches()`` to free
        RAM before a large allocation could not tell that a cache had
        stayed full -- the memory is still held and the next allocation
        still OOMs, but nothing said why.  The walk itself is still
        best-effort and does not raise.
    """
    # Snapshot under the lock so a concurrent registration during the
    # walk doesn't produce a "dictionary changed size during
    # iteration" RuntimeError.  The clear functions themselves take
    # their own per-cache locks; we explicitly release the registry
    # lock before calling them to avoid lock-order inversion with
    # any cache-internal lock.
    with _REGISTRY_LOCK:
        items = list(_CACHE_CLEARERS.items())
    failures: List[str] = []
    for name, fn in items:
        try:
            fn()
        except (ImportError, RuntimeError, AttributeError) as exc:
            # Same narrowed-except as the v4.15 fan-out.  A single
            # clearer failure must not strand the rest of the chain.
            # We swallow rather than re-raise to preserve the v4.15
            # contract that ``clear_asm_caches`` is best-effort -- but we
            # COLLECT it, so the caller learns that the memory it asked
            # for was not actually released (audit Z4).
            failures.append(f"{name} ({type(exc).__name__}: {exc})")
    if failures:
        warnings.warn(
            f"clear_all_registered_caches: {len(failures)} of {len(items)} "
            f"registered cache clearers raised and were skipped, so those "
            f"caches are still holding memory: "
            f"{'; '.join(failures)}.  The walk is best-effort by contract "
            f"(the remaining clearers all ran), but a caller clearing "
            f"caches to make room for a large allocation should not assume "
            f"the memory was freed.",
            RuntimeWarning, stacklevel=2)


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
