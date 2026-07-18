"""G08 / S4-15 (AUDIT_V5_24_2) -- cache-hygiene fixes.

Two defects in the cache-drain path:

1. ``lumenairy_context(clear_caches_on_exit=True)``'s defence-in-depth
   fallback (used when the canonical ``clear_asm_caches`` re-export is
   unavailable) hand-listed only 7 sibling clearers and OMITTED
   berreman / pmm / rcwa / glass / wrapper_merit / eme_jax / bluestein.
   The central registry exists precisely to retire this "fix N, miss
   N+1" pattern; the fallback now walks ``clear_all_registered_caches``.

2. ``_JAX_IFT_SOLVER_CACHE`` (the compiled JAX Newton-IFT solver
   singleton) was never registered with the central registry, so an
   explicit cache drain left its pinned XLA executable resident.  It is
   now enrolled via ``clear_jax_ift_solver_cache``.

Both tests are pure-Python (no JAX / GUI required).
"""
from __future__ import annotations

import lumenairy as la
from lumenairy import _cache_registry
from lumenairy._context import lumenairy_context

# =========================================================================
# (2) _JAX_IFT_SOLVER_CACHE clearer is registered + actually resets.
# =========================================================================

def test_jax_ift_solver_clearer_is_registered():
    """The JAX Newton-IFT solver singleton must now enrol a clearer so a
    cache drain reclaims its pinned XLA executable (pre-fix: absent)."""
    assert 'jax_ift_solver' in la.list_registered_cache_clearers()


def test_clear_jax_ift_solver_cache_resets_singleton():
    """Setting the module singleton to a sentinel and invoking the
    registered clearer (via the public registry walk) must reset it to
    None -- proving the clearer is wired, not just named."""
    import lumenairy.propagators.asymptotic_jax_twin as twin

    sentinel = object()
    twin._JAX_IFT_SOLVER_CACHE = sentinel
    # Direct clearer.
    twin.clear_jax_ift_solver_cache()
    assert twin._JAX_IFT_SOLVER_CACHE is None

    # And the registry walk reaches it too.
    twin._JAX_IFT_SOLVER_CACHE = sentinel
    _cache_registry.clear_all_registered_caches()
    assert twin._JAX_IFT_SOLVER_CACHE is None


# =========================================================================
# (1) The context fallback drains the WHOLE registry, not a hand-list.
# =========================================================================

def test_context_fallback_drains_full_registry(monkeypatch):
    """When the canonical ``clear_asm_caches`` path is unavailable (here:
    forced to raise), the ``clear_caches_on_exit=True`` fallback must
    still drain a registered clearer that was NOT in the pre-fix
    7-clearer hand-list.

    Pre-fix oracle: the fallback enumerated only
    zernike/lg_polynomial/trace_jax/propagate_system/phase_retrieval/
    through_focus/lg_mode_stack, so a synthetic registered clearer would
    be skipped and ``calls`` would stay 0.  Post-fix it walks
    ``clear_all_registered_caches`` and the synthetic clearer fires.
    """
    calls = [0]

    def _synthetic_clearer():
        calls[0] += 1

    name = '_g08_s4_15_synthetic_clearer'
    la.register_cache_clearer(name, _synthetic_clearer)
    try:
        # Force the canonical chain to fail so the fallback runs.
        import lumenairy.propagators.propagation as prop

        def _boom():
            raise RuntimeError('forced: canonical chain unavailable')

        monkeypatch.setattr(prop, 'clear_asm_caches', _boom)

        with lumenairy_context(clear_caches_on_exit=True):
            pass

        assert calls[0] >= 1, (
            'S4-15 regression: the clear_caches_on_exit fallback did not '
            'drain a registered clearer outside the legacy 7-clearer '
            'hand-list.'
        )
    finally:
        _cache_registry._unregister_for_test(name)


def test_context_happy_path_still_drains_registry(monkeypatch):
    """Sanity: with the canonical path intact, a registered clearer is
    still drained (this path already walked the registry via
    ``clear_asm_caches`` -> ``clear_all_registered_caches``)."""
    calls = [0]
    name = '_g08_s4_15_synthetic_clearer_happy'
    la.register_cache_clearer(name, lambda: calls.__setitem__(0, calls[0] + 1))
    try:
        with lumenairy_context(clear_caches_on_exit=True):
            pass
        assert calls[0] >= 1
    finally:
        _cache_registry._unregister_for_test(name)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
