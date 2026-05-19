"""v4.16.0 Agent D: central cache registry (ROADMAP #15).

Pins
----

The v4.14.2 / v4.14.3 audit catalogued a recurring "fix N, miss N+1"
sibling-gap meta-pattern in the cache-clear domain.  Every new cache
addition forced the author to remember to extend the lazy-import fan-
out in ``clear_asm_caches``.  v4.14.3 added the 8th cache and the
meta-pattern recurred 5 ways across v4.14.x alone.

v4.16.0 retires the fan-out: cache authors register their clear
function via ``register_cache_clearer`` at module-import time;
``clear_asm_caches`` walks the registry rather than enumerating clear
calls by hand.  Counter-measure ratio: 1 registry prevents N future
sibling-gap recurrences.

The tests below pin:

* registration mechanics (add, double-register, list);
* the central walker (``clear_all_registered_caches``);
* the new ``clear_asm_caches`` implementation actually walks the
  registry;
* every known v4.16.0 cache is registered (regression baseline);
* the v4.14.1 cache-clear meta-pin still passes (regression check
  that the refactor did NOT break the external contract).

Author: Andrew Traverso -- v4.16.0 / Agent D
"""
from __future__ import annotations

import pytest

import lumenairy as la
from lumenairy import _cache_registry as _cr


# ===========================================================================
# Registration mechanics
# ===========================================================================


@pytest.fixture
def _synth_cache():
    """Provide a fresh synthetic clear-function and unique registry
    key; teardown removes it so the production registry is not
    permanently mutated by the test.
    """
    state = {'cleared': 0}

    def _synth_clear():
        state['cleared'] += 1

    synth_name = '__test_v4_16_0_synth_cache__'
    yield synth_name, _synth_clear, state
    _cr._unregister_for_test(synth_name)


def test_register_cache_clearer_adds_to_registry(_synth_cache):
    """Registering a new clearer adds it to the live registry."""
    name, fn, _state = _synth_cache
    assert name not in _cr.list_registered_cache_clearers()
    la.register_cache_clearer(name, fn)
    assert name in _cr.list_registered_cache_clearers()


def test_double_registration_is_no_op(_synth_cache):
    """Re-registering the same name accepts silently (idempotent).

    Module reloads can re-trigger the ``register_cache_clearer``
    call; the registry must treat the duplicate as a no-op rather
    than warning or raising.
    """
    name, fn, _state = _synth_cache
    la.register_cache_clearer(name, fn)
    # Second call: same name, possibly different function.
    la.register_cache_clearer(name, fn)
    listing = _cr.list_registered_cache_clearers()
    # Count occurrences -- must be exactly 1.
    assert listing.count(name) == 1, (
        f"Double-registration left {listing.count(name)} entries; "
        f"expected exactly 1 (idempotent).")


def test_clear_asm_caches_now_walks_registry(_synth_cache):
    """A clearer registered AFTER ``propagation.py`` was imported
    must still be invoked by ``clear_asm_caches``.

    This is the core counter-measure: pre-v4.16 a new cache had to
    edit ``clear_asm_caches`` by hand; v4.16 the registry walk picks
    it up automatically.
    """
    name, fn, state = _synth_cache
    la.register_cache_clearer(name, fn)

    # Initially no clears have happened.
    assert state['cleared'] == 0

    # Walk should invoke our synthetic clearer.
    la.clear_asm_caches()
    assert state['cleared'] >= 1, (
        f"clear_asm_caches did not invoke registered clearer "
        f"{name!r}; state['cleared']={state['cleared']}.  The "
        f"registry walk path is broken.")


def test_list_returns_sorted():
    """``list_registered_cache_clearers`` returns a sorted list."""
    listing = _cr.list_registered_cache_clearers()
    assert listing == sorted(listing), (
        "list_registered_cache_clearers should return sorted names "
        "for deterministic CI output.")


# ===========================================================================
# Pin the v4.16.0 known caches (regression baseline)
# ===========================================================================
#
# These names are the canonical v4.16.0 cache identifiers.  A future
# release that drops or renames one of these must update this list
# explicitly; the test fails loudly rather than letting the rename
# slip in unnoticed.

V4_16_0_KNOWN_CACHES = {
    'asm_local',
    'lg_mode_stack',
    'lg_polynomial_items',
    'phase_retrieval_kernels',
    'propagate_system_jax',
    'through_focus_scan_jax',
    'trace_jax',
    'wrapper_merit_meshgrid',
    'zernike_basis',
}


def test_all_known_caches_are_registered():
    """Every cache that v4.16.0 commits to dispatching via the registry
    is in fact registered at module-import time.

    This is the inversion of the v4.14.1 ``test_v4_14_1_dispatcher_pin_
    cache_clears`` walker pin: that test asserted every
    submodule's-__all__ ``clear_*`` name is re-exported at top level;
    this test asserts each one is actually wired into the registry.
    """
    registered = set(_cr.list_registered_cache_clearers())
    missing = V4_16_0_KNOWN_CACHES - registered
    assert not missing, (
        f"v4.16.0 known cache names missing from registry: "
        f"{sorted(missing)}.  Each module owning one of these caches "
        f"must call ``register_cache_clearer`` at module-import time."
    )


def test_at_least_eight_caches_registered():
    """The v4.14.3 baseline had 8 sibling caches + the local ASM
    block.  v4.16.0 must therefore register at least 9 clearers
    (one per cache).  Pin the floor so a partial-import regression
    that silently drops a registration is detected immediately.
    """
    listing = _cr.list_registered_cache_clearers()
    assert len(listing) >= 9, (
        f"Only {len(listing)} cache clearers registered: {listing}.  "
        f"v4.16.0 expects at least 9 (one per known cache).")


# ===========================================================================
# Regression: v4.14.1 meta-pin must still pass
# ===========================================================================


def test_v4_14_1_cache_clear_meta_pin_still_passes():
    """The v4.14.1 dispatcher pin walks every submodule's ``__all__``
    for ``clear_*`` names and asserts each is re-exported at top
    level.  The v4.16.0 registry refactor preserves the external
    contract -- ``clear_asm_caches`` and every sibling clearer are
    still importable from ``lumenairy``.
    """
    expected = [
        'clear_asm_caches',
        'clear_zernike_basis_cache',
        'clear_lg_polynomial_cache',
        'clear_lg_mode_stack_cache',
        'clear_through_focus_scan_jax_cache',
        'clear_trace_jax_cache',
        'clear_propagate_system_jax_cache',
        'clear_phase_retrieval_caches',
    ]
    for name in expected:
        assert hasattr(la, name), (
            f"v4.14.1 meta-pin regression: la.{name} missing after "
            f"v4.16.0 registry refactor.")
        assert name in la.__all__, (
            f"v4.14.1 meta-pin regression: {name!r} dropped from "
            f"lumenairy.__all__ after v4.16.0 registry refactor.")
        assert callable(getattr(la, name)), (
            f"v4.14.1 meta-pin regression: la.{name} no longer "
            f"callable.")


# ===========================================================================
# Top-level export surface
# ===========================================================================


def test_register_cache_clearer_at_top_level():
    """``register_cache_clearer`` is part of the v4.16.0 public API."""
    assert hasattr(la, 'register_cache_clearer')
    assert callable(la.register_cache_clearer)
    assert 'register_cache_clearer' in la.__all__


def test_list_registered_cache_clearers_at_top_level():
    """``list_registered_cache_clearers`` is part of the v4.16.0
    public API."""
    assert hasattr(la, 'list_registered_cache_clearers')
    assert callable(la.list_registered_cache_clearers)
    assert 'list_registered_cache_clearers' in la.__all__


def test_clear_all_registered_caches_swallows_exceptions(_synth_cache):
    """If a registered clear function raises ``ImportError`` /
    ``RuntimeError`` / ``AttributeError`` (the v4.13 Phase-2 narrowed
    set), the walker must not strand sibling clearers.
    """
    name, _fn, _state = _synth_cache
    misbehaving = {'count': 0}

    def _misbehaving_clear():
        misbehaving['count'] += 1
        raise ImportError('synthetic failure for test')

    la.register_cache_clearer(name, _misbehaving_clear)

    # Should not raise -- ImportError is in the swallowed set.
    la.clear_asm_caches()

    assert misbehaving['count'] >= 1, (
        "Misbehaving clearer should have been invoked at least once.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
