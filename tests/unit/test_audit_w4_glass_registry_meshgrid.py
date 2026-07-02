"""Wave-4 audit fixes: glass/registry cache lifecycle + meshgrid split.

Pins the v5.17.1 fixes for AUDIT_V5_17_0_2026_07_01_DEEP findings:

* **P2-42** -- the cache-enrollment meta-pin's ``endswith('_CACHE')``
  filter was case-SENSITIVE, so ``glass.py``'s lower-case module caches
  (``_glass_cache``, ``_glass_value_cache``) escaped discovery and their
  missing enrollment went unflagged.  The walker now matches the suffix
  case-insensitively; these tests pin that glass.py's caches ARE
  discovered and that every exemption entry points at a real file (the
  audit also found two stale pre-v5.x ``propagators/asymptotic.py``
  exemption paths).

* **P3-40** -- glass.py's four module caches (``_glass_value_cache``,
  ``_glass_cache``, ``_validity_warned``, ``_kappa_warned``) were
  unbounded and unenrolled, surviving ``clear_asm_caches()`` /
  ``clear_all_registered_caches()`` forever.  Now: ``_glass_value_cache``
  is an LRU (``_GLASS_VALUE_CACHE_SIZE``), and a ``'glass_caches'``
  clearer drains all four -- PRESERVING user-fixed ``_FixedIndex``
  entries (for those names ``_glass_cache`` is the authoritative value
  store; dropping them would make the next lookup raise ValueError).

* **P2-41** -- ``load_material``'s catalog branch re-pointed
  ``GLASS_REGISTRY[name]`` without invalidating ``_glass_cache`` /
  ``_glass_value_cache``, silently serving the OLD refractive index
  forever.  Now it calls ``glass._invalidate_glass_name(name)``.

* **P2-25** -- the wrapper-merit meshgrid cache keyed six
  aperture-INDEPENDENT N x N arrays on the aperture VALUE, so a free
  ``aperture_diameter`` FD sweep duplicated ~56 B/px per perturbed
  value (7.6 GB retained at N=2048).  Now the grid arrays live in the
  aperture-free ``_WRAPPER_MERIT_GRID_CACHE`` and are shared by
  reference; per-aperture entries own only their 1 B/px boolean mask.
  Values are byte-identical (verified pre/post via SHA-256 A/B probe;
  merit repr 0.2182219461209884 identical on a 2-iteration
  design_optimize with aperture_diameter free).

Author: Wave-4 audit implementer -- v5.17.1
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
import lumenairy.glass as G
import lumenairy.optimize.wrapper_merits as wm
import lumenairy.user_library as ul

# The meta-pin module under test for P2-42 (sibling test module; tests/
# is a package so the canonical dotted import resolves under pytest).
from tests.unit import (
    test_v4_16_1_dispatcher_pin_cache_registry_enrollment as _pin,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sweep_nk5(n, start=550e-9, step=1e-12):
    """Populate ``_glass_value_cache`` with ``n`` distinct
    picometre-quantised N-K5 (``'__sellmeier__'`` sentinel) lookups."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(n):
            G.get_glass_index('N-K5', start + i * step)


# ============================================================================
# P2-42: walker discovery + exemption-path freshness
# ============================================================================

class TestP242WalkerCaseInsensitive:

    def test_walker_discovers_lowercase_glass_caches(self):
        """The broadened (case-insensitive) walker MUST discover both
        lower-case glass.py caches -- the exact blindness of P2-42."""
        tree = _pin._file_to_ast(Path(G.__file__))
        names = {n for n, _ in
                 _pin._discover_module_level_cache_assignments(tree)}
        assert {'_glass_cache', '_glass_value_cache'} <= names, (
            f"case-insensitive cache discovery regressed: glass.py "
            f"discovery returned {sorted(names)}")

    def test_glass_module_enrolls_with_registry(self):
        """glass.py now hosts a ``register_cache_clearer`` call, so the
        newly-discovered caches pass the main pin via enrollment (not
        via an exemption)."""
        tree = _pin._file_to_ast(Path(G.__file__))
        assert _pin._module_has_register_cache_clearer_call(tree)

    def test_main_meta_pin_passes_with_broadened_filter(self):
        """The full enrollment pin must hold over the broadened
        discovery set (every newly-visible cache is enrolled or
        exempted with rationale)."""
        _pin.test_every_cache_owning_module_enrolls_with_registry()

    def test_exemption_registry_paths_exist(self):
        """No exemption entry may reference a file that no longer
        exists (the audit found two stale pre-v5.x
        ``propagators/asymptotic.py`` paths -- dead entries hide the
        next real gap behind a false sense of coverage)."""
        stale = [rel for rel, _name in _pin._CACHE_REGISTRY_EXEMPTIONS
                 if not (_REPO_ROOT / rel).exists()]
        assert not stale, f"exemption entries cite missing files: {stale}"


# ============================================================================
# P3-40: glass caches enrolled, drained, bounded
# ============================================================================

class TestP340GlassCacheLifecycle:

    def test_glass_caches_clearer_registered(self):
        assert 'glass_caches' in la.list_registered_cache_clearers()

    def test_clear_asm_caches_drains_glass_state(self):
        """clear_asm_caches() must drain the value cache, both
        warn-once sets, and re-loadable catalogue ``_glass_cache``
        entries -- while PRESERVING user-fixed entries (authoritative
        value stores, not caches)."""
        cat_name = '__w4_catalog_test__'
        fix_name = '__w4_fixed_test__'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # register FIRST -- register_fixed_glass clears the
                # whole value cache as part of its own hygiene.
                ul.register_fixed_glass(fix_name, 1.5)
                # out-of-validity lookup -> _validity_warned entry
                G.get_glass_index('N-BK7', 200e-9)
            _sweep_nk5(50)
            # synthetic catalogue-dispatch cache entry (re-loadable)
            G.GLASS_REGISTRY[cat_name] = ('specs', 'X', 'Y')
            with G._GLASS_CACHE_LOCK:
                G._glass_cache[cat_name] = object()
            assert len(G._glass_value_cache) >= 50
            assert len(G._validity_warned) >= 1

            la.clear_asm_caches()

            assert len(G._glass_value_cache) == 0
            assert len(G._validity_warned) == 0
            assert len(G._kappa_warned) == 0
            assert cat_name not in G._glass_cache, (
                'catalogue-dispatch cache entry must be dropped '
                '(re-loadable from GLASS_REGISTRY)')
            assert fix_name in G._glass_cache, (
                'user-fixed entry must be PRESERVED (authoritative '
                'value store -- dropping it breaks get_glass_index)')
            # the '__thin_lens__' user-fixed entry registered by
            # raytrace.trace at import must also survive and resolve.
            assert G.get_glass_index(fix_name, 550e-9) == 1.5
            assert G.get_glass_index('__thin_lens__', 550e-9) == 1.5
        finally:
            G.GLASS_REGISTRY.pop(cat_name, None)
            G.GLASS_REGISTRY.pop(fix_name, None)
            with G._GLASS_CACHE_LOCK:
                G._glass_cache.pop(cat_name, None)
                G._glass_cache.pop(fix_name, None)

    def test_value_cache_lru_bounded(self, monkeypatch):
        """Sweeping 3x the bound leaves exactly ``bound`` entries."""
        bound = 64
        monkeypatch.setattr(G, '_GLASS_VALUE_CACHE_SIZE', bound)
        with G._GLASS_CACHE_LOCK:
            G._glass_value_cache.clear()
        _sweep_nk5(3 * bound)
        assert len(G._glass_value_cache) == bound

    def test_post_eviction_recompute_byte_identical(self, monkeypatch):
        """An evicted value recomputes to the EXACT same float."""
        bound = 32
        monkeypatch.setattr(G, '_GLASS_VALUE_CACHE_SIZE', bound)
        with G._GLASS_CACHE_LOCK:
            G._glass_value_cache.clear()
        wl0 = 550e-9
        v_first = G.get_glass_index('N-K5', wl0)   # miss -> compute
        v_hit = G.get_glass_index('N-K5', wl0)     # hit
        assert v_hit == v_first
        _sweep_nk5(2 * bound, start=600e-9)        # evict wl0
        key0 = ('N-K5', round(wl0 * 1e12))
        assert key0 not in G._glass_value_cache, 'wl0 should be evicted'
        v_recompute = G.get_glass_index('N-K5', wl0)
        assert v_recompute == v_first, (
            'post-eviction recompute must be byte-identical')


# ============================================================================
# P2-41: load_material catalog branch invalidates stale caches
# ============================================================================

@pytest.fixture
def _tmp_library(tmp_path):
    """Redirect the user library to a temp dir; restore afterwards."""
    old = ul._library_path
    ul.set_library_path(str(tmp_path))
    yield tmp_path
    ul._library_path = old


class TestP241LoadMaterialInvalidation:

    def test_catalog_reload_invalidates_cached_resolution(self, _tmp_library):
        """Mechanism pin (refractiveindex-free): after load_material
        re-points the registry, no stale ``_glass_cache`` object nor
        ``_glass_value_cache`` entries may remain for the name."""
        name = '__w4_stale_test__'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ul.register_fixed_glass(name, 1.5)
            assert name in G._glass_cache
            # plant a value-cache entry for the name as well
            with G._GLASS_CACHE_LOCK:
                G._glass_value_cache[(name, 12345)] = 99.0
            ul.save_material(name, shelf='specs', book='SCHOTT-optical',
                             page='N-BK7')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ul.load_material(name)
            assert G.GLASS_REGISTRY[name] == (
                'specs', 'SCHOTT-optical', 'N-BK7')
            assert name not in G._glass_cache, (
                'P2-41 regression: stale _glass_cache object survives '
                'the catalog re-point and would serve the OLD index')
            assert not any(k[0] == name for k in G._glass_value_cache), (
                'P2-41 regression: stale value-cache entries survive')
        finally:
            G.GLASS_REGISTRY.pop(name, None)
            G._invalidate_glass_name(name)

    def test_catalog_reload_serves_fresh_index(self, _tmp_library):
        """End-to-end audit scenario: fixed 1.5 -> N-BK7 catalog page
        must serve ~1.5168 at the d-line, not the stale 1.5."""
        pytest.importorskip('refractiveindex')
        name = '__w4_stale_live__'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ul.register_fixed_glass(name, 1.5)
                assert G.get_glass_index(name, 587.6e-9) == 1.5
                ul.save_material(name, shelf='specs',
                                 book='SCHOTT-optical', page='N-BK7')
                ul.load_material(name)
                n = G.get_glass_index(name, 587.6e-9)
            assert abs(n - 1.5168) < 5e-4, (
                f'stale index served after catalog reload: {n}')
        finally:
            G.GLASS_REGISTRY.pop(name, None)
            G._invalidate_glass_name(name)


# ============================================================================
# P2-25: meshgrid cache split (aperture-free grid arrays)
# ============================================================================

class TestP225MeshgridSplit:

    def setup_method(self, method):
        wm._clear_wrapper_merit_cache()

    def teardown_method(self, method):
        wm._clear_wrapper_merit_cache()

    def test_grid_arrays_shared_across_apertures(self):
        """Aperture-only-different entries must REFERENCE one shared
        set of the six aperture-independent arrays (was: six full
        duplicates per FD-perturbed aperture value)."""
        e1 = wm._get_wrapper_merit_cache(64, 1e-5, 5e-3, np.complex128)
        e2 = wm._get_wrapper_merit_cache(64, 1e-5, 5.0001e-3, np.complex128)
        for k in ('X', 'Y', 'Y_factor', 'X_factor', 'r_squared', 'E_ones'):
            assert e1[k] is e2[k], (
                f'{k} must be shared by reference across apertures')
        assert e1 is not e2
        assert len(wm._WRAPPER_MERIT_GRID_CACHE) == 1

    def test_retained_bytes_bounded_under_aperture_sweep(self):
        """A 40-value FD aperture sweep retains ONE grid-array set plus
        32 boolean masks -- not 32 x 57 B/px full payloads."""
        N = 64
        for i in range(40):
            wm._get_wrapper_merit_cache(N, 1e-5, 5e-3 + i * 1e-7,
                                        np.complex128)
        assert len(wm._WRAPPER_MERIT_CACHE) == wm._WRAPPER_MERIT_CACHE_SIZE
        seen, total = set(), 0
        for entry in wm._WRAPPER_MERIT_CACHE.values():
            for v in entry.values():
                if isinstance(v, np.ndarray) and id(v) not in seen:
                    seen.add(id(v))
                    total += v.nbytes
        # 5 float64 grids + 1 complex128 = 56 B/px shared once, plus
        # one 1 B/px bool mask per retained entry.
        expected = N * N * (5 * 8 + 16) + 32 * (N * N * 1)
        assert total == expected, (
            f'unique retained bytes {total} != expected {expected} '
            f'(grid set shared once + 32 masks)')

    def test_grid_cache_lru_bounded(self):
        """Sweeping 3x the grid bound leaves exactly ``bound`` grid
        entries (per-aperture cache keeps its own 32-bound)."""
        bound = wm._WRAPPER_MERIT_GRID_CACHE_SIZE
        for i in range(3 * bound):
            wm._get_wrapper_merit_cache(16 + i, 1e-5, 5e-3, np.complex128)
        assert len(wm._WRAPPER_MERIT_GRID_CACHE) == bound

    def test_eviction_does_not_mutate_held_payload(self):
        """A caller-held entry must be byte-stable across eviction of
        both its per-aperture slot and its grid-cache slot, and the
        post-eviction recompute must be byte-identical."""
        N, dx, ap = 32, 7e-6, 173.4e-6
        held = wm._get_wrapper_merit_cache(N, dx, ap, np.complex128)
        snap = {k: held[k].copy() for k in
                ('X', 'Y', 'mask', 'Y_factor', 'X_factor',
                 'r_squared', 'E_ones')}
        # cycle far past both LRU bounds
        for i in range(wm._WRAPPER_MERIT_CACHE_SIZE + 4):
            wm._get_wrapper_merit_cache(40 + i, dx, 5e-3, np.complex128)
        for k, v in snap.items():
            assert np.array_equal(held[k], v), (
                f'eviction mutated caller-held array {k!r}')
        rebuilt = wm._get_wrapper_merit_cache(N, dx, ap, np.complex128)
        assert rebuilt is not held, 'entry should have been evicted'
        for k, v in snap.items():
            assert np.array_equal(rebuilt[k], v), (
                f'post-eviction recompute of {k!r} not byte-identical')

    def test_payload_byte_identical_to_reference_build(self):
        """Cached payload == fresh np.indices reference build, exactly
        (the pre-split construction, per the A/B probe)."""
        N, dx, ap = 48, 7e-6, 173.4e-6
        c = wm._get_wrapper_merit_cache(N, dx, ap, np.complex128)
        Y_idx, X_idx = np.indices((N, N))
        X_ref = (X_idx - N / 2) * dx
        Y_ref = (Y_idx - N / 2) * dx
        r2_ref = X_ref * X_ref + Y_ref * Y_ref
        assert np.array_equal(c['X'], X_ref)
        assert np.array_equal(c['Y'], Y_ref)
        assert np.array_equal(c['r_squared'], r2_ref)
        assert np.array_equal(c['mask'], r2_ref <= (ap / 2.0) ** 2)
        assert np.array_equal(c['Y_factor'], 2.0 * np.pi * Y_ref)
        assert np.array_equal(c['X_factor'], 2.0 * np.pi * X_ref)
        assert c['E_ones'].dtype == np.complex128
        assert np.all(c['E_ones'] == 1.0)

    def test_registered_clearer_drains_both_caches(self):
        """The 'wrapper_merit_meshgrid' registry clearer (via
        clear_asm_caches) must drain the grid sibling too."""
        wm._get_wrapper_merit_cache(64, 1e-5, 5e-3, np.complex128)
        assert len(wm._WRAPPER_MERIT_CACHE) >= 1
        assert len(wm._WRAPPER_MERIT_GRID_CACHE) >= 1
        la.clear_asm_caches()
        assert len(wm._WRAPPER_MERIT_CACHE) == 0
        assert len(wm._WRAPPER_MERIT_GRID_CACHE) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
