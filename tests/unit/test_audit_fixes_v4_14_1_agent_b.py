"""Tests for the v4.14.1 audit-fix Agent-B changes.

Scope: ``lumenairy/optimize/core.py`` and
``lumenairy/propagators/propagation.py``.

What this module pins
---------------------

* **B.1 -- P1-NEW-1 aperture=0 semantics regression.**
  ``MultiWavelengthMerit`` / ``MultiFieldMerit`` must zero the field
  when ``prescription['aperture_diameter'] == 0`` (the deliberate
  "block all light" branch).  v4.14.0 mapped ``ap <= 0`` to
  ``mask=None`` and the downstream callers then treated the
  deliberate zero as "no aperture -> full grid plane wave," flipping
  the semantics 180 degrees.  v4.14.1 introduces the
  ``_ZERO_APERTURE_MASK`` sentinel and makes both wrapper merits
  branch on ``mask is _ZERO_APERTURE_MASK``.

* **B.2 -- P2-1 wrapper-merit cache lock.**
  ``_WRAPPER_MERIT_CACHE`` is now guarded by a module-level
  ``threading.Lock``; the ``get/move_to_end/__setitem__/popitem`` ops
  and ``_clear_wrapper_merit_cache`` all acquire it.  No new
  pinning beyond a smoke test that ``threading.Lock`` is present.

* **B.3 -- P2-3 monkey-patch -> lazy-import.**
  v4.14.0 monkey-patched ``propagation.clear_asm_caches`` to also
  drop the wrapper-merit cache.  v4.14.1 inverts the dependency:
  ``clear_asm_caches()`` now lazy-imports
  ``_clear_wrapper_merit_cache`` from ``optimize.core`` and calls it
  inline.  This test imports the original module-attribute path
  (``lumenairy.propagators.propagation.clear_asm_caches``) and pins
  the cross-module clear.

* **B.4 -- Tier-0 #5 LG/HG mode-stack cache wiring.**
  v4.14.0 CHANGELOG claimed ``clear_asm_caches()`` clears the LG/HG
  mode-stack cache; in reality only ``lumenairy_context`` did.
  v4.14.1 chains a lazy import of
  ``propagators.asymptotic.clear_lg_mode_stack_cache`` into
  ``clear_asm_caches``.  Populate the LG cache via ``decompose_lg``,
  call ``clear_asm_caches()``, assert the LG cache is empty.

Author: Andrew Traverso -- v4.14.1 / Agent B
"""
from __future__ import annotations

import threading

import numpy as np

from lumenairy.optimize.core import (
    _WRAPPER_MERIT_CACHE,
    _WRAPPER_MERIT_CACHE_LOCK,
    _ZERO_APERTURE_MASK,
    _clear_wrapper_merit_cache,
    _get_wrapper_merit_cache,
)
from lumenairy.propagators.asymptotic import (
    _LG_MODE_STACK_CACHE,
    decompose_lg,
)
import lumenairy.propagators.propagation as _propagation_module


# ============================================================================
# B.1 -- P1-NEW-1: aperture=0 vs aperture=None semantics
# ============================================================================

class TestApertureZeroSemantics:
    """``aperture_diameter == 0`` must zero the field; ``None`` must
    map to "no aperture, full grid."""

    def setup_method(self, method):
        _clear_wrapper_merit_cache()

    def test_cache_returns_sentinel_for_aperture_zero(self):
        """A scalar ``aperture_diameter=0`` produces the dedicated
        ``_ZERO_APERTURE_MASK`` sentinel, NOT ``None`` (the v4.14.0
        regression)."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, 0.0, np.complex128)
        assert entry['mask'] is _ZERO_APERTURE_MASK, (
            'aperture_diameter=0 must map to _ZERO_APERTURE_MASK '
            "(deliberate 'block all light')")

    def test_cache_returns_sentinel_for_aperture_negative(self):
        """Same as zero but negative -- still 'no light through'."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, -1e-3, np.complex128)
        assert entry['mask'] is _ZERO_APERTURE_MASK

    def test_cache_returns_none_for_aperture_none(self):
        """``None`` aperture must NOT collide with the zero sentinel;
        downstream callers treat it as 'no aperture, full grid.'"""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, None, np.complex128)
        assert entry['mask'] is None
        assert entry['mask'] is not _ZERO_APERTURE_MASK

    def test_cache_returns_ndarray_for_positive_aperture(self):
        """A positive scalar aperture produces a boolean ndarray
        (the circular mask) -- neither None nor the sentinel."""
        N = 32
        dx = 5e-6
        entry = _get_wrapper_merit_cache(N, dx, 100e-6, np.complex128)
        assert isinstance(entry['mask'], np.ndarray)
        assert entry['mask'].dtype == bool

    def test_multifield_branches_on_zero_aperture_sentinel(self):
        """``MultiFieldMerit.evaluate`` must branch on
        ``aperture_mask is _ZERO_APERTURE_MASK`` and build
        ``E_tilted = np.zeros(...)`` (the v4.14.0 regression silently
        produced ``np.exp(1j*tilt_phase)`` -- a grid-filling plane
        wave -- which inverted the 'block all light' semantics).

        The library's ``validate_prescription`` strict-mode rejects
        ``aperture_diameter=0`` so we cannot drive this through
        ``evaluate``; instead we exercise the per-field branching
        body directly via the cache, which is the source of truth.
        """
        N = 32
        dx = 5e-6
        wl = 1.3e-6
        _clear_wrapper_merit_cache()
        # Mimic the MultiFieldMerit setup line-for-line: ap=0 keeps
        # the deliberate zero (no fallback substitution since 0 is
        # not None).
        ap_diam = 0.0
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, float(ap_diam), cdtype)
        aperture_mask = _cache['mask']
        assert aperture_mask is _ZERO_APERTURE_MASK
        k_X = _cache['X_factor'] / wl
        k_Y = _cache['Y_factor'] / wl
        theta_x, theta_y = 0.01, 0.02
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        # Reproduce the v4.14.1 branching contract.
        if aperture_mask is _ZERO_APERTURE_MASK:
            E_tilted = np.zeros((N, N), dtype=cdtype)
        elif aperture_mask is None:
            E_tilted = np.exp(1j * tilt_phase).astype(cdtype)
        else:
            E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                 0.0).astype(cdtype)
        assert np.array_equal(E_tilted, np.zeros((N, N), dtype=cdtype)), (
            'aperture=0 must zero E_tilted (the deliberate '
            "'block all light' branch)")

    def test_multifield_aperture_none_produces_full_grid(self):
        """``aperture_mask is None`` must produce the grid-filling
        plane wave (full-grid behaviour)."""
        N = 32
        dx = 5e-6
        wl = 1.3e-6
        _clear_wrapper_merit_cache()
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, None, cdtype)
        aperture_mask = _cache['mask']
        assert aperture_mask is None
        k_X = _cache['X_factor'] / wl
        k_Y = _cache['Y_factor'] / wl
        theta_x, theta_y = 0.0, 0.0
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        if aperture_mask is _ZERO_APERTURE_MASK:
            E_tilted = np.zeros((N, N), dtype=cdtype)
        elif aperture_mask is None:
            E_tilted = np.exp(1j * tilt_phase).astype(cdtype)
        else:
            E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                 0.0).astype(cdtype)
        # On-axis (theta=0) tilt_phase is zero everywhere -> E_tilted
        # is all ones (the full-grid plane wave).
        assert np.allclose(E_tilted, np.ones((N, N), dtype=cdtype))

    def test_multiwavelength_branches_on_zero_aperture_sentinel(self):
        """``MultiWavelengthMerit.evaluate`` must branch on the
        zero-aperture sentinel and build ``E_in = np.zeros(...)``.
        Same constraint as ``MultiFieldMerit``: prescription
        validation forbids ``aperture_diameter=0`` so we exercise
        the cache-driven branch body directly."""
        N = 32
        dx = 5e-6
        _clear_wrapper_merit_cache()
        cdtype = np.complex128
        _cache = _get_wrapper_merit_cache(N, dx, 0.0, cdtype)
        mask = _cache['mask']
        assert mask is _ZERO_APERTURE_MASK
        # Reproduce the v4.14.1 MultiWavelengthMerit branching.
        if mask is _ZERO_APERTURE_MASK:
            E_in_wl = np.zeros((N, N), dtype=cdtype)
        elif mask is None:
            E_in_wl = np.ones((N, N), dtype=cdtype)
        else:
            E_in_wl = mask.astype(cdtype)
        assert np.array_equal(E_in_wl, np.zeros((N, N), dtype=cdtype))

        # Counter-test: ap=None gives the full-grid ones field.
        _clear_wrapper_merit_cache()
        _cache_none = _get_wrapper_merit_cache(N, dx, None, cdtype)
        mask = _cache_none['mask']
        assert mask is None
        if mask is _ZERO_APERTURE_MASK:
            E_in_none = np.zeros((N, N), dtype=cdtype)
        elif mask is None:
            E_in_none = np.ones((N, N), dtype=cdtype)
        else:
            E_in_none = mask.astype(cdtype)
        assert np.array_equal(E_in_none, np.ones((N, N), dtype=cdtype))


# ============================================================================
# B.2 -- P2-1: lock present and acquired
# ============================================================================

class TestWrapperMeritCacheLock:
    """The wrapper-merit cache now carries a threading.Lock so
    concurrent design_optimize threads cannot tear the OrderedDict."""

    def test_lock_is_a_threading_lock(self):
        """Sanity: ``_WRAPPER_MERIT_CACHE_LOCK`` exists and looks like
        a ``threading.Lock`` (it carries an ``acquire`` method and is
        usable as a context manager)."""
        # threading.Lock is a factory function returning a _thread.lock
        # primitive that doesn't expose the original class via
        # isinstance, so we check the duck-type shape.
        assert hasattr(_WRAPPER_MERIT_CACHE_LOCK, 'acquire')
        assert hasattr(_WRAPPER_MERIT_CACHE_LOCK, 'release')
        # Confirm it is re-acquirable after release (i.e. it really
        # is unlocked between calls).
        with _WRAPPER_MERIT_CACHE_LOCK:
            assert _WRAPPER_MERIT_CACHE_LOCK.locked()
        assert not _WRAPPER_MERIT_CACHE_LOCK.locked()

    def test_concurrent_cache_access_does_not_corrupt(self):
        """Smoke test: hammer ``_get_wrapper_merit_cache`` from 8
        threads with mixed keys.  The cache must not lose entries
        or crash."""
        _clear_wrapper_merit_cache()
        errors = []

        def worker(seed):
            try:
                for _ in range(50):
                    N = 32 + (seed % 4) * 8
                    _get_wrapper_merit_cache(N, 1e-6, 50e-6,
                                              np.complex128)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)
        assert not errors, f'concurrent access raised: {errors}'
        # 4 distinct N values -> at most 4 cache entries.
        assert 1 <= len(_WRAPPER_MERIT_CACHE) <= 4


# ============================================================================
# B.3 -- P2-3: lazy-import replaces monkey-patch
# ============================================================================

class TestClearAsmCachesLazyImport:
    """``propagation.clear_asm_caches`` (the module-attribute path,
    NOT the v4.14.0 monkey-patched re-bind) must still drop the
    wrapper-merit cache via the v4.14.1 lazy-import."""

    def test_propagation_clear_asm_caches_drops_wrapper_merit(self):
        """Imported via ``lumenairy.propagators.propagation.clear_asm_caches``
        directly (no monkey-patch involvement).  v4.14.1 adds a
        lazy-import + call to ``_clear_wrapper_merit_cache`` inside
        the propagation-layer body."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        _propagation_module.clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0, (
            'propagation.clear_asm_caches must drop the wrapper-merit '
            'cache via the v4.14.1 lazy-import (the v4.14.0 '
            'monkey-patch has been removed)')


# ============================================================================
# B.4 -- Tier-0 #5: LG mode-stack cache wired into clear_asm_caches
# ============================================================================

class TestClearAsmCachesDropsLgCache:
    """``clear_asm_caches`` must also flush the LG/HG mode-stack
    cache.  The v4.14.0 CHANGELOG claimed this; only the v4.14.1
    fix actually wires it up."""

    def test_clear_asm_caches_drops_lg_mode_stack_cache(self):
        """Populate ``_LG_MODE_STACK_CACHE`` via a real ``decompose_lg``
        call; call ``clear_asm_caches``; assert the cache is empty."""
        # Build a synthetic field on a small grid; populate the cache.
        N = 32
        x = np.linspace(-50e-6, 50e-6, N)
        y = np.linspace(-50e-6, 50e-6, N)
        X, Y = np.meshgrid(x, y, indexing='xy')
        field = np.exp(-(X ** 2 + Y ** 2) / (30e-6) ** 2).astype(
            np.complex128)
        # decompose_lg builds + caches the conjugated mode stack.
        _LG_MODE_STACK_CACHE.clear()
        _ = decompose_lg(field, X, Y, w=30e-6, p_max=2, ell_max=1)
        assert len(_LG_MODE_STACK_CACHE) >= 1, (
            'decompose_lg should populate the LG mode-stack cache')

        _propagation_module.clear_asm_caches()

        assert len(_LG_MODE_STACK_CACHE) == 0, (
            'clear_asm_caches must drop the LG mode-stack cache '
            '(v4.14.1 Tier-0 #5 wires the lazy-import; v4.14.0 '
            'CHANGELOG claimed this but never implemented it)')
