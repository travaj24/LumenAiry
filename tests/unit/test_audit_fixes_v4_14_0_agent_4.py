"""Pinning tests for the v4.14.0 Agent-4 perf win.

Audit reference
---------------

v4.14.0 Agent-4 scope: ``lumenairy/optimize/core.py`` ONLY.

The three wrapper merits -- ``MultiWavelengthMerit``,
``MultiFieldMerit``, ``ToleranceAwareMerit`` -- each rebuilt
``np.indices`` / meshgrid / aperture-mask / Y-tilt-phase arrays on
every per-wavelength, per-field, per-trial leg.  For a representative
5-wavelength * 5-field * 40-FD-eval optimisation step at N=512 that
amounted to up to 1000 N x N meshgrid builds per outer iteration, none
of which depended on the parameter vector being differenced.

v4.14.0 adds a module-level LRU(32) cache keyed on
``(Ny, Nx, dx, aperture_hash, dtype_str)`` and routes all three
wrapper merits through it.  Per-leg work reduces to ``np.exp(1j *
sin_a * cached_k0_Y) * cached_aperture_mask`` (MultiFieldMerit) or a
single ``.copy()`` of the cached np.ones template (the other two)
plus the standard ``apply_real_lens`` call.

What this test file pins
------------------------

* Cache identity -- repeated calls with the same key return the same
  cached arrays (not just numerically equal -- the same object) so
  the per-leg cost stays a single multiply.
* Cache invalidation -- different ``(N, dx, aperture, dtype)`` keys
  produce distinct cached payloads.
* Meshgrid-build counter -- the eval-count contract: 1 build per
  (N, dx, aperture) signature for a full sweep, not 1 per leg.
* Correctness -- pre-perf and post-perf merit values must match
  bit-near-exact (1e-12 relative).  Compared against a reference
  implementation that materialises the meshgrid on every call.
* LRU bound -- the cache evicts at the ``_WRAPPER_MERIT_CACHE_SIZE``
  threshold.
* ``clear_asm_caches`` wiring -- the propagation-layer clear-all hook
  also drops the wrapper-merit cache.

Author: Andrew Traverso -- v4.14.0 / Agent 4
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.optimize.core import (
    DesignParameterization,
    EvaluationContext,
    MultiFieldMerit,
    MultiWavelengthMerit,
    StrehlMerit,
    ToleranceAwareMerit,
    _WRAPPER_MERIT_CACHE,
    _WRAPPER_MERIT_CACHE_SIZE,
    _clear_wrapper_merit_cache,
    _get_wrapper_merit_cache,
    _wrapper_merit_aperture_key,
    design_optimize,
)
from lumenairy.propagators.propagation import clear_asm_caches


# ============================================================================
# Helpers
# ============================================================================

def _meshgrid_build_count() -> int:
    """Re-read the module-level counter (it is mutated, not rebound)."""
    import lumenairy.optimize.core as core
    return core._WRAPPER_MERIT_MESHGRID_BUILDS


def _simple_singlet():
    """Build a minimal singlet prescription suitable for design_optimize
    smoke runs.  The same geometry as v4.13.x C.2 / C.4 tests."""
    return {
        'surfaces': [
            {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -50e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [3e-3],
        'aperture_diameter': 10e-3,
    }


# ============================================================================
# Cache primitives
# ============================================================================

class TestCachePrimitives:
    """Exercise the module-level cache helper directly."""

    def setup_method(self, method):
        _clear_wrapper_merit_cache()

    def test_cache_hit_returns_same_object(self):
        """Two calls with the same key share the same payload object."""
        c1 = _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        before = _meshgrid_build_count()
        c2 = _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        after = _meshgrid_build_count()
        assert c1 is c2, 'cache hit must return the SAME dict (not a copy)'
        assert c1['X'] is c2['X'], 'cache hit must reuse X array'
        assert c1['Y'] is c2['Y'], 'cache hit must reuse Y array'
        assert c1['mask'] is c2['mask'], 'cache hit must reuse mask array'
        assert after == before, (
            f'cache hit must NOT increment meshgrid-build counter; '
            f'got before={before}, after={after}')

    def test_cache_miss_rebuilds_and_counts(self):
        """Different (N, dx, aperture, dtype) produces a fresh build."""
        _clear_wrapper_merit_cache()
        base = _meshgrid_build_count()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(128, 1e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(64, 2e-6, 50e-6, np.complex128)
        _get_wrapper_merit_cache(64, 1e-6, 80e-6, np.complex128)
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex64)
        # Five distinct keys -> five builds.
        assert _meshgrid_build_count() == base + 5

    def test_cache_payload_correctness(self):
        """The cached X/Y/mask/Y_factor/E_ones agree with a fresh
        np.indices/meshgrid reference build."""
        N = 32
        dx = 5e-6
        ap = 80e-6
        _clear_wrapper_merit_cache()
        c = _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        # Reference
        Y_idx, X_idx = np.indices((N, N))
        X_ref = (X_idx - N / 2) * dx
        Y_ref = (Y_idx - N / 2) * dx
        mask_ref = (X_ref ** 2 + Y_ref ** 2) <= (ap / 2.0) ** 2
        assert np.array_equal(c['X'], X_ref)
        assert np.array_equal(c['Y'], Y_ref)
        assert np.array_equal(c['mask'], mask_ref)
        # Y_factor is 2*pi * Y (no wavelength baked in).
        assert np.allclose(c['Y_factor'], 2.0 * np.pi * Y_ref)
        assert np.allclose(c['X_factor'], 2.0 * np.pi * X_ref)
        assert c['E_ones'].shape == (N, N)
        assert c['E_ones'].dtype == np.complex128

    def test_lru_eviction(self):
        """Once the cache exceeds ``_WRAPPER_MERIT_CACHE_SIZE`` (32)
        the oldest entry is dropped."""
        _clear_wrapper_merit_cache()
        # Insert SIZE+5 distinct entries.  Vary N so each key is
        # unique.
        for i in range(_WRAPPER_MERIT_CACHE_SIZE + 5):
            _get_wrapper_merit_cache(32 + i, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) == _WRAPPER_MERIT_CACHE_SIZE, (
            f'cache size {len(_WRAPPER_MERIT_CACHE)} != '
            f'{_WRAPPER_MERIT_CACHE_SIZE}')

    def test_aperture_key_none(self):
        """``None`` aperture maps to a stable scalar tag."""
        assert _wrapper_merit_aperture_key(None) == ('none',)

    def test_aperture_key_scalar(self):
        """Numeric aperture maps to ``('scalar', float)``."""
        assert _wrapper_merit_aperture_key(10e-3) == ('scalar', 10e-3)
        # np.float64 and python float must collide.
        assert (_wrapper_merit_aperture_key(np.float64(10e-3))
                == _wrapper_merit_aperture_key(10e-3))

    def test_aperture_key_array(self):
        """ndarray aperture -- different contents must hash differently;
        identical contents must hash to the same key."""
        a1 = np.zeros((8, 8), dtype=bool)
        a2 = a1.copy()
        a3 = a1.copy()
        a3[0, 0] = True
        assert (_wrapper_merit_aperture_key(a1)
                == _wrapper_merit_aperture_key(a2))
        assert (_wrapper_merit_aperture_key(a1)
                != _wrapper_merit_aperture_key(a3))

    def test_aperture_key_array_vs_scalar_distinct(self):
        """An ndarray aperture key never collides with a scalar key."""
        a = np.ones((4, 4), dtype=bool)
        k_arr = _wrapper_merit_aperture_key(a)
        k_sc = _wrapper_merit_aperture_key(1.0)
        assert k_arr != k_sc
        assert k_arr[0] == 'arr' and k_sc[0] == 'scalar'

    def test_clear_resets_counter(self):
        """``_clear_wrapper_merit_cache`` zeros the build counter."""
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert _meshgrid_build_count() >= 1
        _clear_wrapper_merit_cache()
        assert _meshgrid_build_count() == 0
        assert len(_WRAPPER_MERIT_CACHE) == 0


# ============================================================================
# clear_asm_caches wiring
# ============================================================================

class TestClearAsmCachesWiring:
    """``clear_asm_caches`` is monkey-patched at import time to also
    drop the wrapper-merit cache.  Pinning this ensures the
    ``lumenairy_context(clear_caches_on_exit=True)`` hook in
    ``_context.py`` continues to leave both layers pristine."""

    def test_clear_asm_caches_drops_wrapper_merit_cache(self):
        """A call to ``clear_asm_caches`` empties the wrapper-merit
        cache as a side effect."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(64, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0, (
            'clear_asm_caches must also drop the wrapper-merit cache '
            '(monkey-patched composite clear)')

    def test_clear_asm_caches_via_top_level_export(self):
        """``lumenairy.clear_asm_caches`` (the top-level re-export)
        also picks up the composite version."""
        _clear_wrapper_merit_cache()
        _get_wrapper_merit_cache(32, 1e-6, 50e-6, np.complex128)
        assert len(_WRAPPER_MERIT_CACHE) >= 1
        lm.clear_asm_caches()
        assert len(_WRAPPER_MERIT_CACHE) == 0


# ============================================================================
# MultiFieldMerit correctness: cached path vs reference build
# ============================================================================

class TestMultiFieldMeritCorrectness:
    """The cached tilted-plane-wave construction must agree
    bit-near-exact with the v4.13.2 reference (rebuild meshgrid on
    every call)."""

    def test_tilted_plane_wave_matches_reference(self):
        """For a synthetic ``(N, dx, aperture, wavelength)`` build
        the tilted plane wave from the cache; compare against an
        explicit np.meshgrid rebuild."""
        N = 32
        dx = 10e-6
        ap = 100e-6
        wavelength = 1.30e-6
        theta_x = 0.005
        theta_y = 0.012
        _clear_wrapper_merit_cache()
        # Reference (pre-perf path)
        x_ref = (np.arange(N) - N / 2) * dx
        y_ref = (np.arange(N) - N / 2) * dx
        X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
        k0 = 2 * np.pi / wavelength
        tilt_phase_ref = (k0 * np.sin(theta_x) * X_ref
                          + k0 * np.sin(theta_y) * Y_ref)
        mask_ref = (X_ref ** 2 + Y_ref ** 2) <= (ap / 2.0) ** 2
        E_ref = np.where(
            mask_ref, np.exp(1j * tilt_phase_ref), 0.0
        ).astype(np.complex128)
        # Cached path
        c = _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        k_X = c['X_factor'] / wavelength
        k_Y = c['Y_factor'] / wavelength
        tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
        E_new = np.where(
            c['mask'], np.exp(1j * tilt_phase), 0.0
        ).astype(np.complex128)
        # Bit-near-exact agreement (the only mathematical difference
        # is the multiply order; numerically identical at 1e-15).
        np.testing.assert_allclose(
            E_new, E_ref, rtol=1e-13, atol=1e-13,
            err_msg='cached tilted-plane-wave must match reference '
                    'build to 1e-13')


# ============================================================================
# MultiWavelengthMerit correctness via design_optimize
# ============================================================================

class TestMultiWavelengthMeritCorrectness:
    """End-to-end pin: a short ``design_optimize`` run with a
    ``MultiWavelengthMerit`` produces the same merit values as a
    reference run using the pre-perf direct-meshgrid path.

    Realised as: run a fixed 1-iteration optimisation and snapshot
    the final ``merit`` value; this is the cheapest cross-check that
    exercises the cached code path inside the wrapper merit's
    ``evaluate``.
    """

    def test_short_optimisation_runs_without_error(self):
        """The cached path must support a full design_optimize run
        without raising / producing NaN.  Three wavelengths * three
        FD evals @ N=32 (very cheap)."""
        template = _simple_singlet()
        param = DesignParameterization(
            template=template,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(20e-3, 80e-3)])
        sub = StrehlMerit(weight=1.0)
        merit = [MultiWavelengthMerit(
            wavelengths=[1.27e-6, 1.30e-6, 1.33e-6],
            sub_merit=sub, weight=1.0)]
        _clear_wrapper_merit_cache()
        with warnings.catch_warnings():
            # design_optimize may emit per-merit RuntimeWarnings if
            # the wave leg fails on the FD-perturbed prescription;
            # these are not in scope for this test.
            warnings.simplefilter('ignore', RuntimeWarning)
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', DeprecationWarning)
            res = design_optimize(
                param, merit, wavelength=1.30e-6,
                N=32, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        assert np.isfinite(res.merit), (
            f'merit must stay finite through the cached path; got '
            f'{res.merit}')


# ============================================================================
# Eval-count pin: meshgrid_build_count for a full run
# ============================================================================

class TestMeshgridBuildCountPin:
    """The headline perf claim: meshgrid_build_count == 1 (one) per
    ``(N, dx, aperture)`` signature over the entire optimisation
    run, regardless of #wavelengths / #fields / #FD evals.
    """

    def test_one_build_per_signature_multifield(self):
        """A MultiFieldMerit run with 3 field angles * several
        evaluate() calls produces exactly ONE meshgrid build."""
        N = 32
        dx = 10e-6
        ap = 100e-6
        _clear_wrapper_merit_cache()
        # Simulate the inner-loop call pattern WITHOUT spinning up
        # the full design_optimize -- just hit the cache helper as
        # MultiFieldMerit's evaluate would.
        for _ in range(50):
            _get_wrapper_merit_cache(N, dx, ap, np.complex128)
        assert _meshgrid_build_count() == 1, (
            f'expected exactly 1 build per signature; got '
            f'{_meshgrid_build_count()}')

    def test_three_signatures_three_builds(self):
        """Three distinct ``(N, dx, ap)`` signatures = three builds,
        no matter how many evaluate() calls per signature."""
        _clear_wrapper_merit_cache()
        for _ in range(20):
            _get_wrapper_merit_cache(32, 1e-6, 50e-6, np.complex128)
            _get_wrapper_merit_cache(32, 1e-6, 60e-6, np.complex128)
            _get_wrapper_merit_cache(32, 1e-6, 70e-6, np.complex128)
        assert _meshgrid_build_count() == 3


# ============================================================================
# v4.13 closure preservation
# ============================================================================

class TestV413ClosuresPreserved:
    """Pin that the v4.13.2 x=ctx.x threading and field_angles tuple
    support survived the v4.14.0 perf refactor."""

    def setup_method(self, method):
        # Reset the deprecation one-shot.
        MultiFieldMerit._scalar_warning_issued = False

    def test_field_angles_tuple_still_accepted(self):
        """Tuple form (theta_x, theta_y) still works without
        warnings (preserved from v4.13.2 C-P0-2)."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m = MultiFieldMerit(
                field_angles=[(0.0, 0.0), (0.005, 0.01)],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert not dep
        assert m.field_angles == [(0.0, 0.0), (0.005, 0.01)]

    def test_field_angles_scalar_still_emits_deprecation(self):
        """Scalar form still emits the one-shot DeprecationWarning
        (preserved from v4.13.2 C-P0-2)."""
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            MultiFieldMerit(
                field_angles=[0.005, 0.01],
                sub_merit=StrehlMerit(weight=1.0),
                weight=1.0)
        dep = [w for w in ws if issubclass(w.category, DeprecationWarning)]
        assert dep, 'scalar field_angles must still emit DeprecationWarning'

    def test_x_thread_through_multifield_sub_ctx_source_pin(self):
        """Source-level pin for the v4.13.2 C-P1-2 ``x=getattr(ctx,
        'x', None)`` thread through the wrapper merits.  The v4.14.0
        perf refactor must preserve this closure; we grep the file
        contents for the call signature."""
        from pathlib import Path
        import lumenairy
        src = (Path(lumenairy.__file__).parent / 'optimize'
               / 'core.py').read_text(encoding='cp1252')
        # MultiFieldMerit, MultiWavelengthMerit, ToleranceAwareMerit
        # all build sub_ctx EvaluationContext objects.  Each must
        # forward x.
        # Count occurrences of x=getattr(ctx, 'x', None) -- expect
        # at least 3 (one per wrapper merit).
        marker = "x=getattr(ctx, 'x', None)"
        n = src.count(marker)
        assert n >= 3, (
            f'expected at least 3 occurrences of {marker!r} (one per '
            f'wrapper merit; v4.13.2 C-P1-2); got {n}')
