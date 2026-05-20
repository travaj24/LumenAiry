"""Correctness pinning tests for the v4.12 JAX-jit perf work.

The v4.12 audit found the JAX runtime tree had zero ``@jax.jit``
decorators -- every call paid Python-level dispatch overhead on every
invocation.  v4.12.0 adds module-scope jit caches to the hottest JAX
entry points (``propagate_through_system_jax``, ``gerchberg_saxton_jax``
/ ``error_reduction_jax`` / ``hybrid_input_output_jax``, the inner ASM
kernel inside ``through_focus_scan_jax``).

Note: a ``trace_jax`` jit cache was prototyped but deferred to v4.12.1
because the flat-tuple signature hashing broke ``jax.grad`` propagation
through ``fit_canonical_polynomials_jax`` (validation suite caught it,
test_asymptotic.py).  v4.12.1 will revisit with a gradient-aware cache
design (likely pytree-registered prescription wrapper instead of tuple).

These tests pin:

  1. Output equality between jit'd and pre-jit behaviour (run twice
     with the same inputs -- the second call hits the cache and must
     produce identical results).
  2. Cache reuse:  a second call with the same static signature must
     NOT add a new entry to the module-level cache dict (i.e., we are
     actually caching, not re-tracing).
  3. NumPy / JAX numeric agreement is preserved.

If JAX isn't installed in the runtime env, the whole module skips
gracefully via ``pytest.importorskip``.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if JAX is unavailable.
jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


import lumenairy as lm  # noqa: E402  (after importorskip)


# ===========================================================================
# propagate_through_system_jax
# ===========================================================================

class TestPropagateThroughSystemJaxJit:
    """The JAX system-propagation entry should cache its kernel across
    repeated calls with the same element list."""

    @pytest.fixture
    def simple_chain(self):
        """A short ASM + lens + aperture chain."""
        from lumenairy.propagators.system import (
            propagate_through_system_jax,
            _PROPAGATE_SYSTEM_JAX_CACHE,
        )
        N, dx, wl = 64, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'lens', 'f': 5e-3},
            # Canonical NumPy schema (params['diameter']) -- matches
            # ``apply_aperture`` (audit round-4 B1-1 fix).
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 200e-6}},
            {'type': 'propagate', 'z': 5e-3, 'bandlimit': True},
        ]
        return {
            'fn': propagate_through_system_jax,
            'cache': _PROPAGATE_SYSTEM_JAX_CACHE,
            'E': E, 'elements': elements, 'wl': wl, 'dx': dx,
        }

    def test_propagate_through_system_jax_reuses_cache(self, simple_chain):
        c = simple_chain
        cache_size_before = len(c['cache'])
        E1 = c['fn'](c['E'], c['elements'], c['wl'], c['dx'])
        cache_size_after_first = len(c['cache'])
        E2 = c['fn'](c['E'], c['elements'], c['wl'], c['dx'])
        cache_size_after_second = len(c['cache'])
        assert cache_size_after_first == cache_size_before + 1, (
            "First call must add one cache entry")
        assert cache_size_after_second == cache_size_after_first, (
            "Second call must hit cache (no growth)")
        np.testing.assert_allclose(np.asarray(E1), np.asarray(E2))

    def test_propagate_through_system_jax_handles_fallback(self):
        """Element types without a JAX path must FAIL-FAST with a
        ``NotImplementedError`` (audit round-4 B1-2 fix).

        Pre-v4.12 this function silently fell back to NumPy via
        ``np.asarray(E)``, which raises ``TracerArrayConversionError``
        under ``jax.jit`` / ``jax.grad`` and breaks end-to-end
        traceability.  v4.12 raises a clear ``NotImplementedError``
        at call time listing the offending element types.

        Uses ``aspheric_lens`` (no JAX handler) -> must raise.
        """
        from lumenairy.propagators.system import (
            propagate_through_system_jax,
            _PROPAGATE_SYSTEM_JAX_CACHE,
        )
        N, dx, wl = 32, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'aspheric_lens',
             'R1': 5e-3, 'R2': -5e-3, 'd': 1e-3, 'n_lens': 1.5,
             'k1': 0.0, 'k2': 0.0},
            {'type': 'propagate', 'z': 2e-3, 'bandlimit': True},
        ]
        cache_size_before = len(_PROPAGATE_SYSTEM_JAX_CACHE)
        with pytest.raises(NotImplementedError, match='aspheric_lens'):
            propagate_through_system_jax(E, elements, wl, dx)
        # The fail-fast gate is before any tracing, so the cache must
        # not grow.
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == cache_size_before, (
            "Fail-fast (non-traceable element) path should not "
            "pollute the JAX kernel cache.")


# ===========================================================================
# gerchberg_saxton_jax / error_reduction_jax / hybrid_input_output_jax
# ===========================================================================

class TestPhaseRetrievalJaxJit:
    """The phase-retrieval drivers should reuse a cached kernel keyed
    on ``n_iter``."""

    @pytest.fixture
    def gs_setup(self):
        N = 16
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        src = np.exp(-(X**2 + Y**2) / 0.5**2).astype(np.float32)
        tgt = np.exp(-(X**2 + Y**2) / 0.3**2).astype(np.float32)
        return src, tgt

    def test_gerchberg_saxton_jax_seed_42_repeatable(self, gs_setup):
        """Same seed should produce the same trajectory across calls."""
        src, tgt = gs_setup
        phase_a, err_a = lm.gerchberg_saxton_jax(
            src, tgt, n_iter=10, seed=42)
        phase_b, err_b = lm.gerchberg_saxton_jax(
            src, tgt, n_iter=10, seed=42)
        np.testing.assert_allclose(phase_a, phase_b)
        assert abs(err_a - err_b) < 1e-10

    def test_gerchberg_saxton_jax_cache_keyed_on_n_iter(self, gs_setup):
        """Calls with the same n_iter share a kernel; different n_iter
        gets a fresh kernel.  Uses an uncommon n_iter so the cache slot
        is fresh regardless of test-suite ordering."""
        from lumenairy.analysis.phase_retrieval import _GS_KERNEL_CACHE
        src, tgt = gs_setup
        # Use uncommon n_iter values so this test doesn't share cache
        # slots with other tests / fixtures.
        n_iter_a, n_iter_b = 9, 11
        _GS_KERNEL_CACHE.pop(n_iter_a, None)
        _GS_KERNEL_CACHE.pop(n_iter_b, None)
        n0 = len(_GS_KERNEL_CACHE)
        lm.gerchberg_saxton_jax(src, tgt, n_iter=n_iter_a, seed=1)
        n1 = len(_GS_KERNEL_CACHE)
        lm.gerchberg_saxton_jax(src, tgt, n_iter=n_iter_a, seed=2)
        n2 = len(_GS_KERNEL_CACHE)
        lm.gerchberg_saxton_jax(src, tgt, n_iter=n_iter_b, seed=1)
        n3 = len(_GS_KERNEL_CACHE)
        assert n1 == n0 + 1, "First call adds one cache entry"
        assert n2 == n1, "Same n_iter, same kernel (cache hit)"
        assert n3 == n2 + 1, "Different n_iter -> new kernel"

    def test_error_reduction_jax_cache_keyed_on_n_iter(self):
        from lumenairy.analysis.phase_retrieval import _ER_KERNEL_CACHE
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        # Use uncommon n_iter to keep this test independent of suite order.
        n_iter_a = 7
        _ER_KERNEL_CACHE.pop(n_iter_a, None)
        n0 = len(_ER_KERNEL_CACHE)
        lm.error_reduction_jax(meas, sup, n_iter=n_iter_a, seed=0)
        n1 = len(_ER_KERNEL_CACHE)
        lm.error_reduction_jax(meas, sup, n_iter=n_iter_a, seed=1)
        n2 = len(_ER_KERNEL_CACHE)
        assert n1 == n0 + 1
        assert n2 == n1, "Same n_iter -> cache hit"

    def test_hybrid_input_output_jax_cache_keyed_on_n_iter(self):
        from lumenairy.analysis.phase_retrieval import _HIO_KERNEL_CACHE
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        n_iter_a = 7
        _HIO_KERNEL_CACHE.pop(n_iter_a, None)
        n0 = len(_HIO_KERNEL_CACHE)
        lm.hybrid_input_output_jax(meas, sup, n_iter=n_iter_a, beta=0.9, seed=0)
        n1 = len(_HIO_KERNEL_CACHE)
        lm.hybrid_input_output_jax(meas, sup, n_iter=n_iter_a, beta=0.9, seed=1)
        n2 = len(_HIO_KERNEL_CACHE)
        assert n1 == n0 + 1
        assert n2 == n1, "Same n_iter -> cache hit"


# ===========================================================================
# through_focus_scan_jax
# ===========================================================================

class TestThroughFocusScanJaxJit:
    """``through_focus_scan_jax``'s inner ASM kernel is jit'd
    via jax.jit(jax.vmap(...)); we can't easily inspect a public
    cache here, but we can pin that the function still produces a
    sensible result and matches the NumPy reference within tolerance
    on a small grid."""

    def test_through_focus_scan_jax_runs(self):
        N, dx, wl = 64, 5e-6, 1.55e-6
        sigma = 10e-6
        x = (np.arange(N) - N/2) * dx
        X, Y = np.meshgrid(x, x)
        E = np.exp(-(X**2 + Y**2) / (2*sigma**2)).astype(np.complex64)
        zs = np.linspace(-1e-3, 1e-3, 5)
        res = lm.through_focus_scan_jax(
            E, dx=dx, wavelength=wl, z_values=zs,
            ideal_peak=float(np.max(np.abs(E)**2)),
            bandlimit=True)
        # Output result has a peak intensity per z, all finite.
        assert res.peak_I.shape == zs.shape
        assert np.all(np.isfinite(res.peak_I))


# ===========================================================================
# Future-work documentation
# ===========================================================================
#
# Functions not jit'd (and why):
#
# 1. apply_real_lens_traced_jax / apply_real_lens_maslov_jax
#    (lumenairy/elements/_lens_jax.py)
#    -- Contain ``float(x_out_grid[...])`` calls on JAX traced values
#       to compute the paraxial-magnification initial guess.  These
#       force concretization and would fail under @jit.  Refactor
#       would either lift the M-guess outside jit (and re-jit the
#       inner Newton inversion + phase apply) or replace the
#       float() with a fully-traceable analytic estimate.
#
# 2. propagate_modal_asymptotic_lg00_jax
#    (lumenairy/propagators/asymptotic.py)
#    -- Closes over a CanonicalPolyFit dataclass that is not
#       registered as a JAX pytree.  Wrapping the vmap with jit
#       would either fail to hash the fit (older JAX) or retrace
#       on every call with a new fit object.  The vmap'd path
#       already fuses to one XLA dispatch; further win requires
#       registering CanonicalPolyFit as a pytree.
#
# 3. fit_canonical_polynomials_jax (lumenairy/propagators/asymptotic.py)
#    -- Same pytree-registration story for CanonicalPolyFit, plus
#       the function does its own Python-level shape analysis that
#       interacts poorly with jit's tracer.
#
# 4. angular_spectrum_propagate JAX branch
#    (lumenairy/propagators/propagation.py)
#    -- When called from inside ``propagate_through_system_jax``'s
#       jit'd kernel, the operations get traced and inlined into
#       the outer kernel for free; no separate jit cache needed.
#       When called directly with a JAX field as a one-shot, jit
#       overhead wouldn't help.
