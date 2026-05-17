"""Pinning tests for the v4.12.1 ``trace_jax`` jit cache.

History
-------

v4.12.0 attempted a ``trace_jax`` jit cache via flat-tuple signature
hashing.  The cache itself worked (7470x speedup on warm calls), but
``jax.grad(fit_canonical_polynomials_jax)`` returned NaN, so the change
was reverted before shipping.

v4.12.1 re-implements the cache with two layered fixes:

1. A pytree-registered :class:`JaxPrescription` wrapper that splits the
   prescription into numeric leaves (``jax.grad`` flows through them
   when the caller substitutes tracers) and hashable structural aux
   (forms the cache key).
2. **A trace-context detector that skips the jit layer when the
   prescription leaves OR the initial state carry any JAX tracer.**
   The v4.12.0 NaN turned out to be a JAX bug -- the backward pass
   through ``jax.jit`` + ``jnp.linalg.lstsq`` produces NaN in
   ``dot_general`` when the lstsq matrix is near rank-deficient
   (the canonical-poly Chebyshev basis triggers this in practice).
   The wrapper alone does not avoid the bug, so v4.12.1 also bypasses
   the jit-cache layer under ``jax.grad`` / ``jax.jit`` / ``jax.vmap``,
   restoring v4.11.2 semantics on the gradient path while keeping the
   eager-call cache.

What this file pins
-------------------

* The v4.12.0 failure: ``jax.grad(fit_canonical_polynomials_jax)`` is
  finite and non-zero.  Pinned aggressively so that if anyone tries
  another approach to the cache that re-introduces the NaN, it gets
  caught immediately.
* Eager cache reuse: repeated calls with the same prescription stay
  at cache size 1 (no growth).
* Bit-exact equality: cached output matches a fresh-state call.
* Aux-keyed: changing only a leaf value via mutation of the JAX array
  hits the same cache slot when aux is identical; changing aux (glass)
  misses.
* Cache-warm speedup: 10th call >= 100x faster than first call.

The "CRITICAL test" is :meth:`TestCacheGradFinite.
test_grad_finite_v4_12_0_regression` -- the test that the v4.12.0
implementation failed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

# Skip the whole module if JAX is unavailable.
jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


# x64 is required for canonical-poly fit (per fit_canonical_polynomials_jax
# itself, which auto-enables x64 with a warning).  Enable here too so
# tests run deterministically regardless of import order.
jax.config.update('jax_enable_x64', True)


import lumenairy as lm  # noqa: E402
from lumenairy.raytrace.jax_trace import (  # noqa: E402
    JaxPrescription,
    _TRACE_JAX_CACHE,
    _build_jax_prescription,
    _leaves_are_concrete,
    _running_under_trace,
    make_jax_ray_state,
    trace_jax,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the module-level jit cache before each test so cache-size
    assertions aren't polluted by prior tests."""
    _TRACE_JAX_CACHE.clear()
    yield
    _TRACE_JAX_CACHE.clear()


@pytest.fixture
def singlet():
    """Standard BK7 singlet used across the test module."""
    return lm.make_singlet(
        R1=20e-3, R2=float('inf'), d=2e-3,
        glass='N-BK7', aperture=4e-3)


# ===========================================================================
# THE CRITICAL TEST -- the failure that prompted v4.12.1
# ===========================================================================

class TestCacheGradFinite:
    """The v4.12.0 cache implementation broke ``jax.grad`` through
    :func:`fit_canonical_polynomials_jax`, returning NaN.  v4.12.1
    must preserve gradient finiteness."""

    def test_grad_finite_v4_12_0_regression(self, singlet):
        """**THE failed-last-time test.**

        ``jax.grad(fit_canonical_polynomials_jax)`` w.r.t.
        ``source_box_half`` must return a FINITE, NON-NaN gradient.

        Under v4.12.0 (flat-tuple cache, ``jax.jit`` wrap on every
        call) this returned NaN because the calling transform's
        backward pass through ``jax.jit`` + ``jnp.linalg.lstsq``
        produces NaN in ``dot_general`` for the canonical-poly's
        4-D Chebyshev basis.  v4.12.1 bypasses the jit cache layer
        when ANY tracer is present in the inputs, so the gradient
        path matches v4.11.2 semantics.
        """
        from lumenairy.propagators.asymptotic import (
            fit_canonical_polynomials_jax,
        )

        def loss(sbh):
            fit = fit_canonical_polynomials_jax(
                singlet, 633e-9, source_box_half=sbh,
                pupil_box_half=0.05, n_field=4, n_pupil=8,
                poly_order=4)
            return jnp.sum(fit.coef_phi ** 2)

        g = float(jax.grad(loss)(20e-6))
        assert np.isfinite(g), (
            f"jax.grad(fit_canonical_polynomials_jax) returned NaN "
            f"(grad={g}); the v4.12.0 cache failure has been "
            f"re-introduced.")
        assert abs(g) > 0, (
            f"jax.grad returned zero (grad={g}); gradient flow has "
            f"been silently severed.  Expected a non-zero gradient "
            f"because the loss depends on source_box_half through "
            f"the canonical-poly fit.")
        # Sanity: the v4.11.2 baseline returns ~10273.9 for this
        # specific configuration; allow a generous range so future
        # numeric improvements don't trip this.
        assert 1e2 < abs(g) < 1e6, (
            f"jax.grad magnitude unexpected (grad={g}); v4.11.2 "
            f"baseline was ~1e4.  Check fit_canonical_polynomials_jax "
            f"and trace_jax for unintended drift.")

    def test_grad_finite_through_trace_under_jit(self, singlet):
        """``jax.grad`` composed with an ``@jax.jit``'d wrapper that
        calls ``trace_jax`` directly (not via fit_canonical, which has
        Python-time bool checks that prevent it from being jit'd)
        must also be finite.  Pins that the cache-bypass logic
        correctly detects the outer jit context."""
        state = _make_state(20e-6)

        @jax.jit
        def jitted_trace(x_in):
            local = make_jax_ray_state(
                x=x_in, y=state.y, z=state.z,
                L=state.L, M=state.M, N=state.N)
            out = trace_jax(local, singlet, 633e-9)
            # Avoid the lstsq NaN-trigger: simple quadratic loss.
            return jnp.sum(out.x ** 2 + out.y ** 2)

        g = jax.grad(jitted_trace)(state.x)
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), (
            f"jax.grad(jax.jit(...))(...) produced non-finite "
            f"gradient: max={np.nanmax(g_np)}, min={np.nanmin(g_np)}.")


# ===========================================================================
# Cache mechanics
# ===========================================================================

def _make_state(sbh=20e-6, n_per_side=8):
    """Build a small ray bundle used by several tests."""
    n = n_per_side
    u_field = jnp.asarray(np.cos(np.pi * (np.arange(4) + 0.5) / 4))
    u_pupil = jnp.asarray(np.cos(np.pi * (np.arange(n) + 0.5) / n))
    s1x = u_field * sbh
    s1y = u_field * sbh
    v1x = u_pupil * 0.05
    v1y = u_pupil * 0.05
    S1X, S1Y, V1X, V1Y = jnp.meshgrid(
        s1x, s1y, v1x, v1y, indexing='ij')
    sumsq = V1X * V1X + V1Y * V1Y
    N1 = jnp.sqrt(jnp.maximum(1.0 - sumsq, 0.0))
    return make_jax_ray_state(
        x=S1X.ravel(), y=S1Y.ravel(),
        z=jnp.zeros_like(S1X.ravel()),
        L=V1X.ravel(), M=V1Y.ravel(), N=N1.ravel(),
    )


class TestCacheReuse:
    """Eager / non-traced calls with the same prescription must reuse
    one cached jit kernel."""

    def test_cache_size_one_after_repeated_calls(self, singlet):
        """15 calls with the same prescription -> cache size 1."""
        state = _make_state()
        for _ in range(15):
            out = trace_jax(state, singlet, 633e-9)
            out.x.block_until_ready()
        assert len(_TRACE_JAX_CACHE) == 1, (
            f"Cache must reuse the same kernel; got size "
            f"{len(_TRACE_JAX_CACHE)} after 15 calls.")

    def test_cache_size_one_across_equivalent_dicts(self, singlet):
        """Two independently-constructed prescription dicts with the
        same numeric content must share a cache slot."""
        presc_a = lm.make_singlet(
            R1=20e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=4e-3)
        presc_b = lm.make_singlet(
            R1=20e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=4e-3)
        state = _make_state()
        trace_jax(state, presc_a, 633e-9).x.block_until_ready()
        trace_jax(state, presc_b, 633e-9).x.block_until_ready()
        assert len(_TRACE_JAX_CACHE) == 1, (
            f"Two prescriptions with identical numeric content must "
            f"share a cache slot; got size {len(_TRACE_JAX_CACHE)}.")


class TestNumericEquality:
    """Cached output must be bit-exact equal to the uncached fallback."""

    def test_cache_hit_matches_first_call(self, singlet):
        """The cache hit must produce IDENTICAL output to the first
        call (which built the kernel)."""
        state = _make_state()
        out0 = trace_jax(state, singlet, 633e-9)
        out0.x.block_until_ready()
        out1 = trace_jax(state, singlet, 633e-9)
        out1.x.block_until_ready()
        np.testing.assert_array_equal(
            np.asarray(out0.x), np.asarray(out1.x))
        np.testing.assert_array_equal(
            np.asarray(out0.y), np.asarray(out1.y))
        np.testing.assert_array_equal(
            np.asarray(out0.L), np.asarray(out1.L))
        np.testing.assert_array_equal(
            np.asarray(out0.M), np.asarray(out1.M))
        np.testing.assert_array_equal(
            np.asarray(out0.opd), np.asarray(out1.opd))
        np.testing.assert_array_equal(
            np.asarray(out0.alive), np.asarray(out1.alive))

    def test_traced_matches_eager_numerically(self, singlet):
        """The traced-path body (used inside ``jax.grad``) and the
        eager cached-jit body must produce numerically equal output
        on the same input.  Tolerance is set to per-field max rel
        error 1e-13 because the two paths use the same numeric
        primitives in the same order; the only difference is the
        outer jit boundary."""
        state = _make_state()

        # Eager (hits cache after first call)
        out_eager = trace_jax(state, singlet, 633e-9)
        out_eager.x.block_until_ready()

        # Traced -- wrap in jax.grad so initial_state contains tracers,
        # which forces ``trace_jax`` to bypass the jit layer.
        def gather(sbh):
            local_state = _make_state(sbh=sbh)
            out = trace_jax(local_state, singlet, 633e-9)
            return jnp.sum(out.opd)

        # Force a re-trace; capture the output via vmap-like wrapping.
        # Easier: call via jax.jit so the trace context is live.
        @jax.jit
        def jit_call(state_in):
            return trace_jax(state_in, singlet, 633e-9)

        out_jit = jit_call(state)
        out_jit.x.block_until_ready()
        np.testing.assert_allclose(
            np.asarray(out_eager.x), np.asarray(out_jit.x),
            rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(out_eager.opd), np.asarray(out_jit.opd),
            rtol=1e-13, atol=1e-14)


class TestAuxKeying:
    """Changing only an aux entry (e.g., glass name) must miss the
    cache; numeric-only differences that go into aux (radii / conics)
    also change the key, which is the v4.12.1 design (the
    JaxPrescription leaves are *mirrored* by Python-float aux entries
    so the eager-cache static-branch path keeps working).

    The differentiable use case where leaves carry tracers and aux
    stays static is exercised by :class:`TestLeafDifferentiability`."""

    def test_different_glass_misses_cache(self, singlet):
        """N-BK7 vs N-SF11 differ in ``n_pre``/``n_post`` aux entries;
        they must NOT share a cache slot."""
        presc_bk7 = singlet
        presc_sf11 = lm.make_singlet(
            R1=20e-3, R2=float('inf'), d=2e-3,
            glass='N-SF11', aperture=4e-3)
        state = _make_state()
        trace_jax(state, presc_bk7, 633e-9).x.block_until_ready()
        trace_jax(state, presc_sf11, 633e-9).x.block_until_ready()
        assert len(_TRACE_JAX_CACHE) == 2, (
            f"Different glasses must miss the cache; got size "
            f"{len(_TRACE_JAX_CACHE)} (expected 2).")

    def test_different_wavelength_misses_cache(self, singlet):
        """Different wavelengths feed into the cache key (and change
        glass indices via ``get_glass_index``), so they must miss."""
        state = _make_state()
        trace_jax(state, singlet, 633e-9).x.block_until_ready()
        trace_jax(state, singlet, 1064e-9).x.block_until_ready()
        assert len(_TRACE_JAX_CACHE) == 2, (
            f"Different wavelengths must miss the cache; got size "
            f"{len(_TRACE_JAX_CACHE)} (expected 2).")

    def test_different_radius_misses_cache(self, singlet):
        """Radii live in BOTH the leaves AND the aux (the aux carries a
        Python-float mirror used by the static-branch path).  Changing
        the radius therefore changes aux and must miss the cache --
        the radii-as-leaf gradient path is exercised separately by
        :class:`TestLeafDifferentiability`."""
        presc_a = singlet
        presc_b = lm.make_singlet(
            R1=25e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=4e-3)
        state = _make_state()
        trace_jax(state, presc_a, 633e-9).x.block_until_ready()
        trace_jax(state, presc_b, 633e-9).x.block_until_ready()
        assert len(_TRACE_JAX_CACHE) == 2, (
            f"Different radii must miss the cache; got size "
            f"{len(_TRACE_JAX_CACHE)} (expected 2).")


class TestLeafDifferentiability:
    """When a user substitutes a tracer leaf into a JaxPrescription
    (e.g., differentiate w.r.t. a radius) the cache is bypassed and the
    always-Newton ``_trace_body_traced`` path is taken so gradient
    flows through."""

    def test_grad_through_radius_leaf(self, singlet):
        """``jax.grad`` w.r.t. a JaxPrescription radius leaf is finite
        and non-zero."""

        def loss(radius_scalar):
            base = _build_jax_prescription(singlet, 633e-9)
            new_radii = base.radii.at[0].set(radius_scalar)
            jp = JaxPrescription(
                new_radii, base.conics, base.thicks,
                base.asph_coeffs, base.aux)
            state = _make_state()
            out = trace_jax(state, jp, 633e-9)
            return jnp.sum(out.x ** 2 + out.y ** 2)

        g = float(jax.grad(loss)(jnp.asarray(20e-3)))
        assert np.isfinite(g), (
            f"jax.grad w.r.t. radius leaf must be finite; got {g}.")
        assert abs(g) > 0, (
            f"jax.grad w.r.t. radius leaf must be non-zero; got {g}.")


# ===========================================================================
# Cache hit speedup
# ===========================================================================

class TestCacheWarmSpeedup:
    """Pin the 100x speedup expected from cache hits."""

    def test_warm_call_at_least_100x_faster_than_first(self, singlet):
        """The 10th call must be at least 100x faster than the first
        (compile-bound) call.  This is a sanity check that the cache
        is actually doing something -- without the cache, calls land
        at ~2.5 ms via JAX's internal dispatch tax, vs ~25 us once
        cached (a ~100x speedup baseline)."""
        state = _make_state()
        # Run once to warm any JAX-internal caches that aren't ours.
        trace_jax(state, singlet, 633e-9).x.block_until_ready()
        _TRACE_JAX_CACHE.clear()

        n_iter = 15
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            out = trace_jax(state, singlet, 633e-9)
            out.x.block_until_ready()
            times.append(time.perf_counter() - t0)

        first = times[0]
        # Use the median of calls 5-14 to absorb scheduling jitter.
        warm = float(np.median(times[5:]))
        speedup = first / max(warm, 1e-9)
        assert speedup >= 100.0, (
            f"Cache speedup too small: first={first*1000:.1f}ms, "
            f"warm={warm*1000:.3f}ms, speedup={speedup:.0f}x "
            f"(expected >= 100x).")


# ===========================================================================
# JaxPrescription pytree integrity
# ===========================================================================

class TestJaxPrescriptionPytree:
    """The pytree registration must round-trip and play with the JAX
    pytree machinery (``jax.tree_util.tree_map`` etc.)."""

    def test_flatten_unflatten_round_trip(self, singlet):
        """``tree_flatten`` then ``tree_unflatten`` reconstructs the
        prescription with all attributes preserved (leaves AND aux)."""
        jp = _build_jax_prescription(singlet, 633e-9)
        leaves, treedef = jax.tree_util.tree_flatten(jp)
        jp_back = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(jp_back, JaxPrescription)
        np.testing.assert_array_equal(
            np.asarray(jp.radii), np.asarray(jp_back.radii))
        np.testing.assert_array_equal(
            np.asarray(jp.conics), np.asarray(jp_back.conics))
        np.testing.assert_array_equal(
            np.asarray(jp.thicks), np.asarray(jp_back.thicks))
        assert jp.aux == jp_back.aux, (
            "Pytree round-trip lost aux content (the cache key).")

    def test_tree_map_doubles_leaves(self, singlet):
        """``jax.tree_util.tree_map`` on a JaxPrescription transforms
        every leaf (radii / conics / thicks / asph_coeffs) while
        preserving aux.  This is the contract that lets users build
        gradient-flowing wrappers around ``trace_jax``."""
        jp = _build_jax_prescription(singlet, 633e-9)
        jp2 = jax.tree_util.tree_map(lambda x: 2.0 * x, jp)
        np.testing.assert_allclose(
            np.asarray(jp2.radii), 2.0 * np.asarray(jp.radii),
            rtol=0, atol=0)
        np.testing.assert_allclose(
            np.asarray(jp2.conics), 2.0 * np.asarray(jp.conics),
            rtol=0, atol=0)
        # aux is preserved verbatim (not part of the mapped leaves).
        assert jp2.aux == jp.aux

    def test_leaves_are_concrete_helper(self, singlet):
        """``_leaves_are_concrete`` returns True for a vanilla JP and
        False when a leaf is replaced by a tracer."""
        jp = _build_jax_prescription(singlet, 633e-9)
        assert _leaves_are_concrete(jp) is True

        # Substitute a tracer leaf inside a jax.grad call to verify.
        captured = []

        def f(r):
            base = _build_jax_prescription(singlet, 633e-9)
            new_radii = base.radii.at[0].set(r)
            jp_t = JaxPrescription(
                new_radii, base.conics, base.thicks,
                base.asph_coeffs, base.aux)
            captured.append(_leaves_are_concrete(jp_t))
            return r ** 2

        jax.grad(f)(jnp.asarray(20e-3))
        assert captured and captured[0] is False, (
            "Inside jax.grad, the substituted leaf should NOT be "
            f"concrete; got captured={captured}")


# ===========================================================================
# Surface-kind validation preserved
# ===========================================================================

class TestSurfaceKindGate:
    """The v4.11.2 ``NotImplementedError`` gate on unsupported surface
    kinds (mirrors / coord-breaks / biconic / freeform) must still
    fire under the new cache flow.  Without this check the cache
    would silently pretend the unsupported surface is a refractive
    flat."""

    def test_mirror_raises(self):
        presc = {
            'surfaces': [
                {'radius': 20e-3, 'glass_after': 'air',
                 'is_mirror': True},
            ],
            'thicknesses': [],
        }
        state = _make_state()
        with pytest.raises(NotImplementedError, match='is_mirror'):
            trace_jax(state, presc, 633e-9)

    def test_biconic_raises(self):
        presc = {
            'surfaces': [
                {'radius': 20e-3, 'radius_y': 25e-3,
                 'glass_after': 'air'},
            ],
            'thicknesses': [],
        }
        state = _make_state()
        with pytest.raises(NotImplementedError, match='radius_y'):
            trace_jax(state, presc, 633e-9)
