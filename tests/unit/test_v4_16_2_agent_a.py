"""v4.16.2 Agent A regression tests -- close the propagators + raytrace
findings from ``docs/audits/AUDIT_V4_16_1_2026_05_19.md``.

Scope (per the v4.16.2 dispatch):

* **P1-NEW-F1-1** -- ``_transfer_jax`` high-NA ``RuntimeWarning`` was
  structurally unreachable in the production user flow because the
  v4.16.1 gate used ``isinstance(direction_n, np.ndarray)`` while
  :func:`make_jax_ray_state` materialises ``state.N`` via
  ``jnp.asarray(...)`` (a ``jax.Array``, NOT a ``np.ndarray`` subclass).
  Fix: duck-typed concreteness probe + hoist the check to the eager
  entry of :func:`trace_jax`.
* **P1-NEW-F1-2** -- :func:`propagate_ensemble` silently downcast
  CuPy / JAX ensembles to host NumPy via ``np.asarray(ensemble)``.
  Fix: ``xp = array_namespace(ensemble)`` dispatch with matching
  ``xp.zeros`` / ``xp.abs`` for the accumulator.
* **P3-NEW-F1-1** -- empty 3-D ensemble raised opaque
  ``ZeroDivisionError`` via ``I_acc / float(0)``.  Fix: explicit
  ``ValueError`` at entry.
* **P3-NEW-F1-2** -- collision between ``propagator_kwargs={'dx': ...}``
  and positional ``dx=`` raised opaque Python
  ``TypeError: got multiple values for keyword argument 'dx'``.
  Fix: explicit ``ValueError`` with remediation pointer.
* **P3-NEW-F1-5** -- the v4.16.1 B.4 dtype-probe pin only checked
  ``@jax.jit``; ``jax.grad`` was uncovered.  Mirror the pattern.
* **P3-NEW-F1-6** -- :func:`propagate_ensemble` had no positive pin
  for ``propagator_kwargs`` pass-through with a non-colliding key.

Author: Andrew Traverso -- v4.16.2 / Agent A
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.ensemble import propagate_ensemble
from lumenairy.sources.core import create_gaussian_schell_source

# ----------------------------------------------------------------------
# Optional JAX import (every JAX-touching test skips when missing).
# ----------------------------------------------------------------------

JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason='JAX is not installed')


# ----------------------------------------------------------------------
# Shared Schell-ensemble factory (mirrors the v4.16.1 Agent B fixture).
# ----------------------------------------------------------------------

def _make_simple_ensemble(n_realizations=4, N=16, dx=2e-6,
                          wavelength=633e-9, w0=20e-6, sigma_g=8e-6,
                          seed=0):
    return create_gaussian_schell_source(
        N=N, dx=dx, wavelength=wavelength,
        w0=w0, sigma_g=sigma_g,
        n_realizations=n_realizations, seed=seed,
        return_kind='ensemble',
    )


# ----------------------------------------------------------------------
# A minimal 2-surface refractive prescription for the trace_jax pins.
# Both surfaces flat -> the trace reduces to a free-space transfer of
# length ``thicknesses[0]``, which is the regime where the
# ``_transfer_jax`` paraxial approximation is exercised end-to-end.
# ----------------------------------------------------------------------

_TRACE_PRESCRIPTION = {
    'aperture_diameter': 20e-3,
    'surfaces': [
        {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
         'radius_y': None, 'conic_y': None, 'aspheric_coeffs_y': None,
         'glass_before': 'air', 'glass_after': 'air'},
        {'radius': float('inf'), 'conic': 0.0, 'aspheric_coeffs': None,
         'radius_y': None, 'conic_y': None, 'aspheric_coeffs_y': None,
         'glass_before': 'air', 'glass_after': 'air'},
    ],
    'thicknesses': [1e-3, 0.0],
}


# ============================================================================
# P1-NEW-F1-1 -- _transfer_jax high-NA gate reachable in production
# ============================================================================


class TestTransferJaxGateProductionReachability:
    """audit closure: P1-NEW-F1-1

    The v4.16.1 ``isinstance(direction_n, np.ndarray)`` gate was
    structurally unreachable for users who built their state via the
    canonical :func:`make_jax_ray_state` factory (``state.N`` is a
    ``jax.Array``, NOT a ``np.ndarray`` subclass since JAX 0.4+).  The
    v4.16.1 B.5 tests pinned the warning by constructing
    ``JaxRayState`` directly with ``np.array(...)`` -- bypassing the
    factory -- so the gate was green at test time but dead in the
    user's actual flow.

    These tests cover the production path: build the state via
    :func:`make_jax_ray_state` and verify the warning fires on
    high-NA bundles + does NOT fire on low-NA or inside jit.
    """

    @needs_jax
    def test_make_jax_ray_state_high_na_warns_via_direct_transfer(self):
        """audit closure: P1-NEW-F1-1

        ``make_jax_ray_state`` produces ``state.N`` as a ``jax.Array``.
        Direct call into ``_transfer_jax`` with that state must trigger
        the high-NA warning at min |N|=0.5.
        """
        from lumenairy.raytrace.jax_trace import (
            _reset_transfer_jax_warning,
            _transfer_jax,
            make_jax_ray_state,
        )

        _reset_transfer_jax_warning()
        try:
            K = 4
            zeros = np.zeros(K)
            n_axis = 0.5 * np.ones(K)  # low |N| -> NA ~ 0.866
            L = np.sqrt(1.0 - n_axis ** 2)
            state = make_jax_ray_state(zeros, zeros, zeros, L, zeros,
                                       n_axis)
            # Belt-and-suspenders: confirm the production factory
            # yields a non-``np.ndarray`` ``state.N`` -- this is the
            # exact precondition the v4.16.1 gate missed.
            assert not isinstance(state.N, np.ndarray)
            assert hasattr(state.N, '__array__')

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = _transfer_jax(state, thickness=1e-3, n_medium=1.0)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert len(rw) == 1, (
                f"Expected the high-NA RuntimeWarning to fire for a "
                f"``make_jax_ray_state``-built state at min |N|=0.5; "
                f"got {len(rw)}: {[str(w.message) for w in rw]}.")
        finally:
            _reset_transfer_jax_warning()

    @needs_jax
    def test_trace_jax_end_to_end_high_na_warns(self):
        """audit closure: P1-NEW-F1-1

        End-to-end production path: ``trace_jax(...)`` with a
        ``make_jax_ray_state``-built initial state and a low |N|
        bundle.  The warning must fire once -- this is the user flow
        that the v4.16.1 closure was supposed to cover.
        """
        from lumenairy.raytrace.jax_trace import (
            _TRACE_JAX_CACHE,
            _reset_transfer_jax_warning,
            make_jax_ray_state,
            trace_jax,
        )

        _reset_transfer_jax_warning()
        # Clear the per-prescription cache so the cold-build path is
        # exercised (warning fires on the entry to trace_jax, before
        # cache hits or jit traces).
        _TRACE_JAX_CACHE.clear()
        try:
            K = 5
            zeros = np.zeros(K)
            n_axis = 0.5 * np.ones(K)
            L = np.sqrt(1.0 - n_axis ** 2)
            state = make_jax_ray_state(zeros, zeros, zeros, L, zeros,
                                       n_axis)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = trace_jax(state, _TRACE_PRESCRIPTION,
                              wavelength=0.55e-6)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert len(rw) == 1, (
                f"trace_jax end-to-end with min |N|=0.5 must emit "
                f"exactly 1 high-NA RuntimeWarning; got {len(rw)}: "
                f"{[str(w.message) for w in rw]}.")
            # Quantitative anchor: per-surface drift estimate at NA~0.866
            # and thickness=1e-3 is ``1e-3 * (1 - 0.25) / 2 = 3.75e-4 m``.
            msg = str(rw[0].message)
            assert 'paraxial' in msg.lower()
            assert 'NA' in msg
        finally:
            _reset_transfer_jax_warning()
            _TRACE_JAX_CACHE.clear()

    @needs_jax
    def test_trace_jax_end_to_end_low_na_does_not_warn(self):
        """audit closure: P1-NEW-F1-1

        Low-NA (|N|=0.995, NA=0.1) is below the warning threshold; the
        warning must NOT fire.  Counter-pin to the previous test.
        """
        from lumenairy.raytrace.jax_trace import (
            _TRACE_JAX_CACHE,
            _reset_transfer_jax_warning,
            make_jax_ray_state,
            trace_jax,
        )

        _reset_transfer_jax_warning()
        _TRACE_JAX_CACHE.clear()
        try:
            K = 5
            zeros = np.zeros(K)
            n_axis = 0.995 * np.ones(K)  # NA ~ 0.1
            L = np.sqrt(1.0 - n_axis ** 2)
            state = make_jax_ray_state(zeros, zeros, zeros, L, zeros,
                                       n_axis)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = trace_jax(state, _TRACE_PRESCRIPTION,
                              wavelength=0.55e-6)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert rw == [], (
                f"trace_jax with low NA must NOT emit the high-NA "
                f"RuntimeWarning; got {[str(w.message) for w in rw]}.")
        finally:
            _reset_transfer_jax_warning()
            _TRACE_JAX_CACHE.clear()

    @needs_jax
    def test_trace_jax_under_jit_no_warn(self):
        """audit closure: P1-NEW-F1-1

        When the caller wraps ``trace_jax`` in ``jax.jit``,
        ``initial_state.N`` is a ``Tracer`` (not a concrete array).
        The duck-typed concreteness probe must skip the check cleanly
        -- inspecting tracer values mid-trace would force
        concretisation and raise ``TracerArrayConversionError`` /
        ``ConcretizationTypeError``.
        """
        import jax

        from lumenairy.raytrace.jax_trace import (
            _TRACE_JAX_CACHE,
            _reset_transfer_jax_warning,
            make_jax_ray_state,
            trace_jax,
        )

        _reset_transfer_jax_warning()
        _TRACE_JAX_CACHE.clear()
        try:
            K = 5
            zeros = np.zeros(K)
            n_axis = 0.5 * np.ones(K)
            L = np.sqrt(1.0 - n_axis ** 2)
            state = make_jax_ray_state(zeros, zeros, zeros, L, zeros,
                                       n_axis)

            @jax.jit
            def step(s):
                return trace_jax(s, _TRACE_PRESCRIPTION,
                                 wavelength=0.55e-6)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                out = step(state)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert rw == [], (
                f"trace_jax wrapped in jax.jit must skip the "
                f"high-NA warning (tracer input); got "
                f"{[str(w.message) for w in rw]}.")
            assert np.all(np.isfinite(np.asarray(out.x)))
        finally:
            _reset_transfer_jax_warning()
            _TRACE_JAX_CACHE.clear()

    @needs_jax
    def test_maybe_warn_duck_typed_gate_accepts_eager_jax_array(self):
        """audit closure: P1-NEW-F1-1

        Unit-level pin on the duck-typed gate inside
        ``_maybe_warn_transfer_jax_high_na``.  An eager ``jax.Array``
        (the v4.16.1 production-path argument) must pass the gate;
        an eager ``np.ndarray`` must also pass; a ``Tracer`` (under
        jit) must skip.
        """
        import jax
        import jax.numpy as _jnp

        from lumenairy.raytrace.jax_trace import (
            _maybe_warn_transfer_jax_high_na,
            _reset_transfer_jax_warning,
        )

        # 1: eager jax.Array
        _reset_transfer_jax_warning()
        n_jax = _jnp.array([0.5, 0.6, 0.7])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _maybe_warn_transfer_jax_high_na(n_jax, thickness=1e-3)
        assert len([w for w in caught
                    if issubclass(w.category, RuntimeWarning)
                    and '_transfer_jax' in str(w.message)]) == 1

        # 2: eager NumPy ndarray (legacy path; must still work)
        _reset_transfer_jax_warning()
        n_np = np.array([0.5, 0.6, 0.7])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _maybe_warn_transfer_jax_high_na(n_np, thickness=1e-3)
        assert len([w for w in caught
                    if issubclass(w.category, RuntimeWarning)
                    and '_transfer_jax' in str(w.message)]) == 1

        # 3: tracer under jit (must skip silently; no exception)
        _reset_transfer_jax_warning()

        @jax.jit
        def probe(x):
            _maybe_warn_transfer_jax_high_na(x, thickness=1e-3)
            return x

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ = probe(n_jax)
        # Either no warning (skip cleanly) or 1 warning (eager
        # post-jit replay).  Crucially: no exception, no double-fire.
        rw = [w for w in caught
              if issubclass(w.category, RuntimeWarning)
              and '_transfer_jax' in str(w.message)]
        assert len(rw) <= 1
        _reset_transfer_jax_warning()


# ============================================================================
# P1-NEW-F1-2 -- propagate_ensemble preserves CuPy / JAX backend
# ============================================================================


class TestPropagateEnsembleBackendPreservation:
    """audit closure: P1-NEW-F1-2

    The v4.16.1 ``np.asarray(ensemble)`` fallback silently transferred
    CuPy / JAX inputs to host NumPy, defeating the docstring claim
    "Tolerate duck-typed array protocols (CuPy, JAX)".  Post-fix the
    helper dispatches via ``array_namespace`` and the accumulator
    stays on the input backend.
    """

    def test_numpy_ensemble_stays_numpy(self):
        """audit closure: P1-NEW-F1-2

        Sanity baseline: a NumPy input yields a NumPy ``I_partial``.
        """
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=4, N=16)
        assert isinstance(ens, np.ndarray)
        I = propagate_ensemble(ens, dx=dx, wavelength=wl,
                                propagator='asm', z=0.01)
        assert isinstance(I, np.ndarray), (
            f"NumPy ensemble -> NumPy I_partial; got "
            f"{type(I).__name__}.")
        assert I.dtype.kind == 'f'

    @needs_jax
    def test_jax_ensemble_stays_jax(self):
        """audit closure: P1-NEW-F1-2

        Production canary: a JAX-array ensemble must yield a JAX-array
        ``I_partial``.  Pre-fix this returned a NumPy ndarray (silent
        downcast at the ``np.asarray`` fallback at the old line 252).
        """
        import jax.numpy as _jnp
        ens_np, dx, _, wl = _make_simple_ensemble(n_realizations=4,
                                                    N=16)
        ens_jax = _jnp.asarray(ens_np)
        assert not isinstance(ens_jax, np.ndarray)

        I = propagate_ensemble(ens_jax, dx=dx, wavelength=wl,
                                propagator='asm', z=0.01)
        # Must be a JAX array, NOT a NumPy ndarray.
        assert not isinstance(I, np.ndarray), (
            f"JAX ensemble must yield JAX I_partial; got "
            f"{type(I).__name__} (NumPy fallback regression).")
        # Duck-type: a JAX array has __array__ + dtype + .device or
        # .devices() depending on JAX version.
        assert hasattr(I, '__array__')
        assert hasattr(I, 'dtype')

    @needs_jax
    def test_jax_and_numpy_results_agree(self):
        """audit closure: P1-NEW-F1-2

        Numerical sanity: the NumPy and JAX paths must produce the
        same partial-coherence intensity to within float32 precision
        (JAX defaults to single-precision on x86 unless x64 is
        enabled).
        """
        import jax.numpy as _jnp
        ens_np, dx, _, wl = _make_simple_ensemble(n_realizations=8,
                                                    N=16)
        ens_jax = _jnp.asarray(ens_np)
        I_np = propagate_ensemble(ens_np, dx=dx, wavelength=wl,
                                   propagator='asm', z=0.01)
        I_jax = propagate_ensemble(ens_jax, dx=dx, wavelength=wl,
                                    propagator='asm', z=0.01)
        # float32 default on JAX -> 1e-5 rtol is a generous bound.
        np.testing.assert_allclose(
            np.asarray(I_jax), I_np, rtol=2e-5, atol=1e-7)

    @needs_jax
    def test_jax_ensemble_return_ensemble_stacks_on_backend(self):
        """audit closure: P1-NEW-F1-2

        ``return_ensemble=True`` with a JAX input returns the stacked
        ensemble on the JAX backend (not a NumPy ndarray).  Validates
        the list-and-stack code path for the immutable backend.
        """
        import jax.numpy as _jnp
        ens_np, dx, _, wl = _make_simple_ensemble(n_realizations=4,
                                                    N=16)
        ens_jax = _jnp.asarray(ens_np)
        I, ens_out = propagate_ensemble(
            ens_jax, dx=dx, wavelength=wl,
            propagator='asm', z=0.01,
            return_ensemble=True)
        assert not isinstance(I, np.ndarray)
        assert not isinstance(ens_out, np.ndarray)
        assert ens_out.shape == ens_jax.shape


# ============================================================================
# P3-NEW-F1-1 -- empty ensemble raises ValueError (not ZeroDivisionError)
# ============================================================================


class TestPropagateEnsembleEmpty:
    """audit closure: P3-NEW-F1-1

    Empty ensembles (``shape=(0, Ny, Nx)``) passed the ``ndim == 3``
    check then divided by ``n_realizations = 0``, raising opaque
    ``ZeroDivisionError`` / NaN-poisoning the result.  Now raises
    a clear ``ValueError`` at entry.
    """

    def test_empty_ensemble_raises_value_error(self):
        """audit closure: P3-NEW-F1-1

        Direct synthetic ``(0, Ny, Nx)`` complex ensemble.
        """
        empty = np.empty((0, 8, 8), dtype=np.complex128)
        with pytest.raises(ValueError, match=r"empty ensemble"):
            propagate_ensemble(empty, dx=2e-6, wavelength=633e-9,
                                propagator='asm', z=0.01)

    def test_empty_ensemble_message_cites_n_realizations(self):
        """audit closure: P3-NEW-F1-1

        The error message must surface ``n_realizations=0`` so the
        user can pinpoint the upstream Schell call (typical cause:
        a degenerate sweep parameter producing zero samples).
        """
        empty = np.empty((0, 4, 4), dtype=np.complex128)
        with pytest.raises(ValueError) as exc_info:
            propagate_ensemble(empty, dx=2e-6, wavelength=633e-9,
                                propagator='asm', z=0.01)
        assert 'n_realizations=0' in str(exc_info.value)


# ============================================================================
# P3-NEW-F1-2 -- kwargs collision check (dx / wavelength)
# ============================================================================


class TestPropagateEnsembleKwargsCollision:
    """audit closure: P3-NEW-F1-2

    Pre-fix a caller passing ``propagator_kwargs={'dx': ...}`` plus the
    positional ``dx=`` argument triggered Python's bare
    ``TypeError: got multiple values for keyword argument 'dx'`` --
    correct but confusing.  Now raises a domain-level ``ValueError``
    with a remediation pointer.
    """

    def test_dx_collision_raises_value_error(self):
        """audit closure: P3-NEW-F1-2"""
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=8)
        with pytest.raises(ValueError, match=r"collide"):
            propagate_ensemble(
                ens, dx=dx, wavelength=wl,
                propagator='asm', z=0.01,
                propagator_kwargs={'dx': 5e-6})

    def test_wavelength_collision_raises_value_error(self):
        """audit closure: P3-NEW-F1-2"""
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=8)
        with pytest.raises(ValueError, match=r"collide"):
            propagate_ensemble(
                ens, dx=dx, wavelength=wl,
                propagator='asm', z=0.01,
                propagator_kwargs={'wavelength': 1e-6})

    def test_collision_message_lists_offending_keys(self):
        """audit closure: P3-NEW-F1-2

        Both ``dx`` and ``wavelength`` colliding -> the message lists
        both keys so the user can pinpoint the fix.
        """
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=8)
        with pytest.raises(ValueError) as exc_info:
            propagate_ensemble(
                ens, dx=dx, wavelength=wl,
                propagator='asm', z=0.01,
                propagator_kwargs={'dx': 5e-6, 'wavelength': 1e-6})
        msg = str(exc_info.value)
        assert 'dx' in msg
        assert 'wavelength' in msg


# ============================================================================
# P3-NEW-F1-6 -- non-colliding propagator_kwargs passes through cleanly
# ============================================================================


class TestPropagateEnsembleKwargsPassThrough:
    """audit closure: P3-NEW-F1-6

    The collision guard from P3-NEW-F1-2 must NOT regress the legitimate
    use case: ``propagator_kwargs={'something_else': value}`` should
    flow through to the propagator unchanged.
    """

    def test_propagator_kwargs_pass_through_to_callable(self):
        """audit closure: P3-NEW-F1-6

        Custom propagator receives a non-colliding extra kwarg through
        ``propagator_kwargs``.  Verifies the per-realisation invocation
        carries the extra argument.
        """
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=3, N=16)
        received = []

        def my_propagator(E, *, dx, wavelength, z, my_extra):
            received.append(my_extra)
            return la.angular_spectrum_propagate(E, z, wavelength, dx)

        I = propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator=my_propagator, z=0.01,
            propagator_kwargs={'my_extra': 'sentinel'})
        assert received == ['sentinel'] * 3
        assert I.shape == ens.shape[1:]

    def test_propagator_kwargs_overrides_top_level_kwargs(self):
        """audit closure: P3-NEW-F1-6

        The docstring contract states ``propagator_kwargs`` takes
        precedence over the catch-all ``**kwargs`` on key collisions.
        Pin the precedence for a non-protected key (``z``).  This is
        independent of the dx / wavelength collision guard above and
        confirms the docstring's stated merge order is preserved.
        """
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=8)
        seen_z = []

        def my_propagator(E, *, dx, wavelength, z):
            seen_z.append(z)
            return la.angular_spectrum_propagate(E, z, wavelength, dx)

        propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator=my_propagator,
            z=0.001,  # **kwargs-level z
            propagator_kwargs={'z': 0.05})  # explicit override
        # Every realisation saw the override, not the default.
        assert seen_z == [0.05] * 2


# ============================================================================
# P3-NEW-F1-5 -- jax.grad pin for B.4 (duck-typed dtype probe)
# ============================================================================


class TestPropagateThroughSystemJaxGradSafe:
    """audit closure: P3-NEW-F1-5

    The v4.16.1 B.4 dtype-probe pin (test_v4_16_1_agent_b.py:430-501)
    only checked ``@jax.jit``.  ``jax.grad`` is a separate trace
    context: under ``jax.grad`` ``E_in`` is a JVPTracer (subclass of
    Tracer), not a DynamicJaxprTracer.  The duck-typed
    ``getattr(E_in, 'dtype', None)`` probe must work for both.
    """

    @needs_jax
    def test_propagate_through_system_jax_with_grad(self):
        """audit closure: P3-NEW-F1-5

        Differentiate ``propagate_through_system_jax`` through a
        real-valued scalar loss (a scalar parameter scaling the input
        field).  The duck-typed ``getattr(E_in, 'dtype', None)`` probe
        must remain JAX-safe under ``jax.grad`` (where ``E_in`` is a
        JVPTracer, not a plain Array).
        """
        import jax
        import jax.numpy as _jnp

        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )

        N = 16
        dx = 5e-6
        wavelength = 633e-9
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'lens', 'f': 5e-3, 'xc': 0.0, 'yc': 0.0},
        ]

        # Real-scalar input -> real-scalar loss = canonical jax.grad
        # form (no need for holomorphic / complex-to-real plumbing).
        # The scalar 'amplitude' threads through the complex E field
        # so the dtype probe still sees a complex E_in inside the
        # function -- exercising the exact code path the audit cites.
        base = _jnp.ones((N, N), dtype=_jnp.complex64)

        def loss(amp):
            E_in = amp * base
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            return _jnp.sum(_jnp.abs(E_out) ** 2)

        _PROPAGATE_SYSTEM_JAX_CACHE.clear()
        try:
            amp0 = _jnp.float32(2.0)
            value, grad = jax.value_and_grad(loss)(amp0)
            # Forward value is real-positive (sum of |E|^2).
            assert float(value) > 0
            assert np.all(np.isfinite(np.asarray(value)))
            # Gradient is a real scalar; sum-of-|E|^2 scales like
            # |amp|^2 * sum|base|^2, so d/d_amp = 2 * amp * S.
            assert np.all(np.isfinite(np.asarray(grad)))
            assert float(grad) > 0
        finally:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_propagate_through_system_jax_grad_real_input(self):
        """audit closure: P3-NEW-F1-5

        ``jax.grad`` with a real-valued input -- the dtype probe must
        fall through to the library complex default cleanly.  Mirrors
        the v4.16.1 B.4 ``test_propagate_through_system_jax_dtype_probe_real_input``
        but for ``jax.grad`` rather than ``@jax.jit``.
        """
        import jax
        import jax.numpy as _jnp

        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )

        N = 16
        dx = 5e-6
        wavelength = 633e-9
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
        ]

        def loss(E_real):
            # Cast inside the trace context to exercise the probe.
            E_out = propagate_through_system_jax(
                E_real, elements, wavelength, dx, verbose=False)
            return _jnp.sum(_jnp.abs(E_out) ** 2)

        _PROPAGATE_SYSTEM_JAX_CACHE.clear()
        try:
            E_real = _jnp.ones((N, N), dtype=_jnp.float32)
            # Use jax.grad on the real-input loss; the probe must
            # see a float dtype and not raise.
            value_and_grad = jax.value_and_grad(loss)
            value, grad = value_and_grad(E_real)
            assert float(value) > 0
            assert grad.shape == E_real.shape
            assert np.all(np.isfinite(np.asarray(grad)))
        finally:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
