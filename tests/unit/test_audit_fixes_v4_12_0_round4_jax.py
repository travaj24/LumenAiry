"""Pinning tests for the v4.12.0 round-4 audit Tier-1 JAX fixes.

The round-4 audit (``AUDIT_ROUND4_2026_05_16.md``) identified three
correctness issues in the JAX backends that this test module pins:

* **B1-1 -- JAX/NumPy aperture schemas incompatible.**
  ``system.py`` (JAX path) read ``params.get('radius')`` while NumPy
  ``apply_aperture`` reads ``params.get('diameter')``.  A working
  NumPy element list ported to ``propagate_through_system_jax`` had
  every aperture silently no-op'd because ``params.get('radius')``
  returned ``None``.  Pinning:

    1. Canonical NumPy schema (``params={'diameter': ...}``) works
       end-to-end (no warning, correct aperture clipping).
    2. Legacy schema (``params={'radius': ...}``) still works but
       emits a one-shot :class:`DeprecationWarning`.
    3. All three shapes (circular / rectangular / annular) accept
       both schemas.

* **B1-2 -- ``propagate_through_system_jax`` was not actually JAX-end-
  to-end traceable.**  Pre-v4.12 the function silently fell back to
  NumPy via ``np.asarray(E)`` for unsupported element types, which
  raises ``TracerArrayConversionError`` under ``jax.jit`` /
  ``jax.grad``.  v4.12 raises :class:`NotImplementedError` at call
  time listing the offending element types.  Pinning:

    1. Element lists with only supported types
       (:data:`_TRACEABLE_ELEMENT_TYPES`) run without error.
    2. Lists containing any unsupported type raise
       :class:`NotImplementedError` with the offending type names in
       the error message.
    3. ``_TRACEABLE_ELEMENT_TYPES`` is exposed and contains the
       expected 4-element set.

* **B1-9 -- ``_apply_doe_kick_jax`` blocked gradients.**
  Pre-v4.12 used ``float(period_x)`` (strips trace -> silent zero
  gradient) and ``np.isfinite(period_x)`` (raises on a traced array).
  v4.12 keeps the JAX trace alive via ``jnp.where`` whenever the
  period argument is traced.  Pinning:

    1. A concrete-period call returns finite, sensible state.
    2. ``jax.grad`` w.r.t. ``period_x`` returns a FINITE, NON-ZERO
       gradient (the actual round-3 + round-4 finding).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

# Skip the whole module if JAX is unavailable.
jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')


import lumenairy as lm  # noqa: E402  (after importorskip)
from lumenairy.system import (  # noqa: E402
    _PROPAGATE_SYSTEM_JAX_CACHE,
    _TRACEABLE_ELEMENT_TYPES,
    propagate_through_system_jax,
)


# ===========================================================================
# B1-1 -- JAX/NumPy aperture schema unification
# ===========================================================================

class TestB1_1_ApertureSchemaUnification:
    """``propagate_through_system_jax`` aperture handling must accept
    BOTH the canonical NumPy schema (``diameter`` / ``width_x`` /
    ``inner_diameter``) AND the legacy JAX-only schema (``radius`` /
    ``half_width_x`` / ``inner_radius``).  The legacy form emits a
    one-shot :class:`DeprecationWarning`."""

    @pytest.fixture
    def field(self):
        N, dx, wl = 64, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        return E, dx, wl

    @pytest.fixture(autouse=True)
    def _reset_one_shot_warn(self):
        """Reset the module-level one-shot deprecation latch so each
        test gets a fresh warning emission."""
        import lumenairy.system as lms
        lms._LEGACY_APERTURE_SCHEMA_WARNED = False
        yield
        lms._LEGACY_APERTURE_SCHEMA_WARNED = False

    # ---- circular ----------------------------------------------------

    def test_circular_diameter_canonical_no_warning(self, field):
        """``params={'diameter': D}`` is the canonical NumPy schema:
        must clip the field AND not emit a deprecation warning."""
        E, dx, wl = field
        diameter = 100e-6
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': diameter}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # The aperture must actually clip: some pixels become 0.
        n_zero = int(np.sum(np.abs(E_np) < 1e-9))
        n_one = int(np.sum(np.abs(E_np) > 0.5))
        assert n_zero > 0, (
            "Canonical 'diameter' schema must clip the field "
            "(audit B1-1: pre-fix silently no-op'd).")
        assert n_one > 0, "Some pixels must remain inside the aperture."
        # And no deprecation warning fires.
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep, (
            f"Canonical schema must not emit DeprecationWarning; "
            f"got: {[str(wi.message)[:80] for wi in dep]}")

    def test_circular_radius_legacy_warns_once(self, field):
        """``params={'radius': r}`` is the legacy JAX-only schema:
        still works but emits a one-shot DeprecationWarning."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'radius': 50e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        n_zero = int(np.sum(np.abs(E_np) < 1e-9))
        assert n_zero > 0, "Legacy 'radius' schema must still clip the field."
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert len(dep) >= 1, (
            "Legacy schema must emit at least one DeprecationWarning.")
        msg = str(dep[0].message)
        assert 'radius' in msg or 'legacy' in msg.lower(), (
            f"Deprecation message should mention 'radius'/'legacy'; got: {msg}")

    def test_circular_canonical_and_legacy_agree(self, field):
        """The same circular aperture expressed either way must
        produce the same clipped field (within float tolerance)."""
        E, dx, wl = field
        radius = 50e-6
        diameter = 2 * radius

        E_canon = propagate_through_system_jax(
            E, [{'type': 'aperture', 'shape': 'circular',
                 'params': {'diameter': diameter}}],
            wl, dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            E_leg = propagate_through_system_jax(
                E, [{'type': 'aperture', 'shape': 'circular',
                     'params': {'radius': radius}}],
                wl, dx)

        np.testing.assert_allclose(np.asarray(E_canon),
                                    np.asarray(E_leg), atol=1e-7)

    # ---- rectangular -------------------------------------------------

    def test_rectangular_width_canonical(self, field):
        """``params={'width_x': Wx, 'width_y': Wy}`` is the NumPy
        canonical schema (full widths)."""
        E, dx, wl = field
        N = E.shape[0]
        elements = [
            {'type': 'aperture', 'shape': 'rectangular',
             'params': {'width_x': 40e-6, 'width_y': 60e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # Center pixel inside the aperture: |E| ~ 1.
        cx = cy = N // 2
        assert abs(E_np[cy, cx]) > 0.5
        # Far-corner pixel outside the aperture: |E| == 0.
        assert abs(E_np[0, 0]) < 1e-9
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep, "Canonical schema must not emit DeprecationWarning."

    def test_rectangular_half_width_legacy_warns(self, field):
        """``params={'half_width_x', 'half_width_y'}`` is the legacy
        JAX schema (half-widths) -- must warn."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'rectangular',
             'params': {'half_width_x': 20e-6, 'half_width_y': 30e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # The corner pixel is outside the half-extents -> zero.
        assert abs(E_np[0, 0]) < 1e-9
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert len(dep) >= 1, (
            "Legacy half_width schema must emit DeprecationWarning.")

    # ---- annular -----------------------------------------------------

    def test_annular_diameter_canonical(self, field):
        """``params={'inner_diameter', 'outer_diameter'}`` is the
        canonical schema."""
        E, dx, wl = field
        N = E.shape[0]
        elements = [
            {'type': 'aperture', 'shape': 'annular',
             'params': {'inner_diameter': 40e-6,
                        'outer_diameter': 200e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E_out = propagate_through_system_jax(E, elements, wl, dx)
        E_np = np.asarray(E_out)
        # Origin pixel is inside the inner hole -> zero.
        cx = cy = N // 2
        assert abs(E_np[cy, cx]) < 1e-9
        # An off-axis pixel inside the ring -> ~1.
        # Pixel at (cy, cx + 10) is at x = 10*dx = 50e-6 m (in the ring).
        assert abs(E_np[cy, cx + 10]) > 0.5
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert not dep

    def test_annular_radius_legacy_warns(self, field):
        """``params={'inner_radius', 'outer_radius'}`` is the legacy
        schema -- must warn."""
        E, dx, wl = field
        elements = [
            {'type': 'aperture', 'shape': 'annular',
             'params': {'inner_radius': 20e-6, 'outer_radius': 100e-6}},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            propagate_through_system_jax(E, elements, wl, dx)
        dep = [wi for wi in w if issubclass(wi.category, DeprecationWarning)]
        assert len(dep) >= 1, (
            "Legacy annular schema must emit DeprecationWarning.")

    # ---- cross-backend agreement -------------------------------------

    def test_jax_aperture_matches_numpy(self, field):
        """The MAIN B1-1 finding: an aperture spec built with the
        canonical NumPy schema and passed to BOTH backends must
        produce the same clipped field (within fp32 precision on the
        circle boundary).  Pre-fix the JAX path silently no-op'd
        because it read ``params.get('radius')``."""
        E, dx, wl = field
        # Use a diameter that doesn't land on an integer multiple of
        # dx so the boundary doesn't sit exactly on a grid pixel
        # (different fp32 vs fp64 rounding on ``X**2 + Y**2 <= r**2``
        # can flip a handful of boundary pixels in either direction).
        elements = [
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 97.5e-6}},
        ]
        # NumPy backend.
        E_np_out, _ = lm.propagate_through_system(
            E, elements, wl, dx, method='asm')
        # JAX backend.
        E_jax_out = propagate_through_system_jax(E, elements, wl, dx)
        # Pixel-wise mismatch must be bounded by the perimeter of the
        # disk (~ pi*D/dx ~ 60 pixels); insist no more than ~5% drift.
        diff = np.asarray(E_jax_out) - E_np_out.astype(np.complex64)
        n_mismatch = int(np.sum(np.abs(diff) > 1e-6))
        total = E_np_out.size
        assert n_mismatch < 0.05 * total, (
            f"Cross-backend mismatch must be limited to the disk "
            f"boundary; got {n_mismatch} / {total} mismatched pixels.")
        # And the JAX result is NOT just E_in unchanged (the pre-fix
        # behaviour) -- it's actually clipped.
        n_zero = int(np.sum(np.abs(np.asarray(E_jax_out)) < 1e-9))
        assert n_zero > 0, (
            "JAX aperture must clip the field; pre-fix silently "
            "no-op'd because params.get('radius') returned None.")


# ===========================================================================
# B1-2 -- propagate_through_system_jax fail-fast on non-traceable elements
# ===========================================================================

class TestB1_2_NonTraceableElementFailFast:
    """Element types without a JAX handler must raise
    :class:`NotImplementedError` at call time, listing the offending
    types.  Pre-v4.12 the function silently fell back to NumPy via
    ``np.asarray(E)``, which crashes under ``jax.jit`` / ``jax.grad``."""

    def test_traceable_element_types_constant(self):
        """The module-level constant must exist and contain the
        expected 4-element set."""
        assert isinstance(_TRACEABLE_ELEMENT_TYPES, frozenset)
        assert _TRACEABLE_ELEMENT_TYPES == frozenset(
            {'propagate', 'lens', 'aperture', 'mask'})

    def test_pure_traceable_chain_runs(self):
        """An element list with only traceable types runs without
        error."""
        N, dx, wl = 32, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'lens', 'f': 5e-3},
            {'type': 'aperture', 'shape': 'circular',
             'params': {'diameter': 100e-6}},
            {'type': 'mask',
             'mask': np.ones((N, N), dtype=np.complex64)},
        ]
        E_out = propagate_through_system_jax(E, elements, wl, dx)
        assert E_out.shape == (N, N)
        assert np.all(np.isfinite(np.asarray(E_out)))

    @pytest.mark.parametrize('etype, extra_keys', [
        ('spherical_lens', {'R1': 5e-3, 'R2': -5e-3, 'd': 1e-3,
                             'n_lens': 1.5}),
        ('aspheric_lens',  {'R1': 5e-3, 'R2': -5e-3, 'd': 1e-3,
                             'n_lens': 1.5, 'k1': 0.0, 'k2': 0.0}),
        ('mirror',         {'radius': -100e-3}),
        ('gaussian_aperture', {'sigma': 50e-6}),
        ('turbulence',     {'r0': 1e-2}),
        ('zernike',        {'coefficients': [0.0],
                             'aperture_radius': 1e-3}),
        ('cylindrical_lens', {'f': 5e-3}),
        ('axicon',         {'alpha': 0.01, 'n_axicon': 1.5}),
        ('grin_lens',      {'n0': 1.5, 'g': 0.1, 'd': 1e-3}),
        ('propagate_tilted', {'z': 1e-3}),
    ])
    def test_non_traceable_raises_not_implemented(self, etype, extra_keys):
        """Each non-traceable element type triggers a clear
        ``NotImplementedError`` (one row per element type listed in
        the audit B1-2 finding)."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elem = {'type': etype, **extra_keys}
        with pytest.raises(NotImplementedError) as excinfo:
            propagate_through_system_jax(E, [elem], wl, dx)
        assert etype in str(excinfo.value), (
            f"NotImplementedError message must name the offending "
            f"element type {etype!r}; got: {excinfo.value}")

    def test_spherical_lens_in_mixed_list_raises(self):
        """A mixed list with one ``spherical_lens`` among otherwise
        traceable types must still raise."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'propagate', 'z': 1e-3},
            {'type': 'spherical_lens', 'R1': 5e-3, 'R2': -5e-3,
             'd': 1e-3, 'n_lens': 1.5},
            {'type': 'propagate', 'z': 1e-3},
        ]
        with pytest.raises(NotImplementedError, match='spherical_lens'):
            propagate_through_system_jax(E, elements, wl, dx)

    def test_fail_fast_does_not_pollute_cache(self):
        """The fail-fast check runs BEFORE any kernel build / cache
        insertion, so a NotImplementedError must not pollute the
        kernel cache."""
        N, dx, wl = 16, 5e-6, 1.55e-6
        E = np.ones((N, N), dtype=np.complex64)
        elements = [
            {'type': 'mirror', 'radius': -100e-3},
        ]
        cache_size_before = len(_PROPAGATE_SYSTEM_JAX_CACHE)
        with pytest.raises(NotImplementedError):
            propagate_through_system_jax(E, elements, wl, dx)
        assert len(_PROPAGATE_SYSTEM_JAX_CACHE) == cache_size_before


# ===========================================================================
# B1-9 -- _apply_doe_kick_jax gradient flow w.r.t. grating period
# ===========================================================================

class TestB1_9_DoeKickJaxGradient:
    """The DOE-kick path must keep the JAX trace alive when
    ``period_x`` / ``period_y`` are JAX-traced values, so users can
    ``jax.grad`` w.r.t. grating period.  Pre-v4.12 used
    ``float(period_x)`` (strips trace, zero gradient) and
    ``np.isfinite(period_x)`` (raises on a tracer)."""

    def _make_initial_state(self, n_rays=8):
        """Build a JaxRayState ray bundle parallel to z, slightly
        off-axis so the DOE kick affects (x, y, opd) downstream."""
        from lumenairy.raytrace.jax_trace import make_jax_ray_state
        # Rays at non-zero (x, y) so the OPL contribution ``dL*x +
        # dM*y`` is non-zero and depends on period.
        x = jnp.linspace(-0.5e-3, 0.5e-3, n_rays)
        y = jnp.linspace(-0.3e-3, 0.3e-3, n_rays)
        z = jnp.zeros(n_rays)
        L = jnp.zeros(n_rays)
        M = jnp.zeros(n_rays)
        N = jnp.ones(n_rays)
        opd = jnp.zeros(n_rays)
        alive = jnp.ones(n_rays, dtype=bool)
        return make_jax_ray_state(x=x, y=y, z=z, L=L, M=M, N=N,
                                   opd=opd, alive=alive)

    def test_concrete_period_runs(self):
        """Sanity: with concrete Python floats for period_x /
        period_y, the function returns a finite state."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6
        order_x, order_y = 1, 0
        period_x, period_y = 5e-6, float('inf')
        out = _apply_doe_kick_jax(state, order_x, order_y,
                                    period_x, period_y, wavelength)
        # All output fields finite.
        for name in ('x', 'y', 'L', 'M', 'N', 'opd'):
            arr = np.asarray(getattr(out, name))
            assert np.all(np.isfinite(arr)), f"{name} not finite"
        # The kick along x should equal m*lambda/period.
        dL_expected = order_x * wavelength / period_x
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.zeros(out.L.shape) + dL_expected,
                                    atol=1e-12)

    def test_grad_w_r_t_period_finite_nonzero(self):
        """THE B1-9 PIN: ``jax.grad`` w.r.t. ``period_x`` must return a
        finite, non-zero scalar.  Pre-v4.12 the ``float(period_x)``
        strip produced silent zero gradient (or crash on the
        ``np.isfinite`` of a tracer)."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_x):
            """Apply DOE kick with traced period_x; return a scalar
            function of the result so we can grad it.  Sum of x²+y²
            after a transfer step makes the result period-dependent."""
            order_x, order_y = 1, 0
            period_y = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            # Propagate the ray a fixed thickness so dL contributes to
            # x_out (otherwise dL only shows up at the next surface).
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        grad_fn = jax.grad(trace_and_reduce)
        period_x = jnp.float32(5e-6)
        g = grad_fn(period_x)
        g_np = float(np.asarray(g))
        assert np.isfinite(g_np), (
            f"Gradient w.r.t. period_x must be finite; got {g_np}")
        assert abs(g_np) > 1e-12, (
            f"Gradient w.r.t. period_x must be non-zero (pre-fix "
            f"``float(period_x)`` stripped the trace and produced "
            f"silent zero gradient); got {g_np}")

    def test_grad_w_r_t_period_y_finite_nonzero(self):
        """Same pin for the y-axis grating period."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_y):
            order_x, order_y = 0, 1
            period_x = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        grad_fn = jax.grad(trace_and_reduce)
        period_y = jnp.float32(5e-6)
        g = grad_fn(period_y)
        g_np = float(np.asarray(g))
        assert np.isfinite(g_np)
        assert abs(g_np) > 1e-12

    def test_grad_matches_analytic_sign(self):
        """Sanity-check the sign of the gradient against the analytic
        derivative.  ``trace_and_reduce`` ~ ``(dL * t)^2 * n_rays`` for
        on-axis rays (x_init = 0), where ``dL = lambda / period_x``.
        ``d/dperiod_x[(lambda*t/period)^2 * n] = -2*lambda^2*t^2*n /
        period^3 < 0``.  Even with off-axis rays, the leading
        period-dependent term still has negative derivative."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax

        state = self._make_initial_state()
        wavelength = 1.55e-6

        def trace_and_reduce(period_x):
            order_x, order_y = 1, 0
            period_y = jnp.inf
            new_state = _apply_doe_kick_jax(
                state, order_x, order_y,
                period_x, period_y, wavelength)
            thickness = 10e-3
            x_out = new_state.x + new_state.L * thickness
            y_out = new_state.y + new_state.M * thickness
            return jnp.sum(x_out * x_out + y_out * y_out)

        # Compare JAX grad to forward-difference numeric grad.
        period_x = 5e-6
        eps = 1e-9
        f_plus = float(trace_and_reduce(jnp.float64(period_x + eps)))
        f_minus = float(trace_and_reduce(jnp.float64(period_x - eps)))
        g_fd = (f_plus - f_minus) / (2 * eps)

        g_jax = float(jax.grad(trace_and_reduce)(jnp.float64(period_x)))
        # Sign agreement; numeric value within ~1%.
        assert np.sign(g_jax) == np.sign(g_fd), (
            f"JAX-grad sign ({g_jax}) disagrees with finite-difference "
            f"({g_fd}).")
        rel = abs(g_jax - g_fd) / max(abs(g_fd), 1e-30)
        assert rel < 1e-2, (
            f"JAX-grad / FD mismatch: jax={g_jax:.6e}, fd={g_fd:.6e}, "
            f"rel={rel:.3e}")

    def test_infinite_period_concrete_no_kick(self):
        """Concrete ``inf`` period (1-D grating) must produce zero
        kick along that axis (the no-grating branch)."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6
        out = _apply_doe_kick_jax(
            state, order_x=1, order_y=0,
            period_x=np.inf, period_y=np.inf, wavelength=wavelength)
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.asarray(state.L))
        np.testing.assert_allclose(np.asarray(out.M),
                                    np.asarray(state.M))

    def test_infinite_period_traced_no_kick(self):
        """Traced ``jnp.inf`` period must also produce zero kick (the
        ``jnp.where`` guard).  This is the path that pre-fix raised
        on ``np.isfinite(tracer)``."""
        from lumenairy.raytrace.jax_trace import _apply_doe_kick_jax
        state = self._make_initial_state()
        wavelength = 1.55e-6

        def kick_with_traced_period(period_x):
            return _apply_doe_kick_jax(
                state, 1, 0, period_x, jnp.inf, wavelength)

        # Should NOT raise (pre-fix crashed on np.isfinite(tracer)).
        out = jax.jit(kick_with_traced_period)(jnp.inf)
        np.testing.assert_allclose(np.asarray(out.L),
                                    np.asarray(state.L), atol=1e-12)
