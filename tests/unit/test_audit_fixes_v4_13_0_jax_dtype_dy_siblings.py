"""Pinning tests for the v4.13.0 Track-C audit closures.

Audit reference
---------------

``AUDIT_V4_12_1_2026_05_16.md`` Part 5 / Round 5 carried four
known-limitations items into v4.12.2's CHANGELOG.  v4.13.0 closes the
following three:

* **L2** -- JAX path complex64 hard-casts silently override
  ``set_default_complex_dtype``.  v4.13.0 routes every JAX-side dtype
  resolution through :func:`_resolve_jax_complex_dtype` (and its real
  twin :func:`_resolve_jax_real_dtype`) which read
  :func:`get_default_complex_dtype`.  JIT cache keys that previously
  omitted the dtype now include it.
* **L3** -- :class:`PropagationResult` lost the y-axis pitch for
  anamorphic Fresnel propagation because ``_coerce_field`` extracted
  ``dx_out`` from tuple-returning kernels but silently discarded
  ``dy_out``.  v4.13.0 adds a ``dy`` field on both
  :class:`PropagationResult` (with ``dy_out`` alias) and
  :class:`lumenairy.sources.Source`; defaults to ``dx`` for back-compat.
* **L4a** -- the v4.11.2 pre-flight mirror guard in
  ``apply_real_lens_traced`` was not ported to the three siblings
  (``apply_real_lens_maslov``, ``apply_real_lens_traced_jax``,
  ``apply_real_lens_maslov_jax``).  A hand-built prescription with
  ``surfaces[i]['is_mirror']=True`` slipped past on all three.
* **L4b** -- ``error_reduction(backend='jax')`` /
  ``hybrid_input_output(backend='jax')`` dispatchers silently dropped
  the NumPy-API ``initial_guess`` kwarg.  Mapping
  ``initial_guess`` (complex object-plane field) to the JAX twin's
  ``init_phase`` (real Fourier-plane phase) is not lossless; v4.13.0
  raises ``NotImplementedError`` to force the user to make an
  explicit choice.
* **L4c** -- ``gerchberg_saxton(backend='jax', return_history=True)``
  silently dropped ``return_history`` and returned a 2-tuple instead
  of a 3-tuple.  v4.13.0 emits a ``RuntimeWarning`` and returns a
  synthetic 3-tuple with an empty history list so the return shape
  always matches the NumPy API.

Each test class below pins one closure.

Author: Andrew Traverso
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la


JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason="JAX is not installed")


# ============================================================================
# L2 -- JAX dtype unification via _resolve_jax_complex_dtype
# ============================================================================

class TestL2JaxDtypeUnification:
    """``set_default_complex_dtype(np.complex128)`` propagates to every
    JAX entry point.  Pre-fix the JAX-side hard-casts (``jnp.asarray(
    E_in, dtype=jnp.complex64)``, ``.astype(jnp.complex64)``, and
    ``jax.config.jax_enable_x64`` reads in ``_lens_jax.py``) all
    bypassed ``set_default_complex_dtype`` and gave float32-precision
    answers with no warning.
    """

    def test_resolve_jax_complex_dtype_reads_default(self):
        """When no override is passed, ``_resolve_jax_complex_dtype``
        reads :func:`get_default_complex_dtype`."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp
        from lumenairy.propagators.propagation import (
            _resolve_jax_complex_dtype, set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_complex_dtype() == _jnp.complex128
            set_default_complex_dtype(np.complex64)
            assert _resolve_jax_complex_dtype() == _jnp.complex64
        finally:
            set_default_complex_dtype(prior)

    def test_resolve_jax_complex_dtype_override(self):
        """An explicit per-call override takes precedence over the
        library default."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp
        from lumenairy.propagators.propagation import (
            _resolve_jax_complex_dtype, set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            # Even with default=complex64, an explicit complex128
            # override should win.
            assert _resolve_jax_complex_dtype(np.complex128) == _jnp.complex128
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_complex_dtype(np.complex64) == _jnp.complex64
        finally:
            set_default_complex_dtype(prior)

    def test_resolve_jax_real_dtype_pairs_with_complex(self):
        """The real twin returns ``float64`` paired with ``complex128``
        and ``float32`` paired with ``complex64``."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not installed")
        import jax.numpy as _jnp
        from lumenairy.propagators.propagation import (
            _resolve_jax_real_dtype, set_default_complex_dtype,
        )
        prior = la.get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex128)
            assert _resolve_jax_real_dtype() == _jnp.float64
            set_default_complex_dtype(np.complex64)
            assert _resolve_jax_real_dtype() == _jnp.float32
        finally:
            set_default_complex_dtype(prior)

    @needs_jax
    def test_propagate_through_system_jax_complex128(self):
        """Direction 1: ``set_default_complex_dtype(np.complex128)``
        then run ``propagate_through_system_jax`` -- the JAX result
        must be ``jnp.complex128``.

        Pre-fix the ``E = jnp.asarray(E_in, dtype=jnp.complex64)``
        hard-cast at the top of ``propagate_through_system_jax``
        silently demoted to single precision regardless of the
        configured default.
        """
        import jax.numpy as _jnp
        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.system import (
            propagate_through_system_jax,
            _PROPAGATE_SYSTEM_JAX_CACHE,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            set_default_complex_dtype(np.complex128)
            N = 32
            dx = 5e-6
            wavelength = 633e-9
            # Real-valued input so the dtype comes from the default,
            # not from a complex caller dtype.
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
                {'type': 'lens', 'f': 5e-3, 'xc': 0.0, 'yc': 0.0},
            ]
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            assert E_out.dtype == _jnp.complex128, (
                f"Expected jnp.complex128 with "
                f"set_default_complex_dtype(np.complex128); got "
                f"{E_out.dtype!r}.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_propagate_through_system_jax_complex64(self):
        """Direction 2: ``set_default_complex_dtype(np.complex64)``
        then run ``propagate_through_system_jax`` -- the JAX result
        must be ``jnp.complex64`` (which was also the historical
        hardcoded value, but now via the helper rather than a
        silent override)."""
        import jax.numpy as _jnp
        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.system import (
            propagate_through_system_jax,
            _PROPAGATE_SYSTEM_JAX_CACHE,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            set_default_complex_dtype(np.complex64)
            N = 32
            dx = 5e-6
            wavelength = 633e-9
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            ]
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            assert E_out.dtype == _jnp.complex64, (
                f"Expected jnp.complex64 with "
                f"set_default_complex_dtype(np.complex64); got "
                f"{E_out.dtype!r}.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_jit_cache_key_separates_dtypes(self):
        """Pre-fix the ``_PROPAGATE_SYSTEM_JAX_CACHE`` key omitted dtype,
        so a complex64 call and a complex128 call at the same shape /
        wavelength / dx would collide on a single cached XLA kernel.
        v4.13.0 adds ``str(np.dtype(cdtype))`` to the key so each
        dtype gets its own kernel."""
        from lumenairy.propagators.propagation import (
            set_default_complex_dtype,
        )
        from lumenairy.system import (
            propagate_through_system_jax,
            _PROPAGATE_SYSTEM_JAX_CACHE,
        )
        prior = la.get_default_complex_dtype()
        try:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()
            N = 16
            dx = 5e-6
            wavelength = 633e-9
            E_in = np.ones((N, N), dtype=np.float32)
            elements = [
                {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            ]
            set_default_complex_dtype(np.complex64)
            propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            n_after_c64 = len(_PROPAGATE_SYSTEM_JAX_CACHE)
            set_default_complex_dtype(np.complex128)
            propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            n_after_c128 = len(_PROPAGATE_SYSTEM_JAX_CACHE)
            assert n_after_c128 > n_after_c64, (
                f"JIT cache should grow when dtype changes; before "
                f"complex128 call had {n_after_c64} entries, after "
                f"had {n_after_c128}.  Pre-fix the cache key omitted "
                f"dtype so the two calls collided.")
        finally:
            set_default_complex_dtype(prior)
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()


# ============================================================================
# L3 -- PropagationResult.dy_out + Source.dy
# ============================================================================

class TestL3PropagationResultDy:
    """:class:`PropagationResult` and :class:`Source` carry the y-axis
    pitch so anamorphic Fresnel propagation no longer silently
    discards it.
    """

    def test_propagation_result_has_dy_field(self):
        """Direct field check: ``PropagationResult.dy`` exists and
        defaults to ``dx``."""
        from lumenairy.propagators.result import PropagationResult
        pr = PropagationResult(
            field=np.zeros((4, 4), dtype=np.complex128),
            dx=1.5e-6, wavelength=633e-9,
        )
        assert hasattr(pr, 'dy')
        # Back-compat: square-grid default falls back to dx.
        assert pr.dy == pr.dx

    def test_propagation_result_dy_out_alias(self):
        """``dy_out`` alias mirrors ``dx_out`` (matches the
        tuple-returning kernel naming convention)."""
        from lumenairy.propagators.result import PropagationResult
        pr = PropagationResult(
            field=np.zeros((4, 4), dtype=np.complex128),
            dx=2.0e-6, dy=3.0e-6, wavelength=633e-9,
        )
        assert pr.dx_out == 2.0e-6
        assert pr.dy_out == 3.0e-6

    def test_source_has_dy_field(self):
        """``Source`` carries a ``dy`` attribute that defaults to
        ``dx`` when not given."""
        from lumenairy.sources.core import Source
        src = Source(E=np.zeros((4, 4), dtype=np.complex128),
                     dx=1e-6, wavelength=633e-9)
        assert hasattr(src, 'dy')
        # Back-compat default.
        assert src.dy == src.dx

    def test_source_dy_explicit(self):
        """Explicit ``dy`` is honoured."""
        from lumenairy.sources.core import Source
        src = Source(E=np.zeros((4, 4), dtype=np.complex128),
                     dx=1e-6, dy=2e-6, wavelength=633e-9)
        assert src.dy == 2e-6

    def test_anamorphic_fresnel_threads_dy_out(self):
        """Pin: anamorphic Fresnel propagation (different input dx /
        dy) returns distinct ``dx_out, dy_out`` from the kernel, and
        the wrapped :class:`PropagationResult` exposes both."""
        from lumenairy.propagators.propagation import fresnel_propagate
        from lumenairy.propagators.dispatch import _coerce_field
        from lumenairy.propagators.result import PropagationResult

        N = 32
        dx_in = 5e-6
        dy_in = 1e-6   # rectangular pixels
        wavelength = 633e-9
        z = 5e-3
        x = (np.arange(N) - N / 2) * dx_in
        y = (np.arange(N) - N / 2) * dy_in
        X, Y = np.meshgrid(x, y, indexing='xy')
        E_in = np.exp(-(X * X + Y * Y) / (5 * dx_in) ** 2).astype(np.complex128)

        # Raw kernel returns (E_out, dx_out, dy_out).
        E_out, dx_out_kernel, dy_out_kernel = fresnel_propagate(
            E_in, z, wavelength, dx_in, dy=dy_in)
        # Anamorphic geometry -> distinct output pitches.
        assert abs(dx_out_kernel - dy_out_kernel) > 1e-9 * dx_out_kernel, (
            f"Anamorphic input should yield distinct dx_out / dy_out; "
            f"got dx_out={dx_out_kernel}, dy_out={dy_out_kernel}.")

        # _coerce_field now returns a 3-tuple; ensure dy_out lands.
        field, dx_coerced, dy_coerced = _coerce_field(
            (E_out, dx_out_kernel, dy_out_kernel))
        assert field is E_out
        assert dx_coerced == dx_out_kernel
        assert dy_coerced == dy_out_kernel

        # End-to-end: dispatcher wrap-level thread.
        pr = PropagationResult(
            field=E_out, dx=dx_out_kernel, dy=dy_out_kernel,
            wavelength=wavelength, z=z, method='fresnel',
        )
        assert pr.dx_out == dx_out_kernel
        assert pr.dy_out == dy_out_kernel
        # And the round-trip through to_source preserves dy.
        src = pr.to_source()
        assert src.dx == dx_out_kernel
        assert src.dy == dy_out_kernel


# ============================================================================
# L4a -- mirror guard ported to the three apply_real_lens_* siblings
# ============================================================================

def _hand_built_mirror_prescription():
    """Construct a minimal prescription that puts ``is_mirror=True``
    directly into the surfaces list (the slip-past case the
    apply_real_lens_traced v4.11.2 guard caught for the parent
    function but not the siblings).
    """
    # Two refracting surfaces sandwiching a mirror surface in the
    # middle.  The mirror flag should trigger the guard regardless of
    # whether the rest of the geometry is sensible.
    return {
        'surfaces': [
            {'radius': 0.025, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'conic': 0.0},
            # Hand-built mirror entry the audit flagged.
            {'radius': float('inf'), 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'is_mirror': True,
             'conic': 0.0},
            {'radius': -0.025, 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air',
             'semi_diameter': 0.010, 'conic': 0.0},
        ],
        'aperture_diameter': 0.020,
    }


class TestL4aMirrorGuardSiblings:
    """Pin the mirror guard fires for all three siblings of
    ``apply_real_lens_traced``."""

    def test_apply_real_lens_maslov_raises_on_hand_built_mirror(self):
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_maslov(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    @needs_jax
    def test_apply_real_lens_traced_jax_raises_on_hand_built_mirror(self):
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_traced_jax(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    @needs_jax
    def test_apply_real_lens_maslov_jax_raises_on_hand_built_mirror(self):
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens_maslov_jax(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)


# ============================================================================
# L4b -- JAX-dispatch ``initial_guess`` forwarding / refusal
# ============================================================================

class TestL4bInitialGuessForwarding:
    """v4.13.0 decision: refuse to silently demote a NumPy-API
    ``initial_guess`` (complex object-plane field) into a JAX-API
    ``init_phase`` (real Fourier-plane phase).  Pre-fix the dispatcher
    silently dropped ``initial_guess`` on the JAX path.
    """

    @needs_jax
    def test_error_reduction_jax_with_initial_guess_raises(self):
        """``error_reduction(backend='jax', initial_guess=...)`` raises
        ``NotImplementedError`` with a clear migration message."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        ig = (np.ones((N, N), dtype=np.complex128) * 0.1).astype(np.complex128)
        with pytest.raises(NotImplementedError, match=r'init_phase'):
            la.error_reduction(
                meas, sup, n_iter=5, initial_guess=ig, backend='jax')

    @needs_jax
    def test_hybrid_input_output_jax_with_initial_guess_raises(self):
        """``hybrid_input_output(backend='jax', initial_guess=...)``
        raises ``NotImplementedError``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        ig = (np.ones((N, N), dtype=np.complex128) * 0.1).astype(np.complex128)
        with pytest.raises(NotImplementedError, match=r'init_phase'):
            la.hybrid_input_output(
                meas, sup, n_iter=5, beta=0.9, initial_guess=ig,
                backend='jax')

    @needs_jax
    def test_error_reduction_jax_without_initial_guess_works(self):
        """Sanity: the path still works when ``initial_guess`` is
        not given (the new check only fires when the user passed it
        explicitly)."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        # No initial_guess -> no raise.
        obj, err = la.error_reduction(
            meas, sup, n_iter=5, backend='jax', seed=42)
        assert obj.shape == (N, N)
        assert np.isfinite(err)


# ============================================================================
# L4c -- gerchberg_saxton(backend='jax', return_history=True) shape
# ============================================================================

class TestL4cReturnHistoryShape:
    """v4.13.0 decision: emit ``RuntimeWarning`` and synthesise a
    3-tuple with an empty history list when ``backend='jax'`` is paired
    with ``return_history=True``.  Pre-fix the dispatcher silently
    dropped ``return_history`` so the user expecting a 3-tuple
    received a 2-tuple.
    """

    @needs_jax
    def test_gs_jax_return_history_warns_and_returns_3tuple(self):
        N = 16
        src = np.ones((N, N), dtype=np.float32)
        tgt = np.ones((N, N), dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.gerchberg_saxton(
                src, tgt, n_iter=5, return_history=True, backend='jax')
        # Shape: 3-tuple as the NumPy API contracts.
        assert len(result) == 3, (
            f"Expected 3-tuple (phase, err, history) when "
            f"return_history=True; got len={len(result)}.")
        phase, err, history = result
        assert phase.shape == (N, N)
        assert np.isfinite(err)
        # History is the empty list synthesized by the dispatcher.
        assert history == []
        # Warning text mentions the synthetic / jax limitation.
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        ), (f"Expected a RuntimeWarning about JAX history capture; "
            f"caught {[str(w.message) for w in caught]!r}.")

    @needs_jax
    def test_error_reduction_jax_return_history_warns_and_returns_3tuple(self):
        """L4c sibling: same contract for ``error_reduction``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.error_reduction(
                meas, sup, n_iter=5, return_history=True, backend='jax',
                seed=0)
        assert len(result) == 3
        obj, err, history = result
        assert obj.shape == (N, N)
        assert history == []
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )

    @needs_jax
    def test_hybrid_input_output_jax_return_history_warns_and_returns_3tuple(self):
        """L4c sibling: same contract for ``hybrid_input_output``."""
        N = 16
        meas = np.ones((N, N), dtype=np.float32)
        sup = np.ones((N, N), dtype=bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.hybrid_input_output(
                meas, sup, n_iter=5, beta=0.9, return_history=True,
                backend='jax', seed=0)
        assert len(result) == 3
        obj, err, history = result
        assert obj.shape == (N, N)
        assert history == []
        assert any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )

    @needs_jax
    def test_gs_jax_no_history_still_returns_2tuple(self):
        """Default path (``return_history=False``) still returns a
        2-tuple -- the warning + synthesis only fires when the user
        explicitly asks for history."""
        N = 16
        src = np.ones((N, N), dtype=np.float32)
        tgt = np.ones((N, N), dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.gerchberg_saxton(
                src, tgt, n_iter=5, backend='jax')
        assert len(result) == 2
        assert not any(
            'jax' in str(w.message).lower() and 'history' in str(w.message).lower()
            for w in caught
        )
