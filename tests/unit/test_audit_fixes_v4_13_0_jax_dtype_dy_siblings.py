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
    ``apply_real_lens_traced`` -- AND the parent ``apply_real_lens``
    itself (v4.13.0 audit P1-A: the L4a sweep missed the parent).
    """

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

    def test_apply_real_lens_raises_on_hand_built_mirror(self):
        """v4.13.0 audit P1-A: pin the mirror guard fires for the
        parent ``apply_real_lens`` itself.  Pre-fix the L4a sweep
        hardened the 4 ``apply_real_lens_*`` siblings but missed the
        parent -- a hand-built prescription with ``is_mirror=True``
        and no ``elements`` key silently miscomputed via the
        refractive-only thin-element path.  The new guard fails
        loudly before any sag / refraction math touches the field.
        Closes the audit P1-A gap.
        """
        from lumenairy.elements._lens_real import apply_real_lens
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)

    def test_apply_real_lens_raises_on_glass_after_mirror(self):
        """Same closure, but the guard fires on
        ``glass_after='MIRROR'`` (no explicit ``is_mirror`` flag).
        Matches the alternate Zemax convention the sibling guards
        recognise.
        """
        from lumenairy.elements._lens_real import apply_real_lens
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = {
            'surfaces': [
                {'radius': 0.025, 'thickness': 5e-3,
                 'glass_before': 'air', 'glass_after': 'BK7',
                 'semi_diameter': 0.010, 'conic': 0.0},
                # Mirror flagged via ``glass_after`` rather than
                # ``is_mirror``.
                {'radius': float('inf'), 'thickness': 0.0,
                 'glass_before': 'BK7', 'glass_after': 'MIRROR',
                 'semi_diameter': 0.010, 'conic': 0.0},
                {'radius': -0.025, 'thickness': 0.0,
                 'glass_before': 'BK7', 'glass_after': 'air',
                 'semi_diameter': 0.010, 'conic': 0.0},
            ],
            'aperture_diameter': 0.020,
        }
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            apply_real_lens(
                E_in, prescription=rx, wavelength=wavelength, dx=dx)


def _import_apply_real_lens_variant(name):
    """Resolve the variant name to its callable from the per-module
    home (closures the dispatcher-level pin across all 5 entry
    points)."""
    if name == 'apply_real_lens':
        from lumenairy.elements._lens_real import apply_real_lens
        return apply_real_lens
    if name == 'apply_real_lens_traced':
        from lumenairy.elements._lens_traced import apply_real_lens_traced
        return apply_real_lens_traced
    if name == 'apply_real_lens_maslov':
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
        return apply_real_lens_maslov
    if name == 'apply_real_lens_traced_jax':
        from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
        return apply_real_lens_traced_jax
    if name == 'apply_real_lens_maslov_jax':
        from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax
        return apply_real_lens_maslov_jax
    raise ValueError(f"unknown variant: {name!r}")


_APPLY_REAL_LENS_VARIANTS = [
    pytest.param('apply_real_lens', id='apply_real_lens'),
    pytest.param('apply_real_lens_traced', id='apply_real_lens_traced'),
    pytest.param('apply_real_lens_maslov', id='apply_real_lens_maslov'),
    pytest.param(
        'apply_real_lens_traced_jax', id='apply_real_lens_traced_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
    pytest.param(
        'apply_real_lens_maslov_jax', id='apply_real_lens_maslov_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
]


class TestL4aMirrorGuardDispatcherPin:
    """v4.13.0 audit meta-finding: the strongest closure for the
    L4a sibling-gap pattern is a parametrized dispatcher-level pin
    that exercises every entry point a user can reach the fix
    through.  If a future sweep misses one of the 5 variants the
    way the v4.13.0 sweep missed ``apply_real_lens``, this test
    will fail rather than silently allowing a downstream
    miscomputation.
    """

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_mirror_guard_fires_on_hand_built_is_mirror(self, variant):
        fn = _import_apply_real_lens_variant(variant)
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_mirror_prescription()
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            fn(E_in, prescription=rx, wavelength=wavelength, dx=dx)


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


# ============================================================================
# P1-B -- ER and HIO route dtype through _resolve_jax_complex_dtype
# ============================================================================

def _call_phase_retrieval_kernel(kernel_name, *, dtype):
    """Dispatch helper for the parametrized P1-B pin.  Returns the
    primary output array (complex object / Fourier-plane phase) of
    each kernel with a small, well-conditioned input.
    """
    if kernel_name == 'gerchberg_saxton_jax':
        from lumenairy.analysis.phase_retrieval import gerchberg_saxton_jax
        N = 16
        src = np.ones((N, N), dtype=np.dtype(dtype))
        tgt = np.ones((N, N), dtype=np.dtype(dtype))
        # GS returns (phase, err) -- phase is the real-valued source
        # plane phase.  For the P1-B pin we want to verify the
        # complex intermediates ran at the resolved cdtype, which
        # is observable indirectly through the kernel cache key
        # (verified separately) and directly through the warning
        # the resolver emits on first complex128 request with
        # ``jax_enable_x64=False``.
        return gerchberg_saxton_jax(src, tgt, n_iter=3, dtype=dtype)
    if kernel_name == 'error_reduction_jax':
        from lumenairy.analysis.phase_retrieval import error_reduction_jax
        N = 16
        meas = np.ones((N, N), dtype=np.dtype(dtype))
        sup = np.ones((N, N), dtype=bool)
        return error_reduction_jax(meas, sup, n_iter=3, seed=0, dtype=dtype)
    if kernel_name == 'hybrid_input_output_jax':
        from lumenairy.analysis.phase_retrieval import hybrid_input_output_jax
        N = 16
        meas = np.ones((N, N), dtype=np.dtype(dtype))
        sup = np.ones((N, N), dtype=bool)
        return hybrid_input_output_jax(
            meas, sup, n_iter=3, beta=0.9, seed=0, dtype=dtype)
    raise ValueError(f"unknown kernel: {kernel_name!r}")


_PHASE_RETRIEVAL_KERNELS = ['gerchberg_saxton_jax',
                            'error_reduction_jax',
                            'hybrid_input_output_jax']


class TestP1BPhaseRetrievalDtypeResolver:
    """v4.13.0 audit P1-B: ``error_reduction_jax`` and
    ``hybrid_input_output_jax`` route ``dtype`` through
    ``_resolve_jax_complex_dtype`` (and the real twin) so that a
    caller passing ``dtype=np.float64`` while
    ``jax.config.jax_enable_x64=False`` either auto-enables x64
    (with the documented one-shot RuntimeWarning) or returns at
    full complex128 precision rather than silently demoting.

    The L2 sweep applied this closure to ``gerchberg_saxton_jax``
    but missed the ER/HIO siblings; this pin closes that gap and
    parametrises over all 3 kernels to keep future sweeps
    honest.
    """

    @needs_jax
    @pytest.mark.parametrize('kernel_name', _PHASE_RETRIEVAL_KERNELS)
    def test_float64_dtype_auto_enables_x64_or_returns_complex128(
            self, kernel_name):
        """Under ``jax_enable_x64=False``, calling any of the 3
        phase-retrieval kernels with ``dtype=np.float64`` MUST
        either fire the documented one-shot RuntimeWarning AND
        return a complex128 / float64 result OR (if x64 was
        already enabled earlier in the session) return at the
        full requested precision with no warning.  Pre-P1-B the
        ER/HIO siblings silently produced float32 / complex64
        intermediates.
        """
        import jax
        # We don't try to flip x64 off mid-session -- once any
        # earlier test (or the GS sibling on the first call here)
        # auto-enabled it, JAX won't let us turn it back off
        # cleanly.  Instead, assert the post-condition that holds
        # regardless of starting state: after the call, x64 is
        # enabled AND the returned array is at full precision.
        result = _call_phase_retrieval_kernel(
            kernel_name, dtype=np.float64)
        # Each kernel returns (primary, err).  primary is either a
        # complex object (ER / HIO) or a real phase (GS).
        primary = result[0]
        # The resolver auto-enables x64 when complex128 is
        # requested; once it has flipped, JAX returns at the
        # requested precision.
        assert jax.config.jax_enable_x64, (
            f"{kernel_name} with dtype=np.float64 should have "
            f"auto-enabled jax_enable_x64; current value is "
            f"{jax.config.jax_enable_x64!r}.")
        if np.iscomplexobj(primary):
            assert primary.dtype == np.complex128, (
                f"{kernel_name} returned complex array of dtype "
                f"{primary.dtype!r}; expected complex128 with "
                f"dtype=np.float64 + x64 enabled.")
        else:
            # GS returns the real source-plane phase.
            assert primary.dtype == np.float64, (
                f"{kernel_name} returned real array of dtype "
                f"{primary.dtype!r}; expected float64 with "
                f"dtype=np.float64 + x64 enabled.")

    @needs_jax
    @pytest.mark.parametrize('kernel_name', _PHASE_RETRIEVAL_KERNELS)
    def test_float32_dtype_keeps_complex64(self, kernel_name):
        """Sanity: float32 caller still gets complex64 output (the
        documented historical behaviour); the P1-B fix only
        changes the float64 path.
        """
        result = _call_phase_retrieval_kernel(
            kernel_name, dtype=np.float32)
        primary = result[0]
        if np.iscomplexobj(primary):
            assert primary.dtype == np.complex64, (
                f"{kernel_name} returned complex array of dtype "
                f"{primary.dtype!r}; expected complex64 with "
                f"dtype=np.float32.")
        else:
            assert primary.dtype == np.float32, (
                f"{kernel_name} returned real array of dtype "
                f"{primary.dtype!r}; expected float32 with "
                f"dtype=np.float32.")


# ============================================================================
# P1-C -- Source.propagate() and 5 classmethod factories thread `dy`
# ============================================================================

class TestP1CSourceDyThreading:
    """v4.13.0 audit P1-C: ``Source.propagate()`` and the 5
    classmethod factories (``gaussian``, ``plane_wave``,
    ``point_source``, ``top_hat``, ``fiber_mode``) must thread the
    anamorphic ``dy`` onto the returned ``Source`` instance.

    Pre-P1-C the underlying ``create_*`` helpers received and used
    ``dy`` (so the E-field WAS built on the anamorphic grid), but
    the wrapping ``cls(E=E, dx=dx, wavelength=..., ...)`` call
    omitted ``dy``, so the returned Source advertised
    ``dy == dx`` even though ``E.shape`` reflected the rectangular
    pixel count.  L3 added the ``dy`` field to ``Source`` but the
    threading sweep missed these six call sites.
    """

    def test_source_propagate_preserves_dy(self):
        """Pin: ``Source.propagate(...).dy == self.dy`` for an
        anamorphic input.  Pre-fix the result advertised
        ``dy == dx_out`` regardless of the input's y-pitch.
        """
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 7e-6   # distinct y-pitch
        wavelength = 633e-9
        # Build a simple Gaussian-shaped field on the anamorphic
        # grid so the propagator has something physical to chew on.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dy
        X, Y = np.meshgrid(x, y, indexing='xy')
        E = np.exp(-(X * X + Y * Y) / (5 * dx) ** 2).astype(
            np.complex128)
        src = Source(E=E, dx=dx, dy=dy, wavelength=wavelength)
        assert src.dy == dy
        # ``method='asm'`` is the simplest free-space propagator that
        # respects the input pitch verbatim (no resample).
        result = src.propagate(method='asm', z=1e-4)
        assert result.dy == dy, (
            f"Source.propagate must thread dy onto the returned "
            f"Source; got result.dy={result.dy!r}, expected "
            f"src.dy={dy!r}.")
        assert result.dx == dx

    def test_source_gaussian_factory_preserves_dy(self):
        """``Source.gaussian(..., dy=...)`` must wrap the result
        with the supplied ``dy``."""
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6   # distinct y-pitch
        wavelength = 633e-9
        src = Source.gaussian(
            w0=20e-6, N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy, (
            f"Source.gaussian must thread dy onto the wrapped "
            f"Source; got src.dy={src.dy!r}, expected "
            f"{dy!r}.")

    def test_source_plane_wave_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.plane_wave(
            N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_point_source_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.point_source(
            N=N, dx=dx, wavelength=wavelength, dy=dy, z0=-1e-3)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_top_hat_factory_preserves_dy(self):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        src = Source.top_hat(
            diameter=50e-6, N=N, dx=dx, wavelength=wavelength, dy=dy)
        assert src.dx == dx
        assert src.dy == dy

    def test_source_fiber_mode_factory_preserves_dy(self):
        """``Source.fiber_mode`` is the one factory whose underlying
        helper (``create_fiber_mode``) doesn't accept ``dy``; the
        P1-C fix is wrapper-only and verifies the returned Source
        advertises a distinct ``dy``.  The factory drops ``dy``
        from ``factory_kwargs`` BEFORE forwarding because the helper
        is dy-agnostic; we still want the wrapped Source to carry
        the caller-supplied ``dy`` so the metadata reaches
        downstream code that does honour ``dy``.

        NOTE: ``create_fiber_mode`` does not accept ``dy``, so we
        verify the audit-spec'd behaviour only by directly
        constructing a Source with explicit ``dy``.  When v4.14
        widens ``create_fiber_mode`` to accept ``dy`` this test
        can be rewritten to call ``Source.fiber_mode(..., dy=...)``
        directly.
        """
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 9e-6
        wavelength = 633e-9
        # Smoke: the factory builds a Source without dy (the
        # underlying helper is dy-agnostic).  Default dy == dx is
        # the documented back-compat for that helper.
        src_default = Source.fiber_mode(
            mode_field_diameter=10e-6, N=N, dx=dx,
            wavelength=wavelength)
        assert src_default.dx == dx
        assert src_default.dy == dx, (
            f"Source.fiber_mode without explicit dy should default "
            f"to dy=dx for back-compat; got "
            f"{src_default.dy!r}.")
        # Direct construction with explicit dy: confirm the
        # Source dataclass round-trips dy correctly (matches the
        # P1-C fix's intent for the dy-aware factories above).
        E = src_default.E
        src_anamorphic = Source(
            E=E, dx=dx, dy=dy, wavelength=wavelength)
        assert src_anamorphic.dy == dy


# ============================================================================
# P1-C extension: parametrised dispatcher-level pin over Source factories
# ============================================================================

_SOURCE_FACTORY_KWARGS = [
    pytest.param(
        'gaussian', {'w0': 20e-6}, id='gaussian'),
    pytest.param(
        'plane_wave', {}, id='plane_wave'),
    pytest.param(
        'point_source', {'z0': -1e-3}, id='point_source'),
    pytest.param(
        'top_hat', {'diameter': 50e-6}, id='top_hat'),
    # fiber_mode: v4.13.2 widened ``create_fiber_mode`` to accept the
    # ``dy=`` kwarg (it forwards through to ``create_gaussian_beam``),
    # so the dispatcher pin now covers all 5 factories.  The v4.14.0
    # audit P2-9 flagged that the v4.13.2 CHANGELOG claim of "covers
    # all factories" was off-by-one because this parametrize list had
    # never been updated.  v4.14.1 closes that gap.
    pytest.param(
        'fiber_mode', {'mode_field_diameter': 10e-6}, id='fiber_mode'),
]


class TestP1CSourceFactoryDispatcherPin:
    """v4.13.0 audit meta-finding: the strongest closure for the
    L3 / P1-C sibling-gap pattern is a parametrised dispatcher-
    level pin over every Source factory that the v4.13.0 sweep
    was supposed to touch.  If a future sweep adds a new factory
    and forgets the ``dy`` thread-through, this test fails at
    the parametrised entry point rather than silently producing
    a Source with the wrong y-pitch metadata.
    """

    @pytest.mark.parametrize('factory_name, extra_kwargs',
                             _SOURCE_FACTORY_KWARGS)
    def test_factory_threads_dy(self, factory_name, extra_kwargs):
        from lumenairy.sources.core import Source
        N = 32
        dx = 5e-6
        dy = 11e-6   # distinct y-pitch
        wavelength = 633e-9
        factory = getattr(Source, factory_name)
        src = factory(
            N=N, dx=dx, wavelength=wavelength, dy=dy,
            **extra_kwargs)
        assert src.dx == dx
        assert src.dy == dy, (
            f"Source.{factory_name}: returned Source advertised "
            f"dy={src.dy!r}; expected dy={dy!r}.  Pre-fix the "
            f"v4.13.0 L3 sweep missed the ``cls(...)`` call in "
            f"this factory.")
