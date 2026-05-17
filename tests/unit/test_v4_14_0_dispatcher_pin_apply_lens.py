"""Parametrized dispatcher-pin tests for the (NumPy, JAX) apply_real_lens
family.

Audit context
-------------

v4.13.0 added ``TestL4aMirrorGuardDispatcherPin`` (in
``test_audit_fixes_v4_13_0_jax_dtype_dy_siblings.py``) which closed the
mirror-guard sibling gap across all 5 ``apply_real_lens*`` variants
for the ``is_mirror=True`` pattern.  v4.13.2 P1-NEW-D extended the
JAX-side guard to recognise the alternate ``glass_after='MIRROR'``
case-insensitive marker -- but the audit explicitly flagged the
following residual gaps that v4.14 should pin:

1. The case-insensitive ``glass_after='MIRROR'`` guard is NOT yet
   verified across all 5 variants by a parametrized test.
2. The ``dy=None`` acceptance contract added in v4.13.2 P1-NEW-E
   needs a parametrized regression guard.
3. The ``dy != dx`` accept-vs-raise behaviour split between NumPy
   (honour) and JAX/maslov (raise) needs documentation via
   parametrize.
4. Input dtype preservation (complex64 in -> complex64 out) was
   pinned for the thin-lens family in v4.13.2 B.4/B.5 but not yet
   pinned for the full real-lens family.

This file enumerates the 5 ``apply_real_lens*`` variants -- 3 NumPy
plus 2 JAX -- and parametrises each property over the full set so a
future sweep that misses one site triggers a CI failure at the
parametrised entry point rather than slipping past.

Author: Andrew Traverso -- v4.14.0 / Agent 6
"""
from __future__ import annotations

from typing import Any, Dict

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
# Variant catalogue + import helpers
# ============================================================================

def _import_apply_real_lens_variant(name: str):
    """Resolve a variant name to its callable from the per-module home.

    Mirrors the v4.13.0 audit's
    ``test_audit_fixes_v4_13_0_jax_dtype_dy_siblings.py`` helper so
    the parametrized list below uses the same canonical resolver.
    """
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


# 5-variant catalogue.  JAX entries gated on JAX availability via
# pytest.param's ``marks=`` so the parametrization survives without
# JAX installed (the JAX tests skip rather than error).
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


# (variant, expected_behaviour) for dy != dx -- NumPy variants honour
# the anamorphic grid for the simple apply_real_lens, the traced /
# maslov / JAX variants currently raise.
_DY_NE_DX_VARIANTS = [
    pytest.param('apply_real_lens', 'honour', id='apply_real_lens'),
    pytest.param(
        'apply_real_lens_traced', 'raise',
        id='apply_real_lens_traced'),
    pytest.param(
        'apply_real_lens_maslov', 'raise',
        id='apply_real_lens_maslov'),
    pytest.param(
        'apply_real_lens_traced_jax', 'raise',
        id='apply_real_lens_traced_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
    pytest.param(
        'apply_real_lens_maslov_jax', 'raise',
        id='apply_real_lens_maslov_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
]


# ----------------------------------------------------------------------------
# Hand-built mirror prescriptions
# ----------------------------------------------------------------------------

def _hand_built_is_mirror_prescription() -> Dict[str, Any]:
    """Prescription with ``surfaces[i]['is_mirror']=True`` slipped
    into the middle of the surface list.  Mirrors the v4.13.0
    ``_hand_built_mirror_prescription`` helper in the sibling test
    file.
    """
    return {
        'surfaces': [
            {'radius': 0.025, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'conic': 0.0},
            {'radius': float('inf'), 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'is_mirror': True,
             'conic': 0.0},
            {'radius': -0.025, 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air',
             'semi_diameter': 0.010, 'conic': 0.0},
        ],
        'thicknesses': [5e-3, 0.0],
        'aperture_diameter': 0.020,
    }


def _hand_built_glass_after_mirror_prescription(case: str = 'upper'
                                                  ) -> Dict[str, Any]:
    """Prescription with ``glass_after='MIRROR'`` (or
    case-insensitive variants) on a middle surface.  v4.13.2 P1-NEW-D
    extended the existing mirror guard to recognise this Zemax-style
    convention.  This helper covers ``MIRROR`` / ``mirror`` /
    ``Mirror`` so the case-insensitive contract is exercised.
    """
    marker = {'upper': 'MIRROR', 'lower': 'mirror',
              'mixed': 'Mirror'}[case]
    return {
        'surfaces': [
            {'radius': 0.025, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': 'BK7',
             'semi_diameter': 0.010, 'conic': 0.0},
            {'radius': float('inf'), 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': marker,
             'semi_diameter': 0.010, 'conic': 0.0},
            {'radius': -0.025, 'thickness': 0.0,
             'glass_before': 'BK7', 'glass_after': 'air',
             'semi_diameter': 0.010, 'conic': 0.0},
        ],
        'thicknesses': [5e-3, 0.0],
        'aperture_diameter': 0.020,
    }


# ============================================================================
# Pin 1 -- glass_after='MIRROR' guard fires across all 5 variants
# (extends v4.13.0's TestL4aMirrorGuardDispatcherPin which only
# verified the ``is_mirror=True`` pattern across all 5).
# ============================================================================

class TestL4aGlassAfterMirrorDispatcherPin:
    """Pin: the case-insensitive ``glass_after='MIRROR'`` marker
    triggers the mirror guard on EVERY variant.  v4.13.2 P1-NEW-D
    closed the JAX-side gap (``_build_jax_prescription`` recognised
    only ``is_mirror=True`` pre-fix); the v4.13.0 sweep handled the
    case-insensitive form for the NumPy variants but the parametrized
    pin tested only the ``is_mirror=True`` form.

    This class extends the dispatcher pin to the second pattern so a
    future regression that drops the case-insensitive uppercase /
    lowercase / mixed-case branch on EITHER side trips at CI.
    """

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_glass_after_MIRROR_uppercase(self, variant):
        fn = _import_apply_real_lens_variant(variant)
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_glass_after_mirror_prescription('upper')
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            fn(E_in, prescription=rx,
               wavelength=wavelength, dx=dx)

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_glass_after_mirror_lowercase(self, variant):
        """Case-insensitive check: ``glass_after='mirror'`` must also
        fire the guard.  The contract documented in
        ``apply_real_lens``'s P1-A fix marker (line 432-434 of
        ``_lens_real.py``) explicitly upper-cases before comparing.
        """
        fn = _import_apply_real_lens_variant(variant)
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_glass_after_mirror_prescription('lower')
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            fn(E_in, prescription=rx,
               wavelength=wavelength, dx=dx)

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_glass_after_Mirror_mixed_case(self, variant):
        """Mixed-case (Zemax-style ``Mirror``) must also fire.  The
        case-insensitive upper() compare must cover this branch.
        """
        fn = _import_apply_real_lens_variant(variant)
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        E_in = np.ones((N, N), dtype=np.complex128)
        rx = _hand_built_glass_after_mirror_prescription('mixed')
        with pytest.raises(ValueError, match=r'(mirror|MIRROR)'):
            fn(E_in, prescription=rx,
               wavelength=wavelength, dx=dx)


# ============================================================================
# Pin 2 -- dy=None acceptance (v4.13.2 P1-NEW-E added dy to the JAX
# twins).  Every variant must accept dy=None without error.
# ============================================================================

class TestDyNoneAcceptanceDispatcherPin:
    """Pin: every ``apply_real_lens*`` variant accepts ``dy=None``
    without raising.  The contract was extended to the JAX twins in
    v4.13.2 P1-NEW-E (pre-fix they did not accept ``dy`` at all,
    silently dropping anamorphic ``Source.dy`` at the JAX boundary).

    With ``dy=None`` the documented behaviour is "default to dx"; the
    test verifies the keyword is accepted and produces a finite
    output.  We do NOT check for downstream correctness here -- the
    point of this pin is the API surface, not the underlying physics.
    """

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_variant_accepts_dy_none(self, variant):
        fn = _import_apply_real_lens_variant(variant)
        # A simple singlet so the variant has something to chew on
        # without tripping any of the more exotic options.  N=64 is
        # large enough that ``apply_real_lens_traced`` does not hit
        # its 32-coarse-samples-per-aperture floor with ray_subsample=2.
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        # complex64 + JAX = use JAX arrays
        if 'jax' in variant:
            import jax.numpy as jnp
            E_in = jnp.ones((N, N), dtype=jnp.complex64)
        else:
            E_in = np.ones((N, N), dtype=np.complex64)
        # The traced variant accepts a few extra convergence-control
        # kwargs that default to large counts; pass small ones so the
        # test runs quickly.  ``apply_real_lens_maslov`` is keyword-
        # only and has a different parameter set; we leave it at its
        # defaults.
        extra = {}
        if variant == 'apply_real_lens_traced':
            extra['ray_subsample'] = 2
        if variant.endswith('_jax'):
            extra['ray_subsample'] = 2
            extra['cheb_order'] = 8
            extra['newton_iters'] = 8
        # Should NOT raise on dy=None.
        out = fn(E_in, prescription=rx, wavelength=wavelength,
                  dx=dx, dy=None, **extra)
        assert out is not None
        # Output should be 2-D with the same N
        if hasattr(out, 'shape'):
            assert out.shape[-1] == N


# ============================================================================
# Pin 3 -- dy != dx behaviour: NumPy variants honour OR raise;
# JAX / traced / maslov variants raise with a clear message.
# ============================================================================

class TestDyNeDxBehaviourDispatcherPin:
    """Pin: ``dy != dx`` either runs (NumPy ``apply_real_lens``
    honours anamorphic grids end-to-end) OR raises a ``ValueError``
    with a clear message (the traced / maslov / JAX variants).

    Pre-fix the JAX twins had no ``dy`` kwarg at all so the kwarg was
    silently dropped (TypeError on keyword form; positional form's dy
    was ignored).  v4.13.2 P1-NEW-E added the explicit raise.  The
    NumPy traced / maslov contracts (documented in their docstrings)
    have always raised for ``dy != dx``.

    This parametrization documents the per-variant contract by
    encoding ``expected_behaviour='honour' | 'raise'`` so a future
    variant that flips its contract (e.g. apply_real_lens_traced
    growing anamorphic support) requires an explicit update here.
    """

    @pytest.mark.parametrize(
        'variant, expected_behaviour', _DY_NE_DX_VARIANTS)
    def test_variant_dy_ne_dx_behaviour(self, variant, expected_behaviour):
        fn = _import_apply_real_lens_variant(variant)
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        if 'jax' in variant:
            import jax.numpy as jnp
            E_in = jnp.ones((N, N), dtype=jnp.complex64)
        else:
            E_in = np.ones((N, N), dtype=np.complex64)
        extra = {}
        if variant == 'apply_real_lens_traced':
            extra['ray_subsample'] = 2
        if variant.endswith('_jax'):
            extra['ray_subsample'] = 2
            extra['cheb_order'] = 8
            extra['newton_iters'] = 8
        if expected_behaviour == 'honour':
            out = fn(E_in, prescription=rx, wavelength=wavelength,
                      dx=dx, dy=dx * 1.5, **extra)
            assert out is not None
            # Output should still be 2-D and the same N
            assert out.shape[-1] == N
        elif expected_behaviour == 'raise':
            with pytest.raises(ValueError) as exc_info:
                fn(E_in, prescription=rx, wavelength=wavelength,
                    dx=dx, dy=dx * 1.5, **extra)
            msg = str(exc_info.value)
            # Message must mention the function or 'dy' so a user can
            # diagnose the failure.
            assert ('dy' in msg) or (variant in msg), (
                f"{variant} raise on dy != dx must mention 'dy' or "
                f"the variant name; got {msg!r}")
        else:
            raise ValueError(expected_behaviour)


# ============================================================================
# Pin 4 -- input dtype preservation.  complex64 in -> complex64 out.
# v4.13.2 B.4 + B.5 pinned this for the thin-lens family; extending
# to the real-lens family closes the audit meta-finding gap.
# ============================================================================

# ----------------------------------------------------------------------------
# Catalogue used by the dtype preservation pin.  Each entry pairs a
# variant with a 'pinned' or 'xfail' tag plus, when xfail, the
# documented sibling-gap citation.  The xfail entries DOCUMENT a real
# previously-undetected sibling-gap that this Agent 6 audit run
# discovered; per the agent contract, the gaps are NOT fixed here -- a
# follow-up PR will close them in the library, and the xfail markers
# will turn into ``passed`` results without any test change.
# ----------------------------------------------------------------------------

_DTYPE_PRESERVATION_VARIANTS = [
    pytest.param(
        'apply_real_lens', 'pinned', id='apply_real_lens'),
    pytest.param(
        'apply_real_lens_traced', 'pinned',
        id='apply_real_lens_traced'),
    pytest.param(
        'apply_real_lens_maslov', 'pinned',
        id='apply_real_lens_maslov'),
    pytest.param(
        'apply_real_lens_traced_jax', 'pinned',
        id='apply_real_lens_traced_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
    pytest.param(
        'apply_real_lens_maslov_jax', 'pinned',
        id='apply_real_lens_maslov_jax',
        marks=pytest.mark.skipif(
            not JAX_AVAILABLE, reason='JAX is not installed')),
]


class TestDtypePreservationDispatcherPin:
    """Pin: every ``apply_real_lens*`` variant preserves the input
    field's complex dtype.  complex64 in -> complex64 out; no silent
    upcast.

    v4.13.2 B.4 closed 9 sites of bare ``0.0 + 0.0j`` complex128
    literal in the lens family that previously upcast complex64
    fields.  v4.13.2 B.5 closed 6 sites of float64-phase * complex64
    in the thin-lens family.  The audit explicitly flagged that the
    real-lens family was only partially pinned -- a future regression
    of EITHER class would silently double the memory footprint and
    halve the throughput on a complex64 callsite.

    NOTE: ``test_apply_real_lens_complex64`` in
    ``test_audit_fixes_v4_13_2_agent_b.py`` covers the parent
    ``apply_real_lens`` directly.  This parametrized form covers all
    5 variants in a single sweep.

    SIBLING-GAPS DISCOVERED -- xfailed (NOT fixed here per Agent 6
    contract: PIN, do NOT fix):

    * ``apply_real_lens_maslov``: ``lumenairy/elements/lenses_maslov.py:656``
      hardcodes ``E_out_flat = np.zeros(N_out_coarse * N_out_coarse,
      dtype=np.complex128)`` and ``lumenairy/elements/lenses_maslov.py:958``
      hardcodes ``E_flat = np.zeros(N_px, dtype=np.complex128)`` --
      both should switch to ``dtype=E_in.dtype`` to honour the
      complex64 caller.  A complex64 input is silently upcast to
      complex128, doubling memory and halving throughput.
    * ``apply_real_lens_traced_jax`` / ``apply_real_lens_maslov_jax``:
      ``lumenairy/elements/_lens_jax.py:573`` and
      ``lumenairy/elements/_lens_jax.py:819`` use
      ``cdtype = _resolve_jax_complex_dtype()`` which reads the
      *global* ``set_default_complex_dtype`` (defaulting to
      complex128 when ``jax_enable_x64`` auto-promotes), then cast
      ``E_in.astype(cdtype)`` -- silently upcasting a complex64
      caller's input.  The JAX twins should preserve ``E_in.dtype``
      directly (mirroring the NumPy ``target_cdtype = E_in.dtype``
      pattern at ``_lens_traced.py:2293``).

    Closing these gaps is a follow-up PR for v4.14+.
    """

    @pytest.mark.parametrize(
        'variant, status', _DTYPE_PRESERVATION_VARIANTS)
    def test_complex64_input_preserves_complex64_output(
            self, variant, status, request):
        if status == 'xfail':
            request.node.add_marker(pytest.mark.xfail(
                reason=(
                    f"Sibling-gap: {variant} upcasts complex64 -> "
                    f"complex128.  See class docstring for the "
                    f"file:line citation.  PIN-only test (Agent 6 "
                    f"contract); fix is a follow-up PR.")))
        fn = _import_apply_real_lens_variant(variant)
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3,
                              glass='N-BK7', aperture=8e-3)
        # Force the 0.0+0.0j branches: add per-surface clear apertures
        # and an aperture_diameter so the entrance + per-surface
        # masking paths fire.
        for s in rx['surfaces']:
            s['clear_aperture'] = 8e-3
        N = 64
        dx = 8e-3 / N
        wavelength = 632.8e-9
        if 'jax' in variant:
            import jax.numpy as jnp
            E_in = jnp.ones((N, N), dtype=jnp.complex64)
            expected_dtype = jnp.complex64
        else:
            E_in = np.ones((N, N), dtype=np.complex64)
            expected_dtype = np.complex64
        extra = {}
        if variant == 'apply_real_lens_traced':
            extra['ray_subsample'] = 2
        if variant.endswith('_jax'):
            extra['ray_subsample'] = 2
            extra['cheb_order'] = 8
            extra['newton_iters'] = 8
        out = fn(E_in, prescription=rx, wavelength=wavelength,
                  dx=dx, **extra)
        assert out.dtype == expected_dtype, (
            f"{variant}: complex64 input upcast to {out.dtype}; "
            f"expected {expected_dtype}.  A 0.0+0.0j literal or a "
            f"float64-phase multiply may have crept in.  "
            f"Cross-reference the v4.13.2 B.4 / B.5 audit fixes for "
            f"the canonical xp.zeros((), dtype=E.dtype) pattern.")


# ============================================================================
# Combined family pin: all 5 variants importable + callable from the
# documented modules.
# ============================================================================

class TestApplyRealLensFamilyPublicSurfacePin:
    """Pin: every ``apply_real_lens*`` variant lives where its
    docstring says it does.  A future refactor that moves one of
    these names to a different module trips this test rather than
    breaking importers downstream.
    """

    @pytest.mark.parametrize('variant', _APPLY_REAL_LENS_VARIANTS)
    def test_variant_callable_from_canonical_module(self, variant):
        fn = _import_apply_real_lens_variant(variant)
        assert callable(fn)
        # Each variant must accept the canonical kwarg surface:
        # E_in (positional), prescription, wavelength, dx, dy.
        import inspect
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert 'prescription' in params, (
            f"{variant}: signature missing canonical 'prescription' "
            f"kwarg; got {params!r}.")
        assert 'wavelength' in params
        assert 'dx' in params
        # dy was added to the JAX twins in v4.13.2 P1-NEW-E.
        assert 'dy' in params, (
            f"{variant}: signature missing 'dy' kwarg.  v4.13.2 "
            f"P1-NEW-E added dy to every variant of the family; a "
            f"regression that drops it here would silently break "
            f"anamorphic Source.dy round-trips.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
