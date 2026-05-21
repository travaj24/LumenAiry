"""v5.1.0 Agent C -- regression tests for the
``lumenairy.propagators.propagation`` split.

The v5.1.0 release split the 4103-line
``lumenairy/propagators/propagation.py`` into six submodules:

* ``lumenairy/propagators/fft_infra.py`` -- backend dispatch + plan
                                            cache + ASM transfer-
                                            function caches +
                                            DEFAULT_* knobs +
                                            setters / getters +
                                            ``_validate_propagator_inputs``.
* ``lumenairy/propagators/asm.py``       -- ASM, tilted ASM, batched
                                            ASM, ``apply_fresnel_curvature``,
                                            ``_build_asm_H_square``.
* ``lumenairy/propagators/fresnel.py``   -- single-FFT Fresnel +
                                            Fraunhofer.
* ``lumenairy/propagators/rs.py``        -- Rayleigh-Sommerfeld.
* ``lumenairy/propagators/sas.py``       -- Scalable Angular Spectrum.
* ``lumenairy/propagators/mft.py``       -- MFT / Bluestein
                                            counterparts +
                                            ``resample_field``.

Public-API contract: every name that pre-v5.1.0 was importable from
``lumenairy.propagators.propagation`` (or via ``lumenairy``) must
continue to resolve identically.  This module is the anti-regression
pin for that promise.

These tests are pure import-shape / object-identity checks; they
intentionally don't exercise the numerics.  The propagator numeric
behaviour is covered by ``test_propagation*``, ``test_audit_fixes_v4_*``,
etc.; if any of those start failing after the v5.1.0 split, the
split isn't bit-for-bit and the release must be held back.
"""

from __future__ import annotations

import numpy as np
import pytest

# Names that pre-v5.1.0 ``propagation.py`` exported and that callers
# (in-tree + tests + downstream) import via
# ``from lumenairy.propagators.propagation import <name>``.
#
# This list is the union of:
#   1. every public name in propagation.py's pre-split module body,
#   2. every private helper used by another in-tree module
#      (see Sibling-gap discipline grep sweep below).
#
# Failures of ``test_name_resolves_from_propagation`` are the canonical
# split-regression signal -- the split is not bit-for-bit if any of
# these stops resolving.

_PUBLIC_NAMES = [
    # ASM family
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    'angular_spectrum_propagate_batch',
    'angular_spectrum_propagate_mft',
    'apply_fresnel_curvature',
    # Fresnel / Fraunhofer (natural FFT grid)
    'fresnel_propagate',
    'fraunhofer_propagate',
    # MFT counterparts
    'fresnel_propagate_mft',
    'fraunhofer_propagate_mft',
    # Rayleigh-Sommerfeld
    'rayleigh_sommerfeld_propagate',
    # Scalable Angular Spectrum
    'scalable_angular_spectrum_propagate',
    # Grid bridge
    'resample_field',
    # Default-config knobs (DEFAULT_* + setters / getters)
    'DEFAULT_COMPLEX_DTYPE',
    'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_DY',
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'set_default_real_dtype',
    'get_default_real_dtype',
    'set_default_wave_propagator',
    'get_default_wave_propagator',
    'set_default_dy',
    'get_default_dy',
    # FFT backend
    'CUPY_AVAILABLE',
    'PYFFTW_AVAILABLE',
    'SCIPY_FFT_AVAILABLE',
    'FFTW_THREADS',
    'USE_PYFFTW',
    'USE_SCIPY_FFT',
    'SCIPY_FFT_WORKERS',
    'FFTW_MIN_SIZE',
    'PYFFTW_FALLBACK_ON_ERROR',
    'set_fft_threads',
    'get_fft_threads',
    'set_fft_fallback',
    'set_pyfftw_planner',
    'get_pyfftw_planner',
    'set_fft_plan_cache_size',
    'warmup_fft_plans',
    'set_fft_auto_promote',
    'get_fft_auto_promote',
    'reset_fft_backend',
    # ASM cache control
    'set_asm_cache_size',
    'get_asm_cache_size',
    'clear_asm_caches',
]


_PRIVATE_NAMES = [
    # Private FFT helpers consumed by sibling modules
    '_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
    '_handle_pyfftw_failure',
    '_scipy_or_numpy_fft2', '_scipy_or_numpy_ifft2',
    # Validation helper
    '_validate_propagator_inputs',
    # JAX dtype resolvers
    '_resolve_jax_complex_dtype',
    '_resolve_jax_real_dtype',
    # Cached ASM helpers
    '_build_asm_H_square',
    '_get_or_make_freq_grids',
    '_get_or_make_bandlimit',
    '_h_cache_lookup',
    '_h_cache_store',
    '_entry_bytes',
    # Cache containers + locks
    '_H_CACHE',
    '_FREQ_GRID_CACHE',
    '_BANDLIMIT_CACHE',
    '_ASM_CACHE_LOCK',
    '_PYFFTW_PLAN_CACHE',
    '_PYFFTW_PLAN_LOCK',
    '_PYFFTW_BAD_SHAPES',
    '_get_or_make_plan',
    '_build_plan_entry',
    '_promote_entry_to_measure',
    # Local cache clearer (registered as 'asm_local')
    '_clear_local_asm_caches',
    # Pre-v5.1.0 latches preserved for back-compat introspection
    '_DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED',
    '_DEFAULT_DY_NO_CONSUMER_WARNED',
]


class TestSplitPublicSurfaceFromPropagation:
    """Every legacy public name still imports from
    ``lumenairy.propagators.propagation``."""

    @pytest.mark.parametrize('name', _PUBLIC_NAMES)
    def test_name_resolves_from_propagation(self, name):
        """``from lumenairy.propagators.propagation import <name>`` succeeds."""
        import lumenairy.propagators.propagation as prop
        assert hasattr(prop, name), (
            f"v5.1.0 split regression: {name!r} is no longer "
            f"importable from lumenairy.propagators.propagation")

    @pytest.mark.parametrize('name', _PRIVATE_NAMES)
    def test_private_name_resolves_from_propagation(self, name):
        """Private helpers used by sibling modules must still resolve
        via the legacy module path."""
        import lumenairy.propagators.propagation as prop
        assert hasattr(prop, name), (
            f"v5.1.0 split regression: private name {name!r} is no "
            f"longer importable from lumenairy.propagators.propagation; "
            f"a sibling module (e.g. analysis.detector, analysis."
            f"phase_retrieval) depends on this private path.")

    def test_top_level_lumenairy_namespace(self):
        """The public propagator names re-exported via
        ``lumenairy.__init__`` continue to resolve at the top level."""
        import lumenairy as la
        for name in (
            'angular_spectrum_propagate',
            'angular_spectrum_propagate_tilted',
            'scalable_angular_spectrum_propagate',
            'fresnel_propagate',
            'fraunhofer_propagate',
            'rayleigh_sommerfeld_propagate',
            'resample_field',
            'apply_fresnel_curvature',
            'fresnel_propagate_mft',
            'fraunhofer_propagate_mft',
            'angular_spectrum_propagate_mft',
            'set_default_complex_dtype',
            'get_default_complex_dtype',
            'DEFAULT_COMPLEX_DTYPE',
            'DEFAULT_REAL_DTYPE',
            'DEFAULT_WAVE_PROPAGATOR',
            'DEFAULT_DY',
            'clear_asm_caches',
            'warmup_fft_plans',
            'PYFFTW_AVAILABLE',
            'CUPY_AVAILABLE',
        ):
            assert hasattr(la, name), (
                f"v5.1.0 split regression: la.{name} no longer resolves")


class TestSplitSubmoduleResidency:
    """Each public name resides in the expected new submodule.  This
    pins the topology of the split so a later refactor doesn't
    silently shuffle ownership."""

    def test_fft_infra_owns_dispatch_caches_and_setters(self):
        from lumenairy.propagators import fft_infra
        for n in (
            # FFT dispatchers
            '_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd',
            '_scipy_or_numpy_fft2', '_scipy_or_numpy_ifft2',
            '_handle_pyfftw_failure',
            # Caches
            '_H_CACHE', '_FREQ_GRID_CACHE', '_BANDLIMIT_CACHE',
            '_ASM_CACHE_LOCK', '_PYFFTW_PLAN_CACHE',
            '_PYFFTW_BAD_SHAPES',
            # Cache helpers
            '_h_cache_lookup', '_h_cache_store', '_entry_bytes',
            '_get_or_make_freq_grids', '_get_or_make_bandlimit',
            '_clear_local_asm_caches', 'clear_asm_caches',
            'set_asm_cache_size', 'get_asm_cache_size',
            # pyFFTW plan
            '_get_or_make_plan', '_build_plan_entry',
            '_promote_entry_to_measure',
            # FFT config + setters / getters
            'FFTW_THREADS', 'USE_PYFFTW', 'set_fft_threads',
            'get_fft_threads', 'set_fft_fallback',
            'set_pyfftw_planner', 'get_pyfftw_planner',
            'set_fft_plan_cache_size', 'warmup_fft_plans',
            'set_fft_auto_promote', 'get_fft_auto_promote',
            'reset_fft_backend',
            # DEFAULT_* + their setters / getters
            'DEFAULT_COMPLEX_DTYPE', 'DEFAULT_REAL_DTYPE',
            'DEFAULT_WAVE_PROPAGATOR', 'DEFAULT_DY',
            'set_default_complex_dtype', 'get_default_complex_dtype',
            'set_default_real_dtype', 'get_default_real_dtype',
            'set_default_wave_propagator', 'get_default_wave_propagator',
            'set_default_dy', 'get_default_dy',
            # Validation + JAX dtype resolvers
            '_validate_propagator_inputs',
            '_resolve_jax_complex_dtype', '_resolve_jax_real_dtype',
            # Backend availability flags
            'CUPY_AVAILABLE', 'PYFFTW_AVAILABLE', 'SCIPY_FFT_AVAILABLE',
        ):
            assert hasattr(fft_infra, n), \
                f"{n!r} should live in propagators.fft_infra"

    def test_asm_module_owns_asm_kernels(self):
        from lumenairy.propagators import asm
        for n in ('angular_spectrum_propagate',
                  'angular_spectrum_propagate_tilted',
                  'angular_spectrum_propagate_batch',
                  'apply_fresnel_curvature',
                  '_build_asm_H_square'):
            assert hasattr(asm, n), \
                f"{n!r} should live in propagators.asm"

    def test_fresnel_module_owns_fresnel_and_fraunhofer(self):
        from lumenairy.propagators import fresnel
        for n in ('fresnel_propagate', 'fraunhofer_propagate'):
            assert hasattr(fresnel, n), \
                f"{n!r} should live in propagators.fresnel"

    def test_rs_module_owns_rayleigh_sommerfeld(self):
        from lumenairy.propagators import rs
        assert hasattr(rs, 'rayleigh_sommerfeld_propagate'), \
            "rayleigh_sommerfeld_propagate should live in propagators.rs"

    def test_sas_module_owns_scalable_angular_spectrum(self):
        from lumenairy.propagators import sas
        assert hasattr(sas, 'scalable_angular_spectrum_propagate'), \
            "scalable_angular_spectrum_propagate should live in propagators.sas"

    def test_mft_module_owns_bluestein_propagators(self):
        from lumenairy.propagators import mft
        for n in ('angular_spectrum_propagate_mft',
                  'fresnel_propagate_mft',
                  'fraunhofer_propagate_mft',
                  'resample_field'):
            assert hasattr(mft, n), \
                f"{n!r} should live in propagators.mft"


class TestSplitObjectIdentity:
    """Each public name resolves to the *same* Python object regardless
    of whether the caller imports it from ``propagation`` or the new
    submodule.  Object identity (not just equality) catches the
    common "I accidentally duplicated the function definition" mistake
    after a split.
    """

    @pytest.mark.parametrize('name,submod_path', [
        ('angular_spectrum_propagate', 'lumenairy.propagators.asm'),
        ('angular_spectrum_propagate_tilted', 'lumenairy.propagators.asm'),
        ('angular_spectrum_propagate_batch', 'lumenairy.propagators.asm'),
        ('apply_fresnel_curvature', 'lumenairy.propagators.asm'),
        ('_build_asm_H_square', 'lumenairy.propagators.asm'),
        ('fresnel_propagate', 'lumenairy.propagators.fresnel'),
        ('fraunhofer_propagate', 'lumenairy.propagators.fresnel'),
        ('rayleigh_sommerfeld_propagate', 'lumenairy.propagators.rs'),
        ('scalable_angular_spectrum_propagate', 'lumenairy.propagators.sas'),
        ('angular_spectrum_propagate_mft', 'lumenairy.propagators.mft'),
        ('fresnel_propagate_mft', 'lumenairy.propagators.mft'),
        ('fraunhofer_propagate_mft', 'lumenairy.propagators.mft'),
        ('resample_field', 'lumenairy.propagators.mft'),
        ('set_default_complex_dtype', 'lumenairy.propagators.fft_infra'),
        ('set_default_real_dtype', 'lumenairy.propagators.fft_infra'),
        ('set_default_wave_propagator', 'lumenairy.propagators.fft_infra'),
        ('set_default_dy', 'lumenairy.propagators.fft_infra'),
        ('get_default_complex_dtype', 'lumenairy.propagators.fft_infra'),
        ('clear_asm_caches', 'lumenairy.propagators.fft_infra'),
        ('set_asm_cache_size', 'lumenairy.propagators.fft_infra'),
        ('warmup_fft_plans', 'lumenairy.propagators.fft_infra'),
        ('_fft2', 'lumenairy.propagators.fft_infra'),
        ('_ifft2', 'lumenairy.propagators.fft_infra'),
        ('_validate_propagator_inputs',
         'lumenairy.propagators.fft_infra'),
    ])
    def test_object_identity_across_paths(self, name, submod_path):
        import importlib
        prop_obj = getattr(importlib.import_module(
            'lumenairy.propagators.propagation'), name)
        sub_obj = getattr(importlib.import_module(submod_path), name)
        assert prop_obj is sub_obj, (
            f"{name!r}: propagation re-export and submodule disagree; "
            f"the v5.1.0 split is not back-compat (duplicated "
            f"definition).")


class TestSharedStateAcrossSplit:
    """The split must NOT duplicate the mutable module-level caches
    or DEFAULT_* knobs.  The ``_H_CACHE`` / ``_PYFFTW_PLAN_CACHE`` /
    ``_FREQ_GRID_CACHE`` / etc. live in ``fft_infra`` and any insertion
    via the legacy propagation path must be visible via the submodule
    path (and vice versa).
    """

    def test_h_cache_is_one_object(self):
        from lumenairy.propagators import fft_infra, propagation
        # Same OrderedDict instance (mutable state is shared, not
        # duplicated).
        assert propagation._H_CACHE is fft_infra._H_CACHE

    def test_freq_grid_cache_is_one_object(self):
        from lumenairy.propagators import fft_infra, propagation
        assert propagation._FREQ_GRID_CACHE is fft_infra._FREQ_GRID_CACHE

    def test_bandlimit_cache_is_one_object(self):
        from lumenairy.propagators import fft_infra, propagation
        assert propagation._BANDLIMIT_CACHE is fft_infra._BANDLIMIT_CACHE

    def test_pyfftw_plan_cache_is_one_object(self):
        from lumenairy.propagators import fft_infra, propagation
        assert (propagation._PYFFTW_PLAN_CACHE
                is fft_infra._PYFFTW_PLAN_CACHE)

    def test_asm_cache_lock_is_one_object(self):
        from lumenairy.propagators import fft_infra, propagation
        assert propagation._ASM_CACHE_LOCK is fft_infra._ASM_CACHE_LOCK

    def test_set_default_complex_dtype_visible_via_propagation_attr(self):
        """``set_default_complex_dtype`` mutates ``fft_infra``'s global;
        attribute access via the legacy ``propagation`` module must
        reflect the new value (live forwarding through
        ``__getattr__``).
        """
        from lumenairy.propagators import fft_infra, propagation
        original = fft_infra.DEFAULT_COMPLEX_DTYPE
        try:
            propagation.set_default_complex_dtype(np.complex64)
            assert fft_infra.DEFAULT_COMPLEX_DTYPE == np.dtype(np.complex64)
            # Attribute access on propagation must see the live value.
            assert propagation.DEFAULT_COMPLEX_DTYPE == np.dtype(np.complex64)
        finally:
            propagation.set_default_complex_dtype(original)

    def test_set_default_wave_propagator_visible_via_propagation(self):
        from lumenairy.propagators import fft_infra, propagation
        original = fft_infra.DEFAULT_WAVE_PROPAGATOR
        try:
            propagation.set_default_wave_propagator('fresnel')
            assert fft_infra.DEFAULT_WAVE_PROPAGATOR == 'fresnel'
            assert propagation.DEFAULT_WAVE_PROPAGATOR == 'fresnel'
        finally:
            propagation.set_default_wave_propagator(original)


class TestSplitSmokeNumerics:
    """Minimal end-to-end smoke tests: each propagator runs and
    returns a sensible-shape array.  Numerics are validated by the
    pre-existing test suite; this only confirms the split didn't
    sever any call wiring.
    """

    def _make_field(self, N=64, dx=2e-6, sigma=10e-6):
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        return np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(np.complex128)

    def test_angular_spectrum_smoke(self):
        from lumenairy.propagators.propagation import angular_spectrum_propagate
        E = self._make_field()
        E_out = angular_spectrum_propagate(
            E, z=1e-3, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == E.shape
        assert np.iscomplexobj(E_out)

    def test_fresnel_smoke(self):
        from lumenairy.propagators.propagation import fresnel_propagate
        E = self._make_field()
        E_out, dx_out, dy_out = fresnel_propagate(
            E, z=10e-3, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == E.shape
        assert dx_out > 0 and dy_out > 0

    def test_fraunhofer_smoke(self):
        from lumenairy.propagators.propagation import fraunhofer_propagate
        E = self._make_field()
        E_out, dx_out, dy_out = fraunhofer_propagate(
            E, z=100e-3, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == E.shape
        assert dx_out > 0 and dy_out > 0

    def test_rayleigh_sommerfeld_smoke(self):
        from lumenairy.propagators.propagation import rayleigh_sommerfeld_propagate
        E = self._make_field()
        E_out = rayleigh_sommerfeld_propagate(
            E, z=1e-3, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == E.shape
        assert np.iscomplexobj(E_out)

    def test_sas_smoke(self):
        from lumenairy.propagators.propagation import scalable_angular_spectrum_propagate
        E = self._make_field()
        E_out, dx_out, dy_out = scalable_angular_spectrum_propagate(
            E, z=50e-3, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == E.shape
        assert dx_out > 0 and dy_out > 0

    def test_fresnel_mft_smoke(self):
        from lumenairy.propagators.propagation import fresnel_propagate_mft
        E = self._make_field()
        E_out = fresnel_propagate_mft(
            E, z=10e-3, wavelength=1.31e-6,
            dx_in=2e-6, dx_out=4e-6, N_out=32)
        assert E_out.shape == (32, 32)

    def test_asm_mft_smoke(self):
        from lumenairy.propagators.propagation import angular_spectrum_propagate_mft
        E = self._make_field()
        E_out = angular_spectrum_propagate_mft(
            E, z=1e-3, wavelength=1.31e-6,
            dx_in=2e-6, dx_out=2e-6, N_out=64)
        assert E_out.shape == (64, 64)

    def test_resample_field_smoke(self):
        from lumenairy.propagators.propagation import resample_field
        E = self._make_field()
        E_out, dx_new = resample_field(E, dx_in=2e-6, dx_out=4e-6)
        assert E_out.shape[0] > 0 and E_out.shape[1] > 0
        assert dx_new == 4e-6


class TestPropagationDunderAll:
    """``propagation.__all__`` must include every public name a caller
    could legitimately ``from lumenairy.propagators.propagation
    import *``.
    """

    def test_dunder_all_covers_public_names(self):
        import lumenairy.propagators.propagation as prop
        # All public propagator names AND the public configuration
        # surface MUST appear in __all__.
        for name in (
            'angular_spectrum_propagate', 'fresnel_propagate',
            'fraunhofer_propagate', 'rayleigh_sommerfeld_propagate',
            'scalable_angular_spectrum_propagate',
            'angular_spectrum_propagate_mft',
            'fresnel_propagate_mft', 'fraunhofer_propagate_mft',
            'resample_field', 'apply_fresnel_curvature',
            'DEFAULT_COMPLEX_DTYPE', 'set_default_complex_dtype',
            'clear_asm_caches', 'warmup_fft_plans',
        ):
            assert name in prop.__all__, (
                f"{name!r} should be in "
                f"lumenairy.propagators.propagation.__all__")
