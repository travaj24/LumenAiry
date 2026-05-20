"""v4.16.2 Agent C regression tests.

Covers:
* Feature 1: Sellmeier formula-3 (polynomial) evaluator in glass.py
  + dispatch wiring + (empty) POLYNOMIAL_COEFFICIENTS dict.
* Feature 2-4: 3 pre-v5.0 default-config knobs --
  ``set_default_real_dtype``, ``set_default_wave_propagator``,
  ``set_default_dy``.
* Audit P3-NEW-F1-4: numpy-scalar acceptance in the GLASS_VALIDITY
  consistency check at module load.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import glass as _glass
from lumenairy.glass import (
    POLYNOMIAL_COEFFICIENTS,
    _polynomial_index,
    _check_glass_registry_consistency,
    GLASS_VALIDITY,
)
from lumenairy.propagators.propagation import (
    DEFAULT_REAL_DTYPE,
    DEFAULT_WAVE_PROPAGATOR,
    DEFAULT_DY,
    set_default_real_dtype,
    get_default_real_dtype,
    set_default_wave_propagator,
    get_default_wave_propagator,
    set_default_dy,
    get_default_dy,
)


# ============================================================================
# Feature 1 -- Sellmeier formula-3 (polynomial) evaluator
# ============================================================================

class TestPolynomialIndexEvaluator:
    """Pin the formula-3 math + dispatch wiring."""

    # feature: pre-v5.0 prep -- formula-3 evaluator math
    def test_polynomial_index_constant(self):
        """A constant n^2 = 2.25 polynomial should give n = 1.5 at any
        wavelength."""
        coeffs = (2.25, [])  # n^2 = 2.25 + 0
        n = _polynomial_index(587.6e-9, coeffs)
        assert n == pytest.approx(1.5, abs=1e-12)
        n = _polynomial_index(1.5e-6, coeffs)
        assert n == pytest.approx(1.5, abs=1e-12)

    # feature: pre-v5.0 prep -- formula-3 evaluator math
    def test_polynomial_index_schott_polynomial_form(self):
        """Test against the Schott 6-coefficient polynomial form:
        n^2 = a0 + a1*lam^2 + a2*lam^-2 + a3*lam^-4 + a4*lam^-6 +
        a5*lam^-8

        Use synthetic coefficients that give a known answer at a known
        wavelength.  Pick coefficients so n^2 evaluates exactly to a
        round number.
        """
        # Pick lam = 1 um (lam_um = 1, so all lam^k = 1).
        # n^2 = 2.0 + 0.5 + (-0.1) + 0.05 + (-0.02) + 0.01 = 2.44
        # n = sqrt(2.44) ~= 1.56205
        coeffs = (2.0, [(0.5, 2.0), (-0.1, -2.0), (0.05, -4.0),
                        (-0.02, -6.0), (0.01, -8.0)])
        n = _polynomial_index(1.0e-6, coeffs)
        assert n == pytest.approx(np.sqrt(2.44), abs=1e-10)

    # feature: pre-v5.0 prep -- formula-3 evaluator rejects bad domain
    def test_polynomial_index_negative_n_squared_raises(self):
        """Extrapolation that gives n^2 <= 0 should raise ValueError
        with the glass name and wavelength in the message."""
        coeffs = (-1.0, [])  # n^2 = -1 at any wavelength
        with pytest.raises(ValueError, match="non-positive n\\^2"):
            _polynomial_index(587.6e-9, coeffs, glass_name='SYNTHETIC')

    # feature: pre-v5.0 prep -- POLYNOMIAL_COEFFICIENTS dispatch hookup
    def test_polynomial_coefficients_dict_exists_empty(self):
        """v4.16.2 ships the dict and the evaluator + dispatch wiring,
        but no per-glass coefficients yet (staged for v5.0).  The dict
        must exist and be empty (or otherwise contain only entries
        whose n_d cross-check passes)."""
        assert isinstance(POLYNOMIAL_COEFFICIENTS, dict)
        # We don't assert it's empty -- a future PR may populate it.
        # If non-empty, each entry must be the documented
        # ``(c0, [(c_i, exp_i), ...])`` shape.
        for name, coeffs in POLYNOMIAL_COEFFICIENTS.items():
            assert isinstance(coeffs, tuple) and len(coeffs) == 2, (
                f"POLYNOMIAL_COEFFICIENTS[{name!r}] = {coeffs!r} is not "
                f"a 2-tuple (c0, pairs)")
            c0, pairs = coeffs
            assert isinstance(c0, (int, float)), (
                f"POLYNOMIAL_COEFFICIENTS[{name!r}] c0 is not numeric")
            for pair in pairs:
                assert (isinstance(pair, tuple) and len(pair) == 2), (
                    f"POLYNOMIAL_COEFFICIENTS[{name!r}] pair {pair!r} "
                    f"is not a 2-tuple")

    # feature: pre-v5.0 prep -- dispatch wiring
    def test_get_glass_index_uses_polynomial_fallback(self, monkeypatch):
        """When refractiveindex is unavailable AND the glass is in
        POLYNOMIAL_COEFFICIENTS (not in SELLMEIER_COEFFICIENTS),
        get_glass_index should dispatch to _polynomial_index instead
        of raising ImportError.
        """
        # Stash a synthetic entry that doesn't conflict with anything.
        synth_name = '__TEST_POLY_GLASS__'
        synth_coeffs = (2.25, [])  # n = 1.5 constant
        # Add to all 3 places: registry (tuple-style), POLYNOMIAL_COEFFICIENTS.
        monkeypatch.setitem(_glass.GLASS_REGISTRY, synth_name,
                            ('specs', 'TEST', 'POLY'))
        monkeypatch.setitem(POLYNOMIAL_COEFFICIENTS, synth_name,
                            synth_coeffs)
        # Force the refractiveindex-unavailable branch.
        monkeypatch.setattr(_glass, '_REFRACTIVEINDEX_AVAILABLE', False)
        n = la.get_glass_index(synth_name, 587.6e-9)
        assert n == pytest.approx(1.5, abs=1e-12)


# ============================================================================
# Feature 2 -- set_default_real_dtype
# ============================================================================

class TestSetDefaultRealDtype:
    @pytest.fixture(autouse=True)
    def _restore_default(self):
        original = get_default_real_dtype()
        yield
        set_default_real_dtype(original)

    # feature: pre-v5.0 prep -- set_default_real_dtype
    def test_set_and_get_float64(self):
        set_default_real_dtype(np.float64)
        assert get_default_real_dtype() == np.dtype(np.float64)

    # feature: pre-v5.0 prep -- set_default_real_dtype
    def test_set_and_get_float32(self):
        set_default_real_dtype(np.float32)
        assert get_default_real_dtype() == np.dtype(np.float32)

    # feature: pre-v5.0 prep -- set_default_real_dtype validation
    def test_rejects_complex_dtype(self):
        with pytest.raises(ValueError, match="float32 or"):
            set_default_real_dtype(np.complex128)

    # feature: pre-v5.0 prep -- set_default_real_dtype validation
    def test_rejects_object_dtype(self):
        with pytest.raises(ValueError, match="float32 or"):
            set_default_real_dtype(np.float16)

    # feature: pre-v5.0 prep -- top-level export
    def test_top_level_export(self):
        assert la.set_default_real_dtype is set_default_real_dtype
        assert la.get_default_real_dtype is get_default_real_dtype


# ============================================================================
# Feature 3 -- set_default_wave_propagator
# ============================================================================

class TestSetDefaultWavePropagator:
    @pytest.fixture(autouse=True)
    def _restore_default(self):
        original = get_default_wave_propagator()
        yield
        set_default_wave_propagator(original)

    # feature: pre-v5.0 prep -- set_default_wave_propagator
    def test_set_and_get_asm(self):
        set_default_wave_propagator('asm')
        assert get_default_wave_propagator() == 'asm'

    # feature: pre-v5.0 prep -- set_default_wave_propagator
    def test_set_and_get_fresnel(self):
        set_default_wave_propagator('fresnel')
        assert get_default_wave_propagator() == 'fresnel'

    # feature: pre-v5.0 prep -- set_default_wave_propagator validation
    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="unknown propagator"):
            set_default_wave_propagator('gbd')  # not in the validated set

    # feature: pre-v5.0 prep -- set_default_wave_propagator validation
    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be str"):
            set_default_wave_propagator(123)

    # feature: pre-v5.0 prep -- top-level export
    def test_top_level_export(self):
        assert la.set_default_wave_propagator is set_default_wave_propagator
        assert la.get_default_wave_propagator is get_default_wave_propagator


# ============================================================================
# Feature 4 -- set_default_dy
# ============================================================================

class TestSetDefaultDy:
    @pytest.fixture(autouse=True)
    def _restore_default(self):
        original = get_default_dy()
        yield
        set_default_dy(original)

    # feature: pre-v5.0 prep -- set_default_dy
    def test_set_none(self):
        set_default_dy(None)
        assert get_default_dy() is None

    # feature: pre-v5.0 prep -- set_default_dy
    def test_set_positive_float(self):
        set_default_dy(5e-6)
        assert get_default_dy() == pytest.approx(5e-6)

    # feature: pre-v5.0 prep -- set_default_dy validation
    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive finite"):
            set_default_dy(-1e-6)

    # feature: pre-v5.0 prep -- set_default_dy validation
    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive finite"):
            set_default_dy(0.0)

    # feature: pre-v5.0 prep -- set_default_dy validation
    def test_rejects_infinite(self):
        with pytest.raises(ValueError, match="positive finite"):
            set_default_dy(float('inf'))

    # feature: pre-v5.0 prep -- top-level export
    def test_top_level_export(self):
        assert la.set_default_dy is set_default_dy
        assert la.get_default_dy is get_default_dy


# ============================================================================
# audit closure: P3-NEW-F1-4 -- numpy scalars in GLASS_VALIDITY check
# ============================================================================

class TestGlassValidityNumpyScalarAcceptance:
    # audit closure: P3-NEW-F1-4
    def test_numpy_int_scalars_accepted(self, monkeypatch):
        """User registering a custom glass with numpy-scalar validity
        bounds must not trip the consistency check."""
        # Register a synthetic glass + validity with numpy scalars.
        synth_name = '__TEST_NPSCALAR_GLASS__'
        monkeypatch.setitem(_glass.GLASS_REGISTRY, synth_name,
                            lambda wl: 1.5)
        monkeypatch.setitem(GLASS_VALIDITY, synth_name,
                            (np.int32(300), np.int32(700)))
        # Re-run the consistency check -- must not raise.
        _check_glass_registry_consistency()

    # audit closure: P3-NEW-F1-4
    def test_numpy_float_scalars_accepted(self, monkeypatch):
        synth_name = '__TEST_NPFLOAT_GLASS__'
        monkeypatch.setitem(_glass.GLASS_REGISTRY, synth_name,
                            lambda wl: 1.5)
        monkeypatch.setitem(GLASS_VALIDITY, synth_name,
                            (np.float32(300e-9), np.float64(700e-9)))
        _check_glass_registry_consistency()

    # audit closure: P3-NEW-F1-4
    def test_non_numeric_still_rejected(self, monkeypatch):
        """The numpy-scalar widening must not weaken the actual numeric
        check -- str/object types still raise."""
        synth_name = '__TEST_BADTYPE_GLASS__'
        monkeypatch.setitem(_glass.GLASS_REGISTRY, synth_name,
                            lambda wl: 1.5)
        monkeypatch.setitem(GLASS_VALIDITY, synth_name,
                            ('300nm', '700nm'))
        with pytest.raises(RuntimeError,
                           match="2-tuple of numeric"):
            _check_glass_registry_consistency()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
