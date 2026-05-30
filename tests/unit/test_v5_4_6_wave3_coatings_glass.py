"""v5.4.6 Wave 3 regression pins: coatings/glass cluster.

- P2-1 : TiO2 / Ta2O5 advertised range tightened to cited Sellmeier validity.
- P3-4 : glass._sellmeier_index / _polynomial_index negative / NaN guards.
- P3-5 : lossless multilayer energy conservation R + T == 1.
- P3-6 : broadband_ar_v_coat layer-order convention (ambient-side first).
- P3-7 : the coatings + glass Sellmeier evaluators share _guard_wavelength.
- F-33 : BaF2 bundled Sellmeier corrected to the Li-1980 fit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.coatings import (
    COATING_MATERIAL_REGISTRY,
    broadband_ar_v_coat,
    coating_reflectance,
    quarter_wave_ar,
)
from lumenairy.glass import (
    _polynomial_index,
    _sellmeier_index,
    get_glass_index,
)

_BK7 = ((1.03961212, 0.231792344, 1.01046945),
        (0.00600069867, 0.0200179144, 103.560653))


# ---- F-33: BaF2 corrected coefficients --------------------------------

def test_baf2_index_matches_li1980():
    """BaF2 must match the Li-1980 fit (n(587.6 nm) ~ 1.4744), not the
    pre-fix low-precision row (~0.4-0.5% high)."""
    n = get_glass_index('BaF2', 587.6e-9)
    assert abs(n - 1.4744) < 1.5e-3, f"BaF2 n(587.6nm)={n}, expected ~1.4744"


# ---- P2-1: range tightening -------------------------------------------

def test_tio2_ta2o5_ranges_match_cited_validity():
    assert COATING_MATERIAL_REGISTRY['TiO2']['range'] == (430e-9, 1530e-9)
    assert COATING_MATERIAL_REGISTRY['Ta2O5']['range'] == (500e-9, 1000e-9)


# ---- P3-4: dispersion-evaluator guards --------------------------------

def test_sellmeier_negative_wavelength_warns_and_abs():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        a = _sellmeier_index(-587.6e-9, _BK7)
    b = _sellmeier_index(587.6e-9, _BK7)
    assert np.isclose(a, b), "Sellmeier must be sign-symmetric (use |lambda|)"
    assert any('negative' in str(x.message).lower() for x in w)


def test_sellmeier_nan_wavelength_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _sellmeier_index(float('nan'), _BK7)
    assert any('nan' in str(x.message).lower() for x in w)


def test_polynomial_negative_wavelength_raises():
    poly = (1.0, [(1.0, 2.0), (0.5, -2.0)])  # has an odd/negative exponent term
    with pytest.raises(ValueError, match='negative'):
        _polynomial_index(-500e-9, poly)


# ---- P3-5: energy conservation R + T == 1 for lossless stacks ---------

@pytest.mark.parametrize('angle_deg', [0.0, 30.0, 55.0])
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_lossless_stack_energy_conservation(angle_deg, pol):
    """A stack of purely real-index layers on a real substrate must
    conserve energy: R + T == 1 at any AOI for both polarisations."""
    layers = quarter_wave_ar(1.52, 550e-9)  # real-n single layer
    R, T, _ = coating_reflectance(
        layers, 550e-9, angle=np.radians(angle_deg),
        n_substrate=1.52, n_ambient=1.0, polarization=pol)
    assert np.allclose(R + T, 1.0, atol=1e-9), (
        f"R+T={float(R + T)} != 1 at {angle_deg} deg, pol={pol}")
    assert float(R) <= 1.0 + 1e-12 and float(T) <= 1.0 + 1e-12


def test_multilayer_energy_conservation_spectral():
    layers = broadband_ar_v_coat(1.52, 550e-9)
    wl = np.linspace(500e-9, 600e-9, 11)
    R, T, _ = coating_reflectance(layers, wl, n_substrate=1.52, polarization='avg')
    assert np.allclose(R + T, 1.0, atol=1e-9)


# ---- P3-6: V-coat layer-order convention (ambient-side first) ---------

def test_v_coat_layer_order_is_antireflective_not_hr():
    """broadband_ar_v_coat returns layers ambient-side first.  In that
    order the design is anti-reflective; feeding it substrate-first
    (flipped) behaves like an HR stack.  Pins the convention so a future
    refactor cannot silently invert it."""
    wl, n_sub = 550e-9, 1.52
    v = broadband_ar_v_coat(n_sub, wl)
    R_asis, _, _ = coating_reflectance(v, wl, n_substrate=n_sub, polarization='avg')
    R_flip, _, _ = coating_reflectance(list(reversed(v)), wl,
                                       n_substrate=n_sub, polarization='avg')
    assert float(R_asis) < 0.5 * float(R_flip), (
        f"As-returned order R={float(R_asis):.3f} must be much lower than "
        f"the flipped (HR) order R={float(R_flip):.3f} -- ambient-side-first "
        f"AR convention.")
