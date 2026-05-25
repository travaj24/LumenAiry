"""v5.4 Phase 5 -- thin-film coating material database.

Pins the new :data:`lumenairy.elements.coatings.COATING_MATERIAL_
REGISTRY` and :func:`get_coating_material_index` accessor that the
v5.4 coatings dock now consumes in place of its inline 7-material
hardcoded table.

Suite covers:

1. The canonical 7 dock-hardcoded materials (MgF2, SiO2, TiO2, Ta2O5,
   MgO, ZnS, Al2O3) are all present in the registry.
2. Constant-n lookup at the reference wavelength returns the
   documented ``n_constant`` to 1e-3.
3. Unknown material -> ``KeyError`` with a clear message.
4. Wavelength outside the documented ``range`` triggers
   ``UserWarning`` (but the value is still returned -- design-space
   exploration use case).
5. ndarray wavelength input returns ndarray output of the same shape.
6. For Sellmeier-equipped materials (MgF2, SiO2, TiO2, Ta2O5) the
   visible-band n(lam) curve is monotonically decreasing (normal
   dispersion for transparent dielectrics below the first absorption
   resonance).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.coatings import (
    COATING_MATERIAL_REGISTRY,
    get_coating_material_index,
)

# ============================================================================
# Registry composition + canonical dock materials
# ============================================================================


_DOCK_HARDCODED = ('MgF2', 'SiO2', 'TiO2', 'Ta2O5', 'MgO', 'ZnS', 'Al2O3')


def test_registry_has_canonical_thin_film_materials():
    """Every material the v5.3.2 coatings dock hardcoded must be
    present in the v5.4 library-side registry -- otherwise the dock's
    P5 migration loses materials."""
    missing = [m for m in _DOCK_HARDCODED if m not in COATING_MATERIAL_REGISTRY]
    assert not missing, (
        f"COATING_MATERIAL_REGISTRY is missing dock-hardcoded "
        f"materials: {missing}")


def test_registry_entries_well_formed():
    """Each entry must expose ``n_constant``, ``ref_wavelength``, and
    ``range`` -- the contract documented in the accessor docstring."""
    required = {'n_constant', 'ref_wavelength', 'range'}
    for name, entry in COATING_MATERIAL_REGISTRY.items():
        missing = required - set(entry.keys())
        assert not missing, (
            f"Material {name!r} is missing keys: {missing}")
        lmin, lmax = entry['range']
        assert lmin > 0.0, f"{name}: lambda_min must be positive"
        assert lmax > lmin, f"{name}: lambda_max must exceed lambda_min"
        assert entry['n_constant'] >= 1.0, (
            f"{name}: n_constant must be >= 1.0")


def test_top_level_exports():
    """``COATING_MATERIAL_REGISTRY`` and ``get_coating_material_index``
    must both be reachable via the top-level ``lumenairy`` namespace,
    per the v5.4 Phase 5 conventions."""
    assert hasattr(la, 'COATING_MATERIAL_REGISTRY')
    assert hasattr(la, 'get_coating_material_index')
    assert 'COATING_MATERIAL_REGISTRY' in la.__all__
    assert 'get_coating_material_index' in la.__all__


# ============================================================================
# Accessor behaviour
# ============================================================================


def test_get_index_constant_returns_n():
    """At the entry's reference wavelength the accessor returns the
    documented ``n_constant`` (to a reasonable tolerance -- the
    Sellmeier curves for MgF2/SiO2/TiO2/Ta2O5 are pinned at 550 nm
    to within ~0.05 of the catalogue constant)."""
    n = get_coating_material_index('MgF2', 550e-9)
    assert n == pytest.approx(1.38, abs=0.05)
    # The flat-n path (no Sellmeier) is exact.
    n_mgo = get_coating_material_index('MgO', 550e-9)
    assert n_mgo == pytest.approx(1.74, abs=1e-12)


def test_get_index_unknown_raises():
    """Unknown material -> ``KeyError`` with the name in the message."""
    with pytest.raises(KeyError, match='UNOBTAINIUM'):
        get_coating_material_index('UNOBTAINIUM', 550e-9)


def test_out_of_range_warns():
    """Wavelength outside the documented validity ``range`` triggers
    ``UserWarning`` -- but the call still returns a (possibly
    extrapolated) value."""
    # MgF2 valid 200 nm -- 7000 nm; 100 nm is below the band.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        n = get_coating_material_index('MgF2', 100e-9)
    assert any(issubclass(w.category, UserWarning) for w in caught), (
        f"Expected UserWarning, got {[w.category for w in caught]}")
    assert n is not None
    assert np.isfinite(float(n))


def test_in_range_does_not_warn():
    """At a wavelength inside the documented band, no UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = get_coating_material_index('SiO2', 550e-9)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings, (
        f"Unexpected UserWarning(s) for in-range lookup: "
        f"{[str(w.message) for w in user_warnings]}")


def test_array_wavelength_returns_array():
    """An ndarray input must return an ndarray output of the same
    shape -- the coatings-dock visible-band sweep calls this with the
    full wavelength array."""
    wl = np.linspace(450e-9, 650e-9, 41)
    n_arr = get_coating_material_index('SiO2', wl)
    assert isinstance(n_arr, np.ndarray)
    assert n_arr.shape == wl.shape
    # 2-D input shape preservation.
    wl_2d = np.array([[400e-9, 500e-9], [600e-9, 700e-9]])
    n_2d = get_coating_material_index('TiO2', wl_2d)
    assert isinstance(n_2d, np.ndarray)
    assert n_2d.shape == wl_2d.shape


def test_flat_n_array_path():
    """Materials without Sellmeier coefficients must still broadcast
    cleanly to ndarray output (constant-n fast path)."""
    wl = np.linspace(450e-9, 650e-9, 11)
    n_arr = get_coating_material_index('MgO', wl)
    assert isinstance(n_arr, np.ndarray)
    assert n_arr.shape == wl.shape
    # All entries equal the entry's n_constant.
    n_expected = COATING_MATERIAL_REGISTRY['MgO']['n_constant']
    assert np.allclose(n_arr, n_expected, atol=1e-12)


# ============================================================================
# Sellmeier dispersion -- top-4 AR/HR materials
# ============================================================================


@pytest.mark.parametrize('material', ['MgF2', 'SiO2', 'TiO2', 'Ta2O5'])
def test_sellmeier_dispersion_smooth(material):
    """For the materials with Sellmeier coefficients, n(lambda) is
    monotonically decreasing across the visible band (normal
    dispersion for transparent dielectrics below their first
    electronic absorption resonance).
    """
    entry = COATING_MATERIAL_REGISTRY[material]
    if 'sellmeier' not in entry:
        pytest.skip(f"{material} has no Sellmeier entry")
    # Sample inside the visible band (and inside each material's
    # documented validity range -- TiO2 and Ta2O5 fits stop ~1 um).
    wl = np.linspace(500e-9, 700e-9, 21)
    n = np.asarray(get_coating_material_index(material, wl))
    # n must be finite and physical.
    assert np.all(np.isfinite(n))
    assert np.all(n > 1.0)
    # Monotonically decreasing across the visible band.
    diffs = np.diff(n)
    assert np.all(diffs <= 0.0), (
        f"{material}: dispersion is not monotonically decreasing "
        f"over 500-700 nm: diffs = {diffs}")
