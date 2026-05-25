"""v5.4.1 -- audit P2 NEW: TiO2 / Ta2O5 Sellmeier NaN-at-1um fix.

The v5.4 Phase 5 registry encoded the 1-term DeVore-1951 (TiO2) and
Bright-2013 (Ta2O5) Sellmeier formulae into a 3-pole template by
padding the unused slots with dummy ``(B, C) = (0.0, 1.0)``.  The
``C = 1.0 um^2`` dummy pole forces ``lam2 - C = 0`` at
``lam = 1.0 um``, producing ``0/0 = NaN`` even though
``B = 0``.  Both materials document validity that includes 1 um:
TiO2 range 400-5000 nm, Ta2O5 range 350-8000 nm.

The fix is two-fold:

* registry-side: dummy poles set to ``(0.0, 0.0)`` -- now
  ``lam2 - C = lam2 > 0`` for every positive wavelength;
* function-side: :func:`_coating_sellmeier` short-circuits any term
  with ``B == 0`` (defensive backstop against future registry entries
  that reuse the 3-pole template for a 1- or 2-term Sellmeier).

Suite pins both fixes and a regression check at the 550 nm reference
wavelength so the bug fix does not silently shift the catalogue
values used elsewhere in the library.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.coatings import (
    COATING_MATERIAL_REGISTRY,
    _coating_sellmeier,
    get_coating_material_index,
)

# ============================================================================
# Direct NaN-at-1um regression
# ============================================================================


def test_tio2_at_1um_returns_finite():
    """``get_coating_material_index('TiO2', 1 um)`` must return a
    finite real refractive index.  Before the v5.4.1 fix the dummy
    pole ``C2 = 1.0 um^2`` caused ``lam2 - C2 = 0`` at lam = 1 um,
    yielding NaN.  TiO2 DeVore-1951 at 1 um sits around 2.47-2.50."""
    n = get_coating_material_index('TiO2', 1.0e-6)
    assert np.isfinite(n), (
        f"TiO2(1 um) returned non-finite value: {n!r}")
    assert isinstance(n, float)
    # Loose sanity band -- DeVore single-pole at 1 um.
    assert 2.40 < n < 2.55, (
        f"TiO2(1 um) = {n} outside expected DeVore Sellmeier band "
        f"[2.40, 2.55]")


def test_ta2o5_at_1um_returns_finite():
    """``get_coating_material_index('Ta2O5', 1 um)`` must return a
    finite real refractive index.  Bright-2013 single-pole Sellmeier
    at 1 um sits around 2.10-2.17."""
    n = get_coating_material_index('Ta2O5', 1.0e-6)
    assert np.isfinite(n), (
        f"Ta2O5(1 um) returned non-finite value: {n!r}")
    assert isinstance(n, float)
    assert 2.05 < n < 2.20, (
        f"Ta2O5(1 um) = {n} outside expected Bright Sellmeier band "
        f"[2.05, 2.20]")


# ============================================================================
# Realistic vis-NIR sweep
# ============================================================================


def test_vis_nir_sweep_no_nan():
    """A 100 nm-step vis-NIR sweep across the documented TiO2 / Ta2O5
    validity bands must return all-finite arrays -- 1 um sits in the
    middle of both bands, so the buggy code reliably reproduced the
    NaN here when users called the accessor with ``np.arange``."""
    lams = np.arange(400e-9, 5000e-9, 100e-9)
    with warnings.catch_warnings():
        # The DeVore fit's documented validity is 0.43-1.53 um, but
        # the registry range was relaxed to 0.4-5 um for design-space
        # exploration -- the test just needs finiteness, so suppress
        # any extrapolation warnings.
        warnings.simplefilter('ignore', UserWarning)
        n_tio2 = get_coating_material_index('TiO2', lams)
        n_ta2o5 = get_coating_material_index('Ta2O5', lams)
    assert np.all(np.isfinite(n_tio2)), (
        f"TiO2 vis-NIR sweep contains non-finite values at indices "
        f"{np.where(~np.isfinite(n_tio2))[0].tolist()}")
    assert np.all(np.isfinite(n_ta2o5)), (
        f"Ta2O5 vis-NIR sweep contains non-finite values at indices "
        f"{np.where(~np.isfinite(n_ta2o5))[0].tolist()}")


# ============================================================================
# Defensive B == 0 short-circuit in _coating_sellmeier
# ============================================================================


def test_sellmeier_short_circuit_skips_zero_B():
    """Construct a synthetic 3-pole coefficient bundle where one slot
    has ``B = 0`` but ``C = lam^2`` (in um^2).  Without the
    short-circuit guard this would produce NaN; with the guard it must
    return the contribution from the remaining (non-zero-B) terms.

    The synthetic SiO2-like fit uses one real Malitson term plus a
    booby-trapped (0, lam^2) pole.  The expected answer is just the
    one-term Sellmeier result.
    """
    lam_um = 1.0
    lam_m = lam_um * 1e-6
    lam2 = lam_um * lam_um  # um^2

    # First slot: well-behaved Malitson-like pole.
    B1 = 0.6961663
    C1 = 0.0684043 ** 2
    # Second slot: BOOBY-TRAPPED -- B=0 but C == lam2 (would NaN
    # without the guard since 0 / 0 = NaN in NumPy float math).
    B2, C2 = 0.0, lam2
    # Third slot: also zero, with arbitrary C far from lam2.
    B3, C3 = 0.0, 99.9

    coeffs = ((B1, B2, B3), (C1, C2, C3))
    n = _coating_sellmeier(lam_m, coeffs)
    n_expected = float(np.sqrt(1.0 + B1 * lam2 / (lam2 - C1)))
    assert np.isfinite(n), (
        f"_coating_sellmeier returned non-finite for booby-trapped "
        f"B=0, C=lam2 term: {n!r}")
    assert float(n) == pytest.approx(n_expected, abs=1e-12), (
        f"_coating_sellmeier with B=0 short-circuit must equal the "
        f"single-term Sellmeier: got {float(n)}, expected {n_expected}")


# ============================================================================
# Regression: 550 nm reference-wavelength values unchanged
# ============================================================================


# These values were computed by hand from the pre-fix coefficients at
# lam = 550 nm; the (0.0, 0.0) and B == 0 short-circuit changes must
# not perturb them since the B == 0 terms contributed zero in the
# pre-fix formula too (at any lam != 1 um exactly).  Holding them
# fixed prevents the fix from silently shifting any catalogue value
# downstream consumers depend on.
_TIO2_550_PREFIX = 2.5832  # DeVore: sqrt(1 + 4.99048*0.3025/(0.3025 - 0.19086**2))
_TA2O5_550_PREFIX = 2.2269  # Bright: sqrt(1 + 3.582*0.3025/(0.3025 - 0.16986**2))


def test_existing_550nm_values_unchanged():
    """Pin the 550 nm catalogue values to the pre-fix numbers so the
    NaN fix cannot silently shift the at-reference-wavelength index
    of either material."""
    n_tio2 = get_coating_material_index('TiO2', 550e-9)
    n_ta2o5 = get_coating_material_index('Ta2O5', 550e-9)
    assert n_tio2 == pytest.approx(_TIO2_550_PREFIX, abs=1e-3), (
        f"TiO2(550 nm) shifted from {_TIO2_550_PREFIX} to {n_tio2} "
        f"-- the NaN fix must not perturb in-band values")
    assert n_ta2o5 == pytest.approx(_TA2O5_550_PREFIX, abs=1e-3), (
        f"Ta2O5(550 nm) shifted from {_TA2O5_550_PREFIX} to "
        f"{n_ta2o5} -- the NaN fix must not perturb in-band values")


# ============================================================================
# Literature-band sanity at 1 um
# ============================================================================


def test_value_at_1um_within_literature_band():
    """TiO2 and Ta2O5 at 1 um must land in the typical literature
    bands.  Loose tolerance: DeVore-1951 and Bright-2013 are
    visible-band fits, so 1 um is at the edge / just past the
    validity-band right edge.  The point is to catch gross
    coefficient corruption, not to pin to 4 decimals."""
    n_tio2 = get_coating_material_index('TiO2', 1.0e-6)
    n_ta2o5 = get_coating_material_index('Ta2O5', 1.0e-6)
    # TiO2 rutile literature at 1 um: ~2.45-2.55 (ordinary ray).
    assert 2.45 <= n_tio2 <= 2.55, (
        f"TiO2(1 um) = {n_tio2} outside typical literature band "
        f"[2.45, 2.55]")
    # Ta2O5 amorphous literature at 1 um: ~2.05-2.15.
    assert 2.05 <= n_ta2o5 <= 2.20, (
        f"Ta2O5(1 um) = {n_ta2o5} outside typical literature band "
        f"[2.05, 2.20]")


# ============================================================================
# Registry-level pin: dummy poles are 0.0 (not 1.0)
# ============================================================================


@pytest.mark.parametrize('material', ['TiO2', 'Ta2O5'])
def test_dummy_poles_are_zero(material):
    """Belt-and-braces: pin the registry shape so a future edit can't
    silently reintroduce ``(1.0, 1.0)`` dummy poles -- which would
    make the NaN reappear at lam = 1 um with only the function-side
    guard catching it."""
    coeffs = COATING_MATERIAL_REGISTRY[material]['sellmeier']
    (B1, B2, B3), (C1, C2, C3) = coeffs
    # Only the first slot has a real Sellmeier term for these two
    # materials; the other two slots must be (0.0, 0.0).
    assert (B2, C2) == (0.0, 0.0), (
        f"{material}: second pole must be (0.0, 0.0) dummy, got "
        f"({B2}, {C2}) -- see v5.4.1 audit P2 NEW fix")
    assert (B3, C3) == (0.0, 0.0), (
        f"{material}: third pole must be (0.0, 0.0) dummy, got "
        f"({B3}, {C3}) -- see v5.4.1 audit P2 NEW fix")
