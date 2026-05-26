"""v5.4.5 -- audit P2 batch: coating-material edge-case hardening.

Three surgical fixes to :mod:`lumenairy.elements.coatings`:

* **P2-1** -- ``get_coating_material_index`` now emits a
  :class:`UserWarning` on negative wavelengths.  The Sellmeier formula
  uses ``lam2 = (wl*1e6)**2`` and so is sign-symmetric
  (``n(-lam) == n(+lam)``); a single negative value buried in an array
  would otherwise silently return a plausible value at ``|lam|``.
* **P2-2** -- explicit NaN detection BEFORE the range check.  IEEE 754
  comparisons against NaN return ``False``, so the range check would
  silently miss a NaN-poisoned input.
* **P2-3** -- ``COATING_MATERIAL_REGISTRY['SiO2']['range']`` tightened
  from ``(200e-9, 8000e-9)`` to ``(200e-9, 6700e-9)`` to match the
  Malitson 1965 source citation (0.21-6.7 um); the formula returns
  non-physical ``n < 1`` for fused silica in air beyond ~6.7 um.  Range
  check also upgraded from ``>`` to ``>=`` so a wavelength exactly at
  ``lmax`` triggers the warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.coatings import (
    COATING_MATERIAL_REGISTRY,
    get_coating_material_index,
)

# ============================================================================
# P2-1: negative wavelength
# ============================================================================


def test_negative_wavelength_warns_and_returns_symmetric():
    """``n(material, -500 nm) == n(material, +500 nm)`` exactly AND a
    :class:`UserWarning` with ``"negative wavelength"`` in the message
    fires.  Sellmeier symmetry is mathematically real -- the function
    must not crash, but it MUST surface the suspicious input to the
    caller."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        n_neg = get_coating_material_index('TiO2', -500e-9)
        n_pos = get_coating_material_index('TiO2', 500e-9)

    # Symmetry: exact equality (squared wavelength erases sign).
    assert n_neg == n_pos, (
        f"Sellmeier symmetry broken: n(-500nm)={n_neg!r} vs "
        f"n(+500nm)={n_pos!r}")

    # Warning surfaced.
    neg_msgs = [str(w.message) for w in caught
                if 'negative wavelength' in str(w.message)]
    assert neg_msgs, (
        f"Expected 'negative wavelength' UserWarning; got: "
        f"{[str(w.message) for w in caught]}")


# ============================================================================
# P2-2: NaN wavelength
# ============================================================================


def test_nan_wavelength_warns_and_returns_nan():
    """Array containing NaN must:

    * preserve finite values at the non-NaN slots,
    * yield NaN at the NaN slot,
    * fire a :class:`UserWarning` with ``"NaN wavelength"`` in the
      message.
    """
    wl = np.array([500e-9, np.nan, 600e-9])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        n = get_coating_material_index('TiO2', wl)

    assert np.isfinite(n[0]), f"n[0] should be finite, got {n[0]!r}"
    assert np.isnan(n[1]),    f"n[1] should be NaN, got {n[1]!r}"
    assert np.isfinite(n[2]), f"n[2] should be finite, got {n[2]!r}"

    nan_msgs = [str(w.message) for w in caught
                if 'NaN wavelength' in str(w.message)]
    assert nan_msgs, (
        f"Expected 'NaN wavelength' UserWarning; got: "
        f"{[str(w.message) for w in caught]}")


# ============================================================================
# P2-1 + P2-2 combined
# ============================================================================


def test_negative_AND_nan_both_warn():
    """When BOTH a negative value AND a NaN are present, BOTH warnings
    must fire (order doesn't matter; both must be present).  This pins
    the design choice that the two guards are independent -- a single
    catch-all warning would be ambiguous about which problem the caller
    needs to fix."""
    wl = np.array([-500e-9, np.nan])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_coating_material_index('TiO2', wl)

    msgs = [str(w.message) for w in caught]
    has_neg = any('negative wavelength' in m for m in msgs)
    has_nan = any('NaN wavelength' in m for m in msgs)

    assert has_neg, (
        f"Expected 'negative wavelength' warning among: {msgs}")
    assert has_nan, (
        f"Expected 'NaN wavelength' warning among: {msgs}")


# ============================================================================
# P2-3: SiO2 registry range tightening
# ============================================================================


def test_sio2_range_tightened_to_malitson_validity():
    """Registry-level pin: SiO2 range must match the Malitson 1965
    source citation (0.21-6.7 um) it advertises in its inline
    comment."""
    rng = COATING_MATERIAL_REGISTRY['SiO2']['range']
    assert rng == (200e-9, 6700e-9), (
        f"SiO2 range expected (200e-9, 6700e-9), got {rng!r}")


def test_sio2_at_old_lmax_now_warns():
    """7.5 um -- the audit's canonical "n<1 returned without warning"
    case.  Now SiO2's lmax is 6.7 um, so 7.5 um is out of range and
    MUST warn.  Pins that the user is no longer silently handed a
    non-physical (n<1 in air) extrapolation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_coating_material_index('SiO2', 7.5e-6)

    range_msgs = [str(w.message) for w in caught
                  if 'validity' in str(w.message)]
    assert range_msgs, (
        f"Expected validity-range UserWarning at 7.5 um; got: "
        f"{[str(w.message) for w in caught]}")


def test_sio2_at_new_lmax_exact_warns():
    """Query at EXACTLY lmax = 6.7 um.  The fix changes the range
    check from ``>`` to ``>=``, so a sample sitting on the boundary
    triggers the warning rather than slipping through.  Closes the
    "n=0.642 at exactly lmax with no warning" case from the audit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_coating_material_index('SiO2', 6.7e-6)

    range_msgs = [str(w.message) for w in caught
                  if 'validity' in str(w.message)]
    assert range_msgs, (
        f"Expected validity-range UserWarning at lmax exactly; got: "
        f"{[str(w.message) for w in caught]}")


# ============================================================================
# Negative pin: valid input must NOT warn
# ============================================================================


def test_no_warning_for_valid_wavelength():
    """550 nm sits well inside every coating-material range.  No
    UserWarning of any kind must fire (sanity check that the new
    guards do not have a false-positive on canonical visible
    wavelengths)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        get_coating_material_index('SiO2', 550e-9)

    user_warnings = [w for w in caught
                     if issubclass(w.category, UserWarning)]
    assert not user_warnings, (
        f"Unexpected UserWarning(s) for valid 550 nm input: "
        f"{[str(w.message) for w in user_warnings]}")
