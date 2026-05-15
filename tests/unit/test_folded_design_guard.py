"""Unit tests for the folded-design wave-optics silent-drop guard.

A prescription with fold mirrors silently propagated the unfolded-
equivalent path through ``apply_real_lens*`` in 4.0-4.7; the bug was
flagged on the 3.7.8 release-notes deferred-items list and re-flagged
by the 4.7 audit.  The fix (4.8) raises a ValueError unless the
caller acknowledges the unfolded equivalent or splits the prescription
explicitly via ``split_prescription_at_mirrors``.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def _folded_rx_with_mirror():
    """Build a tiny folded prescription that mimics what load_zemax_zmx
    emits: a refracting singlet, then a flat fold mirror.  ``surfaces``
    is the refractive-only list; ``elements`` keeps the mirror."""
    glass = 'N-BK7'
    surf_a = {'radius': 0.05, 'conic': 0.0, 'aspheric_coeffs': None,
              'glass_before': 'AIR', 'glass_after': glass}
    surf_b = {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
              'glass_before': glass, 'glass_after': 'AIR'}
    mirror = {'element_type': 'mirror', 'radius': np.inf, 'conic': 0.0,
              'clear_aperture': 5e-3}
    rx = {
        'surfaces': [surf_a, surf_b],
        'thicknesses': [3e-3],
        # 3 mm singlet, 50 mm to mirror -- the second 50 mm thickness
        # would normally appear in all_thicknesses but is folded out
        # of the refractive 'thicknesses' list by the loader.
        'elements': [
            {**surf_a, 'element_type': 'surface'},
            {**surf_b, 'element_type': 'surface'},
            mirror,
        ],
        'all_thicknesses': [3e-3, 50e-3],
        'aperture_diameter': 5e-3,
        'name': 'folded-test',
    }
    return rx


class TestFoldGuard:
    """apply_real_lens / _traced / _maslov must refuse to silently drop
    a fold mirror from a prescription that has one."""

    def test_apply_real_lens_raises_on_unacknowledged_mirror(self):
        rx = _folded_rx_with_mirror()
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='mirror'):
            la.apply_real_lens(E, prescription=rx, wavelength=1.31e-6,
                                dx=10e-6)

    def test_apply_real_lens_accepts_explicit_unfolded_equivalent(self):
        rx = _folded_rx_with_mirror()
        rx['allow_unfolded_equivalent'] = True
        E = np.ones((64, 64), dtype=complex)
        # Should run without raising.
        out = la.apply_real_lens(
            E, prescription=rx, wavelength=1.31e-6, dx=10e-6)
        assert out.shape == E.shape
        assert np.all(np.isfinite(out))

    def test_apply_real_lens_traced_raises(self):
        rx = _folded_rx_with_mirror()
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='mirror'):
            la.apply_real_lens_traced(
                E, prescription=rx, wavelength=1.31e-6, dx=10e-6)

    def test_apply_real_lens_maslov_raises(self):
        rx = _folded_rx_with_mirror()
        E = np.ones((64, 64), dtype=complex)
        with pytest.raises(ValueError, match='mirror'):
            la.apply_real_lens_maslov(
                E, prescription=rx, wavelength=1.31e-6, dx=10e-6)

    def test_pure_refractive_prescription_unaffected(self):
        """A bare singlet prescription with no mirrors must still pass
        through apply_real_lens without raising."""
        rx = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        E = np.ones((64, 64), dtype=complex)
        out = la.apply_real_lens(
            E, prescription=rx, wavelength=1.31e-6, dx=10e-6)
        assert out.shape == E.shape


class TestSplitPrescriptionAtMirrors:
    """The new helper that lets users walk fold-mirror systems
    explicitly."""

    def test_pure_refractive_yields_single_leg(self):
        rx = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        legs = la.split_prescription_at_mirrors(rx)
        assert len(legs) == 1
        assert legs[0]['kind'] == 'refractive'
        # The singlet's surfaces survived the trip.
        assert len(legs[0]['prescription']['surfaces']) == 2

    def test_folded_yields_refractive_then_mirror(self):
        rx = _folded_rx_with_mirror()
        legs = la.split_prescription_at_mirrors(rx)
        assert len(legs) == 2
        assert legs[0]['kind'] == 'refractive'
        assert legs[0]['prescription']['surfaces'] is not None
        assert len(legs[0]['prescription']['surfaces']) == 2
        assert legs[1]['kind'] == 'mirror'
        # Mirror metadata preserved.
        assert legs[1]['element']['element_type'] == 'mirror'

    def test_has_mirrors_helper(self):
        rx = _folded_rx_with_mirror()
        assert la.has_mirrors(rx) is True
        rx_plain = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                                     glass='N-BK7', aperture=5e-3)
        assert la.has_mirrors(rx_plain) is False

    def test_segment_legs_runnable_through_apply_real_lens(self):
        """A refractive segment from split_prescription_at_mirrors must
        be consumable by apply_real_lens without triggering the
        fold-guard (no 'elements' carried over)."""
        rx = _folded_rx_with_mirror()
        legs = la.split_prescription_at_mirrors(rx)
        ref_leg = next(leg for leg in legs if leg['kind'] == 'refractive')
        E = np.ones((64, 64), dtype=complex)
        # Should NOT raise: segment is mirror-free by construction.
        out = la.apply_real_lens(
            E, prescription=ref_leg['prescription'],
            wavelength=1.31e-6, dx=10e-6)
        assert out.shape == E.shape
