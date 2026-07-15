"""Unit tests for ``combine_prescriptions`` -- concatenating several lens
prescriptions + air gaps into ONE multi-element prescription for single-pass
whole-group propagation (the inverse of ``split_prescription_at_mirrors``)."""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def _singlet(f_sign=+1, ap=25e-3):
    R = f_sign * 50e-3
    return la.make_singlet(R, -R, 4e-3, 'N-BK7', aperture=ap)


class TestCombinePrescriptions:
    def test_two_elements_surfaces_thicknesses_aperture(self):
        a = _singlet(ap=25e-3)
        b = _singlet(ap=30e-3)
        g = la.combine_prescriptions([a, b], gaps=20e-3)
        # 2 singlets -> 4 surfaces, 3 thicknesses [tc, gap, tc]
        assert len(g['surfaces']) == 4
        assert len(g['thicknesses']) == 3
        np.testing.assert_allclose(g['thicknesses'], [4e-3, 20e-3, 4e-3])
        # combined aperture = max of the elements'
        assert g['aperture_diameter'] == pytest.approx(30e-3)

    def test_scalar_gap_broadcasts(self):
        a = _singlet()
        g = la.combine_prescriptions([a, a, a], gaps=15e-3)
        assert len(g['surfaces']) == 6
        np.testing.assert_allclose(
            g['thicknesses'], [4e-3, 15e-3, 4e-3, 15e-3, 4e-3])

    def test_explicit_gap_list(self):
        a = _singlet()
        g = la.combine_prescriptions([a, a, a], gaps=[10e-3, 30e-3])
        np.testing.assert_allclose(
            g['thicknesses'], [4e-3, 10e-3, 4e-3, 30e-3, 4e-3])

    def test_feeds_system_abcd(self):
        """The combined prescription is consumable by the propagation engine."""
        a = _singlet()
        g = la.combine_prescriptions([a, a], gaps=20e-3)
        M, efl, bfl, ffl = la.system_abcd_prescription(g, 1.31e-6)
        assert np.isfinite(efl) and efl > 0            # a real converging group

    def test_wrong_gap_count_raises(self):
        a = _singlet()
        with pytest.raises(ValueError):
            la.combine_prescriptions([a, a, a], gaps=[10e-3])   # need 2 gaps

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            la.combine_prescriptions([], gaps=[])

    def test_glass_discontinuity_raises(self):
        a = _singlet()
        b = _singlet()
        b = dict(b)
        b['surfaces'] = [dict(s) for s in b['surfaces']]
        b['surfaces'][0]['glass_before'] = 'N-SF11'    # not air -> discontinuity
        with pytest.raises(ValueError):
            la.combine_prescriptions([a, b], gaps=10e-3)

    def test_single_element_passthrough(self):
        a = _singlet()
        g = la.combine_prescriptions([a], gaps=[])
        assert len(g['surfaces']) == len(a['surfaces'])
        assert len(g['thicknesses']) == len(a['thicknesses'])
