"""v5.4.6 regression pins for the harder delegated findings.

- F-15: split_prescription_at_mirrors preserves the surface<->mirror
        propagation distances on each mirror leg.
- F-18: the Richards-Wolf dock worker computes a valid focal field
        (was permanently broken by a fabricated signature).
- F-29: the prescription exporters default the aperture stop to the
        prescription's own stop, not surface 0 (lossless round trip).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest

import lumenairy as la


def test_split_prescription_preserves_mirror_distances():
    from lumenairy.io.prescriptions_transforms import (
        split_prescription_at_mirrors,
    )
    rx = {
        'surfaces': [
            {'radius': 0.1, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'air'},
            {'radius': -0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'air'},
        ],
        'elements': [
            {'element_type': 'surface', 'radius': 0.1, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'air', 'surf_num': 1},
            {'element_type': 'mirror', 'radius': -0.2, 'conic': 0.0,
             'semi_diameter': 0.01, 'surf_num': 2},
            {'element_type': 'surface', 'radius': -0.05, 'conic': 0.0,
             'glass_before': 'air', 'glass_after': 'air', 'surf_num': 3},
        ],
        'all_thicknesses': [0.1, 0.05],  # len == len(elements) - 1
    }
    legs = split_prescription_at_mirrors(rx)
    mirror_legs = [leg for leg in legs if leg['kind'] == 'mirror']
    assert mirror_legs, "expected a mirror leg"
    ml = mirror_legs[0]
    assert 'distance_in' in ml and 'distance_out' in ml
    assert np.isclose(ml['distance_in'], 0.1)
    assert np.isclose(ml['distance_out'], 0.05)


@pytest.mark.skipif(
    importlib.util.find_spec('PySide6') is None, reason="PySide6 not installed")
def test_richards_wolf_dock_compute_runs():
    from lumenairy.ui.richards_wolf_dock import _rw_compute
    res = _rw_compute(NA=0.6, wavelength=633e-9, polarization='linear_x',
                      N=48, dx_m=200e-9, z_offset_m=0.0)
    assert hasattr(res, 'Ex') and hasattr(res, 'Ey') and hasattr(res, 'Ez')
    assert res.Ex.shape == (48, 48)
    assert np.isfinite(res.Ex).all()
    assert float((np.abs(res.Ex) ** 2).max()) > 0.0


@pytest.mark.parametrize('fmt', ['codev', 'quadoa'])
def test_exporter_preserves_stop_index_round_trip(fmt):
    rx = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 0.01, 'is_stop': False},
            {'radius': 0.04, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 0.008, 'is_stop': True},
            {'radius': -0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'air', 'semi_diameter': 0.01, 'is_stop': False},
        ],
        'thicknesses': [0.005, 0.02],
        'aperture_diameter': 0.016,
        'stop_index': 1,
    }
    d = tempfile.mkdtemp()
    if fmt == 'codev':
        p = os.path.join(d, 't.seq')
        la.io.export_codev_seq(rx, p, wavelength=633e-9)
        rx2 = la.io.load_codev_seq(p)
    else:
        p = os.path.join(d, 't.qos')
        la.io.export_quadoa_qos(rx, p, wavelength=633e-9)
        rx2 = la.io.load_quadoa_qos(p)
    assert rx2.get('stop_index') == 1, (
        f"{fmt} round-trip relocated the stop to {rx2.get('stop_index')}")
