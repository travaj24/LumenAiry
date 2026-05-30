"""v5.4.6 Wave 8 regression pins: raytrace + codegen cluster.

- F-14 : codegen negates the mirror radius (prescription raytrace
         convention R<0=concave -> apply_mirror wave-side R>0=concave).
- F-8  : ray_fan_data / opd_fan_data build a valid chief ray on an
         internal-stop system (make_ray no longer fed ep_z as L).

P3-2 (conic-domain NaN derivative) and P3-3 (tangent-ray acceptance) are
covered by the raytrace suite remaining green; they are surgical and have
no isolated reproducer here.
"""
from __future__ import annotations

import re

import numpy as np

import lumenairy as la


def test_codegen_negates_mirror_radius():
    """A concave prescription mirror (R<0) must be emitted to apply_mirror
    with the wave-side sign (R>0), so the generated script reproduces the
    prescription's focusing direction."""
    R_presc = -0.5  # concave focusing mirror in raytrace convention
    rx = {
        'surfaces': [{'radius': R_presc, 'conic': 0.0,
                      'aspheric_coeffs': None,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [0.0],
        'elements': [{'element_type': 'mirror', 'radius': R_presc,
                      'conic': 0.0, 'glass_before': 'air',
                      'glass_after': 'air', 'semi_diameter': 5e-3,
                      'surf_num': 1}],
        'all_thicknesses': [0.0],
        'coord_breaks': [],
    }
    script = la.io.generate_simulation_script(
        rx, wavelength=633e-9, N=64, dx=10e-6, source_sigma=2e-3,
        include_plotting=False, include_analysis=False)
    assert 'apply_mirror' in script
    m = re.search(r'apply_mirror\([^)]*radius=([-\d.eE+]+)', script)
    assert m, f"no apply_mirror radius found in:\n{script}"
    emitted = float(m.group(1))
    assert np.isclose(emitted, -R_presc), (
        f"codegen emitted radius={emitted}; expected the negated "
        f"prescription radius {-R_presc} (raytrace->wave-side conversion)")


def test_ray_fan_internal_stop_chief_ray():
    """ray_fan_data must build a finite chief ray on an internal-stop
    system (post-F-8 / post-seidel-F-2)."""
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.ray_fan import ray_fan_data

    prescription = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 0.01, 'is_stop': False},
            {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 0.008, 'is_stop': True},
            {'radius': -0.05, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'air', 'semi_diameter': 0.01, 'is_stop': False},
        ],
        'thicknesses': [0.005, 0.02, 0.0],
    }
    surfaces = surfaces_from_prescription(prescription)
    # Just needs to run without raising (UnboundLocalError / phantom-z L).
    out = ray_fan_data(surfaces, semi_aperture=0.008, wavelength=550e-9,
                       field_angle=np.radians(1.0))
    assert out is not None
