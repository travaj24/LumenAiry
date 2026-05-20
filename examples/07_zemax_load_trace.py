"""
Lumenairy example 7 -- load a Zemax prescription and trace a ray fan.

Demonstrates the v4.14.x Zemax loader + raytrace bridge (audit
AUDIT_V4_16_0_DEEP P1-3 / UX cleanup item 22): no prior example
in the suite exercises ``load_zemax_zmx`` end-to-end.

Tries to load a real ``.zmx`` file from the validation fixture set
(Thorlabs AC254-100-C achromat); falls back to a programmatically
built N-BK7 singlet if the fixture is missing, so the example always
runs from a fresh checkout.  Either path ends in a ray-fan trace and
prints the RMS spot size at the image plane.

Run::

    python examples/07_zemax_load_trace.py

Author: Andrew Traverso -- v4.16.1 / Agent D.
"""
from __future__ import annotations

import os

import numpy as np

import lumenairy as la


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Try to load a real Zemax .zmx file from the validation
    #    fixture set; fall back to a programmatic singlet if missing.
    # ------------------------------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    zmx_path = os.path.join(
        repo_root, 'validation', 'real_lens_opd',
        'zemax_prescriptions', 'AC254_100_C.zmx',
    )

    wavelength = 587.56e-9  # d-line
    if os.path.exists(zmx_path):
        print(f'Loading Zemax prescription: {zmx_path}')
        prescription = la.load_zemax_zmx(zmx_path)
        print(f"  name              = {prescription.get('name')}")
        print(f"  aperture_diameter = "
              f"{prescription.get('aperture_diameter')*1e3:.2f} mm")
        print(f"  n_surfaces        = {len(prescription['surfaces'])}")
        # Most .zmx files set the object at infinity; pin an explicit
        # finite object distance for the OPD/spot evaluation below.
        prescription['object_distance'] = 1.0
        semi_ap = 0.45 * prescription['aperture_diameter']
    else:
        print(f'Zemax fixture not present at {zmx_path};')
        print('  falling back to a programmatic N-BK7 singlet.')
        prescription = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=4e-3,
            glass='N-BK7', aperture=10e-3,
        )
        prescription['object_distance'] = 1.0
        semi_ap = 4.5e-3

    # ------------------------------------------------------------------
    # 2. Build the sequential surface list from the prescription.
    # ------------------------------------------------------------------
    surfaces = la.surfaces_from_prescription(prescription)
    print(f'  surfaces_from_prescription -> {len(surfaces)} surfaces')

    # ------------------------------------------------------------------
    # 3. Trace a tangential ray fan (11 rays in y across the pupil).
    # ------------------------------------------------------------------
    fan = la.make_fan(
        axis='y', semi_aperture=semi_ap, n_rays=11,
        field_angle=0.0, wavelength=wavelength,
    )
    result = la.trace(fan, surfaces, wavelength=wavelength)

    # ------------------------------------------------------------------
    # 4. Diagnostics: how many rays survived, RMS spot at image plane.
    # ------------------------------------------------------------------
    img = result.image_rays
    alive = np.asarray(img.alive, dtype=bool)
    n_alive = int(alive.sum())
    print(f'  rays alive at image plane: {n_alive} / {len(fan.x)}')

    if n_alive >= 2:
        xs = np.asarray(img.x)[alive]
        ys = np.asarray(img.y)[alive]
        rms = float(np.sqrt(np.mean(xs**2 + ys**2)))
        print(f'  RMS transverse radius:     {rms*1e6:.3f} um')
    else:
        print('  RMS transverse radius:     (no rays survived)')


if __name__ == '__main__':
    main()
