"""WP-C2 round 2, defect D6 -- the JAX tracer has no domain clamp, so the two
backends' vignetting differs by a whole annulus on a ball lens.

`surface._sphere_normal` is NaN outside ``h**2/R**2 < 0.9999`` and
`_surface_sag_derivative` applies the same gate, so the NumPy tracer kills
every ray past the clamp with ``RAY_NAN`` on BOTH normal routes.
``jax_trace._refract_jax`` builds the pure-spherical normal from the
intersection's own ``z`` as ``(x, y, z - R)/R`` and applies no gate at all.

This is PRE-EXISTING -- WP-C2 did not cause it and does not change it -- and
the maintainer's decision (ledger section 1, item 1.4) is to leave it and
document it rather than clamp JAX to match, because clamping would move JAX's
answers on ball-lens-like prescriptions.  This probe is the measurement that
decision rests on, re-measured on both builds.

Usage:  OMP_NUM_THREADS=1 ... python r2_jax_clamp.py --root <tree>
            --out r2_jax_clamp_win.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

WL = 587.5618e-9
CLAMP = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    import lumenairy
    assert os.path.abspath(lumenairy.__file__).startswith(root)
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', True)
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('numpy', np.__version__, 'jax', jax.__version__,
          'python', sys.version.split()[0])

    from lumenairy.raytrace import surfaces_from_prescription as sfp
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    from lumenairy.raytrace.surface import RAY_NAN, RayBundle
    from lumenairy.raytrace.trace import trace

    R = 0.0125
    pres = dict(name='ball', aperture_diameter=2 * R,
                surfaces=[dict(radius=R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None, glass_before='air',
                               glass_after='N-BK7'),
                          dict(radius=-R, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None,
                               glass_before='N-BK7', glass_after='air')],
                thicknesses=[2 * R])
    surfs = sfp(pres)

    n = 40000
    r = R * np.linspace(0.9990, 0.999999, n)
    z = np.zeros(n)
    past = (r / R) > math.sqrt(CLAMP)
    n_past = int(past.sum())

    st = make_jax_ray_state(x=r, y=z.copy(), z=z.copy(), L=z.copy(),
                            M=z.copy(), N=np.ones(n))
    ja = np.asarray(trace_jax(st, pres, WL).alive, dtype=bool)

    arms = {}
    for rn, sn in (('surface', 'generic'), ('exit', 'analytic'),
                   ('surface', 'analytic'), ('exit', 'generic')):
        rb = RayBundle(x=r.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                       M=z.copy(), N=np.ones(n), wavelength=WL,
                       alive=np.ones(n, dtype=bool), opd=np.zeros(n))
        ir = trace(rb, surfs, WL, output_filter='last',
                   renormalize=rn, sphere_normal=sn).image_rays
        a = np.asarray(ir.alive, dtype=bool)
        ec = np.asarray(ir.error_code)
        arms[f'{rn}/{sn}'] = {
            'alive_past_the_clamp': int((a & past).sum()),
            'ray_nan_past_the_clamp': int(((ec == RAY_NAN) & past).sum()),
        }
    payload = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__, 'jax': jax.__version__,
        'n_rays': n, 'R_m': R, 'clamp': CLAMP,
        'rays_past_the_clamp': n_past,
        'cpu': arms,
        'jax_alive_past_the_clamp': int((ja & past).sum()),
        'cpu_kills_all_past_the_clamp': all(
            v['alive_past_the_clamp'] == 0 for v in arms.values()),
        'jax_keeps_all_past_the_clamp': int((ja & past).sum()) == n_past,
        'disagreeing_rays': int((ja & past).sum()),
    }
    print(f'{n_past} of {n} rays past the clamp')
    for k, v in arms.items():
        print(f'  CPU {k:18s} alive {v["alive_past_the_clamp"]:5d}  '
              f'RAY_NAN {v["ray_nan_past_the_clamp"]:5d}')
    print(f'  JAX                    alive '
          f'{payload["jax_alive_past_the_clamp"]:5d}')
    print('alive-flag disagreements:', payload['disagreeing_rays'])

    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
