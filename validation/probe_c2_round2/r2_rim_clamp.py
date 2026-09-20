"""WP-C2 round 2, defect D9 -- what a real design actually meets at the rim.

Two facts the Migration note did not carry, both re-measured here on the
running build:

1. **The band does not exist on the meridian.**  The two domain gates --
   ``(x*x + y*y)/(R*R)`` in ``surface._sphere_normal`` against
   ``(1 + conic) * sqrt(x*x + y*y)**2 / R**2`` in
   ``_surface_sag_derivative`` -- are located by bisection on the running
   build at ``y = 0`` for radii of both signs.  If they bisect to the SAME
   float, a meridional fan cannot enter the band at all.

2. **What IS reachable is the CLAMP, and it is pre-existing.**  A ball lens
   or hemisphere has its clear semi-diameter AT ``|R|`` by construction, so a
   rim-packed bundle crosses ``h = 0.99995 |R|``.  This counts how many of
   those rays die ``RAY_NAN`` -- a numerical-fault code, not
   ``RAY_APERTURE`` -- through BOTH normal routes, which is the number the
   Migration note should carry.

Usage:  OMP_NUM_THREADS=1 ... python r2_rim_clamp.py --root <tree>
            --out r2_rim_clamp_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

WL = 587.5618e-9
CLAMP = 0.9999


def _gate(fn, R, azimuth):
    """Bisect the largest ``h/|R|`` this route still accepts."""
    import numpy as np
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        h = mid * abs(R)
        x = np.array([h * np.cos(azimuth)])
        y = np.array([h * np.sin(azimuth)])
        if fn(x, y, R):
            lo = mid
        else:
            hi = mid
    return lo


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
    print('lumenairy.__file__ =', lumenairy.__file__)
    print('numpy', np.__version__, 'python', sys.version.split()[0])

    from lumenairy.raytrace import surface as surf
    from lumenairy.raytrace.core import Surface
    from lumenairy.raytrace.surface import RAY_NAN
    from lumenairy.raytrace.trace import _make_bundle, trace

    payload = {'lumenairy_file': lumenairy.__file__,
               'python': sys.version.split()[0],
               'numpy': np.__version__}

    # ---- 1. the two gates, on the meridian and off it -------------------
    def _analytic_ok(x, y, R):
        nz = surf._sphere_normal(x, y, R)[2]
        return bool(np.all(np.isfinite(nz)))

    def _generic_ok(x, y, R):
        s = Surface(radius=R, conic=0.0, thickness=0.0, glass_before='air',
                    glass_after='N-BK7', semi_diameter=np.inf)
        dz = surf._surface_sag_derivatives_xy(x, y, s)
        return bool(np.all(np.isfinite(dz[0])) and np.all(np.isfinite(dz[1])))

    radii = [0.0020, -0.0020, 0.0125, -0.0125, 0.0515, -0.1200, 0.5, -1.0]
    meridian = {}
    for R in radii:
        a = _gate(_analytic_ok, R, 0.0)
        g = _gate(_generic_ok, R, 0.0)
        meridian[repr(R)] = {'analytic': a, 'generic': g,
                             'identical': a == g,
                             'width_ulp': abs(a - g) / np.spacing(a)}
    payload['meridian_gates'] = meridian
    payload['meridian_band_is_empty'] = all(
        v['identical'] for v in meridian.values())
    print('meridian gates identical at all', len(radii), 'radii:',
          payload['meridian_band_is_empty'])
    print('  gate =', meridian[repr(radii[0])]['analytic'])

    # ---- 2. the clamp, on a ball lens and a hemisphere ------------------
    def _rim_bundle(n, semi, frac_lo=0.99900, frac_hi=1.0):
        rng = np.random.default_rng(20260920)
        u = rng.random(n)
        r = semi * (frac_lo + (frac_hi - frac_lo) * u)
        th = 2.0 * np.pi * rng.random(n)
        x, y = r * np.cos(th), r * np.sin(th)
        z = np.zeros(n)
        return _make_bundle(x, y, z, z, WL)

    R = 0.0125
    ball = [Surface(radius=R, conic=0.0, thickness=2 * R,
                    glass_before='air', glass_after='N-BK7',
                    semi_diameter=R),
            Surface(radius=-R, conic=0.0, thickness=0.010,
                    glass_before='N-BK7', glass_after='air',
                    semi_diameter=R)]
    hemi = [Surface(radius=R, conic=0.0, thickness=R,
                    glass_before='air', glass_after='N-BK7',
                    semi_diameter=R),
            Surface(radius=np.inf, conic=0.0, thickness=0.010,
                    glass_before='N-BK7', glass_after='air',
                    semi_diameter=R)]

    n_rays = 60000
    rows = {}
    for name, surfs in (('ball_lens', ball), ('hemisphere', hemi)):
        rays = _rim_bundle(n_rays, R)
        h_over_R = np.hypot(np.asarray(rays.x), np.asarray(rays.y)) / abs(R)
        past = int(np.sum(h_over_R ** 2 >= CLAMP))
        arms = {}
        for label, kw in (
                ('generic/surface',
                 dict(sphere_normal='generic', renormalize='surface')),
                ('analytic/exit',
                 dict(sphere_normal='analytic', renormalize='exit')),
                ('library default', {})):
            res = trace(_rim_bundle(n_rays, R), surfs, WL,
                        output_filter='last', **kw)
            f = res.image_rays
            code = np.asarray(f.error_code)
            alive = np.asarray(f.alive)
            arms[label] = {
                'alive': int(alive.sum()),
                'ray_nan': int(np.sum(code == RAY_NAN)),
                'dead': int((~alive).sum()),
            }
        rows[name] = {
            'n_rays': n_rays, 'R_m': R, 'semi_diameter_m': R,
            'max_h_over_R': float(h_over_R.max()),
            'rays_past_the_clamp': past,
            'fraction_past_the_clamp': past / n_rays,
            'arms': arms,
            'both_routes_agree': (
                arms['generic/surface'] == arms['analytic/exit']),
        }
        print(f'{name}: {past} of {n_rays} past the clamp '
              f'({100.0 * past / n_rays:.2f} %), RAY_NAN '
              f'{arms["generic/surface"]["ray_nan"]} (generic) / '
              f'{arms["analytic/exit"]["ray_nan"]} (analytic), '
              f'routes agree {rows[name]["both_routes_agree"]}')
    payload['clamp'] = rows

    with open(args.out, 'w', encoding='ascii') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
