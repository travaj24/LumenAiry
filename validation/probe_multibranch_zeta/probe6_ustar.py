"""Probe 6 -- the zeta-curvature bound u* over every fixture and plane.

Geometry only (no oracle, no rasterisation): re-fit the module's own two-branch
band with a quadratic ``zeta = kappa u + q u^2`` and report
``u* = 0.1 kappa / |q|`` -- the radius at which the band's OWN measured
curvature makes the linear normal form 10 % wrong -- in Airy lengths, against
the dark fill's ``_AIRY_TAIL_CELLS``.

Usage:  python probe6_ustar.py <out.json> [<fixture>:<z0>:<z1>:<n> ...]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES                        # noqa: E402
from probe5_tail import quad_band_fit                # noqa: E402

DEFAULT = ['V:1565:1755:20', 'C:3280:3680:20', 'D:1725:1985:20',
           'W1:1740:2060:20', 'W2:4400:4500:6']


def main():
    import lumenairy as la
    dest = sys.argv[1]
    specs = sys.argv[2:] or DEFAULT
    from lumenairy.elements._lens_traced_uniform import _AIRY_TAIL_CELLS
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'AIRY_TAIL_CELLS': float(_AIRY_TAIL_CELLS)}
    print(json.dumps(hdr), flush=True)
    rows = []
    for spec in specs:
        name, z0, z1, n = spec.split(':')
        fx = FIXTURES[name]
        print(f'--- {name}: {fx["note"]}')
        for z in np.linspace(float(z0), float(z1), int(n)) * 1e-6:
            qb = quad_band_fit(la, fx, float(z))
            if qb is None:
                continue
            zx = qb['l_airy'] / qb['band'] if qb['band'] > 0 else float('inf')
            r = {'fixture': name, 'z_um': z * 1e6, 'zeta_extrapolation': zx,
                 **{k: float(v) for k, v in qb.items()}}
            rows.append(r)
            print(f"  z={z * 1e6:9.2f} zx={zx:10.4g} band/l_airy="
                  f"{qb['band_in_l_airy']:9.4g} lin_resid={qb['lin_resid']:.4f} "
                  f"q={qb['q']:12.5g} u*/l_airy={qb['u_star_in_l_airy']:9.4g}",
                  flush=True)
    good = [r for r in rows if np.isfinite(r['u_star_in_l_airy'])
            and r['lin_resid'] <= 0.15]
    if good:
        us = np.array([r['u_star_in_l_airy'] for r in good])
        print(f'\nu*/l_airy over {us.size} planes with a linear-gate-passing '
              f'band: min {us.min():.4g}  p05 {np.percentile(us, 5):.4g}  '
              f'median {np.median(us):.4g}  max {us.max():.4g}')
        print(f'dark fill depth _AIRY_TAIL_CELLS = {hdr["AIRY_TAIL_CELLS"]}; '
              f'margin to the smallest u*: {us.min() / hdr["AIRY_TAIL_CELLS"]:.3g}x')
    with open(dest, 'w', encoding='cp1252') as fh:
        json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
    print('wrote', dest)


if __name__ == '__main__':
    main()
