"""Probe 3 -- cheap fold scan: which planes of a fixture carry a single fold
ring the completion's own gates accept, and what zeta ladder they span.

Uses only the module's meridional fold tracer (no rasterisation), so a 400-plane
scan costs seconds.  Prints z, r_c, kappa, l_airy, the grid gate, the two-branch
band and zeta_extrapolation = l_airy / band.

Usage:  python probe3_foldscan.py <fixture> <z0_um> <z1_um> <nz>
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES                        # noqa: E402


def scan(la, fx, zs):
    m = la.elements._lens_traced_uniform
    presc = fx['prescription']
    dx, N, wl = fx['dx'], fx['N'], fx['wavelength']
    k0 = 2.0 * np.pi / wl
    ap = presc.get('aperture_diameter')
    lr = 0.5 * float(ap) * 0.98 if ap is not None else 0.5 * N * dx
    rows = []
    for z in zs:
        f = m._trace_meridional_fold(presc, wl, float(z), 1.0, lr, 4000)
        r = {'z_um': z * 1e6, 'ok': bool(f['ok']),
             'reason': f.get('reason'), 'n_turn': int(f.get('n_turn') or 0)}
        if f['ok']:
            kappa = f['kappa']
            l_airy = 1.0 / (k0 ** (2.0 / 3.0) * kappa)
            band = float(f.get('band') or 0.0)
            r.update(r_c_um=f['r_c'] * 1e6, kappa=kappa,
                     l_airy_um=l_airy * 1e6, resolved=bool(l_airy >= 1.2 * dx),
                     band_nm=band * 1e9,
                     zeta_extrap=(l_airy / band if band > 0 else float('inf')))
        rows.append(r)
    return rows


def main():
    import lumenairy as la
    fx = FIXTURES[sys.argv[1]]
    zs = np.linspace(float(sys.argv[2]), float(sys.argv[3]),
                     int(sys.argv[4])) * 1e-6
    print(f'lumenairy: {la.__file__}  dx={fx["dx"] * 1e6:.2f} um  '
          f'gate l_airy >= {1.2 * fx["dx"] * 1e6:.2f} um')
    for r in scan(la, fx, zs):
        if not r['ok']:
            print(f"z={r['z_um']:9.2f}  --  {r['reason']} (n_turn={r['n_turn']})")
        else:
            print(f"z={r['z_um']:9.2f}  r_c={r['r_c_um']:8.3f} um  "
                  f"kappa={r['kappa']:9.4g}  l_airy={r['l_airy_um']:7.3f} um  "
                  f"resolved={r['resolved']!s:5s}  band={r['band_nm']:10.2f} nm "
                  f" zeta_x={r['zeta_extrap']:10.4g}")


if __name__ == '__main__':
    main()
