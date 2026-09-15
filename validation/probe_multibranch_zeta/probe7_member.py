"""Probe 7 -- the member-selection discriminator.

The completion wins on WP-B7b's fixture and loses on VERIFY-B7b's and on two
more, at every ``zeta_extrapolation``.  The two fixture families differ in how
hard the APERTURE truncates the Gaussian (WP-B7b's semi-aperture is 2.0 w0, the
others 1.36-1.5 w0), and the members that are NOT branch sums -- the ray-to-wave
hand-off -- carry the aperture-edge diffraction exactly while the fold
completion models only the fold.  This sweeps the beam width at a FIXED optic
and plane and reports the ordering.

Usage:  python probe7_member.py <fixture> <z_um> <w0_um,w0_um,...> <out.json>
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES, gauss                 # noqa: E402
import oracle as orc                                 # noqa: E402


def main():
    import lumenairy as la
    fxname, z_um, w0s, dest = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    base = FIXTURES[fxname]
    z = z_um * 1e-6
    dx, N, wl = base['dx'], base['N'], base['wavelength']
    presc = base['prescription']
    semi = 0.5 * float(presc['aperture_diameter']) * 0.98
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'fixture': fxname, 'z_um': z_um, 'semi_launch_um': semi * 1e6}
    print(json.dumps(hdr), flush=True)
    rows = []
    for w0_um in [float(v) for v in w0s.split(',')]:
        fx = dict(base, w0=w0_um * 1e-6)
        E_in = gauss(N, dx, fx['w0'])
        E_or, _, _, _ = orc.oracle_field(fx, z, n_fan=9000, n_rho=3600)
        out = {'w0_um': w0_um, 'semi_over_w0': semi / (w0_um * 1e-6),
               'edge_amplitude': float(np.exp(-(semi / (w0_um * 1e-6)) ** 2))}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            E_u, d = la.elements._lens_traced_uniform.\
                apply_real_lens_traced_uniform(
                    E_in, prescription=presc, wavelength=wl, dx=dx,
                    output_plane_distance=z, return_diagnostics=True)
            E_rd = np.asarray(la.apply_real_lens_traced(
                E_in, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z, amplitude_model='ray_density',
                caustic='wave', n_workers=1, on_undersample='silent'))
            E_wv = np.asarray(la.apply_real_lens_traced(
                E_in, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z, amplitude_model='screen',
                caustic='wave', n_workers=1, on_undersample='silent'))
        out['zeta_extrapolation'] = (None if d.get('zeta_extrapolation') is None
                                     else float(d['zeta_extrapolation']))
        out['fell_back'] = bool(d.get('fell_back'))
        out['uniform'] = orc.fidelity(E_u, E_or)
        out['ray_density'] = orc.fidelity(E_rd, E_or)
        out['wave'] = orc.fidelity(E_wv, E_or)
        out['winner'] = max(('uniform', 'ray_density', 'wave'),
                            key=lambda k: out[k])
        out['uniform_minus_ray_density'] = out['uniform'] - out['ray_density']
        rows.append(out)
        print(f"w0={w0_um:7.1f} um  semi/w0={out['semi_over_w0']:5.2f}  "
              f"edge_amp={out['edge_amplitude']:9.3e}  zx="
              f"{out['zeta_extrapolation']!s:>9.9s}  uniform={out['uniform']:.4f} "
              f"ray_density={out['ray_density']:.4f} wave={out['wave']:.4f}  "
              f"uni-rd={out['uniform_minus_ray_density']:+.4f}  "
              f"-> {out['winner']}", flush=True)
        with open(dest, 'w', encoding='cp1252') as fh:
            json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
    print('wrote', dest)


if __name__ == '__main__':
    main()
