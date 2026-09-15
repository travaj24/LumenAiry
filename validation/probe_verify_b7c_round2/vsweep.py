"""Is the 7 % band a property of the QUANTITY, or of the fixture set?

Every published reading is taken at the default ``ray_subsample=2`` on one
grid per optic.  The estimator argument behind "a converged render reads 1"
is that the mapped triangles are spread over many pixels -- which is a joint
property of the LAUNCH lattice (``ray_subsample``) and the OUTPUT pitch
(``dx``), neither of which the derivation varies.  This probe varies both at
planes whose FIELD is good, and asks whether the reading stays inside the
band.

A healthy, high-fidelity plane that reads above ``_PIXEL_CONTINUITY_MAX`` at
a legal setting is a FALSE REFUSAL and a defect.

Usage:  python vsweep.py <out.json> <fixture> <z_um> [<z_um> ...]
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import vfixtures as FX
import vroracle as OR
import vscan

SUBS = (1, 2, 3, 4, 6, 8)
DX_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)


def one(fx, z, sub, N, dx, oracle='asm'):
    from lumenairy.elements import _lens_traced_uniform as U
    E_in = FX.gauss(N, dx, fx['w0'])
    row = dict(fixture=fx['name'], z_um=z * 1e6, sub=sub, N=N,
               dx_um=dx * 1e6)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E, d = U.apply_real_lens_traced_uniform(
                E_in, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=dx,
                output_plane_distance=float(z), ray_subsample=sub,
                return_diagnostics=True)
        row['warnings'] = sorted({str(x.message)[:60] for x in w})
    except Exception as exc:                              # noqa: BLE001
        row['error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
        return row
    for k in ('pixel_continuity', 'multibranch_pixel_continuity',
              'multibranch_power_ratio_bracketed', 'reason', 'fell_back',
              'n_branch_max', 'pixel_continuity_of'):
        v = d.get(k)
        row[k] = (float(v) if isinstance(v, (int, float, np.floating))
                  and not isinstance(v, bool) else v)
    gx = dict(fx)
    gx['N'], gx['dx'] = N, dx
    E_or, P_in = vscan.oracle_2d(gx, z, kind=oracle)
    row['fidelity'] = OR.fidelity(E, E_or)
    row['power_over_oracle'] = (OR.power(E, dx)
                                / max(OR.power(E_or, dx), 1e-300))
    return row


def main():
    out, name = sys.argv[1], sys.argv[2]
    zs = [float(a) * 1e-6 for a in sys.argv[3:]]
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    vscan._disable_bars()
    fx = FX.FIXTURES[name]
    rows = []
    for z in zs:
        for sub in SUBS:
            r = one(fx, z, sub, fx['N'], fx['dx'])
            rows.append(r)
            print(json.dumps(r), flush=True)
        for s in DX_SCALES:
            if s == 1.0:
                continue
            dx = fx['dx'] * s
            N = int(round(fx['N'] / s / 2) * 2)
            r = one(fx, z, 2, N, dx)
            rows.append(r)
            print(json.dumps(r), flush=True)
        with open(out, 'w', encoding='cp1252') as f:
            json.dump(rows, f, indent=1)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
