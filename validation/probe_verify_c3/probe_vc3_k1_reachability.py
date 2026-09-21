"""VERIFY-WP-C3 -- WHEN is the one-step Collins readout route reachable at all?

K1 = space_term + angle_term, angle_term = 2 dx theta/lambda in [0, 1]
(quantised in 2/N; it is the envelope's own angular fill against the grid
Nyquist, and ``_collins_containment_radius`` SATURATES at the outermost
``fftfreq`` bin when the grid already clipped the tail).  So:

  * angle_term == 1 EXACTLY  =>  K1 = 1 + space_term > 1  =>  the one-step
    readout is UNREACHABLE at ANY final distance and ANY N;
  * space_term = 2 dx |A| r / (lambda |B|) does NOT vanish with a long final
    leg: |A| = |1 + z/R| -> |z|/|R|, so space_term -> 2 dx r/(lambda |R_exit|),
    a FLOOR set by the exit reference, not by the leg.

This probe measures both terms on real chain exits over a grid of
configurations and reports where, if anywhere, K1 <= 1.
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE)
LAM = 1.31e-6


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def decompose(env, R, z, dx, lam=LAM):
    Rx, _Ry, _ = C._parse_carrier(R, 'p')
    A, B, _, _ = C._collins_envelope_abcd(Rx, z, np.inf)
    n = np.shape(env)[-1]
    r, _, th, _ = C._collins_input_box(env, dx, dx, lam, C._COLLINS_TAIL_FRAC)
    ang = 2.0 * dx * th / lam
    sp = 2.0 * dx * abs(A) * r / abs(B) / lam
    return dict(A=A, B=B, R=Rx, r_mm=r * 1e3, theta_mrad=th * 1e3,
                angle_term=ang, angle_bin_j=round(ang * n / 2.0),
                n_over_2=n // 2, angle_saturated=bool(ang >= 1.0 - 1e-15),
                space_term=sp, k1=sp + ang,
                space_floor_long_leg=2.0 * dx * r / (lam * abs(Rx))
                if np.isfinite(Rx) and Rx != 0 else None)


def main():
    from tests.unit.test_audit2609_b4_collins_transport import (
        _singlet, _CHAIN_TKW)
    out = {'lumenairy': lumenairy.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'rows': []}
    for n, dx, w, f in ((256, 60e-6, 4.5e-3, 60e-3),
                        (256, 60e-6, 1.2e-3, 300e-3),
                        (512, 15e-6, 0.8e-3, 300e-3),
                        (512, 8e-6, 0.30e-3, 300e-3),
                        (1024, 4e-6, 0.30e-3, 300e-3),
                        (1024, 2e-6, 0.10e-3, 300e-3)):
        presc = _singlet(2 * f, -2 * f, 3e-3, 'N-BK7',
                         max(6e-3, 4 * w), 'p')
        groups = [{'prescription': presc, 'gap_before': 10e-3}]
        env = gauss(n, dx, w)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = C.propagate_traced_carrier_chain(
                    env, groups, LAM, dx, r_in=np.inf, ray_subsample=16,
                    n_workers=1, traced_kwargs=_CHAIN_TKW,
                    final_leg='paraxial', final_distance=0.0,
                    transport='sziklas')
        except Exception as exc:                        # noqa: BLE001
            out['rows'].append(dict(N=n, dx_um=dx * 1e6, w_mm=w * 1e3,
                                    err=f'{type(exc).__name__}: {exc}'[:160]))
            continue
        e, R, dxe = res.field, res.R, res.dx
        dxe = float(dxe[0]) if isinstance(dxe, tuple) else float(dxe)
        row = dict(N=n, in_dx_um=dx * 1e6, w_mm=w * 1e3, f_mm=f * 1e3,
                   exit_dx_um=dxe * 1e6, by_final_distance={})
        for z in (8e-3, 50e-3, 200e-3, 1.0, 5.0):
            row['by_final_distance'][f'{z}'] = decompose(e, R, z, dxe)
        out['rows'].append(row)
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'k1_reachability_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    for r in out['rows']:
        if 'err' in r:
            print('ERR', r['N'], r['err'])
            continue
        print(f"N={r['N']:5d} dx_in={r['in_dx_um']:7.2f}um w={r['w_mm']:6.3f}mm"
              f" exit_dx={r['exit_dx_um']:8.3f}um")
        for z, d in r['by_final_distance'].items():
            print(f"    z={z:>6}  angle={d['angle_term']:.6f}"
                  f" (j={d['angle_bin_j']}/{d['n_over_2']},"
                  f" sat={d['angle_saturated']})"
                  f"  space={d['space_term']:.6f}  K1={d['k1']:.6f}"
                  f"  route={'collins' if d['k1'] <= 1 else 'sziklas'}")
    print('WROTE', p)


if __name__ == '__main__':
    main()
