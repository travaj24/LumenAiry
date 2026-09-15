"""The verifier's plane scan.

ONE call per plane, on ONE tree, with the two refusal bars patched to
``inf`` at runtime (``lumenairy/`` is NOT edited) so that the reading and the
FIELD it would have refused come from the SAME call -- the builder joined a
head-tree reading to a base-tree field across two processes, which cannot see
a reading that disagrees with the field it was taken on.

The shipped decision is then RE-DERIVED from the recorded reading with the
shipped constants, which were imported before the patch.  ``vdecision.py``
spot-checks the re-derivation against the real, un-patched call.

Fidelity is scored against ``vroracle``: the angular-spectrum arm by default
(exact for the scalar Helmholtz equation, no Debye / paraxial / azimuthal
approximation), with the ``J0`` and exact-azimuth arms available for the
oracle-ceiling table.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import vfixtures as FX
import vroracle as OR


def _bars():
    from lumenairy.elements import _lens_traced_uniform as U
    return float(U._MB_PIXEL_CONTINUITY_MAX), float(U._MB_PIXEL_CONTINUITY_MIN), \
        float(U._MB_POWER_RATIO_MAX), float(U._MB_POWER_RATIO_MIN)


def _disable_bars():
    from lumenairy.elements import _lens_traced_uniform as U
    U._MB_PIXEL_CONTINUITY_MAX = float('inf')
    U._MB_PIXEL_CONTINUITY_MIN = 0.0
    U._MB_POWER_RATIO_MAX = float('inf')
    U._MB_POWER_RATIO_MIN = 0.0


def _asm_grid(fx, refine):
    need = 2.2 * float(fx['prescription']['aperture_diameter'])
    dxf = fx['dx'] / refine
    Nf = max(int(np.ceil(need / dxf)), refine * fx['N'])
    if (Nf - refine * fx['N']) % 2:
        Nf += 1
    return Nf


def oracle_2d(fx, z, kind='asm', n_fan=6000, refine=4, n_rho=1600,
              frac=0.9999, safety=6.0):
    presc, wl = fx['prescription'], fx['wavelength']
    N, dx, w0 = fx['N'], fx['dx'], fx['w0']
    h, y, opl, amp, yl, P_in = OR.exit_field(presc, wl, w0, z, n_fan=n_fan)
    if kind == 'asm':
        return OR.asm_field(y, opl, amp, z, wl, N, dx, refine=refine,
                            Nf=_asm_grid(fx, refine)), P_in
    rho_max = 0.5 * N * dx * np.sqrt(2.0) * 1.001
    rho = np.linspace(0.0, rho_max, n_rho)
    E = OR.rs_j0(y, opl, amp, z, wl, rho)
    if kind == 'exact':
        rc = OR.core_radius(rho, E, frac=frac)
        core = rho <= rc
        E = E.copy()
        E[core] = OR.rs_exact(y, opl, amp, z, wl, rho[core], safety=safety)
    return OR.to_2d(rho, E, N, dx), P_in


def scan(fx, zs, oracle='asm', ray_subsample=2, n_fan_lib=4000, **okw):
    from lumenairy.elements import _lens_traced_uniform as U
    cmax, cmin, pmax, pmin = _bars()
    _disable_bars()
    rows = []
    for z in zs:
        E_in = FX.input_field(fx)
        row = dict(fixture=fx['name'], z_um=z * 1e6, N=fx['N'],
                   dx_um=fx['dx'] * 1e6, ray_subsample=ray_subsample)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                E, d = U.apply_real_lens_traced_uniform(
                    E_in, prescription=fx['prescription'],
                    wavelength=fx['wavelength'], dx=fx['dx'],
                    output_plane_distance=float(z),
                    ray_subsample=ray_subsample, n_fan=n_fan_lib,
                    return_diagnostics=True)
            row['warnings'] = sorted({str(x.message)[:70] for x in w})
        except Exception as exc:                       # noqa: BLE001
            row['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        for k in ('pixel_continuity', 'pixel_continuity_of',
                  'pixel_continuity_decision', 'multibranch_pixel_continuity',
                  'power_ratio', 'power_ratio_triangles',
                  'multibranch_power_ratio_bracketed', 'power_ratio_decision',
                  'reason', 'fell_back', 'n_branch_max',
                  'zeta_extrapolation', 'grid_power', 'launched_power',
                  'r_c', 'fit_residual'):
            v = d.get(k)
            row[k] = (float(v) if isinstance(v, (int, float, np.floating))
                      and not isinstance(v, bool) else v)
        # the shipped decision, re-derived from the recorded reading
        c = row.get('pixel_continuity')
        if c is None:
            row['shipped_decision'] = 'not_measured'
        elif c > cmax:
            row['shipped_decision'] = 'REFUSED'
        elif c < cmin:
            row['shipped_decision'] = 'not_converged_loss'
        else:
            row['shipped_decision'] = 'ok'
        b = row.get('multibranch_power_ratio_bracketed')
        row['power_arm_refuses'] = bool(b is not None and b > pmax)
        # oracle
        if oracle == 'none':
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        E_or, P_in = oracle_2d(fx, z, kind=oracle, **okw)
        row['fidelity'] = OR.fidelity(E, E_or)
        row['power_over_oracle'] = (OR.power(E, fx['dx'])
                                    / max(OR.power(E_or, fx['dx']), 1e-300))
        row['oracle'] = oracle
        row['oracle_power'] = OR.power(E_or, fx['dx'])
        row['P_in'] = P_in
        rows.append(row)
        print(json.dumps(row), flush=True)
    U._MB_PIXEL_CONTINUITY_MAX, U._MB_PIXEL_CONTINUITY_MIN = cmax, cmin
    U._MB_POWER_RATIO_MAX, U._MB_POWER_RATIO_MIN = pmax, pmin
    return rows


def parse_zs(spec):
    out = []
    for part in spec.split(','):
        if ':' in part:
            a, b, n = part.split(':')
            out.extend(np.linspace(float(a), float(b), int(n)).tolist())
        else:
            out.append(float(part))
    return [z * 1e-6 for z in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('zs')
    ap.add_argument('out')
    ap.add_argument('--oracle', default='asm')
    ap.add_argument('--builder', action='store_true')
    ap.add_argument('--sub', type=int, default=2)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    fx = FX.builder(a.fixture) if a.builder else FX.FIXTURES[a.fixture]
    rows = scan(fx, parse_zs(a.zs), oracle=a.oracle, ray_subsample=a.sub)
    with open(a.out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0],
                       numpy=np.__version__, fixture=fx['name'],
                       note=fx.get('note'), rows=rows), f, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
