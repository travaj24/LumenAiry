"""E5 / R3-1 -- what a CONVERGED render actually reads, and why it is not 1.

The round-2 report's load-bearing claim is that the reading has a FIXED
reference: "a converged point-sampled quadrature deposits the same total
power at any pitch ... so this reads **1 exactly**, on any optic, at any
plane, at any grid", which is what makes the bar "a tolerance on a known
value rather than a boundary between two moving populations".  The round-2
verification measured 0.999529 .. 1.000207 on one optic's eight planes and
attributed the residue to the two renders not spanning the same window (E5).

This probe measures the distribution instead of quoting a range:

* over MANY optics and MANY planes chosen by a criterion that does not read
  the arbiter (far from every caustic the geometry probe found, and past no
  focus), and
* over a GRID ladder on one optic, because E5's mechanism is an edge effect
  and must therefore scale like 1/N if that is what it is.

Run it on the pre-fix tree and again on the post-fix tree; the two JSONs are
the before/after of the pixel-centre alignment.

Usage:  python r3control.py <out.json> [--oracle asm|none]
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import r3fixtures as FX
import r3ladder as LAD
import r3oracle as OR
import r3scan as SC


def converged_planes(G, name, n=6):
    """Planes at least 25 % of the paraxial focal distance SHORT of the fold
    window's near edge -- i.e. where the landing map is single-valued and far
    from its turning point, so the render is converged for a geometric
    reason and not because the arbiter says so."""
    g = G.get(name)
    if g is None or 'error' in g:
        return []
    fp = float(g['f_paraxial'])
    r = g.get('fold_z_range')
    near = float(r[0]) if r else 0.90 * fp
    hi = near - 0.25 * fp
    lo = 0.20 * fp
    if hi <= lo:
        hi = 0.75 * near
        lo = 0.20 * near
    return list(np.linspace(lo, hi, n))


def read_only(fx, z, ray_subsample=2, N=None, dx=None):
    from lumenairy.elements import _lens_traced_uniform as U
    g = dict(fx)
    if N is not None:
        g['N'] = N
    if dx is not None:
        g['dx'] = dx
    E_in = FX.input_field(g)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        E, d = U.apply_real_lens_traced_uniform(
            E_in, prescription=g['prescription'], wavelength=g['wavelength'],
            dx=g['dx'], output_plane_distance=float(z),
            ray_subsample=ray_subsample, n_fan=4000, return_diagnostics=True)
    return E, d, g, [str(x.message)[:60] for x in w]


def main():
    out_path = sys.argv[1]
    oracle = 'none' if '--oracle' in sys.argv and 'none' in sys.argv else 'asm'
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced_uniform as U
    cmax, cmin, pmax, pmin = SC._bars()
    SC._disable_bars()
    G = LAD.geom()
    rows = []
    try:
        # --- 1. many optics, many converged planes ------------------------
        for nm in FX.OPTICS:
            for z in converged_planes(G, nm):
                fx = FX.FIXTURES[nm]
                try:
                    E, d, g, w = read_only(fx, z)
                except Exception as exc:                    # noqa: BLE001
                    rows.append(dict(kind='optic', fixture=nm, z_um=z * 1e6,
                                     error=f'{type(exc).__name__}: {exc}'))
                    continue
                row = dict(kind='optic', fixture=nm, z_um=z * 1e6,
                           N=g['N'], dx_um=g['dx'] * 1e6, warnings=w,
                           pixel_continuity=d.get('pixel_continuity'),
                           pixel_continuity_of=d.get('pixel_continuity_of'),
                           multibranch_pixel_continuity=d.get(
                               'multibranch_pixel_continuity'),
                           reason=d.get('reason'),
                           fell_back=d.get('fell_back'),
                           n_branch_max=d.get('n_branch_max'),
                           returned_power=OR.power(E, g['dx']))
                if oracle != 'none':
                    E_or, P_in = SC.oracle_2d(g, z)
                    row['fidelity'] = OR.fidelity(E, E_or)
                    row['power_over_oracle'] = (
                        row['returned_power']
                        / max(OR.power(E_or, g['dx']), 1e-300))
                rows.append(row)
                print(json.dumps(row), flush=True)
        # --- 2. the GRID ladder: if E5's mechanism is an edge effect the
        #        residue must fall like 1/N ------------------------------
        fx = FX.FIXTURES['C']
        z = 12.0e-3
        for N in (128, 192, 256, 384, 512, 768, 1024):
            dx = 4.00e-6 * (512.0 / N)
            try:
                E, d, g, w = read_only(fx, z, N=N, dx=dx)
            except Exception as exc:                        # noqa: BLE001
                rows.append(dict(kind='grid', fixture='C', N=N,
                                 error=f'{type(exc).__name__}: {exc}'))
                continue
            row = dict(kind='grid', fixture='C', z_um=z * 1e6, N=N,
                       dx_um=dx * 1e6, warnings=w,
                       pixel_continuity=d.get('pixel_continuity'),
                       multibranch_pixel_continuity=d.get(
                           'multibranch_pixel_continuity'),
                       reason=d.get('reason'), fell_back=d.get('fell_back'),
                       returned_power=OR.power(E, dx))
            rows.append(row)
            print(json.dumps(row), flush=True)
        # --- 3. the same window at a FIXED pitch but growing N (the window
        #        grows, the sampling does not) ----------------------------
        for N in (256, 384, 512, 640, 896):
            try:
                E, d, g, w = read_only(fx, z, N=N, dx=4.00e-6)
            except Exception as exc:                        # noqa: BLE001
                rows.append(dict(kind='window', fixture='C', N=N,
                                 error=f'{type(exc).__name__}: {exc}'))
                continue
            row = dict(kind='window', fixture='C', z_um=z * 1e6, N=N,
                       dx_um=4.0, warnings=w,
                       pixel_continuity=d.get('pixel_continuity'),
                       multibranch_pixel_continuity=d.get(
                           'multibranch_pixel_continuity'),
                       reason=d.get('reason'), fell_back=d.get('fell_back'),
                       returned_power=OR.power(E, 4.00e-6))
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        U._MB_PIXEL_CONTINUITY_MAX, U._MB_PIXEL_CONTINUITY_MIN = cmax, cmin
        U._MB_POWER_RATIO_MAX, U._MB_POWER_RATIO_MIN = pmax, pmin
    cs = [r['pixel_continuity'] for r in rows
          if r.get('pixel_continuity') is not None]
    summary = dict(n=len(cs), min=min(cs) if cs else None,
                   max=max(cs) if cs else None,
                   max_abs_dev=max(abs(c - 1.0) for c in cs) if cs else None)
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__, summary=summary,
                       rows=rows), f, indent=1)
    print(json.dumps(summary, default=float))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
