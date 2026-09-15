"""The refusal bar's own denominator -- does the BRACKET have a blind spot the
size of the bar?

``_MB_POWER_RATIO_MAX`` reads ``min(power_ratio, power_ratio_triangles)``.
The second denominator ``_p_in_tri`` counts the WHOLE launch power of every
triangle that touches the grid, so it is an UPPER bound where ``p_in`` is a
lower one, and the multibranch's own comment records the two separating by up
to 3.26x on the delta-audit's D3 geometry (a 6 mm aperture on a 1.2 mm grid at
``ray_subsample=8``).  That comment justifies the bracket with "costs no
detection power, because a real point-focus blow-up is 1e5x ... and the
audit's silent pre-focus band is 4-8x -- decades outside either", which was
written when the only consumer was a WARNING at 2.0 on a quantity read in
decades.  WP-B7c makes the same bracket gate a hard REFUSAL at 2.0, so the
question is whether the separation the bracket introduces is now comparable to
the bar itself.

This probe measures, on the D3 geometry and on the VERIFY fixtures shrunk onto
a grid that holds a small fraction of the aperture, the two ratios and their
separation at planes through focus, and reports every plane where
``power_ratio > 2.0 >= bracket`` -- a field the multibranch's own gain arm
calls a blow-up and the completion nonetheless returns.

Usage: python probe4_bracket.py <out.json>
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    _MB_POWER_RATIO_MAX, apply_real_lens_traced_uniform)


def d3_singlet():
    """The delta-audit's D3 air-focus singlet, as the A3 suite builds it."""
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
        {'radius': -25e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 3e-3}],
        'thicknesses': [3e-3, 40e-3]}


def row(presc, wl, E, dx, z, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, d = apply_real_lens_traced_multibranch(
            E, prescription=presc, wavelength=wl, dx=dx,
            output_plane_distance=z, return_diagnostics=True, **kw)
    pr = d.get('power_ratio')
    prt = d.get('power_ratio_triangles')
    vals = [float(v) for v in (pr, prt) if v is not None and np.isfinite(v)]
    br = min(vals) if vals else None
    out = {'z': z, 'power_ratio': pr, 'power_ratio_triangles': prt,
           'bracket': br, 'n_branch_max': d.get('n_branch_max'),
           'launched_power': d.get('launched_power'),
           'launched_power_triangles': d.get('launched_power_triangles'),
           'mb_warned_gain': bool([w for w in rec if 'reconstructed grid power'
                                   in str(w.message)]),
           'blind': (pr is not None and pr > _MB_POWER_RATIO_MAX
                     and br is not None and br <= _MB_POWER_RATIO_MAX)}
    if out['launched_power'] and out['launched_power_triangles']:
        out['denominator_spread'] = (out['launched_power_triangles']
                                     / out['launched_power'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, ud = apply_real_lens_traced_uniform(
                E, prescription=presc, wavelength=wl, dx=dx,
                output_plane_distance=z, return_diagnostics=True, **kw)
        out['uni_decision'] = ud.get('power_ratio_decision')
        out['uni_reason'] = ud.get('reason')
        out['uni_refused'] = False
    except RuntimeError:
        out['uni_refused'] = True
        out['uni_decision'] = 'refused_energy_gain'
    return out


def main(argv):
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'bar': _MB_POWER_RATIO_MAX, 'blocks': {}}

    # --- block 1: the D3 geometry the bracket was introduced for
    presc = d3_singlet()
    N, dx, wl = 48, 25e-6, 1.0e-6
    xs = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    for tag, zs in (('d3_48px_near', np.linspace(38e-3, 48e-3, 21)),
                    ('d3_48px_far', np.linspace(90e-3, 130e-3, 21))):
        rows = []
        for z in zs:
            rows.append(row(presc, wl, E, dx, float(z), ray_subsample=8))
        res['blocks'][tag] = rows
    for npx in (24, 32, 64):
        xs2 = (np.arange(npx) - npx / 2.0) * dx
        X2, Y2 = np.meshgrid(xs2, xs2)
        E2 = np.exp(-(X2 ** 2 + Y2 ** 2) / (1.2e-3) ** 2).astype(
            np.complex128)
        rows = []
        for z in np.linspace(90e-3, 130e-3, 17):
            rows.append(row(presc, wl, E2, dx, float(z), ray_subsample=8))
        res['blocks'][f'd3_{npx}px_far'] = rows

    # --- block 2: the VERIFY fixtures on a grid far smaller than the aperture
    for nm, zs in (('S', np.linspace(3200e-6, 3290e-6, 19)),
                   ('M', np.linspace(2250e-6, 2360e-6, 23)),
                   ('Q', np.linspace(5400e-6, 5500e-6, 21))):
        fx = FX.FIXTURES[nm]
        for shrink, sub in ((8, 8), (4, 4)):
            N2 = fx['N'] // shrink
            E2 = FX.gauss(N2, fx['dx'], fx['w0'])
            rows = []
            for z in zs:
                rows.append(row(fx['prescription'], fx['wavelength'], E2,
                                fx['dx'], float(z), ray_subsample=sub))
            res['blocks'][f'{nm}_N{N2}_sub{sub}'] = rows

    blind = []
    for k, rows in res['blocks'].items():
        for r in rows:
            if r['blind']:
                blind.append({'block': k, **r})
    res['n_blind'] = len(blind)
    res['blind_rows'] = blind
    res['max_denominator_spread'] = max(
        [r.get('denominator_spread') or 0.0
         for rows in res['blocks'].values() for r in rows] or [0.0])
    with open(argv[1], 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"bar={_MB_POWER_RATIO_MAX}  blind rows={len(blind)}  "
          f"max denominator spread={res['max_denominator_spread']:.4g}")
    for b in blind:
        print(f"  {b['block']} z={b['z'] * 1e3:.4f} mm  "
              f"power_ratio={b['power_ratio']:.4g} "
              f"triangles={b['power_ratio_triangles']:.4g} "
              f"bracket={b['bracket']:.4g} nbr={b['n_branch_max']} "
              f"mb_warned={b['mb_warned_gain']} "
              f"uni={b['uni_decision']}/{b.get('uni_reason')}")


if __name__ == '__main__':
    main(sys.argv)
