"""WP-C3 round 2 -- D7: does a ``final_leg='exact'`` chain's RESULT move?

The CHANGELOG and the Migration guide said "``final_leg='exact'`` does not
move on either setting".  The exact final leg itself runs no carrier
transport, but every GAP leg does, so the returned field moves whenever a gap
leg does.  VERIFY-WP-C3 D7 measured 24.7x-557x in peak on the PRE-round-2
tree, where the moving legs were the aliased flat-reference ones; this
re-measures it after that hole is closed, so the Migration sentence quotes a
number that is still true.

    python r2_final_leg_exact.py <tree> <out.json>
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
OUT = sys.argv[2]

import lumenairy                                              # noqa: E402
from lumenairy.propagators import carrier as C                # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)
LAM = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(r1, r2, t, ap):
    return {'name': 'p', 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [
                {'radius': r1, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': r2, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


PRESC = {'f60': singlet(60e-3, -60e-3, 3e-3, 14e-3),
         'f120': singlet(120e-3, -120e-3, 6e-3, 25.4e-3)}


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


CASES = [
    ('f60-1grp-bare', 'f60', 1, 256, 3.0e-3, 5e-3, False),
    ('f60-2grp-bare', 'f60', 2, 256, 3.0e-3, 5e-3, False),
    ('f60-2grp-readout', 'f60', 2, 256, 3.0e-3, 5e-3, True),
    ('f120-2grp-bare', 'f120', 2, 512, 3.0e-3, 15e-3, False),
    ('f120-2grp-readout', 'f120', 2, 512, 3.0e-3, 15e-3, True),
    ('f60-2grp-w1.5-bare', 'f60', 2, 512, 1.5e-3, 5e-3, False),
]


def main():
    import inspect
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'default_transport': inspect.signature(
               C.propagate_traced_carrier_chain
           ).parameters['transport'].default,
           'rows': {}}
    for tag, pk, ng, n, w, fd, ro in CASES:
        dx = 10.24e-3 / n
        gs = [{'prescription': PRESC[pk], 'gap_before': 20e-3}]
        if ng == 2:
            gs.append({'prescription': PRESC[pk], 'gap_before': 15e-3})
        kw = dict(r_in=np.inf, ray_subsample=16, n_workers=1,
                  traced_kwargs=TKW, final_leg='exact', final_distance=fd)
        if ro:
            kw['focus_readout'] = dict(dx_out=0.5e-6, N_out=64)
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                r = C.propagate_traced_carrier_chain(
                    gauss(n, dx, w), gs, LAM, dx, **kw)
                rec['outcome'] = 'returned'
                rec['peak'] = float(np.max(np.abs(r.field) ** 2))
                rec['power'] = float((np.abs(r.field) ** 2).sum())
            except BaseException as exc:                      # noqa: BLE001
                rec['outcome'] = 'raised'
                rec['exc'] = type(exc).__name__
                rec['msg'] = str(exc)[:160]
        out['rows'][tag] = rec
        print('%-20s %-9s peak=%s' % (tag, rec['outcome'],
                                      rec.get('peak')))
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('DEFAULT =', out['default_transport'], '| WROTE', OUT)


if __name__ == '__main__':
    main()
