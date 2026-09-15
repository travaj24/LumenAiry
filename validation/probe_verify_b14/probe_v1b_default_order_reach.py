"""VERIFY-B14 arm 1b -- is the C8 defect class reachable at the SHIPPED
default fit order (16), or did WP-A26 make the two fixtures museum pieces?

The WP report pins the fixtures at ``decentred_fit_poly_order=10`` because the
manufactured lobe is unreachable at 16 ON THAT ONE GEOMETRY.  The question that
decides whether the test layer is the right layer is the converse: does ANY
nearby geometry still manufacture light at the default?  This sweeps the
``_GHOST`` construction's own free parameters (the aberration coefficient, the
decentre, the beam width and the free leg) at the default order and reports the
bound-off / bound-on ratio -- the same quantity the fixtures assert on.
"""
from __future__ import annotations

import itertools
import json
import sys

sys.path.insert(0, __file__.rsplit('probe_v1b', 1)[0])
from probe_v1_halo_order import _GHOST, _call, _halo  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements import _lens_traced as LT  # noqa: E402


def main():
    order = None if len(sys.argv) < 2 else (
        None if sys.argv[1] == 'default' else int(sys.argv[1]))
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'v1b.json'
    rows = []
    alphas = (3.0, 5.0, 8.0, 12.0, 20.0)
    cxs = (0.0, 0.75e-3, 1.5e-3)
    zs = (6e-3, 12e-3)
    for alpha, cx, z in itertools.product(alphas, cxs, zs):
        spec = dict(_GHOST)
        spec['alpha'] = alpha
        spec['z'] = z
        row = {'alpha': alpha, 'cx': cx, 'z': z}
        try:
            Foff, _ = _call(spec, bound=False, guard=True, order=order, cx=cx)
            Fon, _ = _call(spec, bound=True, guard=True, order=order, cx=cx)
            h_off = _halo(Foff, spec, cx=cx)
            h_on = _halo(Fon, spec, cx=cx)
            row['off'], row['on'] = h_off, h_on
            row['ratio'] = (h_off / h_on) if h_on else None
        except Exception as exc:
            row['err'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(row, flush=True)
    res = {'lumenairy_file': la.__file__,
           'default_order': int(LT._DECENTRED_FIT_POLY_ORDER),
           'order_used': order, 'rows': rows,
           'max_ratio': max((r['ratio'] for r in rows
                             if isinstance(r.get('ratio'), float)),
                            default=None),
           'max_off': max((r['off'] for r in rows
                           if isinstance(r.get('off'), float)), default=None)}
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('MAX ratio', res['max_ratio'], 'MAX off', res['max_off'],
          'lib', la.__file__)


if __name__ == '__main__':
    main()
