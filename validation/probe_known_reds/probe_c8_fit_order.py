"""Probe 3: the c7 / c8 stimulus against the decentred fit order.

WP-A26 (bbb6c02d) raised ``_DECENTRED_FIT_POLY_ORDER`` 10 -> 16.  The c7 / c8
fixtures were calibrated against the order-10 fit's extrapolation.  This probe
walks the order and reports the manufactured halo beyond 3 w, so the stimulus
is a STATED parameter of the fixture rather than an inherited default.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from probe_c8_guard_reach import _GHOST, call, halo  # noqa: E402
import lumenairy as la  # noqa: E402


def main():
    assert 'lum_reds' in la.__file__, la.__file__
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS')},
           'ladder': {}}
    for order in (None, 6, 8, 10, 12, 14, 16, 18):
        kw = {} if order is None else {'decentred_fit_poly_order': order}
        row = {}
        for bound in (False, True):
            F, msgs = call(_GHOST, bound=bound, guard=True, **kw)
            row['bound' if bound else 'unbound'] = {
                'halo_3w': halo(F, _GHOST),
                'peak': float(np.abs(F).max()),
                'power': float((np.abs(F) ** 2).sum()),
                'n_warnings': len(msgs),
                'halo_check_fired': any('HALO self-check FAILED' in m
                                        for m in msgs)}
        row['ratio_unbound_over_bound'] = (
            row['unbound']['halo_3w'] / row['bound']['halo_3w']
            if row['bound']['halo_3w'] > 0 else None)
        out['ladder'][str(order)] = row
        print(order, json.dumps(row))
    tag = os.environ.get('PROBE_TAG', 'default')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'c8_fit_order_{tag}.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
