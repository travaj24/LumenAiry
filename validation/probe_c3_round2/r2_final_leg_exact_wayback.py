"""WP-C3 round 2 -- D7 on the 103-key harness's OWN chain fixture.

``r2_final_leg_exact.py`` finds ``final_leg='exact'`` bit-identical on six
ordinary two-group relays after the round-2 fix; three of the 103
archive-to-archive keys still move.  This measures the size of that move on
the fixture those keys use, so the Migration sentence can quote it.

    python r2_final_leg_exact_wayback.py <tree> <out.json>
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, 'validation', 'probe_verify_c3'))
OUT = sys.argv[2]

import lumenairy                                              # noqa: E402
from lumenairy.propagators import carrier as C                # noqa: E402
import probe_wayback as W                                     # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)


def main():
    import inspect
    env, dx, r_in, groups = W.chain_fixture()
    base = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=W.TKW,
                final_leg='exact')
    fr = dict(dx_out=0.4e-6, N_out=64)
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'default_transport': inspect.signature(
               C.propagate_traced_carrier_chain
           ).parameters['transport'].default, 'rows': {}}
    for tag, kw in (('C5-final-leg-exact-bare', dict(final_distance=9e-3)),
                    ('C5-final-leg-exact-readout',
                     dict(final_distance=9e-3, focus_readout=fr))):
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                r = C.propagate_traced_carrier_chain(
                    env, groups, W.WL_IR, dx, **dict(base, **kw))
                rec['outcome'] = 'returned'
                rec['peak'] = float(np.max(np.abs(r.field) ** 2))
                rec['power'] = float((np.abs(r.field) ** 2).sum())
                import hashlib
                a = np.ascontiguousarray(r.field, dtype=np.complex128)
                rec['sha'] = hashlib.sha256(a.tobytes()).hexdigest()[:16]
                rec['dx'] = repr(r.dx)
                rec['R'] = repr(r.R)
                rec['n_stages'] = len(r.stages)
                rec['sum_abs'] = float(np.abs(a).sum())
                rec['argmax'] = int(np.argmax(np.abs(a)))
            except BaseException as exc:                      # noqa: BLE001
                rec.update(outcome='raised', exc=type(exc).__name__,
                           msg=str(exc)[:160])
        out['rows'][tag] = rec
        print('%-30s %-9s peak=%r' % (tag, rec['outcome'], rec.get('peak')))
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('DEFAULT =', out['default_transport'], '| WROTE', OUT)


if __name__ == '__main__':
    main()
