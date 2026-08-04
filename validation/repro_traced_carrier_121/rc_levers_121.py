# D121 RESIDUAL CLOSURE -- the WEIGHTED off-centre fit branch's own levers.
#
# Once every group has crossed the niche-C1 decentre gate the residual is made
# on the D1/D7 WEIGHTED branch.  That branch differs from the historical
# concentric one in three things that flip together:
#
#   * the hard NaN sample mask becomes a weighted restriction
#     (``_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8``);
#   * the ray-map fit order rises to ``_DECENTRED_FIT_POLY_ORDER = 10``;
#   * the beam radius is measured about the CENTRE rather than the ORIGIN.
#
# This prices each lever separately, all script-side.  A lever that moves EE3
# by a meaningful fraction of the residual is a candidate fix; one that does
# not, is not.
#
# usage:  ORDERS='-1,0 -4,0' python rc_levers_121.py
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import rc_readout_121 as RD                                    # noqa: E402
import lumenairy.elements._lens_traced as _LT                  # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3

ARMS = [
    ('shipped', []),
    ('fit order 6', [(_LT, '_DECENTRED_FIT_POLY_ORDER', 6)]),
    ('fit order 8', [(_LT, '_DECENTRED_FIT_POLY_ORDER', 8)]),
    ('fit order 12', [(_LT, '_DECENTRED_FIT_POLY_ORDER', 12)]),
    ('disc w 1e-4', [(_LT, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 1e-4)]),
    ('disc w 1e-2', [(_LT, '_FIT_DISC_OUTSIDE_WEIGHT_REL', 1e-2)]),
    ('resid deg 2', [(_LT, '_REMAP_RESID_EIKONAL_DEGREE', 2)]),
    ('resid deg 6', [(_LT, '_REMAP_RESID_EIKONAL_DEGREE', 6)]),
]


def main():
    orders = os.environ.get('ORDERS', '-1,0 -4,0').split()
    rn, rs, clip, nlo = 1024, 4, 3.0, 321
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        L, M = m * LAM / period, n * LAM / period
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        a = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1, post, BACK)
        r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                      nl=a[7])
        I = np.ascontiguousarray(r['I'])
        cx, cy = RD.centroid(I, r['ax'])
        orc = RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100
        print(f"\n########## ORDER ({m:+d},{n:+d})   oracle (true ceiling) "
              f"EE3 = {orc:.4f} ##########", flush=True)
        print(f"  {'arm':>14} {'EE3 area':>9} {'residual':>9} "
              f"{'EE6 area':>9} {'FWHM um':>8} {'sha':>10}", flush=True)
        for name, patch in ARMS:
            t0 = time.time()
            with Patch(patch):
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                b = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            rr = H.rs_spot(*b[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                           nl=b[7])
            J = np.ascontiguousarray(rr['I'])
            ccx, ccy = RD.centroid(J, rr['ax'])
            e3 = RD.ee(J, rr['ax'], ccx, ccy, 3e-6, 'area') * 100
            e6 = RD.ee(J, rr['ax'], ccx, ccy, 6e-6, 'area') * 100
            print(f"  {name:>14} {e3:9.4f} {orc - e3:9.4f} {e6:9.4f} "
                  f"{rr['fwhm'] * 1e6:8.3f} "
                  f"{hashlib.sha256(J.tobytes()).hexdigest()[:8]:>10}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
