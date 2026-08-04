# D121 RESIDUAL CLOSURE -- is the residual keyed to the niche-C1 DECENTRE GATE?
#
# ``_lens_traced._DECENTRE_GATE_W_FRAC = 0.05`` selects, per element call,
# between the historical CONCENTRIC ray-fit path (hard NaN sample mask,
# origin-referenced beam radius) and the D1/D7 OFF-CENTRE path (weighted
# restriction, fit order ``_DECENTRED_FIT_POLY_ORDER = 10``).  Design 121's
# per-group chief-ray decentre grows linearly with the DOE order, so each
# group crosses that gate at a different fraction of the first order -- and
# ``rc_tilt_121.py`` finds the residual FLAT at 0.047-0.049 points out to
# 0.2 of the first order and 0.93 at the first order, which is where the gate
# sits.
#
# This forces the branch instead of letting the geometry choose it:
#
#   'shipped'     the gate as it ships (0.05 w)
#   'concentric'  gate = +inf   -- every call takes the historical path
#   'offcentre'   gate = 0.0    -- every call takes the D1/D7 path
#                                 (with the pixel floor also 0, this is the
#                                  pre-C1 ``bool(_bcx or _bcy)`` selector)
#
# If the residual is the branch, 'concentric' and 'offcentre' will bracket it
# and one of them will collapse it.  If both reproduce the shipped column, the
# gate is not the mechanism.
#
# SCRIPT-SIDE ONLY -- no library file is edited.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_gate_121.py
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

ARMS = [('shipped', []),
        ('concentric', [(_LT, '_DECENTRE_GATE_W_FRAC', float('inf'))]),
        ('offcentre', [(_LT, '_DECENTRE_GATE_W_FRAC', 0.0),
                       (_LT, '_DECENTRE_GATE_PIXELS', 0.0)])]


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    rn, rs, clip, nlo = 1024, 4, 3.0, 321
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print(f"\n{'order':>8} {'oracle':>9}" + ''.join(
        f"{n:>12}" for n, _ in ARMS) + ''.join(
        f"{'res ' + n:>12}" for n, _ in ARMS), flush=True)
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
        t0 = time.time()
        a = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1, post, BACK)
        r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                      nl=a[7])
        I = np.ascontiguousarray(r['I'])
        cx, cy = RD.centroid(I, r['ax'])
        orc = RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100
        vals, shas = [], []
        for _name, patch in ARMS:
            with Patch(patch):
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                b = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            rr = H.rs_spot(*b[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                           nl=b[7])
            J = np.ascontiguousarray(rr['I'])
            ccx, ccy = RD.centroid(J, rr['ax'])
            vals.append(RD.ee(J, rr['ax'], ccx, ccy, 3e-6, 'area') * 100)
            shas.append(hashlib.sha256(J.tobytes()).hexdigest()[:8])
        print(f"{o:>8} {orc:9.4f}" + ''.join(f"{v:12.4f}" for v in vals)
              + ''.join(f"{orc - v:12.4f}" for v in vals)
              + f"   shas {'/'.join(shas)}  [{time.time() - t0:.0f}s]",
              flush=True)
    print("\nNULL: on axis all three arms must be BYTE-IDENTICAL "
          "(zero decentre -> no branch to choose).", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
