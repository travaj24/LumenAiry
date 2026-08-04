# D121 RESIDUAL CLOSURE -- the residual-eikonal fit degree, per order.
#
# ``_REMAP_RESID_EIKONAL_DEGREE`` is the total degree of ``a_fit``, the smooth
# model of the input residual eikonal that niche C6 launches along.  The C6
# derivation says what is left after that launch is
#
#     1/2 grad(a - a_fit)^T H^-1 grad(a - a_fit)
#
# -- quadratic in what the FIT MISSES.  That is the brief's candidate term,
# and this sweeps the one knob that changes it.
#
# The library's own table (``_REMAP_RESID_EIKONAL_DEGREE``'s docstring) was
# taken at (-4,-2) ONLY, where degree 4 -> 6 moves the chain end to end by
# +0.23 EE3 points and costs 1.25e-02 of the input power as a ghost.  This
# asks the same question at every order, in EE3 against the true ceiling and
# area-exact.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' DEGS='3,4,5,6' python rc_resdeg_121.py
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


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    degs = [int(v) for v in os.environ.get('DEGS', '3,4,5,6').split(',')]
    rn, rs, clip, nlo = 1024, 4, 3.0, 321
    print(FI._provenance(), flush=True)
    print(f"shipped _REMAP_RESID_EIKONAL_DEGREE = "
          f"{_LT._REMAP_RESID_EIKONAL_DEGREE}, cap = "
          f"{_LT._REMAP_RESID_DEGREE_CAP}", flush=True)
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
        print(f"  {'degree':>7} {'EE3 area':>9} {'residual':>9} "
              f"{'EE6 area':>9} {'FWHM um':>8} {'sha':>10}", flush=True)
        for d in degs:
            t0 = time.time()
            with Patch([(_LT, '_REMAP_RESID_EIKONAL_DEGREE', d)]):
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                b = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            rr = H.rs_spot(*b[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                           nl=b[7])
            J = np.ascontiguousarray(rr['I'])
            ccx, ccy = RD.centroid(J, rr['ax'])
            e3 = RD.ee(J, rr['ax'], ccx, ccy, 3e-6, 'area') * 100
            e6 = RD.ee(J, rr['ax'], ccx, ccy, 6e-6, 'area') * 100
            mark = '  <- SHIPPED' if d == 4 else ''
            print(f"  {d:>7} {e3:9.4f} {orc - e3:9.4f} {e6:9.4f} "
                  f"{rr['fwhm'] * 1e6:8.3f} "
                  f"{hashlib.sha256(J.tobytes()).hexdigest()[:8]:>10}   "
                  f"[{time.time() - t0:.0f}s]{mark}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
