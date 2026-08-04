# D121 RESIDUAL CLOSURE -- the C6 stationary-phase launch on the WEIGHTED
# fit branch.
#
# ``rc_tilt_121.py`` finds the residual FLAT at 0.047 points while every group
# sits below the niche-C1 decentre gate (0.05 w) and STEPPING UP as groups
# cross it onto the D1/D7 weighted off-centre fit branch.  ``_lens_traced``'s
# own ``REMAP_STATIONARY_PHASE_FIT_GUARD`` docstring already names that
# combination:
#
#   "With ``REMAP_STATIONARY_PHASE_LAUNCH = False`` both orders are clean
#    (g4 7.75e-09 / 1.92e-08, 0.25x / 0.32x of their ceilings), so the lobe is
#    C6's and it is made on the WEIGHTED branch."
#
# -- written about the (-2,0)/(-3,0) HALO, in the halo currency.  This asks the
# EE question the same way: how much of the chain-vs-oracle EE3 residual is
# that lobe?
#
# Arms (all script-side; no library file is edited):
#   shipped        C6 launch on, fit guard off, C8 bound on  -- as it ships
#   C6 off         REMAP_STATIONARY_PHASE_LAUNCH = False
#   fit guard on   REMAP_STATIONARY_PHASE_FIT_GUARD = True
#   C8 off         REMAP_INVERSE_SUPPORT_BOUND = False
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_c6_121.py
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
        ('C6 off', [(_LT, 'REMAP_STATIONARY_PHASE_LAUNCH', False)]),
        ('fitguard on', [(_LT, 'REMAP_STATIONARY_PHASE_FIT_GUARD', True)]),
        ('C8 off', [(_LT, 'REMAP_INVERSE_SUPPORT_BOUND', False)])]


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    names = [n for n, _ in ARMS]
    rn, rs, clip, nlo = 1024, 4, 3.0, 321
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print("\nEE3 (area-exact) per arm, and the residual against the "
          "CARRY=1 exact-ray oracle", flush=True)
    print(f"{'order':>8} {'oracle':>9}" + ''.join(f"{n:>13}" for n in names)
          + '   ' + ''.join(f"{'res:' + n:>13}" for n in names), flush=True)
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
        vals, e6s, shas = [], [], []
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
            e6s.append(RD.ee(J, rr['ax'], ccx, ccy, 6e-6, 'area') * 100)
            shas.append(hashlib.sha256(J.tobytes()).hexdigest()[:8])
        print(f"{o:>8} {orc:9.4f}" + ''.join(f"{v:13.4f}" for v in vals)
              + '   ' + ''.join(f"{orc - v:13.4f}" for v in vals)
              + f"   [{time.time() - t0:.0f}s]", flush=True)
        print(f"{'':>8} {'EE6:':>9}" + ''.join(f"{v:13.4f}" for v in e6s)
              + f"   shas {'/'.join(shas)}", flush=True)
    print("\nNULL: on axis 'shipped' and 'fitguard on' DIFFER by construction "
          "(the guard's whole reach is the concentric branch);\n"
          "      on (-4,0)/(-4,-2) they must be byte-identical (no group is "
          "concentric there).", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
