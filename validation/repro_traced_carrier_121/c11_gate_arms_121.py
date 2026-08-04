# C11 -- the design-121 per-order table with the PHYSICAL decentre gate.
#
# Same instrument, arms, readout and scoring as ``rc_gate_121.py`` (which this
# file is a copy of, with only ARMS changed), so the numbers are directly
# comparable to that runner's own output.
#
#   'C11'    the shipped default -- ``DECENTRED_FIT_ARBITER = True``: both ray
#            fits are built from the already-traced samples and the one that
#            reproduces the traced OPL better WHERE THE LIGHT IS is used
#   'v532'   ``DECENTRED_FIT_ARBITER = False`` -- the pure
#            ``_DECENTRE_GATE_W_FRAC = 0.05`` selector, i.e. v5.32.0 exactly.
#            This is the FAIL-BEFORE arm and it must reproduce the recorded
#            per-order table bit for bit.
#   'default' NOTHING patched -- whatever the module ships.  Since the flag
#            ships OFF, this arm must be BYTE-IDENTICAL to 'v532', which is
#            what makes "flag-off is a no-op" a measurement on the DESIGN
#            rather than an argument about the code.
#
# SCRIPT-SIDE ONLY -- no library file is edited by this runner.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python c11_gate_arms_121.py
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

ARMS = [('default', []),
        ('C11', [(_LT, 'DECENTRED_FIT_ARBITER', True)]),
        ('v532', [(_LT, 'DECENTRED_FIT_ARBITER', False)])]


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
