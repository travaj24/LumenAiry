# D121 RESIDUAL CLOSURE -- is the AREA-EXACT residual converged?
#
# ``D121_FINAL_CLOSURE`` S3.4/S6.2 established the instrument band and the
# discretisation dependence with the SHIPPED hard pixel mask, which carries a
# +-0.45-point quantisation that does not cancel between arms
# (``rc_score_121.py`` section 3).  Every one of those conclusions has to be
# re-taken on the area-exact mask before the residual it leaves can be called
# real.  This script sweeps, per arm:
#
#   ORACLE   the launch density NL          (its only discretisation)
#   CHAIN    the chain grid RN, the element ray_subsample RS, the readout crop
#            CLIP and the readout upsample UP
#
# and scores every row with the SAME area-exact mask, so a moving number is a
# moving FIELD.
#
# usage:  ORD=-1,0 python rc_converge_121.py
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import rc_readout_121 as RD                                    # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3


def score(x0, y0, amp, ph0, p, q, surfs, nl, xci, yci, dxo=0.4e-6, nout=61):
    r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, BACK, xci, yci,
                  dx_out=dxo, n_out=nout, nl=nl)
    I = np.ascontiguousarray(r['I'])
    cx, cy = RD.centroid(I, r['ax'])
    return (RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100,
            RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100,
            r['ee3'] * 100, r['fwhm'] * 1e6)


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-1,0').split(','))
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    L, M = m * LAM / period, n * LAM / period
    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])
    print(f"\n########## ORDER ({m:+d},{n:+d}) -- AREA-EXACT MASK ##########")
    hdr = (f"  {'arm / knob':34s} {'EE3 area':>9} {'EE6 area':>9} "
           f"{'EE3 hard':>9} {'FWHM um':>8}")

    print("\nORACLE (CARRY=1) vs launch density NL")
    print(hdr)
    env, R, dx, _P = C.chain_a(n=1024, rs=4)
    base = {}
    for nl0 in (161, 241, 321, 481, 641):
        t0 = time.time()
        a = FI.oracle_launch(env, R, dx, L, M, nl0, 3.0, 1, post, BACK)
        v = score(*a[:7], a[7], xci, yci)
        base[('oracle', nl0)] = v
        print(f"  {f'NL={nl0}':34s} {v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f} "
              f"{v[3]:8.3f}   [{time.time() - t0:.0f}s]", flush=True)

    print("\nORACLE (CARRY=1) vs readout crop CLIP   (NL=321)")
    print(hdr)
    for cl in (2.5, 3.0, 3.5, 4.0):
        t0 = time.time()
        a = FI.oracle_launch(env, R, dx, L, M, 321, cl, 1, post, BACK)
        v = score(*a[:7], a[7], xci, yci)
        print(f"  {f'CLIP={cl}':34s} {v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f} "
              f"{v[3]:8.3f}   [{time.time() - t0:.0f}s]", flush=True)

    print("\nCHAIN (taper off, split exact) vs RN / RS / CLIP / UP")
    print(hdr)
    for rn, rs, cl, up in ((1024, 4, 3.0, 1), (1024, 2, 3.0, 1),
                           (2048, 4, 3.0, 1), (1024, 8, 3.0, 1),
                           (1024, 4, 2.5, 1), (1024, 4, 4.0, 1),
                           (1024, 4, 3.0, 2)):
        t0 = time.time()
        e2, R2, d2, _P2 = C.chain_a(n=rn, rs=rs)
        res, _w = FI.run_chain(post, e2, R2, d2, L, M, rs, 'off', 0)
        a = FI.chain_launch(res, L, M, 9999, cl, up, post, BACK, 'exact')
        v = score(*a[:7], a[7], xci, yci)
        print(f"  {f'RN={rn} RS={rs} CLIP={cl} UP={up}':34s} {v[0]:9.4f} "
              f"{v[1]:9.4f} {v[2]:9.4f} {v[3]:8.3f}   "
              f"[{time.time() - t0:.0f}s]", flush=True)

    print("\nREADOUT LATTICE (both arms, NL=321 / RN=1024 RS=4): "
          "the area mask must be lattice-independent")
    print(hdr)
    for dxo, nout in ((0.4e-6, 61), (0.2e-6, 121), (0.1e-6, 241)):
        for which in ('oracle', 'chain'):
            t0 = time.time()
            if which == 'oracle':
                a = FI.oracle_launch(env, R, dx, L, M, 321, 3.0, 1, post, BACK)
            else:
                res, _w = FI.run_chain(post, env, R, dx, L, M, 4, 'off', 0)
                a = FI.chain_launch(res, L, M, 9999, 3.0, 1, post, BACK,
                                    'exact')
            v = score(*a[:7], a[7], xci, yci, dxo=dxo, nout=nout)
            print(f"  {f'{which} dx_out={dxo * 1e6:g}um n={nout}':34s} "
                  f"{v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f} {v[3]:8.3f}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
