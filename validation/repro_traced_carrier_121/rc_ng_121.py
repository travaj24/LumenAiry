# D121 RESIDUAL CLOSURE -- WHICH GROUP costs the residual?
#
# ``hybrid_localize_121``'s ``n_chain`` semantics: the chain propagates the
# first ``ng`` post-DOE groups, and the EXACT ray trace + Rayleigh-Sommerfeld
# integral finishes the remaining ``6 - ng`` groups plus the trailing leg.
# ``ng = 0`` IS the exact-ray oracle (with the DOE-plane residual phase carried,
# i.e. this campaign's true ceiling).  The step from ``ng`` to ``ng+1`` is
# therefore the cost of putting group ``ng+1`` through the chain instead of
# through exact rays -- with EVERYTHING downstream held identical.
#
# Two things differ from the campaign's own sweep and both matter:
#   * the launch is split against the EXACT niche-C5 eikonal (the converged
#     readout of D121_FINAL_CLOSURE S3.3), not the parabola;
#   * EE is scored with an AREA-EXACT circular mask, because the shipped hard
#     pixel mask carries a +-0.45-point quantisation that does NOT cancel
#     between arms (rc_score_121.py section 3).
#
# usage:  ORDERS='0,0 -1,0 -4,0' NGMAX=6 python rc_ng_121.py
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
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3
_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    orders = os.environ.get('ORDERS', '0,0 -1,0 -4,0').split()
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    nlo = int(os.environ.get('NLO', '321'))
    ngmax = int(os.environ.get('NGMAX', '6'))
    # launch-lattice decimation for the INTERMEDIATE planes.  At ng = 1 the
    # beam is 6.3 mm on a 51 um pitch, so CLIP=3 asks for 743^2 = 552 k rays
    # and the Rayleigh-Sommerfeld integral costs 12x what it does at ng = 6.
    # The envelope is band-limited, so a stride sweep must be FLAT -- which is
    # asserted by running the same order at NL=9999 and comparing (below).
    nlc = int(os.environ.get('NL', '9999'))
    dxo, nout, clip = 0.4e-6, 61, 3.0
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    store = {}
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
        print(f"\n########## ORDER ({m:+d},{n:+d}) ##########", flush=True)
        print(f"  {'ng':>3} {'EE3 area':>9} {'d(EE3)':>8} {'EE6 area':>9} "
              f"{'d(EE6)':>8} {'FWHM um':>8} {'env step':>9} {'rays':>7}",
              flush=True)
        prev3 = prev6 = None
        for ng in range(0, ngmax + 1):
            t0 = time.time()
            if ng == 0:
                (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
                 st) = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1,
                                        post, BACK)
                seen = {}
            else:
                res, seen = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0,
                                         ngroups=ng)
                (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
                 st) = FI.chain_launch(res, L, M, nlc, clip, 1, post, BACK,
                                       'exact', ngroups=ng)
            r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, BACK, xci, yci,
                          dx_out=dxo, n_out=nout, nl=nl)
            I = np.ascontiguousarray(r['I'])
            cx, cy = RD.centroid(I, r['ax'])
            e3 = RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100
            e6 = RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100
            store[f"I_{o}_{ng}"] = I
            store[f"ax_{o}_{ng}"] = r['ax']
            store[f"c_{o}_{ng}"] = np.array([cx, cy])
            d3 = '' if prev3 is None else f"{e3 - prev3:+8.4f}"
            d6 = '' if prev6 is None else f"{e6 - prev6:+8.4f}"
            prev3, prev6 = e3, e6
            print(f"  {ng:>3} {e3:9.4f} {d3:>8} {e6:9.4f} {d6:>8} "
                  f"{r['fwhm'] * 1e6:8.3f} {st:9.4f} {r['n_rays']:7d}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            for t, k in seen.items():
                print(f"        [warn x{k}] {t}", flush=True)
    fn = os.path.join(_HERE, "_rc_ng.npz")
    np.savez_compressed(fn, **store)
    print(f"\nsaved {fn}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
