# D121 RESIDUAL CLOSURE -- the residual as a CONTINUOUS function of tilt.
#
# The area-exact residual is 0.048 points on axis, 0.934 at the SMALLEST
# non-zero DOE order, and then falls monotonically to 0.305 at the largest.
# Six integer orders cannot tell a DISCONTINUITY (the chain's tilted branch
# costs a fixed overhead the moment ``L != 0``) from a FUNCTION OF TILT with a
# maximum somewhere below the first order.  Nothing in the chain requires the
# tilt to be a DOE order, so this sweeps it continuously in units of the
# first order's ``L1 = lambda / period = 11.516 mrad``:
#
#   f = 0 exactly     -- the untilted branch
#   f = 1e-4 .. 0.5   -- the tilted branch at a tilt far below any real order
#   f = 1, 2, 3, 4    -- the real orders, as a cross-check against rc_readout
#
# BOTH arms move with ``f`` (the oracle launches the same tilt), so the
# difference is always chain-vs-oracle at one geometry.  EE is area-exact.
#
# usage:  FRACS='0 0.0001 0.01 0.1 0.3 0.5 1 2 3 4' python rc_tilt_121.py
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


def main():
    fracs = [float(v) for v in os.environ.get(
        'FRACS', '0 0.0001 0.001 0.01 0.1 0.3 0.5 1 2 3 4').split()]
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    nlo = int(os.environ.get('NLO', '321'))
    dxo, nout, clip = 0.4e-6, 61, 3.0
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    L1 = LAM / period
    print(f"\nL1 = lambda/period = {L1 * 1e3:.4f} mrad;  "
          f"tilt applied as (L, M) = (-f*L1, 0)\n", flush=True)
    hdr = (f"  {'f':>8} {'L mrad':>9} {'oracle':>9} {'chain':>9} "
           f"{'residual':>9} {'orc EE6':>9} {'chn EE6':>9} {'res EE6':>9} "
           f"{'FWHM o':>7} {'FWHM c':>7}")
    print(hdr, flush=True)
    print('  ' + '-' * (len(hdr) - 2), flush=True)
    for f in fracs:
        t0 = time.time()
        L, M = -f * L1, 0.0
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        out = {}
        for which in ('oracle', 'chain'):
            if which == 'oracle':
                a = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1, post,
                                     BACK)
            else:
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                a = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=dxo, n_out=nout,
                          nl=a[7])
            I = np.ascontiguousarray(r['I'])
            cx, cy = RD.centroid(I, r['ax'])
            out[which] = (RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100,
                          RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100,
                          r['fwhm'] * 1e6)
        o, c = out['oracle'], out['chain']
        print(f"  {f:>8.4g} {L * 1e3:>9.4f} {o[0]:>9.4f} {c[0]:>9.4f} "
              f"{o[0] - c[0]:>9.4f} {o[1]:>9.4f} {c[1]:>9.4f} "
              f"{o[1] - c[1]:>9.4f} {o[2]:>7.3f} {c[2]:>7.3f}   "
              f"[{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
