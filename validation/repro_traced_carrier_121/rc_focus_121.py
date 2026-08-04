# D121 RESIDUAL CLOSURE -- is the residual a WAVEFRONT defect or a FOCUS
# OFFSET?
#
# The area-exact residual is ~1 % of the energy moved out of a 4 um circle
# into the 4-10 um annulus, with the FWHM matching to 0.02-0.06 um.  A small
# residual DEFOCUS does exactly that and nothing else: at design 121's exit NA
# 0.295 a **1.6 um** longitudinal offset is worth 0.0154 waves rms, which is
# precisely the equivalent sigma of a 0.93-point EE3 deficit.  A wavefront
# defect and a focus offset are indistinguishable at ONE plane and trivially
# distinguishable across a through-focus scan:
#
#   FOCUS OFFSET   both arms reach the SAME peak EE3, at different dz
#   WAVEFRONT      the chain's peak EE3 is lower than the oracle's, wherever
#                  each is put
#
# Both arms are scanned through the same planes.  ``rs_spot`` is called with
# the surface list built for the nominal pull-back and a Rayleigh-Sommerfeld
# distance of ``back + dz``, so ``dz`` moves the OBSERVATION plane and nothing
# else -- the launch, the trace and the tile centre are untouched.  EE is
# area-exact about each plane's own centroid.
#
# usage:  ORDERS='0,0 -1,0 -4,0 -4,-2' DZ='-8,8,17' python rc_focus_121.py
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
    orders = os.environ.get('ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    lo, hi, nz = os.environ.get('DZ', '-8,8,17').split(',')
    dzs = np.linspace(float(lo), float(hi), int(nz)) * 1e-6
    rn, rs, nlo, clip = 1024, 4, 321, 3.0
    dxo, nout = 0.4e-6, 61
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    summary = []
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
        print(f"  {'dz um':>7} {'orc EE3':>9} {'chn EE3':>9} {'resid':>8} "
              f"{'orc EE6':>9} {'chn EE6':>9} {'orc FWHM':>9} {'chn FWHM':>9}",
              flush=True)
        curves = {}
        for which in ('oracle', 'chain'):
            t0 = time.time()
            if which == 'oracle':
                a = FI.oracle_launch(env, R, dx, L, M, nlo, clip, 1, post,
                                     BACK)
            else:
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                a = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            e3, e6, fw = [], [], []
            for dz in dzs:
                r = H.rs_spot(*a[:7], BACK + float(dz), xci, yci,
                              dx_out=dxo, n_out=nout, nl=a[7])
                I = np.ascontiguousarray(r['I'])
                cx, cy = RD.centroid(I, r['ax'])
                e3.append(RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100)
                e6.append(RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100)
                fw.append(r['fwhm'] * 1e6)
            curves[which] = (np.array(e3), np.array(e6), np.array(fw))
            print(f"     [{which} {time.time() - t0:.0f}s]", flush=True)
        oe3, oe6, ofw = curves['oracle']
        ce3, ce6, cfw = curves['chain']
        for i, dz in enumerate(dzs):
            print(f"  {dz * 1e6:>7.2f} {oe3[i]:9.4f} {ce3[i]:9.4f} "
                  f"{oe3[i] - ce3[i]:8.4f} {oe6[i]:9.4f} {ce6[i]:9.4f} "
                  f"{ofw[i]:9.3f} {cfw[i]:9.3f}", flush=True)

        def _pk(v):
            """Sub-sample peak of the EE3(dz) curve by a 3-point parabola."""
            i = int(np.argmax(v))
            if 0 < i < v.size - 1:
                d = v[i - 1] - 2 * v[i] + v[i + 1]
                f = 0.0 if abs(d) < 1e-300 else 0.5 * (v[i - 1] - v[i + 1]) / d
                h = float(dzs[1] - dzs[0])
                return dzs[i] + f * h, v[i] - 0.25 * (v[i - 1] - v[i + 1]) * f
            return dzs[i], v[i]
        zo, po = _pk(oe3)
        zc, pc = _pk(ce3)
        summary.append((o, zo, po, zc, pc))
        print(f"  BEST FOCUS  oracle dz={zo * 1e6:+.3f} um EE3={po:.4f}   "
              f"chain dz={zc * 1e6:+.3f} um EE3={pc:.4f}   "
              f"|dz| shift {abs(zc - zo) * 1e6:.3f} um   "
              f"PEAK residual {po - pc:+.4f}", flush=True)
    print("\n\n================ BEST-FOCUS SUMMARY ================")
    print(f"{'order':>8} {'orc dz um':>10} {'orc EE3':>9} {'chn dz um':>10} "
          f"{'chn EE3':>9} {'dz shift':>9} {'peak resid':>11}")
    for o, zo, po, zc, pc in summary:
        print(f"{o:>8} {zo * 1e6:10.3f} {po:9.4f} {zc * 1e6:10.3f} "
              f"{pc:9.4f} {(zc - zo) * 1e6:9.3f} {po - pc:11.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
