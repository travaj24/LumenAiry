# D121 FINAL CLOSURE -- the per-order table, in ONE process.
#
# Runs, for every DOE order asked for:
#   * the exact-ray oracle WITHOUT the DOE-plane residual phase (CARRY=0) --
#     the ceiling every prior table in this campaign quotes;
#   * the exact-ray oracle WITH it (CARRY=1) -- the ceiling for the field the
#     chain is actually handed, converged in the launch density;
#   * the chain as shipped, read out as ``hybrid_localize_121`` reads it
#     (parabola split) -- the number the CHANGELOG quotes;
#   * the chain as shipped, read out against the EXACT eikonal (instrument);
#   * the chain with the conversion taper off, both readouts.
#
# Every arm goes through ONE readout (``hybrid_localize_121.rs_spot``).
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python fc_table_121.py
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    nlo = int(os.environ.get('NLO', '321'))
    back, nout, dxo, clip = 5.0e-3, 61, 0.4e-6, 3.0
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)

    arms = [('oracle CARRY=0', 'oracle', 0, None, None),
            ('oracle CARRY=1', 'oracle', 1, None, None),
            ('chain  taper=on   split=parabola', 'chain', 0, 'on',
             'parabola'),
            ('chain  taper=on   split=exact', 'chain', 0, 'on', 'exact'),
            ('chain  taper=off  split=parabola', 'chain', 0, 'off',
             'parabola'),
            ('chain  taper=off  split=exact', 'chain', 0, 'off', 'exact')]
    rows = {}
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
        print(f"\n########## ORDER ({m:+d},{n:+d})   "
              f"L,M = {L * 1e3:+.3f},{M * 1e3:+.3f} mrad ##########",
              flush=True)
        print(f"  {'arm':34s} {'EE3 %':>9} {'EE6 %':>9} {'FWHM um':>8} "
              f"{'env step':>9} {'RS p99.9':>9}", flush=True)
        for name, arm, carry, taper, split in arms:
            t0 = time.time()
            if arm == 'oracle':
                (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
                 st) = FI.oracle_launch(env, R, dx, L, M, nlo, clip, carry,
                                        post, back)
            else:
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, taper, 0)
                (x0, y0, amp, ph0, p, q, surfs, nl, h, w,
                 st) = FI.chain_launch(res, L, M, 9999, clip, 1, post, back,
                                       split)
            r = H.rs_spot(x0, y0, amp, ph0, p, q, surfs, back, xci, yci,
                          dx_out=dxo, n_out=nout, nl=nl)
            _, wp999, _ = FI._wrapped_step(x0, y0, amp, ph0, p, q, surfs,
                                           back, xci, yci, nl, nout, dxo)
            rows[(o, name)] = r
            print(f"  {name:34s} {r['ee3'] * 100:9.4f} {r['ee6'] * 100:9.4f} "
                  f"{r['fwhm'] * 1e6:8.3f} {st:9.4f} {wp999:9.4f}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        c0 = rows[(o, 'oracle CARRY=0')]['ee3'] * 100
        c1 = rows[(o, 'oracle CARRY=1')]['ee3'] * 100
        sh = rows[(o, 'chain  taper=on   split=parabola')]['ee3'] * 100
        si = rows[(o, 'chain  taper=on   split=exact')]['ee3'] * 100
        bo = rows[(o, 'chain  taper=off  split=exact')]['ee3'] * 100
        print(f"  DECOMPOSITION ({m:+d},{n:+d}):  shipped {sh:.3f} vs true "
              f"ceiling {c1:.3f}  = {c1 - sh:+.3f} points", flush=True)
        print(f"     ceiling correction (oracle carried the DOE residual): "
              f"{c1 - c0:+.3f}   [old ceiling {c0:.3f}]", flush=True)
        print(f"     INSTRUMENT (readout split parabola -> exact): "
              f"{si - sh:+.3f}", flush=True)
        print(f"     LIBRARY    (conversion taper on -> off):      "
              f"{bo - si:+.3f}", flush=True)
        print(f"     RESIDUAL   (chain best vs true ceiling):      "
              f"{c1 - bo:+.3f}", flush=True)

    print("\n\n================ SUMMARY (EE3 %) ================")
    hdr = ['ord', 'orc C0', 'orc C1', 'ship', 'ship+I', 'off+P', 'off+E',
           'gap0', 'gapF']
    print(('{:>8}' * len(hdr)).format(*hdr))
    for o in orders:
        v = [rows[(o, a[0])]['ee3'] * 100 for a in arms]
        print(f"{o:>8}" + ''.join(f"{x:8.3f}" for x in v)
              + f"{v[0] - v[2]:8.3f}{v[1] - v[5]:8.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
