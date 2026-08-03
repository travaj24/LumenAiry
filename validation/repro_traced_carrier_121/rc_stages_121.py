# D121 RESIDUAL CLOSURE -- the chain's OWN per-group diagnostics, per order.
#
# The residual left after the area-exact mask is smooth but NOT monotone in
# |order|: it is ~0 on axis, jumps to its maximum at the SMALLEST non-zero
# tilt, and then falls as the tilt grows.  Anything that explains it has to
# have that shape.  This dumps every per-stage number the chain already
# computes -- grid pitch, chief-ray decentre, gap NA, the dropped quartic sag,
# the decentred-fit fraction, the tilt-ramp per-pixel step -- for each order,
# so a quantity with that shape can be looked for rather than guessed at.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_stages_121.py
import os
import sys
import warnings

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402

LAM = C.LAM
K0 = 2.0 * np.pi / LAM


def main():
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    keys = None
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        L, M = m * LAM / period, n * LAM / period
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            res, seen = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
        st = [s for s in res.stages if not s.get('target')]
        print(f"\n########## ORDER ({m:+d},{n:+d})  "
              f"L,M = {L * 1e3:+.3f},{M * 1e3:+.3f} mrad ##########",
              flush=True)
        if keys is None:
            keys = sorted({k for s in st for k in s})
            print(f"  stage keys: {keys}\n", flush=True)
        hdr = (f"  {'g':>2} {'dx um':>8} {'R_in mm':>11} {'R_out mm':>11} "
               f"{'x_c um':>10} {'y_c um':>10} {'gap NA':>8} "
               f"{'phi_drop':>9} {'tiltstep':>9} {'w mm':>7} {'dec/w':>7}")
        print(hdr, flush=True)
        for i, s in enumerate(st):
            d = float(s.get('dx', np.nan))
            xc = float(s.get('x_c_out', 0.0))
            yc = float(s.get('y_c_out', 0.0))
            Lo = float(s.get('L_out', L))
            Mo = float(s.get('M_out', M))
            na = float(s.get('gap_na', np.nan))
            pd = float(s.get('gap_phi_drop', np.nan))
            w = float(s.get('w_out', s.get('w', np.nan)))
            step = K0 * np.hypot(Lo, Mo) * d
            print(f"  {i:>2} {d * 1e6:8.3f} "
                  f"{float(s.get('R_in', np.nan)) * 1e3:11.4f} "
                  f"{float(s.get('R_out', np.nan)) * 1e3:11.4f} "
                  f"{xc * 1e6:10.2f} {yc * 1e6:10.2f} {na:8.4f} "
                  f"{pd:9.4f} {step:9.4f} {w * 1e3:7.4f} "
                  f"{(np.hypot(xc, yc) / w if w > 0 else np.nan):7.4f}",
                  flush=True)
        for t, k in seen.items():
            print(f"     [warn x{k}] {t}", flush=True)
        # the FULL dict of the last stage, once, so nothing is missed
        if o == orders[0]:
            print("\n  --- last stage, verbatim ---", flush=True)
            for k in sorted(st[-1]):
                v = st[-1][k]
                if isinstance(v, (int, float, np.floating)):
                    print(f"      {k:28s} {v!r}", flush=True)
                else:
                    print(f"      {k:28s} <{type(v).__name__}>", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
