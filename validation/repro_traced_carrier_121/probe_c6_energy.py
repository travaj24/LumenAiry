# Niche C6 adversarial check: WHERE does the exit power go under the
# stationary-phase launch, and does the augmented ray map FOLD?
#
# probe_c6_element.py measured the exit phase error falling 0.0659 -> 0.0138
# waves at order (-4,-2) but ALSO an exit power rising 0.9962 -> 1.0437 of the
# input.  A ray-density amplitude cannot manufacture power unless the map
# folds (two entrance points on one exit pixel) or the polynomial model is
# being extrapolated somewhere it has no business being.  This script prints
# the element's own warnings, the radial distribution of the exit power, and
# a taper / degree sensitivity sweep.
#
# usage:  ORD=-4,-2 python probe_c6_energy.py
import os
import sys
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import probe_c6_element as E6
import lumenairy.elements._lens_traced as LT
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import (TiltedCarrier,
                                             _input_beam_amp_radius)

LAM = P.LAM
K0 = P.K0


def run(E_in, presc, dx, car, rs, flag, deg=None, tin=None, tout=None):
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
           LT._REMAP_RESID_TAPER_IN, LT._REMAP_RESID_TAPER_OUT)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = flag
    if deg is not None:
        LT._REMAP_RESID_EIKONAL_DEGREE = int(deg)
    if tin is not None:
        LT._REMAP_RESID_TAPER_IN = float(tin)
    if tout is not None:
        LT._REMAP_RESID_TAPER_OUT = float(tout)
    d = {}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                ray_subsample=rs, _remap_launch_out=d, **E6.OPTS))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
         LT._REMAP_RESID_TAPER_IN, LT._REMAP_RESID_TAPER_OUT) = old
    msgs = {}
    for w in wl:
        t = str(w.message)[:110]
        msgs[t] = msgs.get(t, 0) + 1
    return out, d, msgs


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    rs = int(os.environ.get('RS', '4'))
    E_in, _Eo, carv, dx = E6.get_call(m, n, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    p_in = float((np.abs(E_in) ** 2).sum())
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car, surfs)
    Rr = np.hypot(Xg - float(xo[0]), Yg - float(yo[0]))

    print(f"order ({m},{n})  entrance w {w*1e3:.4f} mm  exit chief "
          f"({float(xo[0])*1e3:+.4f},{float(yo[0])*1e3:+.4f}) mm  dx "
          f"{dx*1e6:.4f} um")
    print()
    cases = [('OFF', dict(flag=False))]
    for d_ in [int(v) for v in os.environ.get('DEG', '2,4,6,8').split(',')]:
        cases.append((f'ON deg={d_}', dict(flag=True, deg=d_)))
    for ti, to in [(1.0, 1.5), (1.5, 2.5), (2.5, 4.0)]:
        cases.append((f'ON taper {ti}-{to} w',
                      dict(flag=True, tin=ti, tout=to)))
    ref = None
    for lbl, kw in cases:
        F, dg, msgs = run(E_in, presc, dx, car, rs, **kw)
        pw = np.abs(F) ** 2
        tot = float(pw.sum())
        if ref is None:
            ref = F
        # radial power distribution about the exit chief ray, in units of the
        # exit beam's own scale (use the 3 um-class core vs the far field)
        bins = [0.2e-3, 0.5e-3, 1.0e-3, 2.0e-3, 4.0e-3, 1e9]
        frac = []
        lo = 0.0
        for b in bins:
            frac.append(float(pw[(Rr >= lo) & (Rr < b)].sum()) / p_in)
            lo = b
        print(f"  {lbl:18s} P/Pin {tot/p_in:.5f}  deg "
              f"{str(dg.get('degree','-')):>2}  gres "
              f"{dg.get('grad_a_residual_rms', float('nan')):.6f}  "
              f"radial P/Pin " + " ".join(f"{v:.5f}" for v in frac))
        for t, c in msgs.items():
            print(f"        [warn x{c}] {t}")
    print()
    print("radial bins are |r - exit chief| < 0.2 / 0.5 / 1 / 2 / 4 / inf mm")


if __name__ == '__main__':
    sys.exit(main())
