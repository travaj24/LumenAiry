# Conservation audit (2026-07-31): the DEFENSIBLE halo radius, and the
# sampling-adequacy statement for every field the audit measures.
#
# WHY.  A halo metric "power beyond N mm" is worthless unless N is derived from
# the optics rather than chosen.  This script derives it: it traces design
# 121's last group EXACTLY (``wfe_probe_common.trace_forward``, the same
# Zemax-validated skew trace the element itself uses, on the element's own
# launch geometry) and asks, per DOE order, HOW FAR OUT the illuminated pupil
# can land.  Any returned power beyond that radius cannot be traced to
# illuminated input -- it is manufactured, not merely misplaced.
#
# Reported per order:
#   * the exit radius containing 1 - eps of the LAUNCHED power, for
#     eps = 1e-3 .. 1e-12, from the exact trace of the element's own input
#     amplitude (so the radius is weighted by where the light actually is);
#   * the hull of the launch disc at r <= 2w and r <= 3w (2w is the ray-fit
#     disc, ``fit_radius_beam_factor=2.0``; 3w is the e^-9 amplitude contour);
#   * the exit NA against this grid's Nyquist angle, and the fraction of exit
#     POWER above it -- the number that says whether the far-halo RADII are
#     trustworthy after propagation;
#   * the exit field's own amplitude-weighted wrapped nearest-neighbour phase
#     step at p99.9 against pi -- the sampling-adequacy statement, a p99.9 and
#     never a max (SCOPE S8 artefact 5(a), DIAG artefact 4).
#
# LOCAL-ONLY, read-only on the library.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _d121_common as C                                          # noqa: E402
import probe_c6_element as E6                                     # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
from lumenairy.elements._lens_traced import (TiltedCarrier,       # noqa: E402
                                             _input_beam_amp_radius)

LAM = C.LAM


def nn_step_p999(E, weight_floor=0.0):
    """Amplitude-weighted wrapped nearest-neighbour phase step, p99.9 [rad].

    The weight of a step is the SMALLER of the two pixels' powers, so a step
    across a pixel carrying no light cannot set the statistic.  p99.9, never a
    max."""
    ph = np.angle(E)
    w = np.abs(E) ** 2
    dxs = np.abs(np.angle(np.exp(1j * (ph[:, 1:] - ph[:, :-1])))).ravel()
    dys = np.abs(np.angle(np.exp(1j * (ph[1:, :] - ph[:-1, :])))).ravel()
    wx = np.minimum(w[:, 1:], w[:, :-1]).ravel()
    wy = np.minimum(w[1:, :], w[:-1, :]).ravel()
    v = np.concatenate([dxs, dys])
    ww = np.concatenate([wx, wy])
    keep = ww > weight_floor * ww.max()
    v, ww = v[keep], ww[keep]
    if v.size == 0:
        return float('nan')
    o = np.argsort(v)
    cw = np.cumsum(ww[o])
    cw = cw / cw[-1]
    return float(v[o][np.searchsorted(cw, 0.999)])


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -4,0 -4,-2').split()]
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))
    nl = int(os.environ.get('NL', '401'))

    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)

    print("EXACT TRACED EXIT HULL of design 121's LAST GROUP, per DOE order.")
    print("  Launch geometry is the element's own: the captured chain carrier "
          "(R, L, M, x0, y0) and input amplitude.")
    print(f"  Launch lattice {nl}x{nl} over the full co-moving grid.")
    print()
    for (m, n) in orders:
        E_in, _Eo, carv, dx = E6.get_call(m, n, rn=rn, rs=rs)
        car = TiltedCarrier(*carv)
        N = E_in.shape[0]
        w = float(_input_beam_amp_radius(E_in, dx, dx,
                                         centre=(car.x0, car.y0)))
        half = (N / 2) * dx
        t = np.linspace(-half, half, nl)
        U, V = np.meshgrid(t, t)
        xo, yo, _psi, _L, _M, alive = P.trace_forward(U.ravel(), V.ravel(),
                                                      car, surfs)
        # the element's own input amplitude at each launch point
        fx = U.ravel() / dx + N / 2.0
        fy = V.ravel() / dx + N / 2.0
        ii = np.clip(np.rint(fx).astype(int), 0, N - 1)
        jj = np.clip(np.rint(fy).astype(int), 0, N - 1)
        a = np.abs(E_in)[jj, ii]
        # exit chief ray
        xc, yc, _q1, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0],
                                                     car, surfs)
        re = np.hypot(xo - float(xc[0]), yo - float(yc[0]))
        ok = alive & np.isfinite(re)
        pw = (a ** 2) * ok
        tot = float(pw.sum())
        o = np.argsort(re[ok])
        cum = np.cumsum(pw[ok][o]) / tot
        print(f"order ({m},{n})  w {w * 1e3:.4f} mm  dec "
              f"{np.hypot(car.x0, car.y0) / w:.3f} w  exit chief "
              f"({float(xc[0]) * 1e3:+.4f},{float(yc[0]) * 1e3:+.4f}) mm  "
              f"live {ok.mean() * 100:.2f} %")
        print("   exit radius containing 1-eps of the LAUNCHED power:")
        for eps in (1e-3, 1e-6, 1e-9, 1e-12):
            i = np.searchsorted(cum, 1.0 - eps)
            i = min(i, re[ok][o].size - 1)
            print(f"     eps {eps:>7.0e} -> {re[ok][o][i] * 1e3:8.4f} mm")
        # THE ORACLE REFERENCE FOR THE HALO METRIC.  The exact trace of the
        # element's own input amplitude says how much power CAN legitimately
        # reach each shell.  Any returned gN above this is manufactured.
        print("   exact-ray prediction of the halo shells (launched power "
              "landing beyond N mm):")
        for rr in (1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3):
            print(f"     g{rr * 1e3:.0f} <= {float(pw[re > rr].sum()) / tot:.4e}")
        # and the exact-ray SECOND MOMENT -- the r2m reference.  Ray optics
        # only; at r_rms ~ 0.84 mm on a ~1.2 mm exit beam the diffractive
        # correction is far below the 1.7 % clean spread this is compared to.
        # ARTEFACT GUARD: dead rays come back with NaN coordinates at the
        # tilted orders, and ``0.0 * NaN`` is NaN, so the naive weighted sum
        # silently returned NaN at (-4,0)/(-4,-2) while the shell sums (which
        # compare ``re > r``, False for NaN) were unaffected.  Restrict to
        # finite radii explicitly; the excluded rays carry zero weight.
        _f = np.isfinite(re)
        _tot_f = float(pw[_f].sum())
        _r2 = float(np.sqrt((pw[_f] * re[_f] ** 2).sum() / max(_tot_f, 1e-300)))
        _c = _f & (re < 1.0e-3)
        _pc = float(pw[_c].sum())
        _r2c = float(np.sqrt((pw[_c] * re[_c] ** 2).sum() / max(_pc, 1e-300)))
        print(f"     [finite-radius weight retained {_tot_f / tot:.9f} "
              f"of launched power]")
        print(f"     r_rms(exact ray) {_r2 * 1e3:.4f} mm   "
              f"core(<1mm) {_r2c * 1e3:.4f} mm")
        for f in (1.0, 2.0, 3.0, 4.0):
            d = np.hypot(U.ravel() - car.x0, V.ravel() - car.y0) <= f * w
            sel = ok & d
            if sel.any():
                print(f"   hull of the launch disc r <= {f:.0f}w "
                      f"({f * w * 1e3:6.3f} mm): exit max "
                      f"{re[sel].max() * 1e3:8.4f} mm   p99.9 "
                      f"{np.percentile(re[sel], 99.9) * 1e3:8.4f} mm")
        # exit NA vs this grid's Nyquist, power-weighted
        na = np.sqrt(_L ** 2 + _M ** 2)
        na_ny = LAM / (2.0 * dx)
        wt = pw
        frac = float(wt[(na > na_ny) & ok].sum()) / tot
        print(f"   exit NA: max(amp>=e^-4) "
              f"{float(na[ok & (a >= np.exp(-4) * a.max())].max()):.4f}   "
              f"grid Nyquist NA {na_ny:.4f} (dx {dx * 1e6:.3f} um)   "
              f"power above Nyquist {frac:.4e}")
        print(f"   grid is {float(na[ok & (a >= np.exp(-4) * a.max())].max()) / na_ny:.1f}x "
              f"short of lambda/(2 NA_exit)")
        print(f"   SAMPLING: E_in  nn-step p99.9 {nn_step_p999(E_in):.4f} rad "
              f"(limit pi = {np.pi:.4f})")
        print(flush=True)


if __name__ == '__main__':
    sys.exit(main())
