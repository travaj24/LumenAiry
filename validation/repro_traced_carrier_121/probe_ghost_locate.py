# Niche C6 ON-AXIS GHOST -- WHERE is it, and does any real ray reach there?
#
# probe_ghost_c6.py establishes THAT the ghost exists (1.03e-3 of Pin beyond
# 4 mm at 33 % of peak amplitude, on the shipped concentric hard-mask branch at
# residual degree 4) and that it is a FIT-CONDITIONING effect -- it moves with
# the fit disc radius, the restriction method, the fit order and the D1 weight
# constant.  This script asks the one question that decides whether it is a
# FOLD of the ray map (two entrance points on one exit pixel -- a real, if
# badly modelled, physical branch) or pure EXTRAPOLATION of the fitted inverse
# (no ray reaches there at all):
#
#   trace the a_fit congruence EXACTLY over the whole launch square and measure
#   the traced exit HULL.  Any returned power outside that hull is power the
#   element manufactured -- the polynomial forward map was inverted where its
#   own data says nothing lands.
#
# The exact trace is wfe_probe_remap.trace_total with the LIBRARY's own fitted
# residual eikonal (no reimplementation of the model).
#
# usage:  ORDERS='0,0' DEG=4 python probe_ghost_locate.py
import os
import sys
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
import probe_c6_element as E6
import probe_ghost_c6 as G
import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_traced import (TiltedCarrier,
                                             _input_beam_amp_radius)

LAM = P.LAM
K0 = P.K0


class _FitResid(object):
    """``rf``-shaped wrapper around the library's own residual-eikonal model
    (same shim probe_c6_split.py uses)."""

    def __init__(self, eik):
        self.eik = eik

    def ev(self, x, y):
        a, gx, gy = self.eik._eval(np.asarray(x), np.asarray(y))
        return a, gx, gy


def fit_eik(E_in, W, dx, car, w, rs, deg):
    old = (LT._REMAP_RESID_EIKONAL_DEGREE, LT._REMAP_RESID_DEGREE_CAP)
    LT._REMAP_RESID_EIKONAL_DEGREE = int(deg)
    LT._REMAP_RESID_DEGREE_CAP = max(int(deg), LT._REMAP_RESID_DEGREE_CAP)
    try:
        # ray_fit_radius: the element's own ray-fit disc
        # (fit_radius_beam_factor=2.0), which since 2026-07-31 sets the model's
        # RADIAL FREEZE circle -- see ``_REMAP_RESID_FREEZE_MARGIN``.
        return LT._fit_residual_eikonal(E_in, W, LAM, dx, dx,
                                        (car.x0, car.y0), w, stride=rs,
                                        ray_fit_radius=2.0 * w)
    finally:
        (LT._REMAP_RESID_EIKONAL_DEGREE, LT._REMAP_RESID_DEGREE_CAP) = old


def main():
    warnings.filterwarnings('ignore')
    m, n = (int(v) for v in os.environ.get('ORDERS', '0,0').split(','))
    deg = int(os.environ.get('DEG', '4'))
    rs = int(os.environ.get('RS', '4'))
    nl = int(os.environ.get('NLAUNCH', '257'))

    E_in, _Eo, carv, dx = E6.get_call(m, n, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _mm = P.carrier_parts(car, Xg, Yg)
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    p_in = float((np.abs(E_in) ** 2).sum())
    xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car, surfs)
    xc_e, yc_e = float(xo[0]), float(yo[0])
    Rr = np.hypot(Xg - xc_e, Yg - yc_e)

    # ---- the TRUE exit hull of the a_fit congruence over the launch square --
    # The element launches on the coarse lattice spanning the whole grid; trace
    # that same square exactly.
    eik = fit_eik(E_in, W, dx, car, w, rs, deg)
    rfa = _FitResid(eik)
    t = np.linspace(ax[0], ax[-1], nl)
    U, V = np.meshgrid(t, t)
    for lbl, rr in (('grad(W) only', None), ('grad(W + a_fit)', rfa)):
        exo, eyo, _psi, _L, _M, alive = RM.trace_total(U.ravel(), V.ravel(),
                                                       car, surfs, rr)
        rad = np.hypot(exo - xc_e, eyo - yc_e)
        ok = alive & np.isfinite(rad)
        # the hull that MATTERS is the one the beam illuminates: restrict the
        # launch disc to where the input actually carries power
        rin = np.hypot(U.ravel() - car.x0, V.ravel() - car.y0)
        for flab, sel in (('whole launch square', ok),
                          ('launch r <= 2 w', ok & (rin <= 2.0 * w)),
                          ('launch r <= 3 w', ok & (rin <= 3.0 * w))):
            print("  exit radius, %-16s %-20s  max %8.4f mm   p99.9 "
                  "%8.4f mm   alive %d/%d"
                  % (lbl, flab, rad[sel].max() * 1e3,
                     np.percentile(rad[sel], 99.9) * 1e3,
                     int(sel.sum()), rad.size))
    print()

    cases = [('C6 OFF          mask  o6', dict(flag=False)),
             ('C6 deg%d         mask  o6' % deg, dict(flag=True, deg=deg)),
             ('C6 deg%d         wght  o10' % deg,
              dict(flag=True, deg=deg, force_dec=True)),
             ('C6 deg%d  mask o6  inv=fit' % deg,
              dict(flag=True, deg=deg, inversion_method='fit')),
             ('C6 OFF    mask o6  inv=fit',
              dict(flag=False, inversion_method='fit'))]
    hdr = ("%-28s %8s %10s %10s %10s %10s %10s" %
           ('config', 'P/Pin', 'P(r>4mm)', 'r_min/mm', 'r_max/mm',
            'r_peak/mm', 'amp/peak'))
    print(hdr)
    print('-' * len(hdr))
    for lbl, kw in cases:
        try:
            F, _dg, _fl = G.element(E_in, presc, dx, car, rs, **kw)
        except Exception as exc:                                # noqa: BLE001
            print("%-28s FAILED %s: %s" % (lbl, type(exc).__name__,
                                           str(exc)[:60]))
            continue
        pw = np.abs(F) ** 2
        far = (Rr > 4.0e-3) & np.isfinite(pw)
        gp = float(pw[far].sum()) / p_in
        if gp > 1e-9:
            sel = far & (pw > 1e-6 * pw.max())
            rsel = Rr[sel]
            ipk = np.argmax(np.where(far, pw, 0.0))
            iy, ix = np.unravel_index(ipk, pw.shape)
            print("%-28s %8.5f %10.3e %10.4f %10.4f %10.4f %10.3e"
                  % (lbl, float(pw.sum()) / p_in, gp, rsel.min() * 1e3,
                     rsel.max() * 1e3, float(Rr[iy, ix]) * 1e3,
                     float(np.abs(F[iy, ix]) / np.abs(F).max())))
        else:
            print("%-28s %8.5f %10.3e %10s %10s %10s %10s"
                  % (lbl, float(pw.sum()) / p_in, gp, '-', '-', '-', '-'))
    print()
    print("If r_min of the ghost exceeds the traced exit hull above, no ray "
          "of the a_fit congruence reaches it: the power is manufactured by "
          "inverting the FITTED map outside its own data.")


if __name__ == '__main__':
    sys.exit(main())
