# Niche C6: is the leftover error the MODEL or the IMPLEMENTATION?
#
# probe_c6_element.py finds a non-monotone degree response: the exit wavefront
# error against the exact-ray oracle falls to 0.0074 waves at fit degree 3 and
# then RISES to a ~0.014-wave plateau at degree 4-6, even though the model's
# own slope residual keeps improving (3.0e-4 -> 1.0e-4 -> 8.5e-5 rad).  A
# monotone theory cannot produce that, so one of the two legs must be measured
# separately.  Three phases of the SAME exit nodes:
#
#   ELEMENT   arg E_out                                     (what ships)
#   MODEL-A   k0*[W + a_fit + OPL](x_fit*) + arg resid_leftover(x_fit*)
#             -- what the C6 launch INTENDS: the exact a_fit-congruence,
#             Newton-inverted exactly, with the leftover residual evaluated by
#             band-limited interpolation at that congruence's own foot
#   ORACLE-B  k0*[W + a + OPL](x_*)                          (the exact-ray
#             oracle: rays along the TOTAL grad(W + a))
#
#   ELEMENT - MODEL-A   the IMPLEMENTATION (fits, Newton, OPL upsample,
#                       phasor sampling)
#   MODEL-A - ORACLE-B  the MODEL (what the polynomial a_fit does not capture)
#
# usage:  ORD=-4,-2 DEG=2,3,4,6 python probe_c6_split.py
import os
import sys
import time
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
import probe_c6_element as E6
import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_traced import (TiltedCarrier,
                                             _input_beam_amp_radius)

LAM = P.LAM
K0 = P.K0


class _FitResid(object):
    """``rf``-shaped wrapper around the library's OWN residual-eikonal model,
    so wfe_probe_remap's exact machinery can trace and invert the a_fit
    congruence with no reimplementation."""

    def __init__(self, eik):
        self.eik = eik

    def ev(self, x, y):
        a, gx, gy = self.eik._eval(np.asarray(x), np.asarray(y))
        return a, gx, gy


def main():
    warnings.filterwarnings('ignore')
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    rs = int(os.environ.get('RS', '4'))
    half = int(os.environ.get('HALF', '96'))
    thresh = float(os.environ.get('THRESH', '0.02'))
    up = int(os.environ.get('UP', '8'))
    degs = [int(v) for v in os.environ.get('DEG', '2,3,4,6').split(',')]

    E_in, _Eo, carv, dx = E6.get_call(m, n, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _mm = P.carrier_parts(car, Xg, Yg)
    resid = np.asarray(E_in) * np.exp(-1j * K0 * W)
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    rf = RM.ResidualField(resid, dx, car.x0, car.y0, 3.4 * w, up=up)

    xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car, surfs)
    ic = int(round(float(xo[0]) / dx + N / 2))
    jc = int(round(float(yo[0]) / dx + N / 2))
    sx = slice(max(ic - half, 0), min(ic + half + 1, N))
    sy = slice(max(jc - half, 0), min(jc + half + 1, N))
    Xp, Yp = Xg[sy, sx], Yg[sy, sx]
    t = np.linspace(-2.6 * w, 2.6 * w, 21)
    U, V = np.meshgrid(t, t)
    gxo, gyo, _q, _q2, _q3, alv = P.trace_forward(
        (U + car.x0).ravel(), (V + car.y0).ravel(), car, surfs)
    gg = alv & np.isfinite(gxo)
    A3 = np.stack([np.ones(int(gg.sum())), gxo[gg], gyo[gg]], axis=1)
    cxx = np.linalg.lstsq(A3, (U + car.x0).ravel()[gg], rcond=None)[0]
    cyy = np.linalg.lstsq(A3, (V + car.y0).ravel()[gg], rcond=None)[0]
    B3 = np.stack([np.ones(Xp.size), Xp.ravel(), Yp.ravel()], axis=1)
    guess = ((B3 @ cxx).reshape(Xp.shape), (B3 @ cyy).reshape(Xp.shape))
    exB = RM.invert_total(Xp, Yp, car, surfs, rf, guess)
    phiB = exB['phi']

    print(f"order ({m},{n})  w {w*1e3:.4f} mm  dec "
          f"{np.hypot(car.x0, car.y0)/w:.3f} w   patch "
          f"{Xp.shape[0]}x{Xp.shape[1]}")
    print()
    hdr = (f"{'deg':>5} {'g resid':>9} {'ELEM-MODELA':>12} {'MODELA-ORCL':>12} "
           f"{'ELEM-ORACLE':>12} {'nnA p99':>8} {'nnB p99':>8} {'npix':>7}")
    print(hdr)
    print('-' * len(hdr))
    for deg in [0] + degs:
        t0 = time.time()
        if deg == 0:
            F, dg, _fl = E6.element(E_in, presc, dx, car, rs, False)
            lbl = 'OFF'
            eik = None
        else:
            F, dg, _fl = E6.element(E_in, presc, dx, car, rs, True, deg=deg)
            lbl = str(deg)
            old = (LT._REMAP_RESID_EIKONAL_DEGREE, LT._REMAP_RESID_DEGREE_CAP)
            LT._REMAP_RESID_EIKONAL_DEGREE = deg
            LT._REMAP_RESID_DEGREE_CAP = max(deg, LT._REMAP_RESID_DEGREE_CAP)
            try:
                # ray_fit_radius: the element is called with
                # fit_radius_beam_factor=2.0 (probe_c6_element.py), and since
                # 2026-07-31 that disc sets the model's RADIAL FREEZE circle
                # (_REMAP_RESID_FREEZE_MARGIN).  Omit it and this script's
                # MODEL-A leg silently freezes at a different radius from the
                # element's, which is exactly the mismatch it exists to rule out.
                eik = LT._fit_residual_eikonal(
                    E_in, W, LAM, dx, dx, (car.x0, car.y0), w, stride=rs,
                    ray_fit_radius=2.0 * w)
            finally:
                (LT._REMAP_RESID_EIKONAL_DEGREE,
                 LT._REMAP_RESID_DEGREE_CAP) = old
        Ep = F[sy, sx]
        amp = np.abs(Ep)
        pk = float(amp.max())
        keep = exB['ok'] & (amp > thresh * pk) & (exB['resid'] < 1e-9)
        if eik is None:
            # MODEL-A degenerates to the shipped model: the carrier congruence
            # with the FULL residual pasted at its foot
            exA = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=guess,
                                         n_iter=16)
            phiA = exA['phi'] + rf.phase(exA['xe'], exA['ye'])
            keepA = keep & exA['ok'] & (exA['resid'] < 1e-9)
        else:
            rfa = _FitResid(eik)
            exA = RM.invert_total(Xp, Yp, car, surfs, rfa, guess)
            # leftover residual (a - a_fit) at the a_fit congruence's own foot
            lo = rf.phase(exA['xe'], exA['ye']) - K0 * eik.value(exA['xe'],
                                                                 exA['ye'])
            phiA = exA['phi'] + lo
            keepA = keep & exA['ok'] & (exA['resid'] < 1e-9)
        rA = P.local_wfe(Ep, phiA, keepA, Xp, Yp)
        rB = P.local_wfe(Ep, phiB, keepA, Xp, Yp)
        zAB = np.exp(1j * (phiA - phiB))
        rAB = P.local_wfe(np.abs(Ep) * zAB, np.zeros_like(phiA), keepA,
                          Xp, Yp)
        print(f"{lbl:>5} "
              f"{dg.get('grad_a_residual_rms', float('nan')):>9.6f} "
              f"{rA['sigma_fid_waves']:>12.5f} {rAB['sigma_fid_waves']:>12.5f} "
              f"{rB['sigma_fid_waves']:>12.5f} {rA['nn_step_p99']:>8.4f} "
              f"{rB['nn_step_p99']:>8.4f} {rB['n_pix']:>7d}   "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print()
    print("All in waves, unwrap-free (piston+tilt-removed coherent fidelity).")
    print("ELEM-MODELA isolates the IMPLEMENTATION; MODELA-ORCL isolates the "
          "MODEL.")


if __name__ == '__main__':
    sys.exit(main())
