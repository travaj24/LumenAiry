# THE LEG OR THE ELEMENT?  Splitting the hybrid's n=5 -> n=6 step in two.
# (DIAG_LAST_GROUP_DECENTRE_2026_07_30.)
#
# hybrid_localize_121.py's n = 5 hand-off is taken at group 4's EXIT vertex,
# and its oracle then does the GAP into group 5, group 5 itself, and the
# trailing leg -- all by exact rays.  Its n = 6 hand-off is at group 5's exit,
# so the chain has done the GAP (a wave transport) AND the element pass.  The
# 24-point step between them is therefore the cost of TWO operations, not one,
# and the scope doc attributes all of it to the element.
#
# This script inserts the missing intermediate point: the chain's OWN group-5
# ENTRANCE field (the exact array ``apply_real_lens_traced`` is handed, captured
# by monkeypatch), launched as exact rays through group 5 + the trailing leg.
#
#   A  chain groups 0..4, ORACLE does gap + group 5      (= hybrid n = 5)
#   B  chain groups 0..4 AND the gap, ORACLE does group 5   <-- the new point
#   C  chain groups 0..4, gap AND group 5                (= hybrid n = 6)
#
#   A -> B  is the COARSE LEG (wave transport across the gap)
#   B -> C  is the ELEMENT PASS
#
# usage:  ORD=-4,-2 NOUT=61 DXO=0.4 python wfe_probe_gap.py
import os
import sys
import time
import dataclasses

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
import wfe_probe_orders as OR
from hybrid_localize_121 import rs_spot
from lumenairy.elements._lens_traced import TiltedCarrier
from lumenairy.raytrace import RayBundle, Surface, trace

LAM = P.LAM
K0 = P.K0


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    nout = int(os.environ.get('NOUT', '61'))
    dxo = float(os.environ.get('DXO', '0.4')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    nlent = int(os.environ.get('NLENT', '221'))
    up = int(os.environ.get('UP', '8'))
    rn = int(os.environ.get('RN', '1024'))
    _pre, post, _g, period = C.geometry()
    print(f"post-DOE group gaps (mm): "
          + ", ".join(f"{i}:{g['gap_before']*1e3:.4f}"
                      for i, g in enumerate(post)))
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)

    # image-plane centre = the exact chief ray of the full post-DOE system
    L = m * LAM / period
    M = n * LAM / period
    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])

    # group 5 + trailing, launched AT the group-5 entrance vertex
    sf = list(surfs)
    sf[-1] = dataclasses.replace(sf[-1], thickness=float(C.TRAILING) - back)
    sf.append(Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                      glass_before='air', glass_after='air', is_mirror=False,
                      thickness=0.0, label='img'))

    # ---- B: the chain's OWN group-5 entrance field, oracle from there ------
    E_in, E_out, carv, dx = OR.get_call(m, n, rn=rn)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _mm = P.carrier_parts(car, Xg, Yg)
    resid = np.asarray(E_in) * np.exp(-1j * K0 * W)
    from lumenairy.elements._lens_traced import _input_beam_amp_radius
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    rf = RM.ResidualField(resid, dx, car.x0, car.y0, 3.4 * w, up=up)
    t = np.linspace(-3.0 * w, 3.0 * w, nlent)
    U, V = np.meshgrid(t, t)
    XE, YE = U + car.x0, V + car.y0
    Wa, La, Ma = P.carrier_parts(car, XE, YE)
    aa, gL, gM = rf.ev(XE, YE)
    amp = rf.amp_at(XE, YE)
    t0 = time.time()
    rB = rs_spot(XE.ravel(), YE.ravel(), amp.ravel(),
                 (K0 * (Wa + aa)).ravel(), (La + gL).ravel(),
                 (Ma + gM).ravel(), sf, back, xci, yci, dx_out=dxo,
                 n_out=nout, nl=nlent)
    print(f"\n  B  chain does groups 0..4 AND the gap; ORACLE does group 5 "
          f"+ trailing")
    print(f"     entrance w {w*1e3:.4f} mm, decentre "
          f"{np.hypot(car.x0, car.y0)*1e3:.4f} mm = "
          f"{np.hypot(car.x0, car.y0)/w:.3f} w,  {nlent}^2 rays, "
          f"RSstep p99.9 {rB['step_w']:.3f} cyc, live {rB['live']*100:.4f} %")
    print(f"     FWHM {rB['fwhm']*1e6:6.3f} um   EE3 {rB['ee3']*100:6.2f}   "
          f"EE6 {rB['ee6']*100:6.2f}   EE12 {rB['ee12']*100:6.2f}   "
          f"[{time.time()-t0:.0f}s]")

    # convergence of B in the launch density
    for nl2 in [int(v) for v in os.environ.get('NLSWEEP', '161,301').split(',')]:
        t2 = np.linspace(-3.0 * w, 3.0 * w, nl2)
        U2, V2 = np.meshgrid(t2, t2)
        X2, Y2 = U2 + car.x0, V2 + car.y0
        W2, L2, M2 = P.carrier_parts(car, X2, Y2)
        a2, g2, h2 = rf.ev(X2, Y2)
        r2 = rs_spot(X2.ravel(), Y2.ravel(), rf.amp_at(X2, Y2).ravel(),
                     (K0 * (W2 + a2)).ravel(), (L2 + g2).ravel(),
                     (M2 + h2).ravel(), sf, back, xci, yci, dx_out=dxo,
                     n_out=nout, nl=nl2)
        print(f"       [launch density {nl2}^2] EE3 {r2['ee3']*100:6.2f}  "
              f"FWHM {r2['fwhm']*1e6:6.3f} um  RSstep {r2['step_w']:.3f}")
    # clip sweep
    for cl in [float(v) for v in os.environ.get('CLIPS', '2.5,3.5').split(',')]:
        t2 = np.linspace(-cl * w, cl * w, nlent)
        U2, V2 = np.meshgrid(t2, t2)
        X2, Y2 = U2 + car.x0, V2 + car.y0
        W2, L2, M2 = P.carrier_parts(car, X2, Y2)
        a2, g2, h2 = rf.ev(X2, Y2)
        r2 = rs_spot(X2.ravel(), Y2.ravel(), rf.amp_at(X2, Y2).ravel(),
                     (K0 * (W2 + a2)).ravel(), (L2 + g2).ravel(),
                     (M2 + h2).ravel(), sf, back, xci, yci, dx_out=dxo,
                     n_out=nout, nl=nlent)
        print(f"       [clip {cl} w] EE3 {r2['ee3']*100:6.2f}  live "
              f"{r2['live']*100:.4f} %")
    # ---- A: the SAME machinery, but launched from GROUP 4's EXIT ---------
    # (hybrid n=5's hand-off plane).  Identical residual model, identical RS
    # integral, identical surfaces plus the 3.3233 mm gap -- so A -> B isolates
    # the CHAIN's wave transport across that gap and nothing else.
    import warnings as _wm
    import lumenairy as _la
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=4)
    with _wm.catch_warnings(record=True):
        _wm.simplefilter('always')
        res4 = _la.propagate_traced_carrier_chain(
            env_doe, [dict(g) for g in post[:5]], LAM, dx_doe,
            r_in=TiltedCarrier(R_doe, L, M), ray_subsample=4, n_workers=8,
            final_distance=0.0, final_leg='paraxial',
            on_decentred_fit='ignore')
    st4 = [q for q in res4.stages if not q.get('target')][-1]
    fld4 = np.asarray(res4.field)
    R4, dx4 = float(res4.R), float(res4.dx)
    L4 = float(st4.get('L_out', 0.0))
    M4 = float(st4.get('M_out', 0.0))
    xc4 = float(st4.get('x_c_out', 0.0))
    yc4 = float(st4.get('y_c_out', 0.0))
    # the chain returns the field in the CHIEF-RAY-TRACKING frame (beam on the
    # grid centre); the true transverse position is (xc4, yc4).
    car4 = TiltedCarrier(R4, L4, M4, 0.0, 0.0)
    n4 = fld4.shape[0]
    a4 = (np.arange(n4) - n4 / 2) * dx4
    X4, Y4 = np.meshgrid(a4, a4)
    W4g, _q1, _q2 = P.carrier_parts(car4, X4, Y4)
    resid4 = fld4 * np.exp(-1j * K0 * W4g)
    w4 = float(np.sqrt(2.0 * (np.abs(resid4) ** 2
                              * (X4 ** 2 + Y4 ** 2)).sum()
                       / max(float((np.abs(resid4) ** 2).sum()), 1e-300)))
    rf4 = RM.ResidualField(resid4, dx4, 0.0, 0.0, 3.4 * w4, up=up)
    sfA = [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                   glass_before='air', glass_after='air', is_mirror=False,
                   thickness=float(post[5]['gap_before']), label='gap')] + sf
    tA = np.linspace(-3.0 * w4, 3.0 * w4, nlent)
    UA, VA = np.meshgrid(tA, tA)
    WA, LA, MA = P.carrier_parts(car4, UA, VA)
    aA, gLA, gMA = rf4.ev(UA, VA)
    rA = rs_spot((UA + xc4).ravel(), (VA + yc4).ravel(),
                 rf4.amp_at(UA, VA).ravel(), (K0 * (WA + aA)).ravel(),
                 (LA + gLA).ravel(), (MA + gMA).ravel(), sfA, back, xci, yci,
                 dx_out=dxo, n_out=nout, nl=nlent)
    print(f"  A  chain does groups 0..4; ORACLE does the "
          f"{post[5]['gap_before']*1e3:.4f} mm GAP + group 5 + trailing")
    print(f"     hand-off w {w4*1e3:.4f} mm, chief ({xc4*1e3:+.4f},"
          f"{yc4*1e3:+.4f}) mm, dx {dx4*1e6:.4f} um, {nlent}^2 rays, "
          f"RSstep p99.9 {rA['step_w']:.3f} cyc, live {rA['live']*100:.4f} %")
    print(f"     FWHM {rA['fwhm']*1e6:6.3f} um   EE3 {rA['ee3']*100:6.2f}   "
          f"EE6 {rA['ee6']*100:6.2f}   EE12 {rA['ee12']*100:6.2f}")
    print("  SPLIT (same machinery on both sides):")
    print(f"     A (oracle does the gap)   EE3 {rA['ee3']*100:6.2f}")
    print(f"     B (chain does the gap)    EE3 {rB['ee3']*100:6.2f}"
          f"   -> COARSE LEG costs {(rB['ee3']-rA['ee3'])*100:+6.2f} points")
    print()
    print("Compare with hybrid_localize_121.py:")
    print("  A (n=5, hand-off at group 4's EXIT, oracle does gap+group5) and")
    print("  C (n=6, chain does gap+group5).")
    print("  A -> B is the COARSE LEG across the gap; B -> C is the ELEMENT.")


if __name__ == '__main__':
    sys.exit(main())
