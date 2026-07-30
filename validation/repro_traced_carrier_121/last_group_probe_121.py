# DECENTRE x TILT probe of design 121's LAST GROUP.
# (SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30, discriminating experiment.)
#
# ``hybrid_localize_121.py`` shows the whole per-order loss appears when the
# LAST group goes through ``apply_real_lens_traced`` (EE3 89.8 -> 65.8 at order
# (-4,-2); 89.9 -> 87.6 on axis), and nothing before it.  At that group the
# congruence carries BOTH a 1.78-beam-radius chief-ray decentre AND a 55 mrad
# entrance tilt.  This script separates them.
#
# For each (tilt, decentre) combination it builds the SAME input field --
# a Gaussian envelope of the measured entrance radius on the measured entrance
# carrier -- and pushes it through the last group two ways:
#
#   CHAIN  : propagate_traced_carrier_chain over that ONE group, i.e. exactly
#            the element pass the design-121 run performs, then an exact-ray +
#            Rayleigh-Sommerfeld transport of the chain's exit field to the
#            image plane (validated in hybrid_localize_121.py -- it reproduces
#            the shipped per-order EE3 to 0.02 points on axis).
#   ORACLE : the SAME input field launched as exact rays and traced through the
#            same surfaces, then the same Rayleigh-Sommerfeld integral.
#
# Every combination is a legitimate physical input field, so the chain/oracle
# gap is a clean measure of the ELEMENT's error as a function of decentre and
# tilt, with the input, the surfaces, the transport and the metric all held
# fixed.
#
# Env: DEC='0,0.5,1.0,1.778' (chief-ray offset in entrance beam radii, along
#      the design's own azimuth), TILTS='0,1' (0 = untilted, 1 = the measured
#      entrance tilt), N (grid), DXG (grid pitch um), NOUT, DXO, BACK, NL.
import os
import sys
import time
import warnings

import numpy as np

import _d121_common as C
from hybrid_localize_121 import _free_surfaces, rs_spot
from lumenairy.propagators.carrier import (_envelope_amp_radius,
                                           carrier_referenced_envelope)
from lumenairy.raytrace import RayBundle, trace
from exact_ray_oracle_121 import _phase_gradient

LAM = C.LAM
K0 = 2 * np.pi / LAM

# --- measured state at the entrance of design 121's last group, order (-4,-2),
# --- from the chain's own stage table (RN=1024, RS=4).  See the SCOPE doc.
R_IN = -21.139e-3
W_IN = 3.6225e-3
DX_G = 33.211e-6
L_IN, M_IN = 0.049073, 0.024537      # group-4 exit cosines = group-5 entrance


def build_input(n, dxg, w, R, L, M, x0, y0):
    u = (np.arange(n) - n / 2) * dxg
    U, V = np.meshgrid(u, u)
    env = np.exp(-(U ** 2 + V ** 2) / w ** 2).astype(np.complex128)
    return env, U, V


def oracle(env, U, V, dxg, R, L, M, x0, y0, surfs, back, xci, yci,
           nout, dxo, nl):
    """Exact-ray + RS for the analytic input field (sphere R about (x0,y0)
    plus tilt (L, M))."""
    n = env.shape[0]
    # decimate to ~nl across the beam
    w = _envelope_amp_radius(env, dxg, dxg)
    half = int(np.ceil(3.0 * w / dxg))
    c = n // 2
    i0, i1 = max(c - half, 0), min(c + half + 1, n)
    e = env[i0:i1, i0:i1]
    Uc, Vc = U[i0:i1, i0:i1], V[i0:i1, i0:i1]
    sd = max(1, int(np.floor(e.shape[0] / nl)))
    e, Uc, Vc = e[::sd, ::sd], Uc[::sd, ::sd], Vc[::sd, ::sd]
    x0g = (Uc + x0).ravel()
    y0g = (Vc + y0).ravel()
    rr2 = (Uc ** 2 + Vc ** 2).ravel()
    den = np.sqrt(R ** 2 + rr2)
    p = Uc.ravel() / den * np.sign(R) + L
    q = Vc.ravel() / den * np.sign(R) + M
    ph0 = K0 * (L * Uc.ravel() + M * Vc.ravel()
                + np.sign(R) * (den - abs(R)))
    return rs_spot(x0g, y0g, np.abs(e).ravel(), ph0, p, q, surfs, back,
                   xci, yci, dx_out=dxo, n_out=nout, nl=e.shape[0])


def chain_then_rs(env, dxg, R, L, M, x0, y0, presc, back, xci, yci,
                  nout, dxo, nl, tkw=None):
    """The chain's own last-group pass, then the same RS transport."""
    grp = [{'prescription': presc, 'gap_before': 0.0}]
    if tkw:
        grp[0]['traced_kwargs'] = tkw
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res = C.la.propagate_traced_carrier_chain(
            env, grp, LAM, dxg, r_in=C.la.TiltedCarrier(R, L, M, x0, y0),
            ray_subsample=4, n_workers=8, final_distance=0.0,
            final_leg='paraxial', on_decentred_fit='ignore')
    wmsg = sorted({str(w.message).split('.')[0][:90] for w in wl})
    fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
    st = [s for s in res.stages if not s.get('target')][-1]
    Lk, Mk = st.get('L_out', 0.0), st.get('M_out', 0.0)
    xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
    envk = carrier_referenced_envelope(fld, Rk, LAM, dxk)
    nn = envk.shape[0]
    u = (np.arange(nn) - nn / 2) * dxk
    envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
    wk = _envelope_amp_radius(envk, dxk, dxk)
    half = int(np.ceil(3.0 * wk / dxk))
    c = nn // 2
    i0, i1 = max(c - half, 0), min(c + half + 1, nn)
    envk = envk[i0:i1, i0:i1]
    u = (np.arange(envk.shape[0]) - (c - i0)) * dxk
    gx, gy, step = _phase_gradient(envk, dxk)
    phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
    sd = max(1, int(np.floor(envk.shape[0] / nl)))
    envk, gx, gy, phres = (envk[::sd, ::sd], gx[::sd, ::sd], gy[::sd, ::sd],
                           phres[::sd, ::sd])
    u = u[::sd]
    U, V = np.meshgrid(u, u)
    r = rs_spot((U + xck).ravel(), (V + yck).ravel(), np.abs(envk).ravel(),
                K0 * ((U ** 2 + V ** 2).ravel() / (2 * Rk)
                      + Lk * U.ravel() + Mk * V.ravel()) + phres.ravel(),
                (U / Rk).ravel() + Lk + gx.ravel() / K0,
                (V / Rk).ravel() + Mk + gy.ravel() / K0,
                _free_surfaces(C.TRAILING, back), back, xci, yci, dx_out=dxo, n_out=nout, nl=envk.shape[0])
    r['step_ph'] = step
    r['warn'] = wmsg
    return r


def main():
    n = int(os.environ.get('N', '1024'))
    dxg = float(os.environ.get('DXG', str(DX_G * 1e6))) * 1e-6
    nout = int(os.environ.get('NOUT', '61'))
    dxo = float(os.environ.get('DXO', '0.4')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    nl = int(os.environ.get('NL', '181'))
    decs = [float(v) for v in os.environ.get('DEC', '0,0.5,1.0,1.778'
                                             ).split(',')]
    tilts = [int(v) for v in os.environ.get('TILTS', '0,1').split(',')]
    _pre, post, _g, _per = C.geometry()
    presc = post[-1]['prescription']
    grp = [{'prescription': presc, 'gap_before': 0.0}]

    print(f"design 121 LAST GROUP probe: R_in {R_IN * 1e3:.3f} mm, w_in "
          f"{W_IN * 1e6:.0f} um, grid {n}^2 @ {dxg * 1e6:.3f} um "
          f"(half-extent {n / 2 * dxg * 1e3:.2f} mm)")
    print(f"  entrance tilt (L,M) = ({L_IN:+.6f},{M_IN:+.6f}) -> "
          f"|k L dx|/pi = {K0 * np.hypot(L_IN, M_IN) * dxg / np.pi:.3f}")
    rows = []
    for it in tilts:
        L = L_IN if it else 0.0
        M = M_IN if it else 0.0
        for d in decs:
            az = np.arctan2(M_IN, L_IN)
            x0 = -d * W_IN * np.cos(az)
            y0 = -d * W_IN * np.sin(az)
            env, U, V = build_input(n, dxg, W_IN, R_IN, L, M, x0, y0)
            # image-plane centre = the exact chief ray of this input
            ch = trace(RayBundle(x=np.array([x0]), y=np.array([y0]),
                                 z=np.zeros(1), L=np.array([L]),
                                 M=np.array([M]),
                                 N=np.array([np.sqrt(1 - L * L - M * M)]),
                                 wavelength=LAM, alive=np.ones(1, bool),
                                 opd=np.zeros(1)),
                       C.post_surfaces(grp), LAM,
                       output_filter='last').image_rays
            xci, yci = float(ch.x[0]), float(ch.y[0])
            surfs = C.post_surfaces(grp, back_off=back)
            t0 = time.time()
            o = oracle(env, U, V, dxg, R_IN, L, M, x0, y0, surfs, back,
                       xci, yci, nout, dxo, nl)
            c_ = chain_then_rs(env, dxg, R_IN, L, M, x0, y0, presc, back,
                               xci, yci, nout, dxo, nl)
            rows.append((it, d, o, c_))
            print(f"  tilt={'ON ' if it else 'OFF'} decentre={d:5.3f} w "
                  f"({np.hypot(x0, y0) * 1e3:6.3f} mm)   [{time.time() - t0:.0f}s]")
            print(f"     ORACLE  FWHM {o['fwhm'] * 1e6:6.3f} um  EE3 "
                  f"{o['ee3'] * 100:6.2f}  EE6 {o['ee6'] * 100:6.2f}")
            print(f"     CHAIN   FWHM {c_['fwhm'] * 1e6:6.3f} um  EE3 "
                  f"{c_['ee3'] * 100:6.2f}  EE6 {c_['ee6'] * 100:6.2f}   "
                  f"(exit envelope phase step {c_['step_ph']:.3f} rad)")
            for wm in c_['warn']:
                print(f"       [warn] {wm}")
    print()
    print("SUMMARY   tilt  decentre/w   oracle EE3   chain EE3   1-F      "
          "sigma(waves)")
    # 1 - F is the fraction of power the chain's image-plane field puts OUTSIDE
    # the oracle's mode: a Strehl-like measure of the ELEMENT's own error that
    # stays sensitive even when the test case is itself aberrated (EE3 alone is
    # not -- a 5 % reference cannot show a 1 % defect).  sigma is the equivalent
    # rms wavefront error from 1 - F = 1 - exp(-(2 pi sigma)^2).
    for it, d, o, c_ in rows:
        A, B = np.asarray(o['E']), np.asarray(c_['E'])
        F = abs(np.vdot(A, B)) ** 2 / (np.vdot(A, A).real * np.vdot(B, B).real)
        F = float(F)
        sig = np.sqrt(max(-np.log(max(F, 1e-12)), 0.0)) / (2 * np.pi)
        print(f"  {'ON ' if it else 'OFF':>6s}  {d:9.3f}   {o['ee3'] * 100:9.2f}"
              f"   {c_['ee3'] * 100:9.2f}  {1 - F:9.5f}  {sig:9.5f}")


if __name__ == '__main__':
    sys.exit(main())
