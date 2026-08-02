# IS MY WAVEFRONT-ERROR NUMBER HONEST?  The closure test.
# (DIAG_LAST_GROUP_DECENTRE_2026_07_30 -- adversarial check on this study's
# own instrument.)
#
# wfe_probe_orders.py measures the shipped last-group exit field against the
# exact-ray oracle at 0.011 waves on axis and 0.036 waves at (-4,-2)
# (Marechal Strehl 0.995 / 0.951).  The hybrid localisation says the same
# element pass costs EE3 89.94 -> 87.60 on axis and 89.80 -> 65.75 at
# (-4,-2), which needs ~0.026 / ~0.089 waves.  My instrument is therefore
# 2.4x LOW in sigma at BOTH ends, i.e. it has a systematic, not a scatter.
#
# This script settles it WITHOUT trusting either number: it builds the ORACLE
# exit field on the SAME grid the element writes to (amplitude = the exact
# ray-tube amplitude of the TOTAL congruence, phase = the exact traced phase),
# and pushes the element field and the oracle field through the IDENTICAL
# readout -- the same re-envelope, the same aliasing-free phase gradient, the
# same ray launch, the same Rayleigh-Sommerfeld integral.  Whatever the
# readout does wrong, it does to both.
#
#   * oracle field reads ~90 and element field reads ~66  ->  the fields
#     really differ that much and my rms metric under-reads; find out why.
#   * both read ~66  ->  the fields agree and the loss is in the READOUT of a
#     field that is correct at its nodes (i.e. an exit-sampling failure, not
#     an element defect).
#
# usage:  ORD=-4,-2 HALF=110 NOUT=131 DXO=0.2 python wfe_probe_readout.py
import os
import sys
import time

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
import wfe_probe_orders as OR
from hybrid_localize_121 import _free_surfaces, rs_spot
from exact_ray_oracle_121 import _phase_gradient
from lumenairy.elements._lens_traced import TiltedCarrier
from lumenairy.propagators.carrier import (_envelope_amp_radius,
                                           carrier_referenced_envelope)
from lumenairy.raytrace import RayBundle, trace

LAM = P.LAM
K0 = P.K0


def readout(E_exit, dx, R_out, L_out, M_out, xc, yc, back, xci, yci,
            nout, dxo, nl, clip=3.0, grid=None):
    """The chain's own exit readout, applied to ANY field on the exit grid.

    Identical construction to last_group_probe_121.chain_then_rs: strip the
    exit sphere + tilt analytically, take the residual envelope's
    ALIASING-FREE nearest-neighbour phase gradient, launch exact rays from
    the exit plane and finish with the Rayleigh-Sommerfeld integral.
    """
    E = np.asarray(E_exit)
    nn = E.shape[0]
    if grid is None:
        ax = (np.arange(nn) - nn / 2) * dx
        Xg, Yg = np.meshgrid(ax, ax)
    else:
        Xg, Yg = grid
    cex = TiltedCarrier(R_out, L_out, M_out, xc, yc)
    Wc, Lc, Mc = P.carrier_parts(cex, Xg, Yg)
    env = E * np.exp(-1j * K0 * Wc)
    ic = int(round((xc - Xg[0, 0]) / dx))
    jc = int(round((yc - Yg[0, 0]) / dx))
    a2 = np.abs(env) ** 2
    tot = float(a2.sum())
    wk = float(np.sqrt(2.0 * (a2 * ((Xg - xc) ** 2 + (Yg - yc) ** 2)).sum()
                       / max(tot, 1e-300)))
    half = int(np.ceil(clip * wk / dx))
    i0, i1 = max(ic - half, 0), min(ic + half + 1, nn)
    j0, j1 = max(jc - half, 0), min(jc + half + 1, nn)
    envk = env[j0:j1, i0:i1]
    Xk, Yk = Xg[j0:j1, i0:i1], Yg[j0:j1, i0:i1]
    Lk, Mk = Lc[j0:j1, i0:i1], Mc[j0:j1, i0:i1]
    Wk = Wc[j0:j1, i0:i1]
    gx, gy, step = _phase_gradient(envk, dx)
    # POWER-WEIGHTED distribution of the envelope's per-pixel step -- the max
    # that _phase_gradient returns is set by a single skirt pixel and reads
    # ~pi even when the core is fully resolved.
    _ph = np.angle(envk)
    _a2 = np.abs(envk) ** 2
    _sx = np.abs(np.angle(np.exp(1j * (_ph[:, 1:] - _ph[:, :-1]))))
    _sy = np.abs(np.angle(np.exp(1j * (_ph[1:, :] - _ph[:-1, :]))))
    _v = np.concatenate([_sx.ravel(), _sy.ravel()])
    _wv = np.concatenate([np.minimum(_a2[:, 1:], _a2[:, :-1]).ravel(),
                          np.minimum(_a2[1:, :], _a2[:-1, :]).ravel()])
    _o = np.argsort(_v)
    _cw = np.cumsum(_wv[_o])
    _cw = _cw / max(_cw[-1], 1e-300)
    step_p50 = float(_v[_o][np.searchsorted(_cw, 0.50)])
    step_p99 = float(_v[_o][np.searchsorted(_cw, 0.99)])
    phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
    sd = max(1, int(np.floor(envk.shape[0] / nl)))
    envk, gx, gy, phres = (envk[::sd, ::sd], gx[::sd, ::sd], gy[::sd, ::sd],
                           phres[::sd, ::sd])
    Xk, Yk, Lk, Mk, Wk = (Xk[::sd, ::sd], Yk[::sd, ::sd], Lk[::sd, ::sd],
                          Mk[::sd, ::sd], Wk[::sd, ::sd])
    r = rs_spot(Xk.ravel(), Yk.ravel(), np.abs(envk).ravel(),
                K0 * Wk.ravel() + phres.ravel(),
                Lk.ravel() + gx.ravel() / K0,
                Mk.ravel() + gy.ravel() / K0,
                _free_surfaces(C.TRAILING, back), back, xci, yci,
                dx_out=dxo, n_out=nout, nl=envk.shape[0])
    r['step_ph'] = step
    r['step_p50'] = step_p50
    r['step_p99'] = step_p99
    r['w_exit'] = wk
    return r


def _exact_exit_cosines(exB, car, surfs, rf):
    xo, yo, psi, Lo, Mo, alive = RM.trace_total(exB['xe'], exB['ye'], car,
                                                surfs, rf)
    sh = exB['xe'].shape
    return Lo.reshape(sh), Mo.reshape(sh)


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    half = int(os.environ.get('HALF', '110'))
    nout = int(os.environ.get('NOUT', '131'))
    dxo = float(os.environ.get('DXO', '0.2')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    nl = int(os.environ.get('NL', '181'))
    up = int(os.environ.get('UP', '8'))

    _pre, post, _g, period = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    E_in, E_out, carv, dx = OR.get_call(m, n)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _m = P.carrier_parts(car, Xg, Yg)
    resid = np.asarray(E_in) * np.exp(-1j * K0 * W)
    from lumenairy.elements._lens_traced import _input_beam_amp_radius
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    rf = RM.ResidualField(resid, dx, car.x0, car.y0, 3.4 * w, up=up)

    # the chain's own exit congruence for this order
    calls, res, _msg = P.run_chain_capture(m=m, n=n)
    st = [s for s in res.stages if not s.get('target')][-1]
    R_out = float(st['R_out'])
    L_out = float(st.get('L_out', 0.0))
    M_out = float(st.get('M_out', 0.0))
    xc = float(st.get('x_c_out', 0.0))
    yc = float(st.get('y_c_out', 0.0))
    print(f"order ({m},{n})  R_out {R_out*1e3:.4f} mm  L,M ({L_out:+.6f},"
          f"{M_out:+.6f})  chief exit ({xc*1e3:+.4f},{yc*1e3:+.4f}) mm  "
          f"dx {dx*1e6:.4f} um")

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

    # ---- the ORACLE exit field on the SAME grid --------------------------
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
    t0 = time.time()
    exB = RM.invert_total(Xp, Yp, car, surfs, rf, guess)
    h = 5e-7
    xe, ye = exB['xe'], exB['ye']
    x1, y1, _q, _q2, _q3, _q4 = RM.trace_total(xe + h, ye, car, surfs, rf)
    x2, y2, _q, _q2, _q3, _q4 = RM.trace_total(xe, ye + h, car, surfs, rf)
    x0f, y0f, _q, _q2, _q3, _q4 = RM.trace_total(xe, ye, car, surfs, rf)
    det = np.abs((x1 - x0f) * (y2 - y0f) - (x2 - x0f) * (y1 - y0f)).reshape(
        Xp.shape) / (h * h)
    a_ex = OR.rf_amp(rf, xe, ye) / np.sqrt(np.maximum(det, 1e-300))
    ok = exB['ok'] & (exB['resid'] < 1e-9) & np.isfinite(a_ex)
    Eo = np.zeros((N, N), dtype=np.complex128)
    Eo[sy, sx] = np.where(ok, a_ex * np.exp(1j * exB['phi']), 0.0)
    # normalise the oracle to the element's exit power over the same support
    Ee = np.array(E_out, copy=True)
    msk = np.zeros((N, N), bool)
    msk[sy, sx] = ok
    s = np.sqrt(float((np.abs(Ee[msk]) ** 2).sum())
                / max(float((np.abs(Eo[msk]) ** 2).sum()), 1e-300))
    Eo *= s
    print(f"  oracle exit field built in {time.time()-t0:.0f}s;  "
          f"power(oracle patch)/power(element whole grid) = "
          f"{float((np.abs(Eo)**2).sum())/float((np.abs(Ee)**2).sum()):.6f}")
    # phase-only and amplitude-only hybrids, to attribute
    with np.errstate(invalid='ignore', divide='ignore'):
        ue = np.where(np.abs(Ee) > 0, Ee / np.maximum(np.abs(Ee), 1e-300), 0)
        uo = np.where(np.abs(Eo) > 0, Eo / np.maximum(np.abs(Eo), 1e-300), 0)
    E_amp_e_ph_o = np.abs(Ee) * uo * msk
    E_amp_o_ph_e = np.abs(Eo) * ue

    # ORACLE relaunched with its EXACT exit direction cosines and EXACT exit
    # phase -- identical field, but with the readout's ONE estimated quantity
    # (the per-node exit direction, taken from the coarse envelope's
    # nearest-neighbour phase gradient) replaced by the truth.  The gap
    # between this row and the ORACLE row is the cost of estimating the exit
    # direction on the exit grid; it is the SAME estimation every downstream
    # consumer of the element's exit field has to make.
    Lo_, Mo_ = _exact_exit_cosines(exB, car, surfs, rf)
    # ---- CONTROL: the SAME oracle congruence launched from the group-5
    # ENTRANCE (the hybrid n=5 construction, which reads 90.00 for this
    # order).  Same residual model, same surfaces, same RS integral -- only
    # the launch PLANE differs.  This validates the residual model and the
    # ray machinery independently of the exit-plane relaunch.
    import dataclasses as _dc
    from lumenairy.raytrace import Surface as _Sf
    _sf = list(surfs)
    _sf[-1] = _dc.replace(_sf[-1], thickness=float(C.TRAILING) - back)
    _sf.append(_Sf(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                   glass_before='air', glass_after='air', is_mirror=False,
                   thickness=0.0, label='img'))
    _nl = int(os.environ.get('NLENT', '221'))
    _t = np.linspace(-3.0 * w, 3.0 * w, _nl)
    _U, _V = np.meshgrid(_t, _t)
    _XE, _YE = _U + car.x0, _V + car.y0
    _Wa, _La, _Ma = P.carrier_parts(car, _XE, _YE)
    _aa, _gL, _gM = rf.ev(_XE, _YE)
    _amp_e = rf.amp_at(_XE, _YE)
    _r_ent = rs_spot(_XE.ravel(), _YE.ravel(), _amp_e.ravel(),
                     (K0 * (_Wa + _aa)).ravel(), (_La + _gL).ravel(),
                     (_Ma + _gM).ravel(), _sf, back, xci, yci,
                     dx_out=dxo, n_out=nout, nl=_nl)
    print(f"{'ORACLE, ENTRANCE launch':>26} {_r_ent['fwhm']*1e6:>9.3f} "
          f"{_r_ent['ee3']*100:>8.2f} {_r_ent['ee6']*100:>8.2f} "
          f"{_r_ent['ee12']*100:>8.2f}   (same congruence, launched at the "
          f"group ENTRANCE: {_nl}^2 rays, RSstep {_r_ent['step_w']:.3f} cyc, "
          f"live {_r_ent['live']*100:.4f} %)")

    _amp_o = np.where(ok, np.abs(Eo[sy, sx]), 0.0)
    _ph_o = np.where(ok, exB['phi'], 0.0)
    r_ex = rs_spot(np.ascontiguousarray(Xp).ravel(),
                   np.ascontiguousarray(Yp).ravel(), _amp_o.ravel(),
                   _ph_o.ravel(), np.where(ok, Lo_, 0.0).ravel(),
                   np.where(ok, Mo_, 0.0).ravel(),
                   _free_surfaces(C.TRAILING, back), back, xci, yci,
                   dx_out=dxo, n_out=nout, nl=Xp.shape[0])
    print(f"{'ORACLE, EXACT cosines':>26} {r_ex['fwhm']*1e6:>9.3f} "
          f"{r_ex['ee3']*100:>8.2f} {r_ex['ee6']*100:>8.2f} "
          f"{r_ex['ee12']*100:>8.2f}   (same nodes, same amplitudes, same "
          f"phases; only the exit DIRECTION is exact.  "
          f"RSstep {r_ex['step_w']:.3f} cyc, {r_ex['n_rays']} rays)")

    fields = [('ELEMENT (shipped)', Ee),
              ('ORACLE  (exact ray)', Eo),
              ('elem amp + oracle phase', E_amp_e_ph_o),
              ('oracle amp + elem phase', E_amp_o_ph_e)]
    print()
    print(f"{'field':>26} {'FWHM(um)':>9} {'EE3 %':>8} {'EE6 %':>8} "
          f"{'EE12 %':>8} {'env p50':>8} {'env p99':>8} {'env max':>8} "
          f"{'RSstep':>7} {'live':>7} {'rays':>7}")
    for name, F in fields:
        t0 = time.time()
        r = readout(F, dx, R_out, L_out, M_out, xc, yc, back, xci, yci,
                    nout, dxo, nl)
        print(f"{name:>26} {r['fwhm']*1e6:>9.3f} {r['ee3']*100:>8.2f} "
              f"{r['ee6']*100:>8.2f} {r['ee12']*100:>8.2f} "
              f"{r['step_p50']:>8.4f} {r['step_p99']:>8.4f} "
              f"{r['step_ph']:>8.4f} {r['step_w']:>7.3f} {r['live']:>7.4f} "
              f"{r['n_rays']:>7d}   [{time.time()-t0:.0f}s]", flush=True)
    print()
    print("env step = the max aliasing-free per-pixel phase step of the "
          "RESIDUAL ENVELOPE that the readout differentiates.  Above ~pi the "
          "launch directions are meaningless FOR THAT FIELD.")

    # ---- GRID REFINEMENT: the same ORACLE field on a finer exit grid -------
    facs = [int(v) for v in os.environ.get('DXFAC', '1,2,4').split(',')]
    if facs == [0]:
        return
    hf = int(os.environ.get('HALFF', '60'))
    print()
    print("GRID REFINEMENT of the EXIT plane.  The SAME exact-ray oracle "
          "field, sampled on a grid refined by DXFAC and read out the SAME "
          "way.  If the readout is sampling-limited this climbs toward the "
          "design's 89.8 % ceiling; if the field is genuinely bad it does "
          "not move.")
    print(f"{'dx (um)':>9} {'nodes':>7} {'FWHM(um)':>9} {'EE3 %':>8} "
          f"{'EE6 %':>8} {'EE12 %':>8} {'env p50':>8} {'env p99':>8} "
          f"{'env max':>8} {'w_exit(um)':>11}")
    for fac in facs:
        t0 = time.time()
        dxf = dx / fac
        nn = 2 * hf * fac + 1
        axf = (np.arange(nn) - (nn - 1) / 2.0) * dxf
        Xf = float(xo[0]) + axf[None, :] + 0.0 * axf[:, None]
        Yf = float(yo[0]) + axf[:, None] + 0.0 * axf[None, :]
        Xf, Yf = np.broadcast_arrays(Xf, Yf)
        Xf, Yf = np.ascontiguousarray(Xf), np.ascontiguousarray(Yf)
        Bf = np.stack([np.ones(Xf.size), Xf.ravel(), Yf.ravel()], axis=1)
        gf = ((Bf @ cxx).reshape(Xf.shape), (Bf @ cyy).reshape(Xf.shape))
        eb = RM.invert_total(Xf, Yf, car, surfs, rf, gf)
        xe_, ye_ = eb['xe'], eb['ye']
        x1, y1, _q, _q2, _q3, _q4 = RM.trace_total(xe_ + h, ye_, car, surfs, rf)
        x2, y2, _q, _q2, _q3, _q4 = RM.trace_total(xe_, ye_ + h, car, surfs, rf)
        x0f2, y0f2, _q, _q2, _q3, _q4 = RM.trace_total(xe_, ye_, car, surfs, rf)
        dt = np.abs((x1 - x0f2) * (y2 - y0f2)
                    - (x2 - x0f2) * (y1 - y0f2)).reshape(Xf.shape) / (h * h)
        af = OR.rf_amp(rf, xe_, ye_) / np.sqrt(np.maximum(dt, 1e-300))
        okf = eb['ok'] & (eb['resid'] < 1e-9) & np.isfinite(af)
        Ef = np.where(okf, af * np.exp(1j * eb['phi']), 0.0)
        r = readout(Ef, dxf, R_out, L_out, M_out, xc, yc, back, xci, yci,
                    nout, dxo, nl, grid=(Xf, Yf))
        print(f"{dxf*1e6:>9.4f} {nn:>7d} {r['fwhm']*1e6:>9.3f} "
              f"{r['ee3']*100:>8.2f} {r['ee6']*100:>8.2f} "
              f"{r['ee12']*100:>8.2f} {r['step_p50']:>8.4f} "
              f"{r['step_p99']:>8.4f} {r['step_ph']:>8.4f} "
              f"{r['w_exit']*1e6:>11.2f}   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == '__main__':
    sys.exit(main())
