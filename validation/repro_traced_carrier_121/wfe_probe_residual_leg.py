# Which leg of ``apply_real_lens_traced`` carries design 121's last-group
# error -- the RAY-MAP leg or the RESIDUAL-TRANSPORT leg?
# (DIAG_LAST_GROUP_DECENTRE_2026_07_30.)
#
# EXACT ALGEBRAIC SPLIT, NO LIBRARY INSTRUMENTATION NEEDED
# --------------------------------------------------------
# On the shipped chain path (amplitude_model='ray_density',
# preserve_input_phase='remap') the element assembles
#
#     E_out = ard_map(X,Y) * exp(i k0 opl_map(X,Y)) * resid_map(X,Y)
#
# (_lens_traced.py Step 3: the screen field's UNIT PHASOR is kept, the
# magnitude is swapped for the ray-density one, and the transported residual
# phasor is multiplied on).  ``ard_map`` and ``opl_map`` depend on the input
# only through |E_in|.  So running the SAME |E_in| twice --
#
#     run A:  E_in = |r| * exp(i k0 W)      -> resid == |r| (real, positive)
#                                           -> resid_map == 1 EXACTLY
#     run B:  E_in =  r  * exp(i k0 W)      -> resid_map = the transported
#                                              input residual phasor
#
# -- gives ``E_out_B / E_out_A == resid_map`` exactly, and ``arg E_out_A ==
# k0 * opl_map``.  Run A is therefore a clean measurement of the RAY-MAP leg
# against the exact ray trace, and the ratio is a clean measurement of the
# RESIDUAL leg against a band-limited evaluation of the same residual at the
# exact pullback point.  Nothing is instrumented and nothing is inferred.
#
# usage:  DEC=0,0.25,0.5,0.75,1.0,1.079,1.25,1.5 python wfe_probe_residual_leg.py
#         SRC=synth ALPHA=1 python wfe_probe_residual_leg.py
import os
import sys
import time
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import TiltedCarrier

LAM = P.LAM
K0 = P.K0
R_IN = RM.R_IN
L_IN, M_IN = RM.L_IN, RM.M_IN
X0_D, Y0_D = RM.X0_D, RM.Y0_D
DX, N = RM.DX, RM.N


def element(E_in, car, presc, kw):
    opts = dict(amplitude_model='ray_density', preserve_input_phase='remap',
                fit_radius_beam_factor=2.0, remap_sampling='full',
                ray_subsample=4, n_workers=8)
    opts.update(kw)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        out = np.asarray(apply_real_lens_traced(
            E_in, prescription=presc, wavelength=LAM, dx=DX, carrier=car,
            **opts))
    msgs = {}
    for w in wl:
        t = str(w.message)[:120]
        msgs[t] = msgs.get(t, 0) + 1
    return out, msgs


def main():
    src = os.environ.get('SRC', 'real')
    decs = [float(v) for v in os.environ.get(
        'DEC', '0,0.25,0.5,0.75,1.0,1.079,1.25,1.5').split(',')]
    alpha = float(os.environ.get('ALPHA', '1'))
    half = int(os.environ.get('HALF', '72'))
    up = int(os.environ.get('UP', '8'))
    w = float(os.environ.get('W', '3.1255')) * 1e-3
    tilt_on = int(os.environ.get('TILT', '1'))
    kw = RM.main.__globals__ and {}
    for it in os.environ.get('KW', '').split(';'):
        if it.strip():
            k, v = it.split('=', 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = {'True': True, 'False': False}.get(v, v)
            kw[k.strip()] = v
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    az = np.arctan2(Y0_D, X0_D)
    ax = (np.arange(N) - N / 2) * DX
    Xg, Yg = np.meshgrid(ax, ax)

    E_in_real, _E_out_real, car_real = RM.real_input()
    car0 = TiltedCarrier(*car_real)
    W0, _l, _m = P.carrier_parts(car0, Xg, Yg)
    resid0 = np.asarray(E_in_real) * np.exp(-1j * K0 * W0)

    print("design 121 LAST GROUP (Lens S25-S27): RAY-MAP leg vs RESIDUAL leg")
    print(f"  source residual = {src!r} (alpha={alpha})  tilt="
          f"{'ON' if tilt_on else 'OFF'}  UP={up}  kw={kw}")
    print(f"  w = {w*1e3:.4f} mm; design decentre "
          f"{np.hypot(X0_D, Y0_D)*1e3:.4f} mm = "
          f"{np.hypot(X0_D, Y0_D)/w:.3f} w")
    _b5, _b8 = _band_report(resid0, DX)
    print(f"  REAL residual band content: {_b5:.3e} of power above 0.5 "
          f"Nyquist, {_b8:.3e} above 0.8 Nyquist "
          f"(the arbiter's Fourier upsample needs this small)")
    print()
    hdr = (f"{'dec/w':>7} {'RAYMAP rms':>11} {'RESID rms':>10} "
           f"{'RESID fid':>10} {'TOTAL vs ORACLE':>16} {'orcl fid':>9} "
           f"{'d(resid)/px':>12} {'npix':>7}")
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for d in decs:
        t0 = time.time()
        x0 = d * w * np.cos(az)
        y0 = d * w * np.sin(az)
        car = TiltedCarrier(R_IN, L_IN if tilt_on else 0.0,
                            M_IN if tilt_on else 0.0, x0, y0)
        Wn, _l2, _m2 = P.carrier_parts(car, Xg, Yg)
        if src == 'real':
            r = RM.fshift(resid0, x0 - X0_D, y0 - Y0_D, DX)
            if alpha != 1.0:
                r = np.abs(r) * np.exp(1j * alpha * np.angle(r))
        else:
            u, v = Xg - x0, Yg - y0
            q = (u * u + v * v) / (w * w)
            r = np.exp(-q) * np.exp(1j * alpha * q * q)
        E_A = (np.abs(r) * np.exp(1j * K0 * Wn)).astype(np.complex128)
        E_B = (r * np.exp(1j * K0 * Wn)).astype(np.complex128)
        oA, mA = element(E_A, car, presc, kw)
        oB, mB = element(E_B, car, presc, kw)
        # exit patch
        xo, yo, _p3, _l3, _m3, _a3 = P.trace_forward([x0], [y0], car, surfs)
        ic = int(round(float(xo[0]) / DX + N / 2))
        jc = int(round(float(yo[0]) / DX + N / 2))
        sx = slice(max(ic - half, 0), min(ic + half + 1, N))
        sy = slice(max(jc - half, 0), min(jc + half + 1, N))
        Xp, Yp = Xg[sy, sx], Yg[sy, sx]
        EA, EB = oA[sy, sx], oB[sy, sx]
        # exact inverse of the CARRIER ray map
        t = np.linspace(-2.6 * w, 2.6 * w, 21)
        U, V = np.meshgrid(t, t)
        gxo, gyo, _q, _q2, _q3, alv = P.trace_forward(
            (U + x0).ravel(), (V + y0).ravel(), car, surfs)
        gg = alv & np.isfinite(gxo)
        A3 = np.stack([np.ones(int(gg.sum())), gxo[gg], gyo[gg]], axis=1)
        cxx = np.linalg.lstsq(A3, (U + x0).ravel()[gg], rcond=None)[0]
        cyy = np.linalg.lstsq(A3, (V + y0).ravel()[gg], rcond=None)[0]
        B3 = np.stack([np.ones(Xp.size), Xp.ravel(), Yp.ravel()], axis=1)
        guess = ((B3 @ cxx).reshape(Xp.shape), (B3 @ cyy).reshape(Xp.shape))
        exA = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=guess,
                                     n_iter=16)
        ampA = np.abs(EA)
        pk = float(ampA.max())
        keep = exA['ok'] & (ampA > 0.02 * pk) & (exA['resid'] < 1e-9)
        # 1. RAY-MAP leg
        rmap = P.local_wfe(EA, exA['phi'], keep, Xp, Yp)
        # 2. RESIDUAL leg: the element's own transported phasor vs a
        #    band-limited evaluation of the SAME residual at the SAME point
        rf = RM.ResidualField(r, DX, x0, y0, 3.4 * w, up=up)
        # self-test of the residual model on the entrance grid nodes
        st_err = _selftest(rf, r, x0, y0, w)
        Q = np.where(np.abs(EA) > 0, EB / np.where(np.abs(EA) > 0, EA, 1.0),
                     0.0)
        phi_ref = rf.phase(exA['xe'], exA['ye'])
        rres = P.local_wfe(Q * np.abs(EB), phi_ref, keep, Xp, Yp)
        # gradient of the transported residual per EXIT pixel (sampling stmt)
        gstep = _resid_exit_step(phi_ref, keep, np.abs(EB))
        # 3. TOTAL vs the exact-ray ORACLE (rays along grad(W + a))
        exB = RM.invert_total(Xp, Yp, car, surfs, rf, guess)
        keepB = keep & exB['ok'] & (exB['resid'] < 1e-9)
        rtot = P.local_wfe(EB, exB['phi'], keepB, Xp, Yp)
        print(f"{d:>7.3f} {rmap['rms_waves']:>11.5f} "
              f"{rres['sigma_fid_waves']:>10.5f} "
              f"{rres['fidelity_notilt']:>10.6f} "
              f"{rtot['sigma_fid_waves']:>16.5f} "
              f"{rtot['fidelity_notilt']:>9.6f} "
              f"{gstep:>12.4f} {rtot['n_pix']:>7d}   [{time.time()-t0:.0f}s]"
              f"  selftest {st_err:.2e} rad")
        rows.append((d, rmap, rres, rtot))
    print()
    print("RAYMAP rms  : rms of arg(E_out_A) - k0*[W+OPL]_exact, waves "
          "(piston+tilt removed, unwrap validated by nn_step below)")
    print("RESID rms   : unwrap-free equivalent rms of "
          "arg(resid_map_element) - arg(resid_bandlimited(xe*,ye*)), waves")
    print("TOTAL vs ORACLE: unwrap-free equivalent rms of the FULL element "
          "field against rays launched along grad(W + a), waves")
    print("d(resid)/px : power-weighted 99th pct per-EXIT-pixel step of the "
          "transported residual phase, rad (pi = 3.1416)")
    print()
    for d, rmap, rres, rtot in rows:
        print(f"  dec {d:5.3f}  raymap: nn_step "
              f"med/p99/max {rmap['nn_step_med']:.4f}/"
              f"{rmap['nn_step_p99']:.4f}/{rmap['nn_step_rad']:.4f} rad, "
              f"fid {rmap['fidelity_notilt']:.6f}  |  resid: nn_step "
              f"med/p99 {rres['nn_step_med']:.4f}/{rres['nn_step_p99']:.4f}, "
              f"wrapped-rms {rres['rms_wrapped_notilt']:.5f} waves  |  total: "
              f"nn_step med/p99 {rtot['nn_step_med']:.4f}/"
              f"{rtot['nn_step_p99']:.4f}")


def _selftest(rf, r, x0, y0, w):
    """|wrapped(rf.phase) - angle(r)| on the ENTRANCE grid nodes the residual
    model must reproduce EXACTLY (they are samples of its own source, and the
    Fourier upsample is exact at them).  Returns the max over the disc that
    carries 99.99 % of the residual power -- a nonzero value is a bug in the
    model, not a property of the element."""
    n = r.shape[0]
    ax = (np.arange(n) - n / 2) * DX
    X, Y = np.meshgrid(ax, ax)
    m = (((X - x0) ** 2 + (Y - y0) ** 2) <= (2.0 * w) ** 2) & \
        (np.abs(r) > 1e-3 * np.abs(r).max())
    got = rf.phase(X[m], Y[m])
    want = np.angle(r[m])
    e = np.abs(np.angle(np.exp(1j * (got - want))))
    return float(e.max())


def _band_report(r, dx, label=''):
    """Fraction of the residual's power above 0.5 and 0.8 of grid Nyquist --
    the Fourier upsample used by the arbiter is exact only for a band-limited
    residual, so this is the precondition, stated."""
    n = r.shape[0]
    F = np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    FX, FY = np.meshgrid(f, f)
    q = np.hypot(FX, FY) / (0.5 / dx)
    tot = F.sum()
    return float(F[q > 0.5].sum() / tot), float(F[q > 0.8].sum() / tot)


def _resid_exit_step(phi, keep, amp):
    dx1 = np.abs(np.angle(np.exp(1j * (phi[:, 1:] - phi[:, :-1]))))
    dy1 = np.abs(np.angle(np.exp(1j * (phi[1:, :] - phi[:-1, :]))))
    kx = keep[:, 1:] & keep[:, :-1]
    ky = keep[1:, :] & keep[:-1, :]
    v = np.concatenate([dx1[kx], dy1[ky]])
    ww = np.concatenate([np.minimum(amp[:, 1:], amp[:, :-1])[kx] ** 2,
                         np.minimum(amp[1:, :], amp[:-1, :])[ky] ** 2])
    if not v.size:
        return float('nan')
    o = np.argsort(v)
    cw = np.cumsum(ww[o])
    cw /= max(cw[-1], 1e-300)
    return float(v[o][np.searchsorted(cw, 0.99)])


if __name__ == '__main__':
    sys.exit(main())
