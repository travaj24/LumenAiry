# WHERE in the chain does design 121's per-order spot quality die?
# (SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30, localisation experiment.)
#
# ``exact_ray_oracle_121.py`` establishes that the DESIGN is diffraction-
# limited at every DOE order (EE3 89.9-90.5 %, WFE rms <= 0.017 waves), so the
# chain's monotone per-order loss is a library defect.  This script bisects it.
#
# METHOD.  For n = 0 .. 6, run ``propagate_traced_carrier_chain`` over the
# FIRST n post-DOE groups (tilted congruence, exact same defaults the shipped
# runner uses), take the field it hands out, and finish the remaining
# 6 - n groups + the trailing leg with the exact-ray + Rayleigh-Sommerfeld
# oracle.  n = 0 is the pure oracle (the design's own answer); n = 6 is the
# chain doing everything except the final free-space leg + readout.  The EE is
# measured IDENTICALLY at every n, so the step between n and n+1 is the cost of
# putting group n through the chain.
#
# HAND-OFF.  The chain returns the reconstructed FULL field on a grid centred
# on the chief ray, plus the carrier radius R and the stage's (L, M, x_c, y_c).
# Rays are launched from that plane with
#     direction  = (u/R + L + grad(phi_res)/k),      u = grid coordinate
#     OPL offset = |u|^2/(2R) + L u_x + M u_y + phi_res/k
# where ``phi_res`` is the RESIDUAL envelope phase (carrier and tilt ramp
# removed analytically -- their gradients are exact).  ``phi_res`` is
# differentiated by ALIASING-FREE wrapped nearest-neighbour differences, and
# the largest per-pixel step over the bright core is printed: it must be << pi
# or the launch directions are meaningless (the exact trap the D7 correction
# block documents).
#
# SAMPLING.  ``UP`` Fourier-upsamples the chain envelope before launch (exact
# for a band-limited envelope).  UP=1 vs UP=2 must agree, or the hand-off grid
# -- not the chain -- is the thing being measured.
#
# Env: ORD='m,n', RN, RS, UP, NOUT, DXO(um), BACK(mm), NMAX, NMIN.
import os
import sys
import time

import numpy as np

import _d121_common as C
from exact_ray_oracle_121 import _bilinear, _phase_gradient
from lumenairy.propagators.carrier import (_envelope_amp_radius,
                                           carrier_referenced_envelope)
from lumenairy.raytrace import RayBundle, Surface, trace

LAM = C.LAM
K0 = 2 * np.pi / LAM


def _free_surfaces(dist, back):
    return [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', is_mirror=False,
                    thickness=float(dist) - float(back), label='launch'),
            Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', is_mirror=False,
                    thickness=0.0, label='img')]


def _fourier_up(a, up):
    if up == 1:
        return a
    n = a.shape[0]
    m = n * up
    F = np.fft.fftshift(np.fft.fft2(a))
    G = np.zeros((m, m), dtype=complex)
    o = (m - n) // 2
    G[o:o + n, o:o + n] = F
    return np.fft.ifft2(np.fft.ifftshift(G)) * (up * up)


def rs_spot(x0, y0, amp, ph0, p, q, surfs, back, xc_img, yc_img,
            dx_out=0.1e-6, n_out=261, nl=None, label=''):
    """Exact trace of the launch bundle + Rayleigh-Sommerfeld to the image."""
    n0 = np.sqrt(np.maximum(1.0 - p * p - q * q, 0.0))
    rb = RayBundle(x=x0.copy(), y=y0.copy(), z=np.zeros_like(x0),
                   L=p.copy(), M=q.copy(), N=n0.copy(), wavelength=LAM,
                   alive=np.ones(x0.size, bool), opd=np.zeros_like(x0))
    ir = trace(rb, surfs, LAM, output_filter='last').image_rays
    alive = np.asarray(ir.alive, bool)
    xe = np.asarray(ir.x)
    ye = np.asarray(ir.y)
    opd = np.asarray(ir.opd)
    Ne = np.asarray(ir.N)
    nl = nl or int(round(np.sqrt(x0.size)))
    h2 = None
    XE = xe.reshape(nl, nl)
    YE = ye.reshape(nl, nl)
    hx = float(x0.reshape(nl, nl)[0, 1] - x0.reshape(nl, nl)[0, 0])
    hy = float(y0.reshape(nl, nl)[1, 0] - y0.reshape(nl, nl)[0, 0])
    J = np.abs(np.gradient(XE, hx, axis=1) * np.gradient(YE, hy, axis=0)
               - np.gradient(XE, hy, axis=0) * np.gradient(YE, hx, axis=1)
               ).ravel()
    good = alive & (amp > 0) & np.isfinite(opd) & (J > 0) & (Ne > 0)
    p_all = float(np.sum((amp ** 2 * n0)[amp > 0]))
    p_ok = float(np.sum((amp ** 2 * n0)[good]))
    W = amp[good] * np.sqrt(n0[good] * J[good] / Ne[good]) * abs(hx * hy)
    ph = K0 * opd[good] + ph0[good]
    xe, ye = xe[good], ye[good]

    ax = (np.arange(n_out) - (n_out - 1) / 2.0) * dx_out
    PX = xc_img + ax
    PY = yc_img + ax
    # sampling check on the ACTUAL integrand phase, at the patch corner
    phi = np.full(x0.size, np.nan)
    phi[good] = ph + K0 * np.sqrt((PX[0] - xe) ** 2 + (PY[0] - ye) ** 2
                                  + back ** 2)
    PH = phi.reshape(nl, nl)
    Wg = np.zeros(x0.size)
    Wg[good] = W
    WG = Wg.reshape(nl, nl)
    wm = max(W.max(), 1e-300)
    st = np.concatenate([
        (np.abs(np.diff(PH, axis=0)) * np.minimum(WG[1:], WG[:-1])).ravel(),
        (np.abs(np.diff(PH, axis=1)) * np.minimum(WG[:, 1:], WG[:, :-1])
         ).ravel()]) / wm
    st = st[np.isfinite(st)]
    # report the DISTRIBUTION: a single max is dominated by a handful of rays
    # at the vignetting edge / ray-map fold, where sqrt(J) spikes.  The
    # quadrature is governed by where the POWER is.
    step_w = float(np.percentile(st, 99.9)) / (2 * np.pi) if st.size else np.nan
    step_mx = float(st.max()) / (2 * np.pi) if st.size else np.nan
    step_md = float(np.percentile(st, 50)) / (2 * np.pi) if st.size else np.nan

    E = np.zeros((n_out, n_out), dtype=np.complex128)
    chunk = max(1, int(6e7 // max(xe.size, 1)))
    t0 = time.time()
    for j0 in range(0, n_out, chunk):
        j1 = min(j0 + chunk, n_out)
        dyv = PY[j0:j1, None] - ye[None, :]
        for i, px in enumerate(PX):
            rho = np.sqrt((px - xe)[None, :] ** 2 + dyv ** 2 + back ** 2)
            E[j0:j1, i] = np.sum((W * back) / (rho * rho)
                                 * np.exp(1j * (ph[None, :] + K0 * rho)),
                                 axis=1)
    I = np.abs(E) ** 2
    tot = float(I.sum())
    Xg, Yg = np.meshgrid(ax, ax)
    cx = float((I * Xg).sum() / tot)
    cy = float((I * Yg).sum() / tot)
    r = np.hypot(Xg - cx, Yg - cy)
    out = {'I': I, 'E': E, 'ax': ax, 'centroid': (xc_img + cx, yc_img + cy),
           'live': p_ok / p_all if p_all else 0.0, 'step_w': step_w,
           'step_mx': step_mx, 'step_md': step_md,
           'n_rays': int(good.sum()), 'time': time.time() - t0}
    for rad in (3e-6, 6e-6, 12e-6):
        out[f'ee{int(rad * 1e6)}'] = float(I[r <= rad].sum()) / tot
    nb = int(min(n_out // 2, 12e-6 / dx_out))
    ring = np.clip((r / dx_out).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cn = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = s[:nb] / np.maximum(cn[:nb], 1)
    prof = prof / prof[0]
    idx = np.where(prof < 0.5)[0]
    if len(idx) and idx[0] > 0:
        i1 = idx[0]
        f = (prof[i1 - 1] - 0.5) / (prof[i1 - 1] - prof[i1])
        out['fwhm'] = 2 * ((i1 - 1 + 0.5) * dx_out + f * dx_out)
    else:
        out['fwhm'] = np.nan
    return out


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    rn = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    up = int(os.environ.get('UP', '1'))
    nout = int(os.environ.get('NOUT', '261'))
    dxo = float(os.environ.get('DXO', '0.1')) * 1e-6
    back = float(os.environ.get('BACK', '5.0')) * 1e-3
    nmin = int(os.environ.get('NMIN', '0'))
    nmax = int(os.environ.get('NMAX', '6'))
    clip = float(os.environ.get('CLIP', '3.0'))
    nl0 = int(os.environ.get('NL', '161'))

    _pre, post, _g, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=rs)
    L = m * LAM / period
    M = n * LAM / period
    print(f"order ({m:+d},{n:+d})  L,M = {L * 1e3:+.3f},{M * 1e3:+.3f} mrad   "
          f"RN={rn} RS={rs} UP={up} NOUT={nout} DXO={dxo * 1e6} BACK="
          f"{back * 1e3} mm")

    # image-plane centre: the exact chief ray of the FULL post-DOE system
    ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                         L=np.array([L]), M=np.array([M]),
                         N=np.array([np.sqrt(1 - L * L - M * M)]),
                         wavelength=LAM, alive=np.ones(1, bool),
                         opd=np.zeros(1)),
               C.post_surfaces(post), LAM, output_filter='last').image_rays
    xci, yci = float(ch.x[0]), float(ch.y[0])

    rows = []
    for nc in range(nmin, nmax + 1):
        t0 = time.time()
        if nc == 0:
            w = _envelope_amp_radius(env_doe, dx_doe, dx_doe)
            h = 2 * clip * w / (nl0 - 1)
            t = (np.arange(nl0) - (nl0 - 1) / 2.0) * h
            X, Y = np.meshgrid(t, t)
            x0, y0 = X.ravel(), Y.ravel()
            amp = _bilinear(np.abs(env_doe), dx_doe, x0, y0)
            den = np.sqrt(R_doe ** 2 + x0 ** 2 + y0 ** 2)
            p = x0 / den * np.sign(R_doe) + L
            q = y0 / den * np.sign(R_doe) + M
            ph0 = K0 * (L * x0 + M * y0
                        + np.sign(R_doe) * (den - abs(R_doe)))
            surfs = C.post_surfaces(post, back_off=back)
            nl = nl0
            step_ph = 0.0
            dxs, ws = dx_doe, w
        else:
            # per-group traced_kwargs override, applied to the LAST group only
            # (TKW='k=v;k=v'), so a sweep changes exactly one element pass.
            _grp = [dict(g) for g in post[:nc]]
            _tkw = os.environ.get('TKW', '')
            if _tkw:
                kv = {}
                for item in _tkw.split(';'):
                    if not item.strip():
                        continue
                    a, b = item.split('=', 1)
                    try:
                        b = int(b)
                    except ValueError:
                        try:
                            b = float(b)
                        except ValueError:
                            b = {'True': True, 'False': False}.get(b, b)
                    kv[a.strip()] = b
                _grp[-1]['traced_kwargs'] = kv
                print(f"     TKW on groups[{nc - 1}]: {kv}")
            import warnings as _wm
            with _wm.catch_warnings(record=True) as _wlist:
                _wm.simplefilter('always')
                res = C.la.propagate_traced_carrier_chain(
                    env_doe, _grp, LAM, dx_doe,
                    r_in=C.la.TiltedCarrier(R_doe, L, M),
                    ray_subsample=rs, n_workers=8, final_distance=0.0,
                    final_leg='paraxial', on_decentred_fit='ignore')
            _seen = {}
            for _w in _wlist:
                _t = str(_w.message)[:150]
                _seen[_t] = _seen.get(_t, 0) + 1
            for _t, _n in _seen.items():
                print(f"     [warn x{_n}] {_t}")
            fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
            st = [s for s in res.stages if not s.get('target')][-1]
            Lk, Mk = st.get('L_out', L), st.get('M_out', M)
            xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
            # residual envelope: strip the parabolic carrier and the tilt ramp
            envk = carrier_referenced_envelope(fld, Rk, LAM, dxk)
            nn = envk.shape[0]
            u = (np.arange(nn) - nn / 2) * dxk
            envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
            envk = _fourier_up(envk, up)
            dxk_u = dxk / up
            nn = envk.shape[0]
            # CROP to the illuminated window before anything else.  Launching
            # the whole co-moving grid (26 mm half-extent for a 6.3 mm beam)
            # wastes 95 % of the rays on amplitude < 1e-15 whose traced OPL is
            # meaningless, and poisons both the 2-D phase unwrap and the
            # sampling diagnostic (measured: a spurious 119-cycle "integrand
            # step" on a grid that is in fact finer than the oracle's).
            wk = _envelope_amp_radius(envk, dxk_u, dxk_u)
            half = int(np.ceil(clip * wk / dxk_u))
            c = nn // 2
            i0 = max(c - half, 0)
            i1 = min(c + half + 1, nn)
            envk = envk[i0:i1, i0:i1]
            nn = envk.shape[0]
            u = (np.arange(nn) - (c - i0)) * dxk_u
            gx, gy, step_ph = _phase_gradient(envk, dxk_u)
            phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
            # decimate to ~NL samples across (exact samples, no interpolation);
            # the envelope is band-limited, so a stride sweep must be flat.
            # DIAGNOSTIC SPLIT (SCOPE study): optionally Gaussian-smooth the
            # amplitude and/or the residual phase before launch.  The true exit
            # field of a diffraction-limited group is smooth on this scale, so
            # if smoothing ONE of them recovers the spot, that component is
            # where the chain's exit field carries spurious structure.
            _asm = float(os.environ.get('AMPSM', '0'))
            _psm = float(os.environ.get('PHSM', '0'))
            if _asm > 0 or _psm > 0:
                from scipy.ndimage import gaussian_filter
                _a = np.abs(envk)
                if _asm > 0:
                    _a = gaussian_filter(_a, _asm)
                if _psm > 0:
                    phres = gaussian_filter(phres, _psm)
                    gx = np.zeros_like(phres)
                    gy = np.zeros_like(phres)
                    gx[:, 1:-1] = (phres[:, 2:] - phres[:, :-2]) / (2 * dxk_u)
                    gy[1:-1, :] = (phres[2:, :] - phres[:-2, :]) / (2 * dxk_u)
                envk = _a * np.exp(1j * phres)
                print(f"     SMOOTHED: amp sigma {_asm} px, phase sigma "
                      f"{_psm} px")
            sd = max(1, int(np.floor(nn / nl0)))
            envk = envk[::sd, ::sd]
            gx, gy, phres = gx[::sd, ::sd], gy[::sd, ::sd], phres[::sd, ::sd]
            u = u[::sd]
            nn = envk.shape[0]
            U, V = np.meshgrid(u, u)
            x0 = (U + xck).ravel()
            y0 = (V + yck).ravel()
            amp = np.abs(envk).ravel()
            p = (U / Rk).ravel() + Lk + gx.ravel() / K0
            q = (V / Rk).ravel() + Mk + gy.ravel() / K0
            ph0 = K0 * ((U ** 2 + V ** 2).ravel() / (2 * Rk)
                        + Lk * U.ravel() + Mk * V.ravel()) + phres.ravel()
            if nc < len(post):
                surfs = C.post_surfaces(post[nc:], back_off=back)
            else:
                surfs = _free_surfaces(C.TRAILING, back)
            nl = nn
            dxs, ws = dxk_u * sd, wk
        r = rs_spot(x0, y0, amp, ph0, p, q, surfs, back, xci, yci,
                    dx_out=dxo, n_out=nout, nl=nl)
        r['step_ph'] = step_ph
        r['dx'] = dxs
        r['w'] = ws
        rows.append((nc, r))
        print(f"  n_chain={nc}  groups {list(range(nc))} via CHAIN, "
              f"{list(range(nc, len(post)))} via ORACLE   "
              f"[{time.time() - t0:.0f}s, {r['n_rays']} rays, "
              f"grid {nl}^2 @ {dxs * 1e6:.3f} um, w {ws * 1e6:.1f} um]")
        print(f"     envelope per-pixel phase step {r['step_ph']:.4f} rad "
              f"({'OK' if r['step_ph'] < 0.5 else 'ALIASED -- launch '
                 'directions unreliable'}); "
              f"integrand step p50/p99.9/max "
              f"{r['step_md']:.4f}/{r['step_w']:.4f}/{r['step_mx']:.3g} cycles "
              f"({'OK' if r['step_w'] < 0.25 else 'CHECK'})")
        print(f"     live power {r['live'] * 100:.4f} %   centroid "
              f"({r['centroid'][0] * 1e6:+.3f},{r['centroid'][1] * 1e6:+.3f}) "
              f"um (chief {xci * 1e6:+.3f},{yci * 1e6:+.3f})")
        print(f"     FWHM {r['fwhm'] * 1e6:6.3f} um   EE3 {r['ee3'] * 100:6.2f}"
              f"   EE6 {r['ee6'] * 100:6.2f}   EE12 {r['ee12'] * 100:6.2f}")
    print()
    print(f"LOCALISATION, order ({m:+d},{n:+d}):  n_chain  FWHM um   EE3 %   "
          f"EE6 %   EE12 %   d(EE3)")
    prev = None
    for nc, r in rows:
        d = '' if prev is None else f"{(r['ee3'] - prev) * 100:+7.2f}"
        prev = r['ee3']
        print(f"  {nc:5d}   {r['fwhm'] * 1e6:7.3f}  {r['ee3'] * 100:6.2f}  "
              f"{r['ee6'] * 100:6.2f}  {r['ee12'] * 100:6.2f}   {d}")


if __name__ == '__main__':
    sys.exit(main())
