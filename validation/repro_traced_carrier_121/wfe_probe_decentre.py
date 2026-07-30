# DECENTRE-SWEPT EXIT WAVEFRONT ERROR of design 121's LAST GROUP element,
# measured against the EXACT ray trace.   (Option C of
# SCOPE_TILTED_COARSE_LEG_TRANSPORT_2026_07_30 section 6.)
#
# WHY THIS IS DIFFERENT FROM last_group_probe_121.py
# --------------------------------------------------
# That probe compared two IMAGE-PLANE fields through a Rayleigh-Sommerfeld
# transport and read a complex overlap, and it under-read the design point by
# 5x.  This one compares the element's OWN RETURNED FIELD, pixel by pixel,
# against the exact-ray exit phase on the SAME nodes.  No transport, no
# integral, no interpolation of the field, no FFT.
#
# THE ARBITER IS COMPLETE ONLY WHEN THE INPUT RESIDUAL IS UNITY
# -------------------------------------------------------------
# ``apply_real_lens_traced``'s returned phase on the shipped chain path
# (amplitude_model='ray_density', preserve_input_phase='remap') is
#
#     arg E_out(X, Y) = k0 * opl_map(X, Y)  +  arg resid(xe(X,Y), ye(X,Y))
#
# where ``resid = E_in * exp(-i k0 W)`` is the carrier-de-chirped input.  So an
# input built as ``A(x,y) * exp(i k0 W(x,y))`` with A REAL and POSITIVE has
# ``resid == 1`` identically, the remap leg contributes EXACTLY 1 (bilinear
# interpolation of a constant unit phasor is that constant), and the arbiter
#
#     Phi(X, Y) = k0 * [ W(xe*, ye*) + OPL_trace(xe*, ye* -> exit vertex) ]
#
# with (xe*, ye*) the EXACT inverse of the traced entrance->exit map is the
# whole answer.  Any difference is then unambiguously the ray-map / fit /
# Newton / OPL-upsample construction.  MODE='real' swaps in the design's own
# residual and needs the residual arbiter (wfe_probe_remap.py).
#
# usage:
#   DEC=0,0.25,0.5,0.75,1.0,1.079,1.25,1.5 python wfe_probe_decentre.py
#   KW='newton_fit=spline'  DEC=0,1.079 python wfe_probe_decentre.py
import os
import sys
import time
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements import _lens_traced as _LT
from lumenairy.elements._lens_traced import TiltedCarrier

LAM = P.LAM
K0 = P.K0

# design 121, last group (Lens S25-S27), order (-4,-2), measured by
# wfe_probe_recon.py from the chain's OWN element call.
R_IN = -0.021139185452405257
L_IN, M_IN = 0.04907347265758019, 0.024536736328790096
X0_D, Y0_D = -0.003016240777531001, -0.0015081203887655004
DX = 33.2112e-6
N = 1024


def parse_kw(s):
    kv = {}
    for item in (s or '').split(';'):
        if not item.strip():
            continue
        a, b = item.split('=', 1)
        b = b.strip()
        try:
            b = int(b)
        except ValueError:
            try:
                b = float(b)
            except ValueError:
                b = {'True': True, 'False': False, 'None': None}.get(b, b)
        kv[a.strip()] = b
    return kv


def build_input(n, dx, w, carrier, extra_phase=None):
    """A(x,y) * exp(i k0 W): residual identically 1 (or exp(i k0 a) if
    ``extra_phase`` is given)."""
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    W, _L, _M = P.carrier_parts(carrier, X, Y)
    u = X - carrier.x0
    v = Y - carrier.y0
    A = np.exp(-(u * u + v * v) / (w * w))
    ph = K0 * W
    if extra_phase is not None:
        ph = ph + K0 * extra_phase(u, v)
    return (A * np.exp(1j * ph)).astype(np.complex128), X, Y


def run_case(w, x0, y0, tilt=True, kw=None, half=72, n=N, dx=DX,
             presc=None, verbose=True, extra_phase=None, amp_check=True):
    L = L_IN if tilt else 0.0
    M = M_IN if tilt else 0.0
    car = TiltedCarrier(R_IN, L, M, x0, y0)
    E_in, X, Y = build_input(n, dx, w, car, extra_phase=extra_phase)
    opts = dict(amplitude_model='ray_density', preserve_input_phase='remap',
                fit_radius_beam_factor=2.0, remap_sampling='full',
                ray_subsample=4, n_workers=8)
    opts.update(kw or {})
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        E_out = np.asarray(apply_real_lens_traced(
            E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
            **opts))
    wmsgs = {}
    for wv in wl:
        t = str(wv.message)[:130]
        wmsgs[t] = wmsgs.get(t, 0) + 1

    surfs = P.element_surfaces(presc)
    xo, yo, _p, _l, _m, _a = P.trace_forward([x0], [y0], car, surfs)
    ic = int(round(float(xo[0]) / dx + n / 2))
    jc = int(round(float(yo[0]) / dx + n / 2))
    sx = slice(max(ic - half, 0), min(ic + half + 1, n))
    sy = slice(max(jc - half, 0), min(jc + half + 1, n))
    Xp, Yp = X[sy, sx], Y[sy, sx]
    Ep = E_out[sy, sx]
    # exact inverse, seeded from a linear fit of a traced lattice
    t = np.linspace(-2.6 * w, 2.6 * w, 21)
    U, V = np.meshgrid(t, t)
    gxo, gyo, _p2, _l2, _m2, alv = P.trace_forward(
        (U + x0).ravel(), (V + y0).ravel(), car, surfs)
    gg = alv & np.isfinite(gxo)
    A3 = np.stack([np.ones(int(gg.sum())), gxo[gg], gyo[gg]], axis=1)
    cxx = np.linalg.lstsq(A3, (U + x0).ravel()[gg], rcond=None)[0]
    cyy = np.linalg.lstsq(A3, (V + y0).ravel()[gg], rcond=None)[0]
    B3 = np.stack([np.ones(Xp.size), Xp.ravel(), Yp.ravel()], axis=1)
    guess = ((B3 @ cxx).reshape(Xp.shape), (B3 @ cyy).reshape(Xp.shape))
    ex = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=guess, n_iter=16)
    amp = np.abs(Ep)
    pk = float(amp.max()) if amp.size else 0.0
    keep = ex['ok'] & (amp > 0.02 * pk) & (ex['resid'] < 1e-9)
    r = P.local_wfe(Ep, ex['phi'], keep, Xp, Yp)
    if r is None:
        return None
    r['warn'] = wmsgs
    r['exact_resid_max'] = float(np.nanmax(ex['resid'][ex['ok']]))
    r['na_exit'] = float(np.nanmax(np.hypot(ex['L'][keep], ex['M'][keep])))
    if amp_check:
        # exact ray-tube amplitude on the same nodes (|E_in| / sqrt|det J|)
        h = 5e-7
        xe, ye = ex['xe'], ex['ye']
        x1, y1, _q, _q2, _q3, _q4 = P.trace_forward(xe + h, ye, car, surfs)
        x2, y2, _q, _q2, _q3, _q4 = P.trace_forward(xe, ye + h, car, surfs)
        x0f, y0f, _q, _q2, _q3, _q4 = P.trace_forward(xe, ye, car, surfs)
        jxx = (x1 - x0f) / h
        jyx = (y1 - y0f) / h
        jxy = (x2 - x0f) / h
        jyy = (y2 - y0f) / h
        det = np.abs(jxx * jyy - jxy * jyx).reshape(Xp.shape)
        uu = xe - x0
        vv = ye - y0
        a_ex = np.exp(-(uu * uu + vv * vv) / (w * w)) / np.sqrt(det)
        s = float((amp[keep] * a_ex[keep]).sum()
                  / max((a_ex[keep] ** 2).sum(), 1e-300))
        r['amp_rel_rms'] = float(np.sqrt(
            ((amp[keep] - s * a_ex[keep]) ** 2).sum()
            / max((amp[keep] ** 2).sum(), 1e-300)))
    return r


def main():
    _pre, post, _g, _per = C.geometry()
    presc = post[-1]['prescription']
    decs = [float(v) for v in os.environ.get(
        'DEC', '0,0.25,0.5,0.75,1.0,1.079,1.25,1.5').split(',')]
    tilts = [int(v) for v in os.environ.get('TILTS', '1').split(',')]
    kw = parse_kw(os.environ.get('KW', ''))
    half = int(os.environ.get('HALF', '72'))
    n = int(os.environ.get('N', str(N)))
    dx = float(os.environ.get('DXG', str(DX * 1e6))) * 1e-6
    # the beam amplitude radius the guard itself measures on the real field
    w = float(os.environ.get('W', '3.1255')) * 1e-3
    az = np.arctan2(Y0_D, X0_D)
    print(f"design 121 LAST GROUP (Lens S25-S27) -- exit WFE vs EXACT RAY "
          f"TRACE, decentre swept")
    print(f"  R_in {R_IN*1e3:.4f} mm  tilt ({L_IN:+.6f},{M_IN:+.6f})  "
          f"w {w*1e3:.4f} mm  grid {n}^2 @ {dx*1e6:.4f} um")
    print(f"  design decentre {np.hypot(X0_D, Y0_D)*1e3:.4f} mm = "
          f"{np.hypot(X0_D, Y0_D)/w:.3f} w   azimuth {np.degrees(az):.2f} deg")
    _kwtxt = repr(kw) if kw else '(shipped chain defaults)'
    print(f"  element kwargs override: {_kwtxt}")
    print(f"  grid Nyquist direction cosine {LAM/(2*dx):.5f}; the exit NA is "
          f"~0.36, i.e. 18x beyond -- the comparison is POINTWISE on the SAME "
          f"nodes, so nothing is interpolated and the sampling-adequacy proof "
          f"is the per-pixel step of the RESIDUAL, printed as nn_step.")
    print()
    hdr = (f"{'tilt':>4} {'dec/w':>7} {'dec(mm)':>8} {'rms(waves)':>11} "
           f"{'piston-only':>12} {'PTV':>8} {'Strehl':>8} {'|amp|err':>9} "
           f"{'nn_step':>8} {'loop':>9} {'npix':>7}")
    print(hdr)
    print('-' * len(hdr))
    for it in tilts:
        for d in decs:
            x0 = d * w * np.cos(az)
            y0 = d * w * np.sin(az)
            t0 = time.time()
            r = run_case(w, x0, y0, tilt=bool(it), kw=kw, half=half, n=n,
                         dx=dx, presc=presc)
            if r is None:
                print(f"{it:>4} {d:>7.3f}  (no pixels)")
                continue
            print(f"{it:>4} {d:>7.3f} {np.hypot(x0,y0)*1e3:>8.4f} "
                  f"{r['rms_waves']:>11.5f} {r['rms_piston_only']:>12.5f} "
                  f"{r['ptv_waves']:>8.4f} {r['strehl_marechal']:>8.4f} "
                  f"{r.get('amp_rel_rms', float('nan')):>9.4f} "
                  f"{r['nn_step_rad']:>8.4f} {r['loop_rad']:>9.1e} "
                  f"{r['n_pix']:>7d}   [{time.time()-t0:.0f}s]")
            for t, c in sorted(r['warn'].items()):
                if 'Newton inversion' in t or 'did not converge' in t:
                    print(f"        [warn x{c}] {t[:110]}")
    print()
    print("nn_step = max wrapped nearest-neighbour step of the RESIDUAL "
          "(element minus exact), rad.  It must sit far below pi = 3.1416 or "
          "the unwrap is not single-valued and the rms is meaningless.")
    print("loop    = max |curl| of the wrapped-difference field: 0 proves the "
          "unwrap is path-independent.")


if __name__ == '__main__':
    sys.exit(main())
