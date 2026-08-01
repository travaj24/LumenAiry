# Niche C6: does launching along grad(W + a_fit) remove the element's
# second-order stationary-phase error?
#
# The measurement is ELEMENT-ONLY and DIFFERENTIAL.  For each DOE order the
# chain's REAL group-5 element call is captured ONCE with the C6 flag OFF (so
# the input, the carrier and the decentre are exactly the post-C5 shipped
# ones), and then the SAME input is pushed through ``apply_real_lens_traced``
# under a list of configurations -- flag OFF, then flag ON at various fit
# degrees / taper radii -- and every one is scored against the exact-ray
# oracle of wfe_probe_remap (rays launched along grad(W + a) with the FULL
# band-limited residual, Newton-inverted onto the same exit nodes).
#
# Nothing is interpolated, FFT-differentiated or unwrapped in the comparison:
# ``wfe_probe_common.local_wfe`` de-references pointwise and reports both an
# unwrap-based rms and an unwrap-FREE fidelity, plus the power-weighted
# nearest-neighbour step distribution that says whether either is legitimate.
#
# Two ADVERSARIAL columns are reported alongside the wavefront error, because
# a launch perturbation that is not contained can manufacture a D1-class ghost
# lobe (a far exit pixel whose Newton pullback lands back inside the bright
# beam and is handed real ray-density amplitude):
#   P/Pin   total exit power over input power
#   ghost   the fraction of the input power landing MORE than 4 mm from the
#           exit chief ray -- 0.000 on the shipped path
#
# usage:  ORDERS='0,0 -2,0 -4,0 -4,-2' python probe_c6_element.py
#         ORDERS='-4,-2' CASES='off,deg2,deg4,deg6,t1.5-2.0' \
#             python probe_c6_element.py
import os
import sys
import time
import warnings

import numpy as np

import wfe_probe_common as P
import _d121_common as C
import wfe_probe_remap as RM
from lumenairy.elements import apply_real_lens_traced
import lumenairy.elements._lens_traced as LT
from lumenairy.elements._lens_traced import (TiltedCarrier,
                                             _input_beam_amp_radius)

LAM = P.LAM
K0 = P.K0
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '_c6_call_%s.npz')

OPTS = dict(amplitude_model='ray_density', preserve_input_phase='remap',
            fit_radius_beam_factor=2.0, remap_sampling='full', n_workers=8)


def get_call(m, n, rn=1024, rs=4):
    """The chain's group-5 element call, captured with C6 OFF (= post-C5
    shipped).  Cached under a C6-specific name so it can never be confused
    with wfe_probe_orders' own cache."""
    fn = CACHE % f"{m}_{n}_{rn}_{rs}"
    if os.path.exists(fn):
        d = np.load(fn)
        return d['E_in'], d['E_out'], tuple(d['car']), float(d['dx'])
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    LT.REMAP_STATIONARY_PHASE_LAUNCH = False
    try:
        calls, _res, _msg = P.run_chain_capture(m=m, n=n, rn=rn, rs=rs)
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old
    rec = calls[-1]
    car = rec['kw']['carrier']
    if not hasattr(car, 'R'):
        car = TiltedCarrier(float(car), 0.0, 0.0, 0.0, 0.0)
    dx = float(rec['kw']['dx'])
    np.savez_compressed(fn, E_in=rec['E_in'], E_out=rec['E_out'], dx=dx,
                        car=np.array([car.R, car.L, car.M, car.x0, car.y0]))
    return rec['E_in'], rec['E_out'], (car.R, car.L, car.M, car.x0, car.y0), dx


def element(E_in, presc, dx, car, rs, flag, deg=None, tin=None, tout=None,
            fitw=None, **over):
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
           LT._REMAP_RESID_FIT_W, LT._REMAP_RESID_DEGREE_CAP)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = flag
    if deg is not None:
        LT._REMAP_RESID_EIKONAL_DEGREE = int(deg)
        LT._REMAP_RESID_DEGREE_CAP = max(int(deg), LT._REMAP_RESID_DEGREE_CAP)
    if fitw is not None:
        LT._REMAP_RESID_FIT_W = float(fitw)
    d = {}
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            out = np.asarray(apply_real_lens_traced(
                E_in, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                ray_subsample=rs, _remap_launch_out=d, **OPTS, **over))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT._REMAP_RESID_EIKONAL_DEGREE,
         LT._REMAP_RESID_FIT_W, LT._REMAP_RESID_DEGREE_CAP) = old
    flags = {'fold': 0, 'energy': 0}
    for _w in wl:
        _t = str(_w.message)
        if 'fold caustic' in _t:
            flags['fold'] += 1
        if 'energy self-check FAILED' in _t:
            flags['energy'] += 1
    return out, d, flags


def parse_cases(spec):
    """'off,deg4,t1.5-2.0,deg4@t1.5-2.0' -> list of (label, kwargs)."""
    out = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if item == 'off':
            out.append(('OFF (shipped)', dict(flag=False)))
            continue
        kw = dict(flag=True)
        bits = []
        for part in item.split('@'):
            if part.startswith('deg'):
                kw['deg'] = int(part[3:])
                bits.append(f"deg{kw['deg']}")
            elif part.startswith('f'):
                kw['fitw'] = float(part[1:])
                bits.append(f"f{kw['fitw']}w")
            elif part in ('on', ''):
                bits.append('default')
            else:
                raise ValueError(f"bad case {item!r}")
        out.append(('ON ' + ' '.join(bits), kw))
    return out


def analyse(m, n, cases, half=96, thresh=0.02, up=8, rn=1024, rs=4):
    E_in, _E_out_chain, carv, dx = get_call(m, n, rn=rn, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _m = P.carrier_parts(car, Xg, Yg)
    resid = np.asarray(E_in) * np.exp(-1j * K0 * W)
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    rf = RM.ResidualField(resid, dx, car.x0, car.y0, 3.4 * w, up=up)

    # the residual's own ray slope over the beam -- the probe's OWN measure,
    # independent of the library's fit
    a_amp = np.abs(resid)
    br = a_amp > 0.05 * a_amp.max()
    _a, gLa, gMa = rf.ev(Xg[br], Yg[br])
    wq = a_amp[br] ** 2
    g_rms = float(np.sqrt((wq * (gLa ** 2 + gMa ** 2)).sum() / wq.sum()))

    # exit patch about the chief ray
    xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car, surfs)
    xc_e, yc_e = float(xo[0]), float(yo[0])
    ic = int(round(xc_e / dx + N / 2))
    jc = int(round(yc_e / dx + N / 2))
    sx = slice(max(ic - half, 0), min(ic + half + 1, N))
    sy = slice(max(jc - half, 0), min(jc + half + 1, N))
    Xp, Yp = Xg[sy, sx], Yg[sy, sx]
    far = np.hypot(Xg - xc_e, Yg - yc_e) > 4.0e-3

    # linear seed for the exact inversion
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

    p_in = float((np.abs(E_in) ** 2).sum())
    rows = []
    # NULL INTERVENTION: two identical flag-OFF runs, establishing the
    # differential floor before any delta is quoted.
    off1, _d1, _f1 = element(E_in, presc, dx, car, rs, False)
    off2, _d2, _f2 = element(E_in, presc, dx, car, rs, False)
    null_eq = bool(np.array_equal(off1, off2))
    null_max = float(np.abs(off1 - off2).max())
    for lbl, kw in cases:
        F, dg, fl = element(E_in, presc, dx, car, rs, **kw)
        Ep = F[sy, sx]
        amp = np.abs(Ep)
        pk = float(amp.max())
        keep = exB['ok'] & (amp > thresh * pk) & (exB['resid'] < 1e-9)
        r = P.local_wfe(Ep, exB['phi'], keep, Xp, Yp)
        pw = np.abs(F) ** 2
        rows.append({'label': lbl, 'wfe': r, 'diag': dg, 'flags': fl,
                     'p_keep': float((amp[keep] ** 2).sum() / p_in),
                     'power': float(pw.sum()) / p_in,
                     'ghost': float(pw[far].sum()) / p_in})
    return {'m': m, 'n': n, 'w': w,
            'dec_w': float(np.hypot(car.x0, car.y0)) / w,
            'grad_a_rms': g_rms, 'rows': rows, 'null_eq': null_eq,
            'null_max': null_max}


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -4,0 -4,-2').split()]
    half = int(os.environ.get('HALF', '96'))
    thresh = float(os.environ.get('THRESH', '0.02'))
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))
    cases = parse_cases(os.environ.get('CASES', 'off,on'))
    print("niche C6: the element's last-group exit error vs the EXACT ray "
          "oracle (rays along grad(W + a)).")
    print("          Input, carrier and decentre are the chain's own, "
          "captured with C6 OFF (= post-C5 shipped).")
    print()
    hdr = (f"{'order':>8} {'config':>18} {'grad a':>9} {'g resid':>9} "
           f"{'deg':>4} {'gmax/rms':>9} {'eng':>4} {'sigmaF':>8} "
           f"{'Strehl':>8} {'nn p99':>7} {'loop':>9} {'P/Pin':>7} "
           f"{'ghost':>8} {'fold':>5}")
    print(hdr)
    print('-' * len(hdr))
    for (m, n) in orders:
        t0 = time.time()
        r = analyse(m, n, cases, half=half, thresh=thresh, rs=rs, rn=rn)
        for row in r['rows']:
            w_ = row['wfe']
            dg = row['diag']
            _ratio = (dg.get('grad_a_fit_max_launch', float('nan'))
                      / max(dg.get('grad_a_rms', float('nan')), 1e-300))
            print(f"{f'({m},{n})':>8} {row['label']:>18} "
                  f"{r['grad_a_rms']:>9.6f} "
                  f"{dg.get('grad_a_residual_rms', float('nan')):>9.6f} "
                  f"{str(dg.get('degree', '-')):>4} {_ratio:>9.2f} "
                  f"{str(dg.get('engaged', '-'))[:4]:>4} "
                  f"{w_['sigma_fid_waves']:>8.5f} "
                  f"{w_['fidelity_notilt']:>8.4f} "
                  f"{w_['nn_step_p99']:>7.4f} {w_['loop_rad']:>9.2e} "
                  f"{row['power']:>7.4f} {row['ghost']:>8.5f} "
                  f"{row['flags']['fold']:>5d}")
        print(f"         [null intervention: two OFF runs array_equal="
              f"{r['null_eq']} max|dE|={r['null_max']:.3e}]  "
              f"dec {r['dec_w']:.3f} w, w {r['w']*1e3:.4f} mm  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        print()
    print("sigmaF  = UNWRAP-FREE equivalent rms wavefront error (waves) from "
          "the piston+tilt-removed coherent fidelity.")
    print("          Trustworthy only where nn p99 << pi = 3.1416; the "
          "unwrap-based twin additionally needs loop ~ 0.")
    print("g resid = rms of grad(a - a_fit), what the polynomial model MISSES "
          "-- the leftover error scales with its SQUARE.")
    print("ghost   = input-power fraction landing > 4 mm from the exit chief "
          "ray (the D1 spurious-lobe failure mode).")


if __name__ == '__main__':
    sys.exit(main())
