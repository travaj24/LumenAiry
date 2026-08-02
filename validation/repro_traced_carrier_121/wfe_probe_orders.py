# The element's last-group exit error vs the EXACT ray trace, ACROSS THE DOE
# FAN -- the experiment that decides whether the mechanism is DECENTRE or the
# size of the transported input RESIDUAL.
# (DIAG_LAST_GROUP_DECENTRE_2026_07_30.)
#
# For each order this runs design 121's real post-DOE chain, captures the REAL
# ``apply_real_lens_traced`` call the chain makes on the last group (input,
# carrier and returned field -- no synthetic stand-in anywhere), and reports
#
#   RAYMAP   arg(E_out with the residual stripped from the INPUT) vs the exact
#            carrier ray trace  -- the fit / Newton / OPL-upsample leg
#   RESID    the element's transported residual phasor vs a band-limited
#            evaluation of the same residual at the exact pullback point
#            -- the 'remap' IMPLEMENTATION
#   TOTAL    the shipped field vs rays launched along grad(W + a)
#            -- the exact-ray oracle, i.e. IMPLEMENTATION + MODEL
#   AMP      relative rms of |E_out| against the oracle's ray-tube amplitude
#   grad a   amplitude-weighted rms of the residual's OWN transverse direction
#            cosines at the entrance -- the quantity the model drops
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python wfe_probe_orders.py
#         HALF=72,96,120 THRESH=0.02 python wfe_probe_orders.py     (convergence)
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
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '_wfe_probe_orders_%s.npz')


def get_call(m, n, rn=1024, rs=4):
    fn = CACHE % f"{m}_{n}_{rn}_{rs}"
    if os.path.exists(fn):
        d = np.load(fn)
        return d['E_in'], d['E_out'], tuple(d['car']), float(d['dx'])
    calls, _res, _msg = P.run_chain_capture(m=m, n=n, rn=rn, rs=rs)
    rec = calls[-1]
    car = rec['kw']['carrier']
    if not hasattr(car, 'R'):
        # the on-axis order has zero tilt AND zero decentre, so the chain
        # takes its NON-tilted branch and hands a scalar conjugate.  The
        # eikonal is identical to TiltedCarrier(R, 0, 0, 0, 0) (both are
        # _compute_carrier's exact sphere), so record it in that form.
        car = TiltedCarrier(float(car), 0.0, 0.0, 0.0, 0.0)
    dx = float(rec['kw']['dx'])
    np.savez_compressed(fn, E_in=rec['E_in'], E_out=rec['E_out'], dx=dx,
                        car=np.array([car.R, car.L, car.M, car.x0, car.y0]))
    return rec['E_in'], rec['E_out'], (car.R, car.L, car.M, car.x0, car.y0), dx


def analyse(m, n, half=96, thresh=0.02, up=8, rn=1024, rs=4, do_raymap=True):
    E_in, E_out, carv, dx = get_call(m, n, rn=rn, rs=rs)
    car = TiltedCarrier(*carv)
    N = E_in.shape[0]
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    W, _l, _m = P.carrier_parts(car, Xg, Yg)
    resid = np.asarray(E_in) * np.exp(-1j * K0 * W)
    _pre, post, _g, _p = C.geometry()
    presc = post[-1]['prescription']
    surfs = P.element_surfaces(presc)
    # beam radius about the carrier centre (the library's own measure)
    from lumenairy.elements._lens_traced import _input_beam_amp_radius
    w = float(_input_beam_amp_radius(E_in, dx, dx, centre=(car.x0, car.y0)))
    dec = float(np.hypot(car.x0, car.y0))
    rf = RM.ResidualField(resid, dx, car.x0, car.y0, 3.4 * w, up=up)

    # residual's own transverse direction cosines over the beam
    a_amp = np.abs(resid)
    br = a_amp > 0.05 * a_amp.max()
    _a, gLa, gMa = rf.ev(Xg[br], Yg[br])
    wq = a_amp[br] ** 2
    g_rms = float(np.sqrt((wq * (gLa ** 2 + gMa ** 2)).sum() / wq.sum()))

    # exit patch about the chief ray
    xo, yo, _q, _q2, _q3, _q4 = P.trace_forward([car.x0], [car.y0], car, surfs)
    ic = int(round(float(xo[0]) / dx + N / 2))
    jc = int(round(float(yo[0]) / dx + N / 2))
    sx = slice(max(ic - half, 0), min(ic + half + 1, N))
    sy = slice(max(jc - half, 0), min(jc + half + 1, N))
    Xp, Yp, Ep = Xg[sy, sx], Yg[sy, sx], np.asarray(E_out)[sy, sx]
    p_patch = float((np.abs(Ep) ** 2).sum()
                    / max((np.abs(E_out) ** 2).sum(), 1e-300))

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

    exA = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=guess, n_iter=16)
    exB = RM.invert_total(Xp, Yp, car, surfs, rf, guess)
    amp = np.abs(Ep)
    pk = float(amp.max())
    keep = (exA['ok'] & exB['ok'] & (amp > thresh * pk)
            & (exA['resid'] < 1e-9) & (exB['resid'] < 1e-9))
    p_keep = float((amp[keep] ** 2).sum()
                   / max((np.abs(E_out) ** 2).sum(), 1e-300))
    out = {'m': m, 'n': n, 'w': w, 'dec': dec, 'dec_w': dec / w,
           'grad_a_rms': g_rms, 'p_patch': p_patch, 'p_keep': p_keep,
           'dx': dx, 'na_exit': float(np.nanmax(
               np.hypot(exA['L'][keep], exA['M'][keep])))}
    rtot = P.local_wfe(Ep, exB['phi'], keep, Xp, Yp)
    out['total'] = rtot
    # amplitude: oracle ray-tube amplitude on the same nodes
    h = 5e-7
    xe, ye = exB['xe'], exB['ye']
    x1, y1, _q, _q2, _q3, _q4 = RM.trace_total(xe + h, ye, car, surfs, rf)
    x2, y2, _q, _q2, _q3, _q4 = RM.trace_total(xe, ye + h, car, surfs, rf)
    x0f, y0f, _q, _q2, _q3, _q4 = RM.trace_total(xe, ye, car, surfs, rf)
    det = np.abs((x1 - x0f) * (y2 - y0f) - (x2 - x0f) * (y1 - y0f)).reshape(
        Xp.shape) / (h * h)
    a_src = np.abs(rf_amp(rf, xe, ye))
    a_ex = a_src / np.sqrt(np.maximum(det, 1e-300))
    s = float((amp[keep] * a_ex[keep]).sum()
              / max((a_ex[keep] ** 2).sum(), 1e-300))
    out['amp_rel_rms'] = float(np.sqrt(
        ((amp[keep] - s * a_ex[keep]) ** 2).sum()
        / max((amp[keep] ** 2).sum(), 1e-300)))
    if do_raymap:
        opts = dict(amplitude_model='ray_density',
                    preserve_input_phase='remap', fit_radius_beam_factor=2.0,
                    remap_sampling='full', ray_subsample=rs, n_workers=8)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            oA = np.asarray(apply_real_lens_traced(
                (np.abs(resid) * np.exp(1j * K0 * W)).astype(np.complex128),
                prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                **opts))
        EA = oA[sy, sx]
        kA = exA['ok'] & (np.abs(EA) > thresh * float(np.abs(EA).max())) \
            & (exA['resid'] < 1e-9)
        out['raymap'] = P.local_wfe(EA, exA['phi'], kA, Xp, Yp)
        Q = np.where(np.abs(EA) > 0, Ep / np.where(np.abs(EA) > 0, EA, 1.0),
                     0.0)
        out['resid'] = P.local_wfe(Q * amp, rf.phase(exA['xe'], exA['ye']),
                                   keep, Xp, Yp)
    return out


def rf_amp(rf, x, y):
    return rf.amp_at(x, y)


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get(
                  'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()]
    halves = [int(v) for v in os.environ.get('HALF', '96').split(',')]
    thresh = float(os.environ.get('THRESH', '0.02'))
    up = int(os.environ.get('UP', '8'))
    do_rm = int(os.environ.get('RAYMAP', '1'))
    print("design 121 LAST GROUP (Lens S25-S27): exit error vs the EXACT ray "
          "trace, ACROSS THE DOE FAN")
    print("  every row is the chain's OWN element call for that order "
          "(captured, not synthesised)")
    print()
    hdr = (f"{'order':>8} {'dec/w':>6} {'w(mm)':>7} {'NAexit':>7} "
           f"{'grad a rms':>11} {'RAYMAP':>8} {'RESID':>8} {'TOTAL':>8} "
           f"{'Strehl':>7} {'AMPerr':>7} {'Ppatch':>7} {'Pkeep':>7} "
           f"{'nn p99':>7}")
    for half in halves:
        print(f"--- HALF={half} px (patch half-width "
              f"{half*33.2112e-3:.2f} mm), amp threshold {thresh}")
        print(hdr)
        print('-' * len(hdr))
        for (m, n) in orders:
            t0 = time.time()
            r = analyse(m, n, half=half, thresh=thresh, up=up,
                        do_raymap=bool(do_rm))
            rm = r.get('raymap')
            rr = r.get('resid')
            print(f"{f'({m},{n})':>8} {r['dec_w']:>6.3f} {r['w']*1e3:>7.4f} "
                  f"{r['na_exit']:>7.4f} {r['grad_a_rms']:>11.6f} "
                  f"{(rm['rms_waves'] if rm else float('nan')):>8.5f} "
                  f"{(rr['sigma_fid_waves'] if rr else float('nan')):>8.5f} "
                  f"{r['total']['sigma_fid_waves']:>8.5f} "
                  f"{r['total']['fidelity_notilt']:>7.4f} "
                  f"{r['amp_rel_rms']:>7.4f} {r['p_patch']:>7.4f} "
                  f"{r['p_keep']:>7.4f} "
                  f"{r['total']['nn_step_p99']:>7.4f}   "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print()
    print("grad a rms = amplitude-weighted rms of the INPUT residual's own "
          "transverse direction cosines (rad).  This is the quantity")
    print("             preserve_input_phase='remap' drops: it launches along "
          "grad(W) alone and evaluates the residual at THAT ray's foot.")
    print("TOTAL      = unwrap-free equivalent rms wavefront error of the "
          "SHIPPED field vs rays launched along grad(W + a), waves.")
    print("Ppatch/Pkeep = fraction of the element's whole-grid exit power "
          "inside the patch / inside the measured mask.")


if __name__ == '__main__':
    sys.exit(main())
