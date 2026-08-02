# RECON for DIAG_LAST_GROUP_DECENTRE: capture design 121's REAL last-group
# element call and measure its exit wavefront error against the EXACT ray
# trace, at the design's own decentre.
#
# usage:  ORD=-4,-2 HALF=72 python wfe_probe_recon.py
import os
import sys
import time

import numpy as np

import wfe_probe_common as P
import _d121_common as C

LAM = P.LAM
K0 = P.K0


def describe_call(rec, idx):
    kw = rec['kw']
    car = kw.get('carrier')
    E_in = rec['E_in']
    dx = float(kw.get('dx'))
    N = E_in.shape[0]
    presc = kw.get('prescription')
    ap = presc.get('aperture_diameter')
    sub = int(kw.get('ray_subsample', 4))
    lr = 0.5 * ap * 1.5 if ap is not None else 0.5 * N * dx
    n_launch = max(8, int(2 * lr / (dx * sub)))
    if n_launch % 2 == 0:
        n_launch += 1
    amp = np.abs(E_in)
    tot = float((amp ** 2).sum())
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    cx = float((amp ** 2 * Xg).sum() / tot)
    cy = float((amp ** 2 * Yg).sum() / tot)
    w2 = float((amp ** 2 * ((Xg - cx) ** 2 + (Yg - cy) ** 2)).sum() / tot)
    print(f"--- element call {idx}: {presc.get('name', '?')}")
    print(f"    N={N} dx={dx*1e6:.4f} um  half-extent {N/2*dx*1e3:.3f} mm")
    print(f"    carrier = {car}")
    if car is not None and hasattr(car, 'R'):
        print(f"      R_in {car.R*1e3:.4f} mm  tilt ({car.L:+.6f},{car.M:+.6f})"
              f"  centre ({car.x0*1e3:+.4f},{car.y0*1e3:+.4f}) mm "
              f"= {np.hypot(car.x0, car.y0)*1e3:.4f} mm")
    print(f"    aperture_diameter = "
          f"{'None' if ap is None else f'{ap*1e3:.4f} mm'}"
          f"   launch_radius {lr*1e3:.4f} mm  n_launch {n_launch}  sub {sub}")
    print(f"    |E_in| centroid ({cx*1e3:+.4f},{cy*1e3:+.4f}) mm   "
          f"2nd-moment radius {np.sqrt(w2)*1e3:.4f} mm")
    print(f"    kwargs: " + ", ".join(
        f"{k}={v!r}" for k, v in sorted(kw.items())
        if k not in ('prescription', 'carrier', 'dx', 'wavelength')))
    return dict(N=N, dx=dx, presc=presc, carrier=car, launch_radius=lr,
                n_launch=n_launch, sub=sub, cx=cx, cy=cy)


def measure(rec, info, half=72, label=''):
    """Exact-trace wavefront error of the returned field over a patch around
    the exit beam."""
    E_out = rec['E_out']
    car = info['carrier']
    dx, N = info['dx'], info['N']
    surfs = P.element_surfaces(info['presc'])
    # exit chief-ray position
    xo, yo, _psi, Lo, Mo, _a = P.trace_forward([car.x0], [car.y0], car, surfs)
    xc_o, yc_o = float(xo[0]), float(yo[0])
    ic = int(round(xc_o / dx + N / 2))
    jc = int(round(yc_o / dx + N / 2))
    sl_x = slice(max(ic - half, 0), min(ic + half + 1, N))
    sl_y = slice(max(jc - half, 0), min(jc + half + 1, N))
    ax = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(ax, ax)
    Xp, Yp = Xg[sl_y, sl_x], Yg[sl_y, sl_x]
    Ep = np.asarray(E_out)[sl_y, sl_x]
    print(f"    exit chief ray ({xc_o*1e3:+.4f},{yc_o*1e3:+.4f}) mm  "
          f"exit cosines ({float(Lo[0]):+.5f},{float(Mo[0]):+.5f})")
    print(f"    patch {Xp.shape} covering "
          f"[{Xp.min()*1e3:+.3f},{Xp.max()*1e3:+.3f}] x "
          f"[{Yp.min()*1e3:+.3f},{Yp.max()*1e3:+.3f}] mm")
    t0 = time.time()
    # initial guess: paraxial-ish scale from the chief plus a linear fit made
    # from a small traced lattice
    g = _guess(Xp, Yp, car, surfs, info)
    ex = P.exact_phase_on_nodes(Xp, Yp, car, surfs, guess=g, n_iter=14,
                                verbose=False)
    print(f"    exact inverse: {time.time()-t0:.1f} s   max|resid| "
          f"{np.nanmax(np.where(ex['ok'], ex['resid'], 0)):.3e} m  "
          f"median {np.nanmedian(ex['resid'][ex['ok']]):.3e} m")
    amp = np.abs(Ep)
    pk = float(amp.max())
    for thr in (0.02, 0.05):
        keep = ex['ok'] & (amp > thr * pk) & (ex['resid'] < 1e-9)
        r = P.local_wfe(Ep, ex['phi'], keep, Xp, Yp)
        if r is None:
            print(f"    [thr {thr}] no pixels")
            continue
        print(f"    [amp>{thr:.2f}pk  n={r['n_pix']}]  rms {r['rms_waves']:.5f}"
              f" waves (piston+tilt out) | {r['rms_piston_only']:.5f} (piston "
              f"only) | PTV {r['ptv_waves']:.4f} | Strehl "
              f"{r['strehl_marechal']:.4f} | phase-fidelity "
              f"{r['fidelity_phase']:.5f}")
        print(f"          SAMPLING: max wrapped NN step of the RESIDUAL "
              f"{r['nn_step_rad']:.4f} rad (pi = 3.1416); loop residual "
              f"{r['loop_rad']:.2e} rad")
    return ex, Ep, Xp, Yp


def _guess(Xp, Yp, car, surfs, info):
    """Initial entrance guess from a coarse traced lattice + linear inverse."""
    w = 3.2e-3
    t = np.linspace(-2.5 * w, 2.5 * w, 21)
    U, V = np.meshgrid(t, t)
    xe = (U + car.x0).ravel()
    ye = (V + car.y0).ravel()
    xo, yo, _p, _l, _m, alive = P.trace_forward(xe, ye, car, surfs)
    g = alive & np.isfinite(xo)
    A = np.stack([np.ones(g.sum()), xo[g], yo[g]], axis=1)
    cxx = np.linalg.lstsq(A, xe[g], rcond=None)[0]
    cyy = np.linalg.lstsq(A, ye[g], rcond=None)[0]
    B = np.stack([np.ones(Xp.size), Xp.ravel(), Yp.ravel()], axis=1)
    return (B @ cxx).reshape(Xp.shape), (B @ cyy).reshape(Xp.shape)


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    half = int(os.environ.get('HALF', '72'))
    ng = int(os.environ.get('NG', '6'))
    t0 = time.time()
    calls, res, msgs = P.run_chain_capture(m=m, n=n, n_groups=ng)
    print(f"chain captured {len(calls)} element calls in {time.time()-t0:.0f}s"
          f"   order ({m},{n})")
    for t, c in sorted(msgs.items()):
        print(f"  [chain warn x{c}] {t}")
    print()
    infos = []
    for i, rec in enumerate(calls):
        infos.append(describe_call(rec, i))
    print()
    which = [int(v) for v in os.environ.get('WHICH',
                                            str(len(calls) - 1)).split(',')]
    for i in which:
        print(f"=== WFE vs EXACT TRACE, element call {i} "
              f"({infos[i]['presc'].get('name', '?')}) ===")
        measure(calls[i], infos[i], half=half)
        print()


if __name__ == '__main__':
    sys.exit(main())
