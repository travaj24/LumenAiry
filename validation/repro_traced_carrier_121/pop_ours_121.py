# POP CROSSCHECK, step 4: OUR two image-plane profiles per DOE order, on the
# SAME lattice the POP grids will be resampled onto.
#
#   method=oracle -> exact skew ray trace + Rayleigh-Sommerfeld
#                    (exact_ray_oracle_121.oracle_spot, unmodified)
#   method=chain  -> propagate_traced_carrier_chain in the SHIPPED
#                    configuration (no CREF / AM / PIP overrides at all, so it
#                    measures the v5.29 default triple + niches C4/C5/C6 as
#                    they sit in the working tree), tilted carrier per order,
#                    final_leg='auto', trailing leg included.
#
# COMMON LATTICE.  dx_out = 0.1 um, N_out = 401 -> +/-20.0 um about each
# order's OWN chief-ray intercept.  Chosen because (a) 0.1 um is the oracle's
# own production pixel and resolves a 3.4 um FWHM with 34 points, (b) +/-20 um
# holds the EE12 circle with 8 um of margin so the log panel shows real halo
# rather than a window edge, and (c) every POP run here has a pixel finer than
# 0.1 um (0.033 um at the production setting), so POP is DOWN-sampled onto it
# and never interpolated up.
#
# Each order's grid is centred on that order's chief ray, which is what makes
# the three methods comparable: POP centres its array on the chief ray too.
#
# usage:
#   python pop_ours_121.py --method oracle --orders 0,0 -1,0 -2,0 -3,0 -4,0 -4,-2
import argparse
import json
import os
import sys
import time

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'pop_profiles')
DX_OUT = 0.1e-6
N_OUT = 401


def parse_orders(vals):
    out = []
    for v in vals:
        a, b = v.split(',')
        out.append((int(a), int(b)))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('oracle', 'chain', 'chain6'),
                   required=True,
                   help="'chain6' is the campaign's own per-order instrument "
                        "(hybrid_localize_121 at n_chain = 6): the traced "
                        "carrier chain does ALL SIX post-DOE groups with the "
                        "shipped defaults, and only the final free-space leg "
                        "+ readout is finished by the exact-ray + "
                        "Rayleigh-Sommerfeld propagator.  The reference "
                        "numbers in the brief (0,0 89.21 / -4,0 88.94 / "
                        "-4,-2 88.49) ARE this instrument, so it is the "
                        "chain panel that can actually be compared with them. "
                        "'chain' is the pure chain all the way to the image "
                        "plane, which on a tilted order drops 37 %% of the "
                        "power outside a +/-20 um readout and is NOT the "
                        "configuration those numbers came from.")
    p.add_argument('--orders', nargs='+',
                   default=['0,0', '-1,0', '-2,0', '-3,0', '-4,0', '-4,-2'])
    p.add_argument('--rn', type=int, default=2048)
    p.add_argument('--rs', type=int, default=4)
    p.add_argument('--nl', type=int, default=161, help='oracle launch grid')
    p.add_argument('--clip', type=float, default=3.0)
    p.add_argument('--back', type=float, default=5.0, help='oracle RS back-off, mm')
    p.add_argument('--dxo', type=float, default=DX_OUT * 1e6)
    p.add_argument('--nout', type=int, default=N_OUT)
    p.add_argument('--nfc', type=int, default=8192)
    p.add_argument('--wf', type=float, default=4.0)
    p.add_argument('--nw', type=int, default=8)
    p.add_argument('--leg', default='paraxial', choices=('auto', 'paraxial',
                                                         'exact'),
                   help="chain final leg.  'paraxial' is what the per-order "
                        "runner fan_multi_121.py defaults to and what produced "
                        "the campaign's per-order EE3 table; 'auto' takes the "
                        "exact high-NA leg, which on a chief-ray-displaced "
                        "order needs an axis-centred retrace window wide "
                        "enough to hold BOTH the axis and the displaced beam "
                        "and costs >10 min per order here.")
    a = p.parse_args(argv)
    os.makedirs(OUTDIR, exist_ok=True)
    dxo = a.dxo * 1e-6

    import _d121_common as C
    from exact_ray_oracle_121 import oracle_spot
    from lumenairy.raytrace import RayBundle, trace

    lam = C.LAM
    _pre, post, _g, period = C.geometry()
    env_doe, R_doe, dx_doe, P_in_chainA = C.chain_a(n=a.rn, rs=a.rs)
    print(f"chain A: env {env_doe.shape} dx {dx_doe * 1e6:.4f} um  "
          f"R {R_doe * 1e3:.4f} mm  P_in {P_in_chainA:.6e}")
    print(f"DOE period {period * 1e6:.4f} um  -> 1 order = "
          f"{lam / period * 1e3:.4f} mrad")

    for (m, n) in parse_orders(a.orders):
        L = m * lam / period
        M = n * lam / period
        # this order's chief ray at the image plane
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=lam, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), lam,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        t0 = time.time()
        tag = (f"{a.method}_m{m}n{n}".replace('-', 'm')
               + ('' if a.method == 'oracle' or a.leg == 'paraxial'
                  else '_' + a.leg))
        print(f"\n=== order ({m:+d},{n:+d})  tilt ({L * 1e3:+.3f},"
              f"{M * 1e3:+.3f}) mrad  chief ({xci * 1e6:+.2f},"
              f"{yci * 1e6:+.2f}) um  [{a.method}] ===", flush=True)

        extra = {}
        P_doe = float(np.sum(np.abs(env_doe) ** 2)) * dx_doe * dx_doe
        if a.method == 'oracle':
            r = oracle_spot(env_doe, R_doe, dx_doe, post, L, M,
                            n_launch=a.nl, clip=a.clip, back=a.back * 1e-3,
                            dx_out=dxo, n_out=a.nout, carry_phase=False,
                            verbose=True, prefix='  ')
            I = np.asarray(r['I'])
            P_win = float(I.sum()) * dxo * dxo
            extra = {k: (float(v) if np.isscalar(v) or isinstance(v, float)
                         else None)
                     for k, v in r.items() if k not in ('I', 'ax', 'centre',
                                                        'centroid')}
            # oracle_spot's ``launch_power`` is sum(amp^2 * cos) WITHOUT the
            # launch-cell area, while its integrand weight carries h^2 -- so
            # the physical launched power needs the h^2 back.
            h = 2.0 * a.clip * float(r['w_doe']) / (a.nl - 1)
            extra['P_launch'] = float(r['launch_power']) * h * h
            extra['P_window'] = P_win
            extra['P_ratio'] = P_win / extra['P_launch']
            extra['P_doe'] = P_doe
            extra['P_launch_over_doe'] = extra['P_launch'] / P_doe
            extra['live_frac'] = float(r['live_frac'])
        elif a.method == 'chain6':
            # VERBATIM the hybrid_localize_121 nc = 6 path: chain over all six
            # post-DOE groups (final_distance = 0, paraxial hand-off plane),
            # then the residual envelope is re-launched as exact rays and the
            # trailing leg + readout is done by the same RS integral the oracle
            # uses -- so chain and oracle differ ONLY in who did groups 0..5.
            import warnings as _wm

            import hybrid_localize_121 as H
            from lumenairy.propagators.carrier import (
                _envelope_amp_radius, carrier_referenced_envelope)
            from exact_ray_oracle_121 import _phase_gradient

            with _wm.catch_warnings(record=True) as _wl:
                _wm.simplefilter('always')
                res = C.la.propagate_traced_carrier_chain(
                    env_doe, post, lam, dx_doe,
                    r_in=C.la.TiltedCarrier(R_doe, L, M),
                    ray_subsample=a.rs, n_workers=a.nw, final_distance=0.0,
                    final_leg='paraxial', on_decentred_fit='ignore')
            seen = {}
            for w in _wl:
                t = str(w.message)[:160]
                seen[t] = seen.get(t, 0) + 1
            for t, k in seen.items():
                print(f"  [warn x{k}] {t}")
            fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
            st = [s for s in res.stages if not s.get('target')][-1]
            Lk, Mk = st.get('L_out', L), st.get('M_out', M)
            xck, yck = st.get('x_c_out', 0.0), st.get('y_c_out', 0.0)
            envk = carrier_referenced_envelope(fld, Rk, lam, dxk)
            nn = envk.shape[0]
            u = (np.arange(nn) - nn / 2) * dxk
            K0 = 2 * np.pi / lam
            envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
            wk = _envelope_amp_radius(envk, dxk, dxk)
            half = int(np.ceil(a.clip * wk / dxk))
            c0 = nn // 2
            i0 = max(c0 - half, 0)
            i1 = min(c0 + half + 1, nn)
            envk = envk[i0:i1, i0:i1]
            nn = envk.shape[0]
            u = (np.arange(nn) - (c0 - i0)) * dxk
            gx, gy, step_ph = _phase_gradient(envk, dxk)
            phres = np.unwrap(np.unwrap(np.angle(envk), axis=1), axis=0)
            # hybrid_localize uses floor() here, which lets the launch grid
            # grow to 215^2 on a tilted order (the crop widens with the tilt)
            # and the RS cost with it; ceil() caps it at a.nl so every order is
            # scored on the SAME launch sampling as the oracle panel.
            sd = max(1, int(np.ceil(nn / a.nl)))
            envk = envk[::sd, ::sd]
            gx, gy, phres = gx[::sd, ::sd], gy[::sd, ::sd], phres[::sd, ::sd]
            u = u[::sd]
            nn = envk.shape[0]
            U, V = np.meshgrid(u, u)
            x0 = (U + xck).ravel()
            y0 = (V + yck).ravel()
            amp = np.abs(envk).ravel()
            pdir = (U / Rk).ravel() + Lk + gx.ravel() / K0
            qdir = (V / Rk).ravel() + Mk + gy.ravel() / K0
            ph0 = K0 * ((U ** 2 + V ** 2).ravel() / (2 * Rk)
                        + Lk * U.ravel() + Mk * V.ravel()) + phres.ravel()
            back = a.back * 1e-3
            surfs = H._free_surfaces(C.TRAILING, back)
            r = H.rs_spot(x0, y0, amp, ph0, pdir, qdir, surfs, back, xci, yci,
                          dx_out=dxo, n_out=a.nout, nl=nn)
            I = np.asarray(r['I'])
            P_win = float(I.sum()) * dxo * dxo
            extra = {'P_doe': P_doe, 'P_window': P_win,
                     'P_ratio': float('nan'),
                     'envelope_phase_step_rad': float(step_ph),
                     'integrand_step_p999_cycles': float(r['step_w']),
                     'live_frac': float(r['live']),
                     'launch_grid': int(nn), 'launch_dx_um': dxk * sd * 1e6,
                     'ee3': float(r['ee3']), 'ee6': float(r['ee6']),
                     'ee12': float(r['ee12']), 'fwhm': float(r['fwhm']),
                     'warnings': list(seen.keys())}
            print(f"    launch {nn}^2 @ {dxk * sd * 1e6:.3f} um, live "
                  f"{r['live'] * 100:.4f} %, envelope step {step_ph:.4f} rad, "
                  f"integrand p99.9 {r['step_w']:.4f} cycles")
            print(f"    FWHM {r['fwhm'] * 1e6:6.3f} um   EE3 "
                  f"{r['ee3'] * 100:6.2f}   EE6 {r['ee6'] * 100:6.2f}   EE12 "
                  f"{r['ee12'] * 100:6.2f}")
        else:
            import lumenairy as la
            import warnings as _wm
            with _wm.catch_warnings(record=True) as _wl:
                _wm.simplefilter('always')
                res = C.la.propagate_traced_carrier_chain(
                    env_doe, post, lam, dx_doe,
                    r_in=la.TiltedCarrier(R_doe, L, M),
                    ray_subsample=a.rs, n_workers=a.nw,
                    final_distance=C.TRAILING, final_leg=a.leg,
                    focus_readout={'dx_out': dxo, 'N_out': a.nout,
                                   'n_fine_cap': a.nfc,
                                   'window_factor': a.wf})
            seen = {}
            for w in _wl:
                t = str(w.message)[:160]
                seen[t] = seen.get(t, 0) + 1
            for t, k in seen.items():
                print(f"  [warn x{k}] {t}")
            E = np.asarray(res.field)
            I = np.abs(E) ** 2
            P_win = float(I.sum()) * dxo * dxo
            extra = {'P_doe': P_doe, 'P_window': P_win,
                     'P_ratio': P_win / P_doe,
                     'warnings': list(seen.keys())}
        ax = (np.arange(a.nout) - (a.nout - 1) / 2.0) * dxo
        path = os.path.join(OUTDIR, tag + '.npz')
        np.savez_compressed(
            path, I=I, ax=ax,
            meta=json.dumps({'method': a.method, 'm': m, 'n': n,
                             'L': L, 'M': M, 'chief': [xci, yci],
                             'dx_out': dxo, 'n_out': a.nout,
                             'rn': a.rn, 'rs': a.rs, 'nl': a.nl,
                             'clip': a.clip, 'back_mm': a.back,
                             'nfc': a.nfc, 'wf': a.wf, 'leg': a.leg,
                             'period': period,
                             'runtime_s': time.time() - t0,
                             **{k: v for k, v in extra.items()
                                if v is not None}}))
        print(f"  -> {path}   [{time.time() - t0:.0f}s]  "
              f"P_window/P_in = {extra.get('P_ratio', float('nan')):.6f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
