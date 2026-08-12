# INVERSE-MAP production build, PRE-BUILD MEASUREMENT: is the niche-C6
# residual-eikonal launch offset a SOURCE SHIFT, or is it aberration?
#
# LOCAL-ONLY.  NO library edit -- the one instrumentation is a pass-through
# wrapper on ``_lens_traced._fit_residual_eikonal`` that records the fitted
# object and returns it unchanged.
#
# WHY THIS DECIDES THE ARCHITECTURE.  ``PROTO_INVERSE_MAP_2026_08_11`` sizes a
# SHARED 4-D map ``G(x_out, y_out; x_src, y_src)`` on the two-parameter source
# label, and covers the C6 residual by WIDENING the source box by
# ``max|grad a_fit| x |R|`` -- i.e. it treats the a_fit launch augmentation as
# an equivalent SOURCE DISPLACEMENT.  That is exact only for the part of
# ``grad a_fit`` that is CONSTANT over the launch lattice (a wavefront tilt).
# Any pupil-VARYING part is a different congruence, which no shared
# source-labelled map can represent at any node count.
#
# So this measures, on the real chain, at the real last group:
#
#     grad a_fit(x, y)  over the launch lattice
#       -> mean (the tilt)                      -> equivalent source shift
#       -> max |grad a_fit - mean| (the rest)   -> the irreducible mismatch
#
# and converts the second into the quantity that matters for the inverse map:
# an EXIT-POSITION displacement, hence an ENTRANCE-POSITION error, hence an
# OPL error in waves.  The proto's parity bar is 1.11e-04 waves.
#
# Run:  python imap_afit_121.py            (~1-3 min, coarse legs only)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import hmap_probe_121 as H  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.elements._lens_traced as LT  # noqa: E402
import lumenairy.propagators.carrier as CAR  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_imap_afit.json')

RS = 4
NW = 1
ORDER = (int(os.environ.get('HM_M', '-4')), int(os.environ.get('HM_N', '-2')))


def run(order=ORDER, verbose=True):
    _pre, groups_post, _gap, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a()
    m, n = order
    car = la.TiltedCarrier(float(R_doe), m * C.LAM / period,
                           n * C.LAM / period)
    o_fit = LT._fit_residual_eikonal
    got = []

    def _fit(*a, **kw):
        out = o_fit(*a, **kw)
        got.append(out)
        return out

    LT._fit_residual_eikonal = _fit
    t0 = time.perf_counter()
    try:
        CAR.propagate_traced_carrier_chain(
            env_doe, groups_post, C.LAM, dx_doe, r_in=car,
            ray_subsample=RS, n_workers=NW, final_distance=0.0)
    finally:
        LT._fit_residual_eikonal = o_fit
    t_chain = time.perf_counter() - t0

    # the geometry the shared map would be built on
    groups_p, rows, R_ent, _per, _rd = H.geometry()
    surfs, _presc = H.group_surfaces(groups_p)
    ap = float(groups_p[-1]['prescription']['aperture_diameter'])
    lr = 0.75 * ap
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + H.R_PUPIL_ORDER
    row = None
    for r in rows:
        if tuple(r['order']) == tuple(order):
            row = r
    if row is None:
        row = rows[0]

    n_lat = 229
    xs = np.linspace(-lr, lr, n_lat)
    PX, PY = np.meshgrid(xs, xs, indexing='ij')

    out = {'order': [m, n], 'n_fits': len(got), 'chain_seconds': t_chain,
           'R_ent_mm': float(R_ent) * 1e3, 'launch_radius_mm': lr * 1e3,
           'r_union_mm': r_union * 1e3, 'fits': []}
    for i, o in enumerate(got):
        if o is None:
            out['fits'].append({'call': i, 'null': True})
            continue
        gx, gy = o.grad(PX.ravel(), PY.ravel())
        gx = np.asarray(gx, dtype=float).reshape(PX.shape)
        gy = np.asarray(gy, dtype=float).reshape(PX.shape)
        rec = {'call': i, 'r_fit_mm': float(o.r_fit) * 1e3,
               'scale_mm': float(o.scale) * 1e3,
               'centre_mm': [float(o.cx) * 1e3, float(o.cy) * 1e3],
               'degree_terms': len(o.terms),
               'diag_grad_max': float(getattr(o, 'diag', {}).get(
                   'grad_a_fit_max_launch', float('nan')))}
        for tag, mask in (('launch_square', np.ones(PX.shape, bool)),
                          ('union_pupil',
                           (PX ** 2 + PY ** 2) <= r_union ** 2),
                          ('fit_disc',
                           ((PX - o.cx) ** 2 + (PY - o.cy) ** 2)
                           <= o.r_fit ** 2)):
            if not mask.any():
                continue
            a, b = gx[mask], gy[mask]
            mx, my = float(a.mean()), float(b.mean())
            dev = np.hypot(a - mx, b - my)
            rec[tag] = {
                'n': int(mask.sum()),
                'grad_max_rad': float(np.hypot(a, b).max()),
                'tilt_mean_rad': [mx, my],
                'tilt_mag_rad': float(np.hypot(mx, my)),
                'nontilt_max_rad': float(dev.max()),
                'nontilt_rms_rad': float(np.sqrt((dev ** 2).mean())),
                'src_shift_from_tilt_mm': float(
                    np.hypot(mx, my) * abs(R_ent)) * 1e3,
                'src_shift_from_nontilt_mm': float(
                    dev.max() * abs(R_ent)) * 1e3}
        out['fits'].append(rec)

    # Convert the LAST fit's non-tilt residual into the quantity the inverse
    # map is judged on, by TRACING the same congruence twice: once along
    # grad W (the shared map's own congruence, at the tilt-corrected label)
    # and once along grad(W + a_fit) (what the element actually launches).
    last = got[-1] if got else None
    if last is not None:
        keep = (PX ** 2 + PY ** 2) <= r_union ** 2
        px, py = PX[keep], PY[keep]
        lx0, ly0 = H.order_angle(R_ent, row, px, py)
        agx, agy = last.grad(px, py)
        agx = np.asarray(agx, dtype=float)
        agy = np.asarray(agy, dtype=float)
        # the library's own launch normalisation (see _lens_traced: the
        # residual gradient is ADDED to the carrier direction cosines)
        ch0 = H.characteristic(surfs, px, py, lx0, ly0)
        ch1 = H.characteristic(surfs, px, py, lx0 + agx, ly0 + agy)
        # ...and the best pure-source-shift stand-in for the a_fit launch:
        # shift the source by the MEAN gradient's equivalent displacement.
        mgx, mgy = float(agx.mean()), float(agy.mean())
        ch2 = H.characteristic(surfs, px, py, lx0 + mgx, ly0 + mgy)
        ok = ch0['alive'] & ch1['alive'] & ch2['alive']
        d_full = np.hypot(ch1['x'][ok] - ch0['x'][ok],
                          ch1['y'][ok] - ch0['y'][ok])
        d_res = np.hypot(ch1['x'][ok] - ch2['x'][ok],
                         ch1['y'][ok] - ch2['y'][ok])
        o_full = np.abs(ch1['opl'][ok] - ch0['opl'][ok]) / C.LAM
        o_res = np.abs(ch1['opl'][ok] - ch2['opl'][ok]) / C.LAM
        out['exit_displacement'] = {
            'n_rays': int(ok.sum()),
            'mean_grad_rad': [mgx, mgy],
            'no_afit_exit_shift_max_um': float(d_full.max()) * 1e6,
            'tilt_only_exit_shift_max_um': float(d_res.max()) * 1e6,
            'no_afit_opl_max_waves': float(o_full.max()),
            'tilt_only_opl_max_waves': float(o_res.max())}

    if verbose:
        print("residual-eikonal LAUNCH GRADIENT, order %s, %d fits, chain "
              "%.1f s" % (str(tuple(order)), len(got), t_chain))
        print("  R at the last group's entrance %.4f mm; launch radius "
              "%.4f mm; union pupil %.4f mm"
              % (out['R_ent_mm'], out['launch_radius_mm'], out['r_union_mm']))
        for rec in out['fits']:
            if rec.get('null'):
                print("  call %d: no fit" % rec['call'])
                continue
            u = rec.get('union_pupil') or rec.get('launch_square')
            print("  call %d (r_fit %.3f mm, %d terms): max|grad a| %.4e rad "
                  "| tilt %.4e rad (= %.4f mm source shift) | NON-TILT max "
                  "%.4e rad rms %.4e rad (= %.4f mm)"
                  % (rec['call'], rec['r_fit_mm'], rec['degree_terms'],
                     u['grad_max_rad'], u['tilt_mag_rad'],
                     u['src_shift_from_tilt_mm'], u['nontilt_max_rad'],
                     u['nontilt_rms_rad'], u['src_shift_from_nontilt_mm']))
        e = out.get('exit_displacement')
        if e:
            print("\n  TRACED, over %d union-pupil rays of the LAST fit:"
                  % e['n_rays'])
            print("    grad(W + a_fit) vs grad(W)          : exit shift "
                  "%.3f um, OPL %.4e waves" % (e['no_afit_exit_shift_max_um'],
                                               e['no_afit_opl_max_waves']))
            print("    grad(W + a_fit) vs grad(W) + TILT   : exit shift "
                  "%.3f um, OPL %.4e waves"
                  % (e['tilt_only_exit_shift_max_um'],
                     e['tilt_only_opl_max_waves']))
            print("    (the parity bar the shared map must clear is "
                  "1.11e-04 waves)")
    return out


if __name__ == '__main__':
    res = run()
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    print("\nwrote %s" % os.path.basename(OUT))
