# HAMILTON-MAP feasibility prototype, step 4: THE COST DECOMPOSITION.
#
# LOCAL-ONLY.  NO library edit -- the two spies below are runtime monkeypatches
# on module attributes, installed and removed inside one process, and both are
# pure pass-throughs (they time and record, they do not alter arguments or
# results).
#
# THE QUESTION.  PROBE_SUM_AT_APERTURE_2026_08_11 sec 7.1 measured the fine
# RE-TRACE of the last group at 76.4 % of a design-121 order's wall clock and
# named a "shared-pupil formulation of the last group's ray map" as the only
# route that could reach a real 32x.  A shared map can only amortize the part
# of that 76.4 % that is actually RAY TRACING.  Everything else the retrace
# does -- the Fourier upsample, the carrier reconstruction on the fine grid,
# the Chebyshev fit of the entrance->exit map, the Newton pull-back of every
# one of ``n_fine**2`` exit pixels, and the field assembly -- is per-order and
# per-output-pixel, and no angular interpolation touches it.
#
# So this script measures, on ONE order at the shipped configuration:
#
#     wall = coarse chain + RETRACE + readout
#     RETRACE = upsample/reconstruct + apply_real_lens_traced
#     apply_real_lens_traced = trace(n_launch**2 rays) + everything else
#
# and reports ``n_launch``, the ray count, and the trace's share.  That share
# is the CEILING on what a shared-pupil map can save per order, before its own
# node count is charged.
#
# Run:  python hmap_cost_121.py run          (one order, ~2-3 min, ~25 GB)
#       python hmap_cost_121.py model        (break-even table, no chain)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_traced as LT  # noqa: E402
import lumenairy.propagators.carrier as CAR  # noqa: E402
import lumenairy.raytrace as RT  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_hmap_cost.json')

# the sum-at-aperture probe's own arm-A configuration, verbatim
RS = 4
NW = 1
DXO = 0.2e-6
TILE = 1024
NFC = int(os.environ.get('NFC', '8192'))
WF = 4.0
LEG = 'auto'
RAMB = float('inf')
ORDER = (int(os.environ.get('HM_M', '0')), int(os.environ.get('HM_N', '0')))


def _peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, 'peak_wset', mi.rss)) / 1e9
    except Exception:
        return float('nan')


def run_one(m, n):
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period)
    env_doe, R_doe, dx_doe, _P = C.chain_a()
    car = la.TiltedCarrier(float(R_doe), m * C.LAM / period,
                           n * C.LAM / period)
    x, y, _L, _M = CAR._chain_chief_ray_at_target(
        groups_post, C.LAM, car, C.TRAILING, 'hmap_cost_121')
    fr = dict(dx_out=DXO, N_out=TILE,
              centre_out=(round(x / DXO) * DXO, round(y / DXO) * DXO),
              n_fine_cap=NFC, window_factor=WF, ram_budget=RAMB)

    cap = {'trace_calls': [], 'elem_calls': []}
    o_trace = RT.trace
    o_ftge = CAR._fine_trace_group_exit
    o_elem = EL.apply_real_lens_traced
    o_ro = CAR.carrier_referenced_exact_focus_readout

    def _trace(rays, surfaces, wavelength, **kw):
        t0 = time.perf_counter()
        out = o_trace(rays, surfaces, wavelength, **kw)
        cap['trace_calls'].append({'n_rays': int(np.size(rays.x)),
                                   'n_surf': int(len(surfaces)),
                                   'seconds': time.perf_counter() - t0})
        return out

    def _elem(*a, **kw):
        i0 = len(cap['trace_calls'])
        t0 = time.perf_counter()
        out = o_elem(*a, **kw)
        dt = time.perf_counter() - t0
        mine = cap['trace_calls'][i0:]
        cap['elem_calls'].append({
            'seconds': dt,
            'dx': float(kw.get('dx', float('nan'))),
            'n_grid': int(np.asarray(a[0] if a else kw['E_in']).shape[-1]),
            'ray_subsample': int(kw.get('ray_subsample', 0)),
            'trace_seconds': float(sum(c['seconds'] for c in mine)),
            'trace_rays': int(sum(c['n_rays'] for c in mine)),
            'n_trace_calls': len(mine)})
        return out

    def _ftge(*a, **kw):
        i0 = len(cap['elem_calls'])
        t0 = time.perf_counter()
        out = o_ftge(*a, **kw)
        cap['t_retrace'] = time.perf_counter() - t0
        cap['retrace_elem'] = cap['elem_calls'][i0:]
        cap['dx_fine'] = float(out[1])
        cap['n_fine'] = int(np.asarray(out[0]).shape[-1])
        cap['cur_dx'] = float(a[2])
        cap['R_in'] = float(a[1])
        return out

    def _ro(*a, **kw):
        t0 = time.perf_counter()
        out = o_ro(*a, **kw)
        cap['t_readout'] = time.perf_counter() - t0
        return out

    # niche C6 residual eikonal: its launch-direction gradient is the amount
    # by which the ACTUAL launch congruence departs from the analytic sphere +
    # tilt, and so the amount by which a SHARED angular domain must be widened
    # (it is order-dependent, hence not absorbable into the shear).
    o_fit = LT._fit_residual_eikonal
    cap['resid_eik'] = []

    def _fit(*a, **kw):
        out = o_fit(*a, **kw)
        cap['resid_eik'].append(out)
        return out

    RT.trace = _trace
    EL.apply_real_lens_traced = _elem
    LT._fit_residual_eikonal = _fit
    CAR._fine_trace_group_exit = _ftge
    CAR.carrier_referenced_exact_focus_readout = _ro
    t0 = time.perf_counter()
    try:
        res = CAR.propagate_traced_carrier_chain(
            env_doe, groups_post, C.LAM, dx_doe, r_in=car,
            ray_subsample=RS, n_workers=NW, final_distance=C.TRAILING,
            focus_readout=fr, final_leg=LEG)
    finally:
        RT.trace = o_trace
        EL.apply_real_lens_traced = o_elem
        LT._fit_residual_eikonal = o_fit
        CAR._fine_trace_group_exit = o_ftge
        CAR.carrier_referenced_exact_focus_readout = o_ro
    wall = time.perf_counter() - t0

    rt = cap.get('retrace_elem') or []
    tr_s = sum(c['trace_seconds'] for c in rt)
    tr_r = sum(c['trace_rays'] for c in rt)
    el_s = sum(c['seconds'] for c in rt)
    n_launch = int(round(np.sqrt(max(tr_r, 1))))
    out = {
        'order': [m, n], 'wall': wall, 'nfc': NFC, 'window_factor': WF,
        'ray_subsample': RS, 'n_workers': NW,
        't_retrace': cap.get('t_retrace'), 't_readout': cap.get('t_readout'),
        'dx_fine': cap.get('dx_fine'), 'n_fine': cap.get('n_fine'),
        'cur_dx': cap.get('cur_dx'),
        'retrace_element_seconds': el_s,
        'retrace_trace_seconds': tr_s,
        'retrace_trace_rays': tr_r,
        'retrace_n_launch': n_launch,
        'retrace_rs_fine': (rt[0]['ray_subsample'] if rt else None),
        'all_trace_calls': cap['trace_calls'],
        'all_elem_calls': cap['elem_calls'],
        'peak_wset_gb': _peak_gb(),
        'lumenairy_version': str(la.__version__),
        'field_shape': list(np.asarray(res.field).shape),
        'grad_a_fit_max_launch': [
            float(getattr(o, 'diag', {}).get('grad_a_fit_max_launch',
                                             float('nan')))
            for o in cap['resid_eik'] if o is not None],
    }
    if out['grad_a_fit_max_launch']:
        print(f"  niche-C6 |grad a_fit| max per element call: "
              + ' '.join(f"{v:.3e}" for v in out['grad_a_fit_max_launch']))
    print(f"order {(m, n)}  wall {wall:8.1f} s   "
          f"retrace {cap.get('t_retrace', float('nan')):7.1f} s "
          f"({100 * cap.get('t_retrace', 0) / wall:.1f} %)   "
          f"readout {cap.get('t_readout', float('nan')):6.1f} s")
    print(f"  retrace grid n_fine {cap.get('n_fine')} at "
          f"dx_fine {cap.get('dx_fine', 0) * 1e6:.4f} um; "
          f"chain cur_dx {cap.get('cur_dx', 0) * 1e6:.4f} um")
    print(f"  the ELEMENT inside the retrace: {el_s:.1f} s, of which "
          f"trace() = {tr_s:.3f} s over {tr_r} rays "
          f"(n_launch ~ {n_launch}, rs_fine {out['retrace_rs_fine']})")
    print(f"  => trace share of the RETRACE  "
          f"{100 * tr_s / max(cap.get('t_retrace', 1), 1e-9):.3f} %")
    print(f"  => trace share of the ORDER    {100 * tr_s / wall:.3f} %")
    print(f"  peak working set {out['peak_wset_gb']:.1f} GB")
    print("  every trace() call in the whole chain:")
    for c in cap['trace_calls']:
        print(f"     {c['n_rays']:12d} rays x {c['n_surf']} surfaces  "
              f"{c['seconds']:8.3f} s")
    return out


def model(cost=None, verbose=True):
    """The break-even arithmetic, from the measured node counts and the
    measured trace cost."""
    import hmap_probe_121 as P
    groups_post, rows, R_ent, _per, _rd = P.geometry()
    presc = groups_post[-1]['prescription']
    ap = float(presc.get('aperture_diameter'))
    # the launch radius apply_real_lens_traced ACTUALLY uses (line 8131:
    # ``aperture`` comes from the prescription, so the grid-derived branch is
    # dead here) -- the same for every order, and axis-centred.
    lr_shipped = 0.5 * ap * 1.5
    reach = max(np.hypot(r['x_c'], r['y_c']) for r in rows)
    r_union = reach + P.R_PUPIL_ORDER
    out = {'aperture_diameter': ap, 'launch_radius_shipped': lr_shipped,
           'r_union_needed': r_union, 'chief_reach': reach,
           'r_pupil_order': P.R_PUPIL_ORDER,
           'union_fits_in_shipped_launch': bool(r_union <= lr_shipped),
           'pupil_area_ratio_union_over_shipped':
               (r_union / lr_shipped) ** 2}
    if verbose:
        print(f"last group aperture_diameter {ap * 1e3:.4f} mm  -> "
              f"apply_real_lens_traced launch radius "
              f"{lr_shipped * 1e3:.4f} mm (0.75 x aperture), AXIS-CENTRED and "
              f"ORDER-INDEPENDENT")
        print(f"the shared map needs {r_union * 1e3:.4f} mm "
              f"(chief reach {reach * 1e3:.4f} + half-window "
              f"{P.R_PUPIL_ORDER * 1e3:.4f})  -> "
              f"{'INSIDE' if r_union <= lr_shipped else 'OUTSIDE'} the launch "
              f"disc the shipped path already uses, by "
              f"{lr_shipped / r_union:.3f}x linear "
              f"({(lr_shipped / r_union) ** 2:.3f}x area)")
    if cost:
        tr = float(cost['retrace_trace_seconds'])
        rt = float(cost['t_retrace'])
        wall = float(cost['wall'])
        rays = int(cost['retrace_trace_rays'])
        out.update({'trace_seconds': tr, 'retrace_seconds': rt,
                    'wall_seconds': wall, 'rays': rays})
        if verbose:
            print()
            print(f"MEASURED, one order: trace {tr:.3f} s of a {rt:.1f} s "
                  f"retrace of a {wall:.1f} s order "
                  f"({100 * tr / rt:.3f} % / {100 * tr / wall:.3f} %)")
        for nodes in (20, 24, 25, 30, 36, 49):
            # map build = nodes x the SAME pupil lattice (it fits inside the
            # shipped launch disc), against 32 x that lattice.
            build = nodes * tr
            direct = 32 * tr
            saved = direct - build
            out[f'saved_s_nodes{nodes}'] = saved
            if verbose:
                print(f"   {nodes:3d} nodes: build {build:8.3f} s vs "
                      f"{direct:8.3f} s of direct traces  -> saves "
                      f"{saved:+8.3f} s of a {32 * wall:.0f} s 32-order run "
                      f"= {100 * saved / (32 * wall):+.4f} %")
    return out


def afit():
    """The niche-C6 residual eikonal's LAUNCH-GRADIENT, per group.

    ``apply_real_lens_traced`` launches along ``grad(W + a_fit)``, not along
    ``grad W``.  ``a_fit`` is fitted to THIS order's own field, so it is
    order-dependent and CANNOT be absorbed into a shared map's congruence
    label: it is an extra angular offset the shared angular domain has to
    cover.  ``max |grad a_fit|`` x ``|R|`` is the equivalent widening of the
    SOURCE-POSITION box.

    Stops the chain at the last group's exit vertex (``final_distance=0``, no
    ``focus_readout``) -- the coarse-grid path, ~1-3 min, no 8192-square
    retrace -- so the numbers are the coarse legs' own ``a_fit``."""
    _pre, groups_post, _gap, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a()
    m, n = ORDER
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
    vals = [float(getattr(o, 'diag', {}).get('grad_a_fit_max_launch',
                                             float('nan')))
            for o in got if o is not None]
    print(f"order {(m, n)}: {len(got)} residual-eikonal fits in "
          f"{time.perf_counter() - t0:.1f} s")
    for i, v in enumerate(vals):
        print(f"  call {i}: max |grad a_fit| over the launch lattice "
              f"= {v:.4e} rad")
    return {'order': [m, n], 'grad_a_fit_max_launch': vals}


def scaling():
    """Element cost vs the OUTPUT grid, with the trace held fixed.

    Calls ``apply_real_lens_traced`` on synthetic Gaussian fields of the last
    group's own shape at n = 1024 / 2048 / 4096, same carrier, same physical
    window, with the trace spy in place.  If the element's cost tracks
    ``n**2`` while ``trace()`` stays flat, the retrace is per-OUTPUT-PIXEL
    work and no angular map can touch it."""
    import hmap_probe_121 as P
    groups_post, rows, R_ent, _per, _rd = P.geometry()
    presc = groups_post[-1]['prescription']
    win = 2.0 * P.R_PUPIL_ORDER
    r0 = rows[0]
    out = []
    o_trace = RT.trace
    calls = []

    def _trace(rays, surfaces, wavelength, **kw):
        t0 = time.perf_counter()
        res = o_trace(rays, surfaces, wavelength, **kw)
        calls.append({'n_rays': int(np.size(rays.x)),
                      'seconds': time.perf_counter() - t0})
        return res

    RT.trace = _trace
    try:
        for nn in (1024, 2048, 4096):
            dx = win / nn
            x = (np.arange(nn) - nn // 2) * dx
            X, Y = np.meshgrid(x, x, indexing='xy')
            w = 0.25 * win
            E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
            car = la.TiltedCarrier(R_ent, r0['L'], r0['M'], 0.0, 0.0)
            calls.clear()
            t0 = time.perf_counter()
            EL.apply_real_lens_traced(
                E, prescription=presc, wavelength=C.LAM, dx=dx, carrier=car,
                ray_subsample=max(1, int(round(4 * 33.2112e-6 / dx))),
                n_workers=1, on_undersample='silent')
            dt = time.perf_counter() - t0
            tr = sum(c['seconds'] for c in calls)
            ry = sum(c['n_rays'] for c in calls)
            out.append({'n': nn, 'dx_um': dx * 1e6, 'element_s': dt,
                        'trace_s': tr, 'trace_rays': ry,
                        'trace_frac': tr / max(dt, 1e-12)})
            print(f"  n={nn:5d} dx={dx * 1e6:7.4f} um: element {dt:8.2f} s, "
                  f"trace {tr:7.3f} s over {ry:9d} rays "
                  f"({100 * tr / max(dt, 1e-12):.4f} %)")
            del E, X, Y
    finally:
        RT.trace = o_trace
    for a, b in zip(out, out[1:]):
        print(f"  {a['n']} -> {b['n']}: element x{b['element_s'] / a['element_s']:.2f} "
              f"(pixel count x4), trace x{b['trace_s'] / max(a['trace_s'], 1e-9):.2f}")
    return {'rows': out}


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    blob = {}
    if os.path.exists(OUT):
        try:
            blob = json.load(open(OUT, encoding='ascii'))
        except (ValueError, OSError):
            blob = {}
    if mode == 'run':
        blob['cost'] = run_one(*ORDER)
    elif mode == 'afit':
        blob.setdefault('afit', {})['%d_%d' % ORDER] = afit()
    elif mode == 'scaling':
        blob['scaling'] = scaling()
    blob['model'] = model(blob.get('cost'))
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump(blob, fh, indent=1, sort_keys=True, default=float)
    print(f"\nwrote {os.path.basename(OUT)}")
