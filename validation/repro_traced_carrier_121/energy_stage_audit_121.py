# Conservation audit (2026-07-31), CHAIN level: the per-stage energy budget of
# design 121's post-DOE chain, per DOE order, with the niche C5 and C6 flags
# toggled.
#
# THE QUESTION.  The campaign scored itself on encircled energy.  EE cannot see
# a field that gains or loses power, and it cannot see a far halo (measured:
# a field carrying 82-121 % of MANUFACTURED energy read as a -0.005 EE3 point
# change).  This instrument scores the SAME shipped chain on conservation
# instead, stage by stage, so that legitimate loss (aperture clipping) is
# separated from unexplained loss/gain.
#
# WHAT IS MEASURED, per post-DOE group k (there are six):
#
#   P_in(k)       = sum |E_in(k)|^2 dx(k)^2          -- whole co-moving grid
#   P_ap(k)       = the same restricted to the group's aperture_diameter
#                   -- THIS is the denominator the library's own energy
#                      self-check uses, so ``elem`` below is exactly the
#                      quantity the guard polices
#   clip(k)       = 1 - P_ap(k)/P_in(k)              -- legitimate vignetting
#   P_out(k)      = sum |E_out(k)|^2 dx(k)^2
#   elem(k)       = P_out(k) / P_ap(k)               -- the ELEMENT's ratio
#   leg(k)        = P_in(k+1) / P_out(k)             -- the FREE-SPACE leg
#                   between group k's exit and group k+1's entrance (the
#                   co-moving carrier transport: window factor, grid extent,
#                   Fourier crop).  A leg is a pure propagation and must
#                   conserve to the grid's own truncation.
#
# End to end, P_out(5)/P_in(0) factorises EXACTLY as
#     prod_k elem(k) * prod_k (1 - clip(k)) * prod_k leg(k)
# and the script prints each factor, so no loss is attributed by assertion.
#
# Powers use ``sum |E|^2 dx^2``, the library's own ``_chain_envelope_stats``
# convention, because the co-moving frame rescales dx per stage; the raw
# ``sum |E|^2`` is printed alongside so the rescaling convention is visible and
# cannot silently supply or absorb energy.
#
# The chain's OWN ``res.stages[k]['power']`` is printed as an independent
# cross-check of this script's arithmetic -- if the two disagree, this
# instrument is wrong, not the library.
#
# LOCAL-ONLY.  No library edit: C5/C6 are module attributes set inside a
# try/finally, and the element capture is ``wfe_probe_common``'s existing
# script-side monkeypatch, extended only to record warnings PER CALL (so the
# energy self-check can be attributed to a stage rather than to the run).
#
# usage:  ORDERS='0,0'   CONFIGS='ship,noC6,noC5' python energy_stage_audit_121.py
#         ORDERS='-4,-2' CONFIGS='ship'           python energy_stage_audit_121.py
import hashlib
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _d121_common as C                                          # noqa: E402
import probe_ghost_c6 as G6                                       # noqa: E402
import wfe_probe_common as P                                      # noqa: E402
import lumenairy as la                                            # noqa: E402
import lumenairy.elements._lens_traced as LT                      # noqa: E402
from lumenairy.elements._lens_traced import TiltedCarrier         # noqa: E402

LAM = C.LAM

CONFIGS = {
    'ship':  {'c5': True,  'c6': True},    # SHIPPED (C4+C5+C6 working tree)
    'noC6':  {'c5': True,  'c6': False},   # C6 ablated
    'noC5':  {'c5': False, 'c6': True},    # C5 ablated
    'noC56': {'c5': False, 'c6': False},   # both ablated
    # -- 2026-07-31 fit-guard decision study ------------------------------
    # ``REMAP_STATIONARY_PHASE_FIT_GUARD`` routes a C6-ENGAGED CONCENTRIC fit
    # disc through D1's weighted restriction (+ D7's raised order) instead of
    # the historical hard NaN mask.  It is inert wherever the disc is already
    # off-centre (every tilted order) and inert wherever C6 is off (the guard
    # is gated on ``_resid_eik is not None``, which requires C6).
    'shipG': {'c5': True,  'c6': True,  'guard': True},
    'noC6G': {'c5': True,  'c6': False, 'guard': True},   # must equal noC6
}


class capture_with_warnings(object):
    """``wfe_probe_common.capture_chain_element_calls`` plus a per-call
    warning census, so the energy self-check can be attributed to a STAGE."""

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        import lumenairy.elements as _el
        self._orig = _el.apply_real_lens_traced
        calls = self.calls

        def _wrapped(E_in, *a, **kw):
            rec = {'kw': dict(kw), 'E_in': np.array(E_in, copy=True)}
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                out = self._orig(E_in, *a, **kw)
            fl = {'energy': 0, 'fold': 0, 'undersample': 0}
            for w in wl:
                t = str(w.message)
                if 'energy self-check FAILED' in t:
                    fl['energy'] += 1
                elif 'fold caustic' in t:
                    fl['fold'] += 1
                elif ('far-halo energy lands at wrong radii' in t
                      or 'NA_exit=' in t):
                    fl['undersample'] += 1
                else:
                    warnings.warn_explicit(w.message, w.category,
                                           w.filename, w.lineno)
            rec['E_out'] = np.array(out, copy=True)
            rec['flags'] = fl
            calls.append(rec)
            return out

        _el.apply_real_lens_traced = _wrapped
        return self

    def __exit__(self, *exc):
        import lumenairy.elements as _el
        _el.apply_real_lens_traced = self._orig
        return False


def run(order, cfg_name, rn=1024, rs=4, n_groups=6):
    """One complete post-DOE chain run.  Returns the capture and the result."""
    m, n = order
    _pre, post, _gap, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=rn, rs=rs)
    L, M = m * LAM / period, n * LAM / period
    cfg = CONFIGS[cfg_name]
    old = (LT.TILTED_CARRIER_EXACT_EIKONAL, LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD)
    LT.TILTED_CARRIER_EXACT_EIKONAL = bool(cfg['c5'])
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(cfg['c6'])
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(cfg.get('guard', False))
    t0 = time.time()
    try:
        with capture_with_warnings() as cap:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                res = la.propagate_traced_carrier_chain(
                    env_doe, [dict(g) for g in post[:n_groups]], LAM, dx_doe,
                    r_in=TiltedCarrier(R_doe, L, M), ray_subsample=rs,
                    n_workers=8, final_distance=0.0, final_leg='paraxial',
                    on_decentred_fit='ignore')
    finally:
        (LT.TILTED_CARRIER_EXACT_EIKONAL,
         LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD) = old
    chain_warn = {}
    for w in wl:
        t = str(w.message)[:110]
        chain_warn[t] = chain_warn.get(t, 0) + 1
    return cap.calls, res, chain_warn, time.time() - t0


def budget(calls, post):
    """Per-stage energy budget.  Returns (rows, totals)."""
    rows = []
    for k, rec in enumerate(calls):
        dx = float(rec['kw']['dx'])
        Ein, Eout = rec['E_in'], rec['E_out']
        N = Ein.shape[0]
        axg = (np.arange(N) - N / 2) * dx
        Xg, Yg = np.meshgrid(axg, axg)
        ap = post[k]['prescription'].get('aperture_diameter')
        i_in = np.abs(Ein) ** 2
        p_in = float(i_in.sum()) * dx * dx
        if ap is None:
            p_ap = p_in
        else:
            p_ap = float(i_in[Xg ** 2 + Yg ** 2
                              <= (ap / 2) ** 2].sum()) * dx * dx
        p_out = float((np.abs(Eout) ** 2).sum()) * dx * dx
        rows.append({
            'k': k, 'dx': dx, 'N': N, 'ap': ap,
            'p_in': p_in, 'p_ap': p_ap, 'p_out': p_out,
            'clip': 1.0 - (p_ap / p_in if p_in > 0 else 1.0),
            'elem': (p_out / p_ap) if p_ap > 0 else float('nan'),
            'raw_in': float(i_in.sum()), 'raw_out': float(
                (np.abs(Eout) ** 2).sum()),
            'flags': rec['flags']})
    for k in range(len(rows) - 1):
        rows[k]['leg_after'] = (rows[k + 1]['p_in'] / rows[k]['p_out']
                                if rows[k]['p_out'] > 0 else float('nan'))
    if rows:
        rows[-1]['leg_after'] = float('nan')
    tot = {}
    if rows:
        tot['end_to_end'] = rows[-1]['p_out'] / rows[0]['p_in']
        tot['prod_elem'] = float(np.prod([r['elem'] for r in rows]))
        tot['prod_keep'] = float(np.prod([1.0 - r['clip'] for r in rows]))
        tot['prod_leg'] = float(np.prod([r['leg_after']
                                         for r in rows[:-1]])) if len(
            rows) > 1 else 1.0
        tot['clip_total'] = 1.0 - tot['prod_keep']
    return rows, tot


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0').split()]
    cfgs = [c.strip() for c in
            os.environ.get('CONFIGS', 'ship,noC6,noC5').split(',') if c.strip()]
    rs = int(os.environ.get('RS', '4'))
    rn = int(os.environ.get('RN', '1024'))

    print("   lib %s  sha256 %s" % (
        os.path.basename(LT.__file__),
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    print(f"   RN={rn} ray_subsample={rs}  post-DOE groups only, "
          f"final_distance=0 final_leg='paraxial' (no readout stage)")
    _lo = 1.0 - (LT._RD_ENERGY_DEFICIT_BASE
                 + LT._RD_ENERGY_DEFICIT_PER_SUB * (rs - 1))
    print(f"   guard band at ray_subsample={rs}: "
          f"[{_lo:.4f}, {1.0 + LT._RD_ENERGY_GAIN_TOL:.4f}]")
    print()
    _pre, post, _g, _p = C.geometry()
    presc_last = post[-1]['prescription']
    surfs_last = P.element_surfaces(presc_last)

    for order in orders:
        # CHAIN-LEVEL NULL INTERVENTION: two identical shipped runs.  No delta
        # below this floor is a measurement.
        if os.environ.get('NULL', '1') == '1':
            _c1, _r1, _w1, _s1 = run(order, cfgs[0], rn=rn, rs=rs)
            _c2, _r2, _w2, _s2 = run(order, cfgs[0], rn=rn, rs=rs)
            _eq = all(np.array_equal(a['E_out'], b['E_out'])
                      for a, b in zip(_c1, _c2))
            _mx = max(float(np.abs(a['E_out'] - b['E_out']).max())
                      for a, b in zip(_c1, _c2))
            print(f"NULL INTERVENTION order {order} config {cfgs[0]}: "
                  f"all six stages array_equal={_eq}  max|dE|={_mx:.3e}")
            del _c1, _c2
        keep = {}
        for cname in cfgs:
            calls, res, cw, secs = run(order, cname, rn=rn, rs=rs)
            rows, tot = budget(calls, post)
            # keep the per-stage exits so guard OFF vs ON can be scored for
            # BYTE IDENTITY, not only for agreement to printed digits
            keep[cname] = [np.array(c['E_out'], copy=True) for c in calls]
            print(f"=== order {order}  config {cname} "
                  f"(C5={CONFIGS[cname]['c5']}, C6={CONFIGS[cname]['c6']}, "
                  f"GUARD={CONFIGS[cname].get('guard', False)})"
                  f"  [{secs:.0f}s] ===")
            hdr = (f"{'grp':>4} {'dx/um':>8} {'ap/mm':>7} {'clip':>10} "
                   f"{'P_in':>12} {'P_out':>12} {'elem':>9} {'leg_after-1':>11} "
                   f"{'ENG':>4} {'FLD':>4} {'NYQ':>4} {'chain P':>12}")
            print(hdr)
            print('-' * len(hdr))
            sp = [s.get('power', float('nan')) for s in res.stages]
            for r in rows:
                cp = sp[r['k']] if r['k'] < len(sp) else float('nan')
                print(f"{r['k']:>4} {r['dx'] * 1e6:>8.3f} "
                      f"{(r['ap'] * 1e3 if r['ap'] else float('nan')):>7.3f} "
                      f"{r['clip']:>10.3e} {r['p_in']:>12.6e} "
                      f"{r['p_out']:>12.6e} {r['elem']:>9.6f} "
                      f"{r['leg_after'] - 1.0:>10.2e} "
                      f"{r['flags']['energy']:>4d} {r['flags']['fold']:>4d} "
                      f"{r['flags']['undersample']:>4d} {cp:>12.6e}")
            print(f"  END TO END  P_out(last)/P_in(first) = "
                  f"{tot['end_to_end']:.6f}")
            print(f"    = prod(elem) {tot['prod_elem']:.6f} "
                  f"x prod(1-clip) {tot['prod_keep']:.6f} "
                  f"x prod(leg) {tot['prod_leg']:.6f}  "
                  f"[check {tot['prod_elem'] * tot['prod_keep'] * tot['prod_leg']:.6f}]")
            print(f"    aperture-clipped (legitimate) total: "
                  f"{tot['clip_total']:.3e}")
            # halo at the LAST group's exit, about its traced exit chief ray
            last = calls[-1]
            car = last['kw']['carrier']
            if not hasattr(car, 'R'):
                car = TiltedCarrier(float(car), 0.0, 0.0, 0.0, 0.0)
            dxl = float(last['kw']['dx'])
            Nl = last['E_out'].shape[0]
            axl = (np.arange(Nl) - Nl / 2) * dxl
            Xl, Yl = np.meshgrid(axl, axl)
            xo, yo, _q1, _q2, _q3, _q4 = P.trace_forward(
                [car.x0], [car.y0], car, surfs_last)
            Rr = np.hypot(Xl - float(xo[0]), Yl - float(yo[0]))
            h = G6.halo(last['E_out'], float((np.abs(last['E_in']) ** 2).sum()),
                        Rr)
            print(f"    HALO at last-group exit (about traced exit chief "
                  f"{float(xo[0]) * 1e3:+.4f},{float(yo[0]) * 1e3:+.4f} mm): "
                  f"g1 {h['g1']:.3e} g2 {h['g2']:.3e} g4 {h['g4']:.3e} "
                  f"g8 {h['g8']:.3e}")
            print(f"        amax4 {h['amax4']:.3e}  r_rms "
                  f"{h['r_rms'] * 1e3:.4f} mm  core "
                  f"{h['r_rms_core'] * 1e3:.4f} mm")
            for t, c in sorted(cw.items(), key=lambda kv: -kv[1])[:6]:
                print(f"    [chain warn x{c}] {t}")
            print(flush=True)
        # pairwise byte identity between the configs run for this order
        names = list(keep)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = keep[names[i]], keep[names[j]]
                eq = all(np.array_equal(u, v) for u, v in zip(a, b))
                mx = max(float(np.abs(u - v).max()) for u, v in zip(a, b))
                per = ''.join('=' if np.array_equal(u, v) else 'X'
                              for u, v in zip(a, b))
                print(f"  IDENTITY {names[i]} vs {names[j]}: "
                      f"all stages array_equal={eq}  max|dE|={mx:.3e}  "
                      f"per-stage [{per}]")
        print(flush=True)


if __name__ == '__main__':
    sys.exit(main())
