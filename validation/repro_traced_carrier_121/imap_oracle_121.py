# INVERSE-MAP production build, S6: THE INDEPENDENT ORACLE.
#
# LOCAL-ONLY.  The only library surface used is the private, opt-in
# ``_imap_out['probe_rc']`` diagnostic (which returns each arm's finalised
# ``opl_map`` / ``ard_map`` sampled at named pixels and retains nothing else),
# plus a pass-through spy on ``_fit_residual_eikonal`` to capture the ONE
# analytic object the oracle must share with both arms.
#
# THE QUESTION S5.4 LEFT.  With the evaluator scoped to the terminal leg the
# design-121 banner reads 3.450 um against the shipped 3.350 um.  Both cannot
# be faithful.  Deciding it needs an oracle that shares NO INVERSION machinery
# with either arm.
#
# WHAT THE ARMS SHARE AND WHAT THEY DO NOT.  Both arms trace the same rays
# through the same surfaces along the same congruence, and both add the same
# analytic entrance eikonal ``W + a_fit``.  That is the PHYSICS and the oracle
# uses it too -- it is what "the exit field of this group" MEANS.  What the
# arms do NOT share, and what is on trial, is the INVERSION: the incumbent
# Newtons a 95 x 95 coarse lattice on a degree-10 forward Chebyshev fit and
# cubically upsamples; the map fits a degree-14 polynomial in exit coordinates
# and evaluates it per pixel.  The oracle uses NEITHER.  It inverts by
# EXACT-TRACE NEWTON: the residual and its Jacobian both come from real traced
# rays (a 5-point stencil of actual ``trace()`` calls), so no polynomial model
# of any degree appears anywhere in the truth.
#
# THE OBSERVABLE is each arm's own ``opl_map`` and ``ard_map`` at named exit
# pixels, against the exact traced OPL and the exact ray-tube amplitude at the
# same pixels.  Scored separately in the CORE (|E| >= e^-2 of peak) and the
# SKIRT (below it), because S5 measured the arms to agree in the first and
# differ by 1.2-5.9 % of peak in the second.
#
# Run:  python imap_oracle_121.py                    (~15 min, ~22 GB)
#       IMAP_NFC=4096 python imap_oracle_121.py      (~5 min, ~6 GB)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import hmap_probe_121 as H  # noqa: E402
import imap_cost_121 as IC  # noqa: E402

import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_imap as IM  # noqa: E402
import lumenairy.elements._lens_traced as LT  # noqa: E402
from lumenairy.elements._lens_traced import _tilted_carrier_parts  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_imap_oracle.json')

NFC = int(os.environ.get('IMAP_NFC', '8192'))
N_RING = int(os.environ.get('ORACLE_RINGS', '40'))
N_AZ = int(os.environ.get('ORACLE_AZ', '48'))
SEED_LAT = int(os.environ.get('ORACLE_SEED', '513'))
NEWTON_H = 2.0e-8          # exact-trace stencil half-step, metres
NEWTON_IT = 6


def _orders():
    raw = os.environ.get('IMAP_ORDERS', '-4,-2;0,0')
    return [tuple(int(v) for v in c.split(',')) for c in raw.split(';')]


# ===========================================================================
# the congruence: the launch direction field BOTH arms use
# ===========================================================================
def _launch_dirs(carrier, a_fit, px, py):
    """``grad(W + a_fit)`` at the entrance heights -- the element's own launch
    rule (``_lens_traced``: the two gradients ADD, with no renormalisation,
    because grad(eikonal) IS the transverse direction cosine)."""
    _W, L, M = _tilted_carrier_parts(carrier, np.asarray(px, dtype=float),
                                     np.asarray(py, dtype=float))
    L = np.asarray(L, dtype=np.float64).ravel().copy()
    M = np.asarray(M, dtype=np.float64).ravel().copy()
    if a_fit is not None:
        gL, gM = a_fit.grad(np.ravel(px), np.ravel(py))
        L += np.asarray(gL, dtype=np.float64).ravel()
        M += np.asarray(gM, dtype=np.float64).ravel()
    return L, M


def _trace_at(surfs, carrier, a_fit, px, py):
    """Exact exit landing + TOTAL OPL (geometric path + the analytic entrance
    eikonal ``W + a_fit``), i.e. exactly the quantity the element's own
    ``opl_grid`` holds before its on-axis reference is removed."""
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    L, M = _launch_dirs(carrier, a_fit, px, py)
    ch = H.characteristic(surfs, px, py, L, M)
    W, _L, _M = _tilted_carrier_parts(carrier, px, py)
    opl = np.asarray(ch['opl'], dtype=np.float64) + np.asarray(W, float)
    if a_fit is not None:
        opl = opl + np.asarray(a_fit.value(px, py), dtype=np.float64)
    return (np.asarray(ch['x'], float), np.asarray(ch['y'], float), opl,
            np.asarray(ch['alive'], bool))


# ===========================================================================
# the EXACT-TRACE inverse -- no polynomial model anywhere
# ===========================================================================
def exact_inverse(surfs, carrier, a_fit, xt, yt, seed_x, seed_y,
                  h=NEWTON_H, iters=NEWTON_IT):
    """Solve ``(x_out, y_out)(x_in, y_in) = (xt, yt)`` by Newton on the EXACT
    TRACE.  The residual is a traced ray; the Jacobian is a 4-point central
    stencil of traced rays.  Every quantity is a real ray -- there is no fit,
    no spline and no Chebyshev anywhere in this function, which is the whole
    point of it."""
    xi = np.asarray(seed_x, dtype=np.float64).copy()
    yi = np.asarray(seed_y, dtype=np.float64).copy()
    n = xi.size
    ok = np.ones(n, dtype=bool)
    for _ in range(iters):
        xo, yo, _o, al = _trace_at(surfs, carrier, a_fit, xi, yi)
        rx = xo - xt
        ry = yo - yt
        xp, yp, _o2, a1 = _trace_at(surfs, carrier, a_fit, xi + h, yi)
        xm, ym, _o3, a2 = _trace_at(surfs, carrier, a_fit, xi - h, yi)
        xq, yq, _o4, a3 = _trace_at(surfs, carrier, a_fit, xi, yi + h)
        xr, yr, _o5, a4 = _trace_at(surfs, carrier, a_fit, xi, yi - h)
        ok &= al & a1 & a2 & a3 & a4
        jxx = (xp - xm) / (2 * h)
        jyx = (yp - ym) / (2 * h)
        jxy = (xq - xr) / (2 * h)
        jyy = (yq - yr) / (2 * h)
        det = jxx * jyy - jxy * jyx
        good = np.abs(det) > 1e-30
        inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        xi = xi - (jyy * rx - jxy * ry) * inv
        yi = yi - (-jyx * rx + jxx * ry) * inv
        ok &= good
    xo, yo, opl, al = _trace_at(surfs, carrier, a_fit, xi, yi)
    resid = np.hypot(xo - xt, yo - yt)
    # the exact forward Jacobian at the converged entrance point, for the
    # ray-tube amplitude |E_in| / sqrt(|det J|)
    xp, yp, _a, a1 = _trace_at(surfs, carrier, a_fit, xi + h, yi)
    xm, ym, _b, a2 = _trace_at(surfs, carrier, a_fit, xi - h, yi)
    xq, yq, _c, a3 = _trace_at(surfs, carrier, a_fit, xi, yi + h)
    xr, yr, _d, a4 = _trace_at(surfs, carrier, a_fit, xi, yi - h)
    detj = (((xp - xm) * (yq - yr) - (xq - xr) * (yp - ym))
            / (4.0 * h * h))
    return {'x_in': xi, 'y_in': yi, 'opl': opl, 'det_j': detj,
            'resid': resid, 'ok': ok & al & a1 & a2 & a3 & a4}


# ===========================================================================
def one_order(order, verbose=True):
    E, kw, meta = IC._retrace_call_args(n_fine=NFC, order=order)
    N = int(meta['n_fine'])
    dx = float(meta['dx'])
    lam = C.LAM
    ax = (np.arange(N) - N // 2) * dx

    # ---- ARM B (incumbent) first, with a spy that captures ``a_fit`` -------
    o_fit = LT._fit_residual_eikonal
    got = []

    def _fit(*a, **k):
        out = o_fit(*a, **k)
        got.append(out)
        return out

    # ---- the seed trace, FIRST, because it also sizes the probe -----------
    # The rings must land INSIDE the traced exit support: outside it neither
    # arm has data and the comparison would measure two extrapolations against
    # a truth that does not exist.  So the support is MEASURED here (the exit
    # extent of a dense forward trace) and the rings are placed inside it,
    # rather than guessed as a fraction of the grid.
    groups_post, _rows_tab, _R_ent, _per, _rd = H.geometry()
    surfs, _presc = H.group_surfaces(groups_post)
    car = kw['carrier']
    _o_fit0 = LT._fit_residual_eikonal

    # probe pixels: concentric rings out to the exit support, so the census
    # spans the CORE and the SKIRT rather than sampling one of them densely.
    # CENTRED ON THIS ORDER'S OWN EXIT FOOTPRINT, not on the grid axis.  A
    # tilted congruence lands off axis (order (-4,-2)'s chief ray exits ~3 mm
    # out), so axis-centred rings would spend most of their samples where that
    # order HAS NO RAYS -- and a comparison of two extrapolations against a
    # truth that does not exist there measures nothing.  The centre and the
    # radius are both MEASURED from a dense forward trace of this congruence.
    _lr0 = 0.75 * float(groups_post[-1]['prescription']['aperture_diameter'])
    _s0 = np.linspace(-_lr0, _lr0, 129)
    _SX0, _SY0 = np.meshgrid(_s0, _s0, indexing='ij')
    _ex, _ey, _eo, _eok = _trace_at(surfs, car, None, _SX0.ravel(),
                                    _SY0.ravel())
    _cx = 0.5 * (float(_ex[_eok].max()) + float(_ex[_eok].min()))
    _cy = 0.5 * (float(_ey[_eok].max()) + float(_ey[_eok].min()))
    _r_sup = float(np.nanmax(np.hypot(_ex[_eok] - _cx, _ey[_eok] - _cy)))
    _r_out = min(0.90 * _r_sup, 0.45 * N * dx)
    rr = np.linspace(0.02, 1.0, N_RING) * _r_out
    th = np.linspace(0.0, 2.0 * np.pi, N_AZ, endpoint=False)
    px = np.concatenate([_cx + r * np.cos(th) for r in rr])
    py = np.concatenate([_cy + r * np.sin(th) for r in rr])
    cols = np.clip(np.rint(px / dx + N // 2).astype(np.intp), 0, N - 1)
    rows = np.clip(np.rint(py / dx + N // 2).astype(np.intp), 0, N - 1)
    xt = ax[cols]
    yt = ax[rows]

    arms = {}
    for tag, flag in (('incumbent', False), ('map', True)):
        rec = {'probe_rc': (rows, cols)}
        old = IM.TRACED_INVERSE_MAP
        IM.inverse_map_cache_clear()
        LT._fit_residual_eikonal = _fit if tag == 'incumbent' else o_fit
        try:
            IM.TRACED_INVERSE_MAP = flag
            t0 = time.perf_counter()
            EL.apply_real_lens_traced(E, _imap_out=rec, **kw)
            rec['seconds'] = time.perf_counter() - t0
        finally:
            IM.TRACED_INVERSE_MAP = old
            LT._fit_residual_eikonal = o_fit
        arms[tag] = rec
    a_fit = got[-1] if got else None

    # ---- THE ORACLE -------------------------------------------------------
    t0 = time.perf_counter()
    lr = 0.75 * float(groups_post[-1]['prescription']['aperture_diameter'])
    sx = np.linspace(-lr, lr, SEED_LAT)
    SX, SY = np.meshgrid(sx, sx, indexing='ij')
    so_x, so_y, _so, s_ok = _trace_at(surfs, car, a_fit, SX.ravel(), SY.ravel())
    so_x = np.where(s_ok, so_x, 1e9)
    so_y = np.where(s_ok, so_y, 1e9)
    # nearest traced landing as the Newton seed (chunked: the full
    # (n_probe, SEED_LAT^2) distance matrix would be 4 GB)
    seed_x = np.empty(xt.size)
    seed_y = np.empty(xt.size)
    sxr, syr = SX.ravel(), SY.ravel()
    CH = max(1, int(4e7 // max(so_x.size, 1)))
    for s in range(0, xt.size, CH):
        e = min(s + CH, xt.size)
        d2 = ((so_x[None, :] - xt[s:e, None]) ** 2
              + (so_y[None, :] - yt[s:e, None]) ** 2)
        k = np.argmin(d2, axis=1)
        seed_x[s:e] = sxr[k]
        seed_y[s:e] = syr[k]
        del d2
    tru = exact_inverse(surfs, car, a_fit, xt, yt, seed_x, seed_y)
    t_oracle = time.perf_counter() - t0

    # the element's own on-axis OPL reference, reproduced exactly: the axis
    # launch node of an ODD lattice is (0, 0).
    _rx, _ry, opl0, _ok0 = _trace_at(surfs, car, a_fit, np.array([0.0]),
                                     np.array([0.0]))
    opl_ref = float(opl0[0])
    opl_truth = tru['opl'] - opl_ref

    # the exact ray-tube amplitude at those pixels
    absin = np.abs(np.asarray(E)).astype(np.float64)
    from scipy.ndimage import map_coordinates as _mc
    a_in = _mc(absin, np.vstack([tru['y_in'] / dx + N / 2.0,
                                 tru['x_in'] / dx + N / 2.0]),
               order=1, mode='constant', cval=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_truth = a_in / np.sqrt(np.abs(tru['det_j']))

    # convergence + validity of the oracle itself
    good = tru['ok'] & (tru['resid'] < 0.02 * dx) & np.isfinite(opl_truth)
    res = {'order': list(order), 'n_fine': N, 'dx': dx,
           'support_radius_mm': _r_sup * 1e3, 'probe_r_out_mm': _r_out * 1e3,
           'probe_centre_mm': [_cx * 1e3, _cy * 1e3],
           'n_probe': int(xt.size), 'n_oracle_valid': int(good.sum()),
           'oracle_resid_max_m': float(np.nanmax(tru['resid'][good]))
           if good.any() else float('nan'),
           'oracle_seconds': t_oracle, 'opl_ref_m': opl_ref,
           'a_fit_terms': (len(a_fit.terms) if a_fit is not None else 0),
           'arms': {}}

    # CORE vs SKIRT, by the arms' own amplitude: the core is the 1/e^2 disc
    # of the incumbent's ray-density amplitude, which is the region every
    # banner metric is computed on.
    ard_i = arms['incumbent'].get('probe_ard')
    ref_pk = (float(np.nanmax(ard_i)) if ard_i is not None
              else float(np.nanmax(amp_truth[good])))
    base = ard_i if ard_i is not None else amp_truth
    core = good & np.isfinite(base) & (base >= np.exp(-2.0) * ref_pk)
    skirt = good & ~core
    res['n_core'] = int(core.sum())
    res['n_skirt'] = int(skirt.sum())

    for tag in ('incumbent', 'map'):
        opl_a = np.asarray(arms[tag]['probe_opl'], dtype=np.float64)
        e = np.abs(opl_a - opl_truth) / lam
        d = {'seconds': arms[tag]['seconds'],
             'n_finite': int(np.isfinite(opl_a[good]).sum())}
        for nm, m in (('all', good), ('core', core), ('skirt', skirt)):
            mm = m & np.isfinite(e)
            d['opl_%s_max_waves' % nm] = (float(e[mm].max()) if mm.any()
                                          else float('nan'))
            d['opl_%s_rms_waves' % nm] = (
                float(np.sqrt((e[mm] ** 2).mean())) if mm.any()
                else float('nan'))
            d['n_%s' % nm] = int(mm.sum())
        ard = arms[tag].get('probe_ard')
        if ard is not None:
            ard = np.asarray(ard, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                ea = np.abs(ard - amp_truth) / np.maximum(amp_truth, 1e-300)
            for nm, m in (('core', core), ('skirt', skirt)):
                mm = m & np.isfinite(ea)
                d['amp_%s_rel_rms' % nm] = (
                    float(np.sqrt((ea[mm] ** 2).mean())) if mm.any()
                    else float('nan'))
        res['arms'][tag] = d

    if verbose:
        a, b = res['arms']['incumbent'], res['arms']['map']
        print("order %-9s n_fine %d dx %.4f um  |  oracle: %d/%d probe "
              "pixels converged, max exit residual %.3e m (%.4f px), %.1f s"
              % (str(tuple(order)), N, dx * 1e6, res['n_oracle_valid'],
                 res['n_probe'], res['oracle_resid_max_m'],
                 res['oracle_resid_max_m'] / dx, t_oracle))
        print("  region        n      INCUMBENT rms / max (waves)      "
              "MAP rms / max (waves)          ratio(rms)")
        for nm in ('all', 'core', 'skirt'):
            r_i, r_m = a['opl_%s_rms_waves' % nm], b['opl_%s_rms_waves' % nm]
            print("  %-12s %-6d %.4e / %.4e        %.4e / %.4e   %.4g"
                  % (nm, a['n_%s' % nm], r_i, a['opl_%s_max_waves' % nm],
                     r_m, b['opl_%s_max_waves' % nm],
                     (r_m / r_i) if r_i > 0 else float('nan')))
        if 'amp_core_rel_rms' in a:
            print("  amplitude rel rms   core: incumbent %.4e  map %.4e   |"
                  "   skirt: incumbent %.4e  map %.4e"
                  % (a['amp_core_rel_rms'], b['amp_core_rel_rms'],
                     a['amp_skirt_rel_rms'], b['amp_skirt_rel_rms']))
    return res


def main():
    rows = [one_order(o) for o in _orders()]
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump({'rows': rows, 'n_fine': NFC}, fh, indent=1,
                  sort_keys=True, default=float)
    print("\nVERDICT INPUT (rms over the ALL region, waves):")
    for r in rows:
        a, b = r['arms']['incumbent'], r['arms']['map']
        print("  order %-9s incumbent %.4e   map %.4e   -> map is %.4gx"
              % (str(tuple(r['order'])), a['opl_all_rms_waves'],
                 b['opl_all_rms_waves'],
                 b['opl_all_rms_waves'] / max(a['opl_all_rms_waves'], 1e-300)))
    print("wrote %s" % os.path.basename(OUT))


if __name__ == '__main__':
    main()
