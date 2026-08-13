# INVERSE-MAP production build: the PER-ORDER acceptance measurement.
#
# LOCAL-ONLY.  No library edit here -- this drives the SHIPPED element both
# ways (``_lens_imap.TRACED_INVERSE_MAP`` False / True) at the retrace
# configuration ``_fine_trace_group_exit`` uses, and reports:
#
#   * the field delta between the two arms (max |dE| / peak, power ratio, and
#     the phase difference in waves over the beam core);
#   * the wall time of each arm, hence the measured per-order saving;
#   * the map's own guard record -- the G8 parity numbers against the very
#     Newton path it replaces, measured at OFF-LATTICE probe points (the
#     held-out NODE probe G8 used before FIX_G8_PROBE_2026_08_12 is retired);
#   * peak working set.
#
# FLAG OFF must be BYTE-IDENTICAL to the branch base.  That is checked here by
# running the OFF arm twice and hashing, and separately by the unit tests; the
# interesting number in this script is the ON-vs-OFF delta, which is the
# accuracy change the feature makes, and the wall delta, which is why it exists.
#
# Run:  python imap_prod_121.py                 (orders -4,-2 / 0,0 / -1,0)
#       IMAP_ORDERS="0,0" python imap_prod_121.py
#       IMAP_NFC=4096 python imap_prod_121.py    (cheaper, ~6 GB/arm)
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import imap_cost_121 as IC  # noqa: E402

import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_imap as IM  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_imap_prod.json')

NFC = int(os.environ.get('IMAP_NFC', '8192'))
REPS = int(os.environ.get('IMAP_REPS', '1'))


def _orders():
    raw = os.environ.get('IMAP_ORDERS', '-4,-2;0,0;-1,0')
    out = []
    for chunk in raw.split(';'):
        a, b = chunk.split(',')
        out.append((int(a), int(b)))
    return out


def _peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, 'peak_wset', mi.rss)) / 1e9
    except Exception:
        return float('nan')


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def one_order(order, verbose=True):
    E, kw, meta = IC._retrace_call_args(n_fine=NFC, order=order)
    lam = C.LAM
    res = {'order': list(order), **{k: v for k, v in meta.items()
                                    if k != 'order'}}
    arms = {}
    for tag, flag in (('off', False), ('on', True)):
        IM.TRACED_INVERSE_MAP = flag
        IM.inverse_map_cache_clear()
        best = None
        for _ in range(REPS):
            rec = {}
            t0 = time.perf_counter()
            out = EL.apply_real_lens_traced(E, _imap_out=rec, **kw)
            dt = time.perf_counter() - t0
            if best is None or dt < best[0]:
                best = (dt, out, rec)
        arms[tag] = best
        res['%s_seconds' % tag] = best[0]
        res['%s_sha' % tag] = _sha(best[1])
        if flag:
            res['guards'] = {k: v for k, v in best[2].items()
                             if isinstance(v, (int, float, str, bool))}
    IM.TRACED_INVERSE_MAP = True
    Eo, En = arms['off'][1], arms['on'][1]
    pk = float(np.abs(Eo).max())
    d = np.abs(En - Eo)
    p_off = float((np.abs(Eo) ** 2).sum())
    p_on = float((np.abs(En) ** 2).sum())
    # phase difference where BOTH arms carry real amplitude (the readout's own
    # territory): 1/e^2 of the peak, i.e. the disc the metrics are taken over.
    core = (np.abs(Eo) >= np.exp(-2.0) * pk) & (np.abs(En) > 0)
    if core.any():
        ph = np.angle(En[core] / Eo[core]) / (2.0 * np.pi)
        res['core_phase_max_waves'] = float(np.abs(ph).max())
        res['core_phase_rms_waves'] = float(np.sqrt((ph ** 2).mean()))
        res['core_amp_rel_max'] = float(
            np.abs(np.abs(En[core]) - np.abs(Eo[core])).max() / pk)
        res['n_core'] = int(core.sum())
    # RADIAL PROFILE of the two arms' disagreement, which is what says WHERE
    # it lives: a difference concentrated in the beam core is the inversion's
    # own accuracy, one concentrated in an annulus at the traced-support rim is
    # a DOMAIN difference, and the two call for opposite fixes.
    n = Eo.shape[0]
    ax = (np.arange(n) - n // 2)
    rr = np.hypot(ax[None, :], ax[:, None]).astype(np.float64)
    a_off, a_on = np.abs(Eo), np.abs(En)
    prof = []
    edges = np.linspace(0.0, float(rr.max()), 13)
    for i in range(len(edges) - 1):
        m = (rr >= edges[i]) & (rr < edges[i + 1])
        if not m.any():
            continue
        prof.append({
            'r_lo_px': float(edges[i]), 'r_hi_px': float(edges[i + 1]),
            'n': int(m.sum()),
            'dE_max_over_peak': float(d[m].max()) / max(pk, 1e-300),
            'amp_off_max_over_peak': float(a_off[m].max()) / max(pk, 1e-300),
            'd_amp_max_over_peak': float(
                np.abs(a_on[m] - a_off[m]).max()) / max(pk, 1e-300),
            'power_off': float((a_off[m] ** 2).sum()),
            'power_on': float((a_on[m] ** 2).sum())})
    res['radial'] = prof
    if verbose:
        print("  RADIAL (r in pixels; |E| relative to the OFF-arm peak)")
        for p in prof:
            print("    r %6.0f-%6.0f  |dE| %.3e  d|E| %.3e  |E|off %.3e  "
                  "power on/off %.6f"
                  % (p['r_lo_px'], p['r_hi_px'], p['dE_max_over_peak'],
                     p['d_amp_max_over_peak'], p['amp_off_max_over_peak'],
                     p['power_on'] / max(p['power_off'], 1e-300)))
    del rr, a_off, a_on
    res['max_abs_dE_over_peak'] = float(d.max()) / max(pk, 1e-300)
    res['power_ratio_on_over_off'] = p_on / max(p_off, 1e-300)
    res['seconds_saved'] = res['off_seconds'] - res['on_seconds']
    res['peak_wset_gb'] = _peak_gb()
    res['wavelength'] = lam
    del Eo, En, d, core, arms
    if verbose:
        print("order %-9s n_fine %d dx %.4f um sub %d n_launch %d"
              % (str(tuple(order)), meta['n_fine'], meta['dx'] * 1e6,
                 meta['ray_subsample'], meta['n_launch']))
        g = res['guards']
        print("  map: degree %d, %d terms, %.1f kB, build %.3f s, %d fit "
              "samples, %d hull facets, det J range %.4f"
              % (g.get('degree', -1), g.get('n_terms', -1),
                 g.get('bytes', 0) / 1e3, g.get('build_seconds', float('nan')),
                 g.get('n_fit_samples', -1), g.get('n_hull_facets', -1),
                 g.get('det_j_range', float('nan'))))
        print("  G8 parity (off-lattice probe points, %d): map %.4e waves vs "
              "incumbent Newton %.4e  -> %.4gx"
              % (g.get('n_parity', -1), g.get('parity_map_opl_waves',
                                              float('nan')),
                 g.get('parity_incumbent_opl_waves', float('nan')),
                 g.get('parity_ratio_opl', float('nan'))))
        print("      entrance position: map %.4e m vs incumbent %.4e m "
              "-> %.4gx"
              % (g.get('parity_map_pos_m', float('nan')),
                 g.get('parity_incumbent_pos_m', float('nan')),
                 g.get('parity_ratio_pos', float('nan'))))
        print("  G7 fit residual: OPL %.4e waves, entrance %.4e m"
              % (g.get('fit_resid_opl_waves', float('nan')),
                 g.get('fit_resid_x_in_m', float('nan'))))
        print("  FIELD  max|dE|/peak %.4e   power ratio %.9f   core phase "
              "max %.4e rms %.4e waves"
              % (res['max_abs_dE_over_peak'],
                 res['power_ratio_on_over_off'],
                 res.get('core_phase_max_waves', float('nan')),
                 res.get('core_phase_rms_waves', float('nan'))))
        print("  WALL   off %.1f s   on %.1f s   SAVED %.1f s (%.1f %%)   "
              "peak %.1f GB"
              % (res['off_seconds'], res['on_seconds'], res['seconds_saved'],
                 100.0 * res['seconds_saved'] / max(res['off_seconds'], 1e-9),
                 res['peak_wset_gb']))
    return res


def census(order=(-4, -2), verbose=True):
    """The LOAD-ROBUST cost measurement: both arms in ONE process, with a
    pass-through spy on ``scipy.ndimage.map_coordinates`` and a clock on the
    inverse map's own block.

    The wall-clock delta between two element calls on a box carrying other
    campaigns is dominated by the box (this study measured +-9 s of scatter on
    a 32 s element).  What is NOT dominated by the box is the BUCKET census:
    which full-grid interpolations each arm runs and what each one costs,
    inside the same process, seconds apart.  That is the number the saving is
    quoted from -- the same device ``PROTO_INVERSE_MAP_2026_08_11`` S4.3 used.
    """
    import scipy.ndimage as _nd
    E, kw, meta = IC._retrace_call_args(n_fine=NFC, order=order)
    out = {'order': list(order), 'n_fine': meta['n_fine'],
           'n_pixels': meta['n_pixels'], 'arms': {}}
    for tag, flag in (('off', False), ('on', True)):
        calls = []
        orig = _nd.map_coordinates

        def _spy(inp, coords, *a, **k):
            t0 = time.perf_counter()
            r = orig(inp, coords, *a, **k)
            calls.append({'in_shape': list(np.shape(inp)),
                          'out_points': int(np.size(coords) // 2),
                          'order': int(k.get('order',
                                             a[1] if len(a) > 1 else 3)),
                          'seconds': time.perf_counter() - t0})
            return r

        IM.TRACED_INVERSE_MAP = flag
        IM.inverse_map_cache_clear()
        _nd.map_coordinates = _spy
        rec = {}
        try:
            t0 = time.perf_counter()
            EL.apply_real_lens_traced(E, _imap_out=rec, **kw)
            dt = time.perf_counter() - t0
        finally:
            _nd.map_coordinates = orig
        full = [c for c in calls
                if c['out_points'] >= meta['n_pixels'] // 2]
        out['arms'][tag] = {
            'element_seconds': dt, 'calls': calls,
            'map_coordinates_total_s': sum(c['seconds'] for c in calls),
            'full_grid_calls': len(full),
            'full_grid_seconds': sum(c['seconds'] for c in full),
            'build_seconds': rec.get('build_seconds'),
            'engaged': rec.get('engaged'), 'refused': rec.get('refused')}
        if verbose:
            print("  [%s] element %.1f s;  map_coordinates %d calls "
                  "%.3f s (of which %d FULL-GRID, %.3f s);  imap build %s"
                  % (tag, dt, len(calls),
                     out['arms'][tag]['map_coordinates_total_s'], len(full),
                     out['arms'][tag]['full_grid_seconds'],
                     ('%.3f s' % rec['build_seconds'])
                     if rec.get('build_seconds') else '--'))
            for i, c in enumerate(calls):
                print("        %2d  in %-14s -> %10d pts  order %d  %7.3f s"
                      % (i, str(tuple(c['in_shape'])), c['out_points'],
                         c['order'], c['seconds']))
    IM.TRACED_INVERSE_MAP = True
    a, b = out['arms']['off'], out['arms']['on']
    out['bucket_saving_seconds'] = (a['full_grid_seconds']
                                    - b['full_grid_seconds']
                                    - (b['build_seconds'] or 0.0))
    if verbose:
        print("  FULL-GRID interpolation bucket: off %.3f s -> on %.3f s; "
              "minus the %.3f s build => %.3f s per order"
              % (a['full_grid_seconds'], b['full_grid_seconds'],
                 b['build_seconds'] or 0.0, out['bucket_saving_seconds']))
    return out


def main():
    if os.environ.get('IMAP_MODE') == 'census':
        blob = {'census': [census(o) for o in _orders()], 'n_fine': NFC}
        with open(OUT.replace('.json', '_census.json'), 'w',
                  encoding='ascii') as fh:
            json.dump(blob, fh, indent=1, sort_keys=True, default=float)
        print('wrote %s' % os.path.basename(OUT.replace('.json',
                                                        '_census.json')))
        return
    rows = []
    for o in _orders():
        rows.append(one_order(o))
        print('')
    blob = {'n_fine': NFC, 'reps': REPS, 'rows': rows,
            'exit_degree': IM._IMAP_EXIT_DEGREE,
            'parity_factor': IM._IMAP_PARITY_FACTOR}
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump(blob, fh, indent=1, sort_keys=True, default=float)
    tot = sum(r['seconds_saved'] for r in rows)
    print("SUMMARY over %d orders: saved %.1f s total, %.1f s/order mean; "
          "32-order projection %.0f s"
          % (len(rows), tot, tot / max(len(rows), 1),
             32.0 * tot / max(len(rows), 1)))
    print("wrote %s" % os.path.basename(OUT))


if __name__ == '__main__':
    main()
