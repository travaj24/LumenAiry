"""VERIFY-WP-C1 ROUND 2 -- the downstream boolean-cast exposure, MEASURED.

VERIFY-C1 measured the DILATION (a grey mask boolean-casts to +1.37 % more
true pixels than the hard mask on a 12281-px disc) but recorded the EXPOSURE as
unmeasurable, and round 2 changed nothing about it and repeated the note.  This
probe measures the exposure: it builds the smallest fixture that reaches each
boolean-casting site with a caller-supplied ``apply_aperture`` result and reads
the quantity the site actually returns, on the hard mask and on the grey mask.

Sites (VERIFY-C1 sec. 2.7):

* ``analysis/plotting.py:1757`` -- ``n_in_ap = count_nonzero(aperture.astype
  (bool))`` fed to ``_auto_n_bins``, which sets ``plot_opd_summary``'s radial
  bin count.  Reached by ``plot_opd_summary(..., aperture=<array>,
  radial_rms_n_bins='auto')``.
* ``analysis/plotting.py:1491`` -- the same cast inside
  ``_radial_rms_profile``, which decides ``r_max`` and which pixels each
  annulus averages.  This is the curve the panel plots.
* ``analysis/plotting.py:1096`` / ``:1724`` -- the NaN masks of the heatmap and
  of the two fan panels.
* ``optimize/wrapper_merits.py:266`` -- ``mask = asarray(aperture, dtype=bool)``
  in the wrapper-merit grid cache; the mask a merit integrates over.

Every reading is a NUMBER the library returns or plots, not a pixel count
alone, so "does the answer move" is answered rather than "could it".

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_boolcast_exposure.py <out.json>
"""
import json
import sys

import numpy as np

import lumenairy  # noqa: F401
from lumenairy.elements.elements import apply_aperture

N, DX, D = 256, 4e-6, 0.5e-3


def masks():
    """The two ways a CALLER builds an aperture array with this library."""
    E = np.ones((N, N), dtype=complex)
    hard = np.real(apply_aperture(E, DX, 'circular', {'diameter': D},
                                  edge='hard'))
    grey = np.real(apply_aperture(E, DX, 'circular', {'diameter': D}))
    return hard, grey


def main(out_path):
    from lumenairy.analysis.plotting import (_auto_n_bins,
                                             _radial_rms_profile)
    hard, grey = masks()
    out = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__}

    n_hard = int(np.count_nonzero(hard.astype(bool)))
    n_grey = int(np.count_nonzero(grey.astype(bool)))
    analytic = np.pi * (D / 2) ** 2 / (DX * DX)
    out['dilation'] = {
        'n_true_hard': n_hard, 'n_true_grey': n_grey,
        'analytic_disc_px': analytic,
        'delta_px': n_grey - n_hard,
        'delta_pct': 100.0 * (n_grey - n_hard) / n_hard,
        'rim_px': int(np.count_nonzero((grey > 0) & (grey < 1))),
    }

    # ---- site plotting.py:1757 -- the auto bin count ----------------------
    bins_hard = int(_auto_n_bins(n_hard, ceiling=32))
    bins_grey = int(_auto_n_bins(n_grey, ceiling=32))
    # and the same quantity on the grids where _auto_n_bins is not saturated
    small = {}
    for n_small in (12, 24, 48, 96, 160, 256):
        Esm = np.ones((n_small, n_small), dtype=complex)
        dsm = 0.6 * n_small * DX
        h = np.real(apply_aperture(Esm, DX, 'circular', {'diameter': dsm},
                                   edge='hard'))
        g = np.real(apply_aperture(Esm, DX, 'circular', {'diameter': dsm}))
        nh = int(np.count_nonzero(h.astype(bool)))
        ng = int(np.count_nonzero(g.astype(bool)))
        small[str(n_small)] = {
            'n_hard': nh, 'n_grey': ng,
            'bins_hard': int(_auto_n_bins(nh, ceiling=32)),
            'bins_grey': int(_auto_n_bins(ng, ceiling=32)),
            'bins_move': int(_auto_n_bins(nh, ceiling=32))
                         != int(_auto_n_bins(ng, ceiling=32)),
        }
    out['site_1757_auto_n_bins'] = {
        'N256_bins_hard': bins_hard, 'N256_bins_grey': bins_grey,
        'N256_bins_move': bins_hard != bins_grey,
        'ladder': small,
        'any_bin_count_moves': any(v['bins_move'] for v in small.values()),
    }

    # ---- site plotting.py:1491 -- the radial RMS curve --------------------
    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt(((xx - (N - 1) / 2.0) * DX) ** 2
                 + ((yy - (N - 1) / 2.0) * DX) ** 2)
    opd = 1e-6 * (0.4 * (rr / (D / 2)) ** 2 - 0.15 * (rr / (D / 2)) ** 4)
    rc_h, rms_h = _radial_rms_profile(opd, DX, DX, hard, n_bins=16)
    rc_g, rms_g = _radial_rms_profile(opd, DX, DX, grey, n_bins=16)
    with np.errstate(invalid='ignore'):
        rel = np.abs(rms_g - rms_h) / np.maximum(np.abs(rms_h), 1e-30)
    out['site_1491_radial_rms'] = {
        'rms_hard': [float(v) for v in rms_h],
        'rms_grey': [float(v) for v in rms_g],
        'identical': bool(np.array_equal(rms_h, rms_g)),
        'max_abs_rel_move': float(np.nanmax(rel)),
        'last_bin_rel_move': float(rel[-1]),
        'r_centres_identical': bool(np.array_equal(rc_h, rc_g)),
    }

    # ---- sites plotting.py:1096 / :1724 -- the NaN masks ------------------
    disp_h = np.where(hard.astype(bool), opd, np.nan)
    disp_g = np.where(grey.astype(bool), opd, np.nan)
    col = (N - 1) // 2
    out['site_1096_1724_nan_masks'] = {
        'n_finite_hard': int(np.count_nonzero(np.isfinite(disp_h))),
        'n_finite_grey': int(np.count_nonzero(np.isfinite(disp_g))),
        'in_aperture_rms_hard': float(np.sqrt(np.nanmean(
            (disp_h - np.nanmean(disp_h)) ** 2))),
        'in_aperture_rms_grey': float(np.sqrt(np.nanmean(
            (disp_g - np.nanmean(disp_g)) ** 2))),
        'in_aperture_pv_hard': float(np.nanmax(disp_h) - np.nanmin(disp_h)),
        'in_aperture_pv_grey': float(np.nanmax(disp_g) - np.nanmin(disp_g)),
        'fan_n_finite_hard': int(np.count_nonzero(
            hard.astype(bool)[:, col])),
        'fan_n_finite_grey': int(np.count_nonzero(
            grey.astype(bool)[:, col])),
    }
    rms_h2 = out['site_1096_1724_nan_masks']['in_aperture_rms_hard']
    rms_g2 = out['site_1096_1724_nan_masks']['in_aperture_rms_grey']
    out['site_1096_1724_nan_masks']['rms_rel_move'] = abs(
        rms_g2 - rms_h2) / abs(rms_h2)

    # ---- site wrapper_merits.py:266 -- the merit's integration mask -------
    from lumenairy.optimize.wrapper_merits import (_get_wrapper_merit_cache,
                                                   _clear_wrapper_merit_cache)
    _clear_wrapper_merit_cache()
    cdtype = np.complex128
    ch = _get_wrapper_merit_cache(N, DX, hard, cdtype)
    cg = _get_wrapper_merit_cache(N, DX, grey, cdtype)
    mh, mg = ch['mask'], cg['mask']
    # what a merit actually integrates: |E|^2 over the mask, on a field with
    # structure at the rim (an Airy-like apodisation), so the rim ring is not
    # weightless.
    amp = np.exp(-((rr / (D / 2)) ** 2) * 0.35) * (1.0 + 0.6 * np.cos(
        8.0 * np.arctan2(yy - (N - 1) / 2.0, xx - (N - 1) / 2.0)))
    inten = amp ** 2
    p_h = float(inten[mh].sum())
    p_g = float(inten[mg].sum())
    out['site_266_wrapper_merit'] = {
        'mask_true_hard': int(mh.sum()), 'mask_true_grey': int(mg.sum()),
        'mask_delta_px': int(mg.sum() - mh.sum()),
        'integrated_power_hard': p_h, 'integrated_power_grey': p_g,
        'integrated_power_rel_move': abs(p_g - p_h) / abs(p_h),
        'masks_identical': bool(np.array_equal(mh, mg)),
        # the SAME quantity the grey mask is FOR: a weighted (non-boolean) sum
        'weighted_power_grey_mask': float((inten * grey).sum()),
        'boolean_overshoot_vs_weighted': (
            p_g - float((inten * grey).sum())) / float((inten * grey).sum()),
    }
    # reachability: which in-library callers can put an ARRAY here?
    out['site_266_reachability'] = {
        'in_library_callers_pass': 'ctx.prescription["aperture_diameter"]',
        'note': ('two of the three in-library call sites float() the value '
                 'first; the third (line 492) forwards it unfloated, so an '
                 'ndarray aperture_diameter in a prescription reaches the '
                 'array branch'),
    }
    _clear_wrapper_merit_cache()

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)

    print('lumenairy:', lumenairy.__file__)
    d = out['dilation']
    print("dilation: hard={0} grey={1} analytic={2:.1f} delta=+{3} "
          "(+{4:.4f} %) rim={5}".format(
              d['n_true_hard'], d['n_true_grey'], d['analytic_disc_px'],
              d['delta_px'], d['delta_pct'], d['rim_px']))
    s = out['site_1757_auto_n_bins']
    print("1757 _auto_n_bins: N256 {0} -> {1} (moves={2}); "
          "any grid moves={3}".format(s['N256_bins_hard'],
                                      s['N256_bins_grey'],
                                      s['N256_bins_move'],
                                      s['any_bin_count_moves']))
    for k, v in s['ladder'].items():
        print("    N={0:>4s} n {1} -> {2}  bins {3} -> {4}  move={5}".format(
            k, v['n_hard'], v['n_grey'], v['bins_hard'], v['bins_grey'],
            v['bins_move']))
    r = out['site_1491_radial_rms']
    print("1491 radial RMS curve: identical={0} max rel move={1:.4e} "
          "last bin={2:.4e}".format(r['identical'], r['max_abs_rel_move'],
                                    r['last_bin_rel_move']))
    m = out['site_1096_1724_nan_masks']
    print("1096/1724 NaN mask: finite {0} -> {1}; in-aperture RMS rel move "
          "{2:.4e}; PV {3:.6e} -> {4:.6e}".format(
              m['n_finite_hard'], m['n_finite_grey'], m['rms_rel_move'],
              m['in_aperture_pv_hard'], m['in_aperture_pv_grey']))
    w = out['site_266_wrapper_merit']
    print("266 wrapper merit: mask {0} -> {1} (+{2}); integrated power rel "
          "move {3:.4e}; boolean overshoot vs the weighted grey mask "
          "{4:.4e}".format(w['mask_true_hard'], w['mask_true_grey'],
                           w['mask_delta_px'],
                           w['integrated_power_rel_move'],
                           w['boolean_overshoot_vs_weighted']))


if __name__ == '__main__':
    main(sys.argv[1])
