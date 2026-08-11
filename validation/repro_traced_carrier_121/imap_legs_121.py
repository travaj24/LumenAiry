# INVERSE-MAP production build, S5 CLOSURE: WHERE does the banner move come
# from?
#
# LOCAL-ONLY.  No library edit.  The instrument is a pass-through wrapper on
# ``lumenairy.elements.apply_real_lens_traced`` (which is what
# ``carrier.py`` imports, locally, at BOTH of its traced call sites -- the
# ordinary chain leg at :6354 and the fine retrace at :7670), so every element
# call the real banner run makes is intercepted at its OWN configuration.
#
# THE QUESTION.  ``BUILD_INVERSE_MAP_2026_08_11`` S4.6 left a 26x gap: the
# per-group field census puts the two arms 7.67e-03 waves rms apart on ONE
# group -- 0.23 % of Strehl -- while the shipping banner moves 6.5 % in peak
# intensity.  Something accumulates over the chain's traced legs that a single
# group does not show.
#
# THE HYPOTHESIS UNDER TEST, from the build doc's own S5: at the COARSE legs
# ``ray_subsample = 4``, so the cubic upsample the model exists to remove
# carries ~``(4/87)^4`` = 1.8e-06 of its fine-retrace error -- there is nothing
# to remove and the model's own least-squares residual is a NET COST.  If that
# is right, the per-leg table shows the coarse legs with a LARGE map fit
# residual and a large field delta, and the fine retrace with a small one.
#
# THE METHOD.  For every element call the chain makes, run it TWICE -- once
# with ``TRACED_INVERSE_MAP = False`` and once with ``True``, on the SAME
# inputs -- record the delta, and RETURN THE OFF ARM.  Returning the off arm
# is what makes the table a DECOMPOSITION rather than a second banner: each
# leg's number is that leg's own contribution given the shipped input, with no
# compounding from the legs before it.  The compounded answer is already known
# (the banner itself).
#
# Run:  python imap_legs_121.py            (~20 min, ~22 GB; one banner run,
#                                           every traced element doubled)
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402,F401  (path + glass registration)

import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_imap as IM  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_imap_legs.json')

LEGS = []
_ORIG = EL.apply_real_lens_traced


def _both_arms(E_in, **kw):
    """Run the shipped element twice and record the difference."""
    rec = {'n_call': len(LEGS)}
    for k in ('dx', 'ray_subsample', 'amplitude_model',
              'preserve_input_phase', 'remap_sampling', 'inversion_method'):
        if k in kw:
            v = kw[k]
            rec[k] = float(v) if isinstance(v, float) else v
    rec['N'] = int(np.shape(E_in)[0])
    out = {}
    for tag, flag in (('off', False), ('on', True)):
        old = IM.TRACED_INVERSE_MAP
        IM.inverse_map_cache_clear()
        g = {}
        try:
            IM.TRACED_INVERSE_MAP = flag
            t0 = time.perf_counter()
            out[tag] = _ORIG(E_in, _imap_out=g, **kw)
            rec['%s_seconds' % tag] = time.perf_counter() - t0
        finally:
            IM.TRACED_INVERSE_MAP = old
        if flag:
            rec['guards'] = {k: v for k, v in g.items()
                             if isinstance(v, (int, float, str, bool))}
    Eo, En = out['off'], out['on']
    pk = float(np.abs(Eo).max())
    d = np.abs(En - Eo)
    p_off = float((np.abs(Eo) ** 2).sum())
    p_on = float((np.abs(En) ** 2).sum())
    rec['peak_off'] = pk
    rec['max_abs_dE_over_peak'] = float(d.max()) / max(pk, 1e-300)
    rec['power_ratio_on_over_off'] = p_on / max(p_off, 1e-300)
    core = (np.abs(Eo) >= np.exp(-2.0) * pk) & (np.abs(En) > 0)
    if core.any():
        ph = np.angle(En[core] / Eo[core]) / (2.0 * np.pi)
        rec['core_phase_max_waves'] = float(np.abs(ph).max())
        rec['core_phase_rms_waves'] = float(np.sqrt((ph ** 2).mean()))
        rec['core_amp_rel_rms'] = float(np.sqrt(np.mean(
            ((np.abs(En[core]) - np.abs(Eo[core])) / pk) ** 2)))
        rec['n_core'] = int(core.sum())
    LEGS.append(rec)
    g = rec.get('guards', {})
    print("  LEG %d  N=%-6d dx=%8.4f um  sub=%-3s  n_launch=%-5s  "
          "%s%s" % (rec['n_call'], rec['N'], rec.get('dx', float('nan')) * 1e6,
                    rec.get('ray_subsample'), g.get('n_launch', '--'),
                    'ENGAGED' if g.get('engaged') else 'refused ',
                    (' [%s]' % g['refused']) if g.get('refused') else ''),
          flush=True)
    if g.get('engaged'):
        print("        map resid OPL %.4e w (rms %.4e)   parity vs incumbent "
              "%.4gx (rms %.4gx)"
              % (g.get('fit_resid_opl_waves', float('nan')),
                 g.get('fit_resid_opl_rms_waves', float('nan')),
                 g.get('parity_ratio_opl', float('nan')),
                 g.get('parity_ratio_opl_rms', float('nan'))), flush=True)
    print("        FIELD max|dE|/pk %.4e   core phase rms %.4e w   "
          "power ratio %.9f"
          % (rec['max_abs_dE_over_peak'],
             rec.get('core_phase_rms_waves', float('nan')),
             rec['power_ratio_on_over_off']), flush=True)
    del En, d, core
    return Eo                      # the chain proceeds on the SHIPPED arm


def coarse_only():
    """The SIX COARSE LEGS alone, without the 8192-square fine retrace.

    Same instrument, same decomposition, ~2-3 min instead of ~25: the chain is
    stopped at the last group's exit vertex (``final_distance=0.0``, no focus
    readout), which is exactly the path ``imap_afit_121.py`` used to measure
    ``grad a_fit`` per group.  This is the arm that tests the hypothesis --
    every one of these legs runs at ``ray_subsample = 4``, the regime the model
    was NOT sized for -- and it does not need the retrace to do it.
    """
    import lumenairy as la
    import lumenairy.propagators.carrier as CAR
    _pre, groups_post, _gap, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a()
    m, n = (int(os.environ.get('HM_M', '-4')), int(os.environ.get('HM_N', '-2')))
    car = la.TiltedCarrier(float(R_doe), m * C.LAM / period,
                           n * C.LAM / period)
    print('COARSE-LEG DRIVER: order %s, %d post-DOE groups, '
          'ray_subsample=4, final_distance=0' % ((m, n), len(groups_post)),
          flush=True)
    EL.apply_real_lens_traced = _both_arms
    try:
        CAR.propagate_traced_carrier_chain(
            env_doe, groups_post, C.LAM, dx_doe, r_in=car,
            ray_subsample=4, n_workers=1, final_distance=0.0)
    finally:
        EL.apply_real_lens_traced = _ORIG
    return LEGS


def main():
    import runpy

    import lumenairy as la
    if os.environ.get('LEGS_MODE') == 'coarse':
        coarse_only()
        _summary()
        return
    # The chain's Newton-pool worker count, overridden loudly for the same
    # reason ``capstone_stageB.py`` overrides it: ``focus_scan_121.py``
    # hard-codes 8, and this driver runs every traced element TWICE, so the
    # pool's per-worker commit lands twice as often on a box already carrying
    # other campaigns.  ``n_workers`` is a SPEED knob -- the pooled and serial
    # inversions are documented and tested to agree -- so this cannot move a
    # number in the table.
    _nw = int(os.environ.get('LEGS_NW', '1'))
    _orig_chain = la.propagate_traced_carrier_chain

    def _chain(*a, **kw):
        kw['n_workers'] = _nw
        return _orig_chain(*a, **kw)

    la.propagate_traced_carrier_chain = _chain
    print('LEG DRIVER: chain n_workers -> %d; every traced element runs '
          'BOTH arms and the chain proceeds on the SHIPPED one' % _nw,
          flush=True)
    EL.apply_real_lens_traced = _both_arms
    try:
        sys.argv = ['focus_scan_121.py']
        runpy.run_path(os.path.join(_HERE, 'focus_scan_121.py'),
                       run_name='__main__')
    finally:
        EL.apply_real_lens_traced = _ORIG
        la.propagate_traced_carrier_chain = _orig_chain
    _summary()


def _summary():
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump({'legs': LEGS}, fh, indent=1, sort_keys=True, default=float)
    print("\nPER-LEG SUMMARY (the chain ran on the SHIPPED arm throughout)")
    print("  %-4s %-7s %-10s %-5s %-9s %-12s %-12s %-12s"
          % ('leg', 'N', 'dx (um)', 'sub', 'engaged', 'map resid w',
             'core phase w', 'dpower'))
    for r in LEGS:
        g = r.get('guards', {})
        print("  %-4d %-7d %-10.4f %-5s %-9s %-12.4e %-12.4e %+.3e"
              % (r['n_call'], r['N'], r.get('dx', float('nan')) * 1e6,
                 r.get('ray_subsample'), bool(g.get('engaged')),
                 g.get('fit_resid_opl_waves', float('nan')),
                 r.get('core_phase_rms_waves', float('nan')),
                 r['power_ratio_on_over_off'] - 1.0))
    print("wrote %s" % os.path.basename(OUT))


if __name__ == '__main__':
    main()
