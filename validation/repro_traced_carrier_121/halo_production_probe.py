# DOES THE v5.32 HALO SELF-CHECK FIRE IN THE PRODUCTION CONFIGURATION?
#
# It fires on design 121's on-axis last group in the DIAGNOSTIC configuration
# (final_distance=0, final_leg='paraxial') -- ``fitguard_branch_map.py`` reads
# it 1 with the fit guard off and 0 with it on.  The PRODUCTION run
# (energy_ee_vs_conservation_121.py, final_leg='exact' + the trailing leg +
# the Bluestein readout) reported ZERO halo warnings on a field whose halo at
# 4 mm is 8.32e-01 of peak.  One of those two statements is wrong and this
# settles which.
#
# It captures every element call of ONE production run, records the halo
# warning per call, and then re-scores the last call script-side at the same
# radius the library would have used -- so a disagreement between "the library
# did not warn" and "the field has a lobe" is attributed to the RADIUS rather
# than assumed to be a plumbing failure.
#
# usage:  LUMEN_PIN=0 ORDERS='0,0' python halo_production_probe.py
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('LUMEN_PIN', '0')

import approx_common as A                                       # noqa: E402,I001
import lumenairy.elements as _EL                                # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402

_RX = re.compile(
    r"amax_halo = ([0-9.e+-]+) of peak beyond ([0-9.e+-]+) mm "
    r"\(([0-9.]+) x the exact-ray exit support radius ([0-9.e+-]+) mm "
    r"about the traced exit centroid \(([0-9.e+-]+), ([0-9.e+-]+)\) mm\)")


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0').split()]
    old_tol = LT._RD_HALO_AMAX_TOL
    old_fac = LT._RD_HALO_RADIUS_FACTOR
    LT._RD_HALO_AMAX_TOL = -1.0        # make every call print its reading
    # ... and shrink the radius so the annulus can never be EMPTY, which is
    # the other way a call produces no reading at all.  The factor sweep is
    # recomputed script-side from the reported hull, so this changes nothing
    # that is reported.
    LT._RD_HALO_RADIUS_FACTOR = float(os.environ.get('FACTOR', '0.05'))
    calls = []
    orig = _EL.apply_real_lens_traced

    def _w(E_in, *a, **kw):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            out = orig(E_in, *a, **kw)
        rec = {'dx': float(kw.get('dx', np.nan)),
               'E_out': np.array(out, copy=True), 'hull': None}
        for w in wl:
            m = _RX.search(str(w.message))
            if m is not None:
                rec['hull'] = (float(m.group(4)) * 1e-3,
                               float(m.group(5)) * 1e-3,
                               float(m.group(6)) * 1e-3)
        calls.append(rec)
        return out

    _EL.apply_real_lens_traced = _w
    try:
        cfgs = [c.strip() for c in
                os.environ.get('CFGS', 'ship').split(',') if c.strip()]
        for order in orders:
          for cfg in cfgs:
            post, env, R, dx, P_in, L, M, centre = A.setup(order)
            calls.clear()
            _g_old = LT.REMAP_STATIONARY_PHASE_FIT_GUARD
            LT.REMAP_STATIONARY_PHASE_FIT_GUARD = (cfg == 'shipG')
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    A.run_chain(post, env, R, dx, L, M, centre)
            finally:
                LT.REMAP_STATIONARY_PHASE_FIT_GUARD = _g_old
            print(f"order {order} config {cfg}: {len(calls)} element calls")
            for k, rec in enumerate(calls):
                if rec['hull'] is None:
                    print(f"  grp{k}: NO halo reading (the check did not run "
                          f"or found no pixel outside the support)")
                    continue
                rh, cx, cy = rec['hull']
                F = rec['E_out']
                n = F.shape[0]
                ax = (np.arange(n) - n / 2) * rec['dx']
                r2 = (ax[None, :] - cx) ** 2 + (ax[:, None] - cy) ** 2
                aF = np.abs(F)
                pk = float(aF.max())
                cells = []
                for f in (1.00, 1.25, 1.50, 2.00, 3.00):
                    m = r2 > (f * rh) ** 2
                    cells.append(f"x{f:.2f}={((float(aF[m].max()) / pk) if m.any() and pk > 0 else float('nan')):.3e}")
                # and the audit's fixed 4 mm radius about the same centroid
                m4 = r2 > 4.0e-3 ** 2
                a4 = (float(aF[m4].max()) / pk) if m4.any() and pk > 0 else 0.0
                print(f"  grp{k}: N={n} dx={rec['dx'] * 1e6:7.3f}um "
                      f"halfgrid={(n / 2) * rec['dx'] * 1e3:7.3f}mm  "
                      f"r_hull={rh * 1e3:7.4f} mm  centroid "
                      f"({cx * 1e3:+.4f},{cy * 1e3:+.4f})  " + ' '.join(cells)
                      + f"  amax@4mm={a4:.3e}", flush=True)
    finally:
        _EL.apply_real_lens_traced = orig
        LT._RD_HALO_AMAX_TOL = old_tol
        LT._RD_HALO_RADIUS_FACTOR = old_fac
    return 0


if __name__ == '__main__':
    sys.exit(main())
