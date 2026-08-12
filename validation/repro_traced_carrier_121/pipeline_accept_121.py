# ACCEPTANCE for validation/pipeline on design 121, scored against the
# PRINTED NUMBERS of PROBE_SUM_AT_APERTURE_2026_08_11.
#
# WHAT IS BEING CLAIMED.  Not "the staged pipeline is plausible" but "the
# staged pipeline REPRODUCES the probe's arm-B numbers to their printed
# digits".  So the comparison is literal: every quantity is formatted with the
# probe's own format and the STRINGS are compared.  A tolerance would let a
# 4th-decimal drift pass as agreement, and the whole point of the probe's
# section 5/6 tables is that they are exact.
#
# HOW THE ARMS MAP ONTO PIPELINE RUNS.  Four runs, ONE workdir, so the
# ``chains`` stage runs once (three archived aperture fields) and the
# aggregate/leg/readout artifacts are keyed per include-set:
#
#   single_<order>   aggregate.include = [that order],  readout.frames = ALL
#                    -> the DIAGONAL is the probe's S5 null control and the
#                       OFF-DIAGONAL is its S6.4 crosstalk census, from one
#                       run.  The shipped per-order path cannot express the
#                       off-diagonal at all: its value is zero by
#                       construction, not by physics.
#   sum_all          aggregate.include = ALL           -> the probe's S6.1
#                       3-order sum on the like-for-like ``crop`` leg.
#
# PISTON COLUMNS ARE RECORDED AND FLAGGED, NOT SCORED, on a library without
# FIX_TILT_QUADRATIC_OPL_2026_08_11 -- see the pipeline's admissibility
# banner and BUILD_PIPELINE_2026_08_11 S1.
#
#   python pipeline_accept_121.py [--workdir DIR] [--cache DIR] [--leg crop]
#
# cp1252-safe ASCII only.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (_ROOT, os.path.join(_ROOT, 'validation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

from pipeline import PipelineSpec  # noqa: E402
from pipeline import driver as PD

ORDERS = ((0, 0), (-2, 0), (-4, -2))
TAGS = {(0, 0): 'p0_p0', (-2, 0): 'm2_p0', (-4, -2): 'm4_m2'}

# ---------------------------------------------------------------------------
# THE PROBE'S OWN PRINTED NUMBERS.  Sources named per block.
# ---------------------------------------------------------------------------
# PROBE_SUM_AT_APERTURE S5 (null control, crop leg), at the precision
# BUILD_CARRIER_FIELD S3 re-printed them to.
NULL = {
    'p0_p0': dict(rel_l2=2.7785e-05, piston=+7.287e-09, ph_rms=1.84e-06,
                  fwhm=3.400e-6, ee3=90.7407, ee6=None, power=1.8727654e-09,
                  power_ratio=0.9999999),
    'm2_p0': dict(rel_l2=1.4026e-04, piston=-7.716e-08, ph_rms=2.33e-05,
                  fwhm=3.400e-6, ee3=90.6343, ee6=None, power=1.8735537e-09,
                  power_ratio=0.9999997),
    'm4_m2': dict(rel_l2=9.3424e-05, piston=-1.741e-08, ph_rms=5.31e-06,
                  fwhm=3.800e-6, ee3=90.0711, ee6=None, power=1.8794228e-09,
                  power_ratio=1.0000001),
}
# PROBE_SUM_AT_APERTURE S6.1 (3-order sum, crop leg).
SUM3 = {
    'p0_p0': dict(fwhm=3.400e-6, ee3=90.740, ee6=99.897, power_ratio=0.9999978,
                  rel_l2=1.796e-03, piston=+2e-07, dee3=-0.001, dee6=0.000),
    'm2_p0': dict(fwhm=3.400e-6, ee3=90.634, ee6=99.861, power_ratio=1.0000001,
                  rel_l2=1.502e-03, piston=+2e-06, dee3=0.000, dee6=0.000),
    'm4_m2': dict(fwhm=3.800e-6, ee3=90.071, ee6=99.851, power_ratio=1.0000001,
                  rel_l2=9.382e-05, piston=-3e-07, dee3=0.000, dee6=0.000),
}
# PROBE_SUM_AT_APERTURE S6.4 (crosstalk, crop leg): ratio of the power frame i
# receives when ONLY order j is summed, to frame i's own diagonal.
XTALK = {
    'p0_p0': {'p0_p0': 1.000, 'm2_p0': 3.210e-06, 'm4_m2': 4.221e-11},
    'm2_p0': {'p0_p0': 2.410e-06, 'm2_p0': 1.000, 'm4_m2': 4.563e-11},
    'm4_m2': {'p0_p0': 3.459e-11, 'm2_p0': 1.053e-11, 'm4_m2': 1.000},
}
# The probe's own bars (S9), for the columns that are scored on a bar rather
# than on a printed digit.
BAR_ENERGY = 4e-05
BAR_EE_POINTS = 0.1


def printed(v, fmt):
    return 'nan' if v is None or not np.isfinite(v) else format(float(v), fmt)


def same_printed(got, want, fmt):
    return printed(got, fmt) == printed(want, fmt)


def load_spec(workdir, cache, leg_mode):
    p = os.path.join(_ROOT, 'validation', 'pipeline', 'specs',
                     'd121_3order_probe.json')
    with open(p, 'r', encoding='cp1252') as fh:
        d = json.load(fh)
    d['workdir'] = os.path.abspath(workdir)
    if cache:
        d['decompose']['params']['cache_dir'] = cache
    d['leg'] = {**d['leg'], 'mode': leg_mode}
    return d


def run_arm(base, include, quiet=True):
    d = json.loads(json.dumps(base))
    d['aggregate'] = {**d['aggregate'], 'include': include}
    spec = PipelineSpec.from_dict(d)
    st = PD.run(spec, log=(lambda *a, **k: None) if quiet else print)
    return st.report['readout']['rows']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=os.path.join(
        _ROOT, 'validation', 'pipeline', '_work', 'd121_3order_probe'))
    ap.add_argument('--cache', default=None)
    ap.add_argument('--leg', default='crop', choices=('crop', 'full'))
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--json', default=None)
    a = ap.parse_args(argv)
    base = load_spec(a.workdir, a.cache, a.leg)
    keys = [TAGS[o] for o in ORDERS]

    print("=" * 78)
    print("PIPELINE ACCEPTANCE -- design 121, 3 orders, scored against")
    print("PROBE_SUM_AT_APERTURE_2026_08_11 (leg mode %r)" % a.leg)
    print("=" * 78)
    t0 = time.perf_counter()
    singles = {}
    for k in keys:
        print(f"\n-- arm: sum ONLY {k}, read ALL frames "
              f"(null control + crosstalk row) --", flush=True)
        singles[k] = run_arm(base, [k], quiet=not a.verbose)
    print("\n-- arm: sum ALL 3 orders --", flush=True)
    allrows = run_arm(base, 'all', quiet=not a.verbose)
    wall = time.perf_counter() - t0

    fails = []
    flagged = []

    # ---- (1) NULL CONTROL -------------------------------------------------
    print("\n" + "=" * 78)
    print("(1) NULL CONTROL -- one order summed, read on its own frame")
    print("    vs PROBE_SUM_AT_APERTURE S5 / BUILD_CARRIER_FIELD S3")
    print("=" * 78)
    hdr = ("  %-8s %-9s %11s %11s %9s %8s %9s %14s %11s"
           % ('order', 'source', 'relL2', 'piston', 'ph_rms', 'FWHM/um',
              'EE3 %', 'window power', 'P/P_A'))
    print(hdr)
    for k in keys:
        r = singles[k][k]
        p = NULL[k]
        got = dict(rel_l2=r['vs_reference']['rel_l2'],
                   piston=r['vs_reference']['piston'],
                   ph_rms=r['vs_reference']['phase_rms_core'],
                   fwhm=r['spot']['fwhm'], ee3=r['spot']['ee3'] * 100.0,
                   power=r['spot']['power'], power_ratio=r['power_ratio'])
        print("  %-8s %-9s %11s %11s %9s %8s %9s %14s %11s"
              % (k, 'PIPELINE', printed(got['rel_l2'], '.4e'),
                 printed(got['piston'], '+.3e'), printed(got['ph_rms'], '.2e'),
                 printed(got['fwhm'] * 1e6, '.3f'), printed(got['ee3'], '.4f'),
                 printed(got['power'], '.7e'),
                 printed(got['power_ratio'], '.7f')))
        print("  %-8s %-9s %11s %11s %9s %8s %9s %14s %11s"
              % ('', 'probe', printed(p['rel_l2'], '.4e'),
                 printed(p['piston'], '+.3e'), printed(p['ph_rms'], '.2e'),
                 printed(p['fwhm'] * 1e6, '.3f'), printed(p['ee3'], '.4f'),
                 printed(p['power'], '.7e'),
                 printed(p['power_ratio'], '.7f')))
        for nm, fmt in (('rel_l2', '.4e'), ('ph_rms', '.2e'),
                        ('fwhm', '.3e'), ('ee3', '.4f'), ('power', '.7e'),
                        ('power_ratio', '.7f')):
            if not same_printed(got[nm], p[nm], fmt):
                fails.append(f"null {k}.{nm}: {printed(got[nm], fmt)} != "
                             f"{printed(p[nm], fmt)}")
        if not same_printed(got['piston'], p['piston'], '+.3e'):
            flagged.append(f"null {k}.piston: {printed(got['piston'], '+.3e')} "
                           f"vs probe {printed(p['piston'], '+.3e')}")

    # ---- (2) THE 3-ORDER SUM ---------------------------------------------
    print("\n" + "=" * 78)
    print("(2) 3-ORDER SUM -- vs PROBE_SUM_AT_APERTURE S6.1")
    print("=" * 78)
    print("  %-8s %-9s %8s %9s %9s %11s %11s %11s"
          % ('frame', 'source', 'FWHM/um', 'EE3 %', 'EE6 %', 'P/P_A',
             'relL2', 'piston'))
    for k in keys:
        r = allrows[k]
        p = SUM3[k]
        got = dict(fwhm=r['spot']['fwhm'], ee3=r['spot']['ee3'] * 100.0,
                   ee6=r['spot']['ee6'] * 100.0,
                   power_ratio=r['power_ratio'],
                   rel_l2=r['vs_reference']['rel_l2'],
                   piston=r['vs_reference']['piston'])
        armA_ee3 = r['reference_spot']['ee3'] * 100.0
        armA_ee6 = r['reference_spot']['ee6'] * 100.0
        print("  %-8s %-9s %8s %9s %9s %11s %11s %11s"
              % (k, 'PIPELINE', printed(got['fwhm'] * 1e6, '.3f'),
                 printed(got['ee3'], '.3f'), printed(got['ee6'], '.3f'),
                 printed(got['power_ratio'], '.7f'),
                 printed(got['rel_l2'], '.3e'),
                 printed(got['piston'], '+.0e')))
        print("  %-8s %-9s %8s %9s %9s %11s %11s %11s"
              % ('', 'probe', printed(p['fwhm'] * 1e6, '.3f'),
                 printed(p['ee3'], '.3f'), printed(p['ee6'], '.3f'),
                 printed(p['power_ratio'], '.7f'),
                 printed(p['rel_l2'], '.3e'), printed(p['piston'], '+.0e')))
        for nm, fmt in (('fwhm', '.3e'), ('ee3', '.3f'), ('ee6', '.3f'),
                        ('power_ratio', '.7f'), ('rel_l2', '.3e')):
            if not same_printed(got[nm], p[nm], fmt):
                fails.append(f"sum3 {k}.{nm}: {printed(got[nm], fmt)} != "
                             f"{printed(p[nm], fmt)}")
        if not same_printed(got['piston'], p['piston'], '+.0e'):
            flagged.append(f"sum3 {k}.piston: {printed(got['piston'], '+.0e')} "
                           f"vs probe {printed(p['piston'], '+.0e')}")
        # and the probe's own BARS, independently of the printed digits
        d_e = abs(got['power_ratio'] - 1.0)
        d_ee3 = abs(got['ee3'] - armA_ee3)
        d_ee6 = abs(got['ee6'] - armA_ee6)
        if d_e > BAR_ENERGY:
            fails.append(f"sum3 {k}: energy delta {d_e:.2e} > "
                         f"{BAR_ENERGY:.0e}")
        if max(d_ee3, d_ee6) > BAR_EE_POINTS:
            fails.append(f"sum3 {k}: EE delta {max(d_ee3, d_ee6):.3f} pt > "
                         f"{BAR_EE_POINTS} pt")
        if not same_printed(got['fwhm'], r['reference_spot']['fwhm'], '.3e'):
            fails.append(f"sum3 {k}: FWHM differs from the shipped path")

    # ---- (3) CROSSTALK ----------------------------------------------------
    print("\n" + "=" * 78)
    print("(3) CROSSTALK -- power in frame i with ONLY order j summed,")
    print("    over frame i's own diagonal.  vs PROBE_SUM_AT_APERTURE S6.4")
    print("=" * 78)
    print("  %-10s %-9s %12s %12s %12s"
          % ('frame i', 'source', *keys))
    for i in keys:
        diag = singles[i][i]['spot']['power']
        got = {j: singles[j][i]['spot']['power'] / diag for j in keys}
        print("  %-10s %-9s %12s %12s %12s"
              % (i, 'PIPELINE', *[printed(got[j], '.3e') for j in keys]))
        print("  %-10s %-9s %12s %12s %12s"
              % ('', 'probe', *[printed(XTALK[i][j], '.3e') for j in keys]))
        for j in keys:
            if not same_printed(got[j], XTALK[i][j], '.3e'):
                fails.append(f"xtalk {i}<-{j}: {printed(got[j], '.3e')} != "
                             f"{printed(XTALK[i][j], '.3e')}")

    # ---- verdict ----------------------------------------------------------
    adm = PD.piston_fix_status()[0]
    print("\n" + "=" * 78)
    print(f"wall {wall:.1f}s   piston fix: {adm.upper()}")
    if flagged:
        print("\nPISTON COLUMNS -- RECORDED, NOT SCORED "
              "(FIX_TILT_QUADRATIC_OPL is %s):" % adm)
        for f in flagged:
            print("   " + f)
        if adm == 'present':
            print("   ...but the fix IS present, so these are real "
                  "disagreements and should be investigated.")
    ok = not fails and not (flagged and adm == 'present')
    print("\nVERDICT: " + ("PASS -- every scored column reproduces the "
                           "probe's printed digits"
                           if ok else "FAIL"))
    for f in fails:
        print("   FAIL " + f)
    if a.json:
        with open(a.json, 'w', encoding='cp1252') as fh:
            json.dump({'singles': singles, 'sum3': allrows, 'wall': wall,
                       'fails': fails, 'flagged': flagged,
                       'piston_fix': adm, 'leg': a.leg}, fh, indent=1,
                      default=float)
        print(f"wrote {a.json}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
