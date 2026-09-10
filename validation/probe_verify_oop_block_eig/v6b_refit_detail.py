"""V6b -- the detail checks on the reference study that V6 raised.

1. The study's LIVE-observable counts are 32 / 28 / 36 / 40; the rule it states
   ("exactly zero in every engine") gives 40 / 36 / 44 / 48 on the same data.
   The difference is exactly the EIGHT Jones components, so this re-runs the
   whole analysis on the no-Jones set to reproduce the study's own counts.
2. C4's "worst observable" column: the study reports one row
   (corner/normal, hybrid-li) as ``3.46e-04 -> 5.34e-04`` -- a RISE -- while
   its summary sentence says "the worst-case distance falling by 1.3x to
   5.4x".  Both selections (largest FIRST distance, largest LAST distance) and
   the full ratio envelope are computed here.
3. Spot-checks of the study's C1 / S2 tables straight out of the committed
   ladder JSON.

Usage: python v6b_refit_detail.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from v6_refit import (  # noqa: E402
    ENGINES,
    P_EDGE_HI,
    P_EDGE_LO,
    P_HI,
    P_LO,
    PAIRS,
    fit_A,
    fit_B,
    fit_power,
    live_keys,
    load_cases,
    series,
)


def keys_no_jones(case):
    return [k for k in live_keys(case) if not k.startswith("J")]


def counts(fitter, keyfn, label):
    print()
    print("#" * 96)
    print(f"# {label}")
    print("#" * 96)
    tot_s = {p: [0, 0] for p in PAIRS}
    tot_a = {p: [0, 0] for p in PAIRS}
    out = []
    for case in load_cases():
        tag = f"{case['fixture']}/{case['mount']}"
        keys = keyfn(case)
        fits = {e: {k: fitter(*series(case, e, k)) for k in keys}
                for e in ENGINES}
        print(f"\n=== {tag}   observables: {len(keys)}")
        row = dict(case=tag, n=len(keys), sound={}, pairs={})
        for e in ENGINES:
            sd = [f for f in fits[e].values() if f["sound"]]
            sg = np.array([f["sigma"] for f in sd]) if sd else np.array([np.nan])
            row["sound"][e] = len(sd)
            print(f"    {e:16s} sound {len(sd):3d}/{len(keys):<3d}  "
                  f"sigma {np.min(sg):.2e} .. {np.max(sg):.2e}  "
                  f"median {np.median(sg):.2e}")
        for A, B in PAIRS:
            ns = nw = na = nwa = 0
            for k in keys:
                fa, fb = fits[A][k], fits[B][k]
                d = abs(fa["f_inf"] - fb["f_inf"])
                s = np.hypot(fa["sigma"], fb["sigma"])
                ok = bool(d <= s) if np.isfinite(s) else False
                na += 1
                nwa += ok
                if fa["sound"] and fb["sound"]:
                    ns += 1
                    nw += ok
            row["pairs"][f"{A}|{B}"] = dict(sound=[nw, ns], all=[nwa, na])
            tot_s[(A, B)][0] += nw
            tot_s[(A, B)][1] += ns
            tot_a[(A, B)][0] += nwa
            tot_a[(A, B)][1] += na
            print(f"    {A + ' vs ' + B:32s} both-sound {ns:3d}/{len(keys):<3d}"
                  f"  within {nw:3d}/{ns:<3d}   ALL-live within {nwa:3d}/{na}")
        out.append(row)
    print()
    print(f"{'pair':34s} {'sound-only':>16s} {'unsound INCLUDED':>18s}")
    tot = {}
    for p in PAIRS:
        w, n = tot_s[p]
        wa, na = tot_a[p]
        tot[f"{p[0]}|{p[1]}"] = dict(sound=[w, n], all=[wa, na])
        print(f"{p[0] + ' vs ' + p[1]:34s} {w:5d}/{n:<5d} "
              f"({100.0 * w / n if n else float('nan'):5.1f}%) "
              f"{wa:6d}/{na:<5d} ({100.0 * wa / na:5.1f}%)")
    return dict(cases=out, totals=tot)


def c4_detail():
    print()
    print("#" * 96)
    print("# C4 detail -- 'the Fourier ladders move TOWARD the staggered "
          "limit'")
    print("#" * 96)
    rows = []
    for case in load_cases():
        tag = f"{case['fixture']}/{case['mount']}"
        keys = keys_no_jones(case)
        stag = {k: fit_A(*series(case, "staggered", k))["f_inf"] for k in keys}
        print(f"\n=== {tag}   observables (no Jones): {len(keys)}")
        print(f"    {'engine':16s} {'closer':>9s} {'monotone':>9s} "
              f"{'worst-FIRST obs':>18s} {'d0 -> d1':>24s} "
              f"{'worst-LAST obs':>17s} {'d0 -> d1':>24s}")
        for e in ENGINES[1:]:
            closer = mono = 0
            bf = (-1.0, None, None)
            bl = (-1.0, None, None)
            ratios = []
            for k in keys:
                _x, y = series(case, e, k)
                d0, d1 = abs(y[0] - stag[k]), abs(y[-1] - stag[k])
                closer += int(d1 < d0)
                st = np.abs(np.diff(y))
                mono += int(np.all(np.diff(st) < 0))
                if d0 > bf[0]:
                    bf = (d0, d1, k)
                if d1 > bl[0]:
                    bl = (d1, d0, k)
                if d1 > 0:
                    ratios.append(d0 / d1)
            rows.append(dict(case=tag, engine=e, closer=closer, n=len(keys),
                             monotone=mono, worst_first=bf[:2],
                             worst_first_key=bf[2],
                             worst_last=[bl[1], bl[0]], worst_last_key=bl[2],
                             ratio_min=float(min(ratios)),
                             ratio_max=float(max(ratios))))
            print(f"    {e:16s} {closer:4d}/{len(keys):<4d} {mono:4d}/"
                  f"{len(keys):<4d} {bf[2]:>18s} {bf[0]:10.2e} -> "
                  f"{bf[1]:10.2e} {bl[2]:>17s} {bl[1]:10.2e} -> {bl[0]:10.2e}")
        print(f"    ratio d(first)/d(last) over all observables: "
              f"{min(r['ratio_min'] for r in rows if r['case'] == tag):.2f}x"
              f" .. {max(r['ratio_max'] for r in rows if r['case'] == tag):.2f}x"
              f"   (a ratio < 1 means the ladder moved AWAY)")
    pct = [100.0 * r["closer"] / r["n"] for r in rows]
    print()
    print(f"'last rung closer than first' over the 12 (case, engine) rows: "
          f"{min(pct):.0f}% .. {max(pct):.0f}%   "
          f"(study claims 86 - 100%)")
    wf = [r["worst_first"][0] / r["worst_first"][1] for r in rows
          if r["worst_first"][1] > 0]
    wl = [r["worst_last"][0] / r["worst_last"][1] for r in rows
          if r["worst_last"][1] > 0]
    print(f"worst-FIRST-distance rows fall by {min(wf):.2f}x .. {max(wf):.2f}x"
          f"   (study claims 'worst-case distance falling by 1.3x to 5.4x')")
    print(f"worst-LAST-distance rows change by {min(wl):.2f}x .. {max(wl):.2f}x"
          f"   (< 1 = it got WORSE)")
    return rows


def c1_spotchecks():
    print()
    print("#" * 96)
    print("# C1 / S2 spot-checks straight out of the committed ladder JSON")
    print("#" * 96)
    cases = {f"{c['fixture']}/{c['mount']}": c for c in load_cases()}
    for tag in cases:
        c = cases[tag]
        _x, y = series(c, "staggered", "sumR0")
        st = np.abs(np.diff(y))
        print(f"\n{tag}  staggered sumR (incident Ex), M = 5..10")
        print("   " + " ".join(f"{v:.9f}" for v in y))
        print("   steps: " + " ".join(f"{v:.2e}" for v in st))
        Rk = [k for k in live_keys(c) if k.startswith("R(")]
        top = max(abs(series(c, "staggered", k)[1][-1]
                      - series(c, "staggered", k)[1][-2]) for k in Rk)
        print(f"   per-order dR, top step: {top:.2e}")
    print()
    print("S2 example table: corner/normal, T(-1, 1)p0")
    c = cases["corner/normal"]
    for e in ENGINES:
        x, y = series(c, e, "T(-1, 1)p0")
        f = fit_A(x, y)
        print(f"   {e:16s} " + " ".join(f"{v:.7f}" for v in y)
              + f"   -> f_inf {f['f_inf']:.7f} +/- {f['sigma']:.1e}, "
                f"p = {f['p']:.1f} {'' if f['sound'] else '(UNSOUND)'}")
    print()
    print("C4 note: hybrid-laurent sum R steps on chiral/conical")
    c = cases["chiral/conical25_40"]
    _x, y = series(c, "hybrid-laurent", "sumR0")
    print("   values " + " ".join(f"{v:.7f}" for v in y))
    print("   steps  " + " ".join(f"{v:.2e}" for v in np.abs(np.diff(y))))
    print()
    print("Section 5 test claim: corner/normal per-order T steps, M = 4..7 "
          "-- NOT in the JSON (the ladder starts at M = 5); re-measured in "
          "v7_tests.py")


def scan_resolution():
    """The study states 400 scan points on [0.25, 10].  The count is not
    cosmetic: one sound-fit count moves by one."""
    print()
    print("#" * 96)
    print("# p-SCAN RESOLUTION -- staggered sound-fit counts vs scan points")
    print("#" * 96)
    cases = load_cases()
    tags = [f"{c['fixture']}/{c['mount']}" for c in cases]
    print(f"    {'points':>7s}  " + "  ".join(f"{x:>22s}" for x in tags))
    rows = []
    for npts in (400, 1201, 4001):
        ps = np.linspace(P_LO, P_HI, npts)
        line = []
        for case in cases:
            keys = keys_no_jones(case)
            n = 0
            for k in keys:
                x, y = series(case, "staggered", k)
                _ss, p, _f, _c, _se = fit_power(x, y, ps)
                if P_EDGE_LO < p < P_EDGE_HI:
                    n += 1
            line.append(f"{n}/{len(keys)}")
        rows.append(dict(points=npts, staggered=line))
        print(f"    {npts:7d}  " + "  ".join(f"{v:>22s}" for v in line))
    print("    (the study's C2 staggered row reads 21/32, 12/28, 10/36, 8/40)")
    return rows


if __name__ == "__main__":
    a_nj = counts(fit_A, keys_no_jones,
                  "fit A (power law, scanned p) -- NO-JONES set "
                  "(reproduces the study's 32/28/36/40)")
    b_nj = counts(fit_B, keys_no_jones,
                  "fit B (pure Aitken/Shanks) -- NO-JONES set")
    c4 = c4_detail()
    sr = scan_resolution()
    c1_spotchecks()
    with open(os.path.join(HERE, "results", "v6b_refit_detail.json"), "w") as fh:
        json.dump(dict(fitA_nojones=a_nj, fitB_nojones=b_nj, c4=c4,
                       scan_resolution=sr), fh, indent=1)
