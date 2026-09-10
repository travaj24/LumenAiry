"""V6 -- an INDEPENDENT re-fit of the committed converged-reference ladders.

The ladder DATA (every rung, every observable, four engines, four cases) is
read from ``validation/probe_pmm2d_staggered_oop_reference/results/*.json``;
the EXTRAPOLATION is re-implemented here from the method description in
``docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_REFERENCE_2026_09_10.md`` S2,
independently of the study's own code:

  fit A (theirs, re-implemented):  f(x) = f_inf + C x^-p, p SCANNED, (f_inf, C)
      by linear least squares at each p; sigma = max of the LS standard error
      of f_inf, the shift when the first rung is dropped, and the distance to
      an Aitken delta^2 extrapolant of the last three rungs; a best p on the
      scan boundary is UNSOUND.
  fit B (mine, wholly different):  a pure AITKEN / Shanks transform of the
      last three rungs -- no model, no fitted rate -- with sigma taken as the
      spread of the two Aitken values available from the last four rungs.

Then, on BOTH extrapolants:
  * the sound-fit counts per engine and case (the study's C2 column);
  * the pairwise agreement counts (C3), with and WITHOUT the unsound-fit
    exclusion -- the cherry-picking check;
  * the "do the Fourier ladders move toward the staggered limit" counts (C4);
  * the top-of-ladder bound (C5), which uses no extrapolation at all.

Usage: python v6_refit.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(os.path.dirname(HERE), "probe_pmm2d_staggered_oop_reference",
                   "results")

import numpy as np  # noqa: E402

ENGINES = ["staggered", "hybrid-laurent", "hybrid-li", "rcwa"]
PAIRS = [("staggered", "hybrid-laurent"), ("staggered", "hybrid-li"),
         ("staggered", "rcwa"), ("hybrid-laurent", "hybrid-li"),
         ("hybrid-laurent", "rcwa"), ("hybrid-li", "rcwa")]

P_LO, P_HI = 0.25, 10.0
N_SCAN = 400          # the study's stated scan resolution
P_EDGE_LO, P_EDGE_HI = 0.30, 9.70


def load_cases():
    cases = []
    for fn in ("t1_corner.json", "t1_chiral.json"):
        d = json.load(open(os.path.join(REF, fn)))
        for c in d["cases"]:
            cases.append(c)
    order = {("corner", "normal"): 0, ("corner", "conical20_35"): 1,
             ("chiral", "conical25_40"): 2, ("chiral", "normal"): 3}
    cases.sort(key=lambda c: order[(c["fixture"], c["mount"])])
    return cases


def series(case, engine, key):
    lad = case["ladders"][engine]
    return np.asarray(lad["x"], float), np.asarray(
        [rung[key] for rung in lad["obs"]], float)


def live_keys(case):
    """An observable is LIVE unless it is EXACTLY zero on every rung of every
    engine (the study's rule for the evanescent orders)."""
    keys = list(case["ladders"]["staggered"]["obs"][0].keys())
    out = []
    for k in keys:
        dead = True
        for e in ENGINES:
            _x, y = series(case, e, k)
            if np.any(y != 0.0):
                dead = False
                break
        if not dead:
            out.append(k)
    return out


# --------------------------------------------------------------------------- #
# fit A -- power-law with a SCANNED rate  (their method, re-implemented)
# --------------------------------------------------------------------------- #
def aitken(y):
    """Aitken delta^2 on the last three entries of ``y``."""
    if y.size < 3:
        return np.nan
    a, b, c = y[-3], y[-2], y[-1]
    den = (c - b) - (b - a)
    if den == 0.0 or not np.isfinite(den):
        return np.nan
    return c - (c - b) ** 2 / den


def fit_power(x, y, ps=None):
    """Least squares in (f_inf, C) at each p on a scan; return the best."""
    if ps is None:
        # 400 points is what the study's S2 states; the count matters -- the
        # chiral/conical staggered sound-fit count reads 10/36 at 400 and
        # 11/36 at 1201 or 4001 (v6b_refit_detail.py section 'p-scan
        # resolution').
        ps = np.linspace(P_LO, P_HI, N_SCAN)
    n = x.size
    best = None
    for p in ps:
        A = np.column_stack([np.ones(n), x ** (-p)])
        coef, res, rank, _sv = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        ss = float(r @ r)
        if best is None or ss < best[0]:
            # standard error of the f_inf coefficient
            dof = max(n - 2, 1)
            s2 = ss / dof
            try:
                cov = np.linalg.inv(A.T @ A) * s2
                se = float(np.sqrt(max(cov[0, 0], 0.0)))
            except np.linalg.LinAlgError:
                se = np.inf
            best = (ss, float(p), float(coef[0]), float(coef[1]), se)
    return best


def fit_A(x, y):
    ss, p, f_inf, C, se = fit_power(x, y)
    # (ii) stability against dropping the FIRST rung
    if x.size >= 4:
        _ss2, _p2, f2, _C2, _se2 = fit_power(x[1:], y[1:])
        d_drop = abs(f_inf - f2)
    else:
        d_drop = np.inf
    # (iii) distance to an independent Aitken extrapolant
    ai = aitken(y)
    d_ait = abs(f_inf - ai) if np.isfinite(ai) else np.inf
    sigma = max(se, d_drop, d_ait)
    sound = (P_EDGE_LO < p < P_EDGE_HI) and np.isfinite(sigma)
    return dict(f_inf=f_inf, sigma=float(sigma), p=p, sound=bool(sound),
                se=se, d_drop=float(d_drop), d_ait=float(d_ait),
                last=float(y[-1]), last_step=float(abs(y[-1] - y[-2])))


# --------------------------------------------------------------------------- #
# fit B -- a pure Aitken/Shanks transform (no model, no fitted rate)
# --------------------------------------------------------------------------- #
def fit_B(x, y):
    a_last = aitken(y)
    a_prev = aitken(y[:-1]) if y.size >= 4 else np.nan
    if not np.isfinite(a_last):
        return dict(f_inf=float(y[-1]), sigma=np.inf, sound=False,
                    last=float(y[-1]))
    sig = abs(a_last - a_prev) if np.isfinite(a_prev) else np.inf
    # the Aitken value must not be further from the top rung than the ladder's
    # own last step by more than a decade -- a diverging transform is unsound
    step = abs(y[-1] - y[-2])
    sane = abs(a_last - y[-1]) <= 100.0 * max(step, 1e-18)
    return dict(f_inf=float(a_last), sigma=float(sig),
                sound=bool(np.isfinite(sig) and sane), last=float(y[-1]))


# --------------------------------------------------------------------------- #
def analyse(fitter, name):
    cases = load_cases()
    print()
    print("#" * 100)
    print(f"# EXTRAPOLANT: {name}")
    print("#" * 100)
    out = {"name": name, "cases": []}
    tot_pair_sound = {p: [0, 0] for p in PAIRS}
    tot_pair_all = {p: [0, 0] for p in PAIRS}
    for case in cases:
        tag = f"{case['fixture']}/{case['mount']}"
        keys = live_keys(case)
        fits = {e: {} for e in ENGINES}
        for e in ENGINES:
            for k in keys:
                x, y = series(case, e, k)
                fits[e][k] = fitter(x, y)
        print(f"\n=== {tag}   live observables: {len(keys)}")
        print(f"    {'engine':16s} {'sound':>7s} {'sigma min':>11s} "
              f"{'sigma max':>11s} {'sigma med':>11s} {'top step max':>13s}")
        crow = dict(case=tag, live=len(keys), engines={}, pairs={})
        for e in ENGINES:
            sd = [f for f in fits[e].values() if f["sound"]]
            sg = np.array([f["sigma"] for f in sd]) if sd else np.array([np.nan])
            steps = [abs(series(case, e, k)[1][-1] - series(case, e, k)[1][-2])
                     for k in keys]
            crow["engines"][e] = dict(
                sound=len(sd), live=len(keys),
                sigma_min=float(np.min(sg)), sigma_max=float(np.max(sg)),
                sigma_med=float(np.median(sg)), top_step=float(np.max(steps)))
            print(f"    {e:16s} {len(sd):3d}/{len(keys):<3d} "
                  f"{np.min(sg):11.2e} {np.max(sg):11.2e} "
                  f"{np.median(sg):11.2e} {np.max(steps):13.2e}")
        # ---- pairwise agreement, sound-only AND all-live
        print(f"    {'pair':30s} {'both sound':>11s} {'within':>8s} "
              f"{'ALL live within':>16s} {'worst z (sound)':>16s}")
        for A, B in PAIRS:
            ns = nw = 0
            na = nwa = 0
            worst_z, worst_k = -1.0, None
            for k in keys:
                fa, fb = fits[A][k], fits[B][k]
                d = abs(fa["f_inf"] - fb["f_inf"])
                s = np.hypot(fa["sigma"], fb["sigma"])
                ok = (d <= s) if np.isfinite(s) else False
                na += 1
                nwa += int(ok)
                if fa["sound"] and fb["sound"]:
                    ns += 1
                    nw += int(ok)
                    z = d / s if s > 0 else (0.0 if d == 0 else np.inf)
                    if np.isfinite(z) and z > worst_z:
                        worst_z, worst_k = z, k
            crow["pairs"][f"{A}|{B}"] = dict(both_sound=ns, within=nw,
                                             all_live=na, all_within=nwa,
                                             worst_z=worst_z, worst_key=worst_k)
            tot_pair_sound[(A, B)][0] += nw
            tot_pair_sound[(A, B)][1] += ns
            tot_pair_all[(A, B)][0] += nwa
            tot_pair_all[(A, B)][1] += na
            print(f"    {A + ' vs ' + B:30s} {ns:4d}/{len(keys):<4d} "
                  f"{nw:4d}/{ns:<3d} {nwa:9d}/{na:<5d} "
                  f"{worst_z:9.2f} ({worst_k})")
        # ---- C4: do the Fourier ladders move TOWARD the staggered limit?
        print(f"    {'engine':16s} {'last closer than first':>23s} "
              f"{'worst |first-stag| -> |last-stag|':>36s}")
        crow["toward"] = {}
        for e in ENGINES[1:]:
            n_closer = 0
            worst = (-1.0, None, None, None)
            for k in keys:
                stag = fits["staggered"][k]["f_inf"]
                _x, y = series(case, e, k)
                d0, d1 = abs(y[0] - stag), abs(y[-1] - stag)
                if d1 < d0:
                    n_closer += 1
                if d0 > worst[0]:
                    worst = (d0, d1, k, None)
            crow["toward"][e] = dict(closer=n_closer, live=len(keys),
                                     worst_first=worst[0], worst_last=worst[1],
                                     worst_key=worst[2])
            print(f"    {e:16s} {n_closer:9d}/{len(keys):<4d} "
                  f"{worst[2]:>20s}  {worst[0]:.2e} -> {worst[1]:.2e}")
        out["cases"].append(crow)
    print()
    print(f"--- summed over the four cases ({name}) ---")
    print(f"{'pair':34s} {'sound-only':>14s} {'ALL live':>14s}")
    out["totals"] = {}
    for p in PAIRS:
        w, n = tot_pair_sound[p]
        wa, na = tot_pair_all[p]
        out["totals"][f"{p[0]}|{p[1]}"] = dict(sound=[w, n], all=[wa, na])
        pct = 100.0 * w / n if n else float("nan")
        pcta = 100.0 * wa / na if na else float("nan")
        print(f"{p[0] + ' vs ' + p[1]:34s} {w:5d}/{n:<5d} ({pct:5.1f}%) "
              f"{wa:5d}/{na:<5d} ({pcta:5.1f}%)")
    return out


# --------------------------------------------------------------------------- #
def c5_bound():
    """C5 -- the top-of-ladder bound.  No extrapolation, no fitting."""
    print()
    print("#" * 100)
    print("# C5  TOP-OF-LADDER per-order bound (no extrapolation)")
    print("#" * 100)
    cases = load_cases()
    rows = []
    for case in cases:
        tag = f"{case['fixture']}/{case['mount']}"
        keys = live_keys(case)
        Rk = [k for k in keys if k.startswith("R(")]
        Tk = [k for k in keys if k.startswith("T(")]
        top = {e: {k: series(case, e, k)[1][-1] for k in keys}
               for e in ENGINES}
        print(f"\n=== {tag}")
        print(f"    {'pair':34s} {'max|dR|':>11s} {'max|dT|':>11s}")
        rec = dict(case=tag, pairs={})
        for A, B in PAIRS:
            dR = max(abs(top[A][k] - top[B][k]) for k in Rk)
            dT = max(abs(top[A][k] - top[B][k]) for k in Tk)
            rec["pairs"][f"{A}|{B}"] = dict(dR=dR, dT=dT)
            print(f"    {A + ' vs ' + B:34s} {dR:11.3e} {dT:11.3e}")
        f_pairs = [p for p in PAIRS if "staggered" not in p]
        s_pairs = [p for p in PAIRS if "staggered" in p]
        fR = [rec["pairs"][f"{a}|{b}"]["dR"] for a, b in f_pairs]
        fT = [rec["pairs"][f"{a}|{b}"]["dT"] for a, b in f_pairs]
        sR = [rec["pairs"][f"{a}|{b}"]["dR"] for a, b in s_pairs]
        sT = [rec["pairs"][f"{a}|{b}"]["dT"] for a, b in s_pairs]
        rec["fourier_mutual_R"] = [min(fR), max(fR)]
        rec["fourier_mutual_T"] = [min(fT), max(fT)]
        rec["ratio_R"] = max(sR) / max(fR)
        rec["ratio_T"] = max(sT) / max(fT)
        rec["stag_nearest_R"] = min(sR)
        rec["stag_vs_rcwa_R"] = rec["pairs"]["staggered|rcwa"]["dR"]
        print(f"    Fourier mutual   R {min(fR):.3e} .. {max(fR):.3e}   "
              f"T {min(fT):.3e} .. {max(fT):.3e}")
        print(f"    ratio stag/Fourier   R {rec['ratio_R']:.2f}x   "
              f"T {rec['ratio_T']:.2f}x")
        print(f"    staggered vs NEAREST Fourier arm on R: "
              f"{rec['stag_nearest_R']:.3e};  vs rcwa: "
              f"{rec['stag_vs_rcwa_R']:.3e}")
        rows.append(rec)
    return rows


def mid_ladder_bound():
    """The verification's original number: staggered M=7 vs rcwa n=9."""
    print()
    print("#" * 100)
    print("# the ORIGINAL bound restated from the same data (M=7 vs rcwa n=9)")
    print("#" * 100)
    for case in load_cases():
        tag = f"{case['fixture']}/{case['mount']}"
        keys = [k for k in live_keys(case) if k.startswith("R(")]
        xs = case["ladders"]["staggered"]["x"]
        xr = case["ladders"]["rcwa"]["x"]
        i7, i9 = xs.index(7), xr.index(9)
        i10, i11 = xs.index(10), xr.index(11)
        d79 = max(abs(series(case, "staggered", k)[1][i7]
                      - series(case, "rcwa", k)[1][i9]) for k in keys)
        d1011 = max(abs(series(case, "staggered", k)[1][i10]
                        - series(case, "rcwa", k)[1][i11]) for k in keys)
        print(f"{tag:26s}  M=7 vs n=9: {d79:.4e}   M=10 vs n=11: {d1011:.4e}")


if __name__ == "__main__":
    res_a = analyse(fit_A, "A -- power law, rate SCANNED on [0.25, 10]")
    res_b = analyse(fit_B, "B -- pure Aitken/Shanks (no model)")
    c5 = c5_bound()
    mid_ladder_bound()
    with open(os.path.join(HERE, "results", "v6_refit.json"), "w") as fh:
        json.dump(dict(fitA=res_a, fitB=res_b, c5=c5), fh, indent=1)
