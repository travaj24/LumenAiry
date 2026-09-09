"""T2 -- render the T1 ladders as the tables the reference report carries.

Reads ``results/t1_<fixture>.json`` and prints, per fixture x mount:

  * each ladder with its successive steps;
  * the extrapolated limit and uncertainty per engine;
  * the pairwise question -- do the engines' extrapolated limits agree within
    their COMBINED uncertainties, per observable?
  * the tightened BOUND on the staggered arm: its distance from the
    three-Fourier-arm consensus, and from its own extrapolated limit.

Observables that are EXACTLY zero in every engine (the evanescent orders) are
excluded: they agree trivially and carry no uncertainty, so including them
would inflate every "n within sigma" count.

Run:
  cd /c/tmp/lum_oopfast && PYTHONPATH=/c/tmp/lum_oopfast \\
    python validation/probe_pmm2d_staggered_oop_reference/t2_summary.py corner
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ENG_ORDER = ["staggered", "hybrid-laurent", "hybrid-li", "rcwa"]

#: the rate ``p`` is scanned on [0.25, 10]; a best fit that lands ON either end
#: is a DIAGNOSED FAILURE of the extrapolation, not a rate -- the ladder has
#: not entered its asymptotic regime and the extrapolant is an artefact of the
#: scan boundary.  Such fits are reported but excluded from the agreement
#: verdict.
P_LO, P_HI = 0.30, 9.70


def pinned(f):
    return not (P_LO < f["p"] < P_HI)


def live_keys(fits, engs):
    keys = []
    for k in sorted(fits[engs[0]]):
        if not (k.startswith("sum") or k.startswith("R(") or k.startswith("T(")):
            continue
        if all(fits[e][k]["f_inf"] == 0.0 for e in engs):
            continue                       # evanescent order: exactly 0 in all
        keys.append(k)
    return keys


def main():
    fix = sys.argv[1] if len(sys.argv) > 1 else "corner"
    d = json.load(open(os.path.join(OUT, f"t1_{fix}.json")))
    for rec in d["cases"]:
        mount = rec["mount"]
        print(f"\n\n### {fix} / {mount}\n")
        print("#### ladders (`sum R`, incident Ex) and their own steps\n")
        print("| engine | rung | `sum R` | step | t [s] |")
        print("|---|---|---|---|---|")
        for eng in ENG_ORDER:
            la = rec["ladders"].get(eng)
            if la is None:
                continue
            prev = None
            for x, o, t in zip(la["x"], la["obs"], la["t"]):
                v = o["sumR0"]
                st = "--" if prev is None else f"{abs(v - prev):.3e}"
                lab = f"M={x}" if eng == "staggered" else f"n={x}"
                print(f"| {eng} | {lab} | {v:.9f} | {st} | {t:.1f} |")
                prev = v
        if "fits" not in rec:
            print("\n(no fits: the run did not finish this mount)")
            continue
        fits = rec["fits"]
        engs = [e for e in ENG_ORDER if e in fits]
        keys = live_keys(fits, engs)

        print("\n#### extrapolated limits (fit `f = f_inf + C x^-p`, `p` "
              "scanned)\n")
        print("| observable | " + " | ".join(engs) + " |")
        print("|---" * (len(engs) + 1) + "|")
        for k in keys:
            cells = []
            for e in engs:
                f = fits[e][k]
                mark = " (P)" if pinned(f) else ""
                cells.append(f"{f['f_inf']:.7f} +/- {f['sigma']:.1e}{mark}")
            print(f"| `{k}` | " + " | ".join(cells) + " |")
        print("\n`(P)` = the fitted rate `p` landed ON the scan boundary "
              f"([{P_LO}, {P_HI}] excluded): the ladder has NOT entered its "
              "asymptotic regime and the extrapolant is an artefact, not a "
              "limit.  Those entries are reported but excluded from the "
              "agreement verdict below.")

        print("\n#### fit health and uncertainty envelope per engine (over "
              f"the {len(keys)} live observables)\n")
        print("| engine | well-conditioned fits | sigma min | sigma max | "
              "median sigma | top-of-ladder step (max) |")
        print("|---|---|---|---|---|---|")
        for e in engs:
            good = [k for k in keys if not pinned(fits[e][k])]
            sg = np.array([fits[e][k]["sigma"] for k in good]) if good else \
                np.array([np.nan])
            ls = np.array([fits[e][k]["last_step"] for k in keys])
            print(f"| {e} | **{len(good)}/{len(keys)}** | {sg.min():.2e} | "
                  f"{sg.max():.2e} | {np.median(sg):.2e} | "
                  f"{np.nanmax(ls):.2e} |")

        print("\n#### pairwise on the WELL-CONDITIONED subset: "
              "|limit_A - limit_B| against `sqrt(sA^2 + sB^2)`\n")
        print("| pair | both fits sound | within combined sigma | "
              "worst observable | d | combined sigma | z = d/sigma |")
        print("|---|---|---|---|---|---|---|")
        for i, a in enumerate(engs):
            for b in engs[i + 1:]:
                sub = [k for k in keys
                       if not pinned(fits[a][k]) and not pinned(fits[b][k])]
                if not sub:
                    print(f"| {a} vs {b} | 0/{len(keys)} | -- | -- | -- | "
                          f"-- | -- |")
                    continue
                worst, n_ok = None, 0
                for k in sub:
                    dd = abs(fits[a][k]["f_inf"] - fits[b][k]["f_inf"])
                    cc = float(np.hypot(fits[a][k]["sigma"],
                                        fits[b][k]["sigma"]))
                    z = 0.0 if dd == 0.0 else (dd / cc if cc > 0 else np.inf)
                    n_ok += int(z <= 1.0)
                    if worst is None or z > worst[3]:
                        worst = (k, dd, cc, z)
                print(f"| {a} vs {b} | {len(sub)}/{len(keys)} | "
                      f"{n_ok}/{len(sub)} | `{worst[0]}` | {worst[1]:.3e} | "
                      f"{worst[2]:.3e} | {worst[3]:.2f} |")

        print("\n#### do the FOURIER ladders move TOWARD the staggered limit?\n")
        print("| engine | monotone rungs | last rung closer to the staggered "
              "limit than the first | |first - stag| -> |last - stag| (worst "
              "observable) |")
        print("|---|---|---|---|")
        stl = {k: fits["staggered"][k]["f_inf"] for k in keys}
        for e in [x for x in engs if x != "staggered"]:
            la = rec["ladders"][e]
            mono = closer = 0
            worst = None
            for k in keys:
                v = np.array([o[k] for o in la["obs"]])
                dv = np.diff(v)
                mono += int(np.all(dv >= 0) or np.all(dv <= 0))
                d0, d1 = abs(v[0] - stl[k]), abs(v[-1] - stl[k])
                closer += int(d1 < d0)
                if worst is None or d1 > worst[2]:
                    worst = (k, d0, d1)
            print(f"| {e} | {mono}/{len(keys)} | {closer}/{len(keys)} | "
                  f"`{worst[0]}` {worst[1]:.2e} -> {worst[2]:.2e} |")

        print("\n#### the BOUND on the staggered arm\n")
        st = fits["staggered"]
        fo = [e for e in engs if e != "staggered"]
        rows = []
        for k in keys:
            lim = [fits[e][k]["f_inf"] for e in fo]
            cons = float(np.mean(lim))
            spread = float(np.max(lim) - np.min(lim))
            rows.append((k, abs(st[k]["last"] - st[k]["f_inf"]),
                         abs(st[k]["last"] - cons), spread))
        wr = max(rows, key=lambda r: r[1])
        wc = max(rows, key=lambda r: r[2])
        ws = max(rows, key=lambda r: r[3])
        print("| quantity | worst observable | value |")
        print("|---|---|---|")
        print(f"| staggered top rung vs its OWN extrapolated limit | "
              f"`{wr[0]}` | **{wr[1]:.3e}** |")
        print(f"| staggered top rung vs the mean of the three Fourier limits | "
              f"`{wc[0]}` | **{wc[2]:.3e}** |")
        print(f"| the three Fourier limits' own SPREAD | `{ws[0]}` | "
              f"**{ws[3]:.3e}** |")

        print("\n#### top-of-ladder distances (no extrapolation), per order\n")
        print("| pair | max abs dR | max abs dT |")
        print("|---|---|---|")
        for i, a in enumerate(engs):
            for b in engs[i + 1:]:
                oa = rec["ladders"][a]["obs"][-1]
                ob = rec["ladders"][b]["obs"][-1]
                kr = [k for k in oa if k.startswith("R(") and k in ob]
                kt = [k for k in oa if k.startswith("T(") and k in ob]
                print(f"| {a} vs {b} | "
                      f"{max(abs(oa[k]-ob[k]) for k in kr):.3e} | "
                      f"{max(abs(oa[k]-ob[k]) for k in kt):.3e} |")

        la = rec["ladders"]["staggered"]
        vals = np.array([o["sumR0"] for o in la["obs"]])
        steps = np.abs(np.diff(vals))
        print(f"\nstaggered `sum R` steps (M = {la['x'][0]}..{la['x'][-1]}): "
              f"{' '.join(f'{s:.3e}' for s in steps)}")
        print("step ratios: "
              + " ".join(f"{steps[i+1]/steps[i]:.3f}"
                         for i in range(steps.size - 1)))
        pr = [o for o in la["obs"][-1] if o.startswith("R(")]
        o1, o0 = la["obs"][-1], la["obs"][-2]
        print(f"staggered per-order top step: dR "
              f"{max(abs(o1[k]-o0[k]) for k in pr):.3e}")


if __name__ == "__main__":
    main()
