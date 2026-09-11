"""V5 -- the population analysis of the V1 census.

Re-derives, from THIS verification's own 216-solve census, every number the
round-2 report's section 4 quotes: the energy populations scored against
set-wrongness, the index-ceiling populations, the false-positive count, the
closest set-right row to the slack, the mildest set-wrong row, and the rows
BOTH detectors miss.
"""
from __future__ import annotations

import argparse
import json
import math


def dec(a, b):
    if a <= 0 or b <= 0:
        return float("nan")
    return math.log10(b / a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    rows = d["rows"]
    C = d["constants"]
    BAR, SLACK, WARN = C["BAR"], C["CEILING_SLACK"], C["WARN"]

    stag = [r for r in rows if r["basis"] == "staggered"]
    nod = [r for r in rows if r["basis"] == "nodal"]
    right = [r for r in nod if not r["set_wrong"]]
    wrong = [r for r in nod if r["set_wrong"]]

    def two(r):
        return max(abs(r["excess"]), abs(r["deficit"]))

    print("census: %d rows (%d staggered, %d nodal: %d set-right, %d set-wrong)"
          % (len(rows), len(stag), len(nod), len(right), len(wrong)))
    print()
    print("=== ENERGY, |R+T-1|, scored against SET-WRONGNESS ===")
    for name, pop in (("STAGGERED", stag), ("NODAL set RIGHT", right),
                      ("NODAL set WRONG", wrong)):
        v = sorted(two(r) for r in pop)
        if v:
            print("  %-18s n=%-4d %.6e .. %.6e" % (name, len(v), v[0], v[-1]))
    if right and wrong:
        print("  OVERLAP: set-right reaches %.6e, set-wrong starts at %.6e "
              "-> %.2f decades of overlap"
              % (max(two(r) for r in right), min(two(r) for r in wrong),
                 dec(min(two(r) for r in wrong), max(two(r) for r in right))))
    print()
    print("=== INDEX CEILING, Re qn - n_max ===")
    for name, pop in (("STAGGERED", stag), ("NODAL set RIGHT", right),
                      ("NODAL set WRONG", wrong)):
        v = sorted(r["ceiling_excess"] for r in pop)
        pos = [x for x in v if x > SLACK]
        if v:
            print("  %-18s n=%-4d worst %.6e   fires on %d"
                  % (name, len(v), v[-1], len(pos)))
    fires_right = [r for r in right if r["ceiling_excess"] > SLACK]
    fires_stag = [r for r in stag if r["ceiling_excess"] > SLACK]
    print("  FALSE POSITIVES (ceiling fires on an undamaged row): %d of %d"
          % (len(fires_right) + len(fires_stag), len(right) + len(stag)))
    for r in fires_right + fires_stag:
        print("     %s %s ceil=%.6e" % (r["name"], r["basis"],
                                        r["ceiling_excess"]))
    undam = right + stag
    closest = max(undam, key=lambda r: r["ceiling_excess"])
    print("  closest UNDAMAGED row to the slack: %s %s at %.6e "
          "(slack %.0e -> %.2f decades below)"
          % (closest["name"], closest["basis"], closest["ceiling_excess"],
             SLACK, dec(abs(closest["ceiling_excess"]), SLACK)
             if closest["ceiling_excess"] < 0 else float("nan")))
    fw = [r for r in wrong if r["ceiling_excess"] > SLACK]
    if fw:
        mild = min(fw, key=lambda r: r["ceiling_excess"])
        print("  mildest SET-WRONG row the ceiling catches: %s at %.6e "
              "(%.2f decades above the slack)"
              % (mild["name"], mild["ceiling_excess"],
                 dec(SLACK, mild["ceiling_excess"])))
    print()
    print("=== THE UNION, and what it misses ===")
    caught_e = [r for r in nod if two(r) > BAR]
    caught_c = [r for r in nod if r["ceiling_excess"] > SLACK]
    caught = {r["name"] for r in caught_e} | {r["name"] for r in caught_c}
    missed = [r for r in wrong if r["name"] not in caught]
    ceil_only = [r for r in wrong
                 if r["ceiling_excess"] > SLACK and two(r) <= BAR]
    e_only = [r for r in wrong
              if r["ceiling_excess"] <= SLACK and two(r) > BAR]
    print("  union catches %d of %d nodal rows; MISSES %d of %d set-wrong"
          % (len(caught), len(nod), len(missed), len(wrong)))
    print("  ceiling-ONLY catches (energy silent): %d" % (len(ceil_only),))
    for r in ceil_only:
        print("     %-26s |R+T-1|=%.4e ceil=%.4e" % (r["name"], two(r),
                                                     r["ceiling_excess"]))
    print("  energy-ONLY catches (ceiling silent): %d" % (len(e_only),))
    print("  MISSED set-wrong rows (both detectors silent):")
    for r in missed:
        print("     %-26s n=%d/%d ref=%s |R+T-1|=%.4e ceil=%.4e emax=%.9g "
              "emin=%.9g" % (r["name"], r["n_inc"], r["n_out"],
                             r.get("ref_counts"), two(r),
                             r["ceiling_excess"], r["emax"], r["emin"]))
    print()
    print("=== THE SUB-POPULATION THE ENERGY BAR OWNS ===")
    own = [r for r in nod if not r["set_wrong"]
           and r["ceiling_excess"] <= SLACK]
    uni = [r for r in own if r["kind"] == "uniform"]
    oth = [r for r in own if r["kind"] != "uniform"]
    for nm, pop in (("accurate family (uniform)", uni),
                    ("structured (ring+segment)", oth)):
        v = sorted(two(r) for r in pop)
        if v:
            print("  %-28s n=%-4d %.6e .. %.6e" % (nm, len(v), v[0], v[-1]))
    if uni and oth:
        top_u = max(two(r) for r in uni)
        bot_o = min(two(r) for r in oth)
        print("  gap: %.6e -> %.6e = %.2f decades; the bar %0.0e sits %.2f "
              "decades above the accurate ceiling and %.2f below the mildest "
              "row it must refuse"
              % (top_u, bot_o, dec(top_u, bot_o), BAR, dec(top_u, BAR),
                 dec(BAR, bot_o)))
        print("  the WARN edge %.0e sits %.2f decades above the accurate "
              "ceiling" % (WARN, dec(top_u, WARN)))
    print()
    print("=== DECISIONS ===")
    for v in ("returned", "REFUSED"):
        print("  nodal %-9s %d" % (v, sum(1 for r in nod
                                          if r["verdict"] == v)))
    print("  refused by energy : %d"
          % sum(1 for r in nod if r["detector"] == "energy"))
    print("  refused by ceiling: %d"
          % sum(1 for r in nod if r["detector"] == "ceiling"))
    print("  staggered refused : %d"
          % sum(1 for r in stag if r["verdict"] == "REFUSED"))
    dfc = [r for r in nod if r["verdict"] == "REFUSED"
           and "DEFICIT" in r.get("message", "")]
    print("  refusals naming a DEFICIT: %d" % (len(dfc),))
    for r in dfc[:6]:
        print("     %-26s emax=%.6g emin=%.6g" % (r["name"], r["emax"],
                                                  r["emin"]))


if __name__ == "__main__":
    main()
