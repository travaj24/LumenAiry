"""ROUND 2, D2 -- derive the TWO-SIDED screen from the measured populations.

Reads the ``r1_passivity_census`` JSON of one arm and asks, of each candidate
screen, the only question ``docs/TESTING_STANDARDS.md`` rule 5 accepts: is
there a GAP on both sides, and how many decades wide?

THE REFERENCE, AND WHY IT IS NOT CIRCULAR.  Every row is solved TWICE on the
same geometry -- once on the legacy ``nodal`` basis and once on the
div-conforming ``staggered`` one.  The staggered basis is the reference for
the CHANNEL SET (not for the energy, which is what is being screened): its own
closure is <= 6.5e-12 on every row of this census, and on wide-basin uniform
fixtures at ``m >= 1`` it returns the closed-form Bessel-zero count.  So

    SET-WRONG  :=  the nodal channel count differs from the staggered one

is evidence of damage that reads NO energy at all, and the energy bar can be
scored against it without assuming its own conclusion.

The closed-form Bessel-zero count is reported alongside as the fully
independent oracle, restricted to the rows where it is trustworthy (uniform
profile, wide basin, ``m >= 1``): the verification measured its own ``m = 0``
bookkeeping off by one and its large-cell misses, so it is a CENSUS oracle and
not a production predicate.

THE THIRD CANDIDATE is deterministic and carries no bar at all: a returned
channel whose axial index ``Re qn`` exceeds the incidence half-space's own
index ``n``.  ``q^2`` is an eigenvalue of ``eps k0^2 + D`` with ``D`` the
transverse operator; for the continuum problem and for any basis whose ``D`` is
negative semi-definite, the Rayleigh quotient bounds ``q^2 <= max(eps) k0^2``,
so ``qn > n`` means ``gamma^2 < 0`` -- impossible, and a signature of the nodal
basis's divergence-violating sea rather than of any arithmetic.

Run:  python validation/probe_fix_bor_round2/r4_bars.py [json ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def decades(a, b):
    if a is None or b is None or a <= 0 or b <= 0 or not np.isfinite(a) \
            or not np.isfinite(b):
        return float("nan")
    return float(np.log10(b / a))


def _key(r):
    return (r["family"], r["m"], r["N"], r["rbl"])


def analyse(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = [r for r in doc["rows"] if "error" not in r]
    arm = doc["arm"]
    print("=" * 78)
    print("%s" % (os.path.basename(path),))
    print("  arm: %s %s / %s / %s thread(s)"
          % (arm["platform"], arm["python"], arm.get("kernel"),
             arm.get("threads")))
    stag = {_key(r): r for r in rows if r["basis"] == "staggered"}
    nod = [r for r in rows if r["basis"] == "nodal"]

    se = [max(abs(r["excess"]), abs(r["deficit"])) for r in stag.values()]
    print("  STAGGERED reference, n=%d: worst |R+T-1| = %.6e"
          % (len(se), max(se)))
    orc = [r for r in stag.values()
           if r["family"] == "uniform" and r["basin"] > 1e-2 and r["m"] >= 1]
    print("    closed-form Bessel count exact on %d of %d trustworthy rows "
          "(uniform, basin > 1e-2, m >= 1)"
          % (sum(1 for r in orc if r["count_matches"]), len(orc)))
    ceil_stag = max(r["ceiling_excess"] for r in stag.values())
    print("    worst staggered Re qn / n - 1 = %+.6e (must be < 0)"
          % (ceil_stag,))

    for r in nod:
        tw = stag.get(_key(r))
        r["_twin"] = None if tw is None else tw["n_channels"]
        r["_set_wrong"] = bool(tw is not None
                               and tw["n_channels"] != r["n_channels"])
        r["_abs"] = max(abs(r["excess"]), abs(r["deficit"]))
        r["_oracle_ok"] = bool(r["family"] == "uniform" and r["basin"] > 1e-2
                               and r["m"] >= 1)
    wrong = [r for r in nod if r["_set_wrong"]]
    right = [r for r in nod if not r["_set_wrong"]]
    print("  NODAL n=%d: %d SET-WRONG, %d SET-RIGHT" % (len(nod), len(wrong),
                                                        len(right)))
    out = dict(arm=arm, n_nodal=len(nod), n_set_wrong=len(wrong),
               n_set_right=len(right), staggered_worst_abs=max(se),
               staggered_worst_ceiling=ceil_stag)

    # ---------------- the energy axis, both sides ----------------
    def band(sel, key, label):
        v = sorted(abs(r[key]) for r in sel)
        if not v:
            return None
        print("    %-34s n=%2d  %.6e .. %.6e" % (label, len(v), v[0], v[-1]))
        return dict(n=len(v), lo=v[0], hi=v[-1])

    print("  -- the ENERGY axis --")
    out["set_right_excess"] = band(right, "excess", "SET-RIGHT  |excess|")
    out["set_right_deficit"] = band(right, "deficit", "SET-RIGHT  |deficit|")
    out["set_right_abs"] = band(right, "_abs", "SET-RIGHT  |R+T-1|")
    out["set_wrong_abs"] = band(wrong, "_abs", "SET-WRONG  |R+T-1|")

    hi = max(r["_abs"] for r in right) if right else float("nan")
    lo = min(r["_abs"] for r in wrong) if wrong else float("nan")
    print("    two-sided |R+T-1| gap: SET-RIGHT worst %.6e vs SET-WRONG "
          "mildest %.6e -> %.2f decades" % (hi, lo, decades(hi, lo)))
    out["two_sided_gap"] = dict(set_right_worst=hi, set_wrong_mildest=lo,
                                decades=decades(hi, lo))

    hie = max(abs(r["excess"]) for r in right) if right else float("nan")
    loe = min(abs(r["excess"]) for r in wrong) if wrong else float("nan")
    print("    one-sided excess gap : SET-RIGHT worst %.6e vs SET-WRONG "
          "mildest %.6e -> %.2f decades" % (hie, loe, decades(hie, loe)))
    out["one_sided_gap"] = dict(set_right_worst=hie, set_wrong_mildest=loe,
                                decades=decades(hie, loe))

    # what each bar catches
    for bar in (1e-3, 1e-4, 1e-6):
        e_fire = [r for r in nod if r["excess"] > bar]
        s_fire = [r for r in nod if r["_abs"] > bar]
        print("    bar %.0e : one-sided fires on %2d rows (%2d SET-RIGHT), "
              "two-sided on %2d (%2d SET-RIGHT)"
              % (bar, len(e_fire), sum(1 for r in e_fire if not r["_set_wrong"]),
                 len(s_fire), sum(1 for r in s_fire if not r["_set_wrong"])))
        out.setdefault("bar_scan", {})["%.0e" % bar] = dict(
            one_sided=len(e_fire),
            one_sided_set_right=sum(1 for r in e_fire if not r["_set_wrong"]),
            two_sided=len(s_fire),
            two_sided_set_right=sum(1 for r in s_fire if not r["_set_wrong"]))

    # rows the DEFICIT half alone adds at the shipped bar
    add = [r for r in nod if r["excess"] <= 1e-3 and abs(r["deficit"]) > 1e-3]
    print("  -- rows the DEFICIT half adds at the shipped 1e-3 bar: %d" % len(add))
    for r in add:
        print("     %-8s m=%d N=%3d rbl=%4.1f  exc=%+.4e def=%+.4e  "
              "nch=%d twin=%s exact=%d  ceil=%+.4e"
              % (r["family"], r["m"], r["N"], r["rbl"], r["excess"],
                 r["deficit"], r["n_channels"], r["_twin"],
                 r["exact_channels"], r["ceiling_excess"]))
    out["deficit_only_rows"] = [
        dict(family=r["family"], m=r["m"], N=r["N"], rbl=r["rbl"],
             excess=r["excess"], deficit=r["deficit"],
             n_channels=r["n_channels"], twin=r["_twin"],
             exact=r["exact_channels"], ceiling_excess=r["ceiling_excess"],
             set_wrong=r["_set_wrong"])
        for r in add]

    # ---------------- the deterministic conjunct ----------------
    print("  -- the INDEX-CEILING conjunct, as a DECISION (Re qn > n) --")
    cw = [r for r in wrong if r["ceiling_excess"] > 0.0]
    cr = [r for r in right if r["ceiling_excess"] > 0.0]
    cs = [r for r in stag.values() if r["ceiling_excess"] > 0.0]
    print("     fires on %d of %d SET-WRONG, %d of %d SET-RIGHT nodal, "
          "%d of %d staggered" % (len(cw), len(wrong), len(cr), len(right),
                                  len(cs), len(stag)))
    ce_wrong = sorted(r["ceiling_excess"] for r in cw)
    ce_right = sorted(r["ceiling_excess"] for r in right)
    ce_stag = sorted(r["ceiling_excess"] for r in stag.values())
    print("     SET-WRONG  ceiling excess when it fires: %.6e .. %.6e"
          % (ce_wrong[0], ce_wrong[-1]) if ce_wrong else "     (never fires)")
    print("     SET-RIGHT  ceiling excess, worst (must be < 0): %+.6e"
          % (ce_right[-1],))
    print("     STAGGERED  ceiling excess, worst (must be < 0): %+.6e"
          % (ce_stag[-1],))
    print("     DEADBAND ROOM: mildest firing %.3e vs worst non-firing %+.3e"
          % (ce_wrong[0] if ce_wrong else float("nan"),
             max(ce_right[-1], ce_stag[-1])))
    out["ceiling_decision"] = dict(
        fires_set_wrong=len(cw), n_set_wrong=len(wrong),
        fires_set_right=len(cr), n_set_right=len(right),
        fires_staggered=len(cs), n_staggered=len(stag),
        mildest_firing=ce_wrong[0] if ce_wrong else None,
        worst_set_right=ce_right[-1] if ce_right else None,
        worst_staggered=ce_stag[-1] if ce_stag else None)

    # ---------------- what the union catches ----------------
    for bar in (1e-3, 1e-4):
        u = [r for r in nod if r["_abs"] > bar or r["ceiling_excess"] > 0.0]
        miss = [r for r in wrong if not (r["_abs"] > bar
                                         or r["ceiling_excess"] > 0.0)]
        fp = [r for r in u if not r["_set_wrong"] and r["_abs"] <= bar]
        print("  -- UNION (|R+T-1| > %.0e OR ceiling) : fires on %d of %d "
              "rows; misses %d of %d SET-WRONG; %d SET-RIGHT caught by the "
              "ceiling alone" % (bar, len(u), len(nod), len(miss), len(wrong),
                                 len(fp)))
        out.setdefault("union", {})["%.0e" % bar] = dict(
            fires=len(u), misses_set_wrong=len(miss), n_set_wrong=len(wrong),
            set_right_by_ceiling=len(fp))

    # the oracle cross-check: where the ceiling fires, is the count wrong?
    oc = [r for r in nod if r["_oracle_ok"]]
    both = [r for r in oc if r["ceiling_excess"] > 0.0
            and not r["count_matches"]]
    fire = [r for r in oc if r["ceiling_excess"] > 0.0]
    print("  -- Bessel-zero cross-check on %d trustworthy nodal rows: the "
          "ceiling fires on %d, and %d of those also disagree with the "
          "closed-form count" % (len(oc), len(fire), len(both)))
    out["bessel_crosscheck"] = dict(n=len(oc), fires=len(fire),
                                    also_count_wrong=len(both))
    return out


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(HERE, "r1_passivity_census_*.json")))
    if not paths:
        print("no census JSON found; run r1_passivity_census.py first")
        return 1
    res = [analyse(p) for p in paths]
    with open(os.path.join(HERE, "r4_bars.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print("\n[probe] wrote %s" % (os.path.join(HERE, "r4_bars.json"),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
