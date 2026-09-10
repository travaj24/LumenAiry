"""ROUND 2, probe 1 -- the two populations and the ARBITER, on the dense grid.

The verification (S4.1) refuted round 1's margins on a 120-delta x 3-degree
grid: the CORRECT population reaches |R+T-1| = 9.867e-05 and the WRONG one
reaches DOWN to +7.142e-03, i.e. BELOW round 1's 1e-2 bar.  This probe

  * re-derives both envelopes on that same grid, on the running build;
  * measures, on EVERY row that reads super-unity above the lowest trigger
    considered (1e-4), what ONE re-solve on the prescribed ``min_feature``
    grid does to the super-unity and to the answer -- the discriminator;
  * scores the resulting REFUSE / RETURN decision at four candidate trigger
    bars against the continuity classification, two-sided.

Nothing in the library is used except the fail-before switch and the geometric
screen; the arbiter is implemented here so the bars can be chosen from the
measurement rather than assumed.

    python validation/probe_pmmstack_sliver_round2/r1_populations.py [out.json]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r_fixtures as F  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRIGGERS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
LOW = min(TRIGGERS)


def arbitrate(d, deg, base, mf):
    """ONE re-solve on the prescribed grid.  Returns the snapped super-unity
    and how far the answer moved, in units of the widest manufactured cell."""
    st2 = F.build(d, deg, mf=mf)
    snap = F.raw(st2)
    return dict(snapped_RplusT=snap[3], move=F.err(base, snap),
                move_both=F.move_both(base, snap))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r1_populations.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()

    deltas = [float(x) for x in np.geomspace(3e-3, 1e-6, 120)]
    rows = []
    n_arb = 0
    for deg in (10, 14, 20):
        ref = F.raw(F.build(0.0, deg))
        for d in deltas:
            st = F.build(d, deg)
            base = F.raw(st)
            e = F.err(base, ref)
            row = dict(degree=deg, delta=d, err=e, err_over_delta=e / d,
                       RplusT=base[3], kind=F.kind_of(e, d))
            mf = F.prescribed_mf(st)
            row["screen_hit"] = mf is not None
            row["passive"] = bool(ps._stack_provably_passive(st))
            row["prescribed_mf"] = mf
            if (mf is not None and row["passive"]
                    and base[3] > 1.0 + LOW):
                row.update(arbitrate(d, deg, base, mf))
                n_arb += 1
            rows.append(row)
        print(f"  degree {deg} done ({time.time() - t0:.0f} s)", flush=True)

    right = [abs(r["RplusT"] - 1.0) for r in rows if r["kind"] == "right"]
    wrong = [r["RplusT"] - 1.0 for r in rows if r["kind"] == "wrong"]
    grey = [r for r in rows if r["kind"] == "grey"]
    print(f"\n  rows {len(rows)}  right {len(right)}  wrong {len(wrong)}  "
          f"grey {len(grey)}")
    print(f"  max |R+T-1| among CORRECT = {max(right):.4e}")
    print(f"  min  (R+T-1) among WRONG  = {min(wrong):.4e}")
    print(f"  arbiter re-solves run: {n_arb} of {len(rows)} rows")

    # ---- decisions at each candidate trigger ---------------------------
    table = {}
    for trig in TRIGGERS:
        r1_ref = r1_ret = 0                 # round 1: refuse iff > 1e-2 bar
        n_ref_wrong = n_ref_right = n_ref_grey = 0
        n_ret_wrong = 0
        for r in rows:
            fires = (r["screen_hit"] and r["passive"]
                     and r["RplusT"] > 1.0 + trig)
            att = fires and r.get("snapped_RplusT", 9e9) <= 1.0 + trig
            if att:
                if r["kind"] == "wrong":
                    n_ref_wrong += 1
                elif r["kind"] == "right":
                    n_ref_right += 1
                else:
                    n_ref_grey += 1
            elif r["kind"] == "wrong":
                n_ret_wrong += 1
            r1 = (r["screen_hit"] and r["passive"]
                  and r["RplusT"] > 1.0 + 1e-2)
            r1_ref += bool(r1)
            r1_ret += bool(not r1 and r["kind"] == "wrong")
        table[f"{trig:g}"] = dict(
            refused_wrong=n_ref_wrong, refused_right=n_ref_right,
            refused_grey=n_ref_grey, returned_wrong=n_ret_wrong,
            round1_refused=r1_ref, round1_returned_wrong=r1_ret)
        print(f"  trigger {trig:g}: refuse wrong {n_ref_wrong}/{len(wrong)}  "
              f"right {n_ref_right}/{len(right)}  grey {n_ref_grey}/"
              f"{len(grey)}  |  wrong-but-RETURNED {n_ret_wrong}")

    # the rows the arbiter could see but did NOT attribute
    surv = [r for r in rows if "snapped_RplusT" in r
            and r["snapped_RplusT"] > 1.0 + 1e-3]
    print(f"\n  rows whose super-unity SURVIVES the snap at 1e-3: {len(surv)}")
    for r in surv[:8]:
        print(f"    deg {r['degree']} d {r['delta']:.4e} kind {r['kind']} "
              f"R+T {r['RplusT']:.6g} -> {r['snapped_RplusT']:.6g}")

    summ = dict(n_rows=len(rows), n_right=len(right), n_wrong=len(wrong),
                n_grey=len(grey), max_absRT1_right=max(right),
                min_RT1_wrong=min(wrong), n_arbiter=n_arb,
                decisions=table, wall_s=time.time() - t0)
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       rows=rows, summary=summ), f, indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
