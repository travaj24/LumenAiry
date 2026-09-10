"""ROUND 2, probe 2 -- the FALSE-NEGATIVE census, before and after.

The verification's S4.3 found 8 rows in 660 samples that round 1 RETURNS and
that the continuity rule does not call correct (5 WRONG, 3 grey), errors to
2.80e-03 at ``R+T-1`` up to +7.14e-03 -- i.e. UNDER round 1's 1e-2 bar and
therefore unwarned.  Its grid is 60 log deltas in 3e-5..1e-6 x degrees
8/10/12/14/16 (the 120 x 3 grid of probe 1 supplies the eighth row).

This re-runs that grid and scores every row three ways: round 1 (refuse iff
``R+T-1`` > 1e-2), the round-2 trigger ladder, and the arbiter's verdict from
one re-solve on the prescribed ``min_feature`` grid.

    python validation/probe_pmmstack_sliver_round2/r2_falseneg.py [out.json]
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
TRIGGERS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "r2_falseneg.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_sliver2" in lib.replace("\\", "/"), lib
    t0 = time.time()

    deltas = [float(x) for x in np.geomspace(3e-5, 1e-6, 60)]
    rows = []
    for deg in (8, 10, 12, 14, 16):
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
            if mf is not None and row["passive"] and base[3] > 1.0 + 1e-6:
                snap = F.raw(F.build(d, deg, mf=mf))
                row["snapped_RplusT"] = snap[3]
                row["move"] = F.err(base, snap)
                row["move_both"] = F.move_both(base, snap)
                row["move0"] = F.err0(base, snap)
                row["same_shape"] = bool(len(base[0]) == len(snap[0]))
                row["snapped_err"] = F.err(snap, ref)
            rows.append(row)
        print(f"  degree {deg} done ({time.time() - t0:.0f} s)", flush=True)

    nw = sum(1 for r in rows if r["kind"] == "wrong")
    ng = sum(1 for r in rows if r["kind"] == "grey")
    nr = sum(1 for r in rows if r["kind"] == "right")
    print(f"\n  rows {len(rows)}  right {nr}  wrong {nw}  grey {ng}")

    table = {}
    for trig in TRIGGERS:
        ref_w = ref_r = ref_g = ret_w = ret_g = 0
        for r in rows:
            fires = (r["screen_hit"] and r["passive"]
                     and r["RplusT"] > 1.0 + trig)
            att = fires and r.get("snapped_RplusT", 9e9) <= 1.0 + trig
            if att:
                ref_w += r["kind"] == "wrong"
                ref_r += r["kind"] == "right"
                ref_g += r["kind"] == "grey"
            else:
                ret_w += r["kind"] == "wrong"
                ret_g += r["kind"] == "grey"
        table[f"{trig:g}"] = dict(refused_wrong=ref_w, refused_right=ref_r,
                                  refused_grey=ref_g, returned_wrong=ret_w,
                                  returned_grey=ret_g)
        print(f"  trigger {trig:g}: refuse wrong {ref_w}/{nw} right {ref_r}/"
              f"{nr} grey {ref_g}/{ng}  |  RETURNED wrong {ret_w} grey {ret_g}")

    # the round-1 misses, listed
    miss = [r for r in rows
            if r["kind"] in ("wrong", "grey") and r["RplusT"] <= 1.0 + 1e-2]
    print(f"\n  ROUND-1 misses on this grid: {len(miss)}")
    for r in sorted(miss, key=lambda x: -x["err"]):
        print(f"    deg {r['degree']:2d} d {r['delta']:.4e} kind {r['kind']:5s}"
              f" err {r['err']:.4e} ({r['err_over_delta']:6.1f}x) R+T-1 "
              f"{r['RplusT'] - 1.0:+.4e} snapped "
              f"{r.get('snapped_RplusT', float('nan')) - 1.0:+.3e}")

    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__),
                       rows=rows, decisions=table,
                       n_right=nr, n_wrong=nw, n_grey=ng,
                       round1_misses=miss, wall_s=time.time() - t0), f,
                  indent=1, default=str)
    print("wrote", out_path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
