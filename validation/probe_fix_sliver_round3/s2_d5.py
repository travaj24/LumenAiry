"""S2 -- the D-5 rows, before and after the relative closure.

D-5 (``docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md`` S8): on a
stack whose SLIVER-FREE truncation super-unity already sits ABOVE the
ABSOLUTE closure bar, the round-2 criterion ``su_snapped <= 1e-5`` can never
be met -- however completely the prescribed snap restores the answer.  The
arbiter then says ``truncation``, the wrong number is RETURNED, and the
warning names a remedy it says will not work.

This probe measures, on the D-5 fixture and on both builds:

  * the sliver-FREE degree ladder, i.e. that the mount's floor is ORDINARY
    truncation and where it sits relative to the two bars;
  * per delta: the returned super-unity, the super-unity on the prescribed
    grid, their ratio (the DROP factor), ``move / w_wide``, the error against
    the sliver-free reference before and after the snap;
  * the ANALYTIC verdict under round 2 and under the candidate round-3
    criterion at several ``_SLIVER_CLOSURE_FRACTION`` values;
  * what the LIBRARY actually decides, and which sentences its warning or
    refusal carries.

The fourth delta (1.1943e-05) is the CONTROL: its answer tracks the
sliver-free reference to 0.032x the wall shift and its snap removes only
1.05x of the super-unity, so no criterion may refuse it.

    python s2_d5.py [out.json]
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from f_fixtures import (  # noqa: E402
    drop_factor,
    gmr,
    guarded,
    prescribed,
    shared_move,
    snapped,
    unguarded,
    verdict_round2,
    verdict_round3,
)

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

#: the three rows D-5 names, plus one CONTROL row that must stay returned
DELTAS = (1.1943e-05, 6.8726e-06, 5.2134e-06, 3.0000e-06)
FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "s2_d5.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    out = dict(lumenairy=lib, python=sys.version.split()[0],
               numpy=np.__version__,
               closure_abs=ps._SLIVER_ATTRIB_CLOSURE,
               closure_frac=getattr(ps, "_SLIVER_CLOSURE_FRACTION", None),
               move_factor=ps._SLIVER_MOVE_FACTOR,
               trigger=ps._SLIVER_TRIGGER_BAR)
    out["degree_ladder_no_sliver"] = {
        str(deg): unguarded(gmr(0.0, deg))["worst"] - 1.0
        for deg in (6, 8, 10, 12, 14)}
    out["provably_passive"] = bool(ps._stack_provably_passive(gmr(0.0, 8)))
    ref = unguarded(gmr(0.0, 8))
    out["reference_super_unity"] = ref["worst"] - 1.0
    print("  degree ladder (sliver-free):",
          {k: f"{v:.5g}" for k, v in out["degree_ladder_no_sliver"].items()})

    rows = {}
    for d in DELTAS:
        st = gmr(d, 8)
        cur = unguarded(st)
        pre = prescribed(st)
        snp = snapped(st, pre["mf"])
        su = max(snp["worst"] - 1.0, 0.0)
        move = shared_move(cur, snp)
        drop = drop_factor(cur["worst"], su)
        refused, msg, _out, warns = guarded(gmr(d, 8))
        r = dict(
            super_unity=cur["worst"] - 1.0,
            super_unity_snapped=su,
            drop_factor=drop,
            move=move, w_wide=pre["w_wide"],
            move_ratio=move / pre["w_wide"],
            err_over_delta=shared_move(cur, ref, pol=1) / d,
            err_snapped_over_delta=shared_move(snp, ref, pol=1) / d,
            verdict_round2=verdict_round2(cur["worst"], su, move,
                                          pre["w_wide"]),
            verdict_round3={
                f"{f:g}": verdict_round3(cur["worst"], su, move,
                                         pre["w_wide"], f) for f in FRACS},
            library_refused=bool(refused),
            library_says_sliver=bool("NEAR-COINCIDENT-WALL SLIVER" in msg),
            library_names_min_feature=bool("pass min_feature=" in msg),
            truncation_note=any("is NOT what moved" in w for w in warns),
            says_min_feature_silences_nothing=any(
                "will silence nothing here" in w for w in warns),
            n_warn=len(warns))
        rows[f"{d:.6e}"] = r
        print(f"  delta={d:.4e}  R+T-1 {r['super_unity']:+.4e} -> snapped "
              f"{r['super_unity_snapped']:+.4e}  ({r['drop_factor']:.4g}x "
              f"drop)  move/w {r['move_ratio']:.4g}  err/delta "
              f"{r['err_over_delta']:.4g} -> {r['err_snapped_over_delta']:.4g}")
        print(f"      round2 {r['verdict_round2']:11s} round3 "
              f"{r['verdict_round3']}")
        print(f"      LIBRARY {'REFUSED' if refused else 'returned'}"
              f"  sliver-message {r['library_says_sliver']}"
              f"  truncation-note {r['truncation_note']}")
    out["rows"] = rows
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
