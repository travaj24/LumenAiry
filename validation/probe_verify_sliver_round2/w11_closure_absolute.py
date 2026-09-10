"""W11 -- the CLOSURE criterion is ABSOLUTE, not relative (defect D-5).

`_SLIVER_ATTRIB_CLOSURE` asks the snapped super-unity to fall below a FIXED
1e-5.  On any stack whose SLIVER-FREE truncation super-unity already sits
above that bar, the criterion can never be met -- however completely the snap
restores the answer.  The arbiter then says `truncation`, the solve RETURNS,
and the message tells the caller that raising ``min_feature`` "will silence
nothing here", which is the opposite of what is measured.

The fixture is a guided-mode-resonance grating in a dense-superstrate grazing
mount whose degree-8 truncation floor is 3.73e-05 -- between the closure bar
and the trigger.

    python w11_closure_absolute.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w6_resonant import gmr  # noqa: E402
from w_fixtures import prescribed, shared_move, snapped, unguarded  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

KW = dict(wl=9.3e-7, nsup=2.4, th=1.22, nsub=complex(1.45, 0.05))
DELTAS = (1.1943e-05, 6.8726e-06, 5.2134e-06, 3.0000e-06)


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w11_closure_absolute.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    out = dict(lumenairy=lib, python=sys.version.split()[0],
               numpy=np.__version__)
    # (i) the mount's truncation floor is PURE truncation: a clean degree
    #     ladder on the SLIVER-FREE stack
    out["degree_ladder_no_sliver"] = {
        str(deg): unguarded(gmr(0.0, deg, **KW))["worst"] - 1.0
        for deg in (6, 8, 10, 12, 14)}
    ref = unguarded(gmr(0.0, 8, **KW))
    out["reference_super_unity"] = ref["worst"] - 1.0
    out["provably_passive"] = bool(
        ps._stack_provably_passive(gmr(0.0, 8, **KW)))
    rows = {}
    for d in DELTAS:
        st = gmr(d, 8, **KW)
        cur = unguarded(st)
        pre = prescribed(st)
        snp = snapped(st, pre["mf"])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                st.solve()
                v = "returned"
            except ValueError as exc:
                v = ("REFUSED"
                     if "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                     else "raised")
        warns = [str(w.message) for w in rec]
        rows[f"{d:.6e}"] = dict(
            verdict=v,
            super_unity=cur["worst"] - 1.0,
            super_unity_snapped=max(snp["worst"] - 1.0, 0.0),
            drop_factor=(cur["worst"] - 1.0)
            / max(snp["worst"] - 1.0, 1e-300),
            move_ratio=shared_move(cur, snp) / pre["w_wide"],
            err_over_delta=shared_move(cur, ref, pol=1) / d,
            err_snapped_over_delta=shared_move(snp, ref, pol=1) / d,
            truncation_note=any("is NOT what moved" in w for w in warns),
            says_min_feature_silences_nothing=any(
                "will silence nothing here" in w for w in warns),
            closure_bar=ps._SLIVER_ATTRIB_CLOSURE,
            move_bar=ps._SLIVER_MOVE_FACTOR)
        r = rows[f"{d:.6e}"]
        print(f"  delta={d:.4e}  R+T-1 = {r['super_unity']:+.4e}"
              f" -> snapped {r['super_unity_snapped']:+.4e}"
              f" (a {r['drop_factor']:.0f}x drop, bar {ps._SLIVER_ATTRIB_CLOSURE:g})")
        print(f"      move/w_wide = {r['move_ratio']:.1f} (bar "
              f"{ps._SLIVER_MOVE_FACTOR:g});  err/delta = "
              f"{r['err_over_delta']:.1f};  SNAPPED err/delta = "
              f"{r['err_snapped_over_delta']:.3g}")
        print(f"      verdict {r['verdict']}, truncation note "
              f"{r['truncation_note']}, "
              f"'raising min_feature will silence nothing' "
              f"{r['says_min_feature_silences_nothing']}")
    out["rows"] = rows
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("  degree ladder (no sliver):", out["degree_ladder_no_sliver"])
    print("  provably passive:", out["provably_passive"])
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
