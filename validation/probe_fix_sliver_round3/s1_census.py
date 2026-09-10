"""S1 -- the two censuses, re-run for the round-3 closure.

  BOX   the 648-configuration realistic staircase box (lossy substrate, dense
        superstrate, theta 1.22-1.44 rad, degree 6-10, wall steps 0.36-3.6 nm
        on a 1.2 um period).  Every configuration in it is CORRECT by the
        campaign's continuity rule, so any refusal here is a FALSE POSITIVE.
        This is the population the relative closure must not disturb, and the
        population whose arbitrated rows set the DROP factor's lower envelope.

  GRID  the 660-row false-negative grid on the O-11 fixture (120 log deltas x
        degrees 10/14/20, plus 60 log deltas x degrees 8/10/12/14/16).  Its
        WRONG rows set the DROP factor's upper envelope.

Per row the probe records the unguarded solve, the solve on the PRESCRIBED
``min_feature`` grid, the two arbiter quantities, the DROP factor, what the
LIBRARY actually decided, and whether a returned answer is bit-identical to
the unguarded one.  The verdicts are then scored ANALYTICALLY under round 2
and under the candidate round-3 criterion at five fractions -- so one run
gives both arms, and the analytic round-2 column can be checked against the
library on a build that still carries round 2.

    python s1_census.py [out.json] [--fast]
"""
import itertools
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from f_fixtures import (  # noqa: E402
    TRIG,
    cbuild,
    classify,
    drop_factor,
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

BAR1 = 1.0e-2                    # round 1's refusal bar
FRACS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3)


def round1_decision(st, worst):
    """Round 1's rule with the shipped screen (which rounds 2 and 3 do not
    change): refuse iff the screen fires on a provably passive stack AND the
    super-unity is above 1e-2."""
    return (ps._sliver_screen(st) is not None) and (worst - 1.0 > BAR1)


def _score(st, cur, ref, d):
    """The row's measured quantities, with both analytic verdicts."""
    e = shared_move(cur, ref, pol=1)
    row = dict(delta=float(d), err=e, err_over_d=e / d,
               kind=classify(e, float(d)), worst=cur["worst"],
               r1_refuse=bool(round1_decision(st, cur["worst"])))
    pre = prescribed(st)
    row["screen"] = pre is not None
    row["arbitrated"] = bool(pre is not None and cur["worst"] - 1.0 > TRIG)
    if row["arbitrated"]:
        snp = snapped(st, pre["mf"])
        su = max(snp["worst"] - 1.0, 0.0)
        move = shared_move(cur, snp)
        row.update(su_snap=su, move=move, w_wide=pre["w_wide"],
                   move_ratio=move / pre["w_wide"],
                   drop=drop_factor(cur["worst"], su),
                   err_snapped_over_d=shared_move(snp, ref, pol=1) / d,
                   v2=verdict_round2(cur["worst"], su, move, pre["w_wide"]),
                   v3={f"{f:g}": verdict_round3(cur["worst"], su, move,
                                                pre["w_wide"], f)
                       for f in FRACS})
    return row


def box(fast=False):
    rows, t0, refs = [], time.time(), {}
    subs = ((1.45 + 0.08j, 2.0 + 0.35j, 3.4 + 1.7j) if not fast
            else (1.45 + 0.08j,))
    degs = (6, 8, 10) if not fast else (8,)
    for nsub, nsup, th, deg, eps, nl in itertools.product(
            subs, (2.4, 3.2), (1.22, 1.33, 1.44), degs, (10.5, 8.0), (2, 4)):
        key = (nsub, nsup, th, deg, eps, nl)
        if key not in refs:
            refs[key] = unguarded(cbuild(0.0, deg, nsub, nsup, th, nl, eps))
        ref = refs[key]
        for d in (3e-3, 1e-3, 3e-4):
            st = cbuild(d, deg, nsub, nsup, th, nl, eps)
            cur = unguarded(st)
            row = _score(st, cur, ref, d)
            refused, msg, out, warns = guarded(
                cbuild(d, deg, nsub, nsup, th, nl, eps))
            row.update(nsub=str(nsub), nsup=nsup, th=th, deg=deg, eps=eps,
                       nl=nl, lib_refuse=bool(refused), n_warn=len(warns),
                       lib_sliver_msg=bool("NEAR-COINCIDENT-WALL SLIVER"
                                           in msg),
                       truncation_note=any("is NOT what moved" in w
                                           for w in warns),
                       bitid=(None if out is None else
                              bool(np.array_equal(out["R"], cur["R"])
                                   and np.array_equal(out["T"], cur["T"])
                                   and np.array_equal(out["o"], cur["o"]))))
            rows.append(row)
    return rows, time.time() - t0


def grid(fast=False):
    rows, t0 = [], time.time()
    if fast:
        plans = ((np.geomspace(3e-3, 1e-6, 20), (14,)),)
    else:
        plans = ((np.geomspace(3e-3, 1e-6, 120), (10, 14, 20)),
                 (np.geomspace(3e-5, 1e-6, 60), (8, 10, 12, 14, 16)))
    for deltas, degrees in plans:
        for deg in degrees:
            ref = unguarded(cbuild(0.0, deg, 1.0, 1.0, 0.15, 2, 9.0))
            for d in deltas:
                st = cbuild(float(d), deg, 1.0, 1.0, 0.15, 2, 9.0)
                cur = unguarded(st)
                row = _score(st, cur, ref, float(d))
                refused, msg, _out, warns = guarded(
                    cbuild(float(d), deg, 1.0, 1.0, 0.15, 2, 9.0))
                row.update(deg=deg, lib_refuse=bool(refused),
                           n_warn=len(warns),
                           lib_sliver_msg=bool("NEAR-COINCIDENT-WALL SLIVER"
                                               in msg),
                           truncation_note=any("is NOT what moved" in w
                                               for w in warns))
                rows.append(row)
    return rows, time.time() - t0


def _envelope(rows, pred, key="drop", how=max):
    vs = [r[key] for r in rows
          if r.get("arbitrated") and pred(r) and np.isfinite(r.get(key, np.nan))]
    return how(vs) if vs else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    out_path = args[0] if args else os.path.join(HERE, "s1_census.json")
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    brows, bt = box(fast)
    print(f"  box: {len(brows)} rows, {bt:.1f} s")
    grows, gt = grid(fast)
    print(f"  grid: {len(grows)} rows, {gt:.1f} s")

    def fp(rows, key):
        return [r for r in rows if r[key] and r["kind"] == "right"]

    def fn(rows, key):
        return [r for r in rows if (not r[key]) and r["kind"] == "wrong"]

    barb = [r for r in brows if r["arbitrated"]]
    garb = [r for r in grows if r["arbitrated"]]
    s = dict(
        lumenairy=lib, python=sys.version.split()[0], numpy=np.__version__,
        closure_abs=ps._SLIVER_ATTRIB_CLOSURE,
        closure_frac=getattr(ps, "_SLIVER_CLOSURE_FRACTION", None),
        box_n=len(brows), box_wall=bt, grid_n=len(grows), grid_wall=gt,
        box_right=sum(1 for r in brows if r["kind"] == "right"),
        box_grey=sum(1 for r in brows if r["kind"] == "grey"),
        box_wrong=sum(1 for r in brows if r["kind"] == "wrong"),
        box_arbitrated=len(barb),
        box_fp_round1=len(fp(brows, "r1_refuse")),
        box_fp_library=len(fp(brows, "lib_refuse")),
        box_refusals_round1=sum(1 for r in brows if r["r1_refuse"]),
        box_refusals_library=sum(1 for r in brows if r["lib_refuse"]),
        box_bitid_returned=sum(1 for r in brows if r["bitid"] is True),
        box_bitid_broken=sum(1 for r in brows if r["bitid"] is False),
        box_truncation_notes=sum(1 for r in brows if r["truncation_note"]),
        box_arb_right_drop_max=_envelope(brows, lambda r: r["kind"] == "right"),
        box_arb_right_drop_min=_envelope(brows, lambda r: r["kind"] == "right",
                                         how=min),
        box_arb_right_move_max=_envelope(brows, lambda r: r["kind"] == "right",
                                         key="move_ratio"),
        box_arb_right_su_min=_envelope(brows, lambda r: r["kind"] == "right",
                                       key="su_snap", how=min),
        grid_right=sum(1 for r in grows if r["kind"] == "right"),
        grid_grey=sum(1 for r in grows if r["kind"] == "grey"),
        grid_wrong=sum(1 for r in grows if r["kind"] == "wrong"),
        grid_arbitrated=len(garb),
        grid_arb_wrong_n=sum(1 for r in garb if r["kind"] == "wrong"),
        grid_arb_wrong_drop_min=_envelope(grows,
                                          lambda r: r["kind"] == "wrong",
                                          how=min),
        grid_fn_round1=len(fn(grows, "r1_refuse")),
        grid_fn_library=len(fn(grows, "lib_refuse")),
        grid_fp_library=len(fp(grows, "lib_refuse")),
    )
    # analytic verdicts vs what the library did, on the arbitrated rows
    for tag, rs in (("box", barb), ("grid", garb)):
        s[f"{tag}_v2_sliver"] = sum(1 for r in rs if r["v2"] == "sliver")
        s[f"{tag}_lib_refuse_arb"] = sum(1 for r in rs if r["lib_refuse"])
        s[f"{tag}_v2_matches_library"] = sum(
            1 for r in rs if (r["v2"] == "sliver") == bool(r["lib_refuse"]))
        for f in FRACS:
            k = f"{f:g}"
            s[f"{tag}_v3_sliver_{k}"] = sum(
                1 for r in rs if r["v3"][k] == "sliver")
            s[f"{tag}_v3_fp_{k}"] = sum(
                1 for r in rs if r["v3"][k] == "sliver" and r["kind"] == "right")
            s[f"{tag}_v3_matches_library_{k}"] = sum(
                1 for r in rs
                if (r["v3"][k] == "sliver") == bool(r["lib_refuse"]))
    s["grid_fn_library_rows"] = sorted(
        (dict(deg=r["deg"], delta=r["delta"], err=r["err"],
              err_over_d=r["err_over_d"], worst=r["worst"],
              su_snap=r.get("su_snap"), move_ratio=r.get("move_ratio"),
              drop=r.get("drop"), arbitrated=r["arbitrated"],
              above_trigger=bool(r["worst"] - 1.0 > TRIG))
         for r in fn(grows, "lib_refuse")),
        key=lambda x: -x["err_over_d"])[:40]
    with open(out_path, "w") as fh:
        json.dump(dict(summary=s, box=brows, grid=grows), fh, indent=1)
    for k, v in s.items():
        if k != "grid_fn_library_rows":
            print(f"  {k:32s} {v}")
    print("  --- the library's false negatives on the grid (worst first) ---")
    for r in s["grid_fn_library_rows"][:12]:
        print(f"    deg {r['deg']:3d} delta {r['delta']:.4e} err/d "
              f"{r['err_over_d']:9.1f} R+T-1 {r['worst'] - 1:+.4e} "
              f"su_snap {r['su_snap']} move {r['move_ratio']} "
              f"drop {r['drop']} trig {r['above_trigger']}")
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
