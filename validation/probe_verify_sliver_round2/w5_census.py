"""W5 -- the CENSUS, independently rebuilt.

  BOX   my own realistic under-converged staircase box (lossy substrate,
        dense superstrate, theta 1.2-1.45 rad, degree 6-10, wall steps
        0.36-3.6 nm), 648 configurations -- DIFFERENT specific values from the
        fix's, inside the same stated ranges.  Counts, per row:
          round 1  refuse iff screen + provably passive + R+T-1 > 1e-2
          round 2  what the LIBRARY does (the guarded solve, end to end)
        A FALSE POSITIVE is a refusal of a row the continuity rule calls
        RIGHT.  Also checks that every RETURNED row is BIT-identical to the
        unguarded answer.

  GRID  my own 660-row wrong/correct grid on the O-11 fixture (120 deltas x
        degrees 10/14/20, plus 60 deltas x degrees 8/10/12/14/16) -- the
        FALSE-NEGATIVE census, round 1 vs round 2, with the residual rows
        characterised (which half of the guard holds each out).

    python w5_census.py [out.json]
"""
import itertools
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lumenairy                                              # noqa: E402
from lumenairy.elements.pmm import PMMStack                   # noqa: E402
from lumenairy.elements.pmm import stack as ps                # noqa: E402
from w_fixtures import classify, prescribed, shared_move      # noqa: E402
from w_fixtures import snapped, unguarded                     # noqa: E402

BAR1 = 1.0e-2                    # round 1's refusal bar
TRIG = 1.0e-3                    # round 2's trigger
P, WL = 1.2e-6, 0.85e-6
A0, B0 = 0.27865, 0.62505
EH = 2.25
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12


def build(d, deg, nsub, nsup, th, nl, eps):
    st = PMMStack(P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=NO_SNAP, far_field_orders=31)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=[(A0 - dd, EH),
                                   (B0 + dd - (A0 - dd), eps),
                                   (1.0 - (B0 + dd), EH)])
    st.set_source(WL, theta=th)
    return st


def guarded_solve(st):
    """(refused, message, payload, warnings)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, _J = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return (False, "", dict(o=o[i], R=np.real(np.asarray(R))[:, i],
                            T=np.real(np.asarray(T))[:, i]),
            [str(w.message) for w in rec])


def round1_decision(st, worst):
    """Round 1's rule, evaluated with the shipped screen (which round 2 did
    not change): refuse iff the geometric screen fires on a provably passive
    stack AND the super-unity is above 1e-2."""
    return (ps._sliver_screen(st) is not None) and (worst - 1.0 > BAR1)


def box():
    rows = []
    t0 = time.time()
    refs = {}
    for nsub, nsup, th, deg, eps, nl in itertools.product(
            (1.45 + 0.08j, 2.0 + 0.35j, 3.4 + 1.7j), (2.4, 3.2),
            (1.22, 1.33, 1.44), (6, 8, 10), (10.5, 8.0), (2, 4)):
        key = (nsub, nsup, th, deg, eps, nl)
        if key not in refs:
            refs[key] = unguarded(build(0.0, deg, nsub, nsup, th, nl, eps))
        ref = refs[key]
        for d in (3e-3, 1e-3, 3e-4):
            st = build(d, deg, nsub, nsup, th, nl, eps)
            cur = unguarded(st)
            e = shared_move(cur, ref, pol=1)
            kind = classify(e, d)
            r1 = round1_decision(st, cur["worst"])
            refused, msg, out, warns = guarded_solve(
                build(d, deg, nsub, nsup, th, nl, eps))
            bitid = None
            if out is not None:
                bitid = bool(
                    np.array_equal(out["R"], cur["R"])
                    and np.array_equal(out["T"], cur["T"])
                    and np.array_equal(out["o"], cur["o"]))
            row = dict(nsub=str(nsub), nsup=nsup, th=th, deg=deg, eps=eps,
                       nl=nl, delta=d, err=e, err_over_d=e / d, kind=kind,
                       worst=cur["worst"], r1_refuse=bool(r1),
                       r2_refuse=bool(refused), bitid=bitid,
                       n_warn=len(warns),
                       truncation_note=any("is NOT what moved" in w
                                           for w in warns),
                       screen=ps._sliver_screen(st) is not None)
            pre = prescribed(st)
            if pre is not None and cur["worst"] - 1.0 > TRIG:
                snp = snapped(st, pre["mf"])
                row.update(su_snap=max(snp["worst"] - 1.0, 0.0),
                           move_ratio=shared_move(cur, snp) / pre["w_wide"],
                           w_wide=pre["w_wide"])
            rows.append(row)
    return rows, time.time() - t0


def grid():
    """The FALSE-NEGATIVE census: 120 deltas x 3 degrees + 60 x 5 = 660."""
    rows = []
    t0 = time.time()
    plans = ((np.geomspace(3e-3, 1e-6, 120), (10, 14, 20)),
             (np.geomspace(3e-5, 1e-6, 60), (8, 10, 12, 14, 16)))
    for deltas, degrees in plans:
        for deg in degrees:
            ref = unguarded(build(0.0, deg, 1.0, 1.0, 0.15, 2, 9.0))
            for d in deltas:
                st = build(float(d), deg, 1.0, 1.0, 0.15, 2, 9.0)
                cur = unguarded(st)
                e = shared_move(cur, ref, pol=1)
                kind = classify(e, float(d))
                r1 = round1_decision(st, cur["worst"])
                refused, _msg, _out, warns = guarded_solve(
                    build(float(d), deg, 1.0, 1.0, 0.15, 2, 9.0))
                row = dict(deg=deg, delta=float(d), err=e, err_over_d=e / d,
                           kind=kind, worst=cur["worst"], r1_refuse=bool(r1),
                           r2_refuse=bool(refused), n_warn=len(warns))
                pre = prescribed(st)
                if pre is not None:
                    row["w_wide"] = pre["w_wide"]
                    if cur["worst"] - 1.0 > 1e-9:
                        snp = snapped(st, pre["mf"])
                        row["su_snap"] = max(snp["worst"] - 1.0, 0.0)
                        row["move_ratio"] = (shared_move(cur, snp)
                                             / pre["w_wide"])
                rows.append(row)
    return rows, time.time() - t0


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w5_census.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    brows, bt = box()
    grows, gt = grid()

    def fp(rows, key):
        return [r for r in rows if r[key] and r["kind"] == "right"]

    def fn(rows, key):
        return [r for r in rows if (not r[key]) and r["kind"] == "wrong"]

    summary = dict(
        lumenairy=lib, python=sys.version.split()[0], numpy=np.__version__,
        box_n=len(brows), box_wall=bt, grid_n=len(grows), grid_wall=gt,
        box_right=sum(1 for r in brows if r["kind"] == "right"),
        box_grey=sum(1 for r in brows if r["kind"] == "grey"),
        box_wrong=sum(1 for r in brows if r["kind"] == "wrong"),
        box_fp_round1=len(fp(brows, "r1_refuse")),
        box_fp_round2=len(fp(brows, "r2_refuse")),
        box_refusals_round1=sum(1 for r in brows if r["r1_refuse"]),
        box_refusals_round2=sum(1 for r in brows if r["r2_refuse"]),
        box_grey_refused_round2=sum(1 for r in brows if r["r2_refuse"]
                                    and r["kind"] == "grey"),
        box_bitid_returned=sum(1 for r in brows if r["bitid"] is True),
        box_bitid_broken=sum(1 for r in brows if r["bitid"] is False),
        box_truncation_notes=sum(1 for r in brows if r["truncation_note"]),
        grid_right=sum(1 for r in grows if r["kind"] == "right"),
        grid_grey=sum(1 for r in grows if r["kind"] == "grey"),
        grid_wrong=sum(1 for r in grows if r["kind"] == "wrong"),
        grid_fn_round1=len(fn(grows, "r1_refuse")),
        grid_fn_round2=len(fn(grows, "r2_refuse")),
        grid_fp_round1=len(fp(grows, "r1_refuse")),
        grid_fp_round2=len(fp(grows, "r2_refuse")),
        grid_wrong_subunity=sum(1 for r in grows if r["kind"] == "wrong"
                                and r["worst"] <= 1.0),
        grid_wrong_min_superunity=min(
            (r["worst"] - 1.0 for r in grows if r["kind"] == "wrong"),
            default=None),
        grid_correct_envelope=max(
            (abs(r["worst"] - 1.0) for r in grows if r["kind"] == "right"),
            default=None),
        box_fp1_err_over_d=[round(r["err_over_d"], 4)
                            for r in fp(brows, "r1_refuse")],
        box_fp1_worst=[round(r["worst"], 6) for r in fp(brows, "r1_refuse")],
    )
    summary["grid_fn_round2_rows"] = sorted(
        (dict(deg=r["deg"], delta=r["delta"], err=r["err"],
              err_over_d=r["err_over_d"], worst=r["worst"],
              su_snap=r.get("su_snap"), move_ratio=r.get("move_ratio"),
              above_trigger=bool(r["worst"] - 1.0 > TRIG))
         for r in fn(grows, "r2_refuse")),
        key=lambda x: -x["err_over_d"])[:40]
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, box=brows, grid=grows), fh, indent=1)
    for k, v in summary.items():
        if k != "grid_fn_round2_rows":
            print(f"  {k:28s} {v}")
    print("  --- the round-2 false negatives (worst first) ---")
    for r in summary["grid_fn_round2_rows"][:12]:
        print(f"    deg {r['deg']:3d} delta {r['delta']:.4e} "
              f"err/d {r['err_over_d']:9.1f} R+T-1 {r['worst'] - 1:+.4e} "
              f"su_snap {r['su_snap']} move {r['move_ratio']} "
              f"trig {r['above_trigger']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
