"""W2 -- the four arbiter bars, RE-DERIVED on my own dense grids.

FIVE fixtures (four of them geometry the fix never tuned on, spanning measured
continuity slopes 0.47 .. 31.4), 150 log-spaced ``delta`` each, three degrees
= 2,250 rows.  Per row, with the guard OFF
throughout so nothing the library decides can feed back into the measurement:

  ref        the exact ``delta -> 0`` solve of the SAME stack
  err        max |dR|,|dT| vs ref, polarization 1 (the campaign's convention)
  err_both   the same over BOTH polarizations (what the library compares)
  kind       'right' (err <= 10 delta) / 'grey' / 'wrong' (err > 100 delta)
  worst      max(R+T) of the unguarded solve
  su_snap    max(R+T)-1 of the re-solve on the PRESCRIBED min_feature grid
  move       max |dR|,|dT| between those two solves, BOTH polarizations
  move_p1    the same on polarization 1 only
  w_wide     the widest manufactured cell (fraction of a period)

From which:
  * the CORRECT population's super-unity envelope, against the 1e-3 trigger
  * the snapped-super-unity split at 1e-5 (WRONG vs TRUNCATION)
  * the move/w_wide split at 100 (CORRECT vs WRONG)
  * how many rows the two builds disagree about

    python w2_populations.py [out.json] [--fast]
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lumenairy                                              # noqa: E402
from w_fixtures import (classify, prescribed, shared_move,     # noqa: E402
                        snapped, unguarded, wbuild)

TRIG = 1.0e-3

# Only ``O11`` reuses the fix's own geometry (so its census can be compared row
# for row); the other four are mine, chosen to span the continuity slope
# ``dR/dx`` -- the quantity the round-2 report's open item R2-D says the MOVE
# criterion silently assumes is O(1).
FIXTURES = {
    # measured continuity slope err/delta in the SMOOTH regime (degree 14,
    # delta 3e-3..1e-4, this build 2026-09-11) in the comment
    "C_nir": dict(period=1.05e-6, wl=0.98e-6, th=0.42, a0=0.2350, b0=0.6650,
                  eh=2.10, ep=4.00, dz=0.15e-6),                    # 0.47
    "O11": dict(period=1.2e-6, wl=0.85e-6, th=0.15, a0=0.27865, b0=0.62505,
                eh=2.25, ep=9.0, dz=0.32e-6 / 4),                   # 1.15
    "B_vis": dict(period=0.74e-6, wl=0.53e-6, th=0.31, a0=0.19137,
                  b0=0.71429, eh=1.96, ep=6.25, dz=0.06e-6),        # 3.15
    "D_tele": dict(period=1.8e-6, wl=1.31e-6, th=0.11, a0=0.31250,
                   b0=0.58750, eh=2.10, ep=11.7, dz=0.10e-6),       # 10.2
    "S_steep": dict(period=0.74e-6, wl=0.53e-6, th=0.31, a0=0.19137,
                    b0=0.71429, eh=1.96, ep=6.25, dz=0.21e-6 / 3),  # 31.4
}
DEGREES = (10, 14, 18)


def rows_for(name, kw, degrees, deltas):
    out = []
    for deg in degrees:
        ref = unguarded(wbuild(0.0, deg, **kw))
        for d in deltas:
            st = wbuild(float(d), deg, **kw)
            cur = unguarded(st)
            e1 = shared_move(cur, ref, pol=1)
            eb = shared_move(cur, ref)
            row = dict(fix=name, deg=deg, delta=float(d), err=e1, err_both=eb,
                       kind=classify(e1, float(d)), worst=cur["worst"])
            pre = prescribed(st)
            row["has_sliver"] = pre is not None
            if pre is not None and cur["worst"] - 1.0 > TRIG:
                snp = snapped(st, pre["mf"])
                row.update(su_snap=max(snp["worst"] - 1.0, 0.0),
                           move=shared_move(cur, snp),
                           move_p1=shared_move(cur, snp, pol=1),
                           w_wide=pre["w_wide"], own=pre["own"],
                           ratio=pre["own"] / pre["w_narrow"],
                           arbitrated=True)
            else:
                row["arbitrated"] = False
            out.append(row)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    out_path = args[0] if args else os.path.join(HERE, "w2_populations.json")
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    n_d = 20 if fast else 150
    degs = (10, 14) if fast else DEGREES
    deltas = np.geomspace(3e-3, 1e-6, n_d)
    t0 = time.time()
    rows = []
    for name, kw in FIXTURES.items():
        t1 = time.time()
        rows += rows_for(name, kw, degs, deltas)
        print(f"  {name}: {len(rows)} rows so far, {time.time() - t1:.1f} s")

    def sel(pred):
        return [r for r in rows if pred(r)]

    corr = sel(lambda r: r["kind"] == "right")
    wrong = sel(lambda r: r["kind"] == "wrong")
    grey = sel(lambda r: r["kind"] == "grey")
    arb = sel(lambda r: r["arbitrated"])
    arb_w = [r for r in arb if r["kind"] == "wrong"]
    arb_r = [r for r in arb if r["kind"] == "right"]
    arb_g = [r for r in arb if r["kind"] == "grey"]

    def rng(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return (min(v), max(v)) if v else (None, None)

    summary = dict(
        lumenairy=lib, python=sys.version.split()[0], numpy=np.__version__,
        n_rows=len(rows), n_correct=len(corr), n_wrong=len(wrong),
        n_grey=len(grey), n_arbitrated=len(arb),
        # (a) the TRIGGER's premise
        correct_superunity_envelope=(
            max(abs(r["worst"] - 1.0) for r in corr) if corr else None),
        correct_superunity_signed_max=(
            max(r["worst"] - 1.0 for r in corr) if corr else None),
        wrong_superunity_min=(
            min(r["worst"] - 1.0 for r in wrong) if wrong else None),
        wrong_superunity_min_abs=(
            min(abs(r["worst"] - 1.0) for r in wrong) if wrong else None),
        # how far a WRONG row's super-unity reaches DOWN, i.e. the trigger's
        # blind band -- and how many wrong rows are inside it
        n_wrong_below_trigger=sum(1 for r in wrong
                                  if r["worst"] - 1.0 <= TRIG),
        n_wrong_subunity=sum(1 for r in wrong if r["worst"] <= 1.0),
        # (b) the CLOSURE bar
        su_snap_wrong=rng(arb_w, "su_snap"),
        su_snap_right=rng(arb_r, "su_snap"),
        su_snap_grey=rng(arb_g, "su_snap"),
        # (c) the MOVE bar
        move_ratio_wrong=rng([dict(m=r["move"] / r["w_wide"]) for r in arb_w],
                             "m"),
        move_ratio_right=rng([dict(m=r["move"] / r["w_wide"]) for r in arb_r],
                             "m"),
        move_ratio_grey=rng([dict(m=r["move"] / r["w_wide"]) for r in arb_g],
                            "m"),
        move_ratio_p1_wrong=rng([dict(m=r["move_p1"] / r["w_wide"])
                                 for r in arb_w], "m"),
        move_ratio_p1_right=rng([dict(m=r["move_p1"] / r["w_wide"])
                                 for r in arb_r], "m"),
        wall=time.time() - t0)
    # the per-fixture envelope, so a family claim is not one fixture's
    for name in FIXTURES:
        c = [r for r in corr if r["fix"] == name]
        w = [r for r in wrong if r["fix"] == name]
        summary[f"env_{name}"] = (max(abs(r["worst"] - 1.0) for r in c)
                                  if c else None)
        summary[f"nc_{name}"], summary[f"nw_{name}"] = len(c), len(w)
        summary[f"wmin_{name}"] = (min(r["worst"] - 1.0 for r in w)
                                   if w else None)
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, rows=rows), fh, indent=1)
    for k, v in summary.items():
        print(f"  {k:34s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
