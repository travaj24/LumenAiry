"""B6 / B7 -- fillets through the LIBRARY: the square pillar (side 0.6, eps
4, period 1.2, height 0.5, air over n = 1.45) with its corners rounded to
r / side in {0.01, 0.02, 0.05, 0.1, 0.2} on the 5 x 5 fillet map, the sharp
square (r = 0, the shipped solver on the 3 x 3 walls 0.3 / 0.9), and the
SAME-AREA square of each fillet (the shipped solver, walls at the
equal-area half side) -- the surrogate B7 refutes.

  fillet <ratio> <M>   -> b6_fillet_r<ratio>_M<M>.json
  square <M>           -> b6_square_M<M>.json   (r = 0)
  eqarea <ratio> <M>   -> b6_eqarea_r<ratio>_M<M>.json
  summary              -> b6_fillet.json

R00 / T00 are polarization-independent for these four-fold-symmetric
pillars (both rows recorded; the summary reads row 1, 'te').
"""
import os
import sys
import time

import _common as C
import numpy as np

SIDE = 0.6


def rec(name, o, R, T, extra):
    i0 = C.idx_of(o, [(0, 0)])[0]
    res = {"env": C.env_record(), **extra,
           "R00": [float(R[0, i0]), float(R[1, i0])],
           "T00": [float(T[0, i0]), float(T[1, i0])],
           "vec": C.vec(o, R, T).tolist(),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))}
    print(name, extra, "R00", res["R00"][1], "T00", res["T00"][1],
          f"clo={res['closure']:.1e}", flush=True)
    C.dump(name, res)


def fillet(ratio, M):
    cm, eps = C.fillet5(ratio, SIDE)
    t0 = time.perf_counter()
    o, R, T, _J = C.solve(cm, eps, M)
    rec(f"b6_fillet_r{ratio}_M{M}.json", o, R, T,
        {"ratio": ratio, "M": M, "t": time.perf_counter() - t0,
         "walls": cm.u_bounds.tolist(),
         "corner_cells": sorted(map(list, C.TS._stag_map_singular_corners(
             cm)))})


def square(M, half=SIDE / 2, name=None, extra=None):
    w = np.array([C.P / 2 - half, C.P / 2 + half])
    eps = np.ones((3, 3), complex)
    eps[1, 1] = C.EPS_P
    t0 = time.perf_counter()
    o, R, T = C.solve_walls(w, w, eps, M)
    rec(name or f"b6_square_M{M}.json", o, R, T,
        {"M": M, "half": half, "t": time.perf_counter() - t0,
         **(extra or {})})


def eqarea(ratio, M):
    rf = ratio * SIDE
    s = np.sqrt(SIDE ** 2 - (4 - np.pi) * rf ** 2)
    square(M, s / 2, f"b6_eqarea_r{ratio}_M{M}.json",
           {"ratio": ratio, "equal_area_side": float(s)})


def summary():
    res = {"env": C.env_record(), "fillet": {}, "square": [], "eqarea": {}}
    for fn in sorted(os.listdir(C.HERE)):
        if not fn.endswith(".json"):
            continue
        if fn.startswith("b6_fillet_r"):
            r = C.load(fn)
            res["fillet"].setdefault(str(r["ratio"]), []).append(
                {k: r[k] for k in ("M", "R00", "T00", "closure", "t")}
                | {"vec": r["vec"]})
        elif fn.startswith("b6_square_M"):
            r = C.load(fn)
            res["square"].append({k: r[k] for k in ("M", "R00", "T00",
                                                    "closure", "t")}
                                 | {"vec": r["vec"]})
        elif fn.startswith("b6_eqarea_r"):
            r = C.load(fn)
            res["eqarea"].setdefault(str(r["ratio"]), []).append(
                {k: r[k] for k in ("M", "R00", "T00", "equal_area_side")})
    for group in [res["square"]] + list(res["fillet"].values()):
        group.sort(key=lambda r: r["M"])
        for a, b in zip(group[:-1], group[1:]):
            a["d_next"] = float(np.max(np.abs(np.array(a["vec"])
                                              - np.array(b["vec"]))))
        for r in group:
            r.pop("vec")
    # the r -> 0 limit: each fillet's TOP rung against the sharp square's
    # top rung, and the constant of a fit T00(r) - T00(0) = c + a r^2 + b r^3
    # through r / side = 0.05, 0.1, 0.2 (c must vanish to the convergence)
    if res["square"] and res["fillet"]:
        sq = res["square"][-1]
        lim = {"square_top_M": sq["M"], "rows": []}
        for k, v in sorted(res["fillet"].items(), key=lambda z: float(z[0])):
            top = v[-1]
            prev = v[-2] if len(v) > 1 else None
            lim["rows"].append({
                "ratio": float(k), "top_M": top["M"],
                "dR00": top["R00"][1] - sq["R00"][1],
                "dT00": top["T00"][1] - sq["T00"][1],
                "own_T00_rung_change": (abs(top["T00"][1] - prev["T00"][1])
                                        if prev else None),
                "own_R00_rung_change": (abs(top["R00"][1] - prev["R00"][1])
                                        if prev else None)})
        sel = [r for r in lim["rows"] if r["ratio"] in (0.05, 0.1, 0.2)]
        if len(sel) == 3:
            rr = np.array([r["ratio"] for r in sel])
            for q in ("dT00", "dR00"):
                y = np.array([r[q] for r in sel])
                A = np.stack([np.ones(3), rr ** 2, rr ** 3], 1)
                lim[f"fit_{q}_c_a_b"] = np.linalg.solve(A, y).tolist()
        res["limit"] = lim
    C.dump("b6_fillet.json", res)
    print(res)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fillet":
        fillet(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "square":
        square(int(sys.argv[2]))
    elif mode == "eqarea":
        eqarea(float(sys.argv[2]), int(sys.argv[3]))
    elif mode == "summary":
        summary()
    else:
        raise SystemExit(mode)
