"""B3 / B4 -- the circular pillar through the LIBRARY, against the planner's
saved 3-D FEM oracle, and the staircase limit.

Fixture (plan 3.3): r = 0.36 = 0.3 p, eps 4 in air, period 1.2, height 0.5,
air over n = 1.45, lambda = 1, normal incidence.

  curved K M      -- one rung of the curved ladder (K = c3 | c5): per-order
                     R / T for both inputs, closure, the four-fold symmetry
                     te(m, n) - tm(n, m), the mapped disk area, node count
                     -> b4_circle_<K>_M<M>.json
  stair k M       -- one rung of the SHIPPED staggered PMM on the 4k-step
                     staircase of the planning probe (walls c +- r i / k, a
                     cell filled when its centre is inside the circle)
                     -> b4_circle_stair_k<k>_M<M>.json
  summary         -- reads every rung + the FEM oracle
                     (``../fem/summary.json``, provenance asserted) and the
                     planner's saved staircase / RCWA JSON -> b4_circle.json

The FEM oracle is E along y ('te', row 1 here); its own error bar is the
largest per-order max-deviation across its three meshes (8.3e-6).
"""
import os
import sys
import time

import _common as C
import numpy as np

ORD_FEM = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)],
           "0,1": [(0, 1), (0, -1)],
           "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}


def disk_area(cm, cells, nq=64):
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(nq)
    A = 0.0
    for sx, sy in cells:
        J1 = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
        J2 = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + J1 * xg
        V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + J2 * xg
        _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
        A += float(np.sum(np.outer(wg, wg) * (xu * yv - xv * yu))) * J1 * J2
    return A


def rung(kind, M):
    cm, eps = C.circle3() if kind == "c3" else C.circle5()
    cells = [tuple(int(v) for v in c) for c in np.argwhere(eps.real > 1)]
    t0 = time.perf_counter()
    o, R, T, _J = C.solve(cm, eps, M)
    t = time.perf_counter() - t0
    bx = C.TS.Basis1D(C.P, cm.u_walls, M)
    by = C.TS.Basis1D(C.P, cm.v_walls, M)
    nq = C.TS._stag_map_nodes(bx, by, cm, M)
    sym = 0.0
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            i = C.idx_of(o, [(m, n)])[0]
            j = C.idx_of(o, [(n, m)])[0]
            sym = max(sym, abs(R[1, i] - R[0, j]), abs(T[1, i] - T[0, j]))
    res = {"env": C.env_record(), "map": kind, "M": M,
           "dof": 2 * (cm.shape[0] * (M - 1)) ** 2, "nq": nq,
           "corner_cells": [list(map(int, k)) for k, _v in
                            C.TS._stag_map_singular_corners(cm).items()],
           "disk_area_rel_err": disk_area(cm, cells) / (np.pi * C.R_CIRC ** 2)
           - 1.0,
           "t": t, "vec": C.vec(o, R, T).tolist(),
           "te": C.table(o, R, T, 1), "tm": C.table(o, R, T, 0),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
           "sym_te_tm_transpose": float(sym)}
    print(kind, M, f"R00te={res['te']['0,0'][0]:.10f} T00te="
          f"{res['te']['0,0'][1]:.10f} clo={res['closure']:.1e} "
          f"sym={sym:.1e} nq={nq} t={t:.0f}s", flush=True)
    C.dump(f"b4_circle_{kind}_M{M}.json", res)


def stair_walls(k):
    c = C.P / 2
    inner = sorted([c - C.R_CIRC * i / k for i in range(1, k + 1)]
                   + [c + C.R_CIRC * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [C.P])
    n = len(w) - 1
    mid = 0.5 * (w[:-1] + w[1:])
    eps = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < C.R_CIRC ** 2:
                eps[i, j] = C.EPS_P
    return w, eps


def stair(k, M):
    w, eps = stair_walls(k)
    t0 = time.perf_counter()
    o, R, T = C.solve_walls(w[1:-1], w[1:-1], eps, M, max_pencil_dof=20000)
    res = {"env": C.env_record(), "k": k, "steps": 4 * k, "M": M,
           "walls": w.tolist(), "t": time.perf_counter() - t0,
           "vec": C.vec(o, R, T).tolist(), "te": C.table(o, R, T, 1)}
    print("stair", k, M, res["te"]["0,0"], f"t={res['t']:.0f}s", flush=True)
    C.dump(f"b4_circle_stair_k{k}_M{M}.json", res)


def fem_oracle():
    """The planner's saved NGSolve oracle, PROVENANCE ASSERTED: the three
    meshes it names, every propagating order present, R + T - 1 below
    1e-6."""
    path = os.path.join(C.HERE, "..", "fem", "summary.json")
    import json
    with open(path) as f:
        d = json.load(f)
    assert d["best_from"] == ["h1.0_e20 p4", "h0.8_e30 p4", "h1.0 p6"], d[
        "best_from"]
    assert abs(d["RplusT"]["value"] - 1.0) < 1e-6
    ref, dev = {}, {}
    for side in ("R", "T"):
        for key, rec in d[side].items():
            for mn in ORD_FEM[key]:
                ref[(side, mn)] = rec["value"]
                dev[(side, mn)] = rec["maxdev"]
    return ref, dev, d


def dist_fem(te_table, ref):
    out = 0.0
    for (side, (m, n)), v in ref.items():
        got = te_table[f"{m},{n}"][0 if side == "R" else 1]
        out = max(out, abs(got - v))
    return out


def summary():
    ref, dev, d = fem_oracle()
    spread = max(dev.values())
    res = {"env": C.env_record(), "fem_spread": spread,
           "fem_best_from": d["best_from"], "curved": {}, "stair": {}}
    for kind, ms in (("c3", range(6, 13)), ("c5", range(4, 9))):
        rows = []
        for M in ms:
            try:
                r = C.load(f"b4_circle_{kind}_M{M}.json")
            except FileNotFoundError:
                continue
            rows.append({"M": M, "dof": r["dof"], "nq": r["nq"],
                         "dist_fem": dist_fem(r["te"], ref),
                         "closure": r["closure"],
                         "sym": r["sym_te_tm_transpose"], "t": r["t"],
                         "area": r["disk_area_rel_err"], "vec": r["vec"]})
        for a, b in zip(rows[:-1], rows[1:]):
            a["d_next"] = float(np.max(np.abs(np.array(a["vec"])
                                              - np.array(b["vec"]))))
        # the planning probe's scratch solver on the same map and rung (plain
        # tensor rule 2M + 8, whitened eig): what the LIBRARY build changed
        p = C.load(f"p3_circle_curved{kind[1]}.json", sub="")
        prs = {r["M"]: r for r in p["runs"]}
        for r in rows:
            q = prs.get(r["M"])
            if q is not None:
                pv = np.array(q["vec_tm"][:9] + q["vec_te"][:9]
                              + q["vec_tm"][9:] + q["vec_te"][9:])
                r["vs_planner_scratch"] = float(np.max(np.abs(
                    np.array(r["vec"]) - pv)))
        res["curved"][kind] = rows
    # the two topologies' top rungs
    try:
        a = np.array(res["curved"]["c3"][-1]["vec"])
        b = np.array(res["curved"]["c5"][-1]["vec"])
        res["c3_top_vs_c5_top"] = float(np.max(np.abs(a - b)))
    except (KeyError, IndexError):
        pass
    for k in (1, 2, 4):
        for fn in sorted(os.listdir(C.HERE)):
            if fn.startswith(f"b4_circle_stair_k{k}_M"):
                r = C.load(fn)
                res["stair"].setdefault(str(k), []).append(
                    {"M": r["M"], "dist_fem": dist_fem(r["te"], ref),
                     "vec": r["vec"]})
    # the planner's saved staircases (the shipped solver on the same walls;
    # 'te' in the planning JSON is E along y as here)
    for k in (1, 2, 4):
        p = C.load(f"p3_circle_stair_k{k}.json", sub="")
        top = p["runs"][-1]
        res["stair"].setdefault(f"planner_k{k}", []).append(
            {"M": top["M"], "dist_fem": dist_fem(top["te"], ref)})
    for k, v in res["curved"].items():
        for r in v:
            r.pop("vec")
    C.dump("b4_circle.json", res)
    print(res)


def direction():
    """B4's direction cosines: each staircase step (shipped solver; the
    planner's saved staircase JSON -- the shipped code is byte-identical)
    against (target - previous staircase), target = the curved answer at a
    given rung or the FEM; 'te' tables, R and T of the nine orders."""
    ref, _dev, _d = fem_oracle()

    def tev(t):
        return np.array([t[f"{m},{n}"][k] for k in (0, 1)
                         for m, n in C.ORD9])
    fem = np.array([ref[("R" if k == 0 else "T", mn)] for k in (0, 1)
                    for mn in C.ORD9])
    st = {k: {r["M"]: tev(r["te"]) for r in
              C.load(f"p3_circle_stair_k{k}.json", sub="")["runs"]}
          for k in (1, 2, 4)}
    cur = {M: tev(C.load(f"b4_circle_c3_M{M}.json")["te"]) for M in (7, 10)}
    res = {"env": C.env_record(), "rows": []}
    for (ka, Ma), (kb, Mb), tgt, name in (
            ((1, 7), (2, 5), cur[7], "unit-test sizes vs curved c3 M=7"),
            ((1, 10), (2, 7), cur[10], "planner tops vs curved c3 M=10"),
            ((2, 7), (4, 4), cur[10], "k2 -> k4 vs curved c3 M=10"),
            ((1, 10), (4, 4), fem, "k1 -> k4 vs the FEM")):
        a, b = st[ka][Ma], st[kb][Mb]
        step, aim = b - a, tgt - a
        res["rows"].append({
            "pair": [[ka, Ma], [kb, Mb]], "target": name,
            "cos": float(step @ aim / np.linalg.norm(step)
                         / np.linalg.norm(aim)),
            "dist_a": float(np.max(np.abs(a - tgt))),
            "dist_b": float(np.max(np.abs(b - tgt)))})
        print(res["rows"][-1], flush=True)
    C.dump("b4_stair_direction.json", res)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "curved":
        rung(sys.argv[2], int(sys.argv[3]))
    elif mode == "stair":
        stair(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "summary":
        summary()
    elif mode == "direction":
        direction()
    else:
        raise SystemExit(mode)
