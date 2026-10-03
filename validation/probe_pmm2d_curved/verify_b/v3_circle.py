"""V3 -- the circular pillar on THIS verifier's maps, at the build's radius
(r = 0.36 = 0.3 p, against the planner's saved FEM) and at two NEW radii
(r = 0.48 = 0.4 p and r = 0.24 = 0.2 p, against this verifier's own NGSolve
runs of the planner's runner, ``fem_circle_r.py``); plus the shipped
solver's staircases of the same circles.

  curved <c3|c5> <r> <M>   -- the curved map (c5 = 5 x 5, inner 0.6)
  stair <k> <r> <M>        -- the 4k-step staircase (walls c +- r i / k, a
                              cell filled when its centre is inside)
Output: v3_<kind>_r<r>_M<M>.json with the 'te' (E along y) and 'tm' rows of
R / T on the nine orders, closure, and the four-fold symmetry residual.
"""
import sys
import time

import _vcommon as C
import numpy as np


def rec(name, o, R, T, extra):
    i = C.idx(o)
    sym = 0.0
    for (m, n) in C.ORD9:
        a, b = C.idx(o, [(m, n), (n, m)])
        sym = max(sym, abs(R[1, a] - R[0, b]), abs(T[1, a] - T[0, b]))
    out = {"te_R": R[1, i], "te_T": T[1, i], "tm_R": R[0, i],
           "tm_T": T[0, i],
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
           "sym_te_tm": float(sym), "orders": C.ORD9}
    out.update(extra)
    C.dump(name, out)


def curved(kind, r, M):
    t0 = time.time()
    cm, eps = (C.vcircle3(r) if kind == "c3" else C.vcircle5(r, 0.6))
    o, R, T, st, warns = C.solve_map(cm, eps, M)
    rec(f"v3_{kind}_r{r}_M{M}.json", o, R, T,
        {"wall_s": time.time() - t0, "warnings": warns, "M": M, "r": r})


def stair(k, r, M):
    t0 = time.time()
    c = C.P / 2
    inner = sorted([c - r * i / k for i in range(1, k + 1)]
                   + [c + r * i / k for i in range(1, k + 1)])
    w = np.array([0.0] + inner + [C.P])
    mid = 0.5 * (w[:-1] + w[1:])
    n = mid.size
    eps = np.ones((n, n), complex)
    for i in range(n):
        for j in range(n):
            if (mid[i] - c) ** 2 + (mid[j] - c) ** 2 < r * r:
                eps[i, j] = C.EPS_P
    o, R, T = C.solve_walls(w[1:-1], w[1:-1], eps, M)
    rec(f"v3_stair{k}_r{r}_M{M}.json", o, R, T,
        {"wall_s": time.time() - t0, "M": M, "r": r, "k": k})


if __name__ == "__main__":
    if sys.argv[1] == "curved":
        curved(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    else:
        stair(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
