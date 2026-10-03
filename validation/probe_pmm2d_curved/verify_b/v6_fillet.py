"""V6 -- the fillet ladder on THIS verifier's fillet map (``vfillet``, built
from Arc.through on vertex images), the sharp-square reference, and the
r -> 0 question.

  fillet <ratio> <M> [grade]  ratio = r / side (side 0.6); 'grade' adds two
                              straight walls INSIDE the pillar at b + g and
                              P - b - g with g = 2 r (a grading of the big
                              pillar cell toward the fillet's centre) -- if
                              the small fillets converge slowly because the
                              BIG neighbouring cells must resolve a field that
                              varies on the scale r, grading fixes it
  square <M> [grade]          the sharp square on the shipped solver, 3 x 3
                              walls; 'grade' adds walls at lo +- g, hi +- g
                              (g = 0.04) toward the corner singularities
Output: v6_fillet_<ratio>_M<M>[_grade].json, v6_square_M<M>[_grade].json
(R00 / T00 of input 'te' and the full nine-order vector, warnings).
"""
import sys
import time

import _vcommon as C
import numpy as np

SIDE = 0.6


def out(name, o, R, T, extra):
    i0 = C.idx(o, [(0, 0)])[0]
    res = {"R00": float(R[1, i0]), "T00": float(T[1, i0]),
           "vec": C.vec(o, R, T),
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))}
    res.update(extra)
    C.dump(name, res)


def fillet(ratio, M, grade=False):
    t0 = time.time()
    rf = ratio * SIDE
    ex = ()
    if grade:
        b = C.P / 2 - SIDE / 2 + rf
        g = 2 * rf
        ex = (b + g, C.P - b - g)
    cm, eps = C.vfillet(SIDE, rf, extra=ex)
    o, R, T, st, w = C.solve_map(cm, eps, M)
    out(f"v6_fillet_{ratio}_M{M}{'_grade' if grade else ''}.json", o, R, T,
        {"ratio": ratio, "M": M, "grade": grade, "walls": cm.u_bounds,
         "warnings": w, "wall_s": time.time() - t0,
         "nq": None})


def square(M, grade=False):
    t0 = time.time()
    lo, hi = C.P / 2 - SIDE / 2, C.P / 2 + SIDE / 2
    if grade:
        g = 0.04
        w = np.array([lo - g, lo, lo + g, hi - g, hi, hi + g])
    else:
        w = np.array([lo, hi])
    n = w.size + 1
    full = np.concatenate([[0.0], w, [C.P]])
    mid = 0.5 * (full[:-1] + full[1:])
    eps = np.ones((n, n), complex)
    inside = (mid > lo) & (mid < hi)
    eps[np.ix_(inside, inside)] = C.EPS_P
    o, R, T = C.solve_walls(w, w, eps, M)
    out(f"v6_square_M{M}{'_grade' if grade else ''}.json", o, R, T,
        {"M": M, "grade": grade, "wall_s": time.time() - t0})


if __name__ == "__main__":
    g = len(sys.argv) > 4 and sys.argv[4] == "grade"
    if sys.argv[1] == "fillet":
        fillet(float(sys.argv[2]), int(sys.argv[3]), g)
    else:
        square(int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "grade")
