"""V11 -- measurements behind the verifier's decision-test bars: the
four-fold symmetry residual of the circle vs a 1e-4 elliptic vertex offset
(mutant m5), and the fail-before staircase at the second FEM radius.
Output: v11_bars_<build>.json"""
import _vcommon as C
import numpy as np
import v3_circle as V3  # noqa: F401

res = {}


def sym(o, R, T):
    s = 0.0
    for (m, n) in C.ORD9:
        a, b = C.idx(o, [(m, n), (n, m)])
        s = max(s, abs(R[1, a] - R[0, b]), abs(T[1, a] - T[0, b]))
    return float(s)


r = 0.36
for tag, (cm, eps) in (("circle", C.vcircle3(r)),
                       ("ellipse_1e-4", C.vellipse3(r * (1 + 1e-4), r)),
                       ("ellipse_1e-6", C.vellipse3(r * (1 + 1e-6), r))):
    o, R, T, _s, _w = C.solve_map(cm, eps, 6)
    res[f"sym_M6_{tag}"] = sym(o, R, T)
    print(tag, res[f"sym_M6_{tag}"], flush=True)
# fail-before at r = 0.24: the 4-step staircase on the shipped solver
c = C.P / 2
rr = 0.24
w = np.array([c - rr, c + rr])
eps = np.ones((3, 3), complex)
eps[1, 1] = C.EPS_P
o, R, T = C.solve_walls(w, w, eps, 7)
res["stair1_r0.24_M7_vec"] = C.vec(o, R, T)
C.dump(f"v11_bars_{C.build_tag()}.json", res)
