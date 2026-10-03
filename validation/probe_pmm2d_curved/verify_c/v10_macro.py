"""V10 -- the over-refusal of the per-edge merge is AVOIDABLE: two circles
of radii 0.30 and 0.40 side by side (period 2.4 x 1.2; the merge raises a
FOLD, v2_merge 'two_circles_r_0.30_0.40') on the plan's MACRO-CELL rule.

Map: each circle's own lone 3 x 3 Phase B map on its half of the cell
([0, 1.2] and [1.2, 2.4]); the union walls SUBDIVIDE each half's macro-cells
and every fine cell evaluates its macro-cell's own blend (RefinedMap's
rule, applied per half).  The halves meet on x = 1.2, where both maps are
the identity, so the composite is continuous.  The other circle's 45-degree
walls cross each half only as NON-material lines (bent harmlessly).

  python v10_macro.py <M>  -> v10_macro_M<M>_<build>.json (+ geometry check)
"""
import sys
import time
import warnings

import numpy as np
from _geom import check, verdict
from _vc import BUILD, dump

from lumenairy.elements.pmm import Circle, PMM2DStackPure, _curvemap as CM

warnings.simplefilter("ignore")
H = 1.2


class TwoHalves(CM.CellMap):
    def __init__(self, A, B, n_square=True):
        self.A, self.B = A, B
        ub = np.unique(np.r_[A.u_bounds, B.u_bounds + H])
        vb = np.unique(np.r_[A.v_bounds, B.v_bounds])
        if n_square:
            vb = list(vb)
            while len(vb) < len(ub):
                k = int(np.argmax(np.diff(vb)))
                vb.insert(k + 1, 0.5 * (vb[k] + vb[k + 1]))
            vb = np.asarray(vb)
        self._init_walls(ub, vb, 2 * H, H)
        self.validate(n=12)

    def _half(self, sx, sy):
        um = 0.5 * (self.u_bounds[sx] + self.u_bounds[sx + 1])
        vm = 0.5 * (self.v_bounds[sy] + self.v_bounds[sy + 1])
        if um < H:
            m, du = self.A, 0.0
        else:
            m, du = self.B, H
        i = int(np.searchsorted(m.u_bounds, um - du) - 1)
        j = int(np.searchsorted(m.v_bounds, vm) - 1)
        return m, du, i, j

    def geom(self, sx, sy, U, V):
        m, du, i, j = self._half(sx, sy)
        X, Y, xu, xv, yu, yv = m.geom(i, j, np.asarray(U) - du, V)
        return X + du, Y, xu, xv, yu, yv

    @property
    def singular_vertices(self):
        out = []
        for m, du in ((self.A, 0.0), (self.B, H)):
            for bsx, bsy, cu, cv in m.singular_vertices:
                u = m.u_bounds[bsx + cu] + du
                v = m.v_bounds[bsy + cv]
                iu = int(np.argmin(np.abs(self.u_bounds - u)))
                iv = int(np.argmin(np.abs(self.v_bounds - v)))
                out.append((iu - cu, iv - cv, cu, cv))
        return sorted(out)

    def _key(self):
        return ("TwoHalves", self.A.fingerprint, self.B.fingerprint)


def build():
    A, _ = CM._circle_map_3x3(H, 0.30)
    B, _ = CM._circle_map_3x3(H, 0.40)
    cm = TwoHalves(A, B)
    nx, ny = cm.shape
    eps = np.ones((nx, ny), complex)
    for i in range(nx):
        for j in range(ny):
            m, du, a, b = cm._half(i, j)
            if (a, b) == (1, 1):
                eps[i, j] = 4.0
    return cm, eps


def main(M):
    cm, eps = build()
    shapes = [Circle(0.6, 0.6, 0.30, 4.0), Circle(1.8, 0.6, 0.40, 4.0)]
    g = check(cm, [eps], [shapes], [1.0])
    t0 = time.time()
    st = PMM2DStackPure(2 * H, H, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=4, cmap=cm)
    st.add_layer(0.5, eps_cell=eps)
    st.set_source(1.0)
    o, R, T, J = st.solve()
    R, T, o = np.asarray(R), np.asarray(T), np.asarray(o)
    res = dict(M=M, grid=list(cm.shape), geometry_verdict=verdict(g),
               geometry=g, closure=float(np.max(np.abs(R.sum(1) + T.sum(1)
                                                       - 1))),
               orders=o.tolist(), R=R.tolist(), T=T.tolist(),
               wall_s=time.time() - t0)
    print({k: v for k, v in res.items() if k not in ("geometry", "orders",
                                                     "R", "T")}, flush=True)
    dump(f"v10_macro_M{M}_{BUILD}.json", res)


if __name__ == "__main__":
    main(int(sys.argv[1]))
