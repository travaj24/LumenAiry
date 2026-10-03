"""V6 shared fixture: the three Phase C V-D3 over-refused layouts, exactly as
VERIFY_PMM2D_CURVED_C section 2 / v10_macro.py / v11_departures define them
(and as build_e2/e2_m_overrefusal.py re-used them), plus the macro-cell
composite map of verify_c/v10_macro.py (copied here, generalised to
off-centre circles) for the ONE-layer supercell reference."""
import numpy as np
from _ve import lumenairy  # noqa: F401  (tree assertion)

from lumenairy.elements.pmm import _curvemap as CM
from lumenairy.elements.pmm.shapes2d import Circle, Rect

H = 1.2
LAYOUTS = {
    # name: (px, py, shape A, shape B)
    "two_circles_r_0.30_0.40": (2.4, 1.2, Circle(0.6, 0.6, 0.30, 4.0),
                                Circle(1.8, 0.6, 0.40, 4.0)),
    "equal_circles_dy_0.05": (2.4, 1.2, Circle(0.6, 0.575, 0.3, 4.0),
                              Circle(1.8, 0.625, 0.3, 4.0)),
    "equal_circles_dy_0.02": (2.4, 1.2, Circle(0.6, 0.59, 0.3, 4.0),
                              Circle(1.8, 0.61, 0.3, 4.0)),
    "rect_touching_circle_30deg": (
        1.2, 1.2, Circle(0.6, 0.6, 0.36, 4.0),
        Rect(0.6 + 0.36 * np.cos(np.pi / 6) + 0.05,
             0.6 + 0.36 * np.sin(np.pi / 6) + 0.05, 0.1, 0.1, 2.25)),
}


class TwoHalves(CM.CellMap):
    """verify_c/v10_macro.py's composite (macro-cell) map, verbatim logic."""

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


def macro(name):
    """(cmap, eps_cell) of the ONE-layer supercell for a two-circle layout."""
    px, py, a, b = LAYOUTS[name]
    A, _ = CM._circle_map_3x3(H, a.r, center=(a.cx, a.cy))
    B, _ = CM._circle_map_3x3(H, b.r, center=(b.cx - H, b.cy))
    cm = TwoHalves(A, B)
    nx, ny = cm.shape
    eps = np.ones((nx, ny), complex)
    for i in range(nx):
        for j in range(ny):
            if cm._half(i, j)[2:] == (1, 1):
                eps[i, j] = 4.0
    return cm, eps
