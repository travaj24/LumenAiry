"""V5b -- the rotated Ellipse's FOLD domain: shipped layout (corners at the
PARAMETRIC 45-degree points) vs a candidate layout (corners where the
OUTWARD NORMAL points at 45 / 135 / 225 / 315 degrees in the lab -- the
exact analogue of the circle's 45-degree points), lone ellipse, aspect x
angle.  The candidate is installed by monkeypatching Ellipse._layout for
rotated ellipses only (the axis-aligned branch is untouched: for angle = 0
both choices are the same points).  Each solvable layout is checked against
the analytic outline (_geom.check).

  python v5b_ellipse_layout.py   -> v5b_ellipse_layout_<build>.json
"""
import numpy as np
from _geom import check, verdict
from _vc import BUILD, dump

from lumenairy.elements.pmm import Ellipse, compile_shapes, shapes2d as SH
from lumenairy.elements.pmm._curvemap import EllipseArc

SHIPPED = SH.Ellipse._layout
_DEG = np.pi / 180


def normal45_layout(self, px, py):
    if self.angle == 0.0:
        return SHIPPED(self, px, py)
    self._check_inside(px, py, strict=True)
    a, b, al = self.a, self.b, self.angle
    t = {}
    for k, psi in (("BL", 225), ("BR", 315), ("TR", 45), ("TL", 135)):
        p = psi * _DEG - al
        t[k] = np.arctan2(b * np.sin(p), a * np.cos(p))
    # keep the parametric angles increasing counter-clockwise from BL
    tBL = t["BL"] % (2 * np.pi)
    tBR = tBL + ((t["BR"] - tBL) % (2 * np.pi))
    tTR = tBR + ((t["TR"] - tBR) % (2 * np.pi))
    tTL = tTR + ((t["TL"] - tTR) % (2 * np.pi))
    P = {k: np.array(self._pt(v)) for k, v in
         (("BL", tBL), ("BR", tBR), ("TR", tTR), ("TL", tTL))}
    u = [0.5 * (P["BL"][0] + P["TL"][0]), 0.5 * (P["BR"][0] + P["TR"][0])]
    v = [0.5 * (P["BL"][1] + P["BR"][1]), 0.5 * (P["TL"][1] + P["TR"][1])]
    verts = {(u[0], v[0]): P["BL"], (u[1], v[0]): P["BR"],
             (u[1], v[1]): P["TR"], (u[0], v[1]): P["TL"]}
    c, ax = (self.cx, self.cy), (a, b)
    edges = [
        SH._HardEdge("h", v[0], u[0], u[1],
                     EllipseArc(c, ax, tBL, tBR, angle=al)),
        SH._HardEdge("h", v[1], u[0], u[1],
                     EllipseArc(c, ax, tTL, tTR, angle=al)),
        SH._HardEdge("v", u[0], v[0], v[1],
                     EllipseArc(c, ax, tBL + 2 * np.pi, tTL, angle=al)),
        SH._HardEdge("v", u[1], v[0], v[1],
                     EllipseArc(c, ax, tBR, tTR, angle=al)),
    ]
    return SH._Layout(u, v, verts, edges, [(u[0], u[1], v[0], v[1])])


OUT = {}
for lay in ("shipped", "normal45"):
    SH.Ellipse._layout = SHIPPED if lay == "shipped" else normal45_layout
    dom = {}
    for asp in (1.05, 1.5, 2.0, 3.0, 5.0):
        row = {}
        for ang in (5, 10, 20, 30, 40, 44.9, -30):
            b = 0.08 if asp >= 3 else 0.2
            a = b * asp
            sh = [Ellipse(0.6, 0.6, a, b, 4.0, angle=np.deg2rad(ang))]
            try:
                cell, xw, yw, cm = compile_shapes(1.2, 1.2, sh, 1.0)
                g = check(cm, [cell], [sh], [1.0])
                row[ang] = verdict(g)
            except ValueError as e:
                row[ang] = "FOLD" if "FOLDS" in str(e) else str(e)[:50]
        dom[f"aspect {asp}"] = row
        print(lay, "aspect", asp, row, flush=True)
    OUT[lay] = dom
SH.Ellipse._layout = SHIPPED
dump(f"v5b_ellipse_layout_{BUILD}.json", OUT)
