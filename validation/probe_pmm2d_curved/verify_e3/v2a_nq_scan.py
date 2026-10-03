"""V2a: where does the NumPy solver's ADAPTIVE node count (_stag_map_nodes)
or its singular-vertex (Duffy) set change along the three shape families?
(The frozen twin cannot see either change.)"""
import sys

from _ve3 import TS, P, dump, np

from lumenairy.elements.pmm import Circle, FilletRect, SinusoidalWall
from lumenairy.elements.pmm.shapes2d import _merge

FAM = {
    "circle_r": (lambda x: [Circle(0.6, 0.6, x, 4.0)], 1.0,
                 np.round(np.arange(0.06, 0.585, 0.02), 4)),
    "fillet_r": (lambda x: [FilletRect(0.6, 0.6, 0.6, 0.5, x, 4.0)], 1.0,
                 np.array([1e-4, 1e-3, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08,
                           0.1, 0.12, 0.15, 0.2, 0.24])),
    "sine_A": (lambda x: [SinusoidalWall("x", 0.6, x, eps=2.25)], 1.0,
               np.array([0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12,
                         0.15, 0.2, 0.25])),
}
Ms = [int(m) for m in sys.argv[1:]] or [4, 5, 6, 7]
out = {}
for fam, (shp, bg, xs) in FAM.items():
    rows = []
    for x in xs:
        try:
            U, V, cm, _c, ident, _m = _merge(P, P, [("layer 1", shp(float(x)),
                                                      bg, None)])
        except ValueError as exc:
            rows.append(dict(x=float(x), refused=str(exc)[:80]))
            continue
        row = dict(x=float(x), sing=len(cm.singular_vertices),
                   ngrid=[len(cm.u_walls) - 1, len(cm.v_walls) - 1])
        for M in Ms:
            bx = TS.Basis1D(P, cm.u_walls, M)
            by = TS.Basis1D(P, cm.v_walls, M)
            row[f"nq_M{M}"] = int(TS._stag_map_nodes(bx, by, cm, M))
        rows.append(row)
        print(fam, row, flush=True)
    out[fam] = rows
dump("v2a_nq_scan.json", out)
