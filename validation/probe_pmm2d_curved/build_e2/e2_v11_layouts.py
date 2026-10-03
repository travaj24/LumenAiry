"""E2-V11 (the Phase E2 verifier's V-E2-D11): the SAME circle on its plain
3 x 3 map and on a grid_hint = 5 refinement, M = 3: the cross-mass against
the separable cross-mass of the two wall grids (the transition is the
identity), and the fail-before with the overlap test disabled (refused)."""
import numpy as np
from _common import TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, compile_shapes

c3 = compile_shapes(P, P, [Circle(0.6, 0.6, 0.36, 4.0)], 1.0)[3]
c5 = compile_shapes(P, P, [Circle(0.6, 0.6, 0.36, 4.0)], 1.0,
                    grid_hint=5)[3]
ga = TS.StagGridOps(P, P, c3.u_walls, c3.v_walls, 3, 1.0, 1.0, cmap=c3)
gb = TS.StagGridOps(P, P, c5.u_walls, c5.v_walls, 3, 1.0, 1.0, cmap=c5)
X = CMM.curved_cross_mass(ga, gb, 6)
g0a = TS.StagGridOps(P, P, c3.u_bounds, c3.v_bounds, 3, 1.0, 1.0)
g0b = TS.StagGridOps(P, P, c5.u_bounds, c5.v_bounds, 3, 1.0, 1.0)
cr = TS.StagCrossOps(g0a, g0b)
Xs = np.zeros_like(X)
Xs[:g0b.qq, :g0a.qq] = np.kron(*cr.C1H())
Xs[g0b.qq:, g0a.qq:] = np.kron(*cr.C2H())
rel = float(np.abs(X - Xs).max() / np.abs(Xs).max())
orig = CMM._same_map_on_overlap
CMM._same_map_on_overlap = lambda *a: False
try:
    CMM.curved_cross_mass(ga, gb, 6)
    fb = "not refused"
except NotImplementedError as ex:
    fb = "refused: " + str(ex)[:100]
CMM._same_map_on_overlap = orig
print("rel", rel, "|", fb)
dump("e2_v11_layouts.json", dict(rel=rel, failbefore=fb))
