"""E2-9: cost -- the per-layer mapped stack (two maps, the curved mortar)
against the shared MERGED map on the NON-overlapping circle + sinusoid pair
(where both exist), at rung M (arg 1).  Wall time of the whole solve and of
the curved cross-masses inside it (timed by wrapping StagCrossOpsMapped),
the pencil sizes.  The box is shared with other agents' jobs, so every wall
time is an UPPER bound; the RATIO within one run is the reading."""
import sys
import time

import numpy as np
from _common import D1, D2, EPS_P, EPS_W, R_CIRC, dump, shapes_stack, solve

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

M = int(sys.argv[1])
lay = [(D1, [Circle(0.6, 0.6, R_CIRC, EPS_P)], 1.0),
       (D2, [SinusoidalWall("x", 0.12, 0.05, eps=EPS_W)], 1.0)]
out = {"M": M}
t0 = time.perf_counter()
st = shapes_stack(lay, M)
assert st._perlayer_fast_ok()
solve(st)
out["merged_wall"] = time.perf_counter() - t0
out["merged_grid"] = list(st._grid)
out["merged_pencil"] = 2 * (st._grid[0] * (M - 1)) ** 2
orig = CMM.StagCrossOpsMapped
tx = []


def timed(ga, gb, tol=None):
    t1 = time.perf_counter()
    o = orig(ga, gb, tol=tol)
    tx.append((time.perf_counter() - t1, o.n, ga.qq, gb.qq))
    return o


CMM.StagCrossOpsMapped = timed
try:
    t0 = time.perf_counter()
    st2 = shapes_stack(lay, M)
    st2._e2_per_layer_maps = True
    solve(st2)
    out["perlayer_wall"] = time.perf_counter() - t0
finally:
    CMM.StagCrossOpsMapped = orig
Ms = st2._perlayer_modal_counts()
out["perlayer_modal_counts"] = Ms
out["perlayer_pencils"] = [2 * (int(np.size(L["wx"]) - 1 if np.ndim(L["wx"])
                                    else L["wx"]) * (m - 1)) ** 2
                           for L, m in zip(st2._layers, Ms)]
out["cross_mass"] = [dict(wall=t, n=n, qq_a=a, qq_b=b) for t, n, a, b in tx]
out["cross_mass_wall"] = float(sum(t for t, *_ in tx))
out["ratio_perlayer_over_merged"] = out["perlayer_wall"] / out["merged_wall"]
print(out)
dump(f"e2_9_cost_M{M}.json", out)
