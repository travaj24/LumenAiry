"""Reproduce the node-inversion RuntimeError of v3g_gridhint (circ_45_near,
grid_hint = 5 on both layers, M = 4) along the adaptive node ladder."""
import json

import numpy as np
import v3g_inv_diag as D  # noqa: E402  (installs the _cut_cell_nodes wrap)
from _ve import dump
from v3g_fix import P
from v3g_geoms import awkward

from lumenairy.elements.pmm import _curvemortar as CMM, twod_staggered as TS
from lumenairy.elements.pmm.shapes2d import compile_shapes

out = {}
for name in ("circ_45_near",):
    for g in (5, 7, None):
        s1, s2 = awkward()[name]
        maps = [compile_shapes(P, P, s, 1.0, **({} if g is None else
                                               {"grid_hint": g}))[3]
                for s in (s1, s2)]
        ga = TS.StagGridOps(P, P, maps[0].u_walls, maps[0].v_walls, 4, 1.0,
                            1.0, cmap=maps[0])
        gb = TS.StagGridOps(P, P, maps[1].u_walls, maps[1].v_walls, 4, 1.0,
                            1.0, cmap=maps[1])
        prev = None
        for n in (8, 12, 18, 27, 41, 62, 93, 96):
            k0 = len(D.FAIL)
            key = f"{name}_g{g}_n{n}"
            try:
                X = CMM.curved_cross_mass(ga, gb, n)
                chg = None if prev is None else float(
                    np.abs(X - prev).max() / np.abs(X).max())
                prev = X
                out[key] = dict(ok=True, change=chg)
            except Exception as ex:  # noqa: BLE001
                out[key] = dict(ok=False, err=f"{type(ex).__name__}: {ex}",
                                fail=D.FAIL[k0:k0 + 1])
            print(key, json.dumps(out[key])[:400], flush=True)
dump("v3g_inv_diag2", out)
