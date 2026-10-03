"""Diagnose 'a quadrature node could not be inverted in the neighbouring
layer's map' on compile_shapes(grid_hint=...) maps: which geometries,
which cells, which points.  Kernel level (curved_cross_mass at fixed n)."""
from _ve import dump
from v3g_fix import P
from v3g_geoms import awkward, pairs

from lumenairy.elements.pmm import _curvemortar as CMM, twod_staggered as TS
from lumenairy.elements.pmm.shapes2d import Circle, compile_shapes

FAIL = []
_orig_ccn = CMM._cut_cell_nodes


def _ccn(Pm, sx, sy, Om, edges, n, cut=True):
    try:
        return _orig_ccn(Pm, sx, sy, Om, edges, n, cut=cut)
    except RuntimeError as ex:
        FAIL.append(dict(primary_cell=(int(sx), int(sy)),
                         primary_walls=[Pm.ub.tolist(), Pm.vb.tolist()],
                         other_walls=[Om.ub.tolist(), Om.vb.tolist()],
                         primary_sing=bool(Pm.sing), other_sing=bool(Om.sing),
                         msg=str(ex)))
        raise


CMM._cut_cell_nodes = _ccn
cases = {}
aw = awkward()
pr = pairs()
cases["circ_45_near"] = (aw["circ_45_near"][0], aw["circ_45_near"][1])
cases["ii_circ_circ"] = (pr["ii_circ_circ"][0], pr["ii_circ_circ"][1])
cases["i_circ_sin"] = (pr["i_circ_sin"][0], pr["i_circ_sin"][1])
cases["same_circle"] = ([Circle(0.52, 0.58, 0.33, 3.0)],
                        [Circle(0.52, 0.58, 0.33, 3.0)])
out = {}
for name, (s1, s2) in (cases.items() if __name__ == "__main__" else []):
    for g1, g2 in ((None, None), (5, 5), (5, None), (None, 5), (7, 7)):
        key = f"{name}_g{g1}_g{g2}"
        k0 = len(FAIL)
        maps = []
        try:
            for s, g in ((s1, g1), (s2, g2)):
                kw = {} if g is None else {"grid_hint": g}
                cell, xw, yw, cm = compile_shapes(P, P, s, 1.0, **kw)
                maps.append(cm)
            ga = TS.StagGridOps(P, P, maps[0].u_walls, maps[0].v_walls, 4,
                                1.0, 1.0, cmap=maps[0])
            gb = TS.StagGridOps(P, P, maps[1].u_walls, maps[1].v_walls, 4,
                                1.0, 1.0, cmap=maps[1])
            X = CMM.curved_cross_mass(ga, gb, 12)
            r = dict(ok=True, grids=[ga.bx.N, gb.bx.N],
                     maptypes=[type(m).__name__ for m in maps])
        except Exception as ex:  # noqa: BLE001
            r = dict(ok=False, err=f"{type(ex).__name__}: {ex}"[:200],
                     fails=FAIL[k0:k0 + 2],
                     maptypes=[type(m).__name__ for m in maps]
                     if len(maps) == 2 else None)
        out[key] = r
        print(key, {k: v for k, v in r.items() if k != "fails"},
              (r.get("fails") or [{}])[0].get("primary_cell") if not r["ok"] else "",
              flush=True)
if __name__ == "__main__":
    dump("v3g_inv_diag", out)
