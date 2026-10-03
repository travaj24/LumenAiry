"""Adaptive cross-mass (the solver's path) on grid_hint (RefinedMap) maps of
ordinary crossing pairs: does the node inversion fail there too?"""
import warnings

import v3g_inv_diag as D  # noqa: E402
from _ve import dump
from v3g_fix import P
from v3g_geoms import pairs

from lumenairy.elements.pmm import _curvemortar as CMM, twod_staggered as TS
from lumenairy.elements.pmm.shapes2d import Circle, compile_shapes

pr = pairs()
cases = {"ii_circ_circ": pr["ii_circ_circ"][:2],
         "i_circ_sin": pr["i_circ_sin"][:2],
         "same_circle": ([Circle(0.52, 0.58, 0.33, 3.0)],
                         [Circle(0.52, 0.58, 0.33, 3.0)])}
out = {}
for name, (s1, s2) in cases.items():
    for g1, g2 in ((5, 5), (5, None), (7, 7)):
        key = f"{name}_g{g1}_g{g2}"
        k0 = len(D.FAIL)
        try:
            maps = [compile_shapes(P, P, s, 1.0, **({} if g is None else
                                                   {"grid_hint": g}))[3]
                    for s, g in ((s1, g1), (s2, g2))]
            gs = [TS.StagGridOps(P, P, m.u_walls, m.v_walls, 4, 1.0, 1.0,
                                 cmap=m) for m in maps]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                X, n, chg = CMM.curved_cross_mass_adaptive(*gs)
            r = dict(ok=True, n=n, change=chg, warn=[str(x.message)[:80]
                                                     for x in w])
        except Exception as ex:  # noqa: BLE001
            r = dict(ok=False, err=f"{type(ex).__name__}: {ex}"[:160],
                     fail_cell=[f["primary_cell"] for f in D.FAIL[k0:k0 + 1]])
        out[key] = r
        print(key, r, flush=True)
dump("v3g_inv_diag3", out)
