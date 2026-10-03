"""E2-N: two CIRCLE maps (both with singular vertices) whose 45-degree points
approach each other -- r_a = 0.36 against r_b = 0.30 .. 0.36 + 1.2e-6 -- at
M = 4: the adaptive node count the cross-mass needs, its last change, and
whether the cap warning fires.  (Concentric circles do not cross, so a real
stack takes the merged map; this measures the kernel's own limit.)"""
import time
import warnings

from _common import CM, TS, P, dump

from lumenairy.elements.pmm import _curvemortar as CMM

out = {}
for r2 in (0.30, 0.34, 0.355, 0.3599, 0.36 + 1.2e-6):
    ca, _ = CM._circle_map_3x3(P, 0.36)
    cb, _ = CM._circle_map_3x3(P, r2)
    ga = TS.StagGridOps(P, P, ca.u_walls, ca.v_walls, 4, 1.0, 1.0, cmap=ca)
    gb = TS.StagGridOps(P, P, cb.u_walls, cb.v_walls, 4, 1.0, 1.0, cmap=cb)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
            res = dict(n=n, change=chg)
        except Exception as ex:
            res = dict(raised=f"{type(ex).__name__}: {str(ex)[:160]}")
    res["warned"] = any("did not settle" in str(x.message) for x in w)
    res["wall"] = time.perf_counter() - t0
    out[f"r2={r2!r}"] = res
    print(r2, res)
dump("e2_n_near_singular.json", out)
