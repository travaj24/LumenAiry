"""V7a -- the curved mortar's cross-mass between two CIRCLE maps whose
singular vertices (the 45-degree points) approach, at M = 4 (build_e2/
e2_n_near_singular.py's setting: 3 x 3 circle maps on the 1.2 cell).

Geometries (arg 1):
  conc   the builder's: CONCENTRIC circles r_a = 0.36, r_b = 0.36 - s
         (do NOT cross; 45-degree points |s| apart radially)
  offx   EQUAL circles r = 0.36, centre b shifted by s in x (they CROSS;
         every 45-degree point s apart)
  diag   circle b of r 0.2 up-right of a on the diagonal: a's upper-right
         45-degree point and b's lower-left one are s*sqrt(2) apart (the
         circles are tangent at s = 0 and disjoint for s > 0: no crossing)
Per separation s (args 2..): the adaptive (n, last change, warning, error
text, wall); then the checks: X at n' = ceil(1.5 n) (a finer fixed rule),
and the same integral with primary='a' and 'b' at n (the self-consistency
arm), all relative max-norm against the adaptive X.
  python v7_near_singular.py <geom> s1 s2 ...  -> v7_ns_<geom>_<tag>_win.json
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump

from lumenairy.elements.pmm import _curvemap as CM, _curvemortar as CMM, twod_staggered as TS

P = 1.2
geom = sys.argv[1]
seps = [float(s) for s in sys.argv[2:]]
tag = sys.argv[2] if len(seps) == 1 else f"{len(seps)}x_{sys.argv[2]}"


def grids(s):
    ca, _ = CM._circle_map_3x3(P, 0.36)
    if geom == "conc":
        cb, _ = CM._circle_map_3x3(P, 0.36 - s)
    elif geom == "diag":
        c = 0.6 + (0.36 + 0.2) / np.sqrt(2.0) + s
        cb, _ = CM._circle_map_3x3(P, 0.2, center=(c, c))
    else:
        cb, _ = CM._circle_map_3x3(P, 0.36, center=(0.6 + s, 0.6))
    ga = TS.StagGridOps(P, P, ca.u_walls, ca.v_walls, 4, 1.0, 1.0, cmap=ca)
    gb = TS.StagGridOps(P, P, cb.u_walls, cb.v_walls, 4, 1.0, 1.0, cmap=cb)
    return ga, gb


def rel(X, Y):
    return float(np.max(np.abs(X - Y)) / max(np.max(np.abs(Y)), 1e-300))


out = {"geom": geom, "M": 4}
for s in seps:
    ga, gb = grids(s)
    res = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
            res.update(n=n, change=chg)
        except Exception as ex:   # noqa: BLE001
            X = None
            res["raised"] = f"{type(ex).__name__}: {ex}"
    res["warnings"] = [str(x.message) for x in w]
    res["wall_adaptive"] = time.perf_counter() - t0
    if X is not None:
        t0 = time.perf_counter()
        for lab, kw in (("fine", dict(n=int(np.ceil(1.5 * n)))),
                        ("primary_a", dict(n=n, primary="a")),
                        ("primary_b", dict(n=n, primary="b"))):
            try:
                Y = CMM.curved_cross_mass(ga, gb, **kw)
                res[f"rel_{lab}"] = rel(Y, X)
            except Exception as ex:   # noqa: BLE001
                res[f"rel_{lab}"] = f"{type(ex).__name__}: {ex}"
        res["wall_checks"] = time.perf_counter() - t0
        res["absmax"] = float(np.max(np.abs(X)))
    out[f"s={s!r}"] = res
    print(geom, s, {k: v for k, v in res.items() if k != "warnings"},
          "nwarn", len(res["warnings"]), flush=True)
dump(f"v7_ns_{geom}_{tag}", out)
