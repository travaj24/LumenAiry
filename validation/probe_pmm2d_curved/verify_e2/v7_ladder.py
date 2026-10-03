"""V7c -- the fixed-n ladder of the curved cross-mass for one near-singular
pair (v7_near_singular.py's geometries, M = 4), to test the adaptive rule's
CAP stop: at the cap the adaptive compares n = 93 against n = 96 (a 3 %
step), a weak convergence test.  Every n of the ladder is computed;
failures (node inversion) are recorded; differences are relative max-norm
against the LARGEST n that succeeded, and between consecutive rungs.
  python v7_ladder.py <geom> <s> n1 n2 ...  -> v7_ladder_<geom>_<s>_win.json
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump

from lumenairy.elements.pmm import _curvemap as CM, _curvemortar as CMM, twod_staggered as TS

warnings.simplefilter("ignore")
P = 1.2
geom, s = sys.argv[1], float(sys.argv[2])
ns = [int(v) for v in sys.argv[3:]]
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
Xs, out = {}, {"geom": geom, "s": s, "rungs": {}}
for n in ns:
    t0 = time.perf_counter()
    try:
        Xs[n] = CMM.curved_cross_mass(ga, gb, n)
        r = {"ok": True}
    except Exception as ex:   # noqa: BLE001
        r = {"ok": False, "raised": f"{type(ex).__name__}: {ex}"}
    r["wall"] = time.perf_counter() - t0
    out["rungs"][str(n)] = r
    print(n, r, flush=True)
good = sorted(Xs)
if good:
    ref = Xs[good[-1]]
    sc = float(np.max(np.abs(ref)))
    for a, b in zip(good[:-1], good[1:]):
        out["rungs"][str(a)]["rel_vs_next_ok"] = float(
            np.max(np.abs(Xs[a] - Xs[b])) / sc)
    for n in good:
        out["rungs"][str(n)]["rel_vs_largest_ok"] = float(
            np.max(np.abs(Xs[n] - ref)) / sc)
    out["largest_ok"] = good[-1]
print({k: {kk: vv for kk, vv in v.items() if kk != "raised"}
       for k, v in out["rungs"].items()})
dump(f"v7_ladder_{geom}_{s!r}", out)
