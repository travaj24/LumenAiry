"""V7b -- does a STACK surface the near-singular mortar limit?  Two-layer
per-layer stacks (lambda 1, period 1.2, air / n 1.45, M 4, n_orders 2):
layer 1 Circle(0.6, 0.6, 0.36, eps 4) t 0.3; layer 2 a second circle
(eps 2.25) t 0.25 that is (geom conc) concentric with radius 0.36 - s, or
(geom offx) radius 0.36 centred at (0.6 + s, 0.6), or (geom diag)
radius 0.2 on the diagonal, s * sqrt 2 from tangency (v7_near_singular).  Records the merge
verdict, the fast path, the solve's outcome (closure or the exception text
and type), the mortar node count reached, and the wall time.
  python v7_stack.py <geom> s1 s2 ...  -> v7_stack_<geom>_<tag>_win.json
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump

from lumenairy.elements.pmm import PMM2DStackPure, _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle

geom = sys.argv[1]
seps = [float(s) for s in sys.argv[2:]]
tag = sys.argv[2] if len(seps) == 1 else f"{len(seps)}x_{sys.argv[2]}"
seen = []
_orig = CMM.curved_cross_mass_adaptive


def spy(ga, gb, tol=None, cap=None):
    X, n, chg = _orig(ga, gb, tol=tol, cap=cap)
    seen.append((n, chg))
    return X, n, chg


CMM.curved_cross_mass_adaptive = spy        # in-process record only
out = {"geom": geom}
for s in seps:
    seen.clear()
    cd = 0.6 + (0.36 + 0.2) / np.sqrt(2.0) + s
    b = {"conc": Circle(0.6, 0.6, 0.36 - s, 2.25),
         "diag": Circle(cd, cd, 0.2, 2.25),
         "offx": Circle(0.6 + s, 0.6, 0.36, 2.25)}[geom]
    res = {}
    t0 = time.perf_counter()
    try:
        st = PMM2DStackPure(1.2, 1.2, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=4, n_orders=2, layer_grids="per-layer")
        st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, 4.0)],
                     background_eps=1.0)
        st.add_layer(0.25, shapes=[b], background_eps=1.0)
        res["merge_refusal"] = st._merge_refusal
        res["fast_ok"] = bool(st._perlayer_fast_ok())
        st.set_source(1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o, R, T, J = st.solve()
        R, T = np.asarray(R), np.asarray(T)
        res["closure"] = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
        res["R"], res["T"] = R, T
        res["warnings"] = sorted({str(x.message)[:300] for x in w})
    except Exception as ex:   # noqa: BLE001
        res["raised"] = f"{type(ex).__name__}: {ex}"
    res["mortar_calls"] = list(seen)
    res["wall"] = time.perf_counter() - t0
    out[f"s={s!r}"] = res
    print(geom, s, {k: v for k, v in res.items()
                    if k not in ("R", "T", "warnings")}, flush=True)
dump(f"v7_stack_{geom}_{tag}", out)
