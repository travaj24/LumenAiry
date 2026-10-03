"""V4/V5 ladder: the circle (layer 1, t 0.3) over a sinusoidal wall
(layer 2, t 0.25), lambda 1, P 1.2, n_sup 1, n_sub 1.45, normal incidence.

usage: v4_ladder.py <arm> <M> <eps_disk> <eps_wall> [x0 A]
arms:
  ref    -- layer_grids='shared' (the merged map; only for non-crossing)
  pl     -- per-layer, merged map DISABLED (_e2_per_layer_maps), q-matching ON
  ploff  -- as pl with the q-matching block removed (monkeypatch: every
            non-homogeneous shape layer without n_modes keeps the stack M)
  nat    -- per-layer as the library routes it (no instrument; for crossing
            pairs this IS the per-layer path, for non-crossing the fast path)
"""
import sys
import time

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure, stack2d_pure as SP
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

arm, M = sys.argv[1], int(sys.argv[2])
ed, ew = complex(sys.argv[3]), complex(sys.argv[4])
x0, A = (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 \
    else (0.12, 0.05)
P = 1.2

if arm == "ploff":
    _orig = SP.PMM2DStackPure._perlayer_modal_counts

    def _off(self):
        Ms = _orig(self)
        out = []
        for L, m in zip(self._layers, Ms):
            own = L.get("own")
            if own is not None and not own.get("homogeneous") \
                    and not L.get("pl_keywords"):
                out.append(int(L["M"]))
            else:
                out.append(m)
        return out
    SP.PMM2DStackPure._perlayer_modal_counts = _off

kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M, n_orders=3)
if arm != "ref":
    kw["layer_grids"] = "per-layer"
st = PMM2DStackPure(P, P, **kw)
st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, ed)], background_eps=1.0)
st.add_layer(0.25, shapes=[SinusoidalWall("x", x0, A, eps=ew)],
             background_eps=1.0)
if arm in ("pl", "ploff"):
    st._e2_per_layer_maps = True
fast = bool(st._perlayer_fast_ok()) if arm != "ref" else None
Ms = st._perlayer_modal_counts() if arm != "ref" else [M, M]
t0 = time.perf_counter()
o, R, T, J = solve(st)
wall = time.perf_counter() - t0
out = dict(arm=arm, M=M, eps_disk=ed, eps_wall=ew, x0=x0, A=A, fast=fast,
           Ms=Ms, orders=o, R=R, T=T, wall=wall,
           closure=np.abs(R.sum(1) + T.sum(1) - 1.0))
print(arm, M, ed, ew, x0, "Ms", Ms, "fast", fast, "wall %.1f" % wall,
      "closure", out["closure"])
tag = f"{ed.real:g}_{ew.real:g}_x{x0:g}"
dump(f"v4_ladder_{arm}_{tag}_M{M}", out)
