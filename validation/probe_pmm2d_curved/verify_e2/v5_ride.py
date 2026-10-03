"""V5 (b): the riding rule.  A homogeneous layer BETWEEN two differently
mapped layers, at the top / bottom, and with n_modes named.

usage: v5_ride.py <stack> <arm> <M> [x0 A]
stacks (layers top -> bottom; sup 1, sub 1.45; disk eps 4 r .36, wall eps 2.25):
  CUS  circle (0.3) / uniform eps 1.7 (0.1) / sinusoid wall (0.25)
  SUC  sinusoid wall (0.25) / uniform eps 1.7 (0.1) / circle (0.3)
  UCS  uniform 1.7 (0.1) on top / circle / sinusoid
  CSU  circle / sinusoid / uniform 1.7 (0.1) at the bottom
  CUSk as CUS with the uniform layer naming n_modes=M (pl_keywords)
  CVS  as CUS with a HOMOGENEOUS SHAPE layer (a sinusoid painted eps 1.7 on
       background 1.7) in the middle
arms:
  ref    layer_grids='shared' (merged map; non-crossing x0 only)
  above  per-layer, merge disabled, library riding (nearest above)
  below  per-layer, merge disabled, rider forced onto the nearest BELOW
  none   per-layer, merge disabled, _e2_no_ride (own grid)
  nat    per-layer, as routed by the library (fast path when it exists)
"""
import sys
import time

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure, stack2d_pure as SP
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

stk, arm, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
x0, A = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 \
    else (0.12, 0.05)
P = 1.2
C = ("S", 0.3, [Circle(0.6, 0.6, 0.36, 4.0)])
S = ("S", 0.25, [SinusoidalWall("x", x0, A, eps=2.25)])
U = ("U", 0.1, 1.7)
UK = ("UK", 0.1, 1.7)
V = ("S", 0.1, [SinusoidalWall("x", 0.6, 0.12, eps=1.7)], 1.7)
lay = {"CUS": [C, U, S], "SUC": [S, U, C], "UCS": [U, C, S],
       "CSU": [C, S, U], "CUSk": [C, UK, S], "CVS": [C, V, S]}[stk]

if arm == "below":
    def _geo_below(self, Ms):
        Ls = self._layers
        geo = [(L["wx"], L["wy"], int(m), L.get("cmap"))
               for L, m in zip(Ls, Ms)]
        n = len(Ls)

        def mapped(i):
            return 0 <= i < n and geo[i][3] is not None
        rider = []
        for i, L in enumerate(Ls):
            own = L.get("own")
            if own is not None:
                rider.append(bool(own.get("homogeneous")))
            elif (L["kind"] in ("uniform", "uniform_tensor")
                  and not L.get("pl_keywords") and L.get("cmap") is None):
                rider.append(mapped(i - 1) or mapped(i + 1))
            else:
                rider.append(False)
        out = list(geo)
        for i in range(n):
            if not rider[i]:
                continue
            j = next((k for k in range(i + 1, n) if not rider[k]), None)
            if j is None:
                j = next(k for k in range(i - 1, -1, -1) if not rider[k])
            out[i] = geo[j]
        return out
    SP.PMM2DStackPure._perlayer_geometry = _geo_below

kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M, n_orders=3)
if arm != "ref":
    kw["layer_grids"] = "per-layer"
st = PMM2DStackPure(P, P, **kw)
for item in lay:
    if item[0] == "S":
        bg = item[3] if len(item) > 3 else 1.0
        st.add_layer(item[1], shapes=item[2], background_eps=bg)
    elif item[0] == "U":
        st.add_layer(item[1], eps=item[2])
    else:
        st.add_layer(item[1], eps=item[2], n_modes=M)
if arm in ("above", "below", "none"):
    st._e2_per_layer_maps = True
if arm == "none":
    st._e2_no_ride = True
info = {}
if arm != "ref":
    info["fast"] = bool(st._perlayer_fast_ok())
    Ms = st._perlayer_modal_counts()
    geo = st._perlayer_geometry(Ms)

    def fp(c):
        return None if c is None else str(c.fingerprint)[:24]
    info["Ms"] = Ms
    info["geo"] = [dict(M=g[2], cmap=fp(g[3]), own_cmap=fp(L.get("cmap")),
                        nx=SP._stag_walls_n(g[0]))
                   for g, L in zip(geo, st._layers)]
    info["merge_refusal"] = st._merge_refusal
t0 = time.perf_counter()
o, R, T, J = solve(st)
out = dict(stack=stk, arm=arm, M=M, x0=x0, A=A, orders=o, R=R, T=T,
           wall=time.perf_counter() - t0,
           closure=np.abs(R.sum(1) + T.sum(1) - 1.0), **info)
print(stk, arm, M, x0, info, "closure", out["closure"],
      "wall %.1f" % out["wall"])
dump(f"v5_ride_{stk}_{arm}_x{x0:g}_M{M}", out)
