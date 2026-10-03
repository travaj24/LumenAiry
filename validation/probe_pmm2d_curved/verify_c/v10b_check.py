"""V10b -- control of the V10 composite (macro-cell) map: two EQUAL circles
(r 0.36) side by side, where the shipped per-edge merge also works: the
composite and the merged map must give the same device to the
discretisation level (and the same closure)."""
import sys
import warnings

import numpy as np
from _vc import BUILD, dump
from v10_macro import TwoHalves

from lumenairy.elements.pmm import Circle, PMM2DStackPure, _curvemap as CM

warnings.simplefilter("ignore")
M = int(sys.argv[1])
r1, r2 = float(sys.argv[2]), float(sys.argv[3])
A, _ = CM._circle_map_3x3(1.2, r1)
B, _ = CM._circle_map_3x3(1.2, r2)
cm = TwoHalves(A, B)
nx, ny = cm.shape
eps = np.ones((nx, ny), complex)
for i in range(nx):
    for j in range(ny):
        if cm._half(i, j)[2:] == (1, 1):
            eps[i, j] = 4.0
out = {"M": M, "r": [r1, r2], "grid": [nx, ny],
       "sing": len(cm.singular_vertices)}
for name in ("composite", "merged"):
    st = PMM2DStackPure(2.4, 1.2, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2,
                        cmap=cm if name == "composite" else None)
    if name == "composite":
        st.add_layer(0.5, eps_cell=eps)
    else:
        try:
            st.add_layer(0.5, shapes=[Circle(0.6, 0.6, r1, 4.0),
                                      Circle(1.8, 0.6, r2, 4.0)],
                         background_eps=1.0)
        except ValueError as e:
            out["merged"] = "RAISE " + str(e)[:80]
            continue
    st.set_source(1.0)
    o, R, T, J = st.solve()
    R, T = np.asarray(R), np.asarray(T)
    out[name] = {"closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
                 "R": R.tolist(), "T": T.tolist(), "grid": list(st._grid)}
if isinstance(out.get("merged"), dict):
    out["diff"] = float(max(
        np.max(np.abs(np.subtract(out["composite"]["R"], out["merged"]["R"]))),
        np.max(np.abs(np.subtract(out["composite"]["T"], out["merged"]["T"])))))
print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                              if kk in ("closure", "grid")})
       for k, v in out.items()}, flush=True)
dump(f"v10b_check_M{M}_r{r1}_{r2}_{BUILD}.json", out)
