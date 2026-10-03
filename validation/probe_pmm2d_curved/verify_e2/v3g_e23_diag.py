"""E2-3 diagnostic: is the stretched stack's closure set by the curved
mortar or by the stretched layers themselves?  Each layer of v3g_e23_stack
ALONE (one layer, no mortar), mapped vs unmapped, closure and difference.
Usage: v3g_e23_diag.py M"""
import sys

import numpy as np
from _ve import dump
from v3g_e23_stack_defs import c1, c2, m1, m2, x1, x2, y1, y2
from v3g_fix import N_SUB, N_SUP, WL, P, PMM2DStackPure

M = int(sys.argv[1])
out = {"M": M}
for name, t, c, m, xw, yw in (("L1", 0.27, c1, m1, x1, y1),
                              ("L2", 0.19, c2, m2, x2, y2)):
    res = {}
    for arm in ("mapped", "unmapped"):
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        if arm == "mapped":
            st.add_layer(t, eps_cell=c, cmap=m)
        else:
            st.add_layer(t, eps_cell=c, x_walls=xw, y_walls=yw)
        st.set_source(WL)
        o, R, T, J = st.solve()
        R, T = np.asarray(R), np.asarray(T)
        res[arm] = (R, T)
        out[f"{name}_{arm}_closure"] = np.abs(R.sum(1) + T.sum(1) - 1)
    out[f"{name}_mapped_vs_unmapped"] = float(max(
        np.abs(res["mapped"][0] - res["unmapped"][0]).max(),
        np.abs(res["mapped"][1] - res["unmapped"][1]).max()))
print(out)
dump(f"v3g_e23_diag_M{M}", out)
