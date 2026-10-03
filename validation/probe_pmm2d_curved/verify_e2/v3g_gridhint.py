"""Does the 96-node cap warning's remedy ('add walls (grid_hint)') work, and
is it reachable from PMM2DStackPure?  circ_45_near (v3g_geoms.awkward) at
M = 4: (a) shapes= (the warning case), (b) compile_shapes(grid_hint=g) per
layer + add_layer(eps_cell=, cmap=); (c) add_layer(shapes=, grid=g)."""
import sys
import warnings

import numpy as np
from _ve import dump
from v3g_fix import N_SUB, NREC, WL, P, PMM2DStackPure
from v3g_geoms import awkward

from lumenairy.elements.pmm.shapes2d import compile_shapes

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
s1, s2 = awkward()["circ_45_near"]
out = {"M": M}
for g in (None, 5, 7):
    k0 = len(NREC)
    st = PMM2DStackPure(P, P, n_modes=M, n_substrate=N_SUB, n_orders=3,
                        layer_grids="per-layer")
    for t, s in ((0.25, s1), (0.2, s2)):
        if g is None:
            st.add_layer(t, shapes=s, background_eps=1.0)
        else:
            cell, xw, yw, cm = compile_shapes(P, P, s, 1.0, grid_hint=g)
            st.add_layer(t, eps_cell=cell, cmap=cm)
    st.set_source(WL)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T, J = st.solve()
    R, T = np.asarray(R), np.asarray(T)
    out[str(g)] = dict(n=[r["n"] for r in NREC[k0:]],
                       change=[r["change"] for r in NREC[k0:]],
                       Mab=[(r["Ma"], r["Mb"], r["Na"], r["Nb"])
                            for r in NREC[k0:]],
                       closure=np.abs(R.sum(1) + T.sum(1) - 1),
                       warn=[str(x.message)[:120] for x in w])
    print(g, out[str(g)], flush=True)
try:
    st = PMM2DStackPure(P, P, n_modes=M, layer_grids="per-layer")
    st.add_layer(0.25, shapes=s1, background_eps=1.0, grid=5)
    out["shapes_with_grid"] = "accepted"
except Exception as ex:  # noqa: BLE001
    out["shapes_with_grid"] = f"{type(ex).__name__}: {ex}"[:300]
print(out["shapes_with_grid"])
dump(f"v3g_gridhint_M{M}", out)
