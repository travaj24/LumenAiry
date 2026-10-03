"""E2-V: the SHIPPED (unmapped, separable) per-layer mortar on the analogue
of the E2-4 vacuum-spacer identity -- a square eps-4 pillar (3 x 3 cells,
side 0.6) with a VACUUM spacer on top on a NON-conforming grid -- against the
pillar alone.  The pillar top-face trace carries the rim (edge) singularity
along the pillar outline; a spacer grid without walls there represents it
only algebraically.  This is the baseline the curved mortar is held to.
arms: 'grid1' (the shipped default: one cell, the neighbour M rule),
'offset3' (3 x 3 walls at 0.2 / 0.8, not the pillar walls), 'conforming'
(the pillar walls: the square match)."""
import sys
import time

import numpy as np
from _common import N_SUB, N_SUP, WL, P, PMM2DStackPure, dump

M = int(sys.argv[1])
cell = np.ones((3, 3), complex)
cell[1, 1] = 4.0
out = {"M": M}
base = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                      n_modes=M, n_orders=3)
base.add_layer(0.3, eps_cell=cell)
base.set_source(WL)
o0, R0, T0, J0 = base.solve()
for arm in ("grid1", "offset3", "conforming"):
    t0 = time.perf_counter()
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    if arm == "grid1":
        st.add_layer(0.25, eps=1.0)
    elif arm == "offset3":
        st.add_layer(0.25, eps=1.0, x_walls=[0.2, 0.8], y_walls=[0.2, 0.8],
                     n_modes=M)
    else:
        st.add_layer(0.25, eps=1.0, grid=3, n_modes=M)
    st.add_layer(0.3, eps_cell=cell)
    st.set_source(WL)
    o, R, T, J = st.solve()
    out[arm] = dict(vs_alone=float(max(np.abs(R - R0).max(),
                                       np.abs(T - T0).max())),
                    closure=np.abs(R.sum(1) + T.sum(1) - 1.0),
                    wall=time.perf_counter() - t0)
    print(arm, out[arm])
dump(f"e2_v_spacer_shipped_M{M}.json", out)
