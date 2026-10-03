"""E2-W: the SHIPPED (unmapped) per-layer mortar on the analogue of the E2-4
"both ways" comparison -- a square eps-4 pillar (x, y in 0.3 .. 0.9, depth
0.3) over a stripe layer (eps 2.25 for x > 0.12, depth 0.25) whose walls are
NOT the pillar's -- per-layer (own walls only, separable mortar) against the
shared union-grid solve.  The per-layer arm's error here is the
non-conforming mortar's own (the pillar's rim trace is cut by the stripe
layer's cells), which the curved mortar inherits."""
import sys

import numpy as np
from _common import N_SUB, N_SUP, WL, P, PMM2DStackPure, dump

M = int(sys.argv[1])
xp = np.array([0.0, 0.3, 0.9, P])
xs = np.array([0.0, 0.12, P])
pil = np.ones((3, 3), complex)
pil[1, 1] = 4.0
stripe = np.ones((2, 2), complex)
stripe[1, :] = 2.25
xu = np.unique(np.concatenate([xp, xs]))
xc = 0.5 * (xu[:-1] + xu[1:])
yu = np.array([0.0, 0.3, 0.6, 0.9, P])
yc = 0.5 * (yu[:-1] + yu[1:])
c1 = np.where((xc[:, None] > 0.3) & (xc[:, None] < 0.9) & (yc[None, :] > 0.3)
              & (yc[None, :] < 0.9), 4.0, 1.0).astype(complex)
c2 = np.where(xc[:, None] > 0.12, 2.25, 1.0) * np.ones((1, yc.size))
out = {"M": M}
st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB, n_modes=M,
                    n_orders=3, layer_grids="per-layer")
st.add_layer(0.3, eps_cell=c1, x_walls=xu, y_walls=yu)
st.add_layer(0.25, eps_cell=c2.astype(complex), x_walls=xu, y_walls=yu)
st.set_source(WL)
o, Ru, Tu, J = st.solve()
for q_match in (False, True):
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    st.add_layer(0.3, eps_cell=pil, x_walls=xp, y_walls=xp)
    Ms = (M if not q_match else -(-3 * (M - 1) // 2) + 1)
    st.add_layer(0.25, eps_cell=stripe, x_walls=xs, y_walls=[0.6],
                 n_modes=Ms)
    st.set_source(WL)
    o, R, T, J = st.solve()
    key = "perlayer_qmatch" if q_match else "perlayer"
    out[key] = dict(vs_union=float(max(np.abs(R - Ru).max(),
                                       np.abs(T - Tu).max())),
                    closure=np.abs(R.sum(1) + T.sum(1) - 1.0), M_stripe=Ms)
    print(key, out[key])
out["union_closure"] = np.abs(Ru.sum(1) + Tu.sum(1) - 1.0)
dump(f"e2_w_shipped_baseline_M{M}.json", out)
