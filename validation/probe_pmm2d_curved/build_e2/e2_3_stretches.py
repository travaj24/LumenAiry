"""E2-3: two layers with DIFFERENT separable stretches (the shipped 1-D ASR
precedent; Phase A's SeparableStretch) through the curved mortar, against
the UNMAPPED shared-grid solve on the union of their physical walls (the
same device, the shipped solver).

Layer 1 (depth 0.3): a pillar eps 4 on the physical walls x, y in
(0.3, 0.9), x stretched by a sine of amplitude 0.06 p; layer 2 (depth 0.25):
eps 2.25 on x in (0.5, 0.8), y in (0.2, 0.7), y stretched by 0.08 p.  The
stretch moves no physical wall (from_physical_walls), so the mapped solve
must converge to the unmapped answer.  Arms per rung M (arg 1):
  mapped    -- per-layer, each layer on its own stretch (curved mortar)
  unmapped  -- per-layer, each layer on its own physical walls, no map
               (the shipped separable mortar)
  shared    -- layer_grids='shared' on the union walls (5 x 5)
Output e2_3_stretches_M<M>.json; e2_3_summary.py differences them against
the shared arm at the top rung."""
import sys
import time

import numpy as np
from _common import CM, N_SUB, N_SUP, WL, P, PMM2DStackPure, dump

M = int(sys.argv[1])
th, ph = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 \
    else (0.0, 0.0)
x1 = np.array([0.0, 0.3, 0.9, P])
y1 = np.array([0.0, 0.3, 0.9, P])
x2 = np.array([0.0, 0.5, 0.8, P])
y2 = np.array([0.0, 0.2, 0.7, P])
c1 = np.ones((3, 3), complex)
c1[1, 1] = 4.0
c2 = np.ones((3, 3), complex)
c2[1, 1] = 2.25
m1 = CM.SeparableStretch.from_physical_walls(x1, y1,
                                             fx=CM.SineStretch(0.06 * P))
m2 = CM.SeparableStretch.from_physical_walls(x2, y2,
                                             fy=CM.SineStretch(0.08 * P))
out = {"M": M, "theta": th, "phi": ph}


def run(arm):
    t0 = time.perf_counter()
    if arm == "shared":
        xu = np.unique(np.concatenate([x1, x2]))
        yu = np.unique(np.concatenate([y1, y2]))
        xc = 0.5 * (xu[:-1] + xu[1:])
        yc = 0.5 * (yu[:-1] + yu[1:])
        cell1 = np.where((xc[:, None] > 0.3) & (xc[:, None] < 0.9)
                         & (yc[None, :] > 0.3) & (yc[None, :] < 0.9), 4.0,
                         1.0).astype(complex)
        cell2 = np.where((xc[:, None] > 0.5) & (xc[:, None] < 0.8)
                         & (yc[None, :] > 0.2) & (yc[None, :] < 0.7), 2.25,
                         1.0).astype(complex)
        cm = CM.IdentityMap(xu, yu, P, P)
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, cmap=None)
        # the union walls through the per-layer spelling on ONE grid
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        st.add_layer(0.3, eps_cell=cell1, x_walls=xu, y_walls=yu)
        st.add_layer(0.25, eps_cell=cell2, x_walls=xu, y_walls=yu)
        del cm
    else:
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        if arm == "mapped":
            st.add_layer(0.3, eps_cell=c1, cmap=m1)
            st.add_layer(0.25, eps_cell=c2, cmap=m2)
        else:
            st.add_layer(0.3, eps_cell=c1, x_walls=x1, y_walls=y1)
            st.add_layer(0.25, eps_cell=c2, x_walls=x2, y_walls=y2)
    st.set_source(WL, theta=th, phi=ph)
    o, R, T, J = st.solve()
    return dict(R=np.asarray(R), T=np.asarray(T),
                closure=np.abs(np.asarray(R).sum(1) + np.asarray(T).sum(1)
                               - 1.0),
                wall=time.perf_counter() - t0)


for arm in ("mapped", "unmapped", "shared"):
    out[arm] = run(arm)
    print(arm, "closure", out[arm]["closure"], "wall", out[arm]["wall"])
ref = out["shared"]
for arm in ("mapped", "unmapped"):
    out[arm + "_vs_shared"] = float(max(np.abs(out[arm]["R"] - ref["R"]).max(),
                                        np.abs(out[arm]["T"] - ref["T"]).max()))
    print(arm, "vs shared", out[arm + "_vs_shared"])
dump(f"e2_3_stretches_M{M}_th{th}_ph{ph}.json", out)
