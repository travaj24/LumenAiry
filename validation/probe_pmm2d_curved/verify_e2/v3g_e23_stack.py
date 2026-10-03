"""E2-3 stack level: a two-layer device on the verifier's walls, each layer
stretched in BOTH axes by its own SineStretch (from_physical_walls -- the
physical device is unchanged), vs the same device on its physical walls
(per-layer, no map: the shipped separable mortar) and vs the union walls
(one grid, square match).  Layer 1 (0.27): eps 3.0 on x 0.2..0.7,
y 0.35..0.95; layer 2 (0.19): eps 2.0 + 0.0i on x 0.45..1.0, y 0.1..0.6.
Usage: v3g_e23_stack.py M [theta phi]"""
import sys
import time

import numpy as np
from _ve import dump
from v3g_fix import N_SUB, N_SUP, NREC, WL, P, PMM2DStackPure

from lumenairy.elements.pmm import _curvemap as CM

M = int(sys.argv[1])
th, ph = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 \
    else (0.0, 0.0)
x1, y1 = np.array([0, 0.2, 0.7, P]), np.array([0, 0.35, 0.95, P])
x2, y2 = np.array([0, 0.45, 1.0, P]), np.array([0, 0.1, 0.6, P])
E1, E2 = 3.0, 2.0
c1 = np.ones((3, 3), complex)
c1[1, 1] = E1
c2 = np.ones((3, 3), complex)
c2[1, 1] = E2
m1 = CM.SeparableStretch.from_physical_walls(
    x1, y1, fx=CM.SineStretch(0.05 * P), fy=CM.SineStretch(-0.07 * P))
m2 = CM.SeparableStretch.from_physical_walls(
    x2, y2, fx=CM.SineStretch(-0.04 * P), fy=CM.SineStretch(0.11 * P))


def run(arm):
    t0 = time.perf_counter()
    k0 = len(NREC)
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    if arm == "mapped":
        st.add_layer(0.27, eps_cell=c1, cmap=m1)
        st.add_layer(0.19, eps_cell=c2, cmap=m2)
    elif arm == "unmapped":
        st.add_layer(0.27, eps_cell=c1, x_walls=x1, y_walls=y1)
        st.add_layer(0.19, eps_cell=c2, x_walls=x2, y_walls=y2)
    else:
        xu, yu = np.unique(np.r_[x1, x2]), np.unique(np.r_[y1, y2])
        xc, yc = 0.5 * (xu[:-1] + xu[1:]), 0.5 * (yu[:-1] + yu[1:])

        def cell(xw, yw, e):
            return np.where((xc[:, None] > xw[1]) & (xc[:, None] < xw[2])
                            & (yc[None, :] > yw[1]) & (yc[None, :] < yw[2]),
                            e, 1.0).astype(complex)
        st.add_layer(0.27, eps_cell=cell(x1, y1, E1), x_walls=xu, y_walls=yu)
        st.add_layer(0.19, eps_cell=cell(x2, y2, E2), x_walls=xu, y_walls=yu)
    st.set_source(WL, theta=th, phi=ph)
    o, R, T, J = st.solve()
    R, T = np.asarray(R), np.asarray(T)
    return dict(R=R, T=T, closure=np.abs(R.sum(1) + T.sum(1) - 1),
                n=[r["n"] for r in NREC[k0:]],
                wall=time.perf_counter() - t0)


out = {"M": M, "theta": th, "phi": ph}
ARMS = ("mapped", "unmapped", "union") if M <= 6 else ("mapped", "unmapped")
for arm in ARMS:
    out[arm] = run(arm)
    print(arm, out[arm]["closure"], out[arm]["n"], out[arm]["wall"],
          flush=True)


def dd(a, b):
    return float(max(np.abs(a["R"] - b["R"]).max(),
                     np.abs(a["T"] - b["T"]).max()))


out["mapped_vs_unmapped"] = dd(out["mapped"], out["unmapped"])
if "union" in out:
    out["mapped_vs_union"] = dd(out["mapped"], out["union"])
    out["unmapped_vs_union"] = dd(out["unmapped"], out["union"])
print({k: out[k] for k in ("mapped_vs_unmapped", "mapped_vs_union",
                           "unmapped_vs_union") if k in out})
dump(f"v3g_e23_stack_M{M}_th{th}_ph{ph}", out)
