"""M2 -- NESTED grids: layer B's uniform lattice is a 2x refinement of layer
A's, so every wall of A lies on B's grid.  The mortar solve (A on N=2, B on
N=4) is compared against the union-grid PMM2DStackPure solve on the REFINED
grid (A re-expressed exactly on N=4, B on N=4) -- the only difference is
layer A's own lateral resolution, so the two must CONVERGE onto each other as
the modal count M rises.  Joint M ladder; wall time reported for both arms."""
import json
import sys
import time

import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

PX = PY = 0.9e-6
WL = 0.60e-6
# A: 2-segment stripe (walls on halves) -- exactly representable on N=2 and N=4
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
# B: 4-segment pattern with a genuine 2-D pillar -- needs N=4
B4 = np.full((4, 4), 2.0 + 0j)
B4[1:3, 1:3] = 9.0
B4[0, 3] = 4.0
tA, tB = 0.20e-6, 0.15e-6
THETA, PHI = 0.20, 0.0
NORD = 2   # both arms: the mortar SUP half-space rides layer 0 (N=2), whose
           # per-axis capacity is q = 2(M-1) -> n_orders <= (q-1)//2 = 2 at M=4.
           # 2 amply covers the propagating set (|m|,|n| <= 1 here).

rows = []
for M in [int(x) for x in (sys.argv[1:] or [4, 5, 6, 7])]:
    t0 = time.perf_counter()
    ref = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    ref.add_layer(tA, eps_cell=refine_cell(A2, 2))
    ref.add_layer(tB, eps_cell=B4)
    ref.set_source(WL, theta=THETA, phi=PHI)
    o0, R0, T0, J0 = ref.solve()
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    mor = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
    mor.add_layer(tA, eps_cell=A2)
    mor.add_layer(tB, eps_cell=B4)
    mor.set_source(WL, theta=THETA, phi=PHI)
    o1, R1, T1, J1 = mor.solve()
    t_mor = time.perf_counter() - t0
    assert np.array_equal(o0, o1)
    scale = max(float(np.max(R0)), float(np.max(T0)))
    dR = float(np.max(np.abs(R0 - R1))) / scale
    dT = float(np.max(np.abs(T0 - T1))) / scale
    dJ = float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0)))
    c0 = [float(abs(R0[p].sum() + T0[p].sum() - 1.0)) for p in (0, 1)]
    c1 = [float(abs(R1[p].sum() + T1[p].sum() - 1.0)) for p in (0, 1)]
    rows.append(dict(M=M, dR=dR, dT=dT, dJ=dJ, clo_ref=max(c0),
                     clo_mortar=max(c1), t_ref=t_ref, t_mortar=t_mor,
                     R0_ref=float(R0[1].max()), R0_mor=float(R1[1].max())))
    print(f"M={M}  dR {dR:9.2e}  dT {dT:9.2e}  dJ {dJ:9.2e} | "
          f"closure ref {max(c0):8.1e} mortar {max(c1):8.1e} | "
          f"t_ref {t_ref:7.1f}s t_mortar {t_mor:7.1f}s "
          f"({t_ref / t_mor:.2f}x)  n_ord_used {mor.n_orders_used}", flush=True)

json.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m2_nested.json",
                     "w"), indent=1)
