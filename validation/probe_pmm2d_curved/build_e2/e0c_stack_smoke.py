"""E0c: the per-layer curved stack end to end -- a circle over a sinusoidal
wall that runs THROUGH it (the stack-wide merge refuses), lossless closure
and timing per M."""
import sys
import time

import numpy as np
from _common import D1, D2, EPS_P, EPS_W, N_SUB, N_SUP, R_CIRC, WL, P, PMM2DStackPure, dump

from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
x0, A = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (
    0.6, 0.12)
st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB, n_modes=M,
                    n_orders=3, layer_grids="per-layer")
st.add_layer(D1, shapes=[Circle(0.6, 0.6, R_CIRC, EPS_P)], background_eps=1.0)
st.add_layer(D2, shapes=[SinusoidalWall("x", x0, A, eps=EPS_W)],
             background_eps=1.0)
print("fast path:", st._perlayer_fast_ok(), "refusal:",
      (st._merge_refusal or "")[:90])
st.set_source(WL)
t0 = time.perf_counter()
o, R, T, J = st.solve()
dt = time.perf_counter() - t0
clos = np.abs(R.sum(1) + T.sum(1) - 1.0)
print("M", M, "closure", clos, "R00", R[:, 40 if R.shape[1] > 40 else 0],
      "t", dt)
dump(f"e0c_stack_smoke_M{M}_x{x0}.json", dict(M=M, closure=clos, R=R, T=T,
                                               wall=dt))
