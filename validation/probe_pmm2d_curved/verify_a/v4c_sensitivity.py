"""V4c -- the mapped solve's sensitivity to operator perturbations far below
the quadrature tolerance: R / T / Jones of the stripe (M = 6) under forced
node counts whose operators agree to ~1e-13 (1024 vs 1000 vs 999), against
the identity map (20 vs 64 nodes, both exact) and the unmapped solve.
Output: v4c_sensitivity_<build>.txt (printed lines; the Windows reading was
taken 2026-10-02 with this script's statements run inline)."""
import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS
from lumenairy.elements.pmm._curvemap import IdentityMap

orig = TS._stag_map_nodes


def run(cm, nq, M=6, n_orders=3):
    TS._stag_map_nodes = (lambda *a, **k: nq)
    try:
        return C.stack_solve(cm, [C.cell("stripe")], M, n_orders=n_orders)
    finally:
        TS._stag_map_nodes = orig


cm = IdentityMap(3, 3, 1.0, 1.0)
a = run(cm, 20)
b = run(cm, 64)
c = C.stack_solve(None, [C.cell("stripe")], 6)
print("ident 20 vs 64", np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max(),
      np.abs(a[3] - b[3]).max())
print("ident 20 vs none", np.abs(a[1] - c[1]).max(), np.abs(a[2] - c[2]).max(),
      np.abs(a[3] - c[3]).max())
cm = C.stretch_map(C.HarmonicStretch(0.10, 0.04, 0.9))
for n_orders in (3, 2):
    x = run(cm, 1024, n_orders=n_orders)
    y = run(cm, 1000, n_orders=n_orders)
    z = run(cm, 999, n_orders=n_orders)
    print("asym no", n_orders, "1024 vs 1000", np.abs(x[1] - y[1]).max(),
          np.abs(x[2] - y[2]).max(), np.abs(x[3] - y[3]).max(),
          "1000 vs 999", np.abs(z[1] - y[1]).max(), np.abs(z[3] - y[3]).max())
    print(" J", x[3])
