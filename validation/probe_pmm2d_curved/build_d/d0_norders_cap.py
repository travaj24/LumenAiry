"""N-3 (Phase B verifier note): the mapped shared-grid stack has no
``n_orders <= (q - 1) // 2`` cap while the per-layer path raises above it.

Measures, on the 3 x 3 circle (q = 3 (M - 1)), the R / T of the nine low
orders at n_orders = 2 (under the cap) and at n_orders above the cap, and the
lossless closure of each.  Since Phase C the mapped incident field is an
exact modal decomposition (window-free) and the far field is a quadrature
INTEGRAL of the represented field against each plane wave -- well defined for
any order -- so the per-layer path's reason for the cap (order slots of the
separable projector aliasing one another inside the least-squares overlap)
does not arise.  Writes d0_norders_cap.json."""
import sys

import _common as C
import numpy as np

out = {}
for M in (4, 5):
    q = 3 * (M - 1)
    cap = (q - 1) // 2
    rows = {}
    for no in sorted({2, cap, cap + 2, cap + 5}):
        cm, eps = C.circle3()
        o, R, T, _J = C.solve(cm, eps, M, n_orders=no)
        v = C.vec(o, R, T)
        clos = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
        rows[no] = {"vec": v, "closure": clos}
    base = rows[2]["vec"]
    out[f"M{M}"] = {"q": q, "cap": cap,
                    "max_low_order_change_vs_n2": {
                        str(k): float(np.max(np.abs(r["vec"] - base)))
                        for k, r in rows.items()},
                    "closure": {str(k): r["closure"] for k, r in rows.items()}}
    print(M, out[f"M{M}"], flush=True)
C.dump("d0_norders_cap.json", out)
sys.exit(0)
