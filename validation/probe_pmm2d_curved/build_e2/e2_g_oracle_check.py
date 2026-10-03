"""E2-G2: is the residual of the CIRCLE graze (kernel vs the verifier's
brute force ~6.5e-7, independent of the graze depth) the kernel's or the
oracle's?  The kernel is continuous across the graze (X moves linearly in
the wall position, 1.92 per unit, delta = -1e-3 .. +1e-3, no jump at 0).
Here: kernel vs brute with the wall just BELOW the circle (delta = -1e-6,
-1e-3: no cut exists, so the kernel has nothing to miss) and touching it
(delta = 0)."""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "verify_e2"))
from _common import dump  # noqa: E402
from v2_brute import blocks_err, brute_cross, circle_map, grid  # noqa: E402

from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402

rows = []
for d in (-1e-3, -1e-6, 0.0):
    ga = grid(circle_map(), 4)
    gb = grid(None, 4, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + d, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = CMM.curved_cross_mass_adaptive(ga, gb)[0]
    Xb = brute_cross(ga, gb, n=20, verbose=False)[0]
    r = dict(delta=d, err=blocks_err(X, Xb, ga.qq, gb.qq)["all"])
    rows.append(r)
    print(r, flush=True)
dump("e2_g_oracle_check.json", dict(rows=rows))
