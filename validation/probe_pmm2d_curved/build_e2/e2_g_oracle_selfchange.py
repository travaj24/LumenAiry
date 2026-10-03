"""E2-G5: the brute force's OWN convergence (n = 20 vs 28) on the circle
pair with the y-wall at 0.21 (below), 0.27 (crossing) and 0.24 + 1e-6 (the
graze), against the kernel.  Run 2026-10-03 on Windows; the remaining two
cases of the original scratch run were stopped (cost); its three printed
lines are recorded in e2_g_oracle_selfchange.json."""
import sys
import warnings

sys.path.insert(0, "C:/tmp/lum_curved_e2/validation/probe_pmm2d_curved/verify_e2")
sys.path.insert(0, "C:/tmp/lum_curved_e2/validation/probe_pmm2d_curved/build_e2")
import numpy as np
from v2_brute import blocks_err, brute_cross, circle_map, grid

from lumenairy.elements.pmm import _curvemortar as CMM

for xw, yw in ((0.45, 0.24 - 0.03), (0.45, 0.24 + 0.03), (0.45, 0.24 + 1e-6), (0.2, 0.24 + 1e-6), (0.2, 0.24 - 0.03)):
    ga = grid(circle_map(), 4)
    gb = grid(None, 4, walls=([0.0, xw, 1.2], [0.0, yw, 1.2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
        X2 = CMM.curved_cross_mass(ga, gb, 2 * n)
    Xb = brute_cross(ga, gb, n=20, verbose=False)[0]
    Xb2 = brute_cross(ga, gb, n=28, verbose=False)[0]
    print(xw, yw, "kernel vs brute", blocks_err(X, Xb, ga.qq, gb.qq)["all"], "brute20 vs 28", blocks_err(Xb2, Xb, ga.qq, gb.qq)["all"], "kernel n vs 2n", np.abs(X - X2).max() / np.abs(X).max(), "n", n, flush=True)
