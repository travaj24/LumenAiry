"""E2-7: THREE layers on THREE different maps -- a circle (layer 1), a
sinusoidal wall along x that crosses it (layer 2), a sinusoidal wall along
y that crosses both (layer 3) -- two curved mortars in one cascade.
Lossless closure per rung; with layer 2 lossy (eps 2.25 + 0.2i), the
absorption of the two LOSSLESS layers (exactly 0 for an energy-consistent
flux form) and the budget sum(A) against 1 - R - T.

Usage: python e2_7_three.py <M>"""
import sys
import time

import numpy as np
from _common import EPS_P, R_CIRC, dump, shapes_stack, solve

from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

M = int(sys.argv[1])
out = {"M": M}


def lay(e2):
    return [(0.25, [Circle(0.6, 0.6, R_CIRC, EPS_P)], 1.0),
            (0.2, [SinusoidalWall("x", 0.6, 0.12, eps=e2)], 1.0),
            (0.2, [SinusoidalWall("y", 0.5, 0.1, eps=1.7)], 1.0)]


t0 = time.perf_counter()
st = shapes_stack(lay(2.25), M)
assert not st._perlayer_fast_ok()
o, R, T, J = solve(st)
out["closure"] = np.abs(R.sum(1) + T.sum(1) - 1.0)
out["wall"] = time.perf_counter() - t0
st = shapes_stack(lay(2.25 + 0.2j), M)
o, R, T, J = solve(st, retain=True)
A = st.layer_absorption()
budget = 1.0 - R.sum(1) - T.sum(1)
out.update(absorption=A, budget=budget, mismatch=np.abs(A.sum(0) - budget),
           lossless_layers=float(max(np.abs(A[0]).max(),
                                     np.abs(A[2]).max())))
out["R"], out["T"] = R, T
print({k: v for k, v in out.items() if k not in ("R", "T")})
dump(f"e2_7_three_M{M}.json", out)
