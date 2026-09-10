"""M3d -- does the 2-D staggered TM channel's ~1e-4 residual belong to the SLANT
or to the VERTICAL basis?  A deeper M-ladder (5..10) on the SAME stripe at
slant 0 and slant 35 deg against the converged covariant 1-D oracle.

If the slanted arm tracks the vertical arm at every M, the residual is the
shipped basis's own wall-normal behaviour on this cell and the slant adds
nothing.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    pmm_jones_1d, pmm_jones_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 0.75
WL, DEPTH = 1.0, 0.30
NR, NG, NSUP, NSUB = 2.0, 1.0, 1.0, 1.5
ORD_CMP = [-2, -1, 0, 1, 2]
CELL = np.array([[NR ** 2, NR ** 2], [NG ** 2, NG ** 2]], dtype=complex)
res = {"lumenairy": assert_worktree(), "rows": []}
t0 = time.time()


def orc(phi_deg, theta, degree=30):
    if phi_deg == 0.0:
        o, R, T, _J = pmm_jones_1d(
            PX, (NR ** 2) * np.eye(3), (NG ** 2) * np.eye(3), NSUB, NSUP,
            DEPTH, 0.5, WL, angle=theta, degree=degree, far_field_orders=15)
    else:
        o, R, T, _J = pmm_jones_1d_slanted(
            PX, (NR ** 2) * np.eye(3), (NG ** 2) * np.eye(3), NSUB, NSUP,
            DEPTH, 0.5, WL, np.deg2rad(phi_deg), angle=theta, degree=degree,
            far_field_orders=15, factorization="covariant")
    return {"tm": {int(m): (float(R[0, i]), float(T[0, i]))
                   for i, m in enumerate(o)},
            "te": {int(m): (float(R[1, i]), float(T[1, i]))
                   for i, m in enumerate(o)}}


def two_d(t, theta, M):
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": CELL, "slant": (t, 0.0)}],
        NSUP, NSUB, WL, M=M, n_orders=4, theta=theta, phi=0.0)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    return {"tm": {m: (float(R[0, i]), float(T[0, i])) for m, i in idx.items()},
            "te": {m: (float(R[1, i]), float(T[1, i])) for m, i in idx.items()}}


def d(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


for phi in (0.0, 35.0):
    ref = orc(phi, 0.0)
    for M in (5, 6, 7, 8, 9, 10):
        tt = time.time()
        d2 = two_d(-np.tan(np.deg2rad(phi)), 0.0, M)
        row = {"phi": phi, "M": M, "dim": int(4 * (2 * (M - 1)) ** 2),
               "tm": d(d2["tm"], ref["tm"]), "te": d(d2["te"], ref["te"]),
               "t_s": time.time() - tt}
        res["rows"].append(row)
        print(f"M3d phi={phi:4.0f} M={M:2d} dim={row['dim']:5d}  "
              f"TM {row['tm']:.2e}  TE {row['te']:.2e}  ({row['t_s']:.1f}s)")

res["wall_s"] = time.time() - t0
with open(os.path.join(OUT, "m3d_tm_deep_ladder.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/m3d_tm_deep_ladder.json  ({res['wall_s']:.1f} s)")
