"""M1c -- is the M1 null residual DISCRETIZATION or CONDITIONING?

M1/M1b leave the uniform-layer null at ~1e-8..1e-10 rather than machine
precision.  A residual that FALLS with the modal count M is discretization (the
slanted generator resolving a homogeneous medium); a FLAT one is the
conditioning of the non-normal 4q^2 pencil.  Worst M1 row: isotropic uniform,
oblique 25 deg, slant 35 deg in x.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
print(assert_worktree())
px = py = 0.9
depth, wl = 0.35, 1.0
cell = np.zeros((2, 2, 3, 3), dtype=complex)
cell[:, :] = 2.25 * np.eye(3)
th, ph = np.deg2rad(25), 0.0
sl = (np.tan(np.deg2rad(35)), 0.0)
rows = []
for M in (4, 5, 6, 7, 8):
    _o, R0, T0, Jr0, Jt0, _i = solve_slant_stack(
        px, py, [{"thickness": depth, "cell": cell, "slant": (0.0, 0.0)}],
        1.0, 1.5, wl, M=M, n_orders=4, theta=th, phi=ph)
    _o, R1, T1, Jr1, Jt1, _j = solve_slant_stack(
        px, py, [{"thickness": depth, "cell": cell, "slant": sl}],
        1.0, 1.5, wl, M=M, n_orders=4, theta=th, phi=ph)
    r = dict(M=M, dof=int(4 * (2 * (M - 1)) ** 2),
             dR=float(np.max(np.abs(R1 - R0))),
             dT=float(np.max(np.abs(T1 - T0))),
             dJr=float(np.max(np.abs(Jr1 - Jr0))),
             dJt=float(np.max(np.abs(Jt1 - Jt0))))
    rows.append(r)
    print(f"M={M} dim={r['dof']:5d} dR {r['dR']:.2e} dT {r['dT']:.2e} "
          f"dJr {r['dJr']:.2e} dJt {r['dJt']:.2e}")
with open(os.path.join(OUT, "m1c_null_ladder.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("WROTE results/m1c_null_ladder.json")
