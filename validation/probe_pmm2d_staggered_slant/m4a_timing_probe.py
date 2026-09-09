"""M4a -- sizing probe: how long is ONE region solve on the M4 union grid?

The pure staggered solver's grid is UNIFORM (``Basis1D`` segments are
``linspace``), so a z-staircase of a slanted pillar must put EVERY slice's walls
on one common uniform grid -- which is what makes a staircase expensive in this
engine.  This measures the price before M4 commits to a ladder.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import SlantSolver, assert_worktree, slant_region_modes  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE, _region_modes,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
print(assert_worktree())
PX = PY = 1.2
K0 = 2 * np.pi
rows = []
for N, M in ((4, 5), (4, 6), (8, 3), (8, 4)):
    q = N * (M - 1)
    cell = np.full((N, N), 1.0 + 0j)
    cell[N // 4:N // 2, N // 4:N // 2] = 4.0
    t = time.perf_counter()
    s = Granet2DTransverseE(PX, PY, N, N, M, cell, alpha0x=0.3, alpha0y=0.2,
                            k0=K0)
    _region_modes(s)
    t_in = time.perf_counter() - t
    t = time.perf_counter()
    s2 = SlantSolver(PX, PY, N, N, M, cell, slant=(0.75, 0.0), alpha0x=0.3,
                     alpha0y=0.2, k0=K0)
    slant_region_modes(s2)
    t_sl = time.perf_counter() - t
    rows.append({"N": N, "M": M, "q2": q * q, "inplane_dim": 2 * q * q,
                 "slant_dim": 4 * q * q, "t_inplane_s": t_in,
                 "t_slant_s": t_sl})
    print(f"  N={N} M={M} q^2={q * q:5d}  in-plane({2 * q * q:5d}) "
          f"{t_in:7.2f}s | slant({4 * q * q:5d}) {t_sl:7.2f}s")
    if t_sl > 240:
        print("  (stopping: past the 4-minute-per-solve budget)")
        break
with open(os.path.join(OUT, "m4a_timing_probe.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("WROTE results/m4a_timing_probe.json")
