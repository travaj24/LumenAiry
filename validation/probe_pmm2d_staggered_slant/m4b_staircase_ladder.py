"""M4b -- the pure-solver STAIRCASE, and the structural reason it is expensive.

The pure staggered solver's segments are a ``linspace`` (``Basis1D.__init__``),
so every slice of a z-staircase must put its walls on ONE uniform union grid of
spacing ``h = px / Nx``.  Two consequences, and they are the finding:

1. the admissible slice counts are exactly the divisors of ``S / h`` (``S`` =
   the total lateral walk) -- the ladder is fixed by the geometry and the grid,
   not chosen;
2. the smallest non-zero per-slice lateral STEP is ``h`` itself.  So the
   staircase's geometric error is bounded below by ``h`` relative to the
   feature, and the ONLY way to reduce it is to refine the union grid -- which
   costs ``(Nx (M-1))^6`` in the region eig.

This runs the SAME geometry as M4 (pillar ``[0.3, 0.6]^2``, ``px = py = 1.2``,
``depth = 0.8``, ``t_x = 0.75`` so ``S = 0.6 = px/2``) on TWO union grids,
``Nx = 8`` (``h = 0.15``, ``n = 1, 2, 4``) and ``Nx = 12`` (``h = 0.1``,
``n = 1, 2, 3, 6``), with LEADING-EDGE sampling (the only one whose walls land
on the grid for every ``n``).  Reference at each grid is the METRIC layer on
that same grid, so the discretization is common-mode and the difference is the
staircase's geometric error alone.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 1.2
WL, DEPTH, TX = 1.0, 0.8, 0.75          # S = TX * DEPTH = 0.6 = PX / 2
NSUP, NSUB = 1.0, 1.5
ORD_CMP = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]
res = {"lumenairy": assert_worktree()}
t00 = time.time()


def cell_at(N, sx):
    h = PX / N
    c = np.full((N, N), 1.0 + 0j)
    for i in range(N):
        for j in range(N):
            xc = (i * h + 0.5 * h - sx) % PX
            yc = (j * h + 0.5 * h) % PY
            if 0.3 <= xc < 0.6 and 0.3 <= yc < 0.6:
                c[i, j] = 4.0
    return c


def po(o, R, T):
    idx = {(int(m), int(n)): i for i, (m, n) in enumerate(o)}
    return {k: (float(R[0, idx[k]]), float(R[1, idx[k]]),
                float(T[0, idx[k]]), float(T[1, idx[k]]))
            for k in ORD_CMP if k in idx}


def dd(a, b):
    return max(max(abs(x - y) for x, y in zip(a[k], b[k]))
               for k in a if k in b)


def run(N, M, th, ph, nsl=None):
    if nsl is None:
        layers = [{"thickness": DEPTH, "cell": cell_at(N, 0.0),
                   "slant": (TX, 0.0)}]
    else:
        h = DEPTH / nsl
        layers = [{"thickness": h,
                   "cell": cell_at(N, TX * DEPTH * i / nsl),
                   "slant": (0.0, 0.0)} for i in range(nsl)]
    tt = time.time()
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, layers, NSUP, NSUB, WL, M=M, n_orders=3, theta=th, phi=ph)
    return (po(o, R, T), time.time() - tt,
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


PLAN = (("normal", 0.0, 0.0, 8, 3, (1, 2, 4)),
        ("conical20_35", np.deg2rad(20), np.deg2rad(35), 8, 3, (1, 2, 4)),
        ("normal", 0.0, 0.0, 12, 3, (1, 2, 3, 6)))

for mount, th, ph, N, M, ns in PLAN:
    key = f"{mount}/Nx{N}_M{M}"
    h = PX / N
    met, tm, cm = run(N, M, th, ph)
    rows = {"h": h, "step_over_pillar_width": h / 0.3,
            "metric": {"t_s": tm, "closure": cm,
                       "dim": int(4 * (N * (M - 1)) ** 2)}}
    print(f"\n[{key}]  h = {h:.3f} ({h / 0.3:.2f} pillar widths)  "
          f"metric layer {tm:6.1f}s  |R+T-1| {cm:.2e}", flush=True)
    for n in ns:
        v, t, clo = run(N, M, th, ph, nsl=n)
        rows[f"n{n}"] = {"vs_metric": dd(v, met), "t_s": t, "closure": clo,
                         "lateral_step": TX * DEPTH / n,
                         "cost_ratio": t / tm}
        print(f"  staircase n={n:2d}  step {TX * DEPTH / n:.3f} "
              f"({TX * DEPTH / n / 0.3:.2f} widths)  vs metric "
              f"{dd(v, met):.3e}  {t:6.1f}s ({t / tm:.2f}x)  "
              f"|R+T-1| {clo:.2e}", flush=True)
    res[key] = rows

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m4b_staircase_ladder.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m4b_staircase_ladder.json  ({res['wall_s']:.1f} s)")
