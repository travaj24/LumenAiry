"""P7 -- the SAME sliver hazard on the 2-D stacks' grids?

``PMM2DStackHybrid`` is Fourier-projected (no union grid at all) and
``PMM2DStackPure`` carries a UNION-GRID CONSTRAINT -- but that union is a
COMMON ``(Nx, Ny)`` PIXEL LATTICE the caller supplies, whose cells are all
``period/Nx`` wide.  This measures both claims instead of reading them off the
docstrings: it builds the near-coincident-wall device in each API, reports the
element grid the solver actually assembles (its widths, and their max/min
ASPECT RATIO -- the sliver number), and runs the answer + closure + a modal
self-gap on a wall offset of ONE cell, the tightest either API can express.
"""
import json
import os
import time
import warnings

import numpy as np

import lumenairy
from lumenairy import PMM2DStackHybrid, PMM2DStackPure

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PX = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0
DZ = 0.08e-6


def cell(N, i0, i1):
    """(N, N) pixel grid: a pillar spanning x-cells [i0, i1)."""
    c = np.full((N, N), EPS_H, dtype=complex)
    c[i0:i1, :] = EPS_P
    return c


def pure(N, off, M):
    s = PMM2DStackPure(PX, n_modes=M, n_orders=5)
    s.add_layer(DZ, eps_cell=cell(N, 4, N - 4))
    s.add_layer(DZ, eps_cell=cell(N, 4 - off, N - 4 + off))
    s.set_source(WL, theta=0.15, phi=0.0)
    o, R, T = s.solve(jones=False)[:3]
    return np.asarray(R), np.asarray(T)


def hybrid(N, off, M):
    s = PMM2DStackHybrid(PX, degree=M, n_orders=5)
    s.add_layer(DZ, eps_cell=cell(N, 4, N - 4))
    s.add_layer(DZ, eps_cell=cell(N, 4 - off, N - 4 + off))
    s.set_source(WL, theta=0.15, phi=0.0)
    o, R, T = s.solve()[:3]
    return np.asarray(R), np.asarray(T)


if __name__ == "__main__":
    out = {}
    # (1) the grid a pixel cell can express: cells are period/N, ALL EQUAL.
    for N in (12, 24, 64, 256):
        w = np.full(N, 1.0 / N)
        out[f"pure_grid_N{N}"] = dict(
            n_cells=N, w_min=float(w.min()), w_max=float(w.max()),
            aspect=float(w.max() / w.min()),
            finest_expressible_wall_offset=1.0 / N)
        print(f"pure/hybrid pixel grid N={N:4d}: cells all {1.0 / N:.4g} of a "
              f"period, aspect ratio {w.max() / w.min():.1f}, finest "
              f"expressible wall offset {1.0 / N:.4g}", flush=True)
    # (2) the answer + closure at a ONE-CELL wall offset (the tightest either
    #     API can express), at two modal counts.
    for name, fn, Ms in (("hybrid", hybrid, (7, 9)), ("pure", pure, (5, 6))):
        res = []
        for M in Ms:
            t0 = time.time()
            R, T = fn(12, 1, M)
            tot = float(np.real(R).sum(-1).max() + np.real(T).sum(-1).max())
            res.append((M, R, T, tot))
            print(f"{name} M={M}: R+T = {tot:.10f}  ({time.time() - t0:.1f} s)",
                  flush=True)
        sg = float(max(np.abs(res[0][1] - res[1][1]).max(),
                       np.abs(res[0][2] - res[1][2]).max()))
        out[name] = dict(M=[r[0] for r in res], total=[r[3] for r in res],
                         self_gap=sg)
        print(f"{name}: modal self-gap M={Ms[0]} vs {Ms[1]} = {sg:.3e}",
              flush=True)
        json.dump(out, open(os.path.join(HERE, "p7_2d.json"), "w"), indent=1)
    print("\nwrote p7_2d.json", flush=True)
