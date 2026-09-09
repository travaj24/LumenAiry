"""M4d -- THE ARBITER for M4b's plateau.

M4b's pure-solver staircase of the x-slanted pillar does not converge toward the
prototype's metric layer at any admissible slice count: `1.40e-01` (n=1),
`2.06e-01` (n=2), `2.07e-01` (n=3) at `Nx = 12`, and the SAME numbers at
`Nx = 8`.  Two readings are consistent with that -- the metric layer is wrong,
or the staircase is nowhere near its own limit -- and energy closure cannot
separate them (`1.7e-05 .. 3.0e-05` on EVERY row, including the ones `2e-01`
apart: the lossless trap).

This adjudicates it with an INDEPENDENT engine on the EXACT M4b geometry
(`px = py = 1.2`, pillar `[0.3, 0.6]^2`, `depth = 0.8`, `t_x = 0.75`, `t_y = 0`,
NORMAL incidence): the shipped hybrid slant metric, on its own `n_orders`
ladder, plus the prototype's own convergence in `(Nx, M)`.  If the hybrid walks
toward the prototype while the staircase sits at `2e-01`, the staircase is the
one that is far from converged.

Also reported: what a VERTICAL pillar (the `n = 1` staircase) is worth, i.e.
how much of the `1.4e-01` is simply "the slant matters".
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402
from lumenairy.elements.pmm import PMM2DStackHybrid  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 1.2
WL, DEPTH, TX = 1.0, 0.8, 0.75
NSUP, NSUB = 1.0, 1.5
ORD_CMP = [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]
res = {"lumenairy": assert_worktree()}
t0 = time.time()


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


print("prototype metric layer -- its own convergence in (Nx, M)", flush=True)
proto = {}
for N, M in ((4, 5), (4, 6), (8, 3), (8, 4), (12, 3)):
    tt = time.time()
    o, R, T, _J, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": cell_at(N, 0.0),
                  "slant": (TX, 0.0)}],
        NSUP, NSUB, WL, M=M, n_orders=3, theta=0.0, phi=0.0)
    proto[f"N{N}_M{M}"] = po(o, R, T)
    print(f"  Nx={N} M={M}  dim={4 * (N * (M - 1)) ** 2:5d}  "
          f"{time.time() - tt:6.1f}s", flush=True)
ref = proto["N8_M4"]
res["proto_selfmove"] = {k: dd(v, ref) for k, v in proto.items() if v is not ref}
print("  self-move vs Nx=8,M=4: " + " ".join(
    f"{k} {v:.2e}" for k, v in res["proto_selfmove"].items()), flush=True)

print("\nhybrid slant metric -- its own n_orders ladder, vs the prototype",
      flush=True)
hyb, prev = {}, None
for no in (5, 7, 9, 11):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          degree=9, n_orders=no, symmetry=False,
                          formulation="li")
    st.add_layer(DEPTH, eps_cell=cell_at(48, 0.0), slant=(-TX, 0.0))
    st.set_source(WL, theta=0.0, phi=0.0)
    tt = time.time()
    out = st.solve()
    el = time.time() - tt
    v = po(out[0], np.atleast_2d(out[1]), np.atleast_2d(out[2]))
    hyb[no] = {"vs_proto": dd(v, ref), "t_s": el,
               "own_step": dd(v, prev) if prev is not None else None}
    print(f"  n_orders={no:2d}  vs prototype {dd(v, ref):.3e}"
          + (f"  (own step {dd(v, prev):.3e})" if prev is not None else "")
          + f"  {el:.1f}s", flush=True)
    prev = v
res["hybrid_ladder"] = hyb

print("\ncontext: how far is a VERTICAL pillar from the slanted answer?",
      flush=True)
o, R, T, _J, _Jt, _i = solve_slant_stack(
    PX, PY, [{"thickness": DEPTH, "cell": cell_at(8, 0.0),
              "slant": (0.0, 0.0)}],
    NSUP, NSUB, WL, M=4, n_orders=3, theta=0.0, phi=0.0)
res["vertical_vs_slanted"] = dd(po(o, R, T), ref)
print(f"  vertical pillar vs slanted: {res['vertical_vs_slanted']:.3e}")

res["wall_s"] = time.time() - t0
with open(os.path.join(OUT, "m4d_arbiter_xslant.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m4d_arbiter_xslant.json  ({res['wall_s']:.1f} s)")
