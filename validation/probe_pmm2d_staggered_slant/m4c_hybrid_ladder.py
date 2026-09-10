"""M4c -- does the INDEPENDENT hybrid converge TOWARD the prototype's answer?

M4 shows the hybrid slant metric landing `5.4e-03` (normal) and `4.2e-02`
(conical) from the prototype's metric layer at ``n_orders = 7``, with the
hybrid's own `5 -> 7` drift still large.  The hybrid has a documented Fourier
truncation floor that the pure staggered solver does not, so the question that
matters is DIRECTION: as the hybrid's truncation is lifted, does it walk toward
the prototype?

Reference: the prototype metric layer at ``Nx = 4, M = 6`` (M4 measures its
distance to the ``Nx = 8, M = 4`` rung at `1.4e-04` / `3.1e-04`, i.e. two to
three decades below the hybrid's own drift, so it is a fixed target on this
scale).  Same geometry as M4.  The hybrid's even-parity fold is OFF.
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
t00 = time.time()


def cell_at(N, sx, sy):
    h = PX / N
    c = np.full((N, N), 1.0 + 0j)
    for i in range(N):
        for j in range(N):
            xc = (i * h + 0.5 * h - sx) % PX
            yc = (j * h + 0.5 * h - sy) % PY
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


for mount, th, ph, slant in (("normal", 0.0, 0.0, (TX, TX)),
                             ("conical20_35", np.deg2rad(20), np.deg2rad(35),
                              (TX, 0.0))):
    o, R, T, _J, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": cell_at(4, 0.0, 0.0),
                  "slant": slant}],
        NSUP, NSUB, WL, M=6, n_orders=3, theta=th, phi=ph)
    ref = po(o, R, T)
    rows = {}
    prev = None
    print(f"\n[{mount}]  prototype metric layer Nx=4 M=6 is the reference",
          flush=True)
    for no in (5, 7, 9, 11):
        st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                              degree=9, n_orders=no, symmetry=False,
                              formulation="li")
        # the hybrid's PUBLIC slant sign is the NEGATIVE of the prototype's
        st.add_layer(DEPTH, eps_cell=cell_at(48, 0.0, 0.0),
                     slant=(-slant[0], -slant[1]))
        st.set_source(WL, theta=th, phi=ph)
        tt = time.time()
        out = st.solve()
        el = time.time() - tt
        v = po(out[0], np.atleast_2d(out[1]), np.atleast_2d(out[2]))
        rows[f"n{no}"] = {"vs_prototype": dd(v, ref), "t_s": el,
                          "own_step": dd(v, prev) if prev is not None else None}
        print(f"  hybrid n_orders={no:2d}  vs prototype {dd(v, ref):.3e}"
              + (f"  (own step {dd(v, prev):.3e})" if prev is not None else "")
              + f"  {el:.1f}s", flush=True)
        prev = v
    res[mount] = rows

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m4c_hybrid_ladder.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m4c_hybrid_ladder.json  ({res['wall_s']:.1f} s)")
