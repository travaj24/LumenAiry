"""V5 -- Li, J. Opt. A 5:345 (2003), Example 1, against the ORIGINAL paper's
Table 1 (read from the PDF, p. 353: columns m = 0, +1, +2 along x (d1 = 2.4
lambda), rows n = -1, 0, +1 along y (d2 = 1.4 lambda); first row of each cell
= the grating as defined, second row = the cross terms reversed, i.e. eps_a
and eps_b interchanged; REFLECTED orders (fig. 3 caption); truncation order
23 with L2 L1).  eps_a = 2.25(xx+yy) + 0.5i(xy - yx) + 2zz (surround),
eps_b = 2.25(xx+yy) - 0.5i(xy - yx) + 2zz (pillar), n(-1) = 1 + 5i (an
INDEX: eps_sub = -24 + 10i), h = lambda, w1/d1 = w2/d2 = 0.5, E in Oxz.

Arms (the verifier's own maps):
  unmapped  M     -- the shipped solver on the 2 x 2 cell
  identity  M     -- IdentityMap on the 2 x 2 walls (tensor route under a map)
  h2        M     -- asymmetric two-harmonic stretch, walls at the PREIMAGES
  sh4       M     -- a 4 x 4 TransfiniteMap whose four vertices INTERIOR to the
                     four material blocks are moved (non-diagonal J; the
                     device is unchanged: every material wall stays straight)
  un4       M     -- the shipped solver on the same 4 x 4 grid (no map)
  *_swap    M     -- the second row (eps_a <-> eps_b)
  sh4 M MUT       -- a _vdcommon.vmutate defect

Output v5_<arm>_M<M>[_<MUT>].json."""
import sys
import time
import warnings

import numpy as np
from _vdcommon import CM, PMM2DStackPure, TwoHarmonicStretch, dump, vmutate

LAM = 1.0e-6
PX, PY = 2.4 * LAM, 1.4 * LAM
LI_B = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
LI_A = np.conj(LI_B)
NSUB = 1.0 + 5.0j
ROW1 = {(0, 0): 0.2980, (1, 0): 0.1195, (2, 0): 0.0222,
        (0, -1): 0.0619, (1, -1): 0.0269, (1, 1): 0.0137}
ROW2 = {(0, 0): 0.2980, (1, 0): 0.1195, (2, 0): 0.0222,
        (0, -1): 0.0619, (1, -1): 0.0137, (1, 1): 0.0269}


def cell(pillar, host, n):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[: n // 2, : n // 2] = pillar
    return c


def setup(arm):
    base = arm.replace("_swap", "")
    n = 4 if base in ("sh4", "un4") else 2
    xw = np.linspace(0.0, PX, n + 1)
    yw = np.linspace(0.0, PY, n + 1)
    if base == "identity":
        cm = CM.IdentityMap(xw, yw, PX, PY)
    elif base == "h2":
        cm = CM.SeparableStretch.from_physical_walls(
            xw, yw, fx=TwoHarmonicStretch(0.07 * PX, 0.04 * PX, 0.9),
            fy=TwoHarmonicStretch(-0.05 * PY, 0.03 * PY, -0.4))
    elif base == "sh4":
        V = np.stack(np.meshgrid(xw, yw, indexing="ij"), axis=-1)
        for (i, j), (dx, dy) in {(1, 1): (0.05, 0.06), (3, 1): (-0.04, 0.05),
                                 (1, 3): (0.06, -0.03),
                                 (3, 3): (-0.05, -0.06)}.items():
            V[i, j] += (dx * 0.5 * PX, dy * 0.5 * PY)
        cm = CM.TransfiniteMap(xw, yw, V, None)
    else:
        cm = None
    eps = cell(LI_A, LI_B, n) if arm.endswith("_swap") else cell(LI_B, LI_A, n)
    return cm, eps


def run(arm, M, mut=None):
    cm, eps = setup(arm)
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=NSUB,
                        n_modes=M, n_orders=4, cmap=cm)
    st.add_layer(LAM, eps_cell=eps)
    st.set_source(LAM)
    return st.solve(jones=True)


if __name__ == "__main__":
    arm, M = sys.argv[1], int(sys.argv[2])
    mut = sys.argv[3] if len(sys.argv) > 3 else None
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    if mut:
        with vmutate(mut):
            o, R, T, J = run(arm, M)
    else:
        o, R, T, J = run(arm, M)
    wall = time.perf_counter() - t0
    o, R, T = np.asarray(o), np.asarray(R), np.asarray(T)
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(o)}
    R0 = {k: float(R[0, idx[k]]) for k in ROW1}
    d1 = max(abs(R0[k] - v) for k, v in ROW1.items())
    d2 = max(abs(R0[k] - v) for k, v in ROW2.items())
    # space-reversal symmetry (m, n) -> (-m, -n) of Li's grating
    sym = max(abs(float(R[0, idx[(a, b)]]) - float(R[0, idx[(-a, -b)]]))
              for (a, b) in ROW1)
    tag = f"v5_{arm}_M{M}" + (f"_{mut}" if mut else "")
    dump(tag + ".json", {"arm": arm, "M": M, "mutation": mut,
                         "R_Ex": {str(k): v for k, v in R0.items()},
                         "maxdev_row1": d1, "maxdev_row2": d2,
                         "space_reversal": sym, "R": R, "T": T,
                         "orders": o, "J": np.asarray(J), "wall_s": wall})
    print(tag, f"row1 {d1:.3e} row2 {d2:.3e} sym {sym:.1e} wall {wall:.0f}s")
