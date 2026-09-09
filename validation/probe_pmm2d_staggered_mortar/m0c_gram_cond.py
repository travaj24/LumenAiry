"""M0c -- the round-off scale of the M1 identity.

The mortar multiplies by the block Gram on both sides of a ``solve`` where the
plain interface does not, so the conforming identity is exact algebraically and
rounds at ``eps * cond(G)``.  This measures ``cond_2(G1)`` / ``cond_2(G2)`` on
the grids M1 uses, so the M1 bar is DERIVED and not fitted to the reading."""
import numpy as np
from mortar2d import guard, GridOps
print("lumenairy:", guard())
PX = PY = 0.9e-6
taux, tauy = np.exp(-1j * 0.31), 1.0 + 0j
eps = np.finfo(float).eps
for N, M in ((2, 5), (3, 5), (3, 6), (2, 6), (6, 4), (3, 8)):
    g = GridOps(PX, PY, N, M, taux, tauy)
    G1 = np.kron(g.V1[0], g.V1[1])
    G2 = np.kron(g.V2[0], g.V2[1])
    c1 = float(np.linalg.cond(G1))
    c2 = float(np.linalg.cond(G2))
    print(f"N={N} M={M} q={g.q:3d}  cond2(G1) {c1:9.2e}  cond2(G2) {c2:9.2e}"
          f"   eps*max {eps*max(c1,c2):9.2e}")
