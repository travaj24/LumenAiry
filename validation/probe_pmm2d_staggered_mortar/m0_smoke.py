"""M0 -- algebra smoke: the 1-D cross-mass reduces to the mass, the Kronecker
apply is exact, and the block Grams built from 1-D factors ARE the eigensolver's
``-Rmat`` blocks."""
import numpy as np
from mortar2d import guard, cross_mass_1d, GridOps, kron_apply
print("lumenairy:", guard())
from lumenairy.elements.pmm.twod_staggered import Granet2DTransverseE

PX = PY = 0.9e-6
a0x, a0y = 0.31 / PX, 0.0
taux, tauy = np.exp(-1j * a0x * PX), np.exp(-1j * a0y * PY)

rows = []
for N, M in [(2, 5), (3, 6), (4, 4)]:
    g = GridOps(PX, PY, N, M, taux, tauy)
    for which, Mref in (("Btilde", g.Mtt_x), ("B", g.Mbb_x)):
        C = cross_mass_1d(g.bx, g.bx, which)
        rows.append((f"N={N} M={M} cross({which},self) vs mass",
                     float(np.max(np.abs(C - Mref))) / float(np.max(np.abs(Mref)))))
    # dense kron vs separable apply
    ga = GridOps(PX, PY, N, M, taux, tauy)
    X = np.random.default_rng(0).normal(size=(ga.qq, 3)) + 0j
    dense = np.kron(ga.V1[0], ga.V1[1]) @ X
    sep = kron_apply(ga.V1[0], ga.V1[1], X)
    rows.append((f"N={N} M={M} kron_apply vs dense",
                 float(np.max(np.abs(dense - sep))) / float(np.max(np.abs(dense)))))
    # Grams vs -Rmat
    cell = np.full((N, N), 2.0 + 0j)
    cell[0, 0] = 6.0
    sol = Granet2DTransverseE(PX, PY, N, N, M, cell, alpha0x=a0x, alpha0y=a0y,
                              k0=2 * np.pi / 0.6e-6)
    G = -sol.Rmat
    qq = sol.q * sol.q
    G1 = np.kron(ga.V1[0], ga.V1[1])
    G2 = np.kron(ga.V2[0], ga.V2[1])
    rows.append((f"N={N} M={M} G1 vs -Rmat[:qq,:qq]",
                 float(np.max(np.abs(G1 - G[:qq, :qq]))) / float(np.max(np.abs(G[:qq, :qq])))))
    rows.append((f"N={N} M={M} G2 vs -Rmat[qq:,qq:]",
                 float(np.max(np.abs(G2 - G[qq:, qq:]))) / float(np.max(np.abs(G[qq:, qq:])))))

for name, v in rows:
    print(f"{name:46s} rel {v:.3e}")
print("MAX", max(v for _n, v in rows))
