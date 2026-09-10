"""M2b -- is the M2 gap the MORTAR or the coarse layer's own resolution?

Per-layer grids make the modal count PER LAYER free, so the question is
answered directly: hold layer B at (N=4, M_B) and walk layer A's modal count
M_A on its own N=2 grid.  If the gap against the union-grid reference collapses
with M_A, the mortar is consistent and the M2 gap is A's under-resolution; if
it plateaus, the mortar carries a variational crime that does not vanish.

Also runs the FAIL-BEFORE control on the 2-D-specific design choice
(``H_BLOCK_SWAP``), and the conforming control (A on N=4 through the mortar
code path)."""
import json
import sys
import time

import mortar2d
import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

PX = PY = 0.9e-6
WL = 0.60e-6
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
B4 = np.full((4, 4), 2.0 + 0j)
B4[1:3, 1:3] = 9.0
B4[0, 3] = 4.0
tA, tB = 0.20e-6, 0.15e-6
THETA, PHI, NORD = 0.20, 0.0, 2
MB = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def obs(R, T, J):
    return np.concatenate([R.ravel(), T.ravel()])


def run_ref(M):
    s = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(tA, eps_cell=refine_cell(A2, 2))
    s.add_layer(tB, eps_cell=B4)
    s.set_source(WL, theta=THETA, phi=PHI)
    return s.solve()


def run_mor(NA, MA, MB, swap=True):
    old = mortar2d.H_BLOCK_SWAP
    mortar2d.H_BLOCK_SWAP = swap
    try:
        s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
        s.add_layer(tA, eps_cell=(A2 if NA == 2 else refine_cell(A2, 2)),
                    n_modes=MA)
        s.add_layer(tB, eps_cell=B4, n_modes=MB)
        s.set_source(WL, theta=THETA, phi=PHI)
        out = s.solve()
    finally:
        mortar2d.H_BLOCK_SWAP = old
    return out, s


# ---- the reference and its OWN convergence -------------------------------
refs = {}
for M in (MB, MB + 2):
    t0 = time.perf_counter()
    refs[M] = run_ref(M)
    print(f"reference union(N=4) M={M}: {time.perf_counter()-t0:6.1f}s  "
          f"sumR(Ey) {refs[M][1][1].sum():.8f}", flush=True)
o0, R0, T0, J0 = refs[MB]
oF, RF, TF, JF = refs[MB + 2]
sc = max(float(np.max(R0)), float(np.max(T0)))
print(f"reference self-gap M={MB} vs {MB+2}: "
      f"{np.abs(obs(R0,T0,J0)-obs(RF,TF,JF)).max()/sc:.3e}\n", flush=True)

rows = []
for MA in [int(x) for x in (sys.argv[2:] or [5, 7, 9, 11, 13])]:
    t0 = time.perf_counter()
    (o1, R1, T1, J1), s = run_mor(2, MA, MB)
    dt = time.perf_counter() - t0
    d_ref = float(np.abs(obs(R1, T1, J1) - obs(R0, T0, J0)).max()) / sc
    d_fin = float(np.abs(obs(R1, T1, J1) - obs(RF, TF, TF)).max()) / sc
    d_fin = float(np.abs(obs(R1, T1, J1) - obs(RF, TF, JF)).max()) / sc
    clo = max(abs(R1[p].sum() + T1[p].sum() - 1.0) for p in (0, 1))
    rows.append(dict(MA=MA, MB=MB, d_ref=d_ref, d_reffine=d_fin,
                     closure=float(clo), t=dt))
    print(f"  mortar A(N=2,M={MA:2d}) | B(N=4,M={MB})  vs ref(M={MB}) "
          f"{d_ref:9.2e}   vs ref(M={MB+2}) {d_fin:9.2e} | "
          f"closure {clo:8.1e} | {dt:6.1f}s", flush=True)

# conforming control through the mortar path
(o2, R2, T2, J2), _ = run_mor(4, MB, MB)
print(f"\nCONFORMING control mortar A(N=4,M={MB}) vs union ref: "
      f"{np.abs(obs(R2,T2,J2)-obs(R0,T0,J0)).max()/sc:.3e}", flush=True)

# fail-before: the H-row block swap
(o3, R3, T3, J3), _ = run_mor(2, max(11, MB), MB, swap=False)
(o4, R4, T4, J4), _ = run_mor(2, max(11, MB), MB, swap=True)
print(f"H_BLOCK_SWAP=False  vs ref(M={MB+2}): "
      f"{np.abs(obs(R3,T3,J3)-obs(RF,TF,JF)).max()/sc:.3e}   "
      f"closure {max(abs(R3[p].sum()+T3[p].sum()-1.0) for p in (0,1)):.2e}")
print(f"H_BLOCK_SWAP=True   vs ref(M={MB+2}): "
      f"{np.abs(obs(R4,T4,J4)-obs(RF,TF,JF)).max()/sc:.3e}   "
      f"closure {max(abs(R4[p].sum()+T4[p].sum()-1.0) for p in (0,1)):.2e}")

json.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m2b_converge.json",
                     "w"), indent=1)
