"""M4c -- the DECISIVE comparison: EQUAL DEGREES OF FREEDOM.

Every M2/M3/M4/M6 gap so far is dominated by the coarse layer's own
resolution, not by the mortar (M1, M0b(b) and M5(a) put the mortar's own error
at 1e-13 .. 1e-16).  But a union grid does not merely 'cost more' -- it SPENDS
its extra degrees of freedom on h-refinement, and the per-layer route spends
its own on p-refinement, so the two must be compared at EQUAL DOF, not at
equal modal count.

The staggered basis makes that exact.  Per axis, q = N (M - 1).  The stripe
pair of M4 has A on N=2 and B on N=3 against a union on N=6, so

    q_union(M) = 6(M-1)  ==  q_A(M_A) = 2(M_A-1)   <=>  M_A = 3M - 2
                         ==  q_B(M_B) = 3(M_B-1)   <=>  M_B = 2M - 1

and at those settings EVERY region eigenproblem in the two arms has exactly
the same dimension 2 q^2.  Whatever separates them is then the h-vs-p question
alone: does h-refinement at the NEIGHBOUR's walls (which only the union grid
can supply, because Granet's lattice is uniform and admits no local
enrichment) buy accuracy that p-refinement on the own-walls grid cannot?

Scored against the exact 1-D ``PMMStack`` oracle, as M4."""
import json
import sys
import time
import warnings

import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
PX = PY = 0.9e-6
WL = 0.60e-6
THETA, NORD = 0.20, 2
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
B3 = np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3], np.complex128)
tA, tB = 0.20e-6, 0.15e-6

s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=14)
s.add_layer(tA, segments=[(1 / 2, 2.0), (1 / 2, 6.0)])
s.add_layer(tB, segments=[(1 / 3, 9.0), (1 / 3, 2.0), (1 / 3, 4.0)])
s.set_source(WL, theta=THETA)
o, R, T = s.solve()[:3]
o = np.asarray(o).ravel()
i = np.argsort(o)
MO, RO, TO = o[i], R[1][i], T[1][i]


def score(orders, R, T):
    oo = np.asarray(orders)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    j = np.argsort(m)
    m, r, t = m[j], R[1][sel][j], T[1][sel][j]
    keep = np.isin(MO, m)
    return max(float(np.abs(r - RO[keep]).max()),
               float(np.abs(t - TO[keep]).max()))


rows = []
for M in [int(x) for x in (sys.argv[1:] or [4, 5, 6])]:
    q = 6 * (M - 1)
    MA, MB = 3 * M - 2, 2 * M - 1
    t0 = time.perf_counter()
    u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    u.add_layer(tA, eps_cell=refine_cell(A2, 3))
    u.add_layer(tB, eps_cell=refine_cell(B3, 2))
    u.set_source(WL, theta=THETA, phi=0.0)
    ou, Ru, Tu = u.solve(jones=False)
    tu = time.perf_counter() - t0
    eu = score(ou, Ru, Tu)
    cu = max(abs(Ru[p].sum() + Tu[p].sum() - 1.0) for p in (0, 1))

    t0 = time.perf_counter()
    mo = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
    mo.add_layer(tA, eps_cell=A2, n_modes=MA)
    mo.add_layer(tB, eps_cell=B3, n_modes=MB)
    mo.set_source(WL, theta=THETA, phi=0.0)
    om, Rm, Tm = mo.solve(jones=False)
    tm = time.perf_counter() - t0
    em = score(om, Rm, Tm)
    cm = max(abs(Rm[p].sum() + Tm[p].sum() - 1.0) for p in (0, 1))
    rows.append(dict(M=M, q=q, MA=MA, MB=MB, err_union=eu, err_mortar=em,
                     clo_union=float(cu), clo_mortar=float(cm),
                     t_union=tu, t_mortar=tm))
    print(f"q={q:3d} (union M={M} | mortar M_A={MA}, M_B={MB}; every region "
          f"eig is {2*q*q}x{2*q*q})\n"
          f"    union  err {eu:9.2e}  closure {cu:8.1e}  {tu:7.1f}s\n"
          f"    mortar err {em:9.2e}  closure {cm:8.1e}  {tm:7.1f}s\n"
          f"    h-vs-p penalty (mortar/union) {em/eu:8.2f}x", flush=True)

json.dump(rows, open(
    "validation/probe_pmm2d_staggered_mortar/m4c_equal_dof.json", "w"), indent=1)
