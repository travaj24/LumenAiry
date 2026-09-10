"""M4b -- the per-layer MODAL COUNT is the lever the union grid cannot pull.

Same stripe pair as M4 (A duty 1/2 on N=2, B duty 1/3 on N=3, genuinely
non-conforming), scored against the exact 1-D ``PMMStack`` oracle.  The union
grid forces ONE modal count on the common refinement N=6; the mortar lets each
layer carry its own, so the coarse layer's under-resolution -- which is what
dominates every M2/M3 gap -- is bought off where it is cheap.

Reported per point: the error against the oracle, the wall time, and the
eig-work proxy sum(dim^3) with dim = 2 (N (M-1))^2."""
import json
import sys
import time
import warnings
import numpy as np
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
PX = PY = 0.9e-6
WL = 0.60e-6
THETA, NORD = 0.20, 2
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
B3 = np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3], np.complex128)
A_SEGS = [(1 / 2, 2.0), (1 / 2, 6.0)]
B_SEGS = [(1 / 3, 9.0), (1 / 3, 2.0), (1 / 3, 4.0)]
tA, tB = 0.20e-6, 0.15e-6

s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=14)
s.add_layer(tA, segments=A_SEGS)
s.add_layer(tB, segments=B_SEGS)
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


def work(pairs):
    return sum((2 * (N * (M - 1)) ** 2) ** 3 for N, M in pairs)


rows = []
print("MORTAR  A(N=2, M_A) | B(N=3, M_B=7)")
for MA in [int(x) for x in (sys.argv[1:] or [7, 9, 11, 13, 15])]:
    t0 = time.perf_counter()
    st = MortarStack2D(PX, PY, n_modes=7, n_orders=NORD)
    st.add_layer(tA, eps_cell=A2, n_modes=MA)
    st.add_layer(tB, eps_cell=B3, n_modes=7)
    st.set_source(WL, theta=THETA, phi=0.0)
    oo, RR, TT = st.solve(jones=False)
    dt = time.perf_counter() - t0
    e = score(oo, RR, TT)
    w = work([(2, MA), (3, 7)])
    rows.append(dict(arm="mortar", MA=MA, MB=7, err=e, t=dt, work=w))
    print(f"  M_A={MA:2d}  err {e:9.2e}  {dt:7.1f}s   eig-work {w:.3e}",
          flush=True)

print("\nUNION grid N=6 (one modal count for the whole stack)")
for M in (4, 5, 6, 7):
    t0 = time.perf_counter()
    st = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    st.add_layer(tA, eps_cell=refine_cell(A2, 3))
    st.add_layer(tB, eps_cell=refine_cell(B3, 2))
    st.set_source(WL, theta=THETA, phi=0.0)
    oo, RR, TT = st.solve(jones=False)
    dt = time.perf_counter() - t0
    e = score(oo, RR, TT)
    w = work([(6, M), (6, M)])
    rows.append(dict(arm="union6", M=M, err=e, t=dt, work=w))
    print(f"  M={M:2d}    err {e:9.2e}  {dt:7.1f}s   eig-work {w:.3e}",
          flush=True)

json.dump(rows, open(
    "validation/probe_pmm2d_staggered_mortar/m4b_perlayer_M.json", "w"),
    indent=1)
