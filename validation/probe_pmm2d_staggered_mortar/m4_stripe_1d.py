"""M4 -- y-uniform STRIPE stack with a DIFFERENT duty cycle per layer, against
the VALIDATED 1-D oracle (``PMMStack`` at degree 14 -- exact for a stripe
stack, no Fourier floor, no corner residual) PER ORDER, and against the
union-grid pure 2-D solve.

Layer A has walls on halves (N=2), layer B on thirds (N=3): the two uniform
lattices are GENUINELY NON-CONFORMING and their common refinement is N=6, so
this is simultaneously the M3 non-conforming measurement WITH AN INDEPENDENT
ORACLE -- the non-conforming remainder is (mortar error) - (union error) at
matched modal count, both scored against the same exact 1-D answer."""
import json
import sys
import time

import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

PX = PY = 0.9e-6
WL = 0.60e-6
THETA = 0.20
NORD = 2
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)            # duty 1/2
B3 = np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3], np.complex128)   # duty 1/3
A_SEGS = [(1 / 2, 2.0), (1 / 2, 6.0)]
B_SEGS = [(1 / 3, 9.0), (1 / 3, 2.0), (1 / 3, 4.0)]
tA, tB = 0.20e-6, 0.15e-6


def n0_row(orders, R, T):
    o = np.asarray(orders)
    sel = o[:, 1] == 0
    m = o[sel, 0]
    i = np.argsort(m)
    return m[i], R[sel][i], T[sel][i]


def oracle_1d(degree=14):
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=degree)
    s.add_layer(tA, segments=A_SEGS)
    s.add_layer(tB, segments=B_SEGS)
    s.set_source(WL, theta=THETA)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]          # row 1 = incident E_y (TE)


mo, ro, to = oracle_1d(14)
mo12, ro12, to12 = oracle_1d(12)
print(f"1-D oracle deg12 vs deg14 self-gap: "
      f"{max(np.abs(ro-ro12).max(), np.abs(to-to12).max()):.2e}", flush=True)


def score(orders, R, T):
    m, r, t = n0_row(orders, R[1], T[1])
    keep = np.isin(mo, m)
    r_ref, t_ref = ro[keep], to[keep]
    m_ref = mo[keep]
    assert np.array_equal(m, m_ref), (m, m_ref)
    direct = max(float(np.abs(r - r_ref).max()), float(np.abs(t - t_ref).max()))
    mirror = max(float(np.abs(r - r_ref[::-1]).max()),
                 float(np.abs(t - t_ref[::-1]).max()))
    return direct, mirror, float(abs(R[1].sum() + T[1].sum() - 1.0))


rows = []
print("\nMORTAR  A(N=2) | B(N=3)  -- genuinely non-conforming")
for M in [int(x) for x in (sys.argv[1:] or [5, 7, 9, 11])]:
    t0 = time.perf_counter()
    s = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(tA, eps_cell=A2)
    s.add_layer(tB, eps_cell=B3)
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    d, mi, clo = score(o, R, T)
    dt = time.perf_counter() - t0
    rows.append(dict(arm="mortar", M=M, err=d, mirror=mi, closure=clo, t=dt))
    print(f"  M={M:2d}  |err|_inf vs 1-D {d:9.2e}  (mirrored {mi:8.2e})  "
          f"closure {clo:8.1e}  {dt:6.1f}s", flush=True)

print("\nUNION grid N=6 (common refinement) -- the reference the mortar replaces")
for M in [int(x) for x in (sys.argv[1:] or [4, 5, 6])][:3]:
    t0 = time.perf_counter()
    s = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(tA, eps_cell=refine_cell(A2, 3))
    s.add_layer(tB, eps_cell=refine_cell(B3, 2))
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    d, mi, clo = score(o, R, T)
    dt = time.perf_counter() - t0
    rows.append(dict(arm="union6", M=M, err=d, mirror=mi, closure=clo, t=dt))
    print(f"  M={M:2d}  |err|_inf vs 1-D {d:9.2e}  (mirrored {mi:8.2e})  "
          f"closure {clo:8.1e}  {dt:6.1f}s", flush=True)

json.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m4_stripe_1d.json",
                     "w"), indent=1)
