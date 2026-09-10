"""F4.4 -- MEASURE the proposed `M_u` rule instead of deriving it.

F4.2 shows a uniform layer on ``grid = 1`` at the stack's default ``M`` reads
3-6x worse than the same layer on ``grid = 2``/``3``.  The rule F4.4 proposes
is to size the uniform layer's modal count to its NEIGHBOURS' ``q`` rather than
to the stack's ``M``::

    M_u = max(q_prev, q_next) / grid + 1

On the F4.2 fixture (``q_A = 2*(8-1) = 14``, ``q_B = 3*(7-1) = 18``) that is
``M_u = 19`` on ``grid = 1``, ``M_u = 10`` on ``grid = 2``, ``M_u = 7`` on
``grid = 3`` -- all three at ``q_u = 18``.  Measured here against the same
exact 1-D oracle, so the rule is a reading rather than an argument.
"""
import json
import os
import time
import warnings

import numpy as np
from mortar2d import guard, MortarStack2D
print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PX = PY = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0
A2 = np.array([[EPS_P, EPS_P], [EPS_H, EPS_H]], complex)
B3 = np.array([[EPS_P] * 3, [EPS_H] * 3, [EPS_H] * 3], complex)
tA, tU, tB = 0.16e-6, 0.11e-6, 0.13e-6
MA, MB, NORD = 8, 7, 2
QA, QB = 2 * (MA - 1), 3 * (MB - 1)
print(f"neighbours: q_A = {QA}, q_B = {QB}  ->  target q_u = {max(QA, QB)}",
      flush=True)


def oracle(theta, deg=14):
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg)
    s.add_layer(tA, segments=[(1 / 2, EPS_P), (1 / 2, EPS_H)])
    s.add_layer(tU, segments=[(1.0, EPS_H)])
    s.add_layer(tB, segments=[(1 / 3, EPS_P), (2 / 3, EPS_H)])
    s.set_source(WL, theta=theta)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


def score(orders, R, T, MO, RO, TO):
    oo = np.asarray(orders)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    j = np.argsort(m)
    m, r, t = m[j], np.asarray(R)[1][sel][j], np.asarray(T)[1][sel][j]
    keep = np.isin(MO, m)
    return max(float(np.abs(r - RO[keep]).max()),
               float(np.abs(t - TO[keep]).max()))


rows = []
for th in (0.00, 0.20, 0.40):
    MO, RO, TO = oracle(th)
    line = []
    for gu, mu in ((1, MB), (1, 19), (2, 10), (3, 7)):
        t0 = time.perf_counter()
        s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
        s.add_layer(tA, eps_cell=A2, n_modes=MA)
        s.add_layer(tU, eps=EPS_H, grid=gu, n_modes=mu)
        s.add_layer(tB, eps_cell=B3, n_modes=MB)
        s.set_source(WL, theta=th, phi=0.0)
        o, R, T = s.solve(jones=False)
        e = score(o, R, T, MO, RO, TO)
        dt = time.perf_counter() - t0
        rows.append(dict(theta=th, grid=gu, M_u=mu, q_u=gu * (mu - 1), err=e,
                         dim_u=2 * (gu * (mu - 1)) ** 2, t=dt,
                         closure=float(max(abs(R[p].sum() + T[p].sum() - 1)
                                           for p in (0, 1)))))
        line.append(f"grid={gu} M_u={mu:2d} (q_u={gu*(mu-1):2d}, eig "
                    f"{2*(gu*(mu-1))**2:4d}): {e:8.2e} [{dt:5.1f}s]")
    print(f"  theta={th:.2f}\n     " + "\n     ".join(line), flush=True)
json.dump(rows, open(os.path.join(HERE, "f4b_mu_rule.json"), "w"), indent=1)
print("\nwrote f4b_mu_rule.json", flush=True)
