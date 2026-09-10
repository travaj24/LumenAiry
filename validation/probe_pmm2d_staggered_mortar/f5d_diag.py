"""F5 gate (d2) DIAGNOSTIC -- the 4-slice non-uniform taper cascade plateaus at
~1.2e-01 against the exact 1-D oracle while a SINGLE layer on the same kind of
grid reaches 4.05e-05 (gate c3).  Before that number is reported as a property
of non-uniform segments, separate the candidates:

  A  4 IDENTICAL slices (a straight pillar at the same arbitrary walls) -- all
     grids equal, so the mortar is BYPASSED.  Isolates the multi-layer
     machinery from the mortar.
  B  the same 4 identical slices with ``force_mortar=True`` -- the M1 identity
     on NON-UNIFORM grids, through 5 forced mortar interfaces.
  C  a 2-slice taper (ONE mortar interface) -- does the error scale with the
     number of non-conforming interfaces?
  D  the wall SEPARATION sweep: the same 2-slice cascade with the second
     slice's walls moved by a shrinking amount.  If the defect is the
     near-coincident-wall integration mesh, it must track this.
  E  per-order breakdown of the 4-slice error.
"""
import json
import os
import warnings

import numpy as np
from mortar2d import guard
print("lumenairy:", guard(), flush=True)
from nonuniform import MortarStackNU
from lumenairy import PMMStack

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PX = PY = 1.2e-6
WL = 0.85e-6
THETA, NORD = 0.15, 2
EPS_H, EPS_P = 2.25, 9.0
THICK = 0.32e-6
NSL = 4
dz = THICK / NSL
XB0, XB1 = (0.1873, 0.7241), (0.2917, 0.6109)
FR = []
for s_ in range(NSL):
    zf = 1.0 - (s_ + 0.5) / NSL
    FR.append((XB0[0] + (XB1[0] - XB0[0]) * zf,
               XB0[1] + (XB1[1] - XB0[1]) * zf))

TILE = np.empty((3, 3), complex)
TILE[0, :] = EPS_H
TILE[1, :] = EPS_P
TILE[2, :] = EPS_H


def oracle(fr, deg=14):
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg)
    for (a, b) in fr:
        s.add_layer(dz, segments=[(a, EPS_H), (b - a, EPS_P), (1.0 - b, EPS_H)])
    s.set_source(WL, theta=THETA)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


def pure(fr, M, force_mortar=False):
    s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
    for (a, b) in fr:
        s.add_layer(dz, eps_cell=TILE, x_walls=[a * PX, b * PX],
                    y_walls=[a * PY, b * PY])
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False, force_mortar=force_mortar)
    oo = np.asarray(o)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    i = np.argsort(m)
    return m[i], R[1][sel][i], T[1][sel][i], \
        float(abs(R[1].sum() + T[1].sum() - 1))


def err(fr, M, force_mortar=False):
    MO, RO, TO = oracle(fr)
    m, R, T, clo = pure(fr, M, force_mortar)
    keep = np.isin(MO, m)
    return (max(float(np.abs(R - RO[keep]).max()),
                float(np.abs(T - TO[keep]).max())), clo,
            m, R - RO[keep], T - TO[keep])


res = {}
print("\n[A] 4 IDENTICAL slices (straight pillar, mortar BYPASSED)", flush=True)
flat = [FR[0]] * 4
rows = []
for M in (4, 5, 6, 7, 9):
    e, clo, _m, _dr, _dt = err(flat, M)
    rows.append(dict(M=M, err=e, closure=clo))
    print(f"  M={M}  err vs EXACT 1-D {e:9.2e}  closure {clo:8.1e}",
          flush=True)
res["A_identical_bypass"] = rows

print("\n[B] the SAME 4 identical slices, mortar FORCED (M1 identity, "
      "non-uniform grids)", flush=True)
rows = []
for M in (5, 7):
    e0, c0, _m, _a, _b = err(flat, M, force_mortar=False)
    e1, c1, _m, _a, _b = err(flat, M, force_mortar=True)
    rows.append(dict(M=M, err_bypass=e0, err_forced=e1, clo_bypass=c0,
                     clo_forced=c1, identity=abs(e1 - e0)))
    print(f"  M={M}  bypass {e0:9.2e} | forced mortar {e1:9.2e}  "
          f"|difference| {abs(e1-e0):9.2e}   closure {c0:8.1e}/{c1:8.1e}",
          flush=True)
res["B_forced_identity"] = rows

print("\n[C] taper truncated to 1 / 2 / 3 / 4 slices (1..4 non-conforming "
      "interfaces)", flush=True)
rows = []
for n in (1, 2, 3, 4):
    for M in (5, 7, 9):
        e, clo, _m, _a, _b = err(FR[:n], M)
        rows.append(dict(n_slices=n, M=M, err=e, closure=clo))
        print(f"  slices={n} M={M}  err {e:9.2e}  closure {clo:8.1e}",
              flush=True)
res["C_n_slices"] = rows

print("\n[D] wall SEPARATION sweep: 2 slices, second slice's walls offset by "
      "delta", flush=True)
rows = []
a0, b0 = FR[0]
for delta in (0.10, 0.0261, 0.005, 1e-3, 1e-5):
    fr2 = [(a0, b0), (a0 - delta, b0 + delta)]
    for M in (7,):
        e, clo, _m, _a, _b = err(fr2, M)
        rows.append(dict(delta=delta, M=M, err=e, closure=clo))
        print(f"  delta={delta:9.5f} (thin union sub-interval) M={M}  "
              f"err {e:9.2e}  closure {clo:8.1e}", flush=True)
res["D_separation"] = rows

print("\n[E] per-order breakdown of the 4-slice taper error", flush=True)
rows = []
for M in (7, 9):
    e, clo, m, dR, dT = err(FR, M)
    print(f"  M={M}: orders {list(m)}", flush=True)
    print(f"        dR {['%.2e' % v for v in dR]}", flush=True)
    print(f"        dT {['%.2e' % v for v in dT]}", flush=True)
    rows.append(dict(M=M, orders=[int(v) for v in m],
                     dR=[float(v) for v in dR], dT=[float(v) for v in dT]))
res["E_per_order"] = rows

json.dump(res, open(os.path.join(HERE, "f5d_diag.json"), "w"), indent=1)
print("\nwrote f5d_diag.json", flush=True)
