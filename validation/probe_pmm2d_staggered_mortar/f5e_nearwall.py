"""F5 follow-up -- the NEAR-COINCIDENT WALL reading, and the deep rung that
proves the taper cascade converges.

**SUPERSEDED IN PART, READ ``f5f_attrib.py`` FIRST.**  The "explosion" this
script reports at ``delta = 1e-4 .. 1e-5`` is NOT the non-uniform mortar: it is
the 1-D ``PMMStack`` ORACLE, whose own degree-12-vs-degree-14 self-gap reads
4.8e-01 / 8.2e+00 at those deltas.  ``f5f_attrib.py`` measures the oracle's
self-gap alongside every comparison and shows the pure non-uniform arm is
SMOOTH and MONOTONE in ``delta`` all the way to 0.  The "weld remedy" section
below is therefore moot and is retained only as the record of how the
attribution was made.


``f5d_diag.py`` [D] found that a 2-slice cascade whose two wall sets differ by
``delta`` (as a fraction of the period) is clean and MONOTONE down to
``delta = 1e-3`` (2.2e-04 at M=7) and then EXPLODES at ``delta = 1e-5``
(9.31e-01) while the lossless closure stays at 1.6e-08 -- i.e. the failure is
ENERGY-INVISIBLE, the most dangerous shape this campaign knows.

This script (1) maps the threshold, (2) instruments the three mortar sites with
the M7 conditioning census so the mechanism is identified rather than guessed,
(3) tests the obvious remedy -- WELD walls closer than a tolerance into one,
i.e. treat the two grids as conforming when their walls agree to that tolerance
-- and (4) runs the 4-slice taper at ``M = 11`` to show the gate-(d2) plateau
is resolution and not a defect.
"""
import json
import os
import sys
import time
import warnings

import mortar2d
import numpy as np
from mortar2d import guard

print("lumenairy:", guard(), flush=True)
from nonuniform import MortarStackNU

from lumenairy import PMMStack

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WHICH = [a.lower() for a in sys.argv[1:]] or ["sweep", "deep"]
res = {}
OUT = os.path.join(HERE, "f5e_nearwall.json")


def dump():
    json.dump(res, open(OUT, "w"), indent=1)


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


def run(fr, M, census=False):
    if census:
        mortar2d.CENSUS = []
    s = MortarStackNU(PX, PY, n_modes=M, n_orders=NORD)
    for (a, b) in fr:
        s.add_layer(dz, eps_cell=TILE, x_walls=[a * PX, b * PX],
                    y_walls=[a * PY, b * PY])
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    cen = mortar2d.CENSUS
    mortar2d.CENSUS = None
    oo = np.asarray(o)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    i = np.argsort(m)
    return (m[i], R[1][sel][i], T[1][sel][i],
            float(abs(R[1].sum() + T[1].sum() - 1)), cen)


def err(fr, M, census=False):
    MO, RO, TO = oracle(fr)
    m, R, T, clo, cen = run(fr, M, census)
    keep = np.isin(MO, m)
    return (max(float(np.abs(R - RO[keep]).max()),
                float(np.abs(T - TO[keep]).max())), clo, cen)


if "sweep" in WHICH:
    print("\n=== the wall-SEPARATION threshold, with the conditioning census "
          "===", flush=True)
    print("    2 slices; slice 2's walls offset by +/- delta * period.  The "
          "roadmap's\n    headline taper moves a wall ~1.8 nm on a 700 nm "
          "period = 2.6e-03 of the period.", flush=True)
    a0, b0 = FR[0]
    rows = []
    for delta in (1e-1, 1e-2, 2.6e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6,
                  1e-6, 1e-8, 0.0):
        fr2 = [(a0, b0), (a0 - delta, b0 + delta)]
        for M in (5, 7):
            e, clo, cen = err(fr2, M, census=True)
            worst = min([c[2] for c in cen], default=float("nan"))
            per = {}
            for site, _n, rc in cen:
                per[site] = min(per.get(site, 1e9), rc)
            rows.append(dict(delta=delta, M=M, err=e, closure=clo,
                             worst_rcond=float(worst),
                             per_site={k: float(v) for k, v in per.items()},
                             n_mortar_calls=len(cen)))
            print(f"  delta={delta:9.2e}  M={M}  err {e:9.2e}  closure "
                  f"{clo:8.1e}  worst mortar rcond {worst:9.2e}  "
                  f"(mortar calls {len(cen)})", flush=True)
        res["sweep"] = rows
        dump()

    print("\n=== the REMEDY: weld walls closer than a tolerance ===",
          flush=True)
    print("    Same sweep, but slice 2's walls are SNAPPED onto slice 1's when "
          "they agree to\n    1e-4 of the period -- the two grids then compare "
          "EQUAL and the interface takes\n    the plain square match (the "
          "identical-grid bypass).", flush=True)
    rows2 = []
    for delta in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6):
        snap = delta <= 1e-4
        fr2 = ([(a0, b0), (a0, b0)] if snap
               else [(a0, b0), (a0 - delta, b0 + delta)])
        e, clo, _c = err(fr2, 7)
        # the exact 1-D oracle for the WELDED geometry is not the same device,
        # so score the welded arm against the UNWELDED oracle: the whole point
        # is that a delta this small must not change the answer.
        MO, RO, TO = oracle([(a0, b0), (a0 - delta, b0 + delta)])
        m, R, T, clo2, _c = run(fr2, 7)
        keep = np.isin(MO, m)
        e2 = max(float(np.abs(R - RO[keep]).max()),
                 float(np.abs(T - TO[keep]).max()))
        rows2.append(dict(delta=delta, snapped=snap, err_vs_true_geometry=e2,
                          closure=clo2))
        print(f"  delta={delta:9.2e}  snapped={snap!s:5s}  err vs the TRUE "
              f"(un-snapped) geometry {e2:9.2e}  closure {clo2:8.1e}",
              flush=True)
    res["weld"] = rows2
    dump()

if "deep" in WHICH:
    print("\n=== the DEEP rung: the 4-slice taper at M=11 (q=30) ===",
          flush=True)
    rows = []
    for tag, fr in (("4-slice taper", FR),
                    ("4 identical slices (no mortar)", [FR[0]] * 4)):
        for M in (9, 11):
            t0 = time.perf_counter()
            e, clo, _c = err(fr, M)
            rows.append(dict(case=tag, M=M, q=3 * (M - 1), err=e, closure=clo,
                             t=time.perf_counter() - t0))
            print(f"  {tag:32s} M={M:2d} (q={3*(M-1)})  err vs EXACT 1-D "
                  f"{e:9.2e}  closure {clo:8.1e}  {rows[-1]['t']:7.1f}s",
                  flush=True)
            res["deep"] = rows
            dump()

dump()
print("\nwrote", OUT, flush=True)
