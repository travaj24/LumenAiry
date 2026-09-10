"""F3 (open item O-10) -- the PER-LAYER ``M`` RECIPE, measured as a surface.

O-10: S7.2 showed that walking ONE layer's modal count PLATEAUS -- the answer
becomes stationary in ``M_A`` while still carrying layer B's error, so a
per-layer solve can be stationary in one knob and wrong.  The build must ship a
recipe.  This script measures the two-knob convergence SURFACE on the F2
corner-dominated pillar pair (layer A: pillar 1/2 on ``N = 2``; layer B: pillar
1/3 on ``N = 3``; conical incidence) and tests one candidate recipe against it:

    RULE: raise ``M`` on the layer whose OWN SINGLE-LAYER residual is larger.

"Own single-layer residual" is measurable without the pair: solve the layer
ALONE between the half-spaces on its own grid at its own ``M`` and compare with
the same layer alone at the top of its own ladder.  That is the quantity the
recipe would have a user (or an auto-refiner) compute, and it costs one extra
single-layer solve per layer per rung -- cheap, because a single-layer solve is
one region eig instead of the stack's.

The surface is scored against the union grid on the common refinement ``N = 6``
at ``M = 6`` (``q = 30``); F2's ladder gives that reference's own uncertainty.
"""
import json
import os
import time
import warnings

import numpy as np
from mortar2d import MortarStack2D, guard, refine_cell

print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
OUT = os.path.join(HERE, "f3_perlayer_M_recipe.json")

PX = PY = 1.2e-6
WL = 0.85e-6
THETA, PHI, NORD = 0.18, 0.35, 2
EPS_H, EPS_P = 2.25, 9.0
tA, tB = 0.16e-6, 0.13e-6
A2 = np.full((2, 2), EPS_H + 0j)
A2[0, 0] = EPS_P
B3 = np.full((3, 3), EPS_H + 0j)
B3[0, 0] = EPS_P

MA_LADDER = (5, 7, 9, 11)          # q_A = 8, 12, 16, 20   (N = 2)
MB_LADDER = (4, 5, 6, 7)           # q_B = 9, 12, 15, 18   (N = 3)


def dump():
    json.dump(res, open(OUT, "w"), indent=1)


def rel(Ra, Ta, Rb, Tb):
    sc = max(float(np.max(np.abs(Rb))), float(np.max(np.abs(Tb))))
    return max(float(np.abs(np.asarray(Ra) - np.asarray(Rb)).max()),
               float(np.abs(np.asarray(Ta) - np.asarray(Tb)).max())) / sc


def single(cell, M, t):
    """One patterned layer alone, on its OWN grid."""
    s = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(t, eps_cell=cell)
    s.set_source(WL, theta=THETA, phi=PHI)
    return s.solve(jones=False)


def pair(MA, MB):
    t0 = time.perf_counter()
    s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
    s.add_layer(tA, eps_cell=A2, n_modes=MA)
    s.add_layer(tB, eps_cell=B3, n_modes=MB)
    s.set_source(WL, theta=THETA, phi=PHI)
    o, R, T = s.solve(jones=False)
    return o, R, T, time.perf_counter() - t0


# ------------------------------------------------------- the reference
print("union-grid reference (N=6): ladder M=5 then M=6", flush=True)
ref = {}
for M in (5, 6):
    t0 = time.perf_counter()
    u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    u.add_layer(tA, eps_cell=refine_cell(A2, 3))
    u.add_layer(tB, eps_cell=refine_cell(B3, 2))
    u.set_source(WL, theta=THETA, phi=PHI)
    ref[M] = u.solve(jones=False)
    print(f"  union M={M} (q={6*(M-1)})  {time.perf_counter()-t0:7.1f}s",
          flush=True)
oR, RR, TR = ref[6]
ref_unc = rel(ref[5][1], ref[5][2], RR, TR)
print(f"  reference = union M=6; its own uncertainty (M=5 gap) {ref_unc:.2e}",
      flush=True)
res["reference"] = dict(M=6, q=30, uncertainty=ref_unc)
dump()

# ------------------------------------- (1) the per-layer OWN residuals
print("\n=== (1) each layer's OWN single-layer residual (the rule's input) ===",
      flush=True)
own = {"A": {}, "B": {}}
_oA, RA_top, TA_top = single(A2, 13, tA)          # q_A = 24
_oB, RB_top, TB_top = single(B3, 9, tB)           # q_B = 24
rowsA, rowsB = [], []
for M in MA_LADDER:
    _o, R, T = single(A2, M, tA)
    own["A"][M] = rel(R, T, RA_top, TA_top)
    rowsA.append(dict(layer="A", M=M, q=2 * (M - 1), residual=own["A"][M]))
    print(f"  layer A (N=2)  M={M:2d} (q={2*(M-1):2d})  own residual "
          f"{own['A'][M]:9.2e}", flush=True)
for M in MB_LADDER:
    _o, R, T = single(B3, M, tB)
    own["B"][M] = rel(R, T, RB_top, TB_top)
    rowsB.append(dict(layer="B", M=M, q=3 * (M - 1), residual=own["B"][M]))
    print(f"  layer B (N=3)  M={M:2d} (q={3*(M-1):2d})  own residual "
          f"{own['B'][M]:9.2e}", flush=True)
res["own_residuals"] = rowsA + rowsB
dump()

# ------------------------------------------------- (2) the 4x4 surface
print("\n=== (2) the two-knob convergence SURFACE (error vs the reference) ===",
      flush=True)
surf = {}
rows = []
print("      " + "".join(f"  M_B={mb:<10d}" for mb in MB_LADDER), flush=True)
for ma in MA_LADDER:
    line = []
    for mb in MB_LADDER:
        o, R, T, dt = pair(ma, mb)
        e = rel(R, T, RR, TR)
        surf[(ma, mb)] = e
        clo = max(abs(float(R[p].sum() + T[p].sum() - 1)) for p in (0, 1))
        dA = 2 * (2 * (ma - 1)) ** 2
        dB = 2 * (3 * (mb - 1)) ** 2
        rows.append(dict(MA=ma, MB=mb, qA=2 * (ma - 1), qB=3 * (mb - 1),
                         err=e, closure=clo, t=dt, dimA=dA, dimB=dB,
                         eig_work=float(dA ** 3 + dB ** 3)))
        line.append(f"{e:9.2e}   ")
    print(f"  M_A={ma:2d} " + "".join(line), flush=True)
    res["surface"] = rows
    dump()

# --------------------------------------------- (3) does the RULE hold?
print("\n=== (3) THE RULE: raise M on the layer with the larger OWN residual ==",
      flush=True)
print("    At each interior surface point, compare the error DROP from one "
      "rung of M_A\n    against one rung of M_B, and check the rule's "
      "prediction.", flush=True)
tests = []
for i, ma in enumerate(MA_LADDER[:-1]):
    for j, mb in enumerate(MB_LADDER[:-1]):
        e0 = surf[(ma, mb)]
        eA = surf[(MA_LADDER[i + 1], mb)]
        eB = surf[(ma, MB_LADDER[j + 1])]
        pred = "A" if own["A"][ma] > own["B"][mb] else "B"
        actual = "A" if eA < eB else "B"
        tests.append(dict(MA=ma, MB=mb, own_A=own["A"][ma], own_B=own["B"][mb],
                          err=e0, err_raise_A=eA, err_raise_B=eB,
                          predicted=pred, actual=actual, hit=pred == actual))
        print(f"  (M_A={ma:2d}, M_B={mb})  own A {own['A'][ma]:8.1e} vs B "
              f"{own['B'][mb]:8.1e} -> predict raise {pred} ; "
              f"err {e0:8.2e} -> A {eA:8.2e} / B {eB:8.2e} -> actual "
              f"{actual}   {'HIT' if pred == actual else 'MISS'}", flush=True)
hits = sum(t["hit"] for t in tests)
print(f"\n  rule hit rate: {hits}/{len(tests)}", flush=True)
res["rule"] = dict(tests=tests, hits=hits, n=len(tests))

# the plateau, restated: hold one knob at the top and walk the other
print("\n=== (4) the PLATEAU, both ways ===", flush=True)
plat = []
for mb_fixed in (MB_LADDER[0], MB_LADDER[-1]):
    line = [f"{surf[(ma, mb_fixed)]:9.2e}" for ma in MA_LADDER]
    plat.append(dict(hold="M_B", value=mb_fixed,
                     errs=[surf[(ma, mb_fixed)] for ma in MA_LADDER]))
    print(f"  hold M_B={mb_fixed}, walk M_A {MA_LADDER}: " + "  ".join(line),
          flush=True)
for ma_fixed in (MA_LADDER[0], MA_LADDER[-1]):
    line = [f"{surf[(ma_fixed, mb)]:9.2e}" for mb in MB_LADDER]
    plat.append(dict(hold="M_A", value=ma_fixed,
                     errs=[surf[(ma_fixed, mb)] for mb in MB_LADDER]))
    print(f"  hold M_A={ma_fixed}, walk M_B {MB_LADDER}: " + "  ".join(line),
          flush=True)
res["plateau"] = plat
dump()
print("\nwrote", OUT, flush=True)
