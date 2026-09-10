"""F2 (open item O-2) -- THE DEVICE REGIME: a corner-dominated 2-D PILLAR PAIR.

O-2 says the equal-DOF advantage measured in S7.3 / S9.2 was established on
STRIPES (smooth in y, one wall set per layer) and on a stripe staircase, and
that the case most likely to REVERSE its sign is the campaign's actual device
class: a crossed 2-D pillar, whose four corners cap the convergence at
ALGEBRAIC, where h-refinement should do relatively better than p-refinement.

Fixture: layer A a pillar 1/2 of the period wide (its own lattice ``N = 2``),
layer B a pillar 1/3 wide (``N = 3``) -- genuinely non-conforming, common
refinement ``N = 6``.  CONICAL incidence, so no symmetry hides an error.

The comparison is at EQUAL DEGREES OF FREEDOM: per axis ``q = N (M - 1)``, so
``q_union(M) = 6(M-1)`` equals ``q_A = 2(M_A - 1)`` at ``M_A = 3M - 2`` and
``q_B = 3(M_B - 1)`` at ``M_B = 2M - 1``; at those settings every region
eigenproblem in both arms has the same dimension ``2 q^2``.

The reference is the UNION grid at the top of its own convergence ladder, and
that ladder is reported so the reference's own uncertainty bounds every
statement made against it.

Usage:  python f2_device_regime.py [scalar] [tensor]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
WHICH = [a.lower() for a in sys.argv[1:]] or ["scalar", "tensor"]

PX = PY = 1.2e-6
WL = 0.85e-6
THETA, PHI = 0.18, 0.35
NORD = 2
EPS_H, EPS_P = 2.25, 9.0
tA, tB = 0.16e-6, 0.13e-6

A2 = np.full((2, 2), EPS_H + 0j)
A2[0, 0] = EPS_P                       # pillar 1/2 x 1/2
B3 = np.full((3, 3), EPS_H + 0j)
B3[0, 0] = EPS_P                       # pillar 1/3 x 1/3

# in-plane (BLOCK-form) LC tensor: director in the xy plane, so e13 = e23 = 0
LC = np.asarray(uniaxial_tensor(1.5, 1.9, np.pi / 2, phi=np.deg2rad(35.0)),
                dtype=complex)
A2T = np.empty((2, 2, 3, 3), complex)
A2T[:, :] = np.eye(3) * EPS_H
A2T[0, 0] = LC

res = {}
OUT = os.path.join(HERE, "f2_device_regime.json")


def dump():
    json.dump(res, open(OUT, "w"), indent=1)


def score(oa, Ra, Ta, ob, Rb, Tb):
    """max over the retained orders and BOTH incident polarizations."""
    assert np.array_equal(np.asarray(oa), np.asarray(ob))
    sc = max(float(np.max(np.abs(Rb))), float(np.max(np.abs(Tb))))
    return max(float(np.abs(np.asarray(Ra) - np.asarray(Rb)).max()),
               float(np.abs(np.asarray(Ta) - np.asarray(Tb)).max())) / sc


def two_sided_closure(R, T):
    return max(abs(float(np.asarray(R)[p].sum() + np.asarray(T)[p].sum() - 1.0))
               for p in (0, 1))


def run_union(M, cellA, cellB):
    t0 = time.perf_counter()
    u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    u.add_layer(tA, eps_cell=refine_cell(cellA, 3))
    u.add_layer(tB, eps_cell=refine_cell(cellB, 2))
    u.set_source(WL, theta=THETA, phi=PHI)
    o, R, T = u.solve(jones=False)
    return o, R, T, time.perf_counter() - t0


def run_mortar(MA, MB, cellA, cellB):
    t0 = time.perf_counter()
    s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
    s.add_layer(tA, eps_cell=cellA, n_modes=MA)
    s.add_layer(tB, eps_cell=cellB, n_modes=MB)
    s.set_source(WL, theta=THETA, phi=PHI)
    o, R, T = s.solve(jones=False)
    return o, R, T, time.perf_counter() - t0


def campaign(tag, cellA, cellB, union_ladder, ref_M, dof_levels):
    print(f"\n===== {tag} =====", flush=True)
    print("-- union-grid REFERENCE ladder (N=6, the common refinement) --",
          flush=True)
    prev = None
    lad, sols = [], {}
    for M in union_ladder:
        o, R, T, dt = run_union(M, cellA, cellB)
        sols[M] = (o, R, T)
        gap = None if prev is None else score(o, R, T, *prev)
        prev = (o, R, T)
        clo = two_sided_closure(R, T)
        lad.append(dict(M=M, q=6 * (M - 1), gap_vs_prev=gap, closure=clo,
                        t=dt, R00=float(np.asarray(R)[1][
                            (np.asarray(o)[:, 0] == 0)
                            & (np.asarray(o)[:, 1] == 0)][0])))
        print(f"  union M={M} (q={6*(M-1):2d}, eig {2*(6*(M-1))**2:5d})  "
              f"R(0,0)={lad[-1]['R00']:.9f}  gap vs prev "
              f"{'   --   ' if gap is None else f'{gap:8.2e}'}  closure "
              f"{clo:8.1e}  {dt:7.1f}s", flush=True)
        res[tag] = dict(ladder=lad)
        dump()
    oR, RR, TR = sols[ref_M]
    ref_unc = lad[-1]["gap_vs_prev"]
    print(f"  -> reference = union M={ref_M}; its OWN uncertainty (last "
          f"ladder gap) = {ref_unc:.2e}", flush=True)

    print("-- EQUAL-DOF: union(M) vs mortar(M_A = 3M-2 on N=2, "
          "M_B = 2M-1 on N=3) --", flush=True)
    rows = []
    for M in dof_levels:
        q = 6 * (M - 1)
        MA, MB = 3 * M - 2, 2 * M - 1
        if M in sols:
            ou, Ru, Tu = sols[M]
            tu = [r["t"] for r in lad if r["M"] == M][0]
        else:
            ou, Ru, Tu, tu = run_union(M, cellA, cellB)
        eu = score(ou, Ru, Tu, oR, RR, TR)
        cu = two_sided_closure(Ru, Tu)
        om, Rm, Tm, tm = run_mortar(MA, MB, cellA, cellB)
        em = score(om, Rm, Tm, oR, RR, TR)
        cm = two_sided_closure(Rm, Tm)
        rows.append(dict(q=q, M=M, MA=MA, MB=MB, err_union=eu, err_mortar=em,
                         clo_union=cu, clo_mortar=cm, ratio=em / eu,
                         t_union=tu, t_mortar=tm,
                         above_ref_uncertainty=bool(min(eu, em) > 3 * ref_unc)))
        print(f"  q={q:2d} (eig {2*q*q:5d})  union M={M} err {eu:9.2e} "
              f"(clo {cu:8.1e}, {tu:6.1f}s) | mortar M_A={MA} M_B={MB} err "
              f"{em:9.2e} (clo {cm:8.1e}, {tm:6.1f}s) | mortar/union "
              f"{em/eu:7.3f}x", flush=True)
        res[tag]["equal_dof"] = rows
        res[tag]["ref_M"] = ref_M
        res[tag]["ref_uncertainty"] = ref_unc
        dump()


if "scalar" in WHICH:
    campaign("scalar pillar pair", A2, B3, [3, 4, 5, 6, 7], 7, [3, 4, 5])
if "tensor" in WHICH:
    campaign("in-plane LC tensor in layer A", A2T, B3, [3, 4, 5, 6], 6, [3, 4])

dump()
print("\nwrote", OUT, flush=True)
