"""M3 -- a y-uniform SLANTED STRIPE against the VALIDATED 1-D slant oracles,
PER ORDER, both polarizations, normal + oblique, slant 10 / 20 / 35 deg.

Oracle: ``pmm_efficiency_1d_slanted`` (scalar, TE and TM) -- the shipped
inclined-coordinate 1-D PMM.  A y-invariant cell in the 2-D staggered solver
must reproduce it order for order at the 2-D solver's own resolution, and the
BAR is the VERTICAL CONTROL (the same cell at slant 0 vs ``pmm_efficiency_1d``)
-- the residual there is what the basis can do on this geometry, and the slanted
arm must TRACK it, not beat it.

The SLANT SIGN is pinned two-sided: a slanted grating is not x-mirror
symmetric, so ``R_{+1} != R_{-1}`` and the wrong sign of ``t`` swaps them.  Both
``t = +tan(phi)`` and ``t = -tan(phi)`` are run against the same oracle; only one
may track.

T3a  sign arbitration (slant 35 deg, normal + oblique, both pols)
T3b  per-order ladder in M for the winning sign, slant 10 / 20 / 35
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree()}
t00 = time.time()

PX = PY = 0.75          # period (wavelength units); wl = 1
WL = 1.0
DEPTH = 0.30
N_RIDGE, N_GROOVE = 2.0, 1.0
N_SUP, N_SUB = 1.0, 1.5
NO = 4                  # 2-D far-field half-width
ORD_CMP = [-2, -1, 0, 1, 2]

CELL = np.array([[N_RIDGE ** 2, N_RIDGE ** 2],
                 [N_GROOVE ** 2, N_GROOVE ** 2]], dtype=complex)


def oracle(phi_deg, theta, pol, degree=22):
    o, R, T = pmm_efficiency_1d_slanted(
        PX, N_RIDGE, N_GROOVE, N_SUB, N_SUP, DEPTH, 0.5, WL,
        np.deg2rad(phi_deg), angle=theta, polarization=pol, degree=degree,
        far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def oracle_vertical(theta, pol, degree=22):
    o, R, T = pmm_efficiency_1d(
        PX, N_RIDGE, N_GROOVE, N_SUB, N_SUP, DEPTH, 0.5, WL,
        angle=theta, polarization=pol, degree=degree, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def two_d(t, theta, M):
    o, R, T, _Jr, _Jt, _i = solve_slant_stack(
        PX, PY, [{"thickness": DEPTH, "cell": CELL, "slant": (t, 0.0)}],
        N_SUP, N_SUB, WL, M=M, n_orders=NO, theta=theta, phi=0.0)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    # row 0 = incident Ex (= TM at phi=0), row 1 = incident Ey (= TE)
    out = {"tm": {}, "te": {}}
    for m, i in idx.items():
        out["tm"][m] = (float(R[0, i]), float(T[0, i]))
        out["te"][m] = (float(R[1, i]), float(T[1, i]))
    ylk = 0.0
    for i, (m, n) in enumerate(o):
        if n != 0:
            ylk = max(ylk, float(np.max(np.abs(R[:, i]))),
                      float(np.max(np.abs(T[:, i]))))
    return out, ylk, float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))


def perorder(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


# ------------------------------------------------------------------ T3a sign
print("T3a  SLANT-SIGN ARBITRATION (M = 7)")
sign_rows = []
for phi_deg in (35.0,):
    for tname, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25))):
        vert2d, _yl, _cl = two_d(0.0, theta, 7)
        for pol in ("te", "tm"):
            ovs = oracle_vertical(theta, pol)
            ctrl = perorder(vert2d[pol], ovs)
            orc = oracle(phi_deg, theta, pol)
            row = {"phi": phi_deg, "mount": tname, "pol": pol, "ctrl": ctrl}
            for sgn in (+1.0, -1.0):
                t = sgn * np.tan(np.deg2rad(phi_deg))
                d2, _yl, _cl = two_d(t, theta, 7)
                row[f"t{'+' if sgn > 0 else '-'}"] = perorder(d2[pol], orc)
            sign_rows.append(row)
            print(f"  phi={phi_deg:.0f} {tname:9s} {pol}  ctrl {ctrl:.2e} | "
                  f"t=+tan {row['t+']:.2e} | t=-tan {row['t-']:.2e}")
res["T3a_sign"] = sign_rows
best_plus = max(r["t+"] for r in sign_rows)
best_minus = max(r["t-"] for r in sign_rows)
SGN = +1.0 if best_plus < best_minus else -1.0
res["sign_winner"] = "+tan" if SGN > 0 else "-tan"
res["sign_worst_winner"] = min(best_plus, best_minus)
res["sign_worst_loser"] = max(best_plus, best_minus)
print(f"  -> winner t = {res['sign_winner']}(phi)  "
      f"({res['sign_worst_winner']:.2e} vs {res['sign_worst_loser']:.2e})")

# ------------------------------------------------------------------ T3b ladder
print("\nT3b  PER-ORDER LADDER IN M (winning sign)")
lad = []
for phi_deg in (0.0, 10.0, 20.0, 35.0):
    for mount, theta in (("normal", 0.0), ("oblique25", np.deg2rad(25))):
        orc = {p: (oracle(phi_deg, theta, p) if phi_deg > 0
                   else oracle_vertical(theta, p)) for p in ("te", "tm")}
        for M in (5, 6, 7, 8):
            t = SGN * np.tan(np.deg2rad(phi_deg))
            d2, ylk, clo = two_d(t, theta, M)
            row = {"phi": phi_deg, "mount": mount, "M": M,
                   "te": perorder(d2["te"], orc["te"]),
                   "tm": perorder(d2["tm"], orc["tm"]),
                   "yleak": ylk, "closure": clo}
            lad.append(row)
            print(f"  phi={phi_deg:4.0f} {mount:9s} M={M} "
                  f"te {row['te']:.2e} tm {row['tm']:.2e} "
                  f"yleak {ylk:.1e} |R+T-1| {clo:.2e}")
res["T3b_ladder"] = lad
res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m3_stripe_vs_1d.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\nWROTE results/m3_stripe_vs_1d.json  ({res['wall_s']:.1f} s)")
