"""M6 -- COST of a slanted region solve, against the two paths it competes with.

T6a  ASSEMBLY + EIG wall clock for one region, at matched ``M``, three arms:
       in-plane 2q^2 (scipy QZ), vertical OUT-OF-PLANE 4q^2 (Cholesky-whitened
       standard eig), SLANTED 4q^2 (the same eig plus six extra kron blocks and
       one pointwise 3x3 congruence).
     The slant's own overhead is the SLANTED-vs-VERTICAL-OOP ratio; the price of
     needing the first-order generator at all is the OOP-vs-in-plane ratio (the
     shipped Stage-B number, re-measured here as a control).
T6b  END-TO-END single-layer solve, slanted vs vertical, same cell.
T6c  the SPEED-WIN arithmetic: how many pure-solver staircase slices does ONE
     slanted solve replace at matched accuracy?  Uses M4's ladder (read back
     from results/m4_pillar.json) plus the per-slice cost measured here.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    SlantSolver, assert_worktree, slant_region_modes, solve_slant_stack,
    tensor_uniaxial,
)
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE, _region_modes, _region_modes_oop,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree()}
t00 = time.time()
PX = PY = 1.2
WL = 1.0
K0 = 2.0 * np.pi / WL
A0X, A0Y = np.sin(np.deg2rad(20)) * np.cos(np.deg2rad(35)) * K0, \
    np.sin(np.deg2rad(20)) * np.sin(np.deg2rad(35)) * K0

scal = np.full((2, 2), 1.0 + 0j)
scal[0, 0] = 4.0
oopc = np.zeros((2, 2, 3, 3), dtype=complex)
oopc[:, :] = np.eye(3)
oopc[0, 0] = tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))


def timeit(fn, rep=3):
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


print("T6a  per-region ASSEMBLY + EIG (Nx = Ny = 2)")
rows = []
for M in (4, 5, 6, 7):
    q = 2 * (M - 1)
    rep = 5 if M <= 5 else 2
    a_in = timeit(lambda: Granet2DTransverseE(PX, PY, 2, 2, M, scal,
                                              alpha0x=A0X, alpha0y=A0Y, k0=K0),
                  rep)
    s_in = Granet2DTransverseE(PX, PY, 2, 2, M, scal, alpha0x=A0X,
                               alpha0y=A0Y, k0=K0)
    e_in = timeit(lambda: _region_modes(s_in), rep)
    a_oop = timeit(lambda: Granet2DTransverseE(PX, PY, 2, 2, M, oopc,
                                               alpha0x=A0X, alpha0y=A0Y,
                                               k0=K0), rep)
    s_oop = Granet2DTransverseE(PX, PY, 2, 2, M, oopc, alpha0x=A0X,
                                alpha0y=A0Y, k0=K0)
    e_oop = timeit(lambda: _region_modes_oop(s_oop), rep)
    a_sl = timeit(lambda: SlantSolver(PX, PY, 2, 2, M, oopc, slant=(0.75, 0.4),
                                      alpha0x=A0X, alpha0y=A0Y, k0=K0), rep)
    s_sl = SlantSolver(PX, PY, 2, 2, M, oopc, slant=(0.75, 0.4), alpha0x=A0X,
                       alpha0y=A0Y, k0=K0)
    e_sl = timeit(lambda: slant_region_modes(s_sl), rep)
    # a SCALAR cell that is SLANTED (the common device case)
    a_sls = timeit(lambda: SlantSolver(PX, PY, 2, 2, M, scal, slant=(0.75, 0.4),
                                       alpha0x=A0X, alpha0y=A0Y, k0=K0), rep)
    s_sls = SlantSolver(PX, PY, 2, 2, M, scal, slant=(0.75, 0.4), alpha0x=A0X,
                        alpha0y=A0Y, k0=K0)
    e_sls = timeit(lambda: slant_region_modes(s_sls), rep)
    row = dict(M=M, qq=int(q * q), dim_inplane=int(2 * q * q),
               dim_oop=int(4 * q * q),
               asm_inplane=a_in, eig_inplane=e_in,
               asm_oop=a_oop, eig_oop=e_oop,
               asm_slant_tensor=a_sl, eig_slant_tensor=e_sl,
               asm_slant_scalar=a_sls, eig_slant_scalar=e_sls)
    row["tot_inplane"] = a_in + e_in
    row["tot_oop"] = a_oop + e_oop
    row["tot_slant_tensor"] = a_sl + e_sl
    row["tot_slant_scalar"] = a_sls + e_sls
    row["slant_over_oop"] = row["tot_slant_tensor"] / row["tot_oop"]
    row["slant_over_inplane"] = row["tot_slant_scalar"] / row["tot_inplane"]
    row["oop_over_inplane"] = row["tot_oop"] / row["tot_inplane"]
    rows.append(row)
    print(f"  M={M} q^2={q * q:4d}  in-plane {row['tot_inplane'] * 1e3:8.1f} ms | "
          f"OOP {row['tot_oop'] * 1e3:8.1f} ms | slant(tensor) "
          f"{row['tot_slant_tensor'] * 1e3:8.1f} ms | slant(scalar) "
          f"{row['tot_slant_scalar'] * 1e3:8.1f} ms  ->  slant/OOP "
          f"{row['slant_over_oop']:.2f}x  slant/in-plane "
          f"{row['slant_over_inplane']:.2f}x  OOP/in-plane "
          f"{row['oop_over_inplane']:.2f}x")
res["T6a_region"] = rows

print("\nT6b  END-TO-END single-layer solve (scalar pillar)")
rows = []
for M in (5, 6, 7):
    tv = timeit(lambda: solve_slant_stack(
        PX, PY, [{"thickness": 0.8, "cell": scal, "slant": (0.0, 0.0)}],
        1.0, 1.5, WL, M=M, n_orders=3, theta=np.deg2rad(20),
        phi=np.deg2rad(35)), 2)
    tsl = timeit(lambda: solve_slant_stack(
        PX, PY, [{"thickness": 0.8, "cell": scal, "slant": (0.75, 0.0)}],
        1.0, 1.5, WL, M=M, n_orders=3, theta=np.deg2rad(20),
        phi=np.deg2rad(35)), 2)
    rows.append({"M": M, "vertical_s": tv, "slanted_s": tsl,
                 "ratio": tsl / tv})
    print(f"  M={M}  vertical {tv:6.3f}s  slanted {tsl:6.3f}s  "
          f"-> {tsl / tv:.2f}x")
res["T6b_end_to_end"] = rows

print("\nT6c  staircase equivalence (read back from M4b)")
try:
    m4b = json.load(open(os.path.join(OUT, "m4b_staircase_ladder.json")))
    eq = {}
    for key, blk in m4b.items():
        if not isinstance(blk, dict) or "metric" not in blk:
            continue
        met_t = blk["metric"]["t_s"]
        eq[key] = {"metric_t_s": met_t, "h": blk["h"], "stair": {}}
        for k, v in blk.items():
            if not k.startswith("n") or not isinstance(v, dict):
                continue
            eq[key]["stair"][k] = {"vs_metric": v["vs_metric"],
                                   "t_s": v["t_s"],
                                   "cost_ratio": v["t_s"] / met_t}
        print(f"  {key}: metric {met_t:.1f}s; " + "; ".join(
            f"{k} {v['vs_metric']:.2e} @ {v['cost_ratio']:.2f}x"
            for k, v in eq[key]["stair"].items()))
    res["T6c_staircase"] = eq
except FileNotFoundError:
    res["T6c_staircase"] = "m4b_staircase_ladder.json not present -- run it first"
    print("  (m4b_staircase_ladder.json not present)")

res["wall_s"] = time.time() - t00
with open(os.path.join(OUT, "m6_cost.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m6_cost.json  ({res['wall_s']:.1f} s)")
