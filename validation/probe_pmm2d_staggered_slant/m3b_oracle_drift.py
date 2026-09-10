"""M3b -- how converged is the 1-D ORACLE itself?

M3's TM residual plateaus at ~1e-4..1e-3 while TE reaches 1e-7.  The honest
question is whose floor that is.  This measures the 1-D oracle's OWN per-order
drift over its ``degree`` ladder on the same geometry, slant 0 / 10 / 20 / 35,
normal + oblique, both polarizations -- if the oracle's own drift is the size of
the M3 residual, the residual is ORACLE-limited and the 2-D slanted arm is not
implicated.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX, WL, DEPTH = 0.75, 1.0, 0.30
NR, NG, NSUP, NSUB = 2.0, 1.0, 1.0, 1.5
ORD_CMP = [-2, -1, 0, 1, 2]
res = {"lumenairy": assert_worktree(), "rows": []}
t0 = time.time()


def run(phi_deg, theta, pol, degree):
    if phi_deg == 0.0:
        o, R, T = pmm_efficiency_1d(PX, NR, NG, NSUB, NSUP, DEPTH, 0.5, WL,
                                    angle=theta, polarization=pol,
                                    degree=degree, far_field_orders=15)
    else:
        o, R, T = pmm_efficiency_1d_slanted(
            PX, NR, NG, NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(phi_deg),
            angle=theta, polarization=pol, degree=degree, far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def d(a, b):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in ORD_CMP if m in a and m in b)


for phi_deg in (0.0, 10.0, 20.0, 35.0):
    for mount, th in (("normal", 0.0), ("oblique25", np.deg2rad(25))):
        for pol in ("te", "tm"):
            ref = run(phi_deg, th, pol, 34)
            row = {"phi": phi_deg, "mount": mount, "pol": pol}
            for deg in (18, 22, 26, 30):
                row[f"d{deg}"] = d(run(phi_deg, th, pol, deg), ref)
            res["rows"].append(row)
            print(f"M3b phi={phi_deg:4.0f} {mount:9s} {pol}  "
                  f"deg18 {row['d18']:.2e} 22 {row['d22']:.2e} "
                  f"26 {row['d26']:.2e} 30 {row['d30']:.2e}  (ref deg34)")

res["worst_d22"] = max(r["d22"] for r in res["rows"])
res["worst_d22_tm"] = max(r["d22"] for r in res["rows"] if r["pol"] == "tm")
res["worst_d22_te"] = max(r["d22"] for r in res["rows"] if r["pol"] == "te")
res["wall_s"] = time.time() - t0
print(f"\noracle own drift at the M3 setting (degree 22): "
      f"TM {res['worst_d22_tm']:.2e}, TE {res['worst_d22_te']:.2e}")
with open(os.path.join(OUT, "m3b_oracle_drift.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"WROTE results/m3b_oracle_drift.json  ({res['wall_s']:.1f} s)")
