"""M1 -- NULL TEST.  A shear of a HOMOGENEOUS medium is a pure coordinate change:
a uniform layer 'slanted' by any phi must give the SAME R / T / Jones as the
unslanted one, because a homogeneous medium has no walls to tilt.

Two arms:
  (a) uniform ISOTROPIC layer;
  (b) uniform ANISOTROPIC layer (in-plane uniaxial, out-of-plane uniaxial,
      gyrotropic, lossy) -- the metric transform eps -> A^-1 eps A^-T must be an
      IDENTITY on observables there too, which is the sharp arm (an isotropic
      cell's covariant tensor is nearly the metric alone).

Reference for each row is the SAME cell at slant (0, 0) through the same driver.
Normal incidence cannot see the shear on the (0,0) order alone (k_t = 0 kills
the convection), so oblique / conical rows are the informative ones.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    assert_worktree,
    solve_slant_stack,
    tensor_uniaxial,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree(), "rows": []}
t00 = time.time()

px = py = 0.9
depth, wl = 0.35, 1.0
M, NO = 5, 4
nsup, nsub = 1.0, 1.5


def uni(t33):
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[:, :] = t33
    return c


tens = {
    "isotropic":  uni(2.25 * np.eye(3)),
    "inplane_ua": uni(tensor_uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25))),
    "oop_ua":     uni(tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))),
    "gyrotropic": uni(np.array([[2.25, 0.6j, 0], [-0.6j, 2.25, 0],
                                [0, 0, 2.25]], dtype=complex)),
    "lossy_oop":  uni(tensor_uniaxial(1.5, 1.7 + 0.08j, np.deg2rad(35),
                                      np.deg2rad(25))),
}
mounts = {"normal": (0.0, 0.0), "oblique25": (np.deg2rad(25), 0.0),
          "conical25_40": (np.deg2rad(25), np.deg2rad(40))}
slants = [(np.tan(np.deg2rad(10)), 0.0),
          (np.tan(np.deg2rad(35)), 0.0),
          (0.0, np.tan(np.deg2rad(35))),
          (np.tan(np.deg2rad(35)), 0.7 * np.tan(np.deg2rad(35)))]
slant_names = ["x10", "x35", "y35", "diag35"]

for tname, cell in tens.items():
    for mname, (th, ph) in mounts.items():
        o0, R0, T0, Jr0, Jt0, _i = solve_slant_stack(
            px, py, [{"thickness": depth, "cell": cell, "slant": (0.0, 0.0)}],
            nsup, nsub, wl, M=M, n_orders=NO, theta=th, phi=ph)
        for sname, sl in zip(slant_names, slants):
            o1, R1, T1, Jr1, Jt1, _j = solve_slant_stack(
                px, py, [{"thickness": depth, "cell": cell, "slant": sl}],
                nsup, nsub, wl, M=M, n_orders=NO, theta=th, phi=ph)
            row = dict(tensor=tname, mount=mname, slant=sname,
                       dR=float(np.max(np.abs(R1 - R0))),
                       dT=float(np.max(np.abs(T1 - T0))),
                       dJr=float(np.max(np.abs(Jr1 - Jr0))),
                       dJt=float(np.max(np.abs(Jt1 - Jt0))),
                       leak=float(np.max(np.abs(
                           np.delete(R1, np.where((o1[:, 0] == 0)
                                                  & (o1[:, 1] == 0))[0],
                                     axis=1)))))
            res["rows"].append(row)
            print(f"M1 {tname:11s} {mname:12s} {sname:7s} dR {row['dR']:.2e} "
                  f"dT {row['dT']:.2e} dJr {row['dJr']:.2e} "
                  f"dJt {row['dJt']:.2e} leak {row['leak']:.2e}")

worst = max(max(r["dR"], r["dT"], r["dJr"], r["dJt"]) for r in res["rows"])
worst_ob = max(max(r["dR"], r["dT"], r["dJr"], r["dJt"])
               for r in res["rows"] if r["mount"] != "normal")
res["worst_all"] = worst
res["worst_oblique_conical"] = worst_ob
res["wall_s"] = time.time() - t00
print(f"\nWORST over all rows {worst:.3e}; oblique/conical only {worst_ob:.3e}")
with open(os.path.join(OUT, "m1_null.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"WROTE results/m1_null.json  ({res['wall_s']:.1f} s)")
