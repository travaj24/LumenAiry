"""M1b -- the FRAME-ANCHOR PHASE, pinned two-sided.

M1 shows the uniform-layer null is EXACT for R, T and the reflection Jones but
that the TRANSMISSION Jones moves by O(0.1-0.6) at oblique / conical incidence
whenever ``t . k_t != 0``.  The prediction (slant_lib docstring) is that this is
NOT an error but the frame-anchor bookkeeping: with the frame anchored at the
layer TOP, the substrate plane sits at ``u = x - t d``, so the transmitted
order-m amplitude carries the unimodular diagonal phase
``exp(-i alpha_m . t d)``.

This scans the three arms -- no correction, ``exp(-i ...)``, ``exp(+i ...)`` --
on the uniform null (where the exact answer is known: the unslanted layer).
The correct sign must collapse dJt to the dR/dJr floor; the other two must not.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, solve_slant_stack, tensor_uniaxial  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = {"lumenairy": assert_worktree(), "rows": []}
t0 = time.time()
px = py = 0.9
depth, wl, M, NO = 0.35, 1.0, 5, 4


def uni(t33):
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[:, :] = t33
    return c


tens = {"isotropic": uni(2.25 * np.eye(3)),
        "oop_ua": uni(tensor_uniaxial(1.5, 1.7, np.deg2rad(35),
                                      np.deg2rad(25)))}
mounts = {"oblique25": (np.deg2rad(25), 0.0),
          "conical25_40": (np.deg2rad(25), np.deg2rad(40))}
slants = {"x10": (np.tan(np.deg2rad(10)), 0.0),
          "x35": (np.tan(np.deg2rad(35)), 0.0),
          "diag35": (np.tan(np.deg2rad(35)), 0.7 * np.tan(np.deg2rad(35)))}
arms = {"none": dict(frame_phase=False),
        "minus": dict(frame_phase=True, phase_sign=-1.0),
        "plus": dict(frame_phase=True, phase_sign=+1.0)}

for tname, cell in tens.items():
    for mname, (th, ph) in mounts.items():
        _o, R0, T0, Jr0, Jt0, _i = solve_slant_stack(
            px, py, [{"thickness": depth, "cell": cell, "slant": (0.0, 0.0)}],
            1.0, 1.5, wl, M=M, n_orders=NO, theta=th, phi=ph)
        for sname, sl in slants.items():
            row = {"tensor": tname, "mount": mname, "slant": sname}
            for aname, kw in arms.items():
                _o1, R1, T1, Jr1, Jt1, _j = solve_slant_stack(
                    px, py,
                    [{"thickness": depth, "cell": cell, "slant": sl}],
                    1.0, 1.5, wl, M=M, n_orders=NO, theta=th, phi=ph, **kw)
                row[aname] = float(np.max(np.abs(Jt1 - Jt0)))
            row["dJr"] = float(np.max(np.abs(Jr1 - Jr0)))
            res["rows"].append(row)
            print(f"M1b {tname:10s} {mname:12s} {sname:7s} "
                  f"none {row['none']:.2e} | -i {row['minus']:.2e} | "
                  f"+i {row['plus']:.2e} | (dJr {row['dJr']:.2e})")

res["worst_minus"] = max(r["minus"] for r in res["rows"])
res["best_none"] = min(r["none"] for r in res["rows"])
res["best_plus"] = min(r["plus"] for r in res["rows"])
res["wall_s"] = time.time() - t0
print(f"\nworst 'minus' arm {res['worst_minus']:.3e}; best 'none' "
      f"{res['best_none']:.3e}; best 'plus' {res['best_plus']:.3e}")
with open(os.path.join(OUT, "m1b_frame_phase.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"WROTE results/m1b_frame_phase.json  ({res['wall_s']:.1f} s)")
