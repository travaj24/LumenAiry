"""M7b -- does the HYBRID really refuse slant x OUT-OF-PLANE?

M7's T7a asked `PMM2DStackHybrid.add_layer`, which ACCEPTED both an in-plane and
an out-of-plane tensor with a slant -- contradicting the method's own docstring
("RESTRICTIONS, all raising: ... combined with OUT-OF-PLANE tensor components").
The claim in this document must be what the engine DOES, so this drives the
whole solve and records the outcome, and -- if it does produce an answer --
compares it to the prototype and to the 1-D oracle on the same y-uniform
out-of-plane stripe M7's T7b uses.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import (  # noqa: E402
    assert_worktree, solve_slant_stack, tensor_uniaxial,
)
from lumenairy.elements.pmm import (  # noqa: E402
    PMM2DStackHybrid, pmm_jones_1d_slanted,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 0.75
WL, DEPTH = 1.0, 0.30
NSUP, NSUB = 1.0, 1.5
ORD = [-2, -1, 0, 1, 2]
TILT_LC = tensor_uniaxial(1.5, 1.7, np.deg2rad(35), np.deg2rad(25))
INPL_LC = tensor_uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25))
res = {"lumenairy": assert_worktree()}
t0 = time.time()
PHI = 35.0
TP = -np.tan(np.deg2rad(PHI))        # the prototype's internal t
TH = np.deg2rad(25)


def px_stripe(N, t33):
    c = np.zeros((N, N, 3, 3), dtype=complex)
    c[:, :] = np.eye(3)
    c[:N // 2, :] = t33
    return c


def cell_stripe(t33):
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[0, :] = t33
    c[1, :] = np.eye(3)
    return c


def d(a, b):
    return max(max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1]))
               for k in ORD if k in a and k in b)


def as_map(o, R, T, row):
    return {int(m): (float(R[row, i]), float(T[row, i]))
            for i, m in enumerate(o)}


# the 1-D oracle and the prototype, on the same stripe
o, R, T, _J = pmm_jones_1d_slanted(
    PX, TILT_LC, np.eye(3), NSUB, NSUP, DEPTH, 0.5, WL, np.deg2rad(PHI),
    angle=TH, degree=30, far_field_orders=15, factorization="convection")
orc = {"ex": as_map(o, R, T, 0), "ey": as_map(o, R, T, 1)}

o2, R2, T2, _Jr, _Jt, _i = solve_slant_stack(
    PX, PY, [{"thickness": DEPTH, "cell": cell_stripe(TILT_LC),
              "slant": (TP, 0.0)}],
    NSUP, NSUB, WL, M=8, n_orders=4, theta=TH, phi=0.0)
idx = {int(m): i for i, (m, n) in enumerate(o2) if n == 0}
pro = {"ex": {m: (float(R2[0, i]), float(T2[0, i])) for m, i in idx.items()},
       "ey": {m: (float(R2[1, i]), float(T2[1, i])) for m, i in idx.items()}}
res["prototype_vs_1d"] = {"ex": d(pro["ex"], orc["ex"]),
                          "ey": d(pro["ey"], orc["ey"])}
print(f"prototype (M=8) vs pmm_jones_1d_slanted: Ex "
      f"{res['prototype_vs_1d']['ex']:.2e}  Ey "
      f"{res['prototype_vs_1d']['ey']:.2e}", flush=True)

for nm, t33 in (("out_of_plane", TILT_LC), ("in_plane", INPL_LC)):
    row = {}
    for stage in ("add_layer", "solve"):
        row[stage] = "not reached"
    try:
        st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                              degree=11, n_orders=9, symmetry=False)
        st.add_layer(DEPTH, eps_tensor_cell=px_stripe(24, t33),
                     slant=(-TP, 0.0))       # the hybrid's PUBLIC sign
        row["add_layer"] = "ACCEPTED"
    except Exception as exc:                                # noqa: BLE001
        row["add_layer"] = f"{type(exc).__name__}: {exc}"
        res[nm] = row
        print(f"{nm}: add_layer -> {row['add_layer'][:160]}", flush=True)
        continue
    try:
        st.set_source(WL, theta=TH, phi=0.0)
        out = st.solve()
        Rh, Th_ = np.atleast_2d(out[1]), np.atleast_2d(out[2])
        hv = {"ex": as_map(out[0], Rh, Th_, 0),
              "ey": as_map(out[0], Rh, Th_, 1)}
        row["solve"] = "ACCEPTED"
        row["vs_1d_ex"] = d(hv["ex"], orc["ex"])
        row["vs_1d_ey"] = d(hv["ey"], orc["ey"])
        row["vs_proto_ex"] = d(hv["ex"], pro["ex"])
        row["vs_proto_ey"] = d(hv["ey"], pro["ey"])
        print(f"{nm}: solve -> ACCEPTED; vs 1-D oracle Ex "
              f"{row['vs_1d_ex']:.2e} Ey {row['vs_1d_ey']:.2e}; "
              f"vs prototype Ex {row['vs_proto_ex']:.2e} Ey "
              f"{row['vs_proto_ey']:.2e}", flush=True)
    except Exception as exc:                                # noqa: BLE001
        row["solve"] = f"{type(exc).__name__}: {exc}"
        print(f"{nm}: solve -> {row['solve'][:200]}", flush=True)
    res[nm] = row

res["wall_s"] = time.time() - t0
with open(os.path.join(OUT, "m7b_hybrid_oop_slant.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"\nWROTE results/m7b_hybrid_oop_slant.json  ({res['wall_s']:.1f} s)")
