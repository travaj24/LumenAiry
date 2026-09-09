"""V2d -- the H gauge flipped on the CHIRAL fixtures too.

v2c walks ``_OOP_H_GAUGE`` on uniform / mixed stacks and on the forced
in-plane reduction.  The brief also asks for the flip on the two chiral
fixtures the rotation gauge is arbitrated on, so that BOTH constants have a
fail-before on BOTH shapes.  Cheap: one extra arm per fixture.

Usage:  PYTHONPATH=<root> python v2d_hgauge_on_chiral.py <root> <out.json>
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm import pmm_jones_1d_segments  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import (  # noqa: E402
    rcwa_jones_1d_segments,
    rcwa_jones_2d,
    uniaxial_tensor,
)

OUT = sys.argv[2]

# ---- the v2a chiral stripe -------------------------------------------------
tA = uniaxial_tensor(1.48, 1.73, 0.62, phi=0.37)
tB = 2.10 * np.eye(3, dtype=complex)
tC = uniaxial_tensor(1.55, 1.80, 1.02, phi=2.20)
SEGS = [(1.0 / 3.0, tA), (1.0 / 3.0, tB), (1.0 / 3.0, tC)]
CHIRAL = np.zeros((3, 3, 3, 3), dtype=complex)
for i, t in enumerate((tA, tB, tC)):
    CHIRAL[i, :, :, :] = t
PX1, WL1, DEP1, NSUB1 = 1.15e-6, 0.70e-6, 0.31e-6, 1.45

# ---- the v2b chiral 2-D cell ----------------------------------------------
t2A = uniaxial_tensor(1.46, 1.74, 0.58, phi=0.31)
t2C = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
CELL2 = np.zeros((3, 3, 3, 3), dtype=complex)
CELL2[:, :] = np.eye(3, dtype=complex)
CELL2[0, 0] = t2A
CELL2[1, 0] = t2C
CELL2[1, 2] = 2.25 * np.eye(3, dtype=complex)
PX2, WL2, DEP2, NSUB2 = 1.10e-6, 0.68e-6, 0.36e-6, 1.50

GAUGES = [(-1j, "-1j (shipped)"), (+1j, "+1j")]
out = {"root": ROOT, "lumenairy": lumenairy.__file__, "cases": []}


def align1(o2, A2, o1, A1):
    o2 = np.asarray(o2)
    sel = np.where(o2[:, 1] == 0)[0]
    m1 = {int(m): j for j, m in enumerate(np.asarray(o1))}
    d = 0.0
    for j in sel:
        k = m1.get(int(o2[j, 0]))
        if k is not None:
            d = max(d, float(np.max(np.abs(np.asarray(A2)[:, j]
                                           - np.asarray(A1)[:, k]))))
    return d


def per_order2(o_a, R_a, T_a, o_b, R_b, T_b):
    mb = {(int(m), int(n)): j for j, (m, n) in enumerate(np.asarray(o_b))}
    dR = dT = 0.0
    for i, (m, nn) in enumerate(np.asarray(o_a)):
        k = mb.get((int(m), int(nn)))
        if k is None:
            continue
        dR = max(dR, float(np.max(np.abs(np.asarray(R_a)[:, i]
                                         - np.asarray(R_b)[:, k]))))
        dT = max(dT, float(np.max(np.abs(np.asarray(T_a)[:, i]
                                         - np.asarray(T_b)[:, k]))))
    return dR, dT


for th, tag in ((np.deg2rad(25.0), "oblique25"), (0.0, "normal")):
    op = pmm_jones_1d_segments(PX1, SEGS, NSUB1, 1.0, DEP1, WL1, angle=th,
                               degree=18, far_field_orders=31, stabilize=False)
    orc = rcwa_jones_1d_segments(PX1, SEGS, NSUB1, 1.0, DEP1, WL1, angle=th,
                                 n_orders=41)
    rec = {"fixture": "chiral_oop_stripe", "mount": tag, "arms": []}
    for g, gname in GAUGES:
        TS._OOP_H_GAUGE = g
        s = pmm_jones_2d_staggered(PX1, PX1, CHIRAL, NSUB1, 1.0, DEP1, WL1,
                                   degree=7, n_orders=5, theta=th)
        a = {"gauge": gname,
             "dR_vs_pmm1d": align1(s[0], s[1], op[0], op[1]),
             "dT_vs_pmm1d": align1(s[0], s[2], op[0], op[2]),
             "dJones_vs_pmm1d": float(np.max(np.abs(np.asarray(s[3])
                                                    - np.asarray(op[3])))),
             "dR_vs_rcwa1d": align1(s[0], s[1], orc[0], orc[1]),
             "dJones_vs_rcwa1d": float(np.max(np.abs(np.asarray(s[3])
                                                     - np.asarray(orc[3])))),
             "sumRT": [float(np.sum(s[1][r]) + np.sum(s[2][r]))
                       for r in (0, 1)]}
        rec["arms"].append(a)
        print(f"[stripe {tag}] gauge={gname:14s} vs pmm1d dR="
              f"{a['dR_vs_pmm1d']:.3e} dT={a['dT_vs_pmm1d']:.3e} dJ="
              f"{a['dJones_vs_pmm1d']:.3e} | vs rcwa1d dR="
              f"{a['dR_vs_rcwa1d']:.3e} | R+T={a['sumRT'][0]:.9f}", flush=True)
    TS._OOP_H_GAUGE = -1j
    out["cases"].append(rec)
    json.dump(out, open(OUT, "w"), indent=1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ref = rcwa_jones_2d(PX2, PX2,
                        np.repeat(np.repeat(CELL2, 13, axis=0), 13, axis=1),
                        NSUB2, 1.0, DEP2, WL2, theta=np.deg2rad(25.0), phi=0.0,
                        n_orders_x=9, n_orders_y=9)
rec = {"fixture": "chiral_2d_cell", "mount": "oblique25", "arms": []}
for g, gname in GAUGES:
    TS._OOP_H_GAUGE = g
    s = pmm_jones_2d_staggered(PX2, PX2, CELL2, NSUB2, 1.0, DEP2, WL2,
                               degree=6, n_orders=3, theta=np.deg2rad(25.0))
    dR, dT = per_order2(*s[:3], *ref[:3])
    a = {"gauge": gname, "dR": dR, "dT": dT,
         "dJones": float(np.max(np.abs(np.asarray(s[3]) - np.asarray(ref[3])))),
         "sumRT": [float(np.sum(s[1][r]) + np.sum(s[2][r])) for r in (0, 1)]}
    rec["arms"].append(a)
    print(f"[2d oblique25] gauge={gname:14s} dR={dR:.3e} dT={dT:.3e} "
          f"dJ={a['dJones']:.3e} R+T={a['sumRT'][0]:.9f}", flush=True)
TS._OOP_H_GAUGE = -1j
out["cases"].append(rec)
json.dump(out, open(OUT, "w"), indent=1)
print("DONE")
