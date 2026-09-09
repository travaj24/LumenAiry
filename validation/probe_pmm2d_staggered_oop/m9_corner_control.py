"""M9 -- is the (3,3) L-cell three-engine disagreement OUT-OF-PLANE or CORNER?

M8 finds all three engines disagreeing at ~2.2e-03 on a (3,3) cell whose
feature is an L (i.e. it has a RE-ENTRANT 270-degree corner, the hardest case
for every method).  That is a big number next to M2 (1e-15 on a uniform OOP
slab) and M5 (4e-05 on a convex OOP pillar), so it must be attributed.

THE CONTROL: run the SAME cell with the out-of-plane entries zeroed, so the
staggered arm is the SHIPPED in-plane discretization (probe candidate
``eform``, proved bit-identical to ``Granet2DTransverseE`` in M0.3 / M3 T3) and
the oracles are on their in-plane paths.  If the disagreement survives, it is
the CORNER and has nothing to do with this prototype.  A convex-pillar cell on
the same (3,3) grid is run alongside as the second half of the two-sided
claim.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m9_corner_control.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
DEPTH = 0.4
NSUB, NSUP = 1.5, 1.0


def upsample(ec, n):
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


def strip_oop(t):
    o = np.array(t, dtype=complex)
    o[0, 2] = o[1, 2] = o[2, 0] = o[2, 1] = 0.0
    return o


def build(kind, er, eg):
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = eg
    if kind == "L (re-entrant corner)":
        for i, j in ((0, 0), (1, 0), (0, 1)):
            e[i, j] = er
    else:                                    # single convex pixel
        e[1, 1] = er
    return e


def main():
    pc.banner("M9 -- corner vs out-of-plane attribution on a (3,3) cell")
    R = {}
    er_oop = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    eg = np.eye(3, dtype=complex)
    for kind in ("L (re-entrant corner)", "single convex pixel"):
        for tag, er in (("OOP tensor", er_oop),
                        ("IN-PLANE (OOP entries zeroed)", strip_oop(er_oop))):
            ec = build(kind, er, eg)
            cand = "a" if tag == "OOP tensor" else "eform"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rc = {}
                for no in (5, 7):
                    fine = upsample(ec, int(np.ceil((4 * no + 1) / 3)))
                    _o, Rm, Tm, J = rcwa_jones_2d(PX, PY, fine, NSUB, NSUP,
                                                  DEPTH, WL, n_orders_x=no,
                                                  n_orders_y=no)
                    rc[no] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
                _o, Rm, Tm, Jh = pmm_jones_2d(PX, PY, ec, NSUB, NSUP, DEPTH,
                                              WL, degree=9, n_orders=11,
                                              stabilize=True)
            hy = (Rm.sum(axis=1), Tm.sum(axis=1), Jh)
            print(f"\n=== {kind}  |  {tag}  (staggered candidate '{cand}') ===")
            print(f"   rcwa(5)  Rtot = {rc[5][0]}  |R+T-1| = "
                  f"{np.max(np.abs(rc[5][0]+rc[5][1]-1)):.2e}")
            print(f"   rcwa(7)  Rtot = {rc[7][0]}  |R+T-1| = "
                  f"{np.max(np.abs(rc[7][0]+rc[7][1]-1)):.2e}   "
                  f"(5->7 movement {np.max(np.abs(rc[7][0]-rc[5][0])):.2e})")
            print(f"   hyb(11)  Rtot = {hy[0]}  |R+T-1| = "
                  f"{np.max(np.abs(hy[0]+hy[1]-1)):.2e}")
            spread = float(np.max(np.abs(rc[7][0] - hy[0])))
            print(f"   ORACLE SPREAD rcwa(7) vs hybrid(11) = {spread:.2e}")
            row = dict(spread=spread)
            for M in (4, 5, 6):
                _o, R2, T2, J2 = pc.solve_slab(PX, PY, ec, NSUB, NSUP, DEPTH,
                                               WL, M=M, candidate=cand)
                Rt, Tt = R2.sum(axis=1), T2.sum(axis=1)
                print(f"   staggered M={M} dim={2*(3*(M-1))**2:4d}  "
                      f"Rtot = {Rt}  |dR| vs hyb = "
                      f"{np.max(np.abs(Rt-hy[0])):.2e}  vs rcwa(7) = "
                      f"{np.max(np.abs(Rt-rc[7][0])):.2e}   own |R+T-1| = "
                      f"{np.max(np.abs(Rt+Tt-1)):.2e}")
                row[f"M{M}_dhyb"] = float(np.max(np.abs(Rt - hy[0])))
                row[f"M{M}_drcwa"] = float(np.max(np.abs(Rt - rc[7][0])))
                row[f"M{M}_closure"] = float(np.max(np.abs(Rt + Tt - 1)))
            R[f"{kind}|{tag}"] = row
    with open(os.path.join(OUT, "m9_corner_control.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m9_corner_control.json")


if __name__ == "__main__":
    main()
