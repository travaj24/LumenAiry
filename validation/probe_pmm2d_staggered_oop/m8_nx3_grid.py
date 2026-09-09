"""M8 -- a (3,3) grid: does the prototype hold on a non-trivial cell?

The plan's method line asks for ``Nx = Ny = 2 AND 3``.  M1-M7 use (2,2); this
runs a (3,3) cell with a NON-CENTRO-SYMMETRIC out-of-plane feature (an L of
tilted-uniaxial pixels), which is also the geometry that breaks the parity
structure M7 T2 looks for.  Measured:

  * both candidates vs the hybrid ``pmm_jones_2d`` and ``rcwa_jones_2d``,
    against the oracles' own mutual spread;
  * POSITION INVARIANCE: cyclically shifting the feature inside the cell must
    leave the TOTAL R / T unchanged (a property of the shipped basis that the
    tensor blocks must not break);
  * energy closure and the forward/backward split count.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m8_nx3_grid.py
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


def lcell(er, eg, sx=0, sy=0):
    """An L of three ``er`` pixels in a (3,3) cell of ``eg``, cyclically
    shifted by ``(sx, sy)``."""
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = eg
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[(i + sx) % 3, (j + sy) % 3] = er
    return e


def upsample(ec, n):
    """Pixel-replicate a coarse eps cell (rcwa_jones_2d refuses a cell sampled
    below 4*n_orders+1 per axis)."""
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


def main():
    pc.banner("M8 -- (3,3) grid, non-centro-symmetric OOP feature")
    R = {}
    er = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    eg = np.eye(3, dtype=complex)
    ec = lcell(er, eg)
    for th, ph in ((0.0, 0.0), (np.deg2rad(20.0), np.deg2rad(35.0))):
        print(f"\n=== theta = {np.rad2deg(th):.0f}, phi = "
              f"{np.rad2deg(ph):.0f} ===")
        orc = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for no in (5, 7):
                fine = upsample(ec, int(np.ceil((4 * no + 1) / ec.shape[0])))
                _o, Rm, Tm, J = rcwa_jones_2d(PX, PY, fine, NSUB, NSUP, DEPTH,
                                              WL, theta=th, phi=ph,
                                              n_orders_x=no, n_orders_y=no)
                orc[("rcwa", no)] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
            for no in (9, 11):
                _o, Rm, Tm, J = pmm_jones_2d(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                             theta=th, phi=ph, degree=9,
                                             n_orders=no, stabilize=True)
                orc[("hyb", no)] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
        for k, v in orc.items():
            print(f"   {k[0]}({k[1]:2d})  Rtot = {v[0]}  |R+T-1| = "
                  f"{np.max(np.abs(v[0]+v[1]-1)):.2e}")
        spread = max(
            float(np.max(np.abs(orc[("rcwa", 7)][0] - orc[("hyb", 11)][0]))),
            float(np.max(np.abs(orc[("rcwa", 7)][1] - orc[("hyb", 11)][1]))))
        spreadJ = float(np.max(np.abs(orc[("rcwa", 7)][2]
                                      - orc[("hyb", 11)][2])))
        print(f"   ORACLE SPREAD rcwa(7) vs hybrid(11): R/T {spread:.2e}  "
              f"Jones {spreadJ:.2e}   <- the derived bar")
        R[f"oracle_spread_th{int(np.rad2deg(th))}"] = dict(RT=spread,
                                                           J=spreadJ)
        print("   cand  M  dim   |dR| vs hyb  |dR| vs rcwa   |dJ| vs hyb  "
              " |R+T-1|   fwd/bwd")
        for cand, Ms in (("a", (4, 5, 6)), ("d", (4, 5))):
            for M in Ms:
                o2, R2, T2, J2, ex = pc.solve_slab(
                    PX, PY, ec, NSUB, NSUP, DEPTH, WL, M=M, theta=th, phi=ph,
                    candidate=cand, return_modes=True)
                Rt, Tt = R2.sum(axis=1), T2.sum(axis=1)
                dh = float(np.max(np.abs(Rt - orc[("hyb", 11)][0])))
                dr = float(np.max(np.abs(Rt - orc[("rcwa", 7)][0])))
                dJ = float(np.max(np.abs(J2 - orc[("hyb", 11)][2])))
                q = 3 * (M - 1)
                print(f"    ({cand}) {M:2d} {2*q*q:5d}  {dh:.2e}        "
                      f"{dr:.2e}     {dJ:.2e}   "
                      f"{np.max(np.abs(Rt+Tt-1)):.2e}  "
                      f"{ex['fidx'].size}/{ex['bidx'].size}")
                R[f"th{int(np.rad2deg(th))}|{cand}|M{M}"] = dict(
                    dR_hyb=dh, dR_rcwa=dr, dJ=dJ,
                    closure=float(np.max(np.abs(Rt + Tt - 1))),
                    nfwd=int(ex["fidx"].size))
        print("   POSITION INVARIANCE (candidate (a), M=5): total R / T under "
              "a cyclic shift of the feature")
        base = None
        for sx, sy in ((0, 0), (1, 0), (1, 2), (2, 2)):
            _o, Rm, Tm, _J = pc.solve_slab(
                PX, PY, lcell(er, eg, sx, sy), NSUB, NSUP, DEPTH, WL, M=5,
                theta=th, phi=ph, candidate="a")
            tot = np.concatenate([Rm.sum(axis=1), Tm.sum(axis=1)])
            if base is None:
                base = tot
            print(f"     shift ({sx},{sy}): Rtot = {tot[:2]}  max|delta| = "
                  f"{np.max(np.abs(tot - base)):.2e}")
            R[f"posinv_th{int(np.rad2deg(th))}_{sx}{sy}"] = float(
                np.max(np.abs(tot - base)))
    with open(os.path.join(OUT, "m8_nx3_grid.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m8_nx3_grid.json")


if __name__ == "__main__":
    main()
