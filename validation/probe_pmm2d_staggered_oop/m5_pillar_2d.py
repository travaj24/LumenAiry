"""M5 -- a genuinely 2-D OOP pillar vs the hybrid ``pmm_jones_2d``.

The hybrid (Fourier-floored, 4Nf OOP generator) is the only other engine in the
library that solves a 2-D out-of-plane cell, so it is the cross-engine oracle
here -- with its OWN floor, which is measured first (its ``n_orders`` ladder)
so the comparison bar is derived rather than assumed.

Also measured: the NO-FLOOR property, two-sided -- the staggered result must be
independent of ``n_orders`` while the hybrid's is not.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m5_pillar_2d.py
"""
import json
import os
import sys
import time
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


def cell(er, eg):
    """One quadrant pillar (0,0) of a (2,2) cell -- a half-period rectangular
    pillar of ``er`` in a background ``eg``."""
    e = np.zeros((2, 2, 3, 3), dtype=complex)
    e[:, :] = eg
    e[0, 0] = er
    return e


def upsample(ec, n):
    """Pixel-replicate a coarse eps cell to (n*Nx, n*Ny).  ``rcwa_jones_2d``
    refuses a cell sampled below 4*n_orders+1 per axis (Fourier aliasing), so
    the RCWA oracle gets an EXACTLY equivalent, finely sampled copy of the same
    geometry."""
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


def totals(o, Rm, Tm):
    return Rm.sum(axis=1), Tm.sum(axis=1)


def main():
    pc.banner("M5 -- 2-D OOP pillar vs the hybrid pmm_jones_2d")
    R = {}
    er = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    eg = np.eye(3, dtype=complex)
    ec = cell(er, eg)
    print(f"  pillar tensor e13 = {er[0,2]:.5f}  e23 = {er[1,2]:.5f}")
    for th, ph in ((0.0, 0.0), (np.deg2rad(20.0), np.deg2rad(35.0))):
        print(f"\n=== theta = {np.rad2deg(th):.0f}, phi = "
              f"{np.rad2deg(ph):.0f} ===")
        print("  hybrid pmm_jones_2d n_orders ladder (its Fourier floor); the")
        print("  hybrid warns on lossless-energy closure at low truncations "
              "here, so\n  stabilize=True is on and the closure is printed:")
        hy = {}
        for no in (7, 9, 11, 13):
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, Rm, Tm, J = pmm_jones_2d(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                            theta=th, phi=ph, degree=9,
                                            n_orders=no, stabilize=True)
            hy[no] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
            print(f"    n_orders={no:2d}  Rtot = {hy[no][0]}  "
                  f"Ttot = {hy[no][1]}  |R+T-1| = "
                  f"{np.max(np.abs(hy[no][0]+hy[no][1]-1)):.2e}"
                  f"  [{time.perf_counter()-t0:.1f} s]")
        top = 13
        floor = max(float(np.max(np.abs(hy[top][0] - hy[11][0]))),
                    float(np.max(np.abs(hy[top][1] - hy[11][1]))))
        floorJ = float(np.max(np.abs(hy[top][2] - hy[11][2])))
        print(f"    hybrid 11->13 movement: R/T {floor:.2e}   Jones "
              f"{floorJ:.2e}   <- the oracle's own floor")
        # SECOND cross-engine oracle: rcwa_jones_2d (Fourier, full 3x3)
        rc = {}
        for no in (7, 11):
            fine = upsample(ec, int(np.ceil((4 * no + 1) / ec.shape[0])))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, Rm, Tm, J = rcwa_jones_2d(PX, PY, fine, NSUB, NSUP, DEPTH,
                                              WL, theta=th, phi=ph,
                                              n_orders_x=no, n_orders_y=no)
            rc[no] = (Rm.sum(axis=1), Tm.sum(axis=1), J)
            print(f"    rcwa_jones_2d n_orders={no:2d}  Rtot = {rc[no][0]}  "
                  f"|R+T-1| = {np.max(np.abs(rc[no][0]+rc[no][1]-1)):.2e}")
        cross = max(float(np.max(np.abs(rc[11][0] - hy[top][0]))),
                    float(np.max(np.abs(rc[11][1] - hy[top][1]))))
        print(f"    hybrid(13) vs rcwa(11): R/T spread {cross:.2e}, Jones "
              f"{np.max(np.abs(rc[11][2]-hy[top][2])):.2e}  <- the ORACLE "
              f"SPREAD (the derived bar)")
        R[f"hybrid_floor_th{int(np.rad2deg(th))}"] = dict(
            RT=floor, J=floorJ, cross=cross,
            crossJ=float(np.max(np.abs(rc[11][2] - hy[top][2]))))
        print("   cand  M   dim    t[s]   |dRtot| vs hybrid  |dTtot|   "
              "|dJones|    |R+T-1|")
        for cand, Ms in (("a", (5, 6, 7, 8)), ("d", (5, 6, 7))):
            for M in Ms:
                t0 = time.perf_counter()
                try:
                    o2, R2, T2, J2 = pc.solve_slab(PX, PY, ec, NSUB, NSUP,
                                                   DEPTH, WL, M=M, theta=th,
                                                   phi=ph, candidate=cand)
                except Exception as exc:                  # noqa: BLE001
                    print(f"    ({cand}) {M:2d}  FAILED {type(exc).__name__}: "
                          f"{exc}")
                    continue
                dt = time.perf_counter() - t0
                Rt, Tt = totals(o2, R2, T2)
                dR = float(np.max(np.abs(Rt - hy[top][0])))
                dT = float(np.max(np.abs(Tt - hy[top][1])))
                dJ = float(np.max(np.abs(J2 - hy[top][2])))
                q = 2 * (M - 1)
                print(f"    ({cand}) {M:2d} {2*q*q:5d} {dt:6.1f}   {dR:.2e}"
                      f"          {dT:.2e}  {dJ:.2e}  "
                      f"{np.max(np.abs(Rt+Tt-1)):.2e}")
                R[f"th{int(np.rad2deg(th))}|{cand}|M{M}"] = dict(
                    dR=dR, dT=dT, dJ=dJ, t=dt,
                    closure=float(np.max(np.abs(Rt + Tt - 1))))
        # no-floor property, two-sided
        print("   NO-FLOOR (staggered, candidate (a), M=7): far-field "
              "n_orders 4 -> 9")
        vals = {}
        for no in (4, 9):
            _o, Rm, Tm, J = pc.solve_slab(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                          M=7, n_orders=no, theta=th, phi=ph,
                                          candidate="a")
            vals[no] = (Rm.sum(axis=1), Tm.sum(axis=1))
        mv = float(np.max(np.abs(np.array(vals[9]) - np.array(vals[4]))))
        print(f"     staggered R/T movement = {mv:.2e}   (hybrid's = "
              f"{floor:.2e})")
        R[f"nofloor_th{int(np.rad2deg(th))}"] = dict(staggered=mv,
                                                     hybrid=floor)
    with open(os.path.join(OUT, "m5_pillar_2d.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m5_pillar_2d.json")


if __name__ == "__main__":
    main()
