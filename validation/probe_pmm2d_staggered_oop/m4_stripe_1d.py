"""M4 -- y-uniform OUT-OF-PLANE stripe grating vs the 1-D engines.

A stripe grating (eps varies in x only) solved on the 2-D staggered cell must
reproduce the 1-D engines PER ORDER and in the order-0 Jones:
  * ``pmm_jones_1d``  -- spectral-element, OOP metric generator, no Fourier floor
  * ``rcwa_jones_1d`` -- GAP7 full-3x3 tensor, many orders

Also measured: Y-MOMENTUM CONSERVATION -- every ``(m, n != 0)`` order of the
2-D solve must be zero (the cell is y-invariant, so nothing may scatter into
a y-order).  A lateral shift of the ridge multiplies order ``m`` by a phase,
so the comparison is on the per-order EFFICIENCIES and on the order-0 Jones
(both shift-invariant).

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m4_stripe_1d.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.pmm import pmm_jones_1d  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_1d  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
DEPTH = 0.4
NSUB, NSUP = 1.5, 1.0


def pick(o2, A2, morders):
    """Rows of a 2-D (2, Nfo) result at the (m, 0) orders."""
    out = np.zeros((2, len(morders)))
    for k, m in enumerate(morders):
        j = np.where((o2[:, 0] == m) & (o2[:, 1] == 0))[0]
        out[:, k] = A2[:, j[0]] if j.size else 0.0
    return out


def yleak(o2, A2):
    j = o2[:, 1] != 0
    return float(np.max(np.abs(A2[:, j]))) if j.any() else 0.0


def main():
    pc.banner("M4 -- y-uniform OOP stripe grating vs the 1-D engines")
    R = {}
    cases = {
        "ridge=tilt35 / groove=air": (pc.uniaxial(1.5, 1.7, 35.0),
                                      np.eye(3, dtype=complex)),
        "ridge=tilt35 lossy / groove=2.25": (
            pc.uniaxial(1.5, 1.7, 35.0, loss=0.08),
            2.25 * np.eye(3, dtype=complex)),
    }
    for cname, (er, eg) in cases.items():
        print(f"\n=== {cname} ===")
        for th in (0.0, np.deg2rad(25.0)):
            o_r, Rr, Tr, Jr = rcwa_jones_1d(PX, er, eg, NSUB, NSUP, DEPTH, 0.5,
                                            WL, angle=th, n_orders=61)
            o_p, Rp, Tp, Jp = pmm_jones_1d(PX, er, eg, NSUB, NSUP, DEPTH, 0.5,
                                           WL, angle=th, degree=18,
                                           far_field_orders=31,
                                           stabilize=False)
            ms = [-1, 0, 1]
            ir = [int(np.where(o_r == m)[0][0]) for m in ms]
            ip = [int(np.where(o_p == m)[0][0]) for m in ms]
            print(f"\n  theta = {np.rad2deg(th):.0f} deg")
            print(f"   rcwa(61) R(m=-1,0,1) row0 = "
                  f"{np.array2string(Rr[0, ir], precision=8)}")
            print(f"   pmm1d(18) R(m=-1,0,1) row0 = "
                  f"{np.array2string(Rp[0, ip], precision=8)}")
            print(f"   rcwa vs pmm1d: max|dR| = "
                  f"{np.max(np.abs(Rr[:, ir] - Rp[:, ip])):.2e}  max|dT| = "
                  f"{np.max(np.abs(Tr[:, ir] - Tp[:, ip])):.2e}  max|dJ| = "
                  f"{np.max(np.abs(Jr - Jp)):.2e}")
            print("    cand  M   dim   max|dR| vs rcwa  max|dT|   max|dJ|   "
                  " y-leak(R)  y-leak(T)   |R+T-1|")
            for cand, Ms in (("a", (5, 6, 7, 8)), ("d", (5, 6, 7))):
                for M in Ms:
                    ec = np.zeros((2, 2, 3, 3), dtype=complex)
                    ec[0, :] = er
                    ec[1, :] = eg
                    try:
                        o2, R2, T2, J2 = pc.solve_slab(
                            PX, PY, ec, NSUB, NSUP, DEPTH, WL, M=M,
                            theta=th, phi=0.0, candidate=cand)
                    except Exception as exc:              # noqa: BLE001
                        print(f"     ({cand}) {M:2d}  FAILED "
                              f"{type(exc).__name__}: {exc}")
                        continue
                    dR = float(np.max(np.abs(pick(o2, R2, ms) - Rr[:, ir])))
                    dT = float(np.max(np.abs(pick(o2, T2, ms) - Tr[:, ir])))
                    dJ = float(np.max(np.abs(J2 - Jr)))
                    q = 2 * (M - 1)
                    print(f"     ({cand}) {M:2d} {2*q*q:5d}   {dR:.2e}      "
                          f"{dT:.2e}  {dJ:.2e}   {yleak(o2, R2):.1e}   "
                          f"{yleak(o2, T2):.1e}   "
                          f"{np.max(np.abs(R2.sum(1)+T2.sum(1)-1)):.2e}")
                    R[f"{cname}|th{int(np.rad2deg(th))}|{cand}|M{M}"] = dict(
                        dR=dR, dT=dT, dJ=dJ, yleakR=yleak(o2, R2),
                        yleakT=yleak(o2, T2),
                        closure=float(np.max(np.abs(R2.sum(1) + T2.sum(1)
                                                    - 1))))
            R[f"{cname}|th{int(np.rad2deg(th))}|oracle_spread"] = dict(
                dR=float(np.max(np.abs(Rr[:, ir] - Rp[:, ip]))),
                dT=float(np.max(np.abs(Tr[:, ir] - Tp[:, ip]))),
                dJ=float(np.max(np.abs(Jr - Jp))))
    with open(os.path.join(OUT, "m4_stripe_1d.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m4_stripe_1d.json")


if __name__ == "__main__":
    main()
