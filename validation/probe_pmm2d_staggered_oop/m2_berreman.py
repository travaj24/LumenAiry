"""M2 -- R / T / Jones of a UNIFORM out-of-plane slab vs Berreman 4x4.

The exact oracle for a uniform layer at any incidence.  Convergence ladder in
``M``, both incident polarizations, normal / oblique 25 deg / conical
(25 deg, phi 40 deg), for candidates (a) and (d).

Also reported (two-sided, never asserted):
  * the ORDER LEAKAGE -- a uniform cell must put all its power in order (0,0);
  * the energy closure ``|R + T - 1|`` for the HERMITIAN (lossless) tensor and
    the deficit for the lossy one.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m2_berreman.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 0.9
DEPTH = 0.35
NSUP, NSUB = 1.0, 1.5
CASES = (("normal", 0.0, 0.0),
         ("oblique25", np.deg2rad(25.0), 0.0),
         ("conical25/40", np.deg2rad(25.0), np.deg2rad(40.0)))


def _tr_oop(e):
    o = np.array(e, dtype=complex)
    o[0, 2], o[2, 0] = e[2, 0], e[0, 2]
    o[1, 2], o[2, 1] = e[2, 1], e[1, 2]
    return o


def run(eps33, cand, M, th, ph, n_orders=2):
    o, Rm, Tm, J = pc.solve_slab(PX, PY, pc.tile(eps33, 2, 2), NSUB, NSUP,
                                 DEPTH, WL, M=M, n_orders=n_orders,
                                 theta=th, phi=ph, candidate=cand)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    Rt, Tt = Rm.sum(axis=1), Tm.sum(axis=1)
    leak = float(max(np.max(np.abs(Rm.sum(axis=1) - Rm[:, p0])),
                     np.max(np.abs(Tm.sum(axis=1) - Tm[:, p0]))))
    return Rt, Tt, J, leak


def main():
    pc.banner("M2 -- uniform OOP slab vs Berreman 4x4")
    R = {}
    # A NON-RECIPROCAL out-of-plane tensor (e13 = conj(e31), Hermitian so still
    # lossless) is the ONLY case that discriminates the OOP TRANSPOSE: the
    # dispersion det(k k^T - |k|^2 I + eps) is invariant under eps -> eps^T, so
    # M1 cannot see a swapped e13/e31 placement -- only the FIELDS can.
    nonrec = pc.uniaxial(1.5, 1.7, 35.0).astype(complex)
    nonrec[0, 2] += 0.22j
    nonrec[2, 0] -= 0.22j
    tensors = {
        "tilt35 (lossless, Hermitian)": pc.uniaxial(1.5, 1.7, 35.0),
        "tilt35 azim40 (lossless)": pc.uniaxial(1.5, 1.7, 35.0, azim_deg=40.0),
        "tilt35 lossy 0.08": pc.uniaxial(1.5, 1.7, 35.0, loss=0.08),
        "tilt35 NON-RECIPROCAL e13=conj(e31)": nonrec,
    }
    for tname, eps33 in tensors.items():
        print(f"\n=== tensor: {tname} ===")
        print(f"    e13 = {eps33[0,2]:.5f}   e23 = {eps33[1,2]:.5f}   "
              f"e33 = {eps33[2,2]:.5f}")
        for cname, th, ph in CASES:
            bR, bT, bJr, _bJt = berreman_jones_1d(
                [(eps33, DEPTH * WL)], NSUB, NSUP, WL, angle=th, phi=ph)
            print(f"\n  {cname}:  Berreman R = [{bR[0]:.9f}, {bR[1]:.9f}]  "
                  f"T = [{bT[0]:.9f}, {bT[1]:.9f}]  "
                  f"R+T-1 = {bR[0]+bT[0]-1:+.2e} / {bR[1]+bT[1]-1:+.2e}")
            print("     cand  M   dim    t[s]   max|dR|   max|dT|   "
                  "max|dJones|  order-leak   |R+T-1|")
            for cand, Ms in (("a", (5, 6, 7, 8, 9, 10)), ("d", (5, 6, 7, 8))):
                for M in Ms:
                    t0 = time.perf_counter()
                    try:
                        Rt, Tt, J, leak = run(eps33, cand, M, th, ph)
                    except Exception as exc:            # noqa: BLE001
                        print(f"     ({cand})  {M:2d}   FAILED: "
                              f"{type(exc).__name__}: {exc}")
                        R[f"{tname}|{cname}|{cand}|M{M}"] = dict(
                            error=f"{type(exc).__name__}: {exc}")
                        continue
                    dt = time.perf_counter() - t0
                    dR = float(np.max(np.abs(Rt - bR)))
                    dT = float(np.max(np.abs(Tt - bT)))
                    dJ = float(np.max(np.abs(J - bJr)))
                    clo = float(np.max(np.abs(Rt + Tt - 1.0)))
                    q = 2 * (M - 1)
                    print(f"     ({cand})  {M:2d} {2*q*q:5d} {dt:6.2f}  "
                          f"{dR:.2e}  {dT:.2e}  {dJ:.2e}   {leak:.1e}   "
                          f"{clo:.2e}")
                    R[f"{tname}|{cname}|{cand}|M{M}"] = dict(
                        dR=dR, dT=dT, dJ=dJ, leak=leak, closure=clo, t=dt)
            # NEGATIVE CONTROLS at the best M: the same solve with the OOP
            # block mis-assembled.  These are the bars the reference numbers
            # above must beat by decades.
            if cname == "conical25/40":
                for vname, ev in (
                        ("drop_oop", eps33 * np.array([[1, 1, 0], [1, 1, 0],
                                                       [0, 0, 1]])),
                        ("negate_oop", eps33 * np.array([[1, 1, -1],
                                                         [1, 1, -1],
                                                         [-1, -1, 1]])),
                        ("transpose_oop", _tr_oop(eps33))):
                    Rt, Tt, J, _lk = run(ev, "a", 8, th, ph)
                    print(f"     NEGATIVE CONTROL {vname:14s} (a) M=8: "
                          f"max|dR| = {np.max(np.abs(Rt - bR)):.2e}  "
                          f"max|dJones| = {np.max(np.abs(J - bJr)):.2e}")
                    R[f"{tname}|{cname}|CONTROL_{vname}"] = dict(
                        dR=float(np.max(np.abs(Rt - bR))),
                        dJ=float(np.max(np.abs(J - bJr))))
    with open(os.path.join(OUT, "m2_berreman.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m2_berreman.json")


if __name__ == "__main__":
    main()
