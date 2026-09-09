"""GATE 1 -- the rotation-gauge sign of the INTEGRATED out-of-plane path.

The shipped staggered basis carries ``exp(-i alpha0 x)`` while its far-field
kernel and its ``eps_cell`` indexing run the other way (GATE 0f / 0e); the
composition is a 180-degree rotation about z of the whole solve.  In-plane
tensor components are invariant under it -- which is why the isotropic and the
Stage-A in-plane paths never had to know -- and out-of-plane components change
sign, so the assembly must apply that sign once
(``twod_staggered._OOP_ROT_SIGN``).

This probe DECIDES the value by measurement, two arms, same build: a UNIFORM
out-of-plane slab through ``pmm_jones_2d_staggered`` against the exact
``berreman_jones_1d``, at normal / oblique / conical, for a lossless, a lossy
and a NON-RECIPROCAL tensor, with ``_OOP_ROT_SIGN`` at -1 and at +1.  The
director azimuth is deliberately DIFFERENT from the incidence azimuth (S6: at
equal azimuths negating the out-of-plane block is an exact symmetry and the
measurement sees nothing).

Run:
  cd /c/tmp/lum_aniso_oopint && PYTHONPATH=/c/tmp/lum_aniso_oopint \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/g1_rot_sign_berreman.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 0.9
DEPTH = 0.35
NSUB, NSUP = 1.5, 1.0

#: director azimuth 25 deg; the conical incidence azimuth below is 40 deg, so
#: the negate-the-out-of-plane-block degeneracy of S6 is NOT hit.
_T_LOSSLESS = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
_T_LOSSY = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0, loss=0.08)
_T_NONREC = np.array(_T_LOSSLESS, dtype=complex)
_T_NONREC[0, 2] = _T_LOSSLESS[0, 2] + 0.22j       # Hermitian, NOT symmetric
_T_NONREC[2, 0] = np.conj(_T_NONREC[0, 2])

CASES = (("lossless tilt35 azim25", _T_LOSSLESS),
         ("lossy tilt35 azim25", _T_LOSSY),
         ("NON-RECIPROCAL tilt35 azim25", _T_NONREC))
MOUNTS = (("normal", 0.0, 0.0),
          ("oblique 25", np.deg2rad(25.0), 0.0),
          ("conical 25/40", np.deg2rad(25.0), np.deg2rad(40.0)))


def main():
    pc.banner("GATE 1 -- _OOP_ROT_SIGN arbitration against berreman_jones_1d")
    res = {}
    for name, t33 in CASES:
        cell = np.broadcast_to(t33, (2, 2, 3, 3)).copy()
        for mount, th, ph in MOUNTS:
            Rb, Tb, Jrb, _Jtb = berreman_jones_1d([(t33, DEPTH)], NSUB, NSUP,
                                                  WL, angle=th, phi=ph)
            print(f"\n=== {name}  |  {mount} ===")
            print(f"    berreman R = {Rb}   |R+T-1| = "
                  f"{np.max(np.abs(Rb + Tb - 1)):.2e}")
            for sign in (-1.0, +1.0):
                TS._OOP_ROT_SIGN = sign
                for M in (6, 8):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        o, R, T, J = TS.pmm_jones_2d_staggered(
                            PX, PY, cell, NSUB, NSUP, DEPTH, WL, degree=M,
                            n_orders=3, theta=th, phi=ph)
                    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
                    dR = float(np.max(np.abs(R.sum(axis=1) - Rb)))
                    dT = float(np.max(np.abs(T.sum(axis=1) - Tb)))
                    dJ = float(np.max(np.abs(J - Jrb)))
                    leak = float(np.max(np.abs(
                        np.delete(R, p0, axis=1)))
                        + np.max(np.abs(np.delete(T, p0, axis=1))))
                    print(f"    rot={sign:+.0f} M={M}  dR = {dR:.3e}  "
                          f"dT = {dT:.3e}  dJones = {dJ:.3e}   "
                          f"order leak = {leak:.1e}")
                    res[f"{name}|{mount}|rot{sign:+.0f}|M{M}"] = dict(
                        dR=dR, dT=dT, dJ=dJ, leak=leak)
    TS._OOP_ROT_SIGN = -1.0
    with open(os.path.join(OUT, "g1_rot_sign_berreman.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote results/g1_rot_sign_berreman.json")


if __name__ == "__main__":
    main()
