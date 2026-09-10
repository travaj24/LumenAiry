"""V2b -- the PUBLIC ``slant`` against the 1-D SLANTED ORACLE, per order, both
signs, and the LOSSLESS TRAP.

A y-uniform binary stripe in the 2-D pure staggered engine is the SAME physical
grating ``pmm_efficiency_1d_slanted`` solves (a different basis, a different
generator, a different far field), so its per-order R and T are an independent
arbiter of the public slant sign -- and per order it is a sharp one, because a
slanted grating is not x-mirror symmetric, so ``R_{+1} != R_{-1}``.

Measured here, both arms every time:

  * ``slant = +tan(slant_angle_1D)`` and ``slant = -tan(...)``, at slants
    10 / 20 / 35 degrees, normal and oblique 25 degrees, TE and TM;
  * the VERTICAL control (slant 0 in both engines) -- the floor this
    comparison can possibly reach, so an accuracy claim is never confused with
    a sign claim;
  * the ORACLE's OWN drift between two degrees, which bounds it from below;
  * the ENERGY CLOSURE of BOTH sign arms, which is the lossless trap: if the
    two close identically, energy carries zero information about the sign.
"""
import numpy as np
from _lib import arm, dump  # noqa: I001

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_1d_slanted,
)
from lumenairy.elements.pmm.twod_staggered import pmm_jones_2d_staggered

WL = 0.68e-6
PX = PY = 0.75 * WL
DEP = 0.30 * WL
NRIDGE, NGROOVE = 2.0, 1.0
DUTY = 0.5
NSUP, NSUB = 1.0, 1.5
M2D = 7
NORD = 3
DEG1D = 24
DEG1D_LO = 20

SLANTS = (0.0, 10.0, 20.0, 35.0)
MOUNTS = {"normal": 0.0, "oblique25": np.deg2rad(25.0)}
CELL = np.array([[NRIDGE ** 2, NRIDGE ** 2], [NGROOVE ** 2, NGROOVE ** 2]],
                dtype=complex)


def two_d(slant, theta):
    o, R, T, J = pmm_jones_2d_staggered(
        PX, PY, CELL, NSUB, NSUP, DEP, WL, degree=M2D, n_orders=NORD,
        theta=theta, phi=0.0, slant=slant)
    o = np.asarray(o)
    sel = o[:, 1] == 0                       # the n = 0 (y) orders
    m = o[sel, 0]
    idx = np.argsort(m)
    return (m[idx], R[:, sel][:, idx], T[:, sel][:, idx],
            float(np.max(R.sum(1) + T.sum(1))))


def one_d(slant_angle, theta, pol, degree=DEG1D):
    o, R, T = pmm_efficiency_1d_slanted(
        PX, NRIDGE, NGROOVE, NSUB, NSUP, DEP, DUTY, WL, float(slant_angle),
        angle=theta, polarization=pol, degree=degree, n_orders=2 * NORD + 1,
        stabilize=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def one_d_vertical(theta, pol, degree=DEG1D):
    o, R, T = pmm_efficiency_1d(
        PX, NRIDGE, NGROOVE, NSUB, NSUP, DEP, DUTY, WL, angle=theta,
        polarization=pol, degree=degree, n_orders=2 * NORD + 1,
        stabilize=False)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _cmp(m2, R2, T2, o1, R1, T1):
    """max per-order |dR|, |dT| over the orders BOTH engines retain."""
    common = [mm for mm in m2 if mm in set(o1.tolist())]
    i2 = [int(np.where(m2 == mm)[0][0]) for mm in common]
    i1 = [int(np.where(o1 == mm)[0][0]) for mm in common]
    dR = float(np.max(np.abs(R2[i2] - R1[i1])))
    dT = float(np.max(np.abs(T2[i2] - T1[i1])))
    return dict(dR=dR, dT=dT, worst=max(dR, dT), orders=[int(c)
                                                         for c in common])


ROW = {"tm": 0, "te": 1}       # 2-D row 0 = incident Ex = TM at phi = 0


def main():
    out = {"geometry": dict(PX=PX, DEP=DEP, WL=WL, duty=DUTY,
                            n_ridge=NRIDGE, n_groove=NGROOVE, M2D=M2D,
                            degree_1d=DEG1D, degree_1d_lo=DEG1D_LO,
                            n_orders=NORD),
           "rows": {}}
    for sdeg in SLANTS:
        t = float(np.tan(np.deg2rad(sdeg)))
        for mname, th in MOUNTS.items():
            key = f"slant{sdeg:g}/{mname}"
            res = {}
            arms = {"plus": (+t, 0.0), "minus": (-t, 0.0)} if sdeg else \
                {"zero": None}
            twod = {}
            for aname, sl in arms.items():
                m2, R2, T2, clo = two_d(sl, th)
                twod[aname] = (m2, R2, T2)
                res[f"closure_{aname}"] = clo
            for pol in ("te", "tm"):
                r = ROW[pol]
                if sdeg == 0.0:
                    o1, R1, T1 = one_d_vertical(th, pol)
                    o1l, R1l, T1l = one_d_vertical(th, pol, DEG1D_LO)
                    m2, R2, T2 = twod["zero"]
                    res[f"{pol}_vertical_control"] = _cmp(m2, R2[r], T2[r],
                                                          o1, R1, T1)
                    res[f"{pol}_oracle_drift"] = _cmp(o1, R1, T1, o1l,
                                                      R1l, T1l)
                    continue
                o1, R1, T1 = one_d(+np.deg2rad(sdeg), th, pol)
                o1l, R1l, T1l = one_d(+np.deg2rad(sdeg), th, pol, DEG1D_LO)
                res[f"{pol}_oracle_drift"] = _cmp(o1, R1, T1, o1l, R1l, T1l)
                for aname in ("plus", "minus"):
                    m2, R2, T2 = twod[aname]
                    res[f"{pol}_{aname}"] = _cmp(m2, R2[r], T2[r],
                                                 o1, R1, T1)
                # and the oracle driven with the OPPOSITE 1-D slant_angle,
                # which must mirror the relation
                o1n, R1n, T1n = one_d(-np.deg2rad(sdeg), th, pol)
                m2, R2, T2 = twod["plus"]
                res[f"{pol}_plus_vs_negative_oracle"] = _cmp(
                    m2, R2[r], T2[r], o1n, R1n, T1n)
            out["rows"][key] = res
            if sdeg:
                print(f"{key}: TE +{res['te_plus']['worst']:.3e} "
                      f"-{res['te_minus']['worst']:.3e} | "
                      f"TM +{res['tm_plus']['worst']:.3e} "
                      f"-{res['tm_minus']['worst']:.3e} | "
                      f"oracle drift TE {res['te_oracle_drift']['worst']:.2e}"
                      f" | closure + {res['closure_plus']:.9f} "
                      f"- {res['closure_minus']:.9f}")
            else:
                print(f"{key}: vertical control TE "
                      f"{res['te_vertical_control']['worst']:.3e} TM "
                      f"{res['tm_vertical_control']['worst']:.3e}")
    dump("v2b_oracle_sign", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
