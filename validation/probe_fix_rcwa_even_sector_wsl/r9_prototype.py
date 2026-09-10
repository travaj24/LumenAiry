"""R9 -- prototype of the branch-cut pin, measured against the un-patched
library on the SAME interpreter.

The candidate ``_sqrt_decay`` widens the on-cut test from the exact
``Re(sqrt(lam^2)) == 0`` to "``lam^2`` lies on the negative real axis to within
the eigensolver's own backward error", and takes the CONJUGATE root there --
which is the outgoing one and keeps ``Re(lam) >= 0``.

Reported per fixture: the failing full-vs-even disagreement, the lossless
closure defect of each path, and the interface conditioning, BEFORE and AFTER.
"""
from __future__ import annotations

import sys
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
import lumenairy.elements.rcwa.twod as TW

CUT_REL = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
_ORIG = C._sqrt_decay


def patched(x):
    xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.max(xp.abs(x))
    on_cut = (x.real < 0) & (xp.abs(x.imag) <= CUT_REL * scale)
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)


IFACE = []


def install_iface():
    orig = C._interface_smatrix

    def iface(Wa, Va, Wb, Vb):
        a = np.linalg.solve(np.asarray(Wb), np.asarray(Wa))
        b = np.linalg.solve(np.asarray(Vb), np.asarray(Va))
        s = np.linalg.svd(np.asarray(a + b), compute_uv=False)
        IFACE.append(float(s[0] / s[-1]) if s[-1] > 0 else float("inf"))
        return orig(Wa, Va, Wb, Vb)
    C._interface_smatrix = iface
    TW._interface_smatrix = iface


def run(tc, n_sub, sym):
    IFACE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TW.rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0,
                               0.2e-6, L.WL_DEFAULT, n_orders_x=5,
                               n_orders_y=5, symmetry=sym)
    return res, list(IFACE)


def main():
    a = L.arm()
    install_iface()
    tc = L.even_sector_cell()
    out = dict(cut_rel=CUT_REL)
    print("### build=%s tree=%s CUT_REL=%.1e" % (a["build"], a["tree"],
                                                 CUT_REL), flush=True)
    for arm_name in ("before", "after"):
        C._sqrt_decay = _ORIG if arm_name == "before" else patched
        full, if_f = run(tc, 1.5, False)
        even, if_e = run(tc, 1.5, True)
        dR = float(np.max(np.abs(full[1] - even[1])))
        dT = float(np.max(np.abs(full[2] - even[2])))
        dJ = float(np.max(np.abs(full[3] - even[3])))
        def_f = float(np.sum(full[1]) + np.sum(full[2]) - 2.0)
        def_e = float(np.sum(even[1]) + np.sum(even[2]) - 2.0)
        out[arm_name] = dict(dR=dR, dT=dT, dJ=dJ, defect_full=def_f,
                             defect_even=def_e, cond_full=if_f, cond_even=if_e,
                             R_full=[float(v) for v in
                                     np.asarray(full[1]).ravel()[:6]],
                             passes=bool(dR < 1e-8 and dJ < 1e-8))
        print("  %-6s dR %.4e dT %.4e dJ %.4e | defect full %+.3e even %+.3e"
              % (arm_name, dR, dT, dJ, def_f, def_e))
        print("         cond(a+b) full %s  even %s"
              % (["%.3e" % c for c in if_f], ["%.3e" % c for c in if_e]),
              flush=True)
    C._sqrt_decay = _ORIG
    L.dump("r9_prototype", out)


if __name__ == "__main__":
    main()
