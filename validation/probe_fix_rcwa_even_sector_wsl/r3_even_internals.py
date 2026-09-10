"""R3 -- WHERE in the solve does the coincidence become an ill-conditioned
operator?

Terms.  The RCWA layer solve builds the layer's modal basis by
eigendecomposing ``M = P Q`` (``W`` = electric eigenvectors, ``V = Q W /
lam`` = magnetic ones), and joins two media across an interface by the
mode-match ``a = Wb^-1 Wa``, ``b = Vb^-1 Va``, whose ``a + b`` is inverted
EXPLICITLY (it is ``S12 = 2 (a+b)^-1`` by definition).  Three condition
numbers therefore decide the accuracy of the whole cascade: ``cond(M)``,
``cond(W)`` (the eigenvector matrix the mode-match inverts through), and
``cond(a + b)`` at each interface.

This probe wraps the ``_core`` globals the solve resolves at call time and
reports all three, for the FULL ``2N`` path and for the EVEN ``N+1`` fold, at a
chosen relative detune of the substrate index away from the fixture's exact
``eps_sub == eps_layer_background == 2.25`` coincidence.

Usage: ``python r3_even_internals.py [detune]`` (default 0.0).
"""
from __future__ import annotations

import sys
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
from lumenairy.elements.rcwa import rcwa_jones_2d

REC = {"eig": [], "iface": []}


def _cond(A):
    A = np.asarray(A)
    if not np.all(np.isfinite(A)):
        return float("inf")
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def install():
    orig_eig_for = C._eig_for
    orig_iface = C._interface_smatrix

    def eig_for(xp):
        base = orig_eig_for(xp)

        def wrapped(A):
            w, v = base(A)
            sw = np.sort_complex(np.asarray(w))
            gaps = np.abs(np.diff(sw)) if sw.size > 1 else np.array([np.nan])
            REC["eig"].append(dict(
                n=int(np.asarray(A).shape[0]), cond_M=_cond(A),
                cond_W=_cond(v),
                eval_absmax=float(np.max(np.abs(w))),
                eval_absmin=float(np.min(np.abs(w))),
                min_gap=float(np.min(gaps)),
                n_gaps_below_1e_8=int(np.sum(gaps < 1e-8)),
                n_gaps_below_1e_12=int(np.sum(gaps < 1e-12))))
            return w, v
        return wrapped

    def iface(Wa, Va, Wb, Vb):
        a = np.linalg.solve(np.asarray(Wb), np.asarray(Wa))
        b = np.linalg.solve(np.asarray(Vb), np.asarray(Va))
        REC["iface"].append(dict(
            n=int(np.asarray(Wa).shape[0]),
            cond_Wb=_cond(Wb), cond_Vb=_cond(Vb),
            cond_apb=_cond(a + b), cond_amb=_cond(a - b),
            amax_a=float(np.max(np.abs(a))),
            amax_b=float(np.max(np.abs(b)))))
        return orig_iface(Wa, Va, Wb, Vb)

    C._eig_for = eig_for
    C._interface_smatrix = iface


def main():
    detune = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    a = L.arm()
    install()
    tc = L.even_sector_cell()
    n_sub = 1.5 * (1.0 + detune)
    out = dict(detune=detune, n_sub=n_sub)
    print("### build=%s tree=%s detune=%.1e n_sub=%.17g"
          % (a["build"], a["tree"], detune, n_sub), flush=True)
    for tag, sym in (("full", False), ("even", True)):
        REC["eig"].clear()
        REC["iface"].clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0,
                                0.2e-6, L.WL_DEFAULT, n_orders_x=5,
                                n_orders_y=5, symmetry=sym)
        R, T = np.asarray(res[1]), np.asarray(res[2])
        out[tag] = dict(eig=list(REC["eig"]), iface=list(REC["iface"]),
                        defect=float(np.sum(R) + np.sum(T) - 2.0),
                        R_max=float(np.max(np.abs(R))))
        print("  %-5s defect %+.4e" % (tag, out[tag]["defect"]))
        for e in REC["eig"]:
            print("    eig  n=%-4d cond(M)=%.3e cond(W)=%.3e  min|lam2 gap|="
                  "%.3e  gaps<1e-8: %d  <1e-12: %d"
                  % (e["n"], e["cond_M"], e["cond_W"], e["min_gap"],
                     e["n_gaps_below_1e_8"], e["n_gaps_below_1e_12"]))
        for i, f in enumerate(REC["iface"]):
            print("    if%-2d n=%-4d cond(Wb)=%.3e cond(Vb)=%.3e "
                  "cond(a+b)=%.3e cond(a-b)=%.3e"
                  % (i, f["n"], f["cond_Wb"], f["cond_Vb"], f["cond_apb"],
                     f["cond_amb"]))
        sys.stdout.flush()
    L.dump("r3_even_internals", out,
           suffix="_d%s" % ("0" if detune == 0 else ("%.0e" % detune)))


if __name__ == "__main__":
    main()
