"""R7 -- WHICH direction is the layer<->substrate mode match singular in?

R3 measured ``cond(a + b) = 1.07e15`` at the layer->substrate interface while
``cond(M) = 1.6e2`` and ``cond(W) = 5.6e3`` at the layer eigenproblem, so the
singularity is CREATED at the mode match, not inherited from the layer's
eigendecomposition.  ``a + b`` is singular exactly when some combination of the
layer's FORWARD modes reproduces a BACKWARD region mode (tangential E and H
both continuous with the backward sign), i.e. when the interface cannot tell
the two apart.

This probe takes the smallest right singular vector of ``a + b``, and reports
(i) how concentrated it is over the modal index, (ii) which layer modes carry
it, (iii) those modes' eigenvalues ``lam``, and (iv) the substrate's ``lam`` for
the SAME order index -- so the answer is a statement about which mode pair the
interface confuses, not a norm.  Run at detune 0 (the coincidence) and at a
detune that removes it.

Usage: ``python r7_nullvector.py [detune]``
"""
from __future__ import annotations

import sys
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
import lumenairy.elements.rcwa.twod as TW

REC = {"eig": [], "iface": []}


def _cond(A):
    A = np.asarray(A)
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("inf"), None
    return (float(s[0] / s[-1]) if s[-1] > 0 else float("inf")), s


def install():
    orig_eig_for = C._eig_for
    orig_iface = C._interface_smatrix

    def eig_for(xp):
        base = orig_eig_for(xp)

        def wrapped(A):
            w, v = base(A)
            REC["eig"].append(dict(lam2=np.asarray(w).copy()))
            return w, v
        return wrapped

    def iface(Wa, Va, Wb, Vb):
        Wa_, Va_ = np.asarray(Wa), np.asarray(Va)
        Wb_, Vb_ = np.asarray(Wb), np.asarray(Vb)
        a = np.linalg.solve(Wb_, Wa_)
        b = np.linalg.solve(Vb_, Va_)
        apb = a + b
        u, s, vh = np.linalg.svd(apb)
        c = vh[-1].conj()                    # smallest right singular vector
        w = np.abs(c) ** 2
        order = np.argsort(w)[::-1]
        REC["iface"].append(dict(
            n=int(Wa_.shape[0]), cond=float(s[0] / s[-1]) if s[-1] else np.inf,
            smin=float(s[-1]), smax=float(s[0]),
            n_sv_below_1e_10_rel=int(np.sum(s < 1e-10 * s[0])),
            participation=float(1.0 / np.sum(w ** 2)),  # inverse participation
            top_idx=[int(i) for i in order[:8]],
            top_wt=[float(w[i]) for i in order[:8]],
            # what the mismatch looks like along that direction
            resid_E=float(np.linalg.norm(a @ c)),
            resid_H=float(np.linalg.norm(b @ c)),
            resid_sum=float(np.linalg.norm(apb @ c))))
        return orig_iface(Wa, Va, Wb, Vb)

    C._eig_for = eig_for
    C._interface_smatrix = iface
    TW._interface_smatrix = iface


def main():
    detune = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    a = L.arm()
    install()
    tc = L.even_sector_cell()
    n_sub = 1.5 * (1.0 + detune)
    print("### build=%s tree=%s detune=%.1e" % (a["build"], a["tree"], detune),
          flush=True)
    out = dict(detune=detune)
    for tag, sym in (("full", False), ("even", True)):
        REC["eig"].clear()
        REC["iface"].clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TW.rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0,
                                   0.2e-6, L.WL_DEFAULT, n_orders_x=5,
                                   n_orders_y=5, symmetry=sym)
        R, T = np.asarray(res[1]), np.asarray(res[2])
        lam2 = REC["eig"][0]["lam2"] if REC["eig"] else np.array([])
        lam = C._sqrt_decay(lam2) if lam2.size else lam2
        recs = []
        for i, f in enumerate(REC["iface"]):
            r = {k: v for k, v in f.items()}
            if lam.size and max(f["top_idx"]) < lam.size:
                r["top_lam"] = [[float(lam[j].real), float(lam[j].imag)]
                                for j in f["top_idx"]]
            recs.append(r)
            print("  %-5s if%-2d n=%-4d cond=%.3e smin/smax=%.3e  "
                  "sv<1e-10rel=%d  participation=%.2f  top=%s"
                  % (tag, i, f["n"], f["cond"], f["smin"] / f["smax"],
                     f["n_sv_below_1e_10_rel"], f["participation"],
                     f["top_idx"][:5]))
            if "top_lam" in r:
                print("        top lam: %s" % ["%.4f%+.4fj" % (z[0], z[1])
                                               for z in r["top_lam"][:5]])
        out[tag] = dict(iface=recs,
                        defect=float(np.sum(R) + np.sum(T) - 2.0),
                        lam_absmin=float(np.min(np.abs(lam)))
                        if lam.size else None)
        print("  %-5s defect %+.4e" % (tag, out[tag]["defect"]), flush=True)
    L.dump("r7_nullvector", out,
           suffix="_d%s" % ("0" if detune == 0 else ("%.0e" % detune)))


if __name__ == "__main__":
    main()
