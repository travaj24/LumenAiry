"""R8 -- the branch-cut census: how many layer modes get the BACKWARD root?

``_sqrt_decay`` maps the layer eigenvalue ``lam^2`` to the modal decay constant
``lam``.  A PROPAGATING mode has ``lam^2`` real NEGATIVE, i.e. it sits exactly
on numpy's principal-square-root branch cut, where the two roots are
``+i|kz|`` (outgoing, the one the S-matrix recursion requires) and ``-i|kz|``
(incoming).  Which one ``numpy.sqrt`` returns is decided by the SIGN OF THE
IMAGINARY PART of ``lam^2`` -- and for a value that comes out of ``eig`` that
sign is rounding noise, roughly ``eps_mach * ||M||``.

``_sqrt_decay`` tries to pin the outgoing root with

    on_cut = r.real == 0

which is an EXACT floating comparison.  For an ``eig`` output ``lam^2 = -s +
i*eta`` with ``|eta| ~ 1e-14``, the root is ``r = eta/(2|r|) + i*sign(eta)*
sqrt(s)`` -- its real part is ~1e-15, NOT zero -- so the pin never fires and
the mode keeps whichever root the last bit of ``eta`` chose.

This probe measures that population directly: for every layer mode it records
``lam^2``, ``lam``, the relative real part ``|Re(lam)| / |lam|`` (the quantity
any tolerant on-cut test must threshold), and whether ``Im(lam) < 0``.

Usage: ``python r8_branch_cut.py [detune]``
"""
from __future__ import annotations

import sys
import warnings

import _lib as L
import numpy as np

import lumenairy.elements.rcwa._core as C
import lumenairy.elements.rcwa.twod as TW

REC = []


def install():
    orig = C._eig_for

    def eig_for(xp):
        base = orig(xp)

        def wrapped(A):
            w, v = base(A)
            REC.append((np.asarray(A).copy(), np.asarray(w).copy()))
            return w, v
        return wrapped
    C._eig_for = eig_for


def report(M, lam2, tag):
    lam = C._sqrt_decay(lam2)
    rel = np.abs(lam.real) / np.maximum(np.abs(lam), 1e-300)
    neg = lam.imag < 0
    nrmM = float(np.max(np.abs(M)))
    print("  %-5s n=%-4d ||M||_max=%.3f" % (tag, lam2.size, nrmM))
    print("      modes with Im(lam) < 0 : %d / %d" % (int(neg.sum()),
                                                      lam2.size))
    if neg.any():
        idx = np.flatnonzero(neg)
        print("      their |Re(lam)|/|lam| : min %.3e  max %.3e"
              % (float(rel[idx].min()), float(rel[idx].max())))
        print("      their Im(lam^2)       : min %.3e  max %.3e"
              % (float(lam2[idx].imag.min()), float(lam2[idx].imag.max())))
        for j in idx[:8]:
            print("        idx %-4d lam^2 = %+.15e %+.3ej   lam = %+.3e %+.15ej"
                  % (j, lam2[j].real, lam2[j].imag, lam[j].real, lam[j].imag))
    # the two-sided separation any tolerant bar needs
    oncut = rel < 1e-6
    print("      |Re(lam)|/|lam| : %d modes < 1e-6 (max %.3e), "
          "%d modes >= 1e-6 (min %.3e)"
          % (int(oncut.sum()), float(rel[oncut].max()) if oncut.any() else -1,
             int((~oncut).sum()),
             float(rel[~oncut].min()) if (~oncut).any() else -1))
    return dict(n=int(lam2.size), n_neg_im=int(neg.sum()),
                normM=nrmM,
                rel_oncut_max=float(rel[rel < 1e-6].max())
                if (rel < 1e-6).any() else None,
                rel_offcut_min=float(rel[rel >= 1e-6].min())
                if (rel >= 1e-6).any() else None,
                neg_idx=[int(i) for i in np.flatnonzero(neg)[:16]],
                neg_lam=[[float(lam[i].real), float(lam[i].imag)]
                         for i in np.flatnonzero(neg)[:16]],
                neg_im_lam2=[float(lam2[i].imag)
                             for i in np.flatnonzero(neg)[:16]])


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
        REC.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TW.rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, n_sub, 1.0,
                                   0.2e-6, L.WL_DEFAULT, n_orders_x=5,
                                   n_orders_y=5, symmetry=sym)
        M, lam2 = REC[0]
        out[tag] = report(M, lam2, tag)
        R, T = np.asarray(res[1]), np.asarray(res[2])
        out[tag]["defect"] = float(np.sum(R) + np.sum(T) - 2.0)
        print("      defect %+.4e" % out[tag]["defect"], flush=True)
    # the substrate's own lam, for contrast: built from EXACT arithmetic
    kz = C._sqrt_forward(np.array([2.25 - 0.0, 2.25 - 1.44], dtype=complex))
    lam_reg = C._sqrt_decay(-np.concatenate([kz, kz]) ** 2)
    print("  region lam (exact arithmetic path): %s"
          % ["%+.3e%+.4fj" % (z.real, z.imag) for z in lam_reg])
    out["region_lam"] = [[float(z.real), float(z.imag)] for z in lam_reg]
    L.dump("r8_branch_cut", out,
           suffix="_d%s" % ("0" if detune == 0 else ("%.0e" % detune)))


if __name__ == "__main__":
    main()
