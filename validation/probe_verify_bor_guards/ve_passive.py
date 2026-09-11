"""TASK E -- is the band's 'must not absorb' side REACHABLE by a PASSIVE
medium?  For eps with Im >= 0 the strip operator's anti-Hermitian part is
positive semidefinite, so every eigenvalue has Im(lam) >= 0 and, for a real
qz2, Im(ky^2) >= 0 -- np.sqrt then already returns Im >= 0 and NO branch
decision is taken.  Measured over 9 decades of passive loss."""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import _vh, ve_fix  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
a = ap.parse_args()
import lumenairy
print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_2d
tot_neg = tot_conj = tot = 0
for Nx in (48, 96, 128):
    for kx0 in (0.0, 0.37):
        for g in (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            for qz2 in (0.0, 26055.8):
                eps = ve_fix.eps_split(Nx, imag=g)
                lam = np.asarray(eme_2d.strip_x_modes(eps, 1.0, Nx,
                                                      20 * np.pi, kx0)[0],
                                 dtype=complex)
                lam = lam[np.lexsort((lam.imag, lam.real))]
                z0 = np.sqrt(lam - qz2 + 0j)
                got = np.asarray(eme_2d._ky_forward(lam, qz2))
                neg = int(np.sum(np.isclose(got, -z0, rtol=1e-13, atol=0)
                                 & ~(got == z0)))
                con = int(np.sum(np.isclose(got, np.conj(z0), rtol=1e-13,
                                            atol=0) & ~(got == z0)))
                tot_neg += neg; tot_conj += con; tot += lam.size
                if neg or con:
                    print("  DECISION FIRED  Nx=%3d kx0=%g Im(eps)=%-8g "
                          "qz2=%-9g neg=%d conj=%d minIm(lam)=%.3e"
                          % (Nx, kx0, g, qz2, neg, con, float(np.min(lam.imag))))
print("  PASSIVE SWEEP: %d modes over 96 configurations -- %d negated, "
      "%d conjugated" % (tot, tot_neg, tot_conj))
