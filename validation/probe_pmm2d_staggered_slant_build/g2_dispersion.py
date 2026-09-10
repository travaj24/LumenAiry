"""B4 -- SHEARED-FRAME DISPERSION against the EXACT quartic roots.

For a UNIFORM cell the slanted generator's eigenvalues must be

    q(m, n) = kz_root(eps_lab; u, v)  +  t_x u + t_y v      (k0 units)

with ``kz_root`` the four EXACT roots of ``det(k k^T - |k|^2 I + eps) = 0`` in
exact polynomial arithmetic and ``t`` the INTERNAL shear (= -slant).  The shear
translates each harmonic's four roots RIGIDLY, which no tensor error can mimic:
this is the sign / factor gate.

Four GAUGE / SHIFT arms are scanned (``a+/a-`` = transverse gauge
``+alpha``/``-alpha``, ``s+/s-`` = shift ``+t.alpha``/``-t.alpha``) and TWO
ABLATIONS are run through the shipped assembly:

  * SIX BLOCKS REMOVED -- congruence kept, ``_slant_rot`` zeroed before the
    re-assembly;
  * CONGRUENCE REMOVED -- the lab tensor assembled vertically, ``_slant_rot``
    set to the rotated shear before the re-assembly.

Both halves of the formulation must be load-bearing, and each is measured.
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
from _lib import _ROOT, uniaxial, write

from lumenairy.elements.pmm.twod_staggered import (
    _OOP_ROT_SIGN,
    Granet2DTransverseE,
)

sys.path.insert(0, os.path.join(_ROOT, "validation",
                                "probe_pmm2d_staggered_oop"))
from probe_common import exact_kz_roots  # noqa: E402

PX = PY = 0.9
K0 = 2.0 * np.pi
M = 8
NG = 2

TIL = uniaxial(1.5, 1.7, np.deg2rad(35.0), np.deg2rad(25.0))
NREC = TIL.copy()
NREC[0, 2] = TIL[0, 2] + 0.30
NREC[2, 0] = TIL[2, 0] - 0.30
ISO = 2.25 * np.eye(3, dtype=complex)
T20 = float(np.tan(np.deg2rad(20.0)))
T35 = float(np.tan(np.deg2rad(35.0)))

res = {}
t00 = time.time()


def spectrum(eps33, slant, mode="full"):
    cell = np.zeros((NG, NG, 3, 3), dtype=complex)
    cell[:, :] = eps33
    kx0, ky0 = 0.25, 0.18
    sol = Granet2DTransverseE(PX, PY, NG, NG, M, cell, alpha0x=kx0 * K0,
                              alpha0y=ky0 * K0, k0=K0,
                              slant=None if mode == "nocong" else slant)
    if mode == "noblocks":
        sol._slant_rot = (0.0, 0.0)
        sol._assemble_oop()
    elif mode == "nocong":
        # keep the LAB tensor (vertical assembly, rot applied inside) but put
        # the six slant blocks in: the congruence is the piece removed.
        sol._slant_rot = (-_OOP_ROT_SIGN * slant[0],
                          -_OOP_ROT_SIGN * slant[1])
        sol._assemble_oop()
    # Bgen is a block Gram (HPD) -> Cholesky-whiten to a standard eig, which
    # is what _region_modes_oop does anyway and is ~10x a QZ here.
    Lc = np.linalg.cholesky(sol.Bgen)
    Ah = sla.solve_triangular(Lc, sol.Agen, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    return np.linalg.eigvals(Ah), kx0, ky0


def worst_root_gap(qv, eps33, kx0, ky0, slant, agn, sgn):
    """``slant`` is the PUBLIC vector; the INTERNAL shear of x = u + t w is
    ``t = -slant``, and the physical arm is ``a+ s+`` in THAT variable."""
    u, v = agn * kx0, agn * ky0
    roots = exact_kz_roots(np.asarray(eps33, dtype=complex), u, v)
    shift = sgn * (-slant[0] * u - slant[1] * v)
    worst = 0.0
    for r in roots:
        worst = max(worst, float(np.min(np.abs(qv - (r + shift)))))
    return worst, roots, shift


CASES = [("uniaxial", TIL, ("none", (0.0, 0.0))),
         ("uniaxial", TIL, ("x20", (T20, 0.0))),
         ("uniaxial", TIL, ("diag35", (T35 / np.sqrt(2), T35 / np.sqrt(2)))),
         ("nonrecip", NREC, ("x20", (T20, 0.0))),
         ("nonrecip", NREC, ("diag35", (T35 / np.sqrt(2), T35 / np.sqrt(2)))),
         ("iso", ISO, ("diag35", (T35 / np.sqrt(2), T35 / np.sqrt(2))))]

print("B4a  GAUGE / SHIFT ARMS (conical Bloch shift, M=%d, (%d,%d) grid)"
      % (M, NG, NG))
arms = []
for tn, tv, (sn, sv) in CASES:
    qv, kx0, ky0 = spectrum(tv, sv)
    row = {"tensor": tn, "slant": sn}
    for agn, alab in ((+1.0, "a+"), (-1.0, "a-")):
        for sgn, slab in ((+1.0, "s+"), (-1.0, "s-")):
            w, _r, _s = worst_root_gap(qv, tv, kx0, ky0, sv, agn, sgn)
            row[alab + slab] = w
    arms.append(row)
    print("  %-9s %-7s  a+s+ %.2e  a+s- %.2e  a-s+ %.2e  a-s- %.2e"
          % (tn, sn, row["a+s+"], row["a+s-"], row["a-s+"], row["a-s-"]))
res["B4a_arms"] = arms
res["B4a_worst_physical"] = max(r["a+s+"] for r in arms)
res["B4a_best_wrong"] = min(min(r["a+s-"], r["a-s+"]) for r in arms
                            if r["slant"] != "none")

print("")
print("B4b  ABLATIONS (both halves of the formulation)")
abl = []
for tn, tv, (sn, sv) in CASES:
    if sn == "none":
        continue
    row = {"tensor": tn, "slant": sn}
    for mode, lab in (("full", "full"), ("noblocks", "noblocks"),
                      ("nocong", "nocong")):
        qv, kx0, ky0 = spectrum(tv, sv, mode)
        w, _r, _s = worst_root_gap(qv, tv, kx0, ky0, sv, +1.0, +1.0)
        row[lab] = w
    abl.append(row)
    print("  %-9s %-7s  full %.2e | six blocks REMOVED %.2e | congruence "
          "REMOVED %.2e" % (tn, sn, row["full"], row["noblocks"],
                            row["nocong"]))
res["B4b_ablations"] = abl
res["B4b_worst_full"] = max(r["full"] for r in abl)
res["B4b_best_noblocks"] = min(r["noblocks"] for r in abl)
res["B4b_best_nocong"] = min(r["nocong"] for r in abl)

print("")
print("B4c  SUM-OF-ROOTS DISCRIMINATOR (the fundamental's four roots)")
sums = []
for tn, tv, (sn, sv) in CASES:
    if tn != "uniaxial":
        continue
    qv, kx0, ky0 = spectrum(tv, sv)
    w, roots, shift = worst_root_gap(qv, tv, kx0, ky0, sv, +1.0, +1.0)
    gen = 0.0
    for r in roots:
        gen += complex(qv[int(np.argmin(np.abs(qv - (r + shift))))])
    sums.append({"tensor": tn, "slant": sn,
                 "gen": [gen.real, gen.imag],
                 "exact": [float(np.real(np.sum(roots) + 4 * shift)),
                           float(np.imag(np.sum(roots) + 4 * shift))]})
    print("  %-9s %-7s  generator %+.6f   exact %+.6f"
          % (tn, sn, gen.real, sums[-1]["exact"][0]))
res["B4c_sums"] = sums

print("")
print("B4d  M-LADDER (uniaxial, diag35)")
lad = []
sv = (T35 / np.sqrt(2), T35 / np.sqrt(2))
for Mi in (4, 5, 6, 7, 8):
    Msave = M
    M = Mi
    qv, kx0, ky0 = spectrum(TIL, sv)
    M = Msave
    w, _r, _s = worst_root_gap(qv, TIL, kx0, ky0, sv, +1.0, +1.0)
    lad.append({"M": Mi, "gap": w})
    print("  M=%d  %.2e" % (Mi, w))
res["B4d_ladder"] = lad

res["wall_s"] = time.time() - t00
write("g2_dispersion", res)
