"""TASK E -- MINIMAL REPRODUCER of the unit-system defect in
``_branch.cut_band``'s literal ``1.0`` floor.

ONE physical problem -- a 1 um period, 96 cells, lambda = 1550 nm, a layer with
a weak GAIN (``Im(eps) = -1e-6``, i.e. ``Im(n) ~ -2e-7``) -- expressed twice:
once with lengths in MICRONS and once with lengths in NANOMETRES.  Identical
physics, identical discretisation, only the length unit differs.

``ky`` carries units of 1/length (``lam`` is an eigenvalue of
``d2/dx2 + eps k0^2``; ``eme_diffraction.mode_match`` multiplies ``qz`` by
``depth``), so ``ky_nm = ky_um / 1000``.  ``cut_band``'s floor at a literal 1.0
is therefore a DIFFERENT physical band in the two unit systems: with a 1 um
cell on 96 points ``max|ky|`` is ~192 /um but only ~0.19 /nm, so the nm run has
its band pinned at the floor and is 5.2x wider in physical terms.

Run on both trees:

    PYTHONPATH=<tree> python ve_repro.py --build pre|post
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["pre", "post"])
    a = ap.parse_args()
    import lumenairy
    print("lumenairy.__file__ =", lumenairy.__file__)
    _vh.require_tree(a.build)
    from lumenairy.elements.eme import eme_2d

    Nx = 96
    eps = np.full(Nx, 2.25 + 0j)
    eps[Nx // 2:] = 12.0 - 1e-6j                 # weak GAIN in the high half
    lam_um = 1.55                                # wavelength

    out = {}
    for unit, L, k0 in (("um", 1.0, 2 * np.pi / lam_um),
                        ("nm", 1000.0, 2 * np.pi / (lam_um * 1000.0))):
        lam = np.asarray(eme_2d.strip_x_modes(eps, L, Nx, k0, 0.0)[0],
                         dtype=complex)
        lam = lam[np.lexsort((lam.imag, lam.real))]
        ky = np.asarray(eme_2d._ky_forward(lam, 0.0))
        out[unit] = ky * (L / 1.0)               # back to 1/um
        print("  %-3s  max|ky| in its own units = %-12.5g  band = %.4g"
              % (unit, float(np.max(np.abs(np.sqrt(lam + 0j)))),
                 1e-9 * max(1.0, float(np.max(np.abs(np.sqrt(lam + 0j)))))))
    d = np.abs(out["um"] - out["nm"])
    n = int(np.sum(d > 1e-6 * np.maximum(np.abs(out["um"]), 1.0)))
    print("\n  %d / %d modes came back on a DIFFERENT ROOT in nm than in um"
          % (n, d.size))
    print("  worst |d ky| (1/um) = %.6g   (spectrum top = %.6g /um)"
          % (float(np.max(d)), float(np.max(np.abs(out["um"])))))
    if n:
        j = int(np.argmax(d))
        print("  example mode %d:  um -> %+.9g%+.4gj      nm -> %+.9g%+.4gj"
              % (j, out["um"][j].real, out["um"][j].imag,
                 out["nm"][j].real, out["nm"][j].imag))
    print("\n  VERDICT[%s]: %s" % (a.build,
                                   "UNIT-DEPENDENT" if n else "unit-invariant"))


if __name__ == "__main__":
    main()
