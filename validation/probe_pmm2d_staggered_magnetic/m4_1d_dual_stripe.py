"""Probe 4 (G4) -- a y-uniform MAGNETIC stripe grating against the library's
1-D diffraction engines, through duality.

ENGINE CENSUS (2026-09-10, ``grep --include='*.py' -riE 'permeability|\\bmu'``
over ``lumenairy/elements/{rcwa,pmm,berreman.py}``): NO 1-D diffraction engine
in the library accepts a permeability.  ``rcwa_jones_1d`` / ``rcwa_efficiency_1d``
/ ``PMMStack`` / ``pmm_jones_1d`` are nonmagnetic (the ``mu`` occurrences in
``pmm/_core.py`` are the eps-free geometric EIGENVALUE ``mu`` of ``Kx2`` and the
slant metric's ``mu^lm = sqrt(g) g^lm``, both coordinate artefacts, not a
material); ``berreman.py`` states ``mu = 1`` in its layer-matrix docstring.  The
only user-settable permeability anywhere is ``eme/eme_2d_vector._build_generator``'s
``mu_xy``, and that is a WAVEGUIDE MODE solver -- it returns propagation
constants of a cross-section, not diffraction orders, so it cannot serve as an
R/T oracle for a grating.

So the 1-D check is made through DUALITY instead, which is strictly better than
"no check": the ELECTRIC stripe (eps patterned, mu = 1) is solvable by BOTH 1-D
engines AND is the dual of the MAGNETIC stripe (eps = 1, mu patterned).  With
vacuum half-spaces the two are the same physical problem with s and p
exchanged, so the magnetic 2-D staggered solve is compared PER ORDER against a
1-D engine's electric solve with the rows swapped.  Chain: magnetic staggered
-> (exact duality) -> 1-D PMM / 1-D RCWA.

The library's two efficiency ROWS are already the s / p POWER responses (each
is normalised by that drive's full ``|E_inc|^2`` through ``einc_sq``), so the
duality statement on R/T is a plain row swap at any incidence.  The JONES,
however, is a TANGENTIAL-amplitude matrix, and duality maps tangential
amplitudes with ``E_t -> (k^ x E)_t``: in a classical mount (phi = 0) that map
is ``A = [[0, -kz], [1/kz, 0]]`` (``kz = cos theta``), so

    J_dual = -A J A^-1                                              [rule 2]

which at normal incidence collapses to ``C J C`` with ``C = [[0,1],[-1,0]]``.
MEASURED at theta = 0.22, M = 8: 1.72e-05 with the ``A`` rule against 3.90e-03
with the normal-incidence one -- the 1/cos(theta) factors are real and this
probe carries them.
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_mag"), \
    lumenairy.__file__

from lumenairy.elements.pmm import pmm_jones_1d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_1d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

P, WL, DEP, TH = 0.90e-6, 0.55e-6, 0.30e-6, 0.22
RIDGE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
GROOVE = 2.10 * np.eye(3, dtype=complex)
EYE = np.eye(3, dtype=complex)
C = np.array([[0.0, 1.0], [-1.0, 0.0]])


def stripe(t_ridge, t_groove):
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = t_groove
    c[0, :] = t_ridge
    return c


def magnetic_arm(M, theta):
    """eps = vacuum everywhere, mu = the stripe -- the DUAL grating."""
    return pmm_jones_2d_staggered(
        P, P, stripe(EYE, EYE), 1.0, 1.0, DEP, WL, mu_cell=stripe(RIDGE, GROOVE),
        degree=M, n_orders=3, theta=theta, phi=0.0)


def compare(M, theta, oracle, tag):
    o, R, T, J = magnetic_arm(M, theta)
    o1, R1, T1, J1 = oracle
    sel = np.asarray(o)[:, 1] == 0
    two = {int(m): i for i, m in zip(np.arange(len(o))[sel],
                                     np.asarray(o)[sel, 0])}
    one = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
    common = [m for m in one if m in two]
    # duality: the magnetic arm's row r is the electric arm's row 1 - r
    dRT = max(max(abs(R[r][two[m]] - R1[1 - r][one[m]]),
                  abs(T[r][two[m]] - T1[1 - r][one[m]]))
              for m in common for r in (0, 1))
    kz = np.cos(theta)
    A = np.array([[0.0, -kz], [1.0 / kz, 0.0]])
    dJ = float(np.max(np.abs(J + A @ J1 @ np.linalg.inv(A))))
    dJ_norm = float(np.max(np.abs(J - C @ J1 @ C)))
    forb = float(max(np.max(np.abs(R[:, ~sel])), np.max(np.abs(T[:, ~sel]))))
    unsw = max(abs(R[r][two[m]] - R1[r][one[m]])
               for m in common for r in (0, 1))
    print(f"{tag:<26s} th={theta:.2f} M={M}  dRT={dRT:.3e}  dJ={dJ:.3e}  "
          f"y-forbidden={forb:.2e}  (unswapped control {unsw:.2e}, "
          f"normal-incidence Jones rule {dJ_norm:.2e})")
    return dRT, dJ, forb


if __name__ == "__main__":
    for theta in (0.0, TH):
        pm = pmm_jones_1d(P, RIDGE, GROOVE, 1.0, 1.0, DEP, 0.5, WL,
                          angle=theta, degree=16, stabilize=False)
        rc = rcwa_jones_1d(P, RIDGE, GROOVE, 1.0, 1.0, DEP, 0.5, WL,
                           angle=theta, n_orders=40)
        o1, R1, T1, J1 = pm
        o2, R2, T2, J2 = rc
        one1 = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
        one2 = {int(m): i for i, m in enumerate(np.asarray(o2).ravel())}
        com = [m for m in one1 if m in one2]
        spread = max(max(abs(R1[r][one1[m]] - R2[r][one2[m]]),
                         abs(T1[r][one1[m]] - T2[r][one2[m]]))
                     for m in com for r in (0, 1))
        print(f"--- theta={theta}: the two 1-D oracles' own mutual spread "
              f"dRT={spread:.3e}, dJ={np.max(np.abs(J1 - J2)):.3e}")
        for M in (5, 6, 7, 8):
            compare(M, theta, pm, "vs pmm_jones_1d")
        compare(8, theta, rc, "vs rcwa_jones_1d")
        print()
