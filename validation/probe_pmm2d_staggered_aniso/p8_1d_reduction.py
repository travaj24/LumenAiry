"""Probe 8 -- G4: y-uniform ANISOTROPIC stripe grating in the staggered 2-D
tensor solver vs the 1-D engines (pmm_jones_1d, rcwa_jones_1d), per order,
both incident polarizations, plus the Jones and the y-momentum conservation."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm import pmm_jones_1d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_1d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

WL = 0.55e-6
P = 0.90e-6
DEP = 0.30e-6
RIDGE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
GROOVE = 2.10 * np.eye(3, dtype=complex)


def cell(nseg):
    c = np.empty((nseg, nseg, 3, 3), dtype=complex)
    c[:] = GROOVE
    c[:nseg // 2, :] = RIDGE          # stripe along y, duty 0.5
    return c


def main():
    for theta in (0.0, 0.22):
        o1, R1, T1, J1 = pmm_jones_1d(P, RIDGE, GROOVE, 1.5, 1.0, DEP, 0.5,
                                      WL, angle=theta, degree=16,
                                      stabilize=False)
        orc, Rr, Tr, Jr = rcwa_jones_1d(P, RIDGE, GROOVE, 1.5, 1.0, DEP, 0.5,
                                        WL, angle=theta, n_orders=40)
        print(f"\n--- theta = {theta} rad ---")
        ir = {int(m): i for i, m in enumerate(np.asarray(orc).ravel())}
        i1o = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
        sh = [m for m in i1o if m in ir]
        print("1-D oracles disagree by (R/T/J):",
              f"{max(abs(R1[r][i1o[m]] - Rr[r][ir[m]]) for m in sh for r in (0, 1)):.2e}",
              f"{max(abs(T1[r][i1o[m]] - Tr[r][ir[m]]) for m in sh for r in (0, 1)):.2e}",
              f"{np.max(np.abs(J1 - Jr)):.2e}")
        for M in (5, 6, 7, 8):
            o, R, T, J = pmm_jones_2d_staggered(P, P, cell(2), 1.5, 1.0, DEP,
                                                WL, degree=M, n_orders=3,
                                                theta=theta, phi=0.0)
            sel = o[:, 1] == 0
            m2 = o[sel, 0]
            ordmap = {int(m): i for i, m in
                      zip(np.arange(len(o))[sel], m2)}
            common = [m for m in np.asarray(o1).ravel().astype(int)
                      if m in ordmap]
            i1 = {int(m): i for i, m in enumerate(np.asarray(o1).ravel())}
            dR = max(abs(R[r][ordmap[m]] - R1[r][i1[m]])
                     for m in common for r in (0, 1))
            dT = max(abs(T[r][ordmap[m]] - T1[r][i1[m]])
                     for m in common for r in (0, 1))
            dJ = float(np.max(np.abs(J - J1)))
            forb = float(np.max(np.abs(R[:, ~sel])) + np.max(np.abs(T[:, ~sel])))
            clo = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
            print(f"M={M}: per-order dR={dR:.3e} dT={dT:.3e} dJones={dJ:.3e}"
                  f"  y-forbidden={forb:.2e}  closure={clo:.2e}")


if __name__ == "__main__":
    main()
