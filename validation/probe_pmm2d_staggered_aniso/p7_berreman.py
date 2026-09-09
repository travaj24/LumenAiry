"""Probe 7 -- G3: UNIFORM in-plane tensor slab on a multi-segment staggered
grid vs the exact Berreman 4x4 oracle, at normal / oblique / conical."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

WL = 1.0e-6
P = 0.40e-6
DEP = 0.55e-6
NS, NC = 1.5, 1.0

LC = uniaxial_tensor(1.5, 1.75, np.pi / 2, phi=0.6)
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)


def cell(t33, n):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = t33
    return c


def run(t33, n, M, theta, phi):
    o, R, T, J = pmm_jones_2d_staggered(P, P, cell(t33, n), NS, NC, DEP, WL,
                                        degree=M, n_orders=2, theta=theta,
                                        phi=phi)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, DEP)], NS, NC, WL,
                                        angle=theta, phi=phi)
    return (np.array([R[0].sum(), R[1].sum()]),
            np.array([T[0].sum(), T[1].sum()]), J, Rb, Tb, jr)


def main():
    print(f"{'tensor':6s} {'n':>2s} {'M':>2s} {'theta':>6s} {'phi':>5s}"
          f" {'dR':>10s} {'dT':>10s} {'dJones':>10s} {'closure':>10s}")
    for name, t33 in (("LC", LC), ("gyro", GYRO)):
        for n in (2, 3):
            for theta, phi in ((0.0, 0.0), (25 * np.pi / 180, 0.0),
                               (25 * np.pi / 180, 40 * np.pi / 180)):
                for M in (5, 7):
                    Rs, Ts, J, Rb, Tb, jr = run(t33, n, M, theta, phi)
                    dR = float(np.max(np.abs(Rs - Rb)))
                    dT = float(np.max(np.abs(Ts - Tb)))
                    dJ = float(np.max(np.abs(J - jr)))
                    clo = float(np.max(np.abs(Rs + Ts - 1.0)))
                    print(f"{name:6s} {n:2d} {M:2d} {np.degrees(theta):6.1f}"
                          f" {np.degrees(phi):5.1f} {dR:10.2e} {dT:10.2e}"
                          f" {dJ:10.2e} {clo:10.2e}")


if __name__ == "__main__":
    main()
