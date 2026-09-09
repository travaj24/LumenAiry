"""Probe 5 -- G2 diagnostics: full order spectrum, both incident pols, the
(2,2)-corner vs (3,3)-centred position invariance, and an M ladder."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

LAM = 1.0e-6
EPS_P = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0],
                  [0.0, 0.0, 2.0]], dtype=complex)
EPS_B, EPS_A = np.conj(EPS_P), EPS_P
N_SUB = np.sqrt(1.0 + 5.0j)


def cell22():
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[:] = EPS_A
    c[0, 0] = EPS_B
    return c


def cell44_centred():
    c = np.empty((4, 4, 3, 3), dtype=complex)
    c[:] = EPS_A
    c[1:3, 1:3] = EPS_B          # centred, w = 0.5 d
    return c


def show(tag, o, R, T):
    prop = [i for i in range(len(o)) if abs(o[i, 0]) <= 2 and abs(o[i, 1]) <= 1]
    print(f"-- {tag}:  sum R = {R[0].sum():.5f}  sum T = {T[0].sum():.5f}"
          f"  R+T = {R[0].sum() + T[0].sum():.6f}")
    for i in sorted(prop, key=lambda j: (o[j, 0], o[j, 1])):
        print(f"     ({o[i,0]:+d},{o[i,1]:+d})  R={R[0][i]:.5f} T={T[0][i]:.5f}"
              f"   | Ey-row R={R[1][i]:.5f} T={T[1][i]:.5f}")


def main():
    dx, dy = 2.4 * LAM, 1.4 * LAM
    for M in (5, 7, 9):
        o, R, T, _ = pmm_jones_2d_staggered(dx, dy, cell22(), N_SUB, 1.0,
                                            LAM, LAM, degree=M, n_orders=3)
        i00 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        i11 = int(np.where((o[:, 0] == 1) & (o[:, 1] == 1))[0][0])
        print(f"M={M}: T00={T[0][i00]:.6f}  T11={T[0][i11]:.6f}  "
              f"R+T={R[0].sum()+T[0].sum():.6f}")
    o, R, T, _ = pmm_jones_2d_staggered(dx, dy, cell22(), N_SUB, 1.0, LAM, LAM,
                                        degree=7, n_orders=3)
    show("(2,2) corner pillar M=7", o, R, T)
    o4, R4, T4, _ = pmm_jones_2d_staggered(dx, dy, cell44_centred(), N_SUB,
                                           1.0, LAM, LAM, degree=5,
                                           n_orders=3)
    show("(4,4) centred pillar M=5", o4, R4, T4)
    print("position invariance maxdev T:",
          f"{np.max(np.abs(T[0] - T4[0])):.3e}")


if __name__ == "__main__":
    main()


def extra_readings():
    """A few DISCRETE alternative readings of the Fig.4 example (not a
    continuous parameter fit): vacuum host, swapped roles."""
    KEYS = [(1, 1), (-1, 1), (0, -1), (0, 0)]
    TAB2 = {(1, 1): 0.0268, (-1, 1): 0.0139, (0, -1): 0.0620, (0, 0): 0.2979}
    vac = np.eye(3, dtype=complex)
    opts = {
        "host=eps_a, pillar=eps_b": (EPS_A, EPS_B),
        "host=vacuum, pillar=eps_b": (vac, EPS_B),
        "host=vacuum, pillar=eps_a": (vac, EPS_A),
        "host=eps_b, pillar=vacuum": (EPS_B, vac),
        "host=eps_a, pillar=vacuum": (EPS_A, vac),
    }
    for dxdy in ((2.4, 1.4), (1.4, 2.4)):
        for name, (h, p) in opts.items():
            c = np.empty((2, 2, 3, 3), dtype=complex)
            c[:] = h
            c[0, 0] = p
            o, R, T, _ = pmm_jones_2d_staggered(
                dxdy[0] * LAM, dxdy[1] * LAM, c, N_SUB, 1.0, LAM, LAM,
                degree=7, n_orders=3)
            idx = {tuple(int(v) for v in r): i for i, r in enumerate(o)}
            v = {k: float(T[0][idx[k]]) for k in KEYS}
            dev = max(abs(v[k] - TAB2[k]) for k in KEYS)
            print(f"d={dxdy}  {name:28s} dev={dev:8.2e}  "
                  + "  ".join(f"{str(k)}={v[k]:.5f}" for k in KEYS))


extra_readings()
