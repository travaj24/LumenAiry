"""Probe 3 -- three-engine cross-check on the Granet Fig.4 geometry AND on a
generic in-plane-anisotropic crossed cell.

If the new staggered tensor assembly is wrong, the two INDEPENDENT engines
(hybrid PMM `pmm_jones_2d`, `rcwa_jones_2d`) disagree with it.  If all three
agree and the PAPER disagrees, the discrepancy is in the paper's geometry /
efficiency definition, not in the assembly.
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402

LAM = 1.0e-6
EPS_B_PAPER = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0],
                        [0.0, 0.0, 2.0]], dtype=complex)
EPS_B = np.conj(EPS_B_PAPER)
EPS_A = EPS_B_PAPER
N_SUB = np.sqrt(1.0 + 5.0j)
KEYS = [(1, 1), (-1, 1), (0, -1), (0, 0)]


def granet_cell(up=1):
    """(2*up, 2*up, 3, 3) pixel cell: pillar = lower-left quadrant."""
    n = 2 * up
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = EPS_A
    c[:up, :up] = EPS_B
    return c


def _pick(o, A):
    idx = {tuple(int(v) for v in r): i for i, r in enumerate(np.asarray(o))}
    return {k: float(A[idx[k]]) for k in KEYS}


def main():
    dx, dy = 2.4 * LAM, 1.4 * LAM
    print("=== Granet Fig.4 geometry: transmitted efficiency, incident E_x ===")
    o, R, T, J = pmm_jones_2d_staggered(dx, dy, granet_cell(1), N_SUB, 1.0,
                                        LAM, LAM, degree=7, n_orders=4)
    stag = _pick(o, T[0])
    print("staggered  M=7  :", {k: round(v, 5) for k, v in stag.items()},
          " R+T =", round(float(R[0].sum() + T[0].sum()), 5))
    for nord in (9, 13):
        oh, Rh, Th, Jh = pmm_jones_2d(dx, dy, granet_cell(1), N_SUB, 1.0,
                                      LAM, LAM, degree=9, n_orders=nord)
        hyb = _pick(oh, Th[0])
        print(f"hybrid  no={nord}  :", {k: round(v, 5) for k, v in hyb.items()},
              " maxdev vs staggered =",
              f"{max(abs(hyb[k] - stag[k]) for k in KEYS):.2e}")
    for nord in (9, 15):
        orc, Rr, Tr, Jr = rcwa_jones_2d(dx, dy, granet_cell(24), N_SUB, 1.0,
                                        LAM, LAM, n_orders=nord)
        rc = _pick(orc, Tr[0])
        print(f"rcwa    no={nord}  :", {k: round(v, 5) for k, v in rc.items()},
              " maxdev vs staggered =",
              f"{max(abs(rc[k] - stag[k]) for k in KEYS):.2e}")
    print("paper SEM Tab.2 :", {(1, 1): 0.0268, (-1, 1): 0.0139,
                                (0, -1): 0.0620, (0, 0): 0.2979})


if __name__ == "__main__":
    main()
