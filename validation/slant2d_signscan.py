"""PHASE A step 1 -- pin the convection sign against the EXACT 1-D oracle.

A y-uniform slanted binary grating solved by the patched 2-D solver must
reproduce the shipped 1-D slanted solver.  All four sign candidates
{+1, -1, +i, -i} are scanned; exactly one is expected to agree, and the
failure of the other three is the two-sided evidence that the winner is
pinned by the oracle rather than assumed.
"""
import numpy as np

import slant2d_proto as SP   # installs the monkeypatch
from lumenairy.elements.pmm import pmm_jones_2d, pmm_jones_1d_slanted

P = 1.0e-6
WL = 633e-9
D = 300e-9
DUTY = 0.5
ER, EG = 4.0 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NX = 480
N_ORD = 7


def run2d(tx, ty, c, theta=0.0, degree=11):
    SP.set_slant(tx, ty, c)
    cell = SP.binary_cell(NX, DUTY, ER, EG)
    out = pmm_jones_2d(P, P, cell, NSUB, NSUP, D, WL,
                       theta=theta, phi=0.0, degree=degree, n_orders=N_ORD)
    SP.set_slant(0.0, 0.0)
    return out


def run1d(phi, theta=0.0, degree=16):
    return pmm_jones_1d_slanted(P, ER * np.eye(3), EG * np.eye(3), NSUB, NSUP,
                                D, DUTY, WL, phi, angle=theta, degree=degree,
                                far_field_orders=2 * N_ORD + 1,
                                factorization="convection")


def slice_2d(res):
    """(orders_m, R, T) restricted to the n=0 row of the 2-D order set."""
    o2, R2, T2, _J = res
    o2 = np.asarray(o2)
    m, n = o2[:, 0], o2[:, 1]
    keep = n == 0
    idx = np.argsort(m[keep])
    return m[keep][idx], np.asarray(R2)[:, keep][:, idx], np.asarray(T2)[:, keep][:, idx]


def slice_1d(res):
    o1, R1, T1, _J = res
    o1 = np.asarray(o1)
    return o1, np.asarray(R1), np.asarray(T1)


def compare(m2, R2, T2, o1, R1, T1):
    """Max per-order abs difference over the common orders, both pols.

    2-D row 0 = x-polarized = TM (E across the grooves); row 1 = y-pol = TE.
    1-D row 0 = TM, row 1 = TE (oned.py delegation note).
    """
    common = np.intersect1d(m2, o1)
    i2 = np.searchsorted(m2, common)
    i1 = np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


if __name__ == "__main__":
    phi_deg = 20.0
    phi = np.radians(phi_deg)
    tx = np.tan(phi)
    print(f"geometry: P={P*1e9:.0f}nm d={D*1e9:.0f}nm duty={DUTY} "
          f"eps {ER.real}/{EG.real} slant={phi_deg}deg (tx={tx:.6f})")

    for theta in (0.0, 0.20):
        o1, R1, T1 = slice_1d(run1d(phi, theta))
        # vertical control: the 2-D solver with NO slant vs 1-D with NO slant
        m2, R2, T2 = slice_2d(run2d(0.0, 0.0, -1j, theta))
        o1v, R1v, T1v = slice_1d(run1d(0.0, theta))
        base = compare(m2, R2, T2, o1v, R1v, T1v)
        print(f"\n--- theta={theta} ---")
        print(f"  VERTICAL control (slant=0, 2-D vs 1-D): {base:.3e}"
              "   <- the floor these comparisons can reach")
        for name, c in (("+1", 1.0 + 0j), ("-1", -1.0 + 0j),
                        ("+i", 1j), ("-i", -1j)):
            try:
                m2, R2, T2 = slice_2d(run2d(tx, 0.0, c, theta))
                d = compare(m2, R2, T2, o1, R1, T1)
                print(f"  c = {name:>2}: max per-order |2-D - 1-D| = {d:.3e}")
            except Exception as e:  # noqa: BLE001
                print(f"  c = {name:>2}: FAILED {type(e).__name__}: {e}")
