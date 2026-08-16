"""PHASE A step 0 -- establish the VERTICAL control floor.

Before any slant number means anything, the 2-D hybrid must reproduce the 1-D
PMM on the SAME y-uniform binary grating with NO slant.  That residual is the
Fourier-truncation floor of the hybrid (twod.py's documented n_orders floor),
and it is the bar every slant comparison is measured against.
"""
import numpy as np
import warnings

import slant2d_proto as SP
from lumenairy.elements.pmm import pmm_jones_2d, pmm_jones_1d

WL = 633e-9
D = 300e-9
DUTY = 0.5
ER, EG = 4.0 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NX = 480


def run2d(P, n_ord, theta, degree=11):
    SP.set_slant(0.0, 0.0)
    cell = SP.binary_cell(NX, DUTY, ER, EG)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T, J = pmm_jones_2d(P, P, cell, NSUB, NSUP, D, WL,
                                  theta=theta, phi=0.0, degree=degree,
                                  n_orders=n_ord)
        warned = any("energy" in str(x.message).lower() for x in w)
    o = np.asarray(o)
    keep = o[:, 1] == 0
    idx = np.argsort(o[keep][:, 0])
    R, T = np.asarray(R), np.asarray(T)
    clos = float(np.sum(R[0]) + np.sum(T[0]) - 1.0)
    return o[keep][:, 0][idx], R[:, keep][:, idx], T[:, keep][:, idx], clos, warned


def run1d(P, theta, n_ff, degree=16):
    o, R, T, J = pmm_jones_1d(P, ER * np.eye(3), EG * np.eye(3), NSUB, NSUP,
                              D, DUTY, WL, angle=theta, degree=degree,
                              far_field_orders=n_ff)
    return np.asarray(o), np.asarray(R), np.asarray(T)


def cmp_orders(m2, R2, T2, o1, R1, T1):
    common = np.intersect1d(m2, o1)
    i2 = np.searchsorted(m2, common)
    i1 = np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


if __name__ == "__main__":
    for P in (500e-9, 700e-9, 1000e-9):
        npro = int(np.floor(P / WL * NSUB)) * 2 + 1
        print(f"\n===== period {P*1e9:.0f} nm  (~{npro} propagating orders in sub) =====")
        for theta in (0.0, 0.20):
            o1, R1, T1 = run1d(P, theta, 21)
            row = []
            for n_ord in (7, 11, 15, 21, 27):
                try:
                    m2, R2, T2, clos, warned = run2d(P, n_ord, theta)
                    d = cmp_orders(m2, R2, T2, o1, R1, T1)
                    flag = "!" if warned else " "
                    row.append(f"n{n_ord}: {d:.2e}{flag}(c{clos:+.0e})")
                except Exception as e:  # noqa: BLE001
                    row.append(f"n{n_ord}: {type(e).__name__}")
            print(f"  theta={theta}: " + "  ".join(row))
