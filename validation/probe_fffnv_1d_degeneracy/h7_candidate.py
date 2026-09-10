"""H7 -- candidate NON-degenerate fixture for
test_fff_nv_stripe_reduces_to_rigorous_1d: closure ladder + the two ratios."""
import os
import sys
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)

from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)
CANDS = {"eg=2.25 (SHIPPED, degenerate)": 2.25,
         "eg=2.10": 2.10, "eg=1.96 (n=1.4)": 1.96, "eg=2.56 (n=1.6)": 2.56}
print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
NS = (11, 13, 15, 17, 19, 21, 25, 31, 41, 51, 61)
print("1-D rigorous closure defect ladder")
print(f"{'fixture':<32}" + "".join(f"{n:>10}" for n in NS))
for tag, g in CANDS.items():
    eg = np.diag([g] * 3).astype(complex)
    row = []
    for n in NS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        row.append(f"{float(np.sum(R1) + np.sum(T1) - 2.0):>10.1e}")
    print(f"{tag:<32}" + "".join(row))

print()
print("2-D arms vs the 1-D reference at n_ref = 61")
print(f"{'fixture':<32}{'No':>4}{'fff_nv clo':>12}{'ef':>11}{'el':>11}"
      f"{'ef/el':>9}{'jf':>11}{'jl':>11}{'jf/jl':>9}")
for tag, g in CANDS.items():
    eg = np.diag([g] * 3).astype(complex)
    cell = _stripe(ER, eg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, J1 = rcwa_jones_1d_segments(
            PX, [(0.5, ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
            n_orders=61)
        for No in (9, 11, 13):
            try:
                _o, Rf, Tf, Jf = rcwa_jones_2d(
                    PX, PX, cell, 1.5, 1.0, DEPTH, WL, n_orders_x=No,
                    n_orders_y=1, formulation="fff_nv", symmetry=False)
                _o, Rl, Tl, Jl = rcwa_jones_2d(
                    PX, PX, cell, 1.5, 1.0, DEPTH, WL, n_orders_x=No,
                    n_orders_y=1, formulation="laurent", symmetry=False)
            except Exception as exc:
                print(f"{tag if No == 9 else '':<32}{No:>4}   RAISED "
                      f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            clo = abs(np.sum(Rf) + np.sum(Tf) - 2.0)
            ef = abs(np.sum(Rf) - np.sum(R1))
            el = abs(np.sum(Rl) - np.sum(R1))
            jf = np.max(np.abs(Jf - J1))
            jl = np.max(np.abs(Jl - J1))
            print(f"{tag if No == 9 else '':<32}{No:>4}{clo:>12.2e}"
                  f"{ef:>11.2e}{el:>11.2e}{ef / el:>9.4f}{jf:>11.2e}"
                  f"{jl:>11.2e}{jf / jl:>9.4f}")
