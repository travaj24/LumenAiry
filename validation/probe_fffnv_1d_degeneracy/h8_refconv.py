"""H8 -- convergence of the 1-D reference on the NON-degenerate fixture."""
import os
import sys
import time
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)

from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)
EG = np.diag([2.10] * 3).astype(complex)
SEG = [(0.5, ER), (0.5, EG)]
prev = None
print(f"numpy {np.__version__}")
print(f"{'n':>5}{'sumR':>18}{'d sumR':>12}{'max|dJ|':>12}{'closure':>12}{'s':>8}")
for n in (21, 31, 41, 51, 61, 81, 101, 141):
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, J1 = rcwa_jones_1d_segments(
            PX, SEG, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
    dt = time.perf_counter() - t
    s = float(np.sum(R1))
    d = "" if prev is None else f"{abs(s - prev[0]):>12.2e}"
    dj = "" if prev is None else f"{float(np.max(np.abs(J1 - prev[1]))):>12.2e}"
    print(f"{n:>5}{s:>18.12f}{d:>12}{dj:>12}"
          f"{float(np.sum(R1) + np.sum(T1) - 2.0):>12.2e}{dt:>8.2f}")
    prev = (s, J1)
