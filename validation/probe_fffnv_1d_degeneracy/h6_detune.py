"""H6 -- the fixture sits on an EXACT index coincidence (no^2 = eg = n_sub^2).
Detune each of the three and watch the closure defect collapse."""
import os
import sys
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


NS = (7, 11, 21, 41, 61)


def defects(no=1.5, eg=2.25, nsub=1.5):
    er = _rot(np.deg2rad(35.0), no, 2.3)
    seg = [(0.5, er), (0.5, np.diag([eg] * 3).astype(complex))]
    out = []
    for n in NS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, seg, nsub, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
        out.append(float(np.sum(R1) + np.sum(T1) - 2.0))
    return out


print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print("detune the GROOVE eps away from no^2 = 2.25 (n_sub stays 1.5):")
print(f"{'rel detune of eg':<24}" + "".join(f"{n:>10}" for n in NS))
for r in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
    print(f"{r:<24.0e}" + "".join(f"{d:>10.2e}" for d in defects(eg=2.25 * (1 + r))))
print()
print("detune the ORDINARY index away from sqrt(eg) = 1.5 (eg, n_sub fixed):")
print(f"{'rel detune of no':<24}" + "".join(f"{n:>10}" for n in NS))
for r in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
    print(f"{r:<24.0e}" + "".join(f"{d:>10.2e}" for d in defects(no=1.5 * (1 + r))))
print()
print("detune the SUBSTRATE index away from 1.5 (er, eg fixed):")
print(f"{'rel detune of n_sub':<24}" + "".join(f"{n:>10}" for n in NS))
for r in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
    print(f"{r:<24.0e}" + "".join(f"{d:>10.2e}" for d in defects(nsub=1.5 * (1 + r))))
