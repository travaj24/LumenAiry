"""H5 -- narrow down WHERE the anisotropic 1-D closure defect comes from."""
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


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)
EG = np.diag([2.25] * 3).astype(complex)
NS = (5, 7, 11, 21, 41, 61)


def run(seg, depth=DEPTH, nsub=1.5, nsup=1.0, tag=""):
    row = []
    for n in NS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, seg, nsub, nsup, depth, WL, theta=0.0, n_orders=n)
        row.append(f"{float(np.sum(R1) + np.sum(T1) - 2.0):>10.2e}")
    print(f"{tag:<48}" + "".join(row))


print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"{'case':<48}" + "".join(f"{n:>10}" for n in NS))
run([(1.0, ER)], tag="UNIFORM rotated tensor (no grating)")
run([(0.5, ER), (0.5, ER)], tag="two IDENTICAL rotated segments")
run([(0.5, ER), (0.5, EG)], tag="THE fixture (rotated | isotropic 2.25)")
run([(0.5, ER), (0.5, EG)], depth=1e-12, tag="THE fixture, depth -> 0")
run([(0.5, ER), (0.5, EG)], depth=0.05e-6, tag="THE fixture, depth 0.05 um")
run([(0.5, ER), (0.5, EG)], depth=2.0e-6, tag="THE fixture, depth 2 um")
run([(0.5, ER), (0.5, EG)], nsub=1.0, tag="THE fixture, matched half-spaces 1/1")
run([(0.5, np.conj(ER)), (0.5, EG)], tag="conjugated ER (same, real tensor)")
E2 = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]], complex)
run([(0.5, E2), (0.5, EG)], tag="different symmetric tensor exy=0.5")
E3 = np.array([[3.0, 0.5j, 0.0], [-0.5j, 2.0, 0.0], [0.0, 0.0, 2.0]], complex)
run([(0.5, E3), (0.5, EG)], tag="GYROTROPIC Hermitian exy=+0.5i")
