"""H2 -- WHICH property of the fixture breaks the 1-D lossless closure?
argv[1] = required lumenairy root."""
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


eg = np.diag([2.25] * 3).astype(complex)
CASES = {
    "rotated 35 deg (THE fixture, exy != 0)": _rot(np.deg2rad(35.0), 1.5, 2.3),
    "same tensor, NOT rotated (diagonal, exy = 0)":
        np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex),
    "ISOTROPIC 2.3^2": np.diag([2.3 ** 2] * 3).astype(complex),
    "rotated 5 deg (weak off-diagonal)": _rot(np.deg2rad(5.0), 1.5, 2.3),
    "rotated 45 deg": _rot(np.deg2rad(45.0), 1.5, 2.3),
    "rotated 35, weak contrast ne=1.55": _rot(np.deg2rad(35.0), 1.5, 1.55),
}
print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"{'case':<46}" + "".join(f"{n:>11}" for n in (11, 21, 31, 41, 61)))
for name, er in CASES.items():
    row = []
    for n in (11, 21, 31, 41, 61):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        row.append(f"{float(np.sum(R1) + np.sum(T1) - 2.0):>11.2e}")
    print(f"{name:<46}" + "".join(row))
