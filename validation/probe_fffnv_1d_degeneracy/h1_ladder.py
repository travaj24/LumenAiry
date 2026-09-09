"""H1 -- the rigorous 1-D lossless-closure ladder for the fff_nv stripe
fixture.  argv[1] = required lumenairy root."""
import os
import sys
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
_HERE = os.path.abspath(lumenairy.__file__).replace("\\", "/").lower()
assert _HERE.startswith(_ROOT), (_HERE, _ROOT)

from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


er = _rot(np.deg2rad(35.0), 1.5, 2.3)
eg = np.diag([2.25] * 3).astype(complex)
SEG = [(0.5, er), (0.5, eg)]

print(f"numpy {np.__version__}  threads OMP={os.environ.get('OMP_NUM_THREADS')} "
      f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"{'n':>4} {'closure defect':>16} {'sumR':>18} {'sound<1e-9':>11}")
best = (1e9, None)
for n in range(5, 62, 2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, J1 = rcwa_jones_1d_segments(
            PX, SEG, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
    d = float(np.sum(R1) + np.sum(T1) - 2.0)
    if abs(d) < best[0]:
        best = (abs(d), n)
    print(f"{n:>4} {d:>16.3e} {float(np.sum(R1)):>18.12f} "
          f"{'YES' if abs(d) < 1e-9 else '':>11}")
print(f"best |defect| = {best[0]:.3e} at n = {best[1]}")
