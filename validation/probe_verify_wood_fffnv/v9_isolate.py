"""V9 -- the two loose ends of Task H (H.3 step 1, and the reference bar's
trend).

A. "It is not the anisotropic solver": the SAME call and geometry with
   different permittivities.  Only the index-coincident cell misbehaves.
B. The 1-D reference's own lossless closure vs the reference truncation, on the
   DETUNED fixture -- the quantity `_ONED_SOUND_CLOSURE` (1e-9) is asserted
   against at `_ONED_REF_ORDERS` = 81, so its margin's trend matters.

    python v9_isolate.py <lumenairy-root> [tag]
"""
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


ER = rot(np.deg2rad(35.0), 1.5, 2.3)


def defect(segments, n, n_sub=1.5, n_sup=1.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, _J = rcwa_jones_1d_segments(
            PX, segments, n_sub, n_sup, DEPTH, WL, theta=0.0, n_orders=int(n))
    return float(np.sum(R1) + np.sum(T1) - 2.0)


EG225 = np.diag([2.25] * 3).astype(complex)
sym = ER.copy()
sym[0, 1] = sym[1, 0] = 0.5                       # a DIFFERENT symmetric tensor
gyro = np.diag([2.25, 2.25, 2.25]).astype(complex)
gyro[0, 1], gyro[1, 0] = 0.5j, -0.5j              # gyrotropic Hermitian
diagonal = np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex)

CASES = [
    ("the fixture: rot 35 deg vs 2.25 (coincident)", [(0.5, ER), (0.5, EG225)],
     dict()),
    ("the same tensor NOT rotated (exy = 0)", [(0.5, diagonal), (0.5, EG225)],
     dict()),
    ("isotropic 5.29 vs 2.25", [(0.5, np.diag([5.29] * 3).astype(complex)),
                                (0.5, EG225)], dict()),
    ("a DIFFERENT symmetric tensor (exy = 0.5)", [(0.5, sym), (0.5, EG225)],
     dict()),
    ("GYROTROPIC Hermitian (exy = +0.5i)", [(0.5, gyro), (0.5, EG225)], dict()),
    ("the fixture, UNIFORM (no grating)", [(1.0, ER)], dict()),
    ("the fixture, half-spaces matched 1.0/1.0",
     [(0.5, ER), (0.5, EG225)], dict(n_sub=1.0, n_sup=1.0)),
    ("the fixture, groove detuned to 2.10",
     [(0.5, ER), (0.5, np.diag([2.10] * 3).astype(complex))], dict()),
]

print("A. which property breaks the closure -- sum R + sum T - 2")
print(f"{'case':46s} " + " ".join(f"{n:>10d}" for n in (11, 21, 31, 41, 61)))
for name, segs, kw in CASES:
    row = [defect(segs, n, **kw) for n in (11, 21, 31, 41, 61)]
    print(f"{name:46s} " + " ".join(f"{v:10.2e}" for v in row))

print()
print("B. the 1-D reference's own closure vs n_ref, DETUNED fixture (2.10)")
EG210 = [(0.5, ER), (0.5, np.diag([2.10] * 3).astype(complex))]
print(f"{'n_ref':>6s} {'|closure|':>12s} {'sum R':>16s} {'margin to 1e-9':>16s}")
for n in (41, 61, 81, 101, 141, 181):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, _J = rcwa_jones_1d_segments(
            PX, EG210, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
    d = abs(float(np.sum(R1) + np.sum(T1) - 2.0))
    print(f"{n:6d} {d:12.4e} {float(np.sum(R1)):16.12f} {1e-9 / d:16.1f}x")
