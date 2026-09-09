"""V8 -- the identity behind the rotation gauge, measured rather than argued.

The build doc reports the same number (7.80e-06) for its T2 ``rot = +1`` row
and its T9 ``negate the out-of-plane block`` row, which suggests the two
controls are the SAME operation.  If they are, then everything known about
"negating the out-of-plane block" transfers to the rot sign, and vice versa --
including exactly when each is invisible.

This measures it directly on one build, one fixture at a time:

    solve(cell, rot = +1)   vs   solve(negate_oop(cell), rot = -1)

must be BIT-IDENTICAL if `_OOP_ROT_SIGN` is nothing but a sign on the four
out-of-plane entries.  Run on a uniform slab, a chiral stripe and a chiral 2-D
cell, at normal / oblique / conical.

Usage:  PYTHONPATH=<root> python v8_rot_equals_negate.py <root> <out.json>
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = sys.argv[2]

tU = uniaxial_tensor(1.52, 1.78, 0.55, phi=0.93)
tA = uniaxial_tensor(1.48, 1.73, 0.62, phi=0.37)
tB = 2.10 * np.eye(3, dtype=complex)
tC = uniaxial_tensor(1.55, 1.80, 1.02, phi=2.20)

UNIFORM = np.zeros((2, 2, 3, 3), dtype=complex)
UNIFORM[:] = tU
STRIPE = np.zeros((3, 3, 3, 3), dtype=complex)
for i, t in enumerate((tA, tB, tC)):
    STRIPE[i, :, :, :] = t
CELL2D = np.zeros((3, 3, 3, 3), dtype=complex)
CELL2D[:, :] = np.eye(3, dtype=complex)
CELL2D[0, 0] = uniaxial_tensor(1.46, 1.74, 0.58, phi=0.31)
CELL2D[1, 0] = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
CELL2D[1, 2] = 2.25 * np.eye(3, dtype=complex)


def negate_oop(c):
    c = c.copy()
    c[..., 0, 2] *= -1
    c[..., 1, 2] *= -1
    c[..., 2, 0] *= -1
    c[..., 2, 1] *= -1
    return c


FIX = [("uniform_slab", UNIFORM, 1.15e-6, 0.70e-6, 0.31e-6, 6),
       ("chiral_stripe", STRIPE, 1.15e-6, 0.70e-6, 0.31e-6, 6),
       ("chiral_2d_cell", CELL2D, 1.10e-6, 0.68e-6, 0.36e-6, 5)]
MOUNTS = [("normal", 0.0, 0.0), ("oblique25", np.deg2rad(25.0), 0.0),
          ("conical25_40", np.deg2rad(25.0), np.deg2rad(40.0))]

out = {"root": ROOT, "lumenairy": lumenairy.__file__, "cases": []}


def solve(cell, px, wl, dep, M, th, ph):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d_staggered(px, px, cell, 1.45, 1.0, dep, wl,
                                      degree=M, n_orders=3, theta=th, phi=ph)


for name, cell, px, wl, dep, M in FIX:
    for mtag, th, ph in MOUNTS:
        orig = TS._OOP_ROT_SIGN
        try:
            TS._OOP_ROT_SIGN = +1.0
            a = solve(cell, px, wl, dep, M, th, ph)
            TS._OOP_ROT_SIGN = orig
            b = solve(negate_oop(cell), px, wl, dep, M, th, ph)
            c = solve(cell, px, wl, dep, M, th, ph)
        finally:
            TS._OOP_ROT_SIGN = orig
        rec = {"fixture": name, "mount": mtag,
               "rotplus_vs_negate_bit_identical":
                   bool(np.array_equal(a[1], b[1])
                        and np.array_equal(a[2], b[2])
                        and np.array_equal(np.asarray(a[3]), np.asarray(b[3]))),
               "rotplus_vs_negate_dR": float(np.max(np.abs(a[1] - b[1]))),
               "rotplus_vs_negate_dJones": float(np.max(np.abs(
                   np.asarray(a[3]) - np.asarray(b[3])))),
               "shipped_vs_rotplus_dR": float(np.max(np.abs(c[1] - a[1]))),
               "shipped_vs_rotplus_dJones": float(np.max(np.abs(
                   np.asarray(c[3]) - np.asarray(a[3]))))}
        out["cases"].append(rec)
        print(f"[{name} {mtag}] rot(+1) vs negate-OOP: bit_identical="
              f"{rec['rotplus_vs_negate_bit_identical']} dR="
              f"{rec['rotplus_vs_negate_dR']:.3e} dJ="
              f"{rec['rotplus_vs_negate_dJones']:.3e} | shipped vs rot(+1) dR="
              f"{rec['shipped_vs_rotplus_dR']:.3e} dJ="
              f"{rec['shipped_vs_rotplus_dJones']:.3e}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
print("DONE")
