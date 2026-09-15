"""D3 mirror-fixture probe.

Measures, on whatever arm it is run on:
  * whether ``cos(deg2rad(45))`` and ``sin(deg2rad(45))`` are bit-identical,
  * the max |A - mirror(A)| of the raw ``uniaxial_tensor`` used by the D3
    fixture, componentwise,
  * the same for the full (S, S, 3, 3) cells the two failing parametrisations
    build,
  * the same after the proposed exact symmetrisation.

Usage:  python probe_d3_mirror.py <out.json>
"""
import json
import os
import struct
import sys

import numpy as np

import lumenairy
from lumenairy.elements.rcwa import uniaxial_tensor

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

_PERM = [1, 0, 2]
S = 96


def _mirror(cell):
    return np.transpose(cell, (1, 0, 2, 3))[:, :, _PERM, :][:, :, :, _PERM]


def _mirror3(t):
    return t[_PERM, :][:, _PERM]


def _cell(kind, tilt, back=1.0 + 0j):
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    c = np.zeros((S, S, 3, 3), dtype=complex)
    c[:] = np.eye(3) * back
    if kind == "square":
        m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 5)
    elif kind == "rect":
        m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 8)
    elif kind == "stripe":
        m = np.abs(ii - S // 2) < S // 5
    elif kind == "uniform":
        m = np.ones((S, S), dtype=bool)
    else:
        m = ((ii - (S - 1) / 2.0) ** 2
             + (jj - (S - 1) / 2.0) ** 2) <= (0.25 * S) ** 2
    c[m] = tilt
    return c


def bits(x):
    return struct.pack(">d", float(x)).hex()


out = {}
out["python"] = sys.version.split()[0]
out["numpy"] = np.__version__
out["lumenairy_file"] = lumenairy.__file__
out["env"] = {k: os.environ.get(k) for k in
              ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS")}

ang = np.deg2rad(45.0)
cp, sp = float(np.cos(ang)), float(np.sin(ang))
out["deg2rad45"] = bits(ang)
out["cos45_bits"] = bits(cp)
out["sin45_bits"] = bits(sp)
out["cos45_eq_sin45"] = bool(cp == sp)
out["cos45_minus_sin45"] = cp - sp

D = uniaxial_tensor(2.0, 2.6, np.deg2rad(40.0), phi=np.deg2rad(45.0))
out["DIAG"] = [[str(v) for v in row] for row in D]
dM = D - _mirror3(D)
out["DIAG_mirror_defect_max"] = float(np.max(np.abs(dM)))
out["DIAG_mirror_defect_components"] = {
    f"{i}{j}": float(abs(dM[i, j])) for i in range(3) for j in range(3)
    if abs(dM[i, j]) > 0.0}
out["DIAG_scale"] = float(np.max(np.abs(D)))
out["DIAG_is_own_mirror"] = bool(np.array_equal(D, _mirror3(D)))

# the proposed exact fix: average the tensor with its own mirror
Dsym = 0.5 * (D + _mirror3(D))
out["DIAGsym_is_own_mirror"] = bool(np.array_equal(Dsym, _mirror3(Dsym)))
out["DIAGsym_shift_from_DIAG"] = float(np.max(np.abs(Dsym - D)))

cells = {}
for kind in ("square", "disk"):
    c = _cell(kind, D)
    cs = _cell(kind, Dsym)
    cells[kind] = {
        "n_pixels_in_mask": int(np.count_nonzero(
            np.abs(c[:, :, 0, 0] - 1.0) > 0)),
        "mask_is_own_transpose": bool(np.array_equal(
            np.abs(c[:, :, 0, 0] - 1.0) > 0,
            (np.abs(c[:, :, 0, 0] - 1.0) > 0).T)),
        "defect_max": float(np.max(np.abs(c - _mirror(c)))),
        "is_own_mirror": bool(np.array_equal(c, _mirror(c))),
        "sym_defect_max": float(np.max(np.abs(cs - _mirror(cs)))),
        "sym_is_own_mirror": bool(np.array_equal(cs, _mirror(cs))),
    }
out["cells"] = cells

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
