"""Gate 2 probe -- scalar cell vs its e*I promotion ON a LAYER cutoff.
argv[1] = the lumenairy root the arm MUST come from."""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
_HERE = os.path.abspath(lumenairy.__file__).replace("\\", "/").lower()
assert _HERE.startswith(_ROOT), (_HERE, _ROOT)

import lumenairy.elements.rcwa._core as _rc  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)

_ORIG = _rc._grazing_safe_wavelength
_SEEN = []


def _spy(wavelength, *a, **kw):
    wl = _ORIG(wavelength, *a, **kw)
    _SEEN.append((float(wavelength), float(wl)))
    return wl


_rc._grazing_safe_wavelength = _spy


def promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


CELL = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=complex)

print("scalar-vs-tensor, uniform eps=4 layer, normal incidence")
for px, wl, note in ((0.5e-6, 1.0e-6, "ON the layer cutoff (wl/px = 2)"),
                     (0.5e-6, 1.0e-6 * (1 - 1e-9), "1e-9 off it"),
                     (0.7e-6, 0.55e-6, "far from any cutoff")):
    _SEEN.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o1, R1, T1 = pmm_efficiency_2d_staggered(
            px, px, CELL, 1.5, 1.0, 0.28e-6, wl, degree=5, n_orders=3,
            polarization="tm")
        _o2, R2, T2, _J2 = pmm_jones_2d_staggered(
            px, px, promote(CELL), 1.5, 1.0, 0.28e-6, wl, degree=5,
            n_orders=3)
    sc, te = _SEEN[0], _SEEN[1]
    print(f"px={px:.3e} wl={wl:.12e}  {note}\n"
          f"    max|dR| = {np.max(np.abs(R1 - R2[0])):.3e}   "
          f"max|dT| = {np.max(np.abs(T1 - T2[0])):.3e}   "
          f"bit-identical = {np.array_equal(R1, R2[0]) and np.array_equal(T1, T2[0])}\n"
          f"    scalar wl {sc[0]:.17e} -> {sc[1]:.17e}  rel {(sc[1]-sc[0])/sc[0]:.3e}\n"
          f"    tensor wl {te[0]:.17e} -> {te[1]:.17e}  rel {(te[1]-te[0])/te[0]:.3e}")

# a MULTI-VALUED scalar cell on a layer cutoff (eps=4 quarter-pillar in eps=1)
CELL2 = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
_SEEN.clear()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o1, R1, T1 = pmm_efficiency_2d_staggered(
        0.5e-6, 0.5e-6, CELL2, 1.5, 1.0, 0.28e-6, 1.0e-6, degree=5,
        n_orders=3, polarization="te")
    _o2, R2, T2, _J2 = pmm_jones_2d_staggered(
        0.5e-6, 0.5e-6, promote(CELL2), 1.5, 1.0, 0.28e-6, 1.0e-6, degree=5,
        n_orders=3)
print(f"patterned cell {{4,1}} ON the eps=4 cutoff: max|dR| = "
      f"{np.max(np.abs(R1 - R2[0])):.3e}  bit-identical = "
      f"{np.array_equal(R1, R2[0])}\n"
      f"    scalar wl rel shift {(_SEEN[0][1]-_SEEN[0][0])/_SEEN[0][0]:.3e}   "
      f"tensor wl rel shift {(_SEEN[1][1]-_SEEN[1][0])/_SEEN[1][0]:.3e}")
