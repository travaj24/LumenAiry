"""V8 -- reproducer for the scalar/tensor Wood-anomaly split (build doc open
item 6).

`PMM2DStackPure.solve` feeds TENSOR layers' permittivity diagonals into
`_grazing_safe_wavelength` and deliberately omits SCALAR layers (so that no
shipped scalar result moves).  The consequence is that a scalar `(Nx, Ny)`
cell and its `e * I` promotion -- which G1 asserts are the same
discretization, and which ARE bit-identical away from a cutoff -- take
DIFFERENT wavelength nudges when an order sits on a LAYER's own Rayleigh
cutoff, and therefore give different answers there.

The reproducer puts order m = 1 exactly on the cutoff of a uniform eps = 4
layer at normal incidence: kt = wl/px = 2 = sqrt(eps_layer).
"""
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)


def promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


CELL = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=complex)   # uniform, n = 2

for px, wl, note in ((0.5e-6, 1.0e-6, "ON the layer cutoff (wl/px = 2)"),
                     (0.5e-6, 1.0e-6 * (1 - 1e-9), "1e-9 off it"),
                     (0.7e-6, 0.55e-6, "far from any cutoff")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o1, R1, T1 = pmm_efficiency_2d_staggered(
            px, px, CELL, 1.5, 1.0, 0.28e-6, wl, degree=5, n_orders=3,
            polarization="tm")
        _o2, R2, T2, _J2 = pmm_jones_2d_staggered(
            px, px, promote(CELL), 1.5, 1.0, 0.28e-6, wl, degree=5,
            n_orders=3)
    print(f"px={px:.3e} wl={wl:.12e}  {note}\n"
          f"    max|dR| = {np.max(np.abs(R1 - R2[0])):.3e}   "
          f"max|dT| = {np.max(np.abs(T1 - T2[0])):.3e}   "
          f"bit-identical = {np.array_equal(R1, R2[0])}")
