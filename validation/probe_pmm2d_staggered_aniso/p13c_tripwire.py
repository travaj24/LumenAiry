"""Probe 13c -- near-Rayleigh-cutoff route to an engineered lossless closure
violation (the documented ~1/sqrt(distance) staggered degradation)."""
import warnings

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
ISO = 16.0 * np.eye(3, dtype=complex)
P, DEP = 0.70e-6, 0.28e-6
c = np.empty((2, 2, 3, 3), dtype=complex)
c[:] = GYRO
c[0, 0] = ISO

for frac in (1e-4, 1e-6, 1e-8, 1e-10):
    wl = P * 1.5 * (1.0 - frac)          # n_substrate = 1.5 cutoff
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o, R, T, J = pmm_jones_2d_staggered(P, P, c, 1.5, 1.0, DEP, wl,
                                            degree=6, n_orders=4)
    dev = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
    msgs = sorted({str(w.message).split(":")[1].strip()[:44] for w in rec})
    print(f"1-frac={frac:8.0e}  |R+T-1| = {dev:.4e}   warnings={msgs}")
