"""Probe 13b -- widen the search for a ROBUST engineered lossless closure
violation (the tripwire's fail-before arm needs decades, not a factor 1.6)."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)


def cell(host, pill, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pill
    return c


WL = 0.55e-6
best = []
for P in (0.7e-6, 1.4e-6, 2.1e-6):
    for dep in (0.28e-6, 0.9e-6):
        for ep in (16.0, 100.0):
            for M in (3, 4, 8):
                pill = ep * np.eye(3, dtype=complex)
                try:
                    o, R, T, J = pmm_jones_2d_staggered(
                        P, P, cell(GYRO, pill), 1.5, 1.0, dep, WL, degree=M,
                        n_orders=5)
                    dev = float(np.max(np.abs(R.sum(axis=1)
                                              + T.sum(axis=1) - 1.0)))
                except Exception as e:                    # noqa: BLE001
                    dev = float("nan")
                    print("RAISED", type(e).__name__, e)
                best.append((dev, P, dep, ep, M))
                print(f"P={P*1e6:.1f}um dep={dep*1e6:.2f}um eps={ep:5.1f} "
                      f"M={M}  |R+T-1| = {dev:.4e}")
