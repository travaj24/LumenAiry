"""Probe 13 -- engineer a LOSSLESS closure violation for the tripwire's
fail-before arm (a ladder through the public API, not a hoped-for state)."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

WL, P, DEP = 0.55e-6, 0.70e-6, 0.28e-6
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)


def cell(host, pill, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pill
    return c


for ep in (4.0, 16.0, 36.0, 100.0):
    pill = ep * np.eye(3, dtype=complex)
    for M in (3, 4, 5, 8):
        try:
            o, R, T, J = pmm_jones_2d_staggered(P, P, cell(GYRO, pill), 1.5,
                                                1.0, DEP, WL, degree=M,
                                                n_orders=4)
            dev = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
            print(f"eps_pillar={ep:6.1f} M={M}  |R+T-1| = {dev:.4e}")
        except Exception as e:                            # noqa: BLE001
            print(f"eps_pillar={ep:6.1f} M={M}  RAISED {type(e).__name__}: {e}")
