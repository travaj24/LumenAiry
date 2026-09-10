"""Is ``test_jones_2d_even_sector_matches_full``'s WSL failure ours?

The WSL suite run of 2026-09-11 reported ONE failure,
``tests/unit/test_v5_14_2_backlog_batch.py::test_jones_2d_even_sector_matches_full``
-- an ``rcwa_jones_2d`` even-parity-fold comparison, on a build whose LAPACK
also printed ``** On entry to DLASCL parameter number 4 had an illegal value``.
It passed on WIN.

``rcwa_jones_2d`` calls NONE of the three functions this branch changed
(``grep -c _interface_smatrix_general lumenairy/elements/rcwa/twod*.py`` == 0;
the branch's ``_guarded_inverse`` change is a no-op for every caller that passes
no ``rcond_refuse`` while ``_INV_CENSUS is None``), so the attribution has to be
MEASURED rather than argued.  This script reproduces the test body verbatim
against WHICHEVER tree ``PYTHONPATH`` points at, so it can be run on the branch
tip AND on ``lum_slfix_pre`` (the branch point) on the same interpreter.
"""
from __future__ import annotations

import _lib as L
import numpy as np

from lumenairy.elements.rcwa import rcwa_jones_2d

_P, _WL = 0.5e-6, 0.6e-6


def main():
    a = L.arm()
    S = 48
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c0, s0 = np.cos(0.7), np.sin(0.7)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 0] = ne2 * c0 * c0 + no2 * s0 * s0
    tc[m, 1, 1] = ne2 * s0 * s0 + no2 * c0 * c0
    tc[m, 0, 1] = tc[m, 1, 0] = (ne2 - no2) * c0 * s0
    tc[m, 2, 2] = no2
    full = rcwa_jones_2d(_P, _P, tc, 1.5, 1.0, 0.2e-6, _WL, n_orders_x=5,
                         n_orders_y=5, symmetry=False)
    even = rcwa_jones_2d(_P, _P, tc, 1.5, 1.0, 0.2e-6, _WL, n_orders_x=5,
                         n_orders_y=5, symmetry=True)
    dR = float(np.max(np.abs(full[1] - even[1])))
    dJ = float(np.max(np.abs(full[3] - even[3])))
    print("%-8s %-22s dR %.6e   dJones %.6e   bar 1e-08 -> %s"
          % (a["build"], a["tree"], dR, dJ,
             "PASS" if (dR < 1e-8 and dJ < 1e-8) else "FAIL"))


if __name__ == "__main__":
    main()
