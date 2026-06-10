"""v5.14.0 -- ``stabilize=`` degree-scan consensus for the 2-D hybrid PMM.

The 1-D PMM's per-order convergence consensus (``_stabilize_scalar`` /
``_stabilize_jones``) wraps the 2-D solvers, stepping through consecutive ODD
degrees.  Pins: a clean base degree returns the per-order answer the consensus
confirms (matching the direct solve to the consensus tolerance), the Jones
variant works, and the scan respects the odd-degree constraint (no even-degree
ValueError escapes)."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.pmm import (
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_jones_2d,
)

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6


def _cell():
    cell = np.full((6, 6), 1.0 + 0j)
    cell[1:3, 1:3] = 6.0
    cell[4:6, 4:5] = 4.0
    return cell


def test_stabilize_scalar_cell_consistent_with_direct():
    cell = _cell()
    o1, R1, T1 = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                        degree=7, n_orders=4)
    res = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                 degree=7, n_orders=4, stabilize=True)
    o2, R2, T2 = res
    assert np.array_equal(o1, o2)
    # the consensus picks the requested degree when it is already converged
    # (per-order agreement within the scan tolerance), so the answers agree
    # to the consensus's own per-order tolerance at worst
    assert np.max(np.abs(T1 - T2)) < 5e-3
    assert res.dof == 2 * 9 ** 2


def test_stabilize_pillar_entry_runs():
    o, R, T = pmm_efficiency_2d(_P, _P, 6.0, 1.0, (0.2 * _P, 0.6 * _P),
                                (0.2 * _P, 0.6 * _P), 1.5, 1.0, _DEP, _WL,
                                degree=7, n_orders=4, stabilize=True)
    # the consensus's energy gate is the 2-D passive tolerance (5e-2); the
    # measured floor at these coarse settings is ~2.2e-2
    assert abs(float(R.sum() + T.sum()) - 1.0) < 5e-2


def test_stabilize_jones_consistent_with_direct():
    cell = _cell()
    tc = np.zeros((6, 6, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = cell
    o1, R1, T1, J1 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=7, n_orders=4)
    o2, R2, T2, J2 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=7, n_orders=4, stabilize=True)
    assert np.array_equal(o1, o2)
    assert np.max(np.abs(T1 - T2)) < 5e-3
    assert np.max(np.abs(J1 - J2)) < 5e-2
