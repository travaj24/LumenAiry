"""Self-convergence of the PURE staggered OUT-OF-PLANE arm on the hardest cell
in the suite: a (3, 3) L with a RE-ENTRANT (270-degree) corner.

This is the ONE claim of the converged-reference study
``docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_REFERENCE_2026_09_10.md`` that is
two-sided, derived, and cheap enough to own a test.  Everything else that study
establishes is a cross-ENGINE comparison whose bar would be a value from
another build's Fourier ladder -- exactly the cross-build pin
``docs/TESTING_STANDARDS.md`` forbids -- so it stays in the report and in the
``validation/probe_pmm2d_staggered_oop_reference/`` ladders.

The claim, both halves measured on THIS build:

* the ladder is NOT already converged at the bottom rung (otherwise "the steps
  shrink" is a statement about roundoff);
* every successive step shrinks by at least a factor of 2, i.e. the
  discretization error is genuinely decaying and not sitting on a floor.

The quantity is the per-order TRANSMITTED efficiency, both incident
polarizations, because a total is mirror-invariant and would hide exactly the
per-order structure this cell exists to stress.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

PX = PY = 1.2e-6
WL = 1.0e-6
DEP = 0.4e-6
NSUB, NSUP = 1.5, 1.0


def l_cell():
    """Three tilted-uniaxial pixels forming an L: the only shape in the suite
    with a re-entrant corner, and the one the out-of-plane build left as an
    open item."""
    er = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = np.eye(3)
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[i, j] = er
    return e


def test_staggered_oop_corner_ladder_steps_shrink_and_start_above_the_floor():
    """M = 4, 5, 6, 7 on the re-entrant-corner out-of-plane cell.

    MEASURED on this build 2026-09-10 (reference-study table C1, probe
    ``t1_ladders.py``; the full ladder runs to M = 10 there): the per-order
    ``T`` steps are 2.51e-04 (4 -> 5), 6.39e-05 (5 -> 6), 9.26e-06 (6 -> 7),
    i.e. shrink factors of 3.9x and 6.9x, and 27x end to end.  Both bars below
    are derived from those numbers with the headroom stated:

    * ``> 1e-05`` on the first step -- 25x under the measured 2.51e-04, and
      four decades above the ~1e-09 a converged fixture would show, so the
      "not already converged" half cannot pass on noise;
    * ``<= 0.5`` on each successive step RATIO -- 2.0x and 3.5x of headroom
      under the measured 0.25 and 0.14.

    These are DISCRETIZATION errors of a deterministic polynomial basis, not
    build noise: the cross-build spread of a fixed-M solve is at the 1e-15
    level (the block-reduction build measured 4.9e-15 between two ALGORITHMS
    on the same pencil), ten decades below the smallest step compared here.
    """
    cell = l_cell()
    vals = []
    for M in (4, 5, 6, 7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, T, _J = pmm_jones_2d_staggered(
                PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M, n_orders=5)
        vals.append(np.asarray(T))
    steps = [float(np.max(np.abs(vals[i + 1] - vals[i])))
             for i in range(len(vals) - 1)]
    # (a) the fixture is genuinely un-converged at the bottom of the ladder
    assert steps[0] > 1e-5, steps
    # (b) every successive step at least halves
    for i in range(len(steps) - 1):
        assert steps[i + 1] <= 0.5 * steps[i], steps
    # (c) and the ladder as a whole has fallen by more than a decade
    assert steps[-1] <= 0.1 * steps[0], steps
