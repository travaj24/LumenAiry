"""Validate the BOR-PMM coupled (E_r,E_phi) vector eigensolver (Milestone 2)
against the open-cladding fiber oracle (itself validated vs the canonical
Okamoto/Snyder-Love characteristic equation).

Gates the THREE-part recipe: q^2 E_z-elimination + wall-normal inverse rule +
consistent divergence-free filter.  Confirms (a) the guided modes match the
oracle, and (b) the intrinsic vector-FEM spurious mode (q~3.76 on the V=4 case,
where the oracle |det| has NO root) is filtered out.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from coupled_radial_eigensolver import guided_modes, radial_coupled_modes
from fiber_oracle import fiber_modes


def _oracle_bound(m, a, e1, e2, k0):
    q = fiber_modes(m, a, e1, e2, k0)
    return q[q > np.sqrt(e2) * k0 + 1e-2]


def test_strong_guided_mode_matches_oracle():
    """The well-confined HE11-type mode (V=4) -> oracle to <1e-3."""
    m, a, e1, e2, k0 = 1, 1.0, 6.0, 2.0, 2.0
    qor = _oracle_bound(m, a, e1, e2, k0)
    gm = guided_modes(m, a, 8.0, 600, e1, e2, k0)
    assert len(gm) >= 1
    qtop = gm[0]["q"].real
    assert np.min(np.abs(qor - qtop)) < 1e-3


def test_multimode_fiber_matches_oracle():
    """V=6 (m=1): all three bound modes recovered, each within the FD floor."""
    m, a, e1, e2, k0 = 1, 1.0, 6.0, 2.0, 3.0
    qor = _oracle_bound(m, a, e1, e2, k0)
    gm = guided_modes(m, a, 8.0, 700, e1, e2, k0)
    qs = np.array([md["q"].real for md in gm])
    # every oracle bound mode has a solver mode within 3e-2 (2nd-order FD floor)
    for qo in qor:
        assert np.min(np.abs(qs - qo)) < 3e-2, f"missed oracle mode {qo}"


def test_spurious_mode_is_filtered():
    """The intrinsic vector-FEM spurious mode (q~3.76, NOT an oracle root) must
    be rejected by the divergence-free filter and absent from guided_modes."""
    m, a, e1, e2, k0 = 1, 1.0, 6.0, 2.0, 2.0
    gm = guided_modes(m, a, 8.0, 600, e1, e2, k0)
    qs = np.array([md["q"].real for md in gm])
    assert np.all(np.abs(qs - 3.76) > 0.1), "spurious q~3.76 leaked through"
    # and it IS present (high reldiv) in the unfiltered spectrum -> the filter
    # is doing real work, not vacuously passing
    allm = radial_coupled_modes(m, 8.0, 600,
                                lambda r: np.where(r <= a, e1, e2), k0)
    near = [md for md in allm if abs(md["q"].real - 3.76) < 0.05
            and abs(md["q"].imag) < 1e-3]
    assert near and near[0]["reldiv"] > 1.0


def test_spurious_divergence_separation():
    """Physical guided modes have small relative divergence; the spurious mode
    is O(1)+ -> a robust separation."""
    m, a, e1, e2, k0 = 1, 1.0, 6.0, 2.0, 2.0
    allm = radial_coupled_modes(m, 8.0, 600,
                                lambda r: np.where(r <= a, e1, e2), k0)
    win = [md for md in allm
           if np.sqrt(e2) * k0 + 1e-2 < md["q"].real < np.sqrt(e1) * k0 - 1e-2
           and abs(md["q"].imag) < 1e-3]
    phys = [md["reldiv"] for md in win if abs(md["q"].real - 4.4363) < 0.05]
    spur = [md["reldiv"] for md in win if abs(md["q"].real - 3.76) < 0.05]
    assert phys and spur
    assert max(phys) < 0.5 and min(spur) > 2.0
