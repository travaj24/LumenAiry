"""v5.8.0 Polynomial Modal Method (PMM, item G) -- 1-D lamellar grating.

A non-Fourier modal solver (subsectional spectral-element / Edee) that converges
spectrally in the polynomial degree with NO accuracy floor -- the property the
ASR coordinate stretch (v5.7) lacked.  Validation anchors:

* the FMM solver (``rcwa_efficiency_1d``) at high order is the convergence
  oracle (PMM must reach the SAME efficiencies),
* a uniform ABSORBING slab must match the FMM slab bit-for-bit (the public<->
  engineering convention discipline -- the worst pitfall during development),
* lossless gratings conserve energy, and PMM keeps converging (no floor).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

# gold-like lossy metal ridge (PUBLIC Im > 0) -- the canonical hard TM case
GOLD = dict(period=1.0e-6, n_ridge=0.2 + 6j, n_groove=1.0, n_substrate=1.0,
            n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5, wavelength=0.8e-6)


def _oracle_T(pol, M=80):
    o, R, T = rcwa_efficiency_1d(**GOLD, polarization=pol, n_orders=M)
    return float(T.sum())


def _pmm_T(pol, degree, **kw):
    o, R, T = pmm_efficiency_1d(**GOLD, polarization=pol, degree=degree, **kw)
    return float(T.sum())


# ---------------------------------------------------------------------------
# converges to the FMM oracle
# ---------------------------------------------------------------------------

def test_pmm_te_converges_to_fmm_oracle():
    assert abs(_pmm_T("te", 16) - _oracle_T("te")) < 5e-5


def test_pmm_tm_converges_to_fmm_oracle():
    assert abs(_pmm_T("tm", 28) - _oracle_T("tm")) < 5e-4


# ---------------------------------------------------------------------------
# NO FLOOR -- the decisive property over ASR.  Error improves with degree, and
# PMM self-converges (consecutive high degrees agree), i.e. no plateau.
# ---------------------------------------------------------------------------

def test_pmm_tm_error_improves_with_degree():
    ref = _oracle_T("tm")
    e_low = abs(_pmm_T("tm", 8) - ref)
    e_high = abs(_pmm_T("tm", 28) - ref)
    assert e_high < e_low                       # monotone improvement (no floor)


def test_pmm_self_converges_no_floor():
    # independent of the oracle's own residual: two high degrees must agree,
    # proving PMM keeps converging rather than plateauing with noise
    assert abs(_pmm_T("tm", 24) - _pmm_T("tm", 36)) < 1e-4
    assert abs(_pmm_T("te", 16) - _pmm_T("te", 28)) < 1e-5


def test_pmm_beats_fmm_at_matched_dof_te():
    # PMM-TE reaches a far lower error than uniform FMM at the same DOF
    ref = _oracle_T("te")
    degree = 12                                 # DOF ~ 2*degree = 24
    e_pmm = abs(_pmm_T("te", degree) - ref)
    of, Rf, Tf = rcwa_efficiency_1d(**GOLD, polarization="te", n_orders=degree)
    e_fmm = abs(float(Tf.sum()) - ref)
    assert e_pmm < e_fmm


# ---------------------------------------------------------------------------
# energy conservation (lossless) + the convention check (absorbing slab)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm_lossless_conserves_energy(pol):
    diel = dict(GOLD, n_ridge=2.5 + 0j)
    o, R, T = pmm_efficiency_1d(**diel, polarization=pol, degree=16)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm_absorbing_slab_matches_fmm(pol):
    # ridge == groove == an absorbing medium -> a plain slab (no grating); PMM
    # and FMM must agree to machine precision (validates the public-convention
    # end-to-end: a mixed convention makes an absorber give R+T > 1).
    slab = dict(period=1.0e-6, n_ridge=1.5 + 0.3j, n_groove=1.5 + 0.3j,
                n_substrate=1.0, n_superstrate=1.0, depth=0.4e-6,
                duty_cycle=0.5, wavelength=0.8e-6)
    o, Rp, Tp = pmm_efficiency_1d(**slab, polarization=pol, degree=10)
    o2, Rf, Tf = rcwa_efficiency_1d(**slab, polarization=pol, n_orders=15)
    assert abs(float(Rp.sum()) - float(Rf.sum())) < 1e-9
    assert abs(float(Tp.sum()) - float(Tf.sum())) < 1e-9
    assert float(Rp.sum() + Tp.sum()) < 1.0     # genuine absorption, not > 1


# ---------------------------------------------------------------------------
# mesh grading (the TM speed lever) helps, and never breaks correctness
# ---------------------------------------------------------------------------

def test_pmm_grading_helps_metal_tm():
    ref = _oracle_T("tm")
    e_graded = abs(_pmm_T("tm", 8, elements_per_region=3, grade=True) - ref)
    e_uniform = abs(_pmm_T("tm", 8, elements_per_region=3, grade=False) - ref)
    assert e_graded < e_uniform                 # corner clustering helps


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_pmm_rejects_oblique():
    with pytest.raises(NotImplementedError, match="normal incidence"):
        pmm_efficiency_1d(**GOLD, polarization="tm", degree=10, angle=0.2)


def test_pmm_rejects_bad_polarization():
    with pytest.raises(ValueError, match="polarization must be"):
        pmm_efficiency_1d(**GOLD, polarization="tem", degree=10)


def test_pmm_rejects_low_degree():
    with pytest.raises(ValueError, match="degree must be"):
        pmm_efficiency_1d(**GOLD, polarization="te", degree=1)


def test_pmm_exported_at_top_level():
    import lumenairy as la
    assert hasattr(la, "pmm_efficiency_1d")
