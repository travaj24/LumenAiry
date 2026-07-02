"""Audit 2026-07-01 P1-03 -- 1-D PMM gain/evanescent incidence-medium guard.

Pre-fix, NO 1-D PMM entry point called ``rcwa._core._require_propagating_
incidence`` (the v5.14.1 suite-wide RCWA audit-P1 guard, already used by every
2-D PMM path): a gain superstrate (public ``Im(n_sup) < 0``, even ``-1e-6``)
flipped the forward ``kz_inc`` root and silently NEGATED every efficiency
(measured ``sum(T) = -12.52`` on pmm_efficiency_1d, ``-53.95`` on the covariant
slanted path, all with R masked to 0 and zero warnings), and an evanescent /
metallic superstrate silently returned ``tot = 44.4``.  These tests FAIL on
the pre-fix code (no ValueError was raised; verified by re-running the audit
probe on the unfixed tree) and pass with the guard at the four shared far-field
kz_inc choke points in ``pmm/_core.py``.  The baseline pins prove the guard is
inert for valid (lossless, absorbing-GRATING, and lossy-superstrate) inputs --
A/B verified byte-identical pre/post fix across 13 solve paths.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_1d_segments,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_segments,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
)

WL, P, DEPTH, DUTY = 1.0e-6, 2.0e-6, 0.3e-6, 0.5
GAIN = 1.0 - 1e-6j          # public n = n - ik: the classic sign-slip
EYE3 = np.eye(3)
SLANT = np.deg2rad(20.0)


# ---------------------------------------------------------------------------
# Gain superstrate must raise on EVERY 1-D entry point / dispatch route
# ---------------------------------------------------------------------------

def test_gain_superstrate_raises_efficiency_1d():
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d(P, 2.0, 1.0, 1.5, GAIN, DEPTH, DUTY, WL,
                          polarization="te", degree=10, stabilize=False)


def test_gain_superstrate_raises_through_stabilize():
    # The stabilize scan must PROPAGATE the guard's ValueError, not launder it
    # into a 'resonance band' RuntimeError or a silent negative-T consensus
    # (pre-fix: stabilize=True returned sum(T) = -12.51 with zero warnings).
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d(P, 2.0, 1.0, 1.5, GAIN, DEPTH, DUTY, WL,
                          polarization="te", degree=10, stabilize=True)


def test_gain_superstrate_raises_jones_1d():
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_jones_1d(P, 4.0 * EYE3, EYE3, 1.5, GAIN, DEPTH, DUTY, WL,
                     degree=10, stabilize=False)


def test_gain_superstrate_raises_segments():
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d_segments(
            P, [(0.5, 2.0), (0.5, 1.0)], 1.5, GAIN, DEPTH, WL,
            polarization="te", degree=10, stabilize=False)


def test_gain_superstrate_raises_jones_segments():
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_jones_1d_segments(
            P, [(0.5, 4.0 * EYE3), (0.5, EYE3)], 1.5, GAIN, DEPTH, WL,
            degree=10, stabilize=False)


def test_gain_superstrate_raises_slanted_scalar():
    # Normal incidence + slant -> the dedicated inclined-coordinate scalar
    # solver (_pmm_slant_solve -> _scalar_farfield_RT choke point).
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d_slanted(
            P, 2.0, 1.0, 1.5, GAIN, DEPTH, DUTY, WL, SLANT,
            polarization="te", degree=10, stabilize=False)


def test_gain_superstrate_raises_slanted_oblique_route():
    # Combined oblique + slant -> delegated to pmm_jones_1d_slanted (covariant).
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_1d_slanted(
            P, 2.0, 1.0, 1.5, GAIN, DEPTH, DUTY, WL, SLANT,
            angle=np.deg2rad(10.0), polarization="te", degree=10,
            stabilize=False)


def test_gain_superstrate_raises_jones_slanted_covariant():
    # Covariant path: eps is conjugated to the INTERNAL exp(+iwt) gauge before
    # the far field -- the guard there takes the internal eps WITHOUT a conj.
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_jones_1d_slanted(
            P, 4.0 * EYE3, EYE3, 1.5, GAIN, DEPTH, DUTY, WL, SLANT,
            degree=10, stabilize=False, factorization="covariant")


def test_gain_superstrate_raises_jones_slanted_segments():
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_jones_1d_slanted_segments(
            P, [(0.5, 4.0 * EYE3), (0.5, EYE3)], 1.5, GAIN, DEPTH, WL,
            SLANT, degree=10, stabilize=False)


# ---------------------------------------------------------------------------
# Evanescent / metallic superstrate must raise (pre-fix: silent tot = 44.4)
# ---------------------------------------------------------------------------

def test_evanescent_superstrate_raises():
    with pytest.raises(ValueError, match="non-propagating"):
        pmm_efficiency_1d(P, 2.0, 1.0, 1.5, 0.1 + 3.0j, DEPTH, DUTY, WL,
                          polarization="te", degree=10, stabilize=False)


# ---------------------------------------------------------------------------
# Valid inputs must be untouched (guard fires only on the incidence MEDIUM)
# ---------------------------------------------------------------------------

def test_lossless_baseline_unchanged():
    o, R, T = pmm_efficiency_1d(P, 2.0, 1.0, 1.5, 1.0, DEPTH, DUTY, WL,
                                polarization="te", degree=10, stabilize=False)
    np.testing.assert_allclose(float(R.sum()), 0.07627557545827729, rtol=1e-9)
    np.testing.assert_allclose(float(T.sum()), 0.9229230722109744, rtol=1e-9)


def test_absorbing_grating_still_allowed():
    # Loss in the GRATING (ridge), lossless superstrate: must not raise.
    o, R, T = pmm_efficiency_1d(P, 2.0 + 0.5j, 1.0, 1.5, 1.0, DEPTH, DUTY,
                                WL, polarization="tm", degree=10,
                                stabilize=False)
    np.testing.assert_allclose(float(R.sum() + T.sum()),
                               0.5588062711904791, rtol=1e-9)


def test_lossy_superstrate_still_allowed():
    # Im(n_sup) > 0 (loss, public convention) remains supported -- the conj
    # into the guard's internal convention must NOT misread loss as gain.
    # (Documented caveat: the incident flux decays, so tot can exceed 1 for
    # strongly lossy superstrates; use stabilize=False there.)
    o, R, T = pmm_efficiency_1d(P, 2.0, 1.0, 1.5, 1.0 + 1e-3j, DEPTH, DUTY,
                                WL, polarization="te", degree=10,
                                stabilize=False)
    tot = float(R.sum() + T.sum())
    np.testing.assert_allclose(tot, 0.9994146084176007, rtol=1e-9)


def test_gain_substrate_not_guarded_matches_rcwa():
    # rcwa's guard covers only the INCIDENCE medium; a gain SUBSTRATE is left
    # to the energy tripwires there, and PMM mirrors that scope exactly.  This
    # pins the parity choice: no incidence-guard ValueError on the substrate.
    o, R, T = pmm_efficiency_1d(P, 2.0, 1.0, 1.5 - 1e-6j, 1.0, DEPTH, DUTY,
                                WL, polarization="te", degree=10,
                                stabilize=False)
    assert float(R.sum()) > 1.0        # nonphysical, caught by energy checks
