"""v5.11.0 multi-region / multi-level 1-D anisotropic RCWA (GAP2).

``rcwa_jones_1d_segments`` generalises :func:`rcwa_jones_1d` (a binary ridge /
groove grating) to an ARBITRARY piecewise-constant 1-D profile: multi-level
staircases (blazed-grating approximations), multi-region cells, and mixed
isotropic / liquid-crystal regions.  It shares the EXACT same solve core
(``_jones_1d_from_profiles``) as ``rcwa_jones_1d``; only the per-component
profile sampling differs.  Both the in-plane (5-component) and the v5.11.0 full
out-of-plane (9-component) tensor paths are supported.

Validation gates (each an independent discriminator):
  (1)  BIT-IDENTITY  -- a 2-segment ``[(duty, ridge), (1-duty, groove)]`` cell
       reproduces ``rcwa_jones_1d(..., duty_cycle=duty)`` bit-for-bit (R, T,
       Jones) for isotropic, in-plane-tensor, and full-tensor ridges, at angles
       0 and 25 deg.
  (2)  MULTI-LEVEL energy + convergence  -- a 4-level isotropic staircase
       conserves energy (lossless R+T=1), a fine staircase whose adjacent
       levels are merged equals the coarser segmentation (segment-merge
       invariance), and the isotropic E_y (TE) channel reproduces the scalar
       ``rcwa_efficiency_1d`` TE solver.
  (3)  FULL-TENSOR segments  -- a 2-region tilted-LC (phi=0 vs phi=pi/2) grating
       conserves energy and is non-trivial (Jones off-diagonal != 0).
  (4)  VALIDATION  -- width fractions not summing to 1 raise; an empty /
       invalid segments list raises; mixed scalar + tensor segments work.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    uniaxial_tensor,
)

WL = 0.55e-6
DEPTH = 0.5e-6
PERIOD = 0.8e-6
DUTY = 0.4
N_SUB, N_SUP = 1.5, 1.0

RIDGE_ISO = (1.8 + 0.0j) ** 2 * np.eye(3)
GROOVE_ISO = (1.0 + 0.0j) ** 2 * np.eye(3)
# in-plane (no off-plane coupling): a theta=pi/2 uniaxial director.
RIDGE_INPLANE = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
GROOVE_INPLANE = (1.0 + 0.0j) ** 2 * np.eye(3)
# full out-of-plane: a tilted theta=pi/4 uniaxial director.
RIDGE_FULL = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)
GROOVE_FULL = (1.0 + 0.0j) ** 2 * np.eye(3)


# ---------------------------------------------------------------------------
# (1) BIT-IDENTITY: 2-segment cell == binary rcwa_jones_1d (R, T, Jones).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,ridge,groove", [
    ("iso", RIDGE_ISO, GROOVE_ISO),
    ("inplane", RIDGE_INPLANE, GROOVE_INPLANE),
    ("full", RIDGE_FULL, GROOVE_FULL),
])
@pytest.mark.parametrize("angle_deg", [0.0, 25.0])
def test_two_segment_equals_binary(name, ridge, groove, angle_deg):
    angle = np.deg2rad(angle_deg)
    o0, R0, T0, J0 = rcwa_jones_1d(
        PERIOD, ridge, groove, N_SUB, N_SUP, DEPTH, DUTY, WL,
        angle=angle, n_orders=9)
    o1, R1, T1, J1 = rcwa_jones_1d_segments(
        PERIOD, [(DUTY, ridge), (1.0 - DUTY, groove)], N_SUB, N_SUP, DEPTH, WL,
        angle=angle, n_orders=9)
    assert np.array_equal(o0, o1)
    assert np.max(np.abs(R1 - R0)) < 1e-13
    assert np.max(np.abs(T1 - T0)) < 1e-13
    assert np.max(np.abs(J1 - J0)) < 1e-13


# ---------------------------------------------------------------------------
# (2) MULTI-LEVEL energy + convergence.
# ---------------------------------------------------------------------------
def _staircase(values):
    """Equal-width isotropic staircase from a list of refractive indices."""
    w = 1.0 / len(values)
    return [(w, (n ** 2) * np.eye(3)) for n in values]


@pytest.mark.parametrize("angle_deg", [0.0, 10.0])
def test_multilevel_energy_conserved(angle_deg):
    # Lossless 4-level staircase in vacuum half-spaces -> R + T = 1.
    levels = _staircase([1.5, 1.7, 1.9, 2.1])
    o, R, T, J = rcwa_jones_1d_segments(
        PERIOD, levels, 1.0, 1.0, DEPTH, WL,
        angle=np.deg2rad(angle_deg), n_orders=15)
    total = R.sum(axis=1) + T.sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-9)


@pytest.mark.parametrize("angle_deg", [0.0, 12.0])
def test_segment_merge_invariance(angle_deg):
    # A staircase whose adjacent levels carry the SAME eps is identical to the
    # coarser segmentation (the fine sampling must not perturb the profile).
    e1 = (1.5 ** 2) * np.eye(3)
    e2 = (1.9 ** 2) * np.eye(3)
    fine = [(0.25, e1), (0.25, e1), (0.25, e2), (0.25, e2)]
    coarse = [(0.5, e1), (0.5, e2)]
    angle = np.deg2rad(angle_deg)
    _, Rf, Tf, Jf = rcwa_jones_1d_segments(
        PERIOD, fine, 1.0, 1.0, DEPTH, WL, angle=angle, n_orders=13)
    _, Rc, Tc, Jc = rcwa_jones_1d_segments(
        PERIOD, coarse, 1.0, 1.0, DEPTH, WL, angle=angle, n_orders=13)
    assert np.max(np.abs(Rf - Rc)) < 1e-12
    assert np.max(np.abs(Tf - Tc)) < 1e-12
    assert np.max(np.abs(Jf - Jc)) < 1e-12
    # And the coarse 2-segment cell equals the binary rcwa_jones_1d.
    _, Rb, Tb, Jb = rcwa_jones_1d(
        PERIOD, e1, e2, 1.0, 1.0, DEPTH, 0.5, WL, angle=angle, n_orders=13)
    assert np.max(np.abs(Rc - Rb)) < 1e-13
    assert np.max(np.abs(Tc - Tb)) < 1e-13


def test_isotropic_te_matches_scalar_solver():
    # For an isotropic grating the E_y (TE) Jones channel must reproduce the
    # scalar TE rcwa_efficiency_1d solver exactly (independent oracle).
    e1, e2 = (1.5 ** 2) * np.eye(3), (1.9 ** 2) * np.eye(3)
    _, R, T, J = rcwa_jones_1d_segments(
        PERIOD, [(0.5, e1), (0.5, e2)], 1.0, 1.0, DEPTH, WL,
        angle=0.0, n_orders=13)
    o, Rte, Tte = rcwa_efficiency_1d(
        PERIOD, 1.5, 1.9, 1.0, 1.0, DEPTH, 0.5, WL,
        angle=0.0, polarization="te", n_orders=13)
    # Row 1 of R/T is the incident-E_y (TE) response.
    assert np.allclose(R[1], Rte, atol=1e-12)
    assert np.allclose(T[1], Tte, atol=1e-12)


# ---------------------------------------------------------------------------
# (3) FULL-TENSOR segments: a 2-region tilted-LC grating.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle_deg", [0.0, 15.0])
def test_full_tensor_segments_energy_and_coupling(angle_deg):
    # Two regions of the SAME tilted (theta=pi/4) uniaxial LC but rotated in
    # azimuth (phi = 0 vs pi/2) -- a genuine full-3x3 multi-region cell.
    lc0 = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.0)
    lc1 = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=np.pi / 2)
    o, R, T, J = rcwa_jones_1d_segments(
        PERIOD, [(0.5, lc0), (0.5, lc1)], 1.0, 1.0, DEPTH, WL,
        angle=np.deg2rad(angle_deg), n_orders=13)
    total = R.sum(axis=1) + T.sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-8)
    # Non-trivial TE<->TM coupling: the off-diagonal Jones terms are non-zero.
    assert abs(J[0, 1]) > 1e-4
    assert abs(J[1, 0]) > 1e-4


# ---------------------------------------------------------------------------
# (4) VALIDATION.
# ---------------------------------------------------------------------------
def test_width_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to"):
        rcwa_jones_1d_segments(
            PERIOD, [(0.4, RIDGE_ISO), (0.4, GROOVE_ISO)],
            1.0, 1.0, DEPTH, WL)


def test_empty_segments_raises():
    with pytest.raises(ValueError, match="non-empty"):
        rcwa_jones_1d_segments(PERIOD, [], 1.0, 1.0, DEPTH, WL)


def test_negative_width_raises():
    with pytest.raises(ValueError, match="width_fraction"):
        rcwa_jones_1d_segments(
            PERIOD, [(-0.5, RIDGE_ISO), (1.5, GROOVE_ISO)],
            1.0, 1.0, DEPTH, WL)


def test_bad_eps_shape_raises():
    bad = np.eye(2)  # not (3, 3) and not a scalar
    with pytest.raises(ValueError):
        rcwa_jones_1d_segments(
            PERIOD, [(0.5, bad), (0.5, GROOVE_ISO)], 1.0, 1.0, DEPTH, WL)


def test_mixed_scalar_and_tensor_segments():
    # A scalar segment (promoted to scalar*I(3)) alongside an in-plane tensor.
    o, R, T, J = rcwa_jones_1d_segments(
        PERIOD, [(0.5, 2.5 + 0.0j), (0.5, RIDGE_INPLANE)],
        1.0, 1.0, DEPTH, WL, angle=0.0, n_orders=11)
    total = R.sum(axis=1) + T.sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-9)


def test_exported_at_top_level():
    assert hasattr(la, "rcwa_jones_1d_segments")
    assert "rcwa_jones_1d_segments" in la.__all__
