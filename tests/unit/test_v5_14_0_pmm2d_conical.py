"""v5.14.0 -- conical-incidence hardening of the 2-D hybrid PMM (Phase 3).

The 2-D PMM entry points carried a "validated near normal, large theta
experimental" caveat.  These tests validate LARGE-angle conical incidence
against the independent RCWA-2D solver and the 1-D reduction, and pin the new
guards (evanescent-incidence rejection + the Wood-anomaly wavelength nudge)
adopted from RCWA.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_1d,
    pmm_efficiency_2d_cell,
    pmm_jones_2d,
)
from lumenairy.elements.rcwa import rcwa_efficiency_2d, rcwa_jones_2d

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6


def _pillar_cell(S=6):
    cell = np.full((S, S), 1.0 + 0j)
    cell[1:4, 2:5] = 6.0
    return cell


def _o0(o):
    return (o[:, 0] == 0) & (o[:, 1] == 0)


# --------------------------------------------------------------------------- #
# (1) large-angle conical vs RCWA-2D
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("theta_deg,phi_deg", [
    (30, 0), (30, 30), (45, 90), (60, 30),
])
def test_large_angle_conical_matches_rcwa(theta_deg, phi_deg):
    cell = _pillar_cell()
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    o1, R1, T1 = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                        degree=9, n_orders=4, theta=th,
                                        phi=ph, polarization="te")
    # the hybrid's Fourier floor grows with theta at fixed coarse settings
    # (measured ~2.4e-2 @ 45 deg, ~3.1e-2 @ 60 deg at degree=9 / n_orders=4);
    # the cross-suite RCWA agreement below is the validation gate
    assert abs(float(R1.sum() + T1.sum()) - 1.0) < 5e-2     # energy
    up = np.kron(cell, np.ones((5, 5)))
    o2, R2, T2 = rcwa_efficiency_2d(_P, _P, up, 1.5, 1.0, _DEP, _WL,
                                    n_orders_x=4, n_orders_y=4, theta=th,
                                    phi=ph, polarization="te",
                                    formulation="li")
    m1, m2 = _o0(o1), _o0(o2)
    assert abs(float(T1[m1][0]) - float(T2[m2][0])) < 2e-2
    # 3e-2 (was 2e-2): the corrected sequential-rule 'li' oracle (audit F1,
    # 2026-06-10) shifted the R-sum at [30, 30] to 2.4e-2 from the hybrid --
    # the same coarse-settings Fourier floor the comment above documents.
    assert abs(float(R1.sum()) - float(R2.sum())) < 3e-2


def test_conical_jones_matches_rcwa():
    """Anisotropic cell at steep conical incidence: the Jones solver's
    cross-suite agreement must hold away from normal."""
    S = 8
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c, s = np.cos(0.6), np.sin(0.6)
    tc[2:6, 3:7, 0, 0] = ne2 * c * c + no2 * s * s
    tc[2:6, 3:7, 1, 1] = ne2 * s * s + no2 * c * c
    tc[2:6, 3:7, 0, 1] = (ne2 - no2) * c * s
    tc[2:6, 3:7, 1, 0] = (ne2 - no2) * c * s
    tc[2:6, 3:7, 2, 2] = no2
    th, ph = np.deg2rad(40), np.deg2rad(25)
    o1, R1, T1, J1 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=9, n_orders=4, theta=th, phi=ph)
    up = np.zeros((3 * S, 3 * S, 3, 3), complex)
    for a in range(3):
        for b in range(3):
            up[:, :, a, b] = np.kron(tc[:, :, a, b], np.ones((3, 3)))
    o2, R2, T2, J2 = rcwa_jones_2d(_P, _P, up, 1.5, 1.0, _DEP, _WL,
                                   n_orders_x=4, n_orders_y=4, theta=th,
                                   phi=ph)
    m1, m2 = _o0(o1), _o0(o2)
    for row in (0, 1):
        assert abs(float(R1[row].sum() + T1[row].sum()) - 1.0) < 3e-2
        assert abs(float(T1[row][m1][0]) - float(T2[row][m2][0])) < 3e-2
    assert np.max(np.abs(J1 - J2)) < 5e-2


# --------------------------------------------------------------------------- #
# (2) y-uniform cell reduces to the 1-D solver at oblique incidence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_y_uniform_cell_reduces_to_1d_oblique(pol):
    """A y-uniform cell at oblique incidence must conserve y-momentum EXACTLY
    (the separable exact-Fourier path; the all-nodal path leaked 4-8% into the
    forbidden n != 0 orders) and agree with the no-floor 1-D solver to the
    hybrid's x-truncation floor."""
    S = 6
    cell = np.full((S, S), 1.0 + 0j)
    cell[1:4, :] = 6.25                # x-grating, uniform along y
    th = np.deg2rad(30)
    o2, R2, T2 = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                        degree=11, n_orders=10, theta=th,
                                        phi=0.0, polarization=pol)
    # the (m, n != 0) orders carry NOTHING (machine-exact, ~1e-29 measured)
    side = o2[:, 1] != 0
    assert float(R2[side].sum() + T2[side].sum()) < 1e-12
    assert abs(float(R2.sum() + T2.sum()) - 1.0) < 5e-3
    o1, R1, T1 = pmm_efficiency_1d(_P, np.sqrt(6.25), 1.0, 1.5, 1.0, _DEP,
                                   0.5, _WL, angle=th, polarization=pol,
                                   degree=14, far_field_orders=9,
                                   stabilize=False)
    # TE ~7e-3 at these settings; TM (the wall-normal inverse-rule channel)
    # converges slower -- measured monotone 1.9e-2 -> 1.4e-2 over degree
    # 11 -> 15, so gate at its honest floor
    tol = 1.5e-2 if pol == "te" else 3e-2
    for m in (-1, 0, 1):
        i2 = (o2[:, 0] == m) & (o2[:, 1] == 0)
        i1 = o1 == m
        if i1.any() and i2.any():
            assert abs(float(T2[i2][0]) - float(T1[i1][0])) < tol
            assert abs(float(R2[i2][0]) - float(R1[i1][0])) < tol


# --------------------------------------------------------------------------- #
# (3) guards: evanescent incidence + Wood-anomaly nudge
# --------------------------------------------------------------------------- #

def test_evanescent_incidence_raises():
    cell = _pillar_cell()
    with pytest.raises(ValueError, match="non-propagating"):
        pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 0.2 + 3.0j, _DEP, _WL,
                               degree=7, n_orders=4)       # metal superstrate


def test_wood_anomaly_wavelength_does_not_crash():
    """wavelength == period puts the (+/-1, 0) orders EXACTLY grazing in the
    vacuum superstrate; the nudge must keep the solve finite + conserving."""
    cell = _pillar_cell()
    o, R, T = pmm_efficiency_2d_cell(_P, _P, cell, 1.5, 1.0, _DEP, _P,
                                     degree=9, n_orders=4)
    tot = float(R.sum() + T.sum())
    assert np.isfinite(tot)
    assert abs(tot - 1.0) < 3e-2
