"""Oblique (off-normal) PMM -- scalar ``pmm_efficiency_1d`` and anisotropic
``pmm_jones_1d`` at ``angle != 0`` (the ``+i kx0`` Bloch shift).

Three physics fixes are exercised here:
  * the ANTISYMMETRIZED convection ``-i kx0 (Cw - Cw^T)`` (TM needs the wall-
    varying ``1/eps`` weight, NOT ``2 Cw``);
  * the NOISE-ROBUST / z-POYNTING-FLUX forward branch selector (the coupled
    tensor modes do not split by ``Im(q)`` at oblique);
  * the per-COLUMN incident-flux normalization (the p-pol ``1/cos^2`` factor).

Oracles: ``rcwa_efficiency_1d`` / ``rcwa_jones_1d`` at matched angle.  The
lossy-metal-TM steep-oblique corner case is resonance-limited and intentionally
NOT asserted (documented limit -> rcwa).  ``angle == 0`` must stay bit-identical
to the normal-incidence solve.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_jones_1d,
    uniaxial_tensor,
)

_C = np.complex128

DIEL = dict(period=0.8e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
            n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5, wavelength=0.55e-6)
LC = dict(period=0.8e-6, n_substrate=1.5, n_superstrate=1.0, depth=0.5e-6,
          duty_cycle=0.5, wavelength=0.55e-6)
LC_RIDGE = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)        # exy != 0
GROOVE = np.eye(3, dtype=_C)


def _align(o, R, T):
    return ({int(a): float(b) for a, b in zip(o, R)},
            {int(a): float(b) for a, b in zip(o, T)})


def _max_eff_err(o1, R1, T1, o2, R2, T2):
    dr1, dt1 = _align(o1, R1, T1)
    dr2, dt2 = _align(o2, R2, T2)
    keys = set(dr1) | set(dr2)
    return max(max(abs(dr1.get(k, 0) - dr2.get(k, 0)) for k in keys),
               max(abs(dt1.get(k, 0) - dt2.get(k, 0)) for k in keys))


# --------------------------------------------------------------------------- #
# scalar oblique
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("ang_deg", [10.0, 25.0, 40.0])
def test_scalar_oblique_matches_fmm(pol, ang_deg):
    ang = np.radians(ang_deg)
    o, R, T = pmm_efficiency_1d(**DIEL, polarization=pol, degree=20, angle=ang)
    orf, Rr, Tr = rcwa_efficiency_1d(**DIEL, polarization=pol, n_orders=31,
                                     angle=ang)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-4         # lossless
    assert _max_eff_err(o, R, T, orf, Rr, Tr) < 1e-3


def test_scalar_angle_zero_bit_identical():
    # angle=0 must be byte-identical to the normal-incidence solve (the Bloch
    # shift terms vanish and the legacy Im(q) branch is used).  stabilize=False
    # isolates the SOLVE path (no degree-scan heuristic in the comparison).
    for pol in ("te", "tm"):
        o0, R0, T0 = pmm_efficiency_1d(**DIEL, polarization=pol, degree=18,
                                       angle=0.0, stabilize=False)
        o, R, T = pmm_efficiency_1d(**DIEL, polarization=pol, degree=18,
                                    stabilize=False)
        assert np.array_equal(o0, o)
        assert np.array_equal(R0, R) and np.array_equal(T0, T)


def test_scalar_oblique_metal_te_robust():
    # metal TE is robust at oblique (only lossy-metal-TM-steep is limited)
    met = dict(period=0.6e-6, n_ridge=0.2 + 3.0j, n_groove=1.0, n_substrate=1.5,
               n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5,
               wavelength=0.55e-6)
    ang = np.radians(35.0)
    o, R, T = pmm_efficiency_1d(**met, polarization="te", degree=22, angle=ang)
    orf, Rr, Tr = rcwa_efficiency_1d(**met, polarization="te", n_orders=41,
                                     angle=ang)
    assert _max_eff_err(o, R, T, orf, Rr, Tr) < 2e-3


# --------------------------------------------------------------------------- #
# anisotropic (Jones) oblique
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ang_deg", [15.0, 30.0, 45.0, 60.0])
def test_jones_oblique_lossless_matches_fmm(ang_deg):
    ang = np.radians(ang_deg)
    o, R, T, J = pmm_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                              LC["n_substrate"], LC["n_superstrate"],
                              LC["depth"], LC["duty_cycle"], LC["wavelength"],
                              angle=ang, degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                                   LC["n_substrate"], LC["n_superstrate"],
                                   LC["depth"], LC["duty_cycle"],
                                   LC["wavelength"], angle=ang, n_orders=25)
    assert np.max(np.abs(J - Jo)) < 5e-4
    # both incident polarizations conserve energy (lossless)
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def test_jones_oblique_lossy_matches_fmm():
    lossy = uniaxial_tensor(1.5 + 0.02j, 1.8 + 0.05j, np.pi / 2, phi=0.3)
    ang = np.radians(30.0)
    o, R, T, J = pmm_jones_1d(LC["period"], lossy, GROOVE, LC["n_substrate"],
                              LC["n_superstrate"], LC["depth"],
                              LC["duty_cycle"], LC["wavelength"], angle=ang,
                              degree=20)
    oo, Ro, To, Jo = rcwa_jones_1d(LC["period"], lossy, GROOVE,
                                   LC["n_substrate"], LC["n_superstrate"],
                                   LC["depth"], LC["duty_cycle"],
                                   LC["wavelength"], angle=ang, n_orders=25)
    assert np.max(np.abs(J - Jo)) < 5e-4
    # PMM and FMM agree on the absorbed power (R+T deficit)
    assert np.max(np.abs((R.sum(1) + T.sum(1)) - (Ro.sum(1) + To.sum(1)))) < 5e-4


def test_jones_oblique_diagonal_decouples():
    # a diagonal (non-rotated) tensor has no Ex<->Ey coupling -> cross-pol ~0
    diag = np.diag([2.1, 2.4, 2.25]).astype(_C)
    o, R, T, J = pmm_jones_1d(LC["period"], diag, GROOVE, LC["n_substrate"],
                              LC["n_superstrate"], LC["depth"],
                              LC["duty_cycle"], LC["wavelength"],
                              angle=np.radians(30.0), degree=18)
    assert abs(J[0, 1]) < 1e-9 and abs(J[1, 0]) < 1e-9          # no cross-pol
    assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def test_jones_angle_zero_bit_identical():
    # stabilize=False isolates the SOLVE path (the angle=0 kx0=0 branch must be
    # byte-identical to the legacy normal-incidence tensor solve).
    o0, R0, T0, J0 = pmm_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                                  LC["n_substrate"], LC["n_superstrate"],
                                  LC["depth"], LC["duty_cycle"],
                                  LC["wavelength"], angle=0.0, degree=18,
                                  stabilize=False)
    o, R, T, J = pmm_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                              LC["n_substrate"], LC["n_superstrate"],
                              LC["depth"], LC["duty_cycle"], LC["wavelength"],
                              degree=18, stabilize=False)
    assert np.array_equal(o0, o)
    assert np.array_equal(R0, R) and np.array_equal(T0, T)
    assert np.array_equal(J0, J)


def test_jones_oblique_all_vacuum_unity():
    # the per-column p-pol flux normalization regression: a uniform vacuum
    # structure must give R=0, T=1 for BOTH incident polarizations at any angle
    # (col-0 over-counted by 1/cos^2 before the fix).
    for ang_deg in (20.0, 40.0):
        o, R, T, J = pmm_jones_1d(LC["period"], GROOVE, GROOVE, 1.0, 1.0,
                                  LC["depth"], LC["duty_cycle"],
                                  LC["wavelength"], angle=np.radians(ang_deg),
                                  degree=14)
        assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-9)
        assert np.max(np.abs(R)) < 1e-9                        # vacuum -> no R


def test_jones_oblique_converges_in_degree():
    ang = np.radians(30.0)
    oo, Ro, To, Jo = rcwa_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                                   LC["n_substrate"], LC["n_superstrate"],
                                   LC["depth"], LC["duty_cycle"],
                                   LC["wavelength"], angle=ang, n_orders=41)
    errs = []
    for deg in (10, 16, 22):
        _o, _R, _T, J = pmm_jones_1d(LC["period"], LC_RIDGE, GROOVE,
                                     LC["n_substrate"], LC["n_superstrate"],
                                     LC["depth"], LC["duty_cycle"],
                                     LC["wavelength"], angle=ang, degree=deg)
        errs.append(np.max(np.abs(J - Jo)))
    assert min(errs) < 1e-4               # converges to the high-order oracle
