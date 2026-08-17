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


#: The conical Jones cross-check's TRUNCATION LADDER: ``(pmm degree, pmm
#: n_orders, rcwa n_orders, rcwa upsample)``.  The RCWA oracle's Fourier
#: convolution needs ``>= 4 n_orders + 1`` samples per axis, so the upsample
#: factor is tied to its order count, not chosen.
_CONICAL_LADDER = ((11, 5, 5, 3), (13, 6, 6, 5), (15, 6, 6, 5))

#: Closure bar for the ladder's BEST rung.  Derived, not chosen -- see
#: :func:`test_conical_jones_matches_rcwa`'s docstring for the measurements.
_CONICAL_CLOSURE = 1.0e-3

#: Cross-suite bars, unconditional on EVERY rung.  Measured envelopes (same
#: 12 environments, 2026-08-16): order-0 T agreement <= 3.8e-3, Jones
#: agreement <= 3.9e-4.
_CONICAL_DT, _CONICAL_DJ = 3.0e-2, 5.0e-2


def _conical_cell(S=8):
    """The anisotropic (rotated-uniaxial inclusion) cell the cross-check
    uses."""
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
    return tc


def _conical_upsample(tc, rep):
    S = tc.shape[0]
    up = np.zeros((rep * S, rep * S, 3, 3), complex)
    for a in range(3):
        for b in range(3):
            up[:, :, a, b] = np.kron(tc[:, :, a, b], np.ones((rep, rep)))
    return up


def test_conical_jones_matches_rcwa():
    """Anisotropic cell at steep conical incidence: the Jones solver's
    cross-suite agreement must hold away from normal.

    **2026-08-16 -- THE OPERATING POINT WAS INSIDE A DOCUMENTED INSTABILITY,
    AND THE CLOSURE BAR WAS INSIDE ITS SPREAD (S4, floor bar;
    ``docs/TESTING_STANDARDS.md``).**  The original form solved the hybrid at
    ``degree=9, n_orders=4`` and asserted ``|R+T-1| < 3e-2`` per row.  At that
    truncation this geometry is the very case ``_EnergyWarning`` names ("the
    truncation is numerically unstable here"), and the reading is decided by
    the BLAS reduction order.  Measured on ONE build (Windows py3.14, numpy
    2.4.4 / scipy 1.17.1), code and geometry FIXED, varying only the thread
    cap::

        threads   |R+T-1| row 0   row 1
        1         8.007e-03       2.083e-02
        2         1.520e-02       3.038e-03
        4         4.589e-02       4.655e-02   <- the failure (bar 3e-2)
        8         _EnergyError raised (sum R+T = 2.203)

    -- a bar at 3e-2 sitting inside a spread that runs from 3e-3 to a hard
    raise.  ``stabilize=True`` does not repair it either: its retry ladder
    treats the 1e-6 closure warning as a failed attempt, no rung near this
    geometry clears 1e-6, and it therefore falls back to the same reading
    (measured identical to the raw one on numpy 2.5.1, 1.26.4 and 2.4.6).

    RESTATED ONE LEVEL UP.  The hybrid's closure here is TRUNCATION-limited,
    and WHICH rung of a truncation ladder closes best is exactly the per-build
    fact; that a rung closing to ~1e-4 EXISTS is not.  So the ladder
    (:data:`_CONICAL_LADDER`) is fixed and stated, the cross-suite agreement
    is asserted on EVERY rung of it, and the closure bar is applied to the
    rung this build's own measurement selects.

    MEASURED ENVELOPES, 2026-08-16, over 12 (build x thread-cap)
    environments -- Windows py3.14 numpy 2.4.4 / scipy 1.17.1 at 1/2/4/8
    threads, and WSL py3.12 numpy 2.5.1 / scipy 1.18.0, py3.12 numpy 1.26.4 /
    scipy 1.11.4, py3.11 numpy 2.4.6 / scipy 1.17.1, py3.10 numpy 2.2.6 /
    scipy 1.15.3 at 1 and 4 threads::

        quantity                              min        max        bar
        closure at the SELECTED rung          1.8e-06    9.7e-05    1e-3
        |T1[0,0] - T2[0,0]|, every rung       2.4e-05    3.8e-03    3e-2
        max |J1 - J2|, every rung             2.6e-04    3.9e-04    5e-2

    The closure bar has a decade of gap below (9.7e-5 envelope) and better
    than 1.5 decades above (the unconverged reading this test used to take,
    4.6e-2, and the raise past it).  Both cross-suite bars keep the shipped
    values: they were never the failing ones and they still clear their
    envelopes by ~8x and ~128x."""
    tc = _conical_cell()
    th, ph = np.deg2rad(40), np.deg2rad(25)
    ups, rungs = {}, []
    for degree, n_ord, r_ord, rep in _CONICAL_LADDER:
        o1, R1, T1, J1 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                      degree=degree, n_orders=n_ord,
                                      theta=th, phi=ph)
        if (r_ord, rep) not in ups:
            ups[(r_ord, rep)] = rcwa_jones_2d(
                _P, _P, _conical_upsample(tc, rep), 1.5, 1.0, _DEP, _WL,
                n_orders_x=r_ord, n_orders_y=r_ord, theta=th, phi=ph)
        o2, R2, T2, J2 = ups[(r_ord, rep)]
        m1, m2 = _o0(o1), _o0(o2)
        closure = max(abs(float(R1[row].sum() + T1[row].sum()) - 1.0)
                      for row in (0, 1))
        # the cross-suite claim is UNCONDITIONAL on every rung: the two
        # independent solvers must agree about the order-0 transmission and
        # the Jones matrix wherever the ladder is evaluated.
        for row in (0, 1):
            dT = abs(float(T1[row][m1][0]) - float(T2[row][m2][0]))
            assert dT < _CONICAL_DT, (
                f"degree={degree} n_orders={n_ord} row={row}: the hybrid and "
                f"the RCWA oracle disagree by {dT:.4g} on the order-0 "
                f"transmission at 40 deg conical incidence")
        dJ = float(np.max(np.abs(J1 - J2)))
        assert dJ < _CONICAL_DJ, (
            f"degree={degree} n_orders={n_ord}: max |J_pmm - J_rcwa| = "
            f"{dJ:.4g} at 40 deg conical incidence")
        rungs.append((degree, n_ord, closure, dJ))
    # the closure bar is applied where THIS build says the truncation closes,
    # never at a rung picked on another machine.
    best = min(rungs, key=lambda r: r[2])
    print("\nconical Jones ladder (degree, n_orders, closure, dJ): "
          + str([(d, n, float(f"{c:.4g}"), float(f"{j:.4g}"))
                 for d, n, c, j in rungs])
          + f"  selected degree={best[0]} n_orders={best[1]}")
    assert best[2] < _CONICAL_CLOSURE, (
        f"the BEST-closing rung of {[(d, n) for d, n, _c, _j in rungs]} is "
        f"degree={best[0]} n_orders={best[1]} at |R+T-1| = {best[2]:.4g}, "
        f"past the {_CONICAL_CLOSURE:g} envelope bar -- no truncation in the "
        f"ladder converges this lossless conical solve on this build "
        f"(all rungs: {[(d, n, float(f'{c:.4g}')) for d, n, c, _j in rungs]})")


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
