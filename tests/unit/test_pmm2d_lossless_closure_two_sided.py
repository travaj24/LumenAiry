"""The 2-D hybrid's lossless-closure tripwire must be TWO-SIDED about 1.0.

Found while measuring the 2-D crossed-PMM accuracy floor
(``docs/audits/EXPERIMENT_PMM2D_FEEC_2026_08_17.md``): ``_warn_lossless_energy_2d``
ESTABLISHES that every permittivity is real -- so ``R+T = 1`` is exact -- and
then applied the passivity window ``-tol <= R+T <= 1+tol`` inherited from
siblings that never establish losslessness.  A near-singular Fourier-projection
coincidence drove ``pmm_efficiency_2d`` to ``R+T = 0.8953`` (10.5 % energy LOSS,
per-order efficiencies ~2x wrong) and it returned SILENTLY, because 0.8953 sits
inside that window.

Per ``docs/TESTING_STANDARDS.md``: every claim below asserts a DECISION, the
pathological state is ENGINEERED through the API (or synthesized directly)
rather than hoped for from the build, and the one solver-driven test asserts an
INVARIANT tying the warning to the predicate on every arm -- so nothing pins a
number this campaign happened to read.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm.twod import (
    _PASSIVE_TOL_2D,
    _warn_lossless_energy_2d,
    pmm_efficiency_2d,
)

P, WL, H = 0.9e-6, 1.55e-6, 0.3e-6
NSUB, NSUP = 1.5, 1.0


def _fake_result(tot, n=4, eff_min=0.0):
    """A synthetic (orders, R, T) whose totals sum to `tot` exactly, with all
    per-order efficiencies >= `eff_min`.  Engineering the state directly is the
    only way to make the DEFICIT claim unconditional on the build."""
    R = np.zeros(n, dtype=float)
    T = np.zeros(n, dtype=float)
    R[0] = eff_min
    T[0] = tot - eff_min
    orders = np.zeros((n, 2), dtype=int)
    return orders, R, T


REAL_EPS = (4.0, 1.0, 1.0, 2.25)          # provably lossless
LOSSY_EPS = (4.0 + 0.1j, 1.0, 1.0, 2.25)  # must be skipped


def _warns_for(result, eps=REAL_EPS):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_lossless_energy_2d(result, eps, "probe")
    return [x for x in w if "energy closure" in str(x.message)]


# --- the decision, both sides -----------------------------------------------
@pytest.mark.parametrize("dev", [-0.5, -0.2, -0.1047, -2 * _PASSIVE_TOL_2D,
                                 2 * _PASSIVE_TOL_2D, 0.2, 0.5])
def test_gross_closure_violation_warns_on_both_sides(dev):
    """A provably-lossless structure off closure by more than the tolerance must
    warn, whether energy was LOST or MANUFACTURED.  -0.1047 is the measured
    real-solver deficit that used to return silently."""
    got = _warns_for(_fake_result(1.0 + dev))
    assert got, f"no warning for a lossless solve with R+T = {1.0 + dev}"
    assert "provably lossless" in str(got[0].message)


@pytest.mark.parametrize("dev", [0.0, 0.5 * _PASSIVE_TOL_2D,
                                 -0.5 * _PASSIVE_TOL_2D])
def test_within_tolerance_stays_silent(dev):
    """Inside the tolerance the guard must stay quiet -- a clean solve is
    byte-unchanged and must not acquire a warning."""
    assert not _warns_for(_fake_result(1.0 + dev))


def test_deficit_is_the_arm_that_regressed():
    """FAIL-BEFORE, stated as the difference between the two predicates: the OLD
    passivity window admits the measured 0.8953 reading, the NEW closure test
    rejects it.  Both are evaluated here, so this test carries its own
    refutation and cannot silently become vacuous."""
    tot = 0.895254364          # measured, cell C, degree=11, n_orders=15
    old_predicate_fires = not (-_PASSIVE_TOL_2D <= tot <= 1.0 + _PASSIVE_TOL_2D)
    new_predicate_fires = abs(tot - 1.0) > _PASSIVE_TOL_2D
    assert not old_predicate_fires, "the old window would have caught it"
    assert new_predicate_fires, "the new closure test must catch it"
    assert _warns_for(_fake_result(tot)), "the shipped guard must catch it"


def test_lossy_input_is_still_skipped():
    """A lossy structure legitimately absorbs, so R+T < 1 is not a defect and
    must not warn -- the two-sided change must not break the loss escape."""
    assert not _warns_for(_fake_result(0.5), eps=LOSSY_EPS)
    assert not _warns_for(_fake_result(0.895254364), eps=LOSSY_EPS)


def test_negative_per_order_still_warns():
    """The per-order non-negativity arm is unchanged by the two-sided fix."""
    assert _warns_for(_fake_result(1.0, eff_min=-2 * _PASSIVE_TOL_2D))


# --- tied to real solves ----------------------------------------------------
def test_solver_warning_matches_the_predicate_on_every_arm():
    """INVARIANT on the real solver over a (degree, n_orders) ladder: a warning
    is emitted if and only if the running build's own (R+T, min-efficiency)
    reading violates the closure gate.  This pins the wiring without pinning any
    value -- every arm is checked on whatever numbers this build produces, so
    LAPACK spread cannot move the pass/fail boundary (it moves both sides
    together)."""
    checked = 0
    for deg in (9, 11):
        for n_orders in (5, 7, 9, 11, 13, 15):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    o, R, T = pmm_efficiency_2d(
                        P, P, 12.25, 1.0, (0.25 * P, 0.5 * P),
                        (0.25 * P, 0.5 * P), NSUB, NSUP, H, WL,
                        degree=deg, n_orders=n_orders, polarization="te")
                except ValueError:
                    continue          # representability refusal: not this claim
                warned = any("energy closure" in str(x.message) for x in w)
            tot = float(np.real(np.sum(R)) + np.real(np.sum(T)))
            eff_min = min(float(np.min(np.real(R))), float(np.min(np.real(T))))
            expect = (abs(tot - 1.0) > _PASSIVE_TOL_2D
                      or eff_min < -_PASSIVE_TOL_2D)
            assert warned == expect, (
                f"degree={deg} n_orders={n_orders}: R+T={tot!r} "
                f"eff_min={eff_min!r} -> warned={warned}, expected {expect}")
            checked += 1
    assert checked >= 8, f"ladder collapsed to {checked} arms; claim is vacuous"
