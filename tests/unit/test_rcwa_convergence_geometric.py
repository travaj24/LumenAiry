"""``rcwa_convergence``: geometric step, relative comparison, error estimate.

The truncation goes ``N -> max(N + 1, ceil(f N))`` (``f`` = 1.5 for one axis,
1.25 per axis for two); every per-order efficiency is compared RELATIVE to
``max(R_m, 1e-3 sum R)`` and every zeroth-order Jones element relative to
``max(|J|, sqrt(1e-3 sum R))``; the largest change ``d`` gives the p = 1 error
bound ``d / (f - 1)`` that is judged against ``rtol``.  ``estimate=True`` adds
a third solve and a per-quantity local exponent (erratic outside [0.5, 4] or
when successive changes turn around).

Fixture A (see ``test_rcwa_stack_auto_formulation.py``), ``'li'``, analytic
strip; reference reflectance ``|r_ref|^2 = 0.02049``.  Rungs measured
2026-11-14 (Windows/MKL; the plan's re-check reproduced them):

    N     R          relative R error
    40    0.016606   18.9 %
    60    0.015668   23.5 %
    61    0.016051   21.6 %
    90    0.017272   15.7 %
    92    0.017301   15.5 %
    135   0.018640    9.0 %
    138   0.018276   10.8 %

The additive check of earlier releases (``bump=4``, ``atol=1e-3``) passed at
N = 61 with the reflectance 23.5 % wrong.
"""
import warnings

import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa import RCWAStack, rcwa_convergence
from lumenairy.elements.rcwa._core import _max_aligned_delta

WL, P, W = 1300e-9, 370e-9, 175e-9
AU, AL, AO, HF, OX, NS = -74 + 6j, -160 + 30j, 3.05, 3.53, 2.5 + 0.5j, 1.447
LAYERS = [(50e-9, AU, 1.0), (1e-9, AO, AO), (7e-9, HF, HF), (5e-9, OX, OX),
          (7e-9, HF, HF), (1e-9, AO, AO), (70e-9, AL, AL)]
R_REF = abs(0.0571572 + 0.1312207j) ** 2
STRIP = [dict(shape="rectangle", eps=AU, center=(W / 2, P / 2), size=(W, P))]


def _fix_a(N):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=N)
    for t, es, eg in LAYERS:
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, shapes=STRIP, eps_background=eg, formulation="li")
    return st.set_source(WL, theta=0.0)


def _r0(res):
    return float(np.asarray(res.efficiencies()[1])[0].sum())


def test_n61_is_not_converged_although_the_additive_check_passed():
    st = _fix_a(61)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res, rep = rcwa_convergence(st)
    assert rep["n_orders_high"] == {"n_orders": 92}
    assert rep["converged"] is False
    assert any("NOT converged" in str(x.message) for x in w)
    # measured: relative R change 7.2 % over a x1.508 step -> estimate 14.2 %
    # (0.1421 on Windows/MKL and WSL/OpenBLAS); true error at N = 92: 15.5 %.
    # The estimate is within 2x of the truth.
    true_err = abs(_r0(res) - R_REF) / R_REF
    assert 0.5 * true_err < rep["err_estimate"] < 2.0 * true_err
    assert rep["factor"] == pytest.approx(92 / 61)
    assert st.nox == 61                                    # restored
    # the absolute test alone (what the additive check applied) passes here
    _res, legacy = rcwa_convergence(st, bump=4, warn=False)
    assert legacy["delta"] <= 1e-3                         # measured 3.8e-4
    assert legacy["n_orders_high"] == {"n_orders": 65}
    assert abs(_r0(_res) - R_REF) / R_REF > 0.2            # 23.5 % wrong


def test_n90_decision_pair():
    """Estimate 14.7 % (true error at 135: 9.0 %): rtol = 0.3 converges,
    rtol = 1e-2 does not."""
    st = _fix_a(90)
    _r, loose = rcwa_convergence(st, rtol=0.3, warn=False)
    _r, tight = rcwa_convergence(st, rtol=1e-2, warn=False)
    assert loose["converged"] is True
    assert tight["converged"] is False
    assert 0.05 < loose["err_estimate"] < 0.3


def _smooth(N):
    """A sinusoidal-profile dielectric grating: exponential convergence."""
    S = 512
    x = np.arange(S) / S
    cell = (2.25 + 0.75 * np.cos(2 * np.pi * x)).astype(complex)[:, None]
    st = RCWAStack(1.0e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=N)
    st.add_layer(0.4e-6, eps_cell=cell)
    return st.set_source(0.8e-6, theta=0.2)


def test_smooth_grating_is_converged_at_its_working_n():
    """Measured: relative change 2.4e-13 at N = 8 (round-off)."""
    _r, rep = rcwa_convergence(_smooth(8))
    assert rep["converged"] is True
    assert rep["delta_rel"] < 1e-6


def test_bump_reproduces_the_additive_step_bit_for_bit():
    st = _fix_a(20)
    hi, rep = rcwa_convergence(st, bump=4, warn=False)
    lo = st.solve()
    st.nox = 24
    try:
        ref_hi = st.solve()
    finally:
        st.nox = 20
    o_lo, R_lo, T_lo = lo.efficiencies()
    o_hi, R_hi, T_hi = ref_hi.efficiencies()
    delta = max(_max_aligned_delta(o_lo, R_lo, o_hi, R_hi),
                _max_aligned_delta(o_lo, T_lo, o_hi, T_hi))
    assert rep["n_orders_high"] == {"n_orders": 24}
    assert rep["delta"] == delta
    assert np.asarray(hi.efficiencies()[1]).tobytes() == \
        np.asarray(R_hi).tobytes()


def test_estimate_flags_the_erratic_window_and_measures_the_exponent():
    _r, rep40 = rcwa_convergence(_fix_a(40), estimate=True, warn=False)
    # rungs 40 / 60 / 90: R 0.016606 / 0.015668 / 0.017272 -- the two changes
    # have opposite signs
    assert rep40["erratic"] is True
    assert rep40["converged"] is False
    assert rep40["n_orders_mid"] == {"n_orders": 60}
    assert rep40["n_orders_high"] == {"n_orders": 90}
    res, rep61 = rcwa_convergence(_fix_a(61), estimate=True, warn=False)
    # rungs 61 / 92 / 138: the reflectance's local exponent is 0.57 and its
    # Richardson estimate 20 % (0.5748 / 0.2033 on both builds; true error at
    # 138: 10.8 %).  (The reflection
    # AMPLITUDE's exponent is 0.45, below the algebraic range, so the window
    # is also flagged erratic -- not asserted: the decision is the point.)
    assert rep61["converged"] is False
    assert rep61["n_orders_high"] == {"n_orders": 138}
    assert 0.5 <= rep61["exponent"] <= 0.7
    true_err = abs(_r0(res) - R_REF) / R_REF
    assert 0.5 * true_err < rep61["err_estimate"] < 3.0 * true_err
    # the smooth grating: already at round-off, nothing to judge
    _r, reps = rcwa_convergence(_smooth(8), estimate=True)
    assert reps["converged"] is True
    assert reps["erratic"] is False


def test_2d_stack_steps_both_axes_by_1p25_and_restores():
    S = 48
    xx = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(xx, xx, indexing="ij")
    cell = np.where((np.abs(X) < 0.25) & (np.abs(Y) < 0.2), 2.5 + 0j, 1.0 + 0j)
    st = lum.RCWAStack(0.6e-6, period_y=0.6e-6, n_superstrate=1.0,
                       n_substrate=1.5, n_orders=6, n_orders_y=4)
    st.add_layer(0.2e-6, eps_cell=cell)
    st.set_source(0.633e-6, theta=0.05)
    _r, rep = rcwa_convergence(st, warn=False)
    assert rep["n_orders_low"] == {"n_orders": 6, "n_orders_y": 4}
    assert rep["n_orders_high"] == {"n_orders": 8, "n_orders_y": 5}
    assert rep["factor"] == pytest.approx(min(8 / 6, 5 / 4))
    assert (st.nox, st.noy) == (6, 4)


def test_undersampled_cell_is_refused_before_the_higher_solve():
    S = 25                                # exactly 4 N + 1 for N = 6
    x = np.arange(S) / S
    st = RCWAStack(1e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=6)
    st.add_layer(0.2e-6, eps_cell=np.where(x < 0.5, 4.0, 1.0).astype(complex))
    st.set_source(0.8e-6)
    with pytest.raises(ValueError, match="rcwa_convergence"):
        rcwa_convergence(st)
    assert st.nox == 6


def test_argument_validation():
    st = _smooth(4)
    with pytest.raises(ValueError, match="not both"):
        rcwa_convergence(st, factor=1.5, bump=4)
    with pytest.raises(ValueError, match="factor"):
        rcwa_convergence(st, factor=1.0)
    with pytest.raises(ValueError, match="bump"):
        rcwa_convergence(st, bump=0)
    with pytest.raises(ValueError, match="rtol"):
        rcwa_convergence(st, rtol=0.0)


def test_callable_compares_returned_jones_amplitudes():
    from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_jones_1d
    kw = dict(period=1e-6, n_substrate=1.5, n_superstrate=1.0, depth=0.3e-6,
              duty_cycle=0.4, wavelength=0.8e-6)
    _o, rep = rcwa_convergence(rcwa_efficiency_1d, n_ridge=0.2 + 5j,
                               n_groove=1.0, polarization="tm", n_orders=10,
                               warn=False, **kw)
    assert rep["delta_amp"] == 0.0                        # no amplitudes
    assert rep["n_orders_high"] == {"n_orders": 15}
    out, rep = rcwa_convergence(rcwa_jones_1d,
                                eps_ridge=(0.2 + 5j) ** 2 * np.eye(3),
                                eps_groove=np.eye(3), n_orders=10,
                                warn=False, **kw)
    assert len(out) >= 4                    # (orders, R, T, jones_reflection)
    assert rep["delta_amp"] > 0.0
