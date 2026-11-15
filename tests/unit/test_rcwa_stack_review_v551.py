"""``rcwa_convergence`` and ``formulation='auto'`` contracts of the RCWA stack.

* An unreadable solver output: ``rcwa_convergence`` raises on a solver output
  it cannot read instead of skipping it with a broad ``except`` (a skipped
  output would leave that quantity out of the verdict silently).
* The LEGACY call form ``rcwa_convergence(..., bump=4, atol=1e-3)`` (an
  additive ``bump`` with ``rtol`` unset) keeps the verdict of releases up to
  5.50 exactly and emits no warning they did not emit; passing ``rtol`` opts
  into the error-estimate verdict.  Measured on a lossless lamellar grating
  of index 3.48 (period 900 nm, fill 0.55, 300 nm, 1550 nm, theta 0.3,
  Laurent), N = 20: 5.50.2 and this release both give ``converged=True``,
  ``delta`` 4.29e-4, no warning; with ``rtol=1e-2`` the estimate is 7.1e-2
  -> not converged (true error at N = 60: 1.07e-2).
* Dispersive layers: a layer left at ``'auto'`` runs the rule its concrete
  twin runs, at every wavelength: a y-varying cell of a 1-D stack
  (``'laurent'``; a rule that resolved the callable without evaluating it
  would run ``'li'``) and, on a 2-D stack, a metal layer (``'li'`` both ways)
  and an exact-zero background (``'laurent'`` both ways).
* ``rcwa_convergence`` on an adaptive-resolution stack with ``eta='auto'``
  freezes the stretch at the base rung for every rung, so the rungs differ
  only in truncation.
* ``RCWAStack.add_layer(segments=)`` validates through the library's one
  segment validator (PMMStack's).
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import RCWAStack, rcwa_convergence, rcwa_jones_1d

WL = 1550e-9


def _hi_stack(N):
    st = RCWAStack(900e-9, n_superstrate=1.0, n_substrate=1.45, n_orders=N)
    st.add_layer(300e-9, segments=[(0.55, 3.48 ** 2), (0.45, 1.0)],
                 formulation="laurent")
    return st.set_source(WL, theta=0.3)


# ---------------------------------------------------------------- unreadable output
class _Unreadable:
    def __array__(self, *a, **k):
        raise ValueError("this output cannot be read as an array")


def test_convergence_does_not_drop_an_unreadable_output():
    def solver(**kw):
        o, R, T, *_ = rcwa_jones_1d(**kw)
        return o, R, T, _Unreadable()

    with pytest.raises(ValueError, match="cannot be read"):
        rcwa_convergence(solver, period=900e-9, eps_ridge=3.48 ** 2,
                         eps_groove=1.0, n_substrate=1.45, n_superstrate=1.0,
                         depth=300e-9, duty_cycle=0.55, wavelength=WL,
                         angle=0.3, n_orders=6)


# ---------------------------------------------------------------- legacy bump
def test_legacy_bump_call_keeps_the_legacy_verdict_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _r, rep = rcwa_convergence(_hi_stack(20), bump=4, atol=1e-3)
    assert rep["converged"] is True
    assert rep["delta"] == pytest.approx(4.29e-4, rel=2e-2)
    assert rep["n_orders_high"] == {"n_orders": 24}
    # the estimate is still reported; asking for it judges by it
    assert rep["err_estimate"] == pytest.approx(7.12e-2, rel=2e-2)
    with pytest.warns(UserWarning, match="NOT converged"):
        _r, rep2 = rcwa_convergence(_hi_stack(20), bump=4, atol=1e-3,
                                    rtol=1e-2)
    assert rep2["converged"] is False
    # a legacy call whose atol fails warns, as releases up to 5.50 did
    with pytest.warns(UserWarning, match="atol"):
        _r, rep3 = rcwa_convergence(_hi_stack(20), bump=4, atol=1e-5)
    assert rep3["converged"] is False
    # legacy default atol is 1e-3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _r, rep4 = rcwa_convergence(_hi_stack(20), bump=4)
    assert rep4["converged"] is True


# ---------------------------------------------------------------- dispersive
def _yvar_cell():
    cell = np.ones((64, 8), complex)
    cell[10:30, :4] = 3.5 ** 2
    return cell


def test_dispersive_y_varying_cell_on_a_1d_stack_runs_laurent_like_its_twin():
    cell = _yvar_cell()

    def jones(eps_cell, **kw):
        st = RCWAStack(800e-9, n_superstrate=1.0, n_substrate=1.45,
                       n_orders=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.add_layer(100e-9, eps_cell=eps_cell, **kw)
            st.set_source(WL, theta=0.2)
            if callable(eps_cell):
                return np.asarray(st.solve_vs_wavelength([WL])[3][0])
            return np.asarray(st.solve().jones_reflection())

    j_conc = jones(cell)
    j_disp = jones(lambda wl: cell)
    # the rule a resolution that does not evaluate the callable would run
    j_li = jones(lambda wl: cell, formulation="li")
    # the sweep and the point solve differ at round-off only (measured
    # 1e-16); the two rules differ at 1e-2
    assert np.max(np.abs(j_conc - j_disp)) < 1e-12
    assert np.max(np.abs(j_li - j_disp)) > 1e-4


def _metal_2d(eps, bg=1.0):
    st = RCWAStack(600e-9, period_y=600e-9, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=3, n_orders_y=3)
    st.add_layer(60e-9, shapes=[dict(shape="rectangle", eps=eps,
                                     size=(300e-9, 300e-9),
                                     center=(300e-9, 300e-9))],
                 eps_background=bg)
    st.add_layer(150e-9, eps=2.1)
    return st.set_source(1.31e-6, theta=0.1)


def test_2d_dispersive_layer_runs_the_rule_of_its_concrete_twin():
    # a metal rectangle: 'li' concretely, and now in the sweep too
    j_conc = np.asarray(_metal_2d(-3 + 1j).solve().jones_reflection())
    j_disp = np.asarray(_metal_2d(lambda wl: -3 + 1j)
                        .solve_vs_wavelength([1.31e-6])[3][0])
    st_la = _metal_2d(-3 + 1j)
    st_la._layers[0].formulation = "laurent"     # the other rule, forced
    j_la = np.asarray(st_la.solve().jones_reflection())
    assert np.max(np.abs(j_conc - j_disp)) < 1e-12
    assert np.max(np.abs(j_la - j_disp)) > 1e-4


def test_2d_dispersive_cell_with_an_exact_zero_runs_laurent():
    cell = np.ones((32, 32), complex)
    cell[8:24, 8:24] = 0.0
    st_c = RCWAStack(600e-9, period_y=600e-9, n_substrate=1.5, n_orders=2,
                     n_orders_y=2)
    st_c.add_layer(60e-9, eps_cell=cell)
    assert st_c.layers[0].formulation == "laurent"
    st_d = RCWAStack(600e-9, period_y=600e-9, n_substrate=1.5, n_orders=2,
                     n_orders_y=2)
    st_d.add_layer(60e-9, eps_cell=lambda wl: cell)
    st_c.set_source(1.31e-6)
    st_d.set_source(1.31e-6)
    assert np.max(np.abs(
        np.asarray(st_c.solve().jones_reflection())
        - np.asarray(st_d.solve_vs_wavelength([1.31e-6])[3][0]))) < 1e-12


# ---------------------------------------------------------------- asr stretch
def test_convergence_freezes_the_auto_stretch_across_rungs():
    # a metal strip on a single-layer spacer (fixture replaced by a generic
    # one; the assertions are fixture-independent bit-for-bit comparisons)
    def stack(asr, N):
        st = RCWAStack(500e-9, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=N, asr=asr)
        st.add_layer(40e-9, segments=[(180 / 500, -90 + 7j),
                                      (320 / 500, 1.0)])
        st.add_layer(19e-9, eps=3.6)
        return st.set_source(1500e-9, theta=0.2)

    st = stack(True, 8)                       # eta='auto' -> 0.3 at N = 8
    top, rep = rcwa_convergence(st, warn=False)
    assert rep["asr_eta"] == pytest.approx(0.3)
    assert st.asr["eta"] == "auto"            # restored
    n_hi = rep["n_orders_high"]["n_orders"]   # 12, where 'auto' gives 0.7
    ref = stack(dict(eta=0.3), n_hi).solve()
    assert np.array_equal(np.asarray(top.jones_reflection()),
                          np.asarray(ref.jones_reflection()))


# ---------------------------------------------------------------- segments
def test_segments_validate_through_the_shared_validator(monkeypatch):
    import lumenairy.elements.pmm._core as pc
    calls = []
    orig = pc._segment_walls

    def spy(period, widths):
        calls.append((period, list(widths)))
        return orig(period, widths)

    monkeypatch.setattr(pc, "_segment_walls", spy)
    RCWAStack(1e-6, n_orders=3).add_layer(
        0.1e-6, segments=[(0.4, 4.0), (0.6, 1.0)])
    assert calls == [(1.0, [0.4, 0.6])]
    with pytest.raises(ValueError, match="finite"):
        RCWAStack(1e-6, n_orders=3).add_layer(
            0.1e-6, segments=[(float("nan"), 4.0), (1.0, 1.0)])
