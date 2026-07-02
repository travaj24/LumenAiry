"""Wave-5 audit fixes: PMM/RCWA cluster (v5.17.0 deep audit).

* P2-13 -- ``PMMStack.layer_absorption(by_material=True)`` dropped a layer
  whose ONLY loss is ``Im(ezz)`` (the per-material density omitted the
  ``Im(ezz)|Ez|^2`` channel): the dict silently missed the layer's correct,
  nonzero flux absorption.  Now the ezz channel is integrated per element
  (the ``internal_field`` Ez reconstruction) and the dict closes against the
  flux totals.
* P2-12 -- ``solve(retain_internal=True)`` rebuilt the full downstream
  Redheffer chain per layer (O(nlay^2) stars); now the linear reverse
  recurrence (the ``RCWAStack._internal_partials`` pattern).  Internal
  fields agree with the old association to float round-off (~1e-15 measured).
* P2-09 -- the stabilize passive gate was ONE-SIDED (``tot <= 1 + tol``): a
  systematically-wrong NEGATIVE-total solve repeated at consecutive degrees
  formed a bogus 'converged cluster' and was returned with ZERO warnings.
  Now two-sided + per-order non-negativity.
* P2-18 -- ``RCWAStack.solve(symmetry=True)`` stored RECENTERING-GAUGE
  per-order amplitudes (efficiencies gauge-invariant, but
  ``per_order_amplitudes`` / ``to_multiorder_field`` phases were those of a
  laterally SHIFTED structure).  The gauge is now undone before storing;
  amplitudes match the ``symmetry=False`` path.
* NEW (wave-3 follow-on) -- ``rcwa/oned.py`` entry points shared the
  back-side-angle hole wave-3 closed for pmm (``sin^2(100 deg) = 0.97``
  slips the evanescent-incidence guard and aliases to the supplementary
  front-side angle): now the same ``|angle| < pi/2`` ``ValueError``.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import _core as pmm_core
from lumenairy.elements.pmm import stack as pmm_stack_mod
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_vs_wavelength,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    rcwa_jones_vs_wavelength,
)

_WL = 0.55e-6
_P = 0.8e-6


# ---------------------------------------------------------------------------
# P2-13: ezz-only-lossy layers are attributed, not dropped
# ---------------------------------------------------------------------------

def _ezz_only_stack():
    st = PMMStack(_P, n_substrate=1.5, degree=10)
    st.add_layer(0.3e-6, eps=np.diag([2.0, 2.0, 2.0 + 1.0j]))
    st.set_source(_WL, angle=0.6)
    st.solve(retain_internal=True)
    return st


def test_p2_13_ezz_only_layer_attributed():
    st = _ezz_only_stack()
    A, mat = st.layer_absorption(by_material=True)
    assert A.sum() > 0.1                      # the layer genuinely absorbs
    assert mat, "ezz-only-lossy layer dropped from the per-material dict"
    tot = sum(v.sum() for v in mat.values())
    assert abs(tot - A.sum()) < 1e-12         # dict closes against flux A


def test_p2_13_mixed_inplane_and_ezz_split():
    """A layer mixing an in-plane-lossy and an ezz-only-lossy segment must
    key BOTH materials (the old density attributed 100% to the in-plane
    one), still summing to the exact flux total."""
    st = PMMStack(_P, n_substrate=1.5, degree=10)
    st.add_layer(0.3e-6, segments=[
        (0.5, np.diag([2.0 + 0.05j, 2.0 + 0.05j, 2.0])),
        (0.5, np.diag([2.0, 2.0, 2.0 + 1.0j]))])
    st.set_source(_WL, angle=0.6)
    st.solve(retain_internal=True)
    A, mat = st.layer_absorption(by_material=True)
    assert len(mat) == 2                      # both lossy materials keyed
    tot = sum(v.sum() for v in mat.values())
    assert abs(tot - A.sum()) < 1e-12
    # the ezz-only material (keyed by its exx = 2+0j) carries a REAL share
    ez_key = [k for k in mat if abs(complex(k).imag) < 1e-14][0]
    assert mat[ez_key].sum() > 0.01


def test_p2_13_isotropic_split_unchanged_invariants():
    """Regression guard: the isotropic-lossy split still closes and keys as
    before (the pre-fix tests' invariants)."""
    st = PMMStack(_P, n_substrate=1.5, degree=10)
    st.add_layer(0.15e-6, segments=[(0.5, 4.0 + 2.0j), (0.5, 1.0)])
    st.set_source(_WL).solve(retain_internal=True)
    A, mat = st.layer_absorption(by_material=True)
    assert complex(4.0 + 2.0j) in mat
    tot = sum(v.sum() for v in mat.values())
    assert abs(tot - A.sum()) < 1e-12


# ---------------------------------------------------------------------------
# P2-12: retain_internal partial cascades are O(nlay), not O(nlay^2)
# ---------------------------------------------------------------------------

def test_p2_12_linear_star_count_and_closure(monkeypatch):
    nlay = 8
    counts = {"n": 0}
    orig = pmm_stack_mod._redheffer_star

    def counting(*a, **k):
        counts["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pmm_stack_mod, "_redheffer_star", counting)
    st = PMMStack(_P, n_substrate=1.5, degree=5)
    for _ in range(nlay):
        st.add_layer(0.05e-6, segments=[(0.5, 4.0 + 0.5j), (0.5, 1.0)])
    st.set_source(_WL, angle=0.1)
    o, R, T, _ = st.solve(retain_internal=True)
    # base cascade 2*nlay, S_above 2*(nlay-1), S_below_bot 2*(nlay-1):
    # linear = 6*nlay - 4 = 44; the old quadratic rebuild needed 86.
    assert counts["n"] <= 6 * nlay - 4, (
        f"retain_internal used {counts['n']} Redheffer stars for nlay={nlay} "
        f"(quadratic rebuild?)")
    # physics unchanged: absorption still closes against the far field.
    # (degree=5 is deliberately coarse for speed, so the closure is
    # CONVERGENCE-limited at ~2e-3 -- the recurrence itself matches the old
    # per-layer rebuild to ~1e-15, measured in the wave-5 A/B probe.)
    A = st.layer_absorption()
    budget = 1.0 - R.sum(axis=1) - T.sum(axis=1)
    assert np.max(np.abs(A.sum(axis=0) - budget)) < 5e-3


# ---------------------------------------------------------------------------
# P2-09: stabilize passive gate rejects negative totals / efficiencies
# ---------------------------------------------------------------------------

def test_p2_09_scalar_gate_rejects_negative_total():
    orders = np.array([-1, 0, 1])

    def neg(_d):
        return orders, np.zeros(3), np.array([-6.0, -6.5, 0.0])  # tot = -12.5

    with pytest.raises(RuntimeError):
        pmm_core._stabilize_scalar(neg, 10, "w5_probe")


def test_p2_09_scalar_gate_rejects_negative_per_order():
    """Total inside [0, 1] but ONE order negative: also non-physical."""
    orders = np.array([-1, 0, 1])

    def mixed(_d):
        return orders, np.zeros(3), np.array([-0.2, 0.9, 0.1])   # tot = 0.8

    with pytest.raises(RuntimeError):
        pmm_core._stabilize_scalar(mixed, 10, "w5_probe")


def test_p2_09_scalar_gate_still_accepts_passive():
    """Control: a physical (lossless) repeated solve is returned silently."""
    orders = np.array([-1, 0, 1])
    R = np.array([0.05, 0.10, 0.05])
    T = np.array([0.20, 0.50, 0.10])                             # tot = 1.0

    def good(_d):
        return orders, R, T

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o, Rr, Tr = pmm_core._stabilize_scalar(good, 10, "w5_probe")
    assert np.allclose(Rr, R) and np.allclose(Tr, T)


def test_p2_09_jones_gate_flags_negative_total():
    orders = np.array([-1, 0, 1])

    def neg(_d):
        R = np.zeros((2, 3))
        T = np.array([[-0.9, -0.5, 0.0]] * 2)                    # tot = -2.8
        return orders, R, T, np.eye(2, dtype=complex)

    with pytest.warns(UserWarning, match="no energy-passive"):
        pmm_core._stabilize_jones(neg, 10, "w5_probe")


# ---------------------------------------------------------------------------
# P2-18: symmetry=True per-order amplitudes match the full-solve gauge
# ---------------------------------------------------------------------------

def _offcentre_stack(sym):
    S = 32
    cell = np.ones((S, S), dtype=complex)
    c0, half = 10, 6                          # pillar centred OFF the origin
    for i in range(S):
        for j in range(S):
            if abs((i - c0 + S // 2) % S - S // 2) <= half and \
               abs((j - c0 + S // 2) % S - S // 2) <= half:
                cell[i, j] = 6.0
    stk = RCWAStack(1.2e-6, period_y=1.2e-6, n_superstrate=1.0,
                    n_substrate=1.5, n_orders=3, n_orders_y=3)
    stk.add_layer(0.2e-6, eps_cell=cell)
    stk.add_layer(0.15e-6, eps_cell=0.5 * (cell + 2.0))
    stk.set_source(1.0e-6)
    return stk.solve(symmetry=sym)


def test_p2_18_symmetry_amplitude_phases_match_full_solve():
    rT = _offcentre_stack(True)
    rF = _offcentre_stack(False)
    _oT, RT, TT = rT.efficiencies()
    _oF, RF, TF = rF.efficiencies()
    assert np.max(np.abs(RT - RF)) < 1e-10    # efficiencies (gauge-invariant)
    assert np.max(np.abs(TT - TF)) < 1e-10
    for port in ("reflection", "transmission"):
        aT = rT.per_order_amplitudes(port)
        aF = rF.per_order_amplitudes(port)
        for c in ("Ex", "Ey"):
            # COMPLEX equality, not just magnitude: the pre-fix gauge left
            # per-order phase ramps up to ~pi (measured 0.34 complex diff
            # at 7e-14 magnitude diff).
            assert np.max(np.abs(aT[c] - aF[c])) < 1e-10
    fT = rT.to_multiorder_field(48, 48, 1.2e-6 / 48)
    fF = rF.to_multiorder_field(48, 48, 1.2e-6 / 48)
    assert np.max(np.abs(fT.Ex - fF.Ex)) < 1e-9
    assert np.max(np.abs(fT.Ey - fF.Ey)) < 1e-9


# ---------------------------------------------------------------------------
# NEW: rcwa 1-D entry points reject back-side incidence (pmm wave-3 mirror)
# ---------------------------------------------------------------------------

_BACK = np.deg2rad(100.0)


def test_rcwa_efficiency_1d_rejects_backside_angle():
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_efficiency_1d(1.0e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.6e-6,
                           angle=_BACK, n_orders=8)
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_efficiency_1d(1.0e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.6e-6,
                           theta=-_BACK, n_orders=8)


def test_rcwa_jones_1d_rejects_backside_angle():
    eps_r, eps_g = 2.0 * np.eye(3), 1.0 * np.eye(3)
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_jones_1d(1.0e-6, eps_r, eps_g, 1.5, 1.0, 0.3e-6, 0.5, 0.6e-6,
                      angle=_BACK, n_orders=8)
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_jones_1d_segments(
            1.0e-6, [(0.5, eps_r), (0.5, eps_g)], 1.5, 1.0, 0.3e-6, 0.6e-6,
            angle=_BACK, n_orders=8)


def test_rcwa_sweeps_reject_backside_angle():
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_efficiency_vs_wavelength(
            1.0e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
            [0.5e-6, 0.6e-6], angle=_BACK, n_orders=8)
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        rcwa_jones_vs_wavelength(
            1.0e-6, 2.0 * np.eye(3), 1.0 * np.eye(3), 1.5, 1.0, 0.3e-6, 0.5,
            [0.5e-6, 0.6e-6], angle=_BACK, n_orders=8)


def test_rcwa_frontside_angle_still_accepted():
    o, R, T = rcwa_efficiency_1d(1.0e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
                                 0.6e-6, angle=0.3, n_orders=8)
    assert abs(R.sum() + T.sum() - 1.0) < 1e-6
