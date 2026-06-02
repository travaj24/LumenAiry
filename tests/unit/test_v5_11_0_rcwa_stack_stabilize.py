"""v5.11.0 RCWAStack.solve(stabilize=True) -- consensus selector for sharp-
resonant metal multilayers (audit AUDIT_RCWA_STACK_RESONANT_CONVERGENCE).

A sharp-resonant metal STACK biases a SINGLE diffraction order (a reflection
null) non-monotonically at isolated n_orders while total power stays bounded --
the isolated-resonance pathology already fixed for pmm_efficiency_1d in v5.10.6,
here in the multilayer S-matrix.  ``stabilize=True`` re-solves a short DOWNWARD
n_orders window and returns the solve whose ZERO-ORDER R/T is in the consensus
cluster (the spikes are the outliers; the consensus is PER-ORDER because total
power does not move).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import (
    _largest_feature_cluster,
    _zero_order_efficiency,
)

S = 48


def _square_cell(eps_hi):
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.where((np.abs(X) < 0.25) & (np.abs(Y) < 0.25), eps_hi, 2.1 + 0j)


def _stack(eps_hi, n_orders):
    st = la.RCWAStack(0.5e-6, period_y=0.5e-6, n_substrate=1.5,
                      n_orders=n_orders, n_orders_y=n_orders)
    st.add_layer(0.10e-6, eps_cell=_square_cell(eps_hi))
    st.add_layer(0.05e-6, eps=2.1)
    st.set_source(0.6e-6, theta=0.001)
    return st


# ---------------------------------------------------------------------------
# the consensus selector mechanism (the spike-healing core)
# ---------------------------------------------------------------------------

def test_consensus_cluster_selects_plateau_over_spikes():
    # a converged plateau {0,1,3} plus two isolated per-order spikes {2,4}
    feats = [np.array([0.80, 0.1, 0.1, 0.0]),
             np.array([0.81, 0.1, 0.1, 0.0]),
             np.array([0.08, 0.5, 0.1, 0.0]),     # spike
             np.array([0.805, 0.1, 0.1, 0.0]),
             np.array([0.14, 0.4, 0.1, 0.0])]     # spike
    cluster = _largest_feature_cluster(feats, 0.02)
    assert sorted(cluster) == [0, 1, 3]            # the plateau, spikes rejected
    assert min(cluster) == 0                        # highest-n_orders consensus


def test_consensus_cluster_all_agree():
    feats = [np.array([0.5, 0.5]) + 1e-3 * i for i in range(4)]
    assert sorted(_largest_feature_cluster(feats, 0.02)) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# end-to-end behaviour
# ---------------------------------------------------------------------------

def test_stabilize_false_is_bit_identical_to_inner_solve():
    # the refactor must leave the default solve byte-for-byte unchanged.
    st = _stack(6.25 + 0j, 8)
    o0, R0, T0 = st.solve().efficiencies()              # stabilize=False default
    o1, R1, T1 = st._solve_once().efficiencies()
    assert np.array_equal(np.asarray(R0), np.asarray(R1))
    assert np.array_equal(np.asarray(T0), np.asarray(T1))


def test_stabilize_no_harm_on_converged_stack():
    # a converged (fast, dielectric) stack: stabilize returns the requested
    # n_orders unchanged (it is in its own consensus cluster) -- no warning.
    st = _stack(6.25 + 0j, 8)
    f0 = _zero_order_efficiency(st.solve())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f1 = _zero_order_efficiency(st.solve(stabilize=True))
        warned = any("consensus" in str(x.message) for x in w)
    assert np.max(np.abs(f0 - f1)) < 1e-12
    assert not warned
    assert st.nox == 8 and st.noy == 8                  # restored


def test_stabilize_under_resolved_warns_and_returns_requested():
    # a hard metal stack at low n_orders has no convergence plateau -> warn +
    # return the requested n_orders (cannot manufacture resolution downward).
    st = _stack((0.2 + 6j) ** 2, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.solve(stabilize=True)
        assert any("consensus" in str(x.message) for x in w)
    assert st.nox == 8 and st.noy == 8                  # restored after the scan


def test_stabilize_conserves_energy():
    st = _stack(6.25 + 0j, 8)
    o, R, T = st.solve(stabilize=True).efficiencies()
    R, T = np.asarray(R), np.asarray(T)
    # (2, N): each incident pol conserves to ~1 (lossless dielectric)
    for pol in (0, 1):
        assert abs(float(R[pol].sum() + T[pol].sum()) - 1.0) < 1e-6


def test_stabilize_window_never_aliases_or_leaks_nox():
    # the downward window stays <= the requested n_orders (cell sized for it),
    # so no aliasing ValueError, and nox/noy are restored even on the 1-D path.
    st = la.RCWAStack(0.5e-6, n_substrate=1.5, n_orders=8)   # 1-D stack
    st.add_layer(0.1e-6, eps_cell=(np.where(
        (np.arange(S) + 0.5) / S < 0.5, 6.25 + 0j, 2.1 + 0j)))
    st.set_source(0.6e-6, theta=0.001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve(stabilize=True)             # must not raise
    assert st.nox == 8 and st.noy == 0
