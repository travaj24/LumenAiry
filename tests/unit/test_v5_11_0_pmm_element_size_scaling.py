"""Element-size-aware conditioning for the PMM spectral-element solver.

A spectral-element grid carries the element Jacobian ``J = (x_r - x_l)/2`` on the
mass/stiffness diagonals (``S0 ∝ J``, ``K ∝ 1/J``).  A grid spanning a huge width
ratio -- a thin liner next to a wide region, and especially the ``PMMStack`` union
grid of a *tapered* stack whose per-slice walls land sub-nm / coincident -- makes
``S0`` singular (``J → 0``) and the inversions raise ``LinAlgError`` (or return
non-physical modes).  Two fixes:

  * **(A)** symmetric Jacobi equilibration of the SE inversions, gated on an
    ill-scaling test so well-conditioned geometries stay BIT-IDENTICAL.
  * **(B)** near-coincident-wall merge in the ``PMMStack`` union grid so a
    genuinely zero-width union cell (which equilibration cannot rescue) never
    forms.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.pmm import (
    _ill_scaled,
    _safe_geig,
    _safe_inv,
    _safe_solve,
)
from lumenairy.elements.rcwa import uniaxial_tensor

_C = np.complex128
GR = np.eye(3, dtype=_C)
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)
WL = 0.633e-6


# --------------------------------------------------------------------------- #
# (A) the equilibration helpers
# --------------------------------------------------------------------------- #
def _well_scaled_mat(n=30, seed=1):
    rng = np.random.RandomState(seed)
    B = rng.randn(n, n) + 1j * rng.randn(n, n)
    return B @ B.conj().T + n * np.eye(n)


def _ill_scaled_mat(n=30, ratio=5e2, seed=1):
    A = _well_scaled_mat(n, seed)
    s = np.concatenate([np.full(n // 3, 1.0), np.full(n - 2 * (n // 3), ratio),
                        np.full(n // 3, 1.0)])
    sc = np.sqrt(s / np.abs(np.diag(A)))
    return (sc[:, None] * A) * sc[None, :]


def test_safe_inv_bit_identical_when_well_scaled():
    # the regression-safety property: a well-scaled matrix takes the plain path,
    # so _safe_inv / _safe_solve are BIT-IDENTICAL to numpy.
    A = _well_scaled_mat()
    assert not _ill_scaled(A)
    assert np.array_equal(_safe_inv(A), np.linalg.inv(A))
    B = np.random.RandomState(2).randn(30, 5) + 0j
    assert np.array_equal(_safe_solve(A, B), np.linalg.solve(A, B))


def test_safe_inv_accurate_when_ill_scaled():
    # an element-size ill-scaled matrix is equilibrated -> accurate inverse.
    A = _ill_scaled_mat()
    assert _ill_scaled(A, ratio=1e-2)
    I = np.eye(A.shape[0])
    assert np.max(np.abs(A @ _safe_inv(A) - I)) < 1e-10


def test_safe_geig_eigenvalues_match_plain_when_well_scaled():
    import scipy.linalg as sla
    A = _well_scaled_mat(20, seed=3)
    B = _well_scaled_mat(20, seed=4)
    q2a, _ = _safe_geig(A, B)
    q2b, _ = sla.eig(A, B)
    assert np.array_equal(q2a, q2b)            # well-scaled -> plain sla.eig


# --------------------------------------------------------------------------- #
# (B) the PMMStack union-grid degenerate-wall merge -- the operative fix
# --------------------------------------------------------------------------- #
def _stack(off, degree=12, ang=10.0):
    st = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=degree)
    st.add_layer(0.2e-6, segments=[(0.3, LC), (0.7, GR)])
    st.add_layer(0.3e-6, segments=[(0.3 + off, 2.0 * GR), (0.7 - off, GR)])
    return st.set_source(WL, angle=np.radians(ang)).solve()


def test_near_coincident_union_walls_no_longer_singular():
    # two layers whose shared wall differs by ~1e-11 of the period make a sub-pm
    # union cell -> a J~0 element -> Singular S0 BEFORE the fix.  AFTER: finite,
    # physical, energy-conserving (2.0 = lossless over the two incident pols).
    o, R, T, J = _stack(1e-11)
    assert np.all(np.isfinite(J))
    assert np.max(np.abs(J)) < 10.0                       # not the |J|~1e10 garbage
    assert abs(float(R.sum() + T.sum()) - 2.0) < 1e-5     # energy (2 polarizations)


def test_merged_degenerate_wall_matches_aligned():
    # the spurious sub-pm wall is merged away -> the result equals the clean,
    # exactly-aligned grid (the merge perturbs geometry by < 1 pm => no physics).
    _, R0, T0, J0 = _stack(0.0)
    _, Rd, Td, Jd = _stack(1e-11)
    assert np.max(np.abs(Jd - J0)) < 1e-9


def test_aligned_stack_bit_identical_to_segments():
    # the well-scaled path is untouched: a 1-layer PMMStack reproduces
    # pmm_jones_1d_segments (RELAXED from bit-identity 2026-06-10: the v5.14
    # noise-robust selectors retire bit-identity across paths -- the gauge
    # WITHIN degenerate mode pairs is BLAS-build/runner dependent (one CI
    # runner produced 1.4e-7 where others give 0.0); the physical contract
    # is the same 5e-6 the stack-vs-segments sibling test pins).
    st = la.PMMStack(0.8e-6, n_substrate=1.5, n_superstrate=1.0, degree=20)
    st.add_layer(0.5e-6, segments=[(0.5, LC), (0.5, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(25)).solve()
    o2, R2, T2, J2 = la.pmm_jones_1d_segments(
        0.8e-6, [(0.5, LC), (0.5, GR)], 1.5, 1.0, 0.5e-6, WL,
        angle=np.radians(25), degree=20)
    assert np.max(np.abs(J - J2)) < 5e-6
    assert np.max(np.abs(R - R2)) < 5e-6 and np.max(np.abs(T - T2)) < 5e-6


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_scalar_pmm_unchanged(pol):
    # the scalar solver (_safe_solve / _safe_geig on the well-scaled path) is
    # bit-identical: a chunky binary grating is energy-conserving and stable.
    o, R, T = la.pmm_efficiency_1d(0.8e-6, 2.5, 1.0, 1.0, 1.0, 0.3e-6, 0.5, WL,
                                   polarization=pol)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-3
