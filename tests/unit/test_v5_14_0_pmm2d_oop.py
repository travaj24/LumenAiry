"""v5.14.0 -- OUT-OF-PLANE full (3, 3) tensors on the 2-D hybrid PMM.

``pmm_jones_2d`` with nonzero ``xz/yz/zx/zy`` routes through the shared
full-3x3 first-order generator (``_layer_eigenmodes_tensor``'s 6-tuple branch,
Li 2003) and the GENERALIZED S-matrix -- the library's FIRST 2-D out-of-plane
solver (rcwa_jones_2d is in-plane only), so every gate here is 1-D-reducible:

* the generator branch with (numerically) vanishing off-plane entries
  reproduces the in-plane path;
* a UNIFORM full-3x3 cell matches the validated 1-D ``pmm_jones_1d`` at
  duty=1 (Berreman-grade);
* a y-uniform out-of-plane grating conserves y-momentum EXACTLY (separable
  path) and matches the validated 1-D full-3x3 solver per order;
* non-reciprocal ``e_xz != e_zx`` reproduces the 1-D total power (which is
  PHYSICALLY != 1 -- the no-auto-balance trap guard: never assert unity).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.pmm import pmm_jones_1d, pmm_jones_2d

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6


def _o0(o):
    return (o[:, 0] == 0) & (o[:, 1] == 0)


def test_vanishing_offplane_matches_inplane_path():
    """exz = 1e-14 forces the generator/generalized-S route; the answer must
    match the in-plane (symmetric-cascade) solve of the same cell."""
    sc = np.full((6, 6), 1.0 + 0j)
    sc[1:4, 2:5] = 6.0
    tc = np.zeros((6, 6, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = sc
    tc_oop = tc.copy()
    tc_oop[:, :, 0, 2] = 1e-14
    tc_oop[:, :, 2, 0] = 1e-14
    o1, R1, T1, J1 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=9, n_orders=4)
    o2, R2, T2, J2 = pmm_jones_2d(_P, _P, tc_oop, 1.5, 1.0, _DEP, _WL,
                                  degree=9, n_orders=4)
    assert np.array_equal(o1, o2)
    assert np.max(np.abs(R1 - R2)) < 1e-6
    assert np.max(np.abs(T1 - T2)) < 1e-6
    assert np.max(np.abs(J1 - J2)) < 1e-5


def test_uniform_full3x3_cell_matches_berreman_and_1d():
    """Uniform out-of-plane cell: the 2-D generator path vs the Berreman 4x4
    oracle (precise; the 2-D uniform branch is exact-diagonal) and the 1-D
    solver (whose OOP metric path has its own ~1e-3 algebraic floor)."""
    from tests.unit._berreman4x4 import berreman_jones
    eps = np.array([[2.9, 0.1, 0.3],
                    [0.1, 2.4, 0.2],
                    [0.3, 0.2, 2.1]], dtype=complex)
    tc = np.zeros((4, 4, 3, 3), complex)
    tc[:, :] = eps
    o2, R2, T2, J2 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=7, n_orders=2)
    J_b = berreman_jones(eps, 1.0, 1.5, _DEP, _WL, 0.0)[0]   # reflection Jones
    assert np.max(np.abs(J2 - J_b)) < 1e-8        # Berreman-grade (2.4e-15)
    o1, R1, T1, J1 = pmm_jones_1d(_P, eps, eps, 1.5, 1.0, _DEP, 0.5, _WL,
                                  degree=10, stabilize=False)
    m2, m1 = _o0(o2), o1 == 0
    for row in (0, 1):                            # 1-D OOP metric-path floor
        assert abs(float(R2[row][m2][0]) - float(R1[row][m1][0])) < 1e-2
        assert abs(float(T2[row][m2][0]) - float(T1[row][m1][0])) < 1e-2


def _y_uniform_cells(exz=0.3, ezx=None):
    ezx = exz if ezx is None else ezx
    ridge = np.array([[2.25, 0.0, exz],
                      [0.0, 2.10, 0.0],
                      [ezx, 0.0, 2.40]], dtype=complex)
    groove = np.eye(3, dtype=complex)
    S = 6
    tc = np.zeros((S, S, 3, 3), complex)
    tc[:, :] = groove
    tc[1:4, :] = ridge
    return tc, ridge, groove


def test_y_uniform_oop_grating_matches_1d():
    """The wall-normal x longitudinal (exz/ezx) channel -- the hard one -- on a
    y-uniform grating.

    ORACLE REPOINTED (AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14): the primary
    reference is now ``rcwa_jones_1d_segments`` -- dispersion-anchored by
    ``test_audit_oop_dispersion.py`` -- at a stable truncation (n_orders=9;
    the old n_orders=8 landed on an unstable truncation post-fix, energy
    +4e-3 with the solver's own warning).  ``pmm_jones_1d``'s independent
    metric-generator OOP path is kept as a LOOSE cross-check only: post-fix
    it sits ~4.5e-2 from BOTH dispersion-anchored engines at m = +/-1 in the
    OOP-coupled polarization (energy exactly 1 -- energy-blind), a known
    open item recorded in the audit doc; its historical '~1e-3 algebraic
    floor' was measured against the pre-fix (wrong) reference."""
    from lumenairy.elements.rcwa import rcwa_jones_1d_segments
    tc, ridge, groove = _y_uniform_cells()
    o2, R2, T2, J2 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=13, n_orders=9)
    # y-momentum conservation is exact on the separable path
    side = o2[:, 1] != 0
    assert float(R2[:, side].sum() + T2[:, side].sum()) < 1e-12
    oR, RR, TR, JR = rcwa_jones_1d_segments(
        _P, [(0.5, ridge), (0.5, groove)], 1.5, 1.0, _DEP, _WL, theta=0.0,
        n_orders=41)
    for m in (-1, 0, 1):
        i2 = (o2[:, 0] == m) & (o2[:, 1] == 0)
        iR = oR == m
        for row in (0, 1):
            assert abs(float(T2[row][i2][0]) - float(TR[row][iR][0])) < 5e-3
            assert abs(float(R2[row][i2][0]) - float(RR[row][iR][0])) < 5e-3
    assert np.max(np.abs(J2 - JR)) < 5e-3
    # loose cross-check vs the (independent, unfixed) 1-D metric OOP path
    o1, R1, T1, J1 = pmm_jones_1d(_P, ridge, groove, 1.5, 1.0, _DEP, 0.5,
                                  _WL, degree=16, stabilize=False)
    for m in (-1, 0, 1):
        i2 = (o2[:, 0] == m) & (o2[:, 1] == 0)
        i1 = o1 == m
        if i1.any() and i2.any():
            for row in (0, 1):
                assert abs(float(T2[row][i2][0])
                           - float(T1[row][i1][0])) < 6e-2
    assert np.max(np.abs(J2 - J1)) < 6e-2


def test_nonreciprocal_total_power_matches_1d_not_unity():
    """exz != ezx (real-asymmetric, non-Hermitian): the total power is
    PHYSICALLY != 1 (precedent: the 1-D slant/convergence suite) -- the gate is
    agreement with the 1-D value, never unity."""
    tc, ridge, groove = _y_uniform_cells(exz=0.3, ezx=0.5)
    o2, R2, T2, _J2 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                   degree=13, n_orders=8)
    o1, R1, T1, _J1 = pmm_jones_1d(_P, ridge, groove, 1.5, 1.0, _DEP, 0.5,
                                   _WL, degree=16, stabilize=False)
    for row in (0, 1):
        e2 = float(R2[row].sum() + T2[row].sum())
        e1 = float(R1[row].sum() + T1[row].sum())
        assert abs(e2 - e1) < 2e-2
    # at least one channel visibly departs from unity (the trap guard)
    e_all = [float(R1[r].sum() + T1[r].sum()) for r in (0, 1)]
    assert max(abs(e - 1.0) for e in e_all) > 1e-4
