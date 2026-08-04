"""Full Popov-Neviere anisotropic off-diagonal FFF for the hybrid PMM:
pmm_jones_2d(formulation='fff_nv').

The PMM mirror of ``rcwa_jones_2d(formulation='fff_nv')``.  For a SEPARABLE
(single-orientation, x- or y-patterned) anisotropic cell the wall-normal is
constant, so the projected tensor operator reduces to the rigorous Li-1996 1-D
anisotropic factorization -- the wall-normal diagonal takes the inverse rule and
the off-diagonal ``Cxy``/``Cyx`` of a rotated director gets its correct
composite.  (PMM's ``'li'`` applies the inverse rule ONLY to the ``E_z``
elimination in the separable branch, so ``'fff_nv'`` is the first correct
in-plane inverse-rule treatment there -- an even bigger gain than in rcwa.)  A
crossed / out-of-plane / JAX cell raises, matching rcwa's honest scoping.

These tests pin: the reduction to the rigorous ``rcwa_jones_1d_segments`` on a
stripe (and faster convergence than laurent), the cross-solver agreement with
``rcwa_jones_2d(fff_nv)``, the lossy absorptance split, the uniform-cell routing,
and the crossed / out-of-plane / JAX guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments

# eig-heavy 2-D fff_nv (degree 11, n_orders 13); the numerics are not
# Python-version-sensitive, so run once in the slow-tests job (keeps the
# fast 4-Python gate under its cap -- the v5.21.1 3.13 runner tipped over
# 40 min).
pytestmark = pytest.mark.slow


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def test_pmm_fff_nv_stripe_reduces_to_rigorous_1d():
    """A rotated-director stripe: pmm fff_nv converges to the rigorous 1-D
    full-tensor solver, and faster than pmm laurent."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    cell = _stripe(er, eg)
    _o, Rref, _T, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=61)
    ref = np.sum(Rref)

    def sumR(No, form):
        _o, R, _T, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=9,
                                     n_orders=No, formulation=form,
                                     symmetry=False)
        return np.sum(R)

    ef, el = abs(sumR(13, "fff_nv") - ref), abs(sumR(13, "laurent") - ref)
    assert ef < 1e-3, f"fff_nv err {ef:.2e} not converged to 1-D"
    assert ef < 0.2 * el, f"fff_nv {ef:.2e} not << laurent {el:.2e}"


def test_pmm_fff_nv_matches_rcwa_fff_nv():
    """Cross-solver: pmm fff_nv and rcwa fff_nv converge to the SAME rigorous
    answer on the same stripe."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    _o, Rp, Tp, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=11,
                                  n_orders=13, formulation="fff_nv",
                                  symmetry=False)
    # REFERENCE TRUNCATION MOVED 13 -> 11 (2026-08-04).  The RCWA reference at
    # n_orders_x = 13 sits ON this cell's measure-zero instability, and the
    # "cross-solver residual" this test was reading was simply that: the
    # reference's OWN lossless-closure violation.  Measured, same cell, same
    # PMM answer, |dT| against the reference's closure defect:
    #
    #     M    RCWA closure    |dT|
    #     11   -2.3e-05        2.9e-04
    #     12   +2.6e-03        3.0e-03
    #     13   +2.6e-02        2.7e-02   <- was pinned here, the WORST of the six
    #     14   +1.6e-03        1.9e-03
    #     15   +1.6e-04        2.1e-03
    #
    # |dT| tracks the closure defect one-for-one at every M, so the quantity
    # being compared was never a factorization floor.  Both engines emit
    # _EnergyWarning at 13 saying exactly this ("the truncation is numerically
    # unstable here and the PER-ORDER efficiencies are suspect").  The 4e-3 bar
    # is UNCHANGED -- widening it would have pinned the instability instead of
    # avoiding it -- and the closure of both engines is now asserted first, so
    # this can never again silently compare against an unstable reference.
    _o, Rr, Tr, _J = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=11, n_orders_y=1,
                                   formulation="fff_nv", symmetry=False)
    # the structure is provably lossless: neither side may be compared unless
    # it conserves.  (Measured: PMM -2.9e-04, RCWA -2.3e-05.)
    assert abs(np.sum(Rp) + np.sum(Tp) - 2.0) < 1e-3, "PMM reference unstable"
    assert abs(np.sum(Rr) + np.sum(Tr) - 2.0) < 1e-3, "RCWA reference unstable"
    # PMM (Laurent-projected) and RCWA (Li-2003 successive) are DIFFERENT
    # factorizations, so they converge to slightly different floors -- the
    # cross-solver residual is a genuine one, not a machine-precision match.
    # 4e-3 keeps the check meaningful (catches gross errors) without flaking.
    assert abs(np.sum(Rp) - np.sum(Rr)) < 4e-3
    assert abs(np.sum(Tp) - np.sum(Tr)) < 4e-3


def test_pmm_fff_nv_lossy_absorptance_split():
    """Lossless-trap guard: on a lossy stripe fff_nv's absorptance tracks the
    rigorous 1-D split at least as closely as laurent."""
    er = _rot(np.deg2rad(35.0), 1.5 + 0.15j, 2.3 + 0.15j)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 13

    def absorptance(form):
        _o, R, T, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=9,
                                    n_orders=No, formulation=form,
                                    symmetry=False)
        return np.mean(1.0 - np.sum(R, 1) - np.sum(T, 1))

    _o, R1, T1, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=No)
    A1 = 1.0 - np.sum(R1) - np.sum(T1)
    assert abs(absorptance("fff_nv") - A1) < abs(absorptance("laurent") - A1) + 1e-9


def test_pmm_fff_nv_uniform_routes():
    """A uniform anisotropic cell (no walls) + fff_nv matches laurent (exact)."""
    e = _rot(np.deg2rad(30.0), 1.5, 2.1)
    cell = np.broadcast_to(e, (8, 8, 3, 3)).copy()
    _o, Rf, Tf, Jf = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=7,
                                  n_orders=5, formulation="fff_nv",
                                  symmetry=False)
    _o, Rl, Tl, Jl = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=7,
                                  n_orders=5, formulation="laurent",
                                  symmetry=False)
    assert np.max(np.abs(Jf - Jl)) < 1e-12


def test_pmm_fff_nv_crossed_and_offplane_and_jax_raise():
    """fff_nv rejects a crossed cell, an out-of-plane tensor, and a JAX cell."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    # crossed (both axes patterned)
    sq = np.zeros((48, 48, 3, 3), complex)
    m = np.zeros((48, 48), bool)
    m[12:36, 12:36] = True
    for i in range(48):
        for j in range(48):
            sq[i, j] = er if m[i, j] else eg
    with pytest.raises(ValueError, match="SEPARABLE"):
        pmm_jones_2d(PX, PX, sq, 1.5, 1.0, DEPTH, WL, degree=7, n_orders=5,
                     formulation="fff_nv", symmetry=False)
    # out-of-plane stripe
    oop = _stripe(er, eg)
    oop[:, :, 0, 2] = oop[:, :, 2, 0] = 0.3
    with pytest.raises(ValueError, match="IN-PLANE only"):
        pmm_jones_2d(PX, PX, oop, 1.5, 1.0, DEPTH, WL, degree=9, n_orders=9,
                     formulation="fff_nv", symmetry=False)
    # JAX cell
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="NumPy only"):
        pmm_jones_2d(PX, PX, jnp.asarray(_stripe(er, eg)), 1.5, 1.0, DEPTH, WL,
                     degree=9, n_orders=9, formulation="fff_nv", symmetry=False,
                     region_layout=np.zeros((64, 8), int))
