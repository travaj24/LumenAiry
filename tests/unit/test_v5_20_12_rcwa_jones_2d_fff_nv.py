"""Full Popov-Neviere anisotropic off-diagonal FFF: rcwa_jones_2d(fff_nv).

``formulation='fff_nv'`` builds the complete tensor operator ``Q = [[eps.C]]
[[C]]^-1`` (Popov & Neviere 2001, JOSA A 18:2886) from a unit normal-vector
field, so ALL FOUR in-plane blocks -- including the off-diagonal ``Cxy``/``Cyx``
of a rotated in-plane director (``exy, eyx != 0``) -- get the correct inverse-
rule treatment (the ``'li'`` diagonal rule leaves the off-diagonal Laurent-
floored).  It reaches the same limit as ``'laurent'`` but converges markedly
faster on sharp anisotropic walls.

The rigorous form is well-conditioned only for a SMOOTH / single-orientation
wall normal (an anisotropic stripe: ``cond([[C]]) ~ O(10)``, order-independent);
a crossed / corner geometry makes ``[[C]]`` ill-conditioned and is REJECTED by a
``cond`` gate (measured 8 vs 3.8e7) so a wrong number can never propagate
silently -- the crossed anisotropic case rides on the (open) matched-coordinate
FFF work.  These tests pin: the scalar operator reduction (== the isotropic
Schuster form for constant N), the exact reduction to the rigorous 1-D
full-tensor solver on a stripe, the faster-than-Laurent convergence, the lossy
absorptance SPLIT (the lossless-trap guard), the cond gate + its expert bypass,
the uniform-cell routing, and the JAX / out-of-plane guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa import twod as _twod
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments


def _rot(phi, no, ne):
    """In-plane rotated uniaxial 3x3 (optic axis at angle phi in the x-y plane)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    """y-uniform (x-periodic) two-material stripe of 3x3 tensors."""
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


PX = 0.7e-6
WL = 1.0e-6
DEPTH = 0.5e-6


def test_fff_nv_scalar_operator_reduction():
    """For eps = eps_scalar*I and a CONSTANT unit N (a y-uniform stripe), the
    tensor operator Q = [[eps C]][[C]]^-1 equals the isotropic Schuster form
    E - Delta[[NN]] (Popov-Neviere App. B) to machine precision."""
    eps_s = np.where((np.arange(64) + 0.5)[:, None] / 64 < 0.5, 6.0, 2.1)
    eps_s = np.broadcast_to(eps_s, (64, 8)).astype(complex)
    orders, _ = _twod._harmonic_orders_2d(9, 1)
    Nx, Ny = _twod._nv_field_2d(eps_s, PX, PX, unit=True)
    assert np.allclose(np.unique(np.round(Nx, 9)), [1.0])   # constant N=(1,0)
    Z = np.zeros_like(eps_s)
    tens = _twod._nv_convolutions_2d_tensor(
        eps_s, Z, Z, eps_s, eps_s, Nx, Ny, orders, 9, 1, np)
    scal = _twod._nv_convolutions_2d(eps_s, Nx, Ny, orders, 9, 1, np)
    for a, b in zip(tens, scal):
        assert np.max(np.abs(a - b)) < 1e-9


def test_fff_nv_stripe_reduces_to_rigorous_1d():
    """A y-uniform rotated-director stripe: fff_nv reduces to the rigorous 1-D
    full-tensor solver AND is more accurate than laurent at the same order."""
    er, eg = _rot(np.deg2rad(35.0), 1.5, 2.3), np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 11
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=No, n_orders_y=1,
                                   formulation="laurent", symmetry=False)
    _o, R1, T1, J1 = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=No)
    # energy closes (2 incident pols, lossless)
    assert abs((np.sum(Rf) + np.sum(Tf)) - 2.0) < 5e-3
    # fff_nv tracks the rigorous 1-D solver, and MORE closely than laurent does
    ef = abs(np.sum(Rf) - np.sum(R1))
    el = abs(np.sum(Rl) - np.sum(R1))
    assert ef < el, f"fff_nv err {ef:.2e} not < laurent err {el:.2e}"
    assert np.max(np.abs(Jf - J1)) < np.max(np.abs(Jl - J1))


def test_fff_nv_beats_laurent_convergence():
    """fff_nv reaches a given accuracy at far lower order than laurent on a
    high-contrast rotated-director stripe (the off-diagonal FFF win)."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    cell = _stripe(er, eg)

    def sumR(No, form):
        _o, R, _T, _J = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                      n_orders_x=No, n_orders_y=1,
                                      formulation=form, symmetry=False)
        return np.sum(R)

    # converged reference: the rigorous 1-D solver at high order
    _o, Rref, _T, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=61)
    ref = np.sum(Rref)
    ef = abs(sumR(9, "fff_nv") - ref)
    el = abs(sumR(9, "laurent") - ref)
    assert ef < 0.5 * el, f"fff_nv {ef:.2e} not < half of laurent {el:.2e}"


def test_fff_nv_lossy_stripe_absorptance_split():
    """Lossless trap guard: on a LOSSY rotated-director stripe, fff_nv's
    absorptance (1 - R - T) tracks the rigorous 1-D solver's SPLIT more closely
    than laurent -- energy closure alone would not police this."""
    er = _rot(np.deg2rad(35.0), 1.5 + 0.15j, 2.3 + 0.15j)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 11

    def absorptance(form):
        _o, R, T, _J = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                     n_orders_x=No, n_orders_y=1,
                                     formulation=form, symmetry=False)
        return 1.0 - np.sum(R, 1) - np.sum(T, 1)      # (2,) per incident pol

    _o, R1, T1, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=No)
    A1 = 1.0 - np.sum(R1) - np.sum(T1)
    Af = np.mean(absorptance("fff_nv"))
    Al = np.mean(absorptance("laurent"))
    assert abs(Af - A1) < abs(Al - A1) + 1e-12


def test_fff_nv_crossed_cell_raises_and_bypass():
    """A crossed / corner (square pillar) anisotropic cell is REJECTED by the
    cond gate; allow_nonseparable_nv=True downgrades it to proceed."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    sq = np.zeros((48, 48, 3, 3), complex)
    m = np.zeros((48, 48), bool)
    m[12:36, 12:36] = True
    for i in range(48):
        for j in range(48):
            sq[i, j] = er if m[i, j] else eg
    with pytest.raises(ValueError, match="ill-conditioned"):
        rcwa_jones_2d(PX, PX, sq, 1.5, 1.0, DEPTH, WL, n_orders_x=9,
                      n_orders_y=9, formulation="fff_nv", symmetry=False)
    # expert bypass SKIPS the cond gate; a separate energy backstop may still
    # fire on a wildly-unstable lossless solve -- but the cond-gate message must
    # be gone (the gate was bypassed).
    try:
        out = rcwa_jones_2d(PX, PX, sq, 1.5, 1.0, DEPTH, WL, n_orders_x=9,
                            n_orders_y=9, formulation="fff_nv", symmetry=False,
                            allow_nonseparable_nv=True)
        assert len(out) == 4
    except Exception as e:                       # noqa: BLE001
        assert "ill-conditioned" not in str(e)


def test_fff_nv_uniform_routes_to_laurent():
    """A UNIFORM anisotropic tensor cell (no walls) + fff_nv routes to laurent
    (which is exact there) and matches it."""
    e = _rot(np.deg2rad(30.0), 1.5, 2.1)
    cell = np.broadcast_to(e, (32, 32, 3, 3)).copy()
    _o, Rf, Tf, Jf = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=5, n_orders_y=5,
                                   formulation="fff_nv", symmetry=False)
    _o, Rl, Tl, Jl = rcwa_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                   n_orders_x=5, n_orders_y=5,
                                   formulation="laurent", symmetry=False)
    assert np.max(np.abs(Rf - Rl)) < 1e-12
    assert np.max(np.abs(Jf - Jl)) < 1e-12


def test_fff_nv_jax_and_offplane_raise():
    """fff_nv rejects a JAX-traced cell (host normal field) and an out-of-plane
    tensor (the out-of-plane anisotropic FFF is not implemented)."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    # out-of-plane: add exz/ezx coupling
    oop = cell.copy()
    oop[:, :, 0, 2] = oop[:, :, 2, 0] = 0.3
    with pytest.raises(ValueError, match="IN-PLANE only"):
        rcwa_jones_2d(PX, PX, oop, 1.5, 1.0, DEPTH, WL, n_orders_x=9,
                      n_orders_y=1, formulation="fff_nv", symmetry=False)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="JAX backend"):
        rcwa_jones_2d(PX, PX, jnp.asarray(cell), 1.5, 1.0, DEPTH, WL,
                      n_orders_x=9, n_orders_y=1, formulation="fff_nv",
                      symmetry=False)
