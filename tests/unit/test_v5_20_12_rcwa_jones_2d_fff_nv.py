"""Full anisotropic off-diagonal FFF: rcwa_jones_2d(formulation='fff_nv').

``formulation='fff_nv'`` builds the Li-2003 successive full-tensor factorization
``ehat = L2 L1(eps)`` (J.Opt.A 5:345; the Smagin-Weiss-Dyakov 2026 ``l+-_tau``
operator), so ALL FOUR in-plane blocks -- including the off-diagonal
``Cxy``/``Cyx`` of a rotated in-plane director (``exy, eyx != 0``) -- get the
correct inverse-rule treatment (the ``'li'`` diagonal rule leaves the
off-diagonal Laurent-floored).  It reaches the same limit as ``'laurent'`` but
converges markedly faster on sharp anisotropic walls.

Unlike the normal-vector projector form ``[[eps.C]][[C]]^-1`` (which inverts an
ill-conditioned 2N x 2N matrix, ``cond ~ 1e7`` for a crossed pillar), the Li-2003
operator inverts ONLY scalar wall-normal elements (plus one N x N block), so it
is well-conditioned even for a CROSSED (both-axis-patterned) cell -- so crossed
anisotropic pillars now CONVERGE (rigorous for axis-aligned / Manhattan cells).
Out-of-plane tensors (`exz, eyz != 0`) are also handled -- the full-3x3 `L2 L1`
plus the `E_z` fold (Li 2003 Eq. 27) -- again converging far faster than laurent.
These tests pin: the operator reduction to the rigorous Li-1996 1-D rule on a
stripe, the exact reduction to the 1-D full-tensor solver + faster-than-Laurent
convergence on a stripe, the lossy absorptance SPLIT (the lossless-trap guard),
the crossed-cell convergence (monotone + beats laurent), the out-of-plane
fast convergence, the uniform-cell routing, and the JAX guard.
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


def test_fff_nv_operator_reduces_to_li1996_on_stripe():
    """The Li-2003 successive operator L2 L1 reduces EXACTLY (machine precision)
    to the rigorous Li-1996 1-D factorization on a y-uniform stripe: the
    wall-normal diagonal Cxx == [[1/exx]]^-1."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)[:2, :2]
    eg = np.diag([2.25, 2.25]).astype(complex)
    Sx = 64
    xm = (np.arange(Sx) + 0.5) / Sx < 0.5
    cell = np.zeros((Sx, 8, 2, 2), complex)
    for ix in range(Sx):
        cell[ix, :] = er if xm[ix] else eg
    orders, _ = _twod._harmonic_orders_2d(9, 1)
    Cxx, Cxy, Cyx, Cyy = _twod._li_convolutions_2d_tensor(
        cell[:, :, 0, 0], cell[:, :, 0, 1], cell[:, :, 1, 0],
        cell[:, :, 1, 1], orders, 9, 1, np)
    inv_exx = np.linalg.inv(
        _twod._eps_convolution_2d(1.0 / cell[:, :, 0, 0], orders, 9, 1))
    assert np.max(np.abs(Cxx - inv_exx)) < 1e-12       # rigorous inverse rule


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


def test_fff_nv_crossed_cell_converges_and_beats_laurent():
    """A CROSSED (both-axis-patterned) rotated-director pillar now CONVERGES
    under fff_nv (Li-2003 L2 L1, well-conditioned) -- monotone and markedly
    faster than laurent on a high-contrast lossy cell, energy closed."""
    th = np.deg2rad(45.0)                          # metal-like eps, rotated 45 deg
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    er = R @ np.diag([-8.0 + 1.2j, -2.0 + 0.8j, -2.0 + 0.8j]).astype(complex) @ R.T
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)

    def _sq(S):
        m = np.zeros((S, S), bool)
        m[S // 4:3 * S // 4, S // 4:3 * S // 4] = True
        c = np.zeros((S, S, 3, 3), complex)
        for i in range(S):
            for j in range(S):
                c[i, j] = er if m[i, j] else eg
        return c

    def sumR(No, form):
        _o, R, _T, _J = rcwa_jones_2d(0.5e-6, 0.5e-6, _sq(max(40, 4 * No + 4)),
                                      1.5, 1.0, 0.25e-6, WL, n_orders_x=No,
                                      n_orders_y=No, formulation=form,
                                      symmetry=False)
        return np.sum(R)

    ref = sumR(17, "fff_nv")                       # best-converging reference
    ef = [abs(sumR(No, "fff_nv") - ref) for No in (5, 7, 9, 11)]
    el = [abs(sumR(No, "laurent") - ref) for No in (5, 7, 9, 11)]
    assert ef[-1] < ef[0]                          # fff_nv converging
    assert ef[-1] < 0.3 * el[-1]                   # >~3x better than laurent
    assert all(ef[i + 1] <= ef[i] + 1e-9 for i in range(len(ef) - 1))  # monotone


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


def _oop_stripe(No):
    """y-uniform stripe of a tilted uniaxial (optic axis tilted about y ->
    exz, ezx != 0, out-of-plane)."""
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    er = Ry @ np.diag([1.5 ** 2, 1.5 ** 2, 2.4 ** 2]).astype(complex) @ Ry.T
    eg = np.diag([2.1, 2.1, 2.1]).astype(complex)
    Sx = max(64, 4 * No + 4)
    xm = (np.arange(Sx) + 0.5) / Sx < 0.5
    cell = np.zeros((Sx, 8, 3, 3), complex)
    for ix in range(Sx):
        cell[ix, :] = er if xm[ix] else eg
    return cell


def test_fff_nv_out_of_plane_converges_fast():
    """OUT-OF-PLANE (exz, ezx != 0) is now supported: the Li-2003 successive
    full-3x3 factorization + the E_z fold.  fff_nv converges FAST (nearly
    order-independent) while laurent climbs slowly TOWARD the same value -- so
    fff_nv reaches the true limit at far lower order.  Energy closes."""
    def sumR(No, form):
        _o, R, T, _J = rcwa_jones_2d(PX, PX, _oop_stripe(No), 1.5, 1.0, DEPTH,
                                     WL, n_orders_x=No, n_orders_y=1,
                                     formulation=form, symmetry=False)
        return np.sum(R), np.sum(R) + np.sum(T)

    conv, e0 = sumR(13, "fff_nv")                  # fff_nv converged value
    assert abs(e0 - 2.0) < 5e-3                     # energy closes
    f7, _ = sumR(7, "fff_nv")
    l7, _ = sumR(7, "laurent")
    l15, _ = sumR(15, "laurent")
    assert abs(f7 - conv) < 2e-5                    # fff_nv ~converged by No=7
    assert abs(l15 - conv) < abs(l7 - conv)         # laurent climbs toward fff_nv
    assert abs(f7 - conv) < 0.3 * abs(l7 - conv)    # fff_nv converges much faster


def test_fff_nv_jax_raises():
    """fff_nv rejects a JAX-traced cell (host-side successive factorization)."""
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="JAX backend"):
        rcwa_jones_2d(PX, PX, jnp.asarray(cell), 1.5, 1.0, DEPTH, WL,
                      n_orders_x=9, n_orders_y=1, formulation="fff_nv",
                      symmetry=False)
