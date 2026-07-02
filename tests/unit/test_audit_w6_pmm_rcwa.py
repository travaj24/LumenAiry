"""Deep-audit 2026-07-01 wave-6 fixes -- pmm / rcwa cluster.

Discriminating regression tests for the CODE fixes:

* P3-26 -- ``_converged_cluster`` is a MUTUAL (pairwise) clique, not an
  anchor star admitting members up to ``2*tol`` apart.
* P3-27 -- the JAX ``PMMStack`` twin solves the uniform half-spaces via ONE
  shared n x n geometric eig (the v5.14.2 backlog-A2 optimization) instead of
  two full traced assemblies + dense 2n x 2n eigs, at unchanged forward
  results and finite gradients.
* P3-33 -- 2-D hybrid PMM ``formulation='li'`` routes the inverse rule to the
  wall-NORMAL E component per patterned axis, restoring the 90-degree
  rotation symmetry that ``'laurent'`` has exactly.
* P3-37 -- ``Granet2DTransverseE`` no longer retains the five dead operator
  matrices (Curl/Kzt/Ktz/G3/Meps33) after assembly.
* P3-38 -- dispersive (callable) ``eps_tensor_cell`` layers accept
  out-of-plane tensors like the static path (same solve result) and enforce
  the same ``|e_zz| > 0`` guard.

(P3-34 / P3-35 are docstring-only corrections -- no runtime surface.)
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm._core import _converged_cluster

# --------------------------------------------------------------------------- #
# P3-26: mutual (clique) convergence cluster
# --------------------------------------------------------------------------- #

def _rec(v):
    """A minimal one-order scan record for ``_aligned_max_diff``."""
    return (np.array([0]), (np.array([v]),), None)


def test_p26_cluster_is_pairwise_not_star():
    # d(A,B) = d(A,C) = 2.9e-3 <= tol, but d(B,C) = 5.8e-3 = ~2*tol: the old
    # anchor-star co-clustered B and C; a mutual clique must not.
    records = [_rec(0.0), _rec(2.9e-3), _rec(-2.9e-3)]
    cl = _converged_cluster(records, [True, True, True], 3e-3, 2)
    assert len(cl) == 2
    vals = [records[i][1][0][0] for i in cl]
    assert abs(vals[0] - vals[1]) <= 3e-3          # pairwise within tol


def test_p26_converged_plateau_unchanged():
    # A genuinely converged plateau (every pair within tol) returns everything
    # -- identical to the pre-fix behavior.
    records = [_rec(0.0), _rec(1e-3), _rec(-1e-3), _rec(0.5e-3)]
    cl = _converged_cluster(records, [True] * 4, 3e-3, 2)
    assert cl == [0, 1, 2, 3]
    # non-passive members stay excluded
    cl2 = _converged_cluster(records, [True, False, True, True], 3e-3, 2)
    assert cl2 == [0, 2, 3]


# --------------------------------------------------------------------------- #
# P3-33: 'li' rotation symmetry on separable cells
# --------------------------------------------------------------------------- #

def _aligned_rot90_diff(res_x, res_y):
    """Max |R/T| difference between an x-patterned solve and its 90-degree
    rotated y-patterned twin, aligned by swapped (m, n) order indices."""
    ox, Rx, Tx = res_x
    oy, Ry, Ty = res_y
    keyx = {(int(m), int(n)): i for i, (m, n) in enumerate(ox)}
    d = 0.0
    for i, (m, n) in enumerate(oy):
        j = keyx[(int(n), int(m))]
        d = max(d, abs(Rx[j] - Ry[i]), abs(Tx[j] - Ty[i]))
    return d


@pytest.mark.parametrize("formulation", ["li", "laurent"])
def test_p33_cell_rot90_symmetry(formulation):
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell

    kw = dict(degree=9, n_orders=3, formulation=formulation)
    cell_x = np.array([[6.25], [1.0]], dtype=complex)   # (2,1) x-patterned
    cell_y = np.array([[6.25, 1.0]], dtype=complex)     # (1,2) y-patterned
    res_x = pmm_efficiency_2d_cell(1.0, 1.0, cell_x, 1.5, 1.0, 0.5, 0.6,
                                   polarization="tm", **kw)
    res_y = pmm_efficiency_2d_cell(1.0, 1.0, cell_y, 1.5, 1.0, 0.5, 0.6,
                                   polarization="te", **kw)
    # pre-fix 'li' broke this at ~4.3e-3 (the inverse rule landed on the
    # tangential Ex for the y-patterned cell); both rules are now exact.
    assert _aligned_rot90_diff(res_x, res_y) < 1e-10


def test_p33_x_uniform_li_tracks_staggered_reference():
    # x-uniform (y-patterned) lamellar cell: the fixed 'li' must sit at the
    # same physics as the independent NO-FLOOR staggered reference (and as
    # 'laurent'), within their respective convergence floors.
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell
    from lumenairy.elements.pmm.twod_staggered import (
        pmm_efficiency_2d_staggered,
    )

    cell_y = np.array([[2.25, 1.0]], dtype=complex)             # (1,2)
    cell_sq = np.array([[2.25, 1.0], [2.25, 1.0]], dtype=complex)
    o_s, R_s, T_s = pmm_efficiency_2d_staggered(
        1.0, 1.0, cell_sq, 1.5, 1.0, 0.3, 0.75, degree=8, n_orders=3,
        polarization="te")
    m0_s = int(np.where((o_s[:, 0] == 0) & (o_s[:, 1] == 0))[0][0])
    for form in ("li", "laurent"):
        o, R, T = pmm_efficiency_2d_cell(
            1.0, 1.0, cell_y, 1.5, 1.0, 0.3, 0.75, polarization="te",
            degree=9, n_orders=3, formulation=form)
        m0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        assert abs(T[m0] - T_s[m0_s]) < 1e-2, form
        assert abs(R[m0] - R_s[m0_s]) < 1e-2, form


# --------------------------------------------------------------------------- #
# P3-37: staggered solver drops the dead operator matrices
# --------------------------------------------------------------------------- #

def test_p37_staggered_no_dead_operator_attrs():
    from lumenairy.elements.pmm.twod_staggered import Granet2DTransverseE

    eps_cell = np.array([[6.25, 1.0], [1.0, 1.0]], dtype=complex)
    sol = Granet2DTransverseE(1.2, 1.2, 2, 2, 4, eps_cell,
                              k0=2 * np.pi / 0.85)
    for dead in ("Curl", "Kzt", "Ktz", "G3", "Meps33"):
        assert not hasattr(sol, dead), dead
    for kept in ("Lmat", "Rmat", "Stt", "Schur", "Et_blocks"):
        assert hasattr(sol, kept), kept


def test_p37_staggered_solve_still_energy_conserving():
    from lumenairy.elements.pmm.twod_staggered import (
        pmm_efficiency_2d_staggered,
    )

    eps_cell = np.array([[6.25, 1.0], [1.0, 1.0]], dtype=complex)
    o, R, T = pmm_efficiency_2d_staggered(1.2, 1.2, eps_cell, 1.5, 1.0, 0.3,
                                          0.95, degree=5, n_orders=2)
    tot = float(R.sum() + T.sum())
    assert abs(tot - 1.0) < 5e-3


# --------------------------------------------------------------------------- #
# P3-38: dispersive OOP tensor layers match the static contract
# --------------------------------------------------------------------------- #

def _tilted_uniaxial_cell():
    c, s = np.cos(0.5), np.sin(0.5)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    t = Rm @ np.diag([2.25, 2.25, 4.0]) @ Rm.T      # exz/ezx != 0
    cell = np.zeros((9, 1, 3, 3), dtype=complex)
    cell[..., :, :] = t
    cell[:4] = np.eye(3) * 1.0
    return cell


def test_p38_dispersive_oop_tensor_matches_static():
    from lumenairy.elements.rcwa import RCWAStack

    static = _tilted_uniaxial_cell()
    s1 = RCWAStack(0.7e-6, n_orders=2, n_substrate=1.5)
    s1.add_layer(0.2e-6, eps_tensor_cell=static)
    s1.set_source(0.633e-6)
    o1, R1, T1 = s1.solve().efficiencies()

    s2 = RCWAStack(0.7e-6, n_orders=2, n_substrate=1.5)
    s2.add_layer(0.2e-6, eps_tensor_cell=lambda wl: static)
    s2.set_source(0.633e-6)
    # pre-fix: raised "the anisotropic path is the z-decoupled in-plane
    # tensor subset ..." even though the identical static layer solves.
    o2, R2, T2, _J2 = s2.solve_vs_wavelength(np.array([0.633e-6]))
    assert np.array_equal(np.asarray(o1), np.asarray(o2))
    assert np.max(np.abs(np.asarray(R1) - R2[0])) < 1e-13
    assert np.max(np.abs(np.asarray(T1) - T2[0])) < 1e-13
    worst = float((R2[0].sum(axis=1) + T2[0].sum(axis=1)).max())
    assert abs(worst - 1.0) < 1e-10                 # lossless energy closure


def test_p38_dispersive_oop_tensor_ezz_guard():
    from lumenairy.elements.rcwa import RCWAStack

    bad = _tilted_uniaxial_cell()
    bad[..., 2, 2] = 0.0
    s = RCWAStack(0.7e-6, n_orders=2, n_substrate=1.5)
    s.add_layer(0.2e-6, eps_tensor_cell=lambda wl: bad)
    s.set_source(0.633e-6)
    with pytest.raises(ValueError, match=r"\|e_zz\| > 0"):
        s.solve_vs_wavelength(np.array([0.633e-6]))


# --------------------------------------------------------------------------- #
# P3-27: JAX stack twin shared-eig half-spaces
# --------------------------------------------------------------------------- #

def test_p27_jax_stack_shared_eig_parity_and_grad():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from lumenairy.elements.pmm import PMMStack
    from lumenairy.elements.pmm import _jax_stack as twin

    P1, WL, T1 = 0.8e-6, 0.633e-6, 0.25e-6

    def _stack(nsub, jax_on):
        j = (lambda v: jnp.asarray(v)) if jax_on else (lambda v: v)
        st = PMMStack(P1, n_substrate=j(nsub), n_superstrate=1.0, degree=8)
        st.add_layer(T1, segments=[(0.4, j(4.0 + 0.5j)), (0.6, 1.0)])
        st.set_source(WL, angle=j(0.3))
        return st

    # STRUCTURAL: the traced assembly must run for the LAYERS only (1 here),
    # not for the two uniform half-spaces (pre-fix: 3 calls).
    calls = {"n": 0}
    orig = twin._jstack_assemble

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    twin._jstack_assemble = counting
    try:
        o1, R1, T1_, J1 = _stack(1.5, True).solve()
    finally:
        twin._jstack_assemble = orig
    assert calls["n"] == 1

    # forward parity vs the NumPy stack (shares the analytic uniform eig)
    o0, R0, T0, J0 = _stack(1.5, False).solve()
    assert np.array_equal(np.asarray(o0), np.asarray(o1))
    assert np.max(np.abs(R0 - np.asarray(R1))) < 1e-12
    assert np.max(np.abs(T0 - np.asarray(T1_))) < 1e-12
    assert np.max(np.abs(J0 - np.asarray(J1))) < 1e-12

    # half-space gradient flows analytically through q^2 = eps - mu
    def loss(n):
        _, _, T, _ = _stack(n, True).solve()
        return T[0].sum()

    g = float(jax.grad(loss)(jnp.asarray(1.5)))
    assert np.isfinite(g) and abs(g) > 0.0
    fd = (float(loss(jnp.asarray(1.5 + 1e-6)))
          - float(loss(jnp.asarray(1.5 - 1e-6)))) / 2e-6
    assert abs(g - fd) / max(abs(fd), 1e-12) < 1e-4, (g, fd)
