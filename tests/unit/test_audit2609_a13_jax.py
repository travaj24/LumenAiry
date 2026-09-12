"""WP-A13 (audit 2026-09-11) -- the PMM 2-D JAX twins.

* **G5 (P1), the twin half** -- ``_jax_stack2d._modes_projected`` carried the
  same legacy ``(Ex <- EpnF, Ey <- EpsF)`` assignment the NumPy stack did, so
  the differentiable stack broke 90 deg rotation invariance under its DEFAULT
  ``formulation='li'`` exactly as the concrete one did.
* **G10 (P2)** -- the twins still built the dense Kronecker projector pair
  ``Tp = kron(Ty, Tx)`` + ``pinv(Tp)`` that the NumPy F5 audit deleted (an
  ``(Nf, N)`` pair, ~1.4 GB at the documented ``_MAX_NODAL_DOF`` ceiling with
  ``n_orders = 11``, plus an ``O(N Nf^2)`` pinv), and ``_static_prep``
  additionally formed a dense ``N x N`` ``diag(1/Mdiag)``.

Follows the house pattern for the optional JAX dependency
(``pytest.importorskip``, x64 enabled per CONVENTIONS sec 10).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStackHybrid,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*energy closure.*")

_WL = 1.0e-6
_P = 0.47e-6
_DEP = 0.3e-6


@pytest.fixture(scope="module")
def jnp():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as _jnp
    return _jnp


def _stripes(S=24):
    cx = np.full((S, S), 1.0 + 0j)
    cx[6:18, :] = 12.25
    return cx, cx.T.copy()


def _jax_stack(jnp, cell, *, formulation="li", symmetry=False):
    st = PMM2DStackHybrid(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                          degree=7, n_orders=3, formulation=formulation,
                          symmetry=symmetry)
    st.add_layer(jnp.asarray(_DEP), eps_cell=cell)   # traced thickness
    st.set_source(_WL, theta=0.0, phi=0.0)
    return st.solve()


def test_g5_jax_stack_twin_routes_the_per_slot_li_operators(jnp):
    """The differentiable stack must satisfy the same 90 deg rotation identity
    the concrete one does.

    BAR 1e-11 on ``T``.  Derivation: the two orientations are transposed
    problems through the same ``jnp.linalg.eig`` custom-VJP, so the floor is
    their independent round-off; MEASURED 3.06e-15.  The pre-fix twin carried
    the identical legacy assignment to the NumPy stack, whose measured break on
    this fixture was ``max|dT| = 4.98e-03`` -- 12 decades above the bar.
    """
    cx, cy = _stripes()
    ox_, _Rx, Tx, _Jx = _jax_stack(jnp, cx)
    oy_, _Ry, Ty, _Jy = _jax_stack(jnp, cy)
    ox_ = np.asarray(ox_)
    oy_ = np.asarray(oy_)
    idx = {tuple(int(v) for v in row): i for i, row in enumerate(oy_)}
    perm = [idx[(int(n), int(m))] for m, n in ox_]
    dT = float(np.max(np.abs(np.asarray(Tx)[0] - np.asarray(Ty)[1][perm])))
    assert dT < 1e-11
    assert float(np.max(np.asarray(Tx))) > 0.5


@pytest.mark.parametrize("which", ["x", "y"])
def test_g5_jax_stack_twin_matches_the_numpy_stack(jnp, which):
    """NumPy / JAX forward parity on BOTH orientations -- the routing fix must
    land on both sides of the twin pair, not just one.

    BAR 1e-11.  MEASURED ``max|dR| <= 2.4e-14``, ``max|dT| <= 2.5e-14``,
    ``max|dJ| <= 1.0e-13`` (the twin's documented ~1e-14 parity)."""
    cx, cy = _stripes()
    cell = cx if which == "x" else cy
    oj, Rj, Tj, Jj = _jax_stack(jnp, cell)
    sn = PMM2DStackHybrid(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                          degree=7, n_orders=3, formulation="li",
                          symmetry=False)
    sn.add_layer(_DEP, eps_cell=cell)
    sn.set_source(_WL, theta=0.0, phi=0.0)
    _on, Rn, Tn, Jn = sn.solve()
    assert np.max(np.abs(np.asarray(Rj) - Rn)) < 1e-11
    assert np.max(np.abs(np.asarray(Tj) - Tn)) < 1e-11
    assert np.max(np.abs(np.asarray(Jj) - Jn)) < 1e-10


def test_g10_the_twins_no_longer_build_the_dense_kronecker_pair():
    """Structural gate: the geometry preps must expose the four PER-AXIS
    projectors and NOT the dense ``(Nf, N)`` pair.  The numbers below are the
    arithmetic of the thing this removes, not a timing."""
    import lumenairy.elements.pmm._jax_twod as jt
    lay = np.zeros((8, 8), dtype=np.int64)
    lay[2:6, 2:6] = 1
    for st in (jt._static_prep(1e-6, 1e-6, 0.3e-6, 0.6e-6, 0.2e-6, 0.7e-6,
                               5, 1, False, 3),
               jt._static_prep_cell(1e-6, 1e-6, lay, 5, 1, False, 3)):
        assert "Tp" not in st and "Tpinv" not in st
        for k in ("Tx", "Txp", "Ty", "Typ", "Nx", "Ny", "NxO", "NyO"):
            assert k in st, k
        # the per-axis pieces really are the small ones: (NxO, Nx), not (Nf, N)
        assert st["Tx"].shape == (st["NxO"], st["Nx"])
        assert st["Ty"].shape == (st["NyO"], st["Ny"])
        dense = 2 * (st["NxO"] * st["NyO"]) * (st["Nx"] * st["Ny"]) * 16
        axes = sum(a.nbytes for a in (st["Tx"], st["Txp"], st["Ty"],
                                      st["Typ"]))
        assert axes < dense


def test_g10_the_frozen_operators_now_equal_the_numpy_path(jnp):
    """A parity BONUS of the factorization: the dense ``Tp @ Gx0 @ Tpinv``
    spelling carried an extra ``Ty Typ`` factor the NumPy path does not, which
    is where the twin's largest measured parity gap lived (``T`` RMS relative
    8.47e-11 in the audit).  Built the factorized way, the frozen constants are
    BIT-IDENTICAL to ``_scalar_projected_ops``'s."""
    import lumenairy.elements.pmm._jax_twod as jt
    from lumenairy.elements.pmm.twod import _build_axis, _scalar_projected_ops
    st = jt._static_prep(1e-6, 1e-6, 0.3e-6, 0.6e-6, 0.2e-6, 0.7e-6, 5, 1,
                         False, 3)
    ax = _build_axis(1e-6, [0.3e-6, 0.6e-6], 5, 1, False)
    ay = _build_axis(1e-6, [0.2e-6, 0.7e-6], 5, 1, False)
    o3 = np.arange(-3, 4)
    lops = _scalar_projected_ops(ax, ay, np.full((3, 3), 2.0 + 0j), o3, o3,
                                 1e-6, 1e-6)
    assert np.array_equal(st["Gx0F"], lops["Gx0F"])
    assert np.array_equal(st["Gy0F"], lops["Gy0F"])
    assert np.array_equal(st["IprojF"], lops["IpxF"])


def test_g10_forward_parity_and_gradients_survive_the_factorization(jnp):
    """The einsum sandwich must reproduce the dense one and stay
    differentiable.

    BARS: 1e-10 relative on the forward (the twin's documented ~1e-14 …
    1e-11 parity; MEASURED ``R`` 2.9e-14, ``T`` 1.6e-13 -- the ``T`` figure is
    2.5 decades BETTER than the audit's pre-fix 8.47e-11); 1e-4 relative on
    AD-vs-central-FD, the gate the JAX twins are held to elsewhere in this
    suite (MEASURED 1.0e-07)."""
    import jax
    bounds = (0.115e-6, 0.355e-6)
    o_n, R_n, T_n = pmm_efficiency_2d(_P, _P, 12.25, 1.0, bounds, bounds, 1.45,
                                      1.0, _DEP, _WL, degree=5, n_orders=3,
                                      polarization="te")[:3]
    o_j, R_j, T_j = pmm_efficiency_2d(_P, _P, jnp.asarray(12.25 + 0j),
                                      jnp.asarray(1.0 + 0j), bounds, bounds,
                                      1.45, 1.0, _DEP, _WL, degree=5,
                                      n_orders=3, polarization="te")[:3]
    for a, b in ((R_n, R_j), (T_n, T_j)):
        num = float(np.sqrt(np.mean((np.asarray(b) - a) ** 2)))
        den = max(float(np.sqrt(np.mean(a ** 2))), 1e-300)
        assert num / den < 1e-10

    def f(e):
        return jnp.sum(pmm_efficiency_2d(
            _P, _P, e, jnp.asarray(1.0 + 0j), bounds, bounds, 1.45, 1.0, _DEP,
            _WL, degree=5, n_orders=3, polarization="te")[2]).real

    e0 = jnp.asarray(12.25 + 0j)
    g = float(np.real(jax.grad(f)(e0)))
    h = 1e-6
    fd = float((f(e0 + h) - f(e0 - h)) / (2 * h))
    assert abs(g - fd) / max(abs(fd), 1e-300) < 1e-4


def test_g10_cell_and_jones_twins_keep_numpy_parity(jnp):
    """The same factorization in ``_static_prep_cell`` /
    ``_jax_twod_jones._proj``.  BAR 1e-10 (MEASURED 4.0e-14 on the cell entry's
    ``T`` and 2.7e-12 on the Jones)."""
    from lumenairy.elements.pmm import pmm_jones_2d
    S = 8
    cell = np.full((S, S), 1.0 + 0j)
    cell[2:6, 2:6] = 12.25
    lay = np.zeros((S, S), dtype=np.int64)
    lay[2:6, 2:6] = 1
    _o, R_n, T_n = pmm_efficiency_2d_cell(0.9e-6, 0.9e-6, cell, 1.45, 1.0,
                                          _DEP, _WL, degree=5, n_orders=3,
                                          polarization="te")[:3]
    _o, R_j, T_j = pmm_efficiency_2d_cell(0.9e-6, 0.9e-6, jnp.asarray(cell),
                                          1.45, 1.0, _DEP, _WL, degree=5,
                                          n_orders=3, polarization="te",
                                          region_layout=lay)[:3]
    assert np.max(np.abs(np.asarray(R_j) - R_n)) < 1e-10
    assert np.max(np.abs(np.asarray(T_j) - T_n)) < 1e-10
    tc = np.zeros((S, S, 3, 3), dtype=complex)
    tc[...] = np.eye(3)
    tc[2:6, 2:6] = 12.25 * np.eye(3)
    Jn = pmm_jones_2d(0.9e-6, 0.9e-6, tc, 1.45, 1.0, _DEP, _WL, degree=5,
                      n_orders=3, formulation="laurent", symmetry=False)[3]
    Jj = pmm_jones_2d(0.9e-6, 0.9e-6, jnp.asarray(tc), 1.45, 1.0, _DEP, _WL,
                      degree=5, n_orders=3, formulation="laurent",
                      region_layout=lay)[3]
    assert np.max(np.abs(np.asarray(Jj) - Jn)) < 1e-10


def test_g10_numpy_only_options_refuse_loudly_on_the_jax_path(jnp):
    """``truncation='circular'``, ``return_jones_transmission`` and
    ``formulation='auto'``'s ``fff_nv`` arm are NumPy-only; the JAX dispatch
    must say so rather than silently returning the rectangular / 4-tuple /
    Laurent answer (the failure shape this partition's slant defect had)."""
    from lumenairy.elements.pmm import pmm_jones_2d
    S = 8
    tc = np.zeros((S, S, 3, 3), dtype=complex)
    tc[...] = np.eye(3)
    tc[2:6, 2:6] = 12.25 * np.eye(3)
    lay = np.zeros((S, S), dtype=np.int64)
    lay[2:6, 2:6] = 1
    kw = dict(degree=5, n_orders=3, region_layout=lay)
    with pytest.raises(NotImplementedError, match="circular.*NumPy only"):
        pmm_jones_2d(0.9e-6, 0.9e-6, jnp.asarray(tc), 1.45, 1.0, _DEP, _WL,
                     truncation="circular", **kw)
    with pytest.raises(NotImplementedError,
                       match="return_jones_transmission.*NumPy only"):
        pmm_jones_2d(0.9e-6, 0.9e-6, jnp.asarray(tc), 1.45, 1.0, _DEP, _WL,
                     return_jones_transmission=True, **kw)
    # 'auto' resolves to 'laurent' on the JAX path rather than raising
    a = pmm_jones_2d(0.9e-6, 0.9e-6, jnp.asarray(tc), 1.45, 1.0, _DEP, _WL,
                     formulation="auto", **kw)[3]
    b = pmm_jones_2d(0.9e-6, 0.9e-6, jnp.asarray(tc), 1.45, 1.0, _DEP, _WL,
                     formulation="laurent", **kw)[3]
    assert np.array_equal(np.asarray(a), np.asarray(b))
    # ... and the stack refuses circular truncation the same way
    cell = np.full((S, S), 1.0 + 0j)
    cell[2:6, 2:6] = 12.25
    st = PMM2DStackHybrid(0.9e-6, 0.9e-6, n_superstrate=1.0, n_substrate=1.45,
                          degree=5, n_orders=3, truncation="circular")
    st.add_layer(jnp.asarray(_DEP), eps_cell=cell)
    st.set_source(_WL, theta=0.0, phi=0.0)
    with pytest.raises(NotImplementedError, match="circular.*NumPy"):
        st.solve()
