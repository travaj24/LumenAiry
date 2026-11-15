"""PMMStack with GAIN media (public ``Im(eps) < 0``).

A gain layer's modes are split into forward / backward by the analytic
continuation from the lossy side (``_core._forward_growth_flip``): a mode
whose root is nearer the real axis follows its power flow, every other mode
its decay.  Before that rule the flux test handed the GROWING root of every
evanescent gain mode to the forward set and the dense and conical solves
returned ``R + T`` of 1e1 .. 1e4 with no warning (patterned gain ridge,
diagonal-tensor gain layer, any gain layer at conical incidence).

References: RCWAStack (inverse rule, ``n_orders`` 40, which agrees with
``n_orders`` 80 to ~1.5e-5 on these cells) and an analytic transfer matrix
for all-uniform stacks.  A gain SUBSTRATE is refused (its transmitted power
was dropped silently).
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.pmm import _core

P, WL, NSUB = 500e-9, 1000e-9, 1.45
X = (np.arange(4000) + 0.5) / 4000


def _pmm(case, theta, phi, degree=14):
    st = la.PMMStack(P, n_substrate=NSUB, n_superstrate=1.0, degree=degree)
    if case == "ridge":
        st.add_layer(80e-9, segments=[(0.4, 12.0 - 0.05j), (0.6, 1.0)])
        st.add_layer(100e-9, eps=2.1)
    elif case == "tensor":
        st.add_layer(80e-9, segments=[(0.4, 12.0), (0.6, 1.0)])
        st.add_layer(100e-9, eps=np.diag([2.25 - 0.05j, 2.3 - 0.05j,
                                          2.25 - 0.05j]))
        st.add_layer(100e-9, eps=2.1)
    else:                                           # "under": iso gain layer
        st.add_layer(80e-9, segments=[(0.4, 12.0), (0.6, 1.0)])
        st.add_layer(100e-9, eps=2.25 - 0.05j)
        st.add_layer(100e-9, eps=2.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.set_source(WL, theta=theta, phi=phi).solve()
    return np.asarray(J), np.sum(R, 1) + np.sum(T, 1)


def _rcwa(case, theta, phi, n_orders=40):
    st = la.RCWAStack(P, n_superstrate=1.0, n_substrate=NSUB,
                      n_orders=n_orders)
    ridge = 12.0 - 0.05j if case == "ridge" else 12.0
    st.add_layer(80e-9, eps_cell=np.where(X < 0.4, ridge, 1.0)
                 .astype(complex)[:, None], formulation="li")
    if case == "tensor":
        st.add_layer(100e-9, eps=np.diag([2.25 - 0.05j, 2.3 - 0.05j,
                                          2.25 - 0.05j]))
    elif case == "under":
        st.add_layer(100e-9, eps=2.25 - 0.05j)
    st.add_layer(100e-9, eps=2.1)
    r = st.set_source(WL, theta=theta, phi=phi).solve()
    _o, R, T = r.efficiencies()
    return np.asarray(r.jones_reflection()), np.sum(R, 1) + np.sum(T, 1)


_INC = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.6)]


@pytest.mark.parametrize("case", ["ridge", "tensor", "under"])
@pytest.mark.parametrize("theta,phi", _INC)
def test_gain_stack_matches_rcwa(case, theta, phi):
    """Measured (degree 14): 1.8e-4 .. 1.9e-4 from RCWAStack in every cell
    (shrinking with degree, 7e-5 at degree 20).  Before: 0.34 .. 20 apart,
    R + T up to 409."""
    Jp, sp = _pmm(case, theta, phi)
    Jr, sr = _rcwa(case, theta, phi)
    assert np.abs(Jp - Jr).max() < 1e-3
    assert np.abs(sp - sr).max() < 1e-3


def test_gain_ridge_converges_in_degree():
    """The PMM error to the RCWA answer falls with degree (4.4e-4 / 1.8e-4 /
    7.2e-5 at degree 10 / 14 / 20, measured), as on a passive ridge."""
    Jr, _ = _rcwa("ridge", 0.4, 0.6)
    errs = [np.abs(_pmm("ridge", 0.4, 0.6, d)[0] - Jr).max()
            for d in (10, 14, 20)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 2e-4


def _tmm_r(eps_list, d_list, n_sup, n_sub, wl):
    """Normal-incidence analytic reflection (tangential-E ratio)."""
    k0 = 2 * np.pi / wl
    M = np.eye(2, dtype=complex)
    for e, d in zip(eps_list, d_list):
        n = np.sqrt(complex(e))
        if n.imag < 0 and n.real < 0:
            n = -n
        dl = n * k0 * d
        M = M @ np.array([[np.cos(dl), -1j * np.sin(dl) / n],
                          [-1j * n * np.sin(dl), np.cos(dl)]])
    B, C = M @ np.array([1.0, n_sub])
    return (n_sup * B - C) / (n_sup * B + C)


@pytest.mark.parametrize("phi", [0.0, 0.6])
def test_uniform_tensor_gain_layer_matches_transfer_matrix(phi):
    """An all-uniform stack with a DIAGONAL-tensor gain layer (the dense
    path -- the uniform fast path takes isotropic layers only) against the
    analytic transfer matrix per polarization at normal incidence (phi only
    rotates the plane of incidence there; it routes through the conical
    solve when nonzero).  Measured 1e-15 .. 1e-14."""
    st = la.PMMStack(P, n_substrate=NSUB, degree=10)
    st.add_layer(80e-9, eps=3.5)
    st.add_layer(100e-9, eps=np.diag([2.25 - 0.05j, 2.3 - 0.05j,
                                      2.25 - 0.05j]))
    st.add_layer(100e-9, eps=2.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T, J = st.set_source(WL, theta=0.0, phi=phi).solve()
    rx = _tmm_r([3.5, 2.25 - 0.05j, 2.1], [80e-9, 100e-9, 100e-9], 1.0,
                NSUB, WL)
    ry = _tmm_r([3.5, 2.3 - 0.05j, 2.1], [80e-9, 100e-9, 100e-9], 1.0,
                NSUB, WL)
    # the lab-frame Jones of a layer with x / y principal axes at normal
    # incidence is diag(r_x, r_y) whatever the plane of incidence
    J = np.asarray(J)
    assert abs(J[0, 1]) < 1e-8 and abs(J[1, 0]) < 1e-8
    assert abs(J[0, 0] - rx) < 1e-8
    assert abs(J[1, 1] - ry) < 1e-8


def test_fail_before_switch_reproduces_the_defect(monkeypatch):
    """``_PMM_GAIN_SECTOR = False`` restores the bare flux rule, and with it
    the growing-evanescent forward set (R + T ~ 10 on the gain ridge)."""
    monkeypatch.setattr(_core, "_PMM_GAIN_SECTOR", False)
    _J, s = _pmm("ridge", 0.0, 0.0)
    assert float(np.max(s)) > 1.5


def test_passive_stack_bit_identical_with_the_rule_on_and_off(monkeypatch):
    st_kw = dict(theta=0.3, phi=0.0)

    def run():
        st = la.PMMStack(P, n_substrate=NSUB, degree=10)
        st.add_layer(80e-9, segments=[(0.4, 12.0 + 0.05j), (0.6, 1.0)])
        st.add_layer(100e-9, eps=np.diag([2.25, 2.3, 2.25]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(st.set_source(WL, **st_kw).solve()[3])
    a = run()
    monkeypatch.setattr(_core, "_PMM_GAIN_SECTOR", False)
    b = run()
    assert np.array_equal(a, b)


def test_gain_substrate_is_refused():
    st = la.PMMStack(P, n_substrate=1.45 - 0.01j, degree=8)
    st.add_layer(100e-9, eps=2.1)
    with pytest.raises(ValueError, match="gain substrate"):
        st.set_source(WL, theta=0.0).solve()
    with pytest.raises(ValueError, match="gain substrate"):
        st.solve_vs_wavelength([WL])
    st2 = la.PMMStack(P, n_substrate=1.45 - 0.01j, degree=8)
    st2.add_layer(100e-9, segments=[(0.4, "M"), (0.6, 1.0)])
    with pytest.raises(ValueError, match="gain substrate"):
        st2.prepare().solve(wavelength=WL, materials={"M": 3.0})


def test_jax_twin_matches_numpy_on_a_gain_ridge():
    """The JAX twin shares the selector (a TRACED passivity flag takes the
    same branch elementwise): measured 1e-13 from the NumPy solve."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def f(e):
        st = la.PMMStack(P, n_substrate=NSUB, degree=12)
        st.add_layer(80e-9, segments=[(0.4, e), (0.6, 1.0)])
        st.add_layer(100e-9, eps=2.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return st.set_source(WL, theta=0.3).solve()[3]
    Jn = np.asarray(f(12.0 - 0.05j))
    Jj = np.asarray(f(jnp.asarray(12.0 - 0.05j)))
    assert np.abs(Jn - Jj).max() < 1e-9
