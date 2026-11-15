"""``RCWAStack`` ``formulation='auto'`` resolves a JAX ``eps_cell`` exactly as
its NumPy twin.

The rule used to resolve by array TYPE: a NumPy cell to ``'li'`` and the same
values as a JAX array (concrete or traced) to ``'laurent'``.  One physical
layer then had two answers (0.162 apart on a smooth-edged metal strip) and an
AD gradient of the JAX function disagreed in SIGN with a finite difference of
the NumPy one (-75.5 against +149.6).  A concrete JAX cell is now read like a
NumPy cell, and a traced one resolves to ``'li'`` as well -- the inverse-rule
blocks are ``xp``-generic Toeplitz builds and inverses, so they trace.  The
one case a tracer cannot answer (a cell with several y-samples on a 1-D
stack, where y-variation decides the rule) raises and asks for an explicit
formulation.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

P, WL, S = 600e-9, 1000e-9, 400
X = (np.arange(S) + 0.5) / S


def _cell(w, xp):
    """A smooth-edged metal strip of width ``w`` (differentiable in ``w``)."""
    x = xp.asarray(X)
    s = (1.0 / (1.0 + xp.exp(-(x - (0.5 - w / 2)) / 0.004))
         - 1.0 / (1.0 + xp.exp(-(x - (0.5 + w / 2)) / 0.004)))
    return (1.0 + (-20 + 2j - 1.0) * s)[:, None]


def _solve(cell):
    st = la.RCWAStack(P, n_superstrate=1.0, n_substrate=1.5, n_orders=20)
    st.add_layer(80e-9, eps_cell=cell)
    st.add_layer(30e-9, eps=2.1)
    st.add_layer(100e-9, eps=-30 + 3j)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = st.set_source(WL, theta=0.2).solve()
    return r, st.layers[0].formulation


def test_concrete_jax_cell_resolves_like_numpy():
    rn, fn = _solve(_cell(0.4, np))
    rj, fj = _solve(_cell(jnp.asarray(0.4), jnp))
    assert fn == fj == "li"
    dJ = np.abs(np.asarray(rn.jones_reflection())
                - np.asarray(rj.jones_reflection())).max()
    assert dJ < 1e-11                       # measured 1.8e-13


def test_ad_gradient_matches_numpy_finite_difference():
    """Measured: AD 149.656, central FD (h 1e-5) 149.642 (the FD step's own
    error); before, AD -75.5."""
    def f(w):
        r, _ = _solve(_cell(w, jnp))
        return jnp.abs(r.jones_reflection()[0, 0]) ** 2
    g = float(jax.grad(f)(jnp.asarray(0.4)))
    h = 1e-5
    fd = (abs(np.asarray(_solve(_cell(0.4 + h, np))[0]
                         .jones_reflection())[0, 0]) ** 2
          - abs(np.asarray(_solve(_cell(0.4 - h, np))[0]
                           .jones_reflection())[0, 0]) ** 2) / (2 * h)
    assert abs(g - fd) < 1e-3 * abs(fd)


def test_traced_cell_resolves_to_li():
    seen = []

    def f(w):
        r, form = _solve(_cell(w, jnp))
        seen.append(form)
        return jnp.abs(r.jones_reflection()[0, 0]) ** 2
    jax.jit(f)(jnp.asarray(0.4))
    assert seen == ["li"]


def test_traced_cell_with_y_samples_on_a_1d_stack_asks_for_a_formulation():
    def f(t):
        cell = jnp.ones((32, 4), dtype=jnp.complex128) * (2.0 + t)
        st = la.RCWAStack(P, n_orders=5)
        st.add_layer(80e-9, eps_cell=cell)
        return 0.0
    with pytest.raises(ValueError, match="formulation='li' or 'laurent'"):
        jax.grad(f)(0.1)


def test_two_d_traced_cell_gradient_matches_numpy():
    """2-D stack: the JAX and NumPy cells both run 'li'; AD vs FD 1.8e-6
    relative (measured)."""
    S2 = 32
    xx = (np.arange(S2) + 0.5) / S2
    XX, YY = np.meshgrid(xx, xx, indexing="ij")
    base = np.where((np.abs(XX - 0.5) < 0.25) & (np.abs(YY - 0.5) < 0.2),
                    6.0 + 0j, 1.0 + 0j)
    dirn = np.where((np.abs(XX - 0.7) < 0.05) & (np.abs(YY - 0.5) < 0.2),
                    1.0, 0.0)

    def f(t, xp):
        st = la.RCWAStack(P, period_y=P, n_superstrate=1.0, n_substrate=1.5,
                          n_orders=3, n_orders_y=3)
        st.add_layer(150e-9, eps_cell=xp.asarray(base) + t * xp.asarray(dirn))
        st.add_layer(50e-9, eps=2.1)
        st.set_source(WL, theta=0.1, phi=0.2)
        _o, R, T = st.solve().efficiencies()
        return R[0].sum() + 0.5 * T[1].sum()
    g = float(jax.grad(lambda t: f(t, jnp))(jnp.asarray(0.3)))
    fd = (float(f(0.3 + 1e-6, np)) - float(f(0.3 - 1e-6, np))) / 2e-6
    assert abs(g - fd) < 1e-4 * abs(fd)
