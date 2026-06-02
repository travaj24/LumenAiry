"""v5.10.3 RCWA 2-D differentiability (audit P1, single-layer).

The stable-eig custom-VJP is dimension-agnostic, so ``rcwa_efficiency_2d``
differentiates a 2-D (crossed-grating / metasurface) figure-of-merit through the
full vector solve -- the basis for 2-D inverse design.  This pins it with a
finite-difference check.

REQUIRES JAX double precision: RCWA's eigenproblem is ill-conditioned in
single precision and JAX silently truncates complex128 -> complex64 unless
``jax_enable_x64`` is set (the solver warns; see ``_warn_if_jax_f32``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.rcwa import rcwa_efficiency_2d  # noqa: E402

S = 24


def _metal_cell(eps_hi):
    x = jnp.linspace(-0.5, 0.5, S)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    return jnp.where((jnp.abs(X) < 0.25) & (jnp.abs(Y) < 0.25), eps_hi,
                     2.1 + 0j)


def _sumT_eps(eps_hi):
    o, R, T = rcwa_efficiency_2d(0.5e-6, 0.5e-6, _metal_cell(eps_hi), 1.5, 1.0,
                                 0.2e-6, 0.6e-6, theta=0.001, n_orders_x=4,
                                 n_orders_y=4, polarization="tm")
    return jnp.real(jnp.sum(T))


def _sumT_depth(depth):
    o, R, T = rcwa_efficiency_2d(0.5e-6, 0.5e-6, _metal_cell(6.0 + 0.5j), 1.5,
                                 1.0, depth, 0.6e-6, theta=0.001, n_orders_x=4,
                                 n_orders_y=4, polarization="tm")
    return jnp.real(jnp.sum(T))


def test_2d_gradient_wrt_cell_permittivity_matches_fd():
    e0 = jnp.asarray(6.0 + 0.5j)
    g = jax.grad(_sumT_eps, holomorphic=False)(e0)
    h = 1e-5
    fd = (_sumT_eps(e0 + h) - _sumT_eps(e0 - h)) / (2 * h)
    assert np.isfinite(float(np.real(g)))
    assert np.allclose(np.real(g), np.real(fd), rtol=1e-3, atol=1e-6)


def test_2d_gradient_wrt_depth_matches_fd():
    d0 = jnp.asarray(0.2e-6)
    g = jax.grad(_sumT_depth)(d0)
    h = 1e-9
    fd = (_sumT_depth(d0 + h) - _sumT_depth(d0 - h)) / (2 * h)
    assert np.allclose(float(g), float(fd), rtol=1e-3)


# ---------------------------------------------------------------------------
# multi-layer RCWAStack autodiff (audit P1 completion)
# ---------------------------------------------------------------------------

import lumenairy as la  # noqa: E402

SS = 48


def _stack_sumT(eps_hi, depth1):
    x = (jnp.arange(SS) + 0.5) / SS - 0.5
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    cell = jnp.where((jnp.abs(X) < 0.25) & (jnp.abs(Y) < 0.25), eps_hi,
                     2.1 + 0j)
    st = la.RCWAStack(0.5e-6, period_y=0.5e-6, n_substrate=1.5, n_orders=6,
                      n_orders_y=6)
    st.add_layer(depth1, eps_cell=cell).add_layer(0.1e-6, eps=2.1)
    st.set_source(0.6e-6, theta=0.001)
    o, R, T = st.solve().efficiencies()
    return jnp.real(jnp.sum(T))


def test_stack_gradient_wrt_cell_permittivity_matches_fd():
    e0 = jnp.asarray(6.0 + 0.5j)
    g = jax.grad(lambda e: _stack_sumT(e, 0.2e-6), holomorphic=False)(e0)
    h = 1e-5
    fd = (_stack_sumT(e0 + h, 0.2e-6) - _stack_sumT(e0 - h, 0.2e-6)) / (2 * h)
    assert np.isfinite(float(np.real(g)))
    assert np.allclose(np.real(g), np.real(fd), rtol=1e-3, atol=1e-6)


def test_stack_gradient_wrt_layer_depth_matches_fd():
    d0 = jnp.asarray(0.2e-6)
    g = jax.grad(lambda d: _stack_sumT(6.0 + 0.5j, d))(d0)
    h = 1e-9
    fd = (_stack_sumT(6.0 + 0.5j, d0 + h)
          - _stack_sumT(6.0 + 0.5j, d0 - h)) / (2 * h)
    assert np.allclose(float(g), float(fd), rtol=1e-2)
