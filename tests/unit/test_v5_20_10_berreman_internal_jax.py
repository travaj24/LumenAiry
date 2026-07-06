"""Differentiable (JAX) Berreman internal observables -- layer_absorption /
internal_field.

``BerremanStack.solve(retain_internal=True)`` now works under a trace: the
per-layer modal-amplitude reconstruction from the retained partial cascades is
backend-generic S-matrix algebra, so ``layer_absorption()`` and
``internal_field()`` intensities differentiate w.r.t. a lossy layer's Im(eps),
the layer indices, thicknesses and the source -- a natural objective for LC
retarder / magneto-optic / lossy dichroic film design.  These tests pin the
gradient vs central FD, the traced-vs-concrete value parity, the energy-closure
invariant, and that the out-of-plane-oblique retain still raises.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.berreman import BerremanStack


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _stack(ei):
    lc = np.diag([1.7 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex)
    st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    st.add_layer(0.12e-6, eps=lc)
    st.add_layer(0.20e-6, eps=(2.4 ** 2 + 1j * ei))       # lossy film
    st.add_layer(0.10e-6, eps=2.1)
    R, T, _J = st.set_source(0.55e-6, theta=0.2).solve(retain_internal=True)
    return st, R, T


def test_layer_absorption_value_parity_and_closure():
    _jax()
    import jax.numpy as jnp
    stc, _Rc, _Tc = _stack(0.3)
    stj, Rj, Tj = _stack(jnp.asarray(0.3))
    Ac = np.asarray(stc.layer_absorption())
    Aj = np.asarray(stj.layer_absorption())
    assert np.max(np.abs(Aj - Ac)) < 1e-12                # traced == concrete
    # honest cross-machinery closure: sum_i A_i == 1 - R - T per pol
    closure = Aj.sum(0) - (1.0 - np.asarray(Rj) - np.asarray(Tj))
    assert np.max(np.abs(closure)) < 1e-9


def test_layer_absorption_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    def loss(ei):
        st, _R, _T = _stack(ei)
        return jnp.asarray(st.layer_absorption())[1, 0]     # lossy layer, pol x

    g = float(jax.grad(loss)(jnp.asarray(0.3)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.3 + h)))
          - float(loss(jnp.asarray(0.3 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9


def test_internal_field_intensity_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    def loss(ei):
        st, _R, _T = _stack(ei)
        f = st.internal_field(0.20e-6, component="E")       # inside the film
        return jnp.abs(f["Ex"]) ** 2

    g = float(jax.grad(loss)(jnp.asarray(0.3)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.3 + h)))
          - float(loss(jnp.asarray(0.3 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9


def test_concrete_retain_internal_unchanged():
    st, _R, _T = _stack(0.3)                                # plain numpy
    A = st.layer_absorption()
    assert A.shape[1] == 2 and np.all(np.asarray(A) > -1e-9)
    f = st.internal_field(0.20e-6, component="all")
    assert set(f) >= {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}


def test_offplane_oblique_retain_raises_under_trace():
    _jax()
    import jax.numpy as jnp
    # tilted uniaxial (out-of-plane) at oblique incidence + traced index
    ct, st_ = np.cos(np.deg2rad(35.0)), np.sin(np.deg2rad(35.0))
    Ry = np.array([[ct, 0, st_], [0, 1, 0], [-st_, 0, ct]])
    oop = Ry @ np.diag([1.8 ** 2, 1.5 ** 2, 1.5 ** 2]) @ Ry.T
    stk = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    stk.add_layer(0.2e-6, eps=oop)
    stk.set_source(0.55e-6, theta=0.3)
    stk.n_sub = jnp.asarray(1.5 + 0j)                       # force the trace
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        stk.solve(retain_internal=True)
