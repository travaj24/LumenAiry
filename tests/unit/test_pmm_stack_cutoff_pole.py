"""A uniform medium EXACTLY at a cut-off of the element grid.

When a uniform layer's ``eps`` equals a geometric eigenvalue ``mu`` of the
grid, that mode has ``q^2 = eps - mu = 0``: the small-eigenvalue clamp
replaced its ``lam`` but its analytic partner column stayed ``w q^2 = 0``, so
the mode-match into the layer was exactly singular.  The NumPy solve fell
back to the dense kernels and raised ``LinAlgError``; the JAX twin, which
cannot branch on finiteness under a trace, returned NaN (eager: the energy
tripwire's "non-finite" error; ``jax.jit`` / ``jax.grad``: NaN silently).
``q^2`` now takes the value the clamped ``lam`` implies (``_core._safe_q2``),
and the solve returns the limit of nearby permittivities.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _pmm_union_grid,
    _tensor3_dict,
    _uniform_geo_eig,
)

P, WL, DEG = 500e-9, 1500e-9, 8
SEGS = [(0.4, -20 + 2j), (0.6, 1.0)]


def _build(e, t=50e-9):
    st = la.PMMStack(P, n_substrate=1.5, degree=DEG)
    st.add_layer(40e-9, segments=SEGS)
    st.add_layer(t, eps=2.1)
    st.add_layer(30e-9, eps=e)
    return st.set_source(WL, theta=0.0)


def _grid_mu():
    st = _build(3.0)
    uw, _ = _pmm_union_grid([L[1] for L in st._layers], st.min_feature / P)
    mats = _build_sem_tensor_segments(
        P, uw, [_tensor3_dict(np.eye(3, dtype=complex))] * len(uw), DEG,
        st.n_el, st._effective_grade(WL))
    return st, uw, mats


def _solve(st):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(st.solve()[3])


def test_numpy_solve_at_an_exact_grid_cutoff_is_the_limit():
    _st, _uw, mats = _grid_mu()
    mu = _uniform_geo_eig(mats, 2 * np.pi / WL, 0.0)[0]
    e = complex(mu[int(np.argmin(np.abs(mu - 9.0)))])     # the +/-1 cut-off
    J = _solve(_build(e))
    J_lim = _solve(_build(e * (1 + 1e-9)))
    assert np.all(np.isfinite(J))
    assert np.abs(J - J_lim).max() < 1e-8                 # measured 2.6e-10


def test_jax_twin_at_its_exact_grid_cutoff_is_finite_and_differentiable():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from lumenairy.elements.pmm import _jax_stack as JS
    from lumenairy.elements.rcwa._core import _jax_twin_eig
    st, uw, mats = _grid_mu()
    k0 = 2 * np.pi / WL
    static = JS._jstack_static(P, uw, DEG, st.n_el, st._effective_grade(WL))
    S0g, Kg, _ = JS._jstack_geo_ops(static)
    Kx2 = (1.0 / (k0 * k0)) * (jnp.linalg.inv(jnp.asarray(S0g))
                               @ jnp.asarray(Kg))
    mj = np.asarray(_jax_twin_eig(Kx2, None, jnp.asarray(S0g), (0.0,))[0])
    e = complex(mj[int(np.argmin(np.abs(mj - 9.0)))])     # the twin's own mu

    def f(t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.abs(_build(e, t).solve()[3][0, 0])
    J_eager = _solve(_build(e, jnp.asarray(50e-9)))
    assert np.all(np.isfinite(J_eager))
    assert np.abs(J_eager - _solve(_build(e))).max() < 1e-10   # meas. 6.8e-13
    v = float(jax.jit(f)(jnp.asarray(50e-9)))
    g = float(jax.grad(f)(jnp.asarray(50e-9)))
    assert np.isfinite(v) and np.isfinite(g)
    h = 1e-12
    fd = (float(f(50e-9 + h)) - float(f(50e-9 - h))) / (2 * h)
    assert abs(g - fd) < 1e-6 * abs(fd)                    # measured 1.5e-10
