"""v5.18.1: the differentiable JAX Jones twin's two ISOTROPIC half-spaces now
share ONE geometry-only eig (backlog A2 / audit P3-27 second half) instead of
two independent full 2n eigs -- mirroring the numpy ``_pmm_jones_solve_core``,
which already does this.

Because the JAX twin now uses the IDENTICAL shared-eig gauge as the numpy
oracle, forward parity is machine-precision (previously the full-eig twin
differed from the oracle by the gauge-equivalence tolerance), and the shared
geometry eig stays differentiable.

REQUIRES JAX double precision (``jax_enable_x64``); skipped without jax.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (after the x64 config, per the module convention)

from lumenairy.elements.pmm.oned import pmm_jones_1d  # noqa: E402

_P, _DEP, _WL = 1.0e-6, 0.4e-6, 1.31e-6
# in-plane anisotropic ridge (the exy off-diagonal makes the Jones matrix
# non-trivial) + an isotropic dielectric groove.
_RIDGE = np.array([[2.25, 0.20, 0.0],
                   [0.20, 2.10, 0.0],
                   [0.0, 0.0, 2.25]], dtype=complex)
_GROOVE = np.array([[1.50, 0.0, 0.0],
                    [0.0, 1.50, 0.0],
                    [0.0, 0.0, 1.50]], dtype=complex)
_KW = dict(degree=12, n_orders=5, stabilize=False)


def test_jones_jax_twin_shared_eig_matches_numpy_oracle():
    """JAX twin (shared-eig) == numpy Jones oracle (shared-eig) to machine
    precision, for R, T, and the full 2x2 Jones."""
    _on, R_n, T_n, J_n = pmm_jones_1d(
        _P, _RIDGE, _GROOVE, 1.5, 1.0, _DEP, 0.5, _WL, **_KW)
    _oj, R_j, T_j, J_j = pmm_jones_1d(
        _P, jnp.asarray(_RIDGE), jnp.asarray(_GROOVE),
        1.5, 1.0, _DEP, 0.5, _WL, **_KW)
    assert np.max(np.abs(R_n - np.asarray(R_j))) < 1e-11
    assert np.max(np.abs(T_n - np.asarray(T_j))) < 1e-11
    assert np.max(np.abs(J_n - np.asarray(J_j))) < 1e-11
    # per-pol energy conserved on both paths (lossless dielectric)
    assert abs(float(np.asarray(R_j).sum() + np.asarray(T_j).sum())
               - float(R_n.sum() + T_n.sum())) < 1e-10


def test_jones_jax_twin_shared_eig_gradient_finite():
    """d(sum|Jones|^2)/d(exy) through the shared-eig twin is finite and its
    real part matches a central finite difference."""
    def loss(exy):
        r = jnp.asarray(_RIDGE).at[0, 1].set(exy).at[1, 0].set(exy)
        _o, _R, _T, J = pmm_jones_1d(
            _P, r, jnp.asarray(_GROOVE), 1.5, 1.0, _DEP, 0.5, _WL, **_KW)
        return jnp.sum(jnp.abs(J) ** 2)

    g = complex(jax.grad(loss)(0.20 + 0.0j))
    assert np.isfinite(g)
    h = 1e-4
    fd = complex((loss(0.20 + h) - loss(0.20 - h)) / (2 * h))
    assert abs(g.real - fd.real) < 1e-4, f'AD {g!r} vs FD {fd!r}'
