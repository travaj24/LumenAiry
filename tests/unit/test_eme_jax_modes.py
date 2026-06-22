"""Differentiable (JAX) twins of the EME eig-based 2-D Bloch mode solvers
(``ref_2d_modes`` scalar Helmholtz, ``ref_2d_modes_vector`` full-vector).

Gates: FWD (twin == NumPy spectrum), GRAD (AD vs central finite difference, on a
loss that genuinely exercises the eig backward rule -- a SPECIFIC sorted
eigenvalue / a sorted spectral-window sum, NOT a trace, which would bypass the
eig VJP), JIT (jit + grad-of-jit trace once), GUARD (the non-differentiable
inputs -- tensor eps, sparse k/sigma, return_vecs, and the root-scan layer
finders -- raise a clear error rather than silently mislead).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)      # eig is ill-conditioned in f32
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.eme import (  # noqa: E402
    layer_vector_modes,
    ref_2d_modes,
    ref_2d_modes_vector,
)

Lx = Ly = 1.0
k0 = 8.0


def _asym_cell(Nx, Ny, lo=2.0, hi=6.0):
    """Asymmetric scalar eps cell (top Helmholtz mode is non-degenerate)."""
    e = np.full((Nx, Ny), lo)
    e[: Nx // 2, : Ny // 3] = hi            # corner block -> no x/y symmetry
    e[Nx - 1, Ny - 1] = 0.5 * (lo + hi)
    return e


def _pillar_cell(Nx, Ny, host=1.0, pillar=12.0):
    """High-contrast pillar (well-separated top band for the window-sum gate)."""
    e = np.full((Nx, Ny), host)
    e[1:4, 1:4] = pillar
    return e


# --------------------------------------------------------------------------- #
#  GATE-FWD : twin spectrum == NumPy oracle spectrum                           #
# --------------------------------------------------------------------------- #
def test_jax_fwd_scalar_matches_numpy():
    Nx = Ny = 6
    e = _asym_cell(Nx, Ny)
    q_np = ref_2d_modes(e, Lx, Ly, Nx, Ny, k0, kx0=0.2, ky0=np.pi / 3)
    q_jx = ref_2d_modes(jnp.asarray(e), Lx, Ly, Nx, Ny, k0, kx0=0.2, ky0=np.pi / 3)
    assert np.max(np.abs(q_np - np.asarray(q_jx))) < 1e-8


def test_jax_fwd_vector_matches_numpy():
    Nx = Ny = 6
    e = _pillar_cell(Nx, Ny)
    q_np = ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.3)
    q_jx = ref_2d_modes_vector(jnp.asarray(e), Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.3)
    assert np.max(np.abs(q_np - np.asarray(q_jx))) < 1e-8


# --------------------------------------------------------------------------- #
#  GATE-GRAD : AD vs central FD (loss exercises the eig backward rule)         #
# --------------------------------------------------------------------------- #
def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_jax_grad_scalar_top_mode_vs_fd():
    """d(top Helmholtz qz^2)/d(eps_pixel) and /d(k0): a SPECIFIC non-degenerate
    eigenvalue -> its gradient is the eigenvector sensitivity (tests the eig VJP,
    not a trace)."""
    Nx = Ny = 6
    e0 = _asym_cell(Nx, Ny)
    kx0, ky0 = 0.2, np.pi / 3

    def loss_eps(epix):
        e = jnp.asarray(e0).at[1, 1].set(epix)
        return ref_2d_modes(e, Lx, Ly, Nx, Ny, k0, kx0=kx0, ky0=ky0)[0]

    ad = float(jax.grad(loss_eps)(jnp.asarray(6.0)))
    fd = float(_central_fd(lambda v: float(loss_eps(jnp.asarray(v))), 6.0, 1e-6))
    assert abs(ad - fd) / max(abs(fd), 1e-9) < 1e-4

    def loss_k0(kk):
        return ref_2d_modes(jnp.asarray(e0), Lx, Ly, Nx, Ny, kk, kx0=kx0, ky0=ky0)[0]

    adk = float(jax.grad(loss_k0)(jnp.asarray(k0)))
    fdk = float(_central_fd(lambda v: float(loss_k0(jnp.asarray(v))), k0, 1e-6))
    assert abs(adk - fdk) / max(abs(fdk), 1e-9) < 1e-4


def test_jax_grad_vector_window_sum_vs_fd():
    """d(sum of the top-4 vector qz^2)/d(eps_pixel) and /d(k0).  The top-4 are the
    two top physical modes (whole +-qz pairs) -- a SORTED spectral-window sum
    (non-analytic in G -> tests the eig VJP) that does not split a degenerate
    pair."""
    Nx = Ny = 6
    e0 = _pillar_cell(Nx, Ny)
    kx0, ky0 = 0.0, 0.3

    def loss_eps(epix):
        e = jnp.asarray(e0).at[2, 2].set(epix)
        return jnp.sum(ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, kx0=kx0, ky0=ky0)[:4])

    ad = float(jax.grad(loss_eps)(jnp.asarray(12.0)))
    fd = float(_central_fd(lambda v: float(loss_eps(jnp.asarray(v))), 12.0, 1e-6))
    assert abs(ad - fd) / max(abs(fd), 1e-9) < 1e-4

    def loss_k0(kk):
        return jnp.sum(
            ref_2d_modes_vector(jnp.asarray(e0), Lx, Ly, Nx, Ny, kk, kx0=kx0, ky0=ky0)[:4])

    adk = float(jax.grad(loss_k0)(jnp.asarray(k0)))
    fdk = float(_central_fd(lambda v: float(loss_k0(jnp.asarray(v))), k0, 1e-6))
    assert abs(adk - fdk) / max(abs(fdk), 1e-9) < 1e-4


# --------------------------------------------------------------------------- #
#  GATE-JIT : jit + grad-of-jit trace and run                                  #
# --------------------------------------------------------------------------- #
def test_jax_jit_and_grad_of_jit():
    Nx = Ny = 6
    e0 = _pillar_cell(Nx, Ny)

    @jax.jit
    def loss(epix):
        e = jnp.asarray(e0).at[2, 2].set(epix)
        return jnp.sum(ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.3)[:4])

    v0 = loss(jnp.asarray(12.0))
    g = jax.grad(loss)(jnp.asarray(12.0))               # grad of jit, traces once
    assert np.isfinite(float(v0)) and np.isfinite(float(g))


# --------------------------------------------------------------------------- #
#  GATE-GUARD : non-differentiable inputs raise (don't silently mislead)       #
# --------------------------------------------------------------------------- #
def test_jax_guard_tensor_eps_raises():
    Nx = Ny = 6
    et = np.zeros((Nx, Ny, 3, 3), dtype=complex)
    et[:, :, 0, 0] = et[:, :, 1, 1] = et[:, :, 2, 2] = 4.0
    with pytest.raises(NotImplementedError, match="scalar isotropic"):
        ref_2d_modes_vector(jnp.asarray(et), Lx, Ly, Nx, Ny, k0)


def test_jax_guard_sparse_and_vecs_raise():
    Nx = Ny = 6
    e = jnp.asarray(_pillar_cell(Nx, Ny))
    with pytest.raises(NotImplementedError, match="DENSE-only|sparse"):
        ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, k=4)
    with pytest.raises(NotImplementedError, match="return_vecs|qz"):
        ref_2d_modes_vector(e, Lx, Ly, Nx, Ny, k0, return_vecs=True)


def test_jax_guard_layer_rootscan_raises():
    """The sigma_min root-scan layer finder is not differentiable -> a JAX strip
    eps must raise (point to the FD-oracle twin), not silently run the NumPy scan."""
    Nx = 8
    strip = (jnp.asarray(_pillar_cell(Nx, 4)[:, 0]), 0.5 * Ly)
    with pytest.raises(NotImplementedError, match="differentiab|root-scan|oracle"):
        layer_vector_modes([strip, strip], Lx, Nx, Ly, k0, (150.0, 252.0))
