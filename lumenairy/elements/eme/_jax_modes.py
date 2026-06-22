"""Differentiable (JAX) twins of the EME eig-based 2-D Bloch mode solvers.

The EME *mode* solvers come in two kinds:

* the **eig-based** oracles ``ref_2d_modes`` (scalar Helmholtz) and
  ``ref_2d_modes_vector`` (full-vector Maxwell) compute the 2-D Bloch
  propagation eigenvalues ``qz^2`` as the spectrum of a single sparse generator
  ``G(eps)`` -- a genuine eigenproblem, hence **AD-friendly**;
* the **root-scan** layer finders ``layer_modes`` / ``layer_vector_modes`` locate
  ``qz^2`` where ``sigma_min(G(qz^2)) = 0`` by a real-axis scan + Brent root-find
  -- that control flow is **not** differentiable as written.

This module provides jnp twins of the two eig-based oracles only.  ``eps(x, y)``
enters each generator LINEARLY (a ``diag(eps)`` term) plus, for the vector
generator, as ``1/eps`` (the ``Ez`` elimination); the finite-difference / Yee
operators are geometry- and ``k0``-only, so they are precomputed in NumPy ONCE
and FROZEN into the trace.  The eigenproblem is solved with the gauge-fixed,
Lorentzian-broadened custom-VJP ``eig`` shared with RCWA / PMM
(:func:`lumenairy.elements.rcwa._core._jax_eig_stable`), so ``qz^2`` is
differentiable w.r.t. ``eps`` and ``k0`` (validated AD-vs-FD).

Scope under JAX (everything else routes to NumPy or raises a clear error):

* **scalar isotropic** ``eps`` (an ``(Nx, Ny)`` grid) only -- the ``(Nx,Ny,3,3)``
  tensor oracle's ``exz`` carries a documented Yee-stagger error and ``eyz`` is
  unimplemented, so a tensor ``eps`` raises on the JAX path;
* **dense** spectrum only -- the sparse shift-invert (``k`` / ``sigma``) has no
  JAX generalized-eig analog;
* ``qz^2`` only -- ``return_vecs`` (field + divergence reconstruction) uses a host
  per-mode loop and stays NumPy-only.

Cost: a DENSE ``eig`` of the ``Nx*Ny`` (scalar) / ``4 Nx Ny`` (vector) generator
plus its VJP (an ``inv(V)``) -- ``O((4 Nx Ny)^3)``.  Keep ``Nx*Ny`` small
(``~12 x 12``) for gradient-based design loops; CPU-only (the custom eig VJP uses
``jnp.linalg.eig`` / ``inv``).
"""
from __future__ import annotations

import threading

import numpy as np

from ..rcwa._core import _jax_eig_stable, _require_jax_x64
from .eme_2d_vector import _yee_ops

# Frozen geometry-only operators, cached by (Nx, Ny, Lx, Ly, kx0, ky0).  Values
# are NumPy dense arrays; converted to jnp.asarray at use (cheap, host->device).
_FROZEN_CACHE: dict = {}
_FROZEN_LOCK = threading.Lock()


def _frozen_key(Nx, Ny, Lx, Ly, kx0, ky0):
    return (int(Nx), int(Ny), float(Lx), float(Ly), float(kx0), float(ky0))


def _frozen_yee_dense(Nx, Ny, Lx, Ly, kx0, ky0):
    """The four Yee difference operators (``DxF, DxB, DyF, DyB``) as dense NumPy
    arrays -- the eps-independent, frozen part of the vector generator."""
    key = ("yee",) + _frozen_key(Nx, Ny, Lx, Ly, kx0, ky0)
    with _FROZEN_LOCK:
        hit = _FROZEN_CACHE.get(key)
    if hit is not None:
        return hit
    ops = tuple(op.toarray().astype(complex) for op in
                _yee_ops(Nx, Ny, Lx, Ly, kx0, ky0))
    with _FROZEN_LOCK:
        _FROZEN_CACHE[key] = ops
    return ops


def _frozen_helmholtz_L(Nx, Ny, Lx, Ly, kx0, ky0):
    """The eps-INDEPENDENT part of the scalar Helmholtz generator: the 5-point
    Bloch Laplacian (everything in ``eme_2d.ref_2d_modes`` except the
    ``eps k0^2`` diagonal).  ``A(eps) = L + k0^2 diag(eps_flat)``."""
    key = ("helm",) + _frozen_key(Nx, Ny, Lx, Ly, kx0, ky0)
    with _FROZEN_LOCK:
        hit = _FROZEN_CACHE.get(key)
    if hit is not None:
        return hit
    hx, hy = Lx / Nx, Ly / Ny
    N = Nx * Ny
    px, py = np.exp(1j * kx0 * Lx), np.exp(1j * ky0 * Ly)

    def ix(i, j):
        return (i % Nx) * Ny + (j % Ny)

    L = np.zeros((N, N), dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            p = ix(i, j)
            L[p, p] += -2 / hx ** 2 - 2 / hy ** 2
            L[p, ix(i + 1, j)] += (px if i == Nx - 1 else 1) / hx ** 2
            L[p, ix(i - 1, j)] += ((1 / px) if i == 0 else 1) / hx ** 2
            L[p, ix(i, j + 1)] += (py if j == Ny - 1 else 1) / hy ** 2
            L[p, ix(i, j - 1)] += ((1 / py) if j == 0 else 1) / hy ** 2
    with _FROZEN_LOCK:
        _FROZEN_CACHE[key] = L
    return L


def _reject_jax_unsupported(fn_name, eps_xy, *, k, return_vecs, mu_xy=None):
    """Raise a clear error for the JAX-path inputs that have no differentiable
    route (tensor eps, sparse shift-invert, field reconstruction, magnetic mu)."""
    import jax.numpy as jnp
    if k is not None:
        raise NotImplementedError(
            f"{fn_name}: the JAX (differentiable) path is DENSE-only; drop the "
            f"sparse shift-invert args (k / sigma) -- there is no JAX generalized "
            f"sparse eig.  Use NumPy for the sparse in-band modes.")
    if return_vecs:
        raise NotImplementedError(
            f"{fn_name}: the JAX path returns qz^2 only; field / divergence "
            f"reconstruction (return_vecs) uses a host per-mode loop -- call on "
            f"NumPy for the mode fields.")
    if mu_xy is not None:
        raise NotImplementedError(
            f"{fn_name}: magnetic mu on the JAX path is not implemented "
            f"(scalar isotropic eps only).")
    a = jnp.asarray(eps_xy)
    if a.ndim != 2:
        raise NotImplementedError(
            f"{fn_name}: the JAX path takes a scalar isotropic (Nx, Ny) eps only "
            f"(got ndim={a.ndim}).  The tensor oracle's exz has a documented "
            f"Yee-stagger error and eyz is unimplemented -- call on NumPy for "
            f"anisotropic eps.")


def _ref_2d_modes_jax(eps_xy, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.0,
                      return_vecs=False, k=None, sigma=None):
    """Differentiable scalar-Helmholtz twin of ``eme_2d.ref_2d_modes``.  Returns
    ``qz^2`` (descending real part) as a jnp array, differentiable w.r.t. ``eps``
    and ``k0``."""
    _require_jax_x64("ref_2d_modes")
    _reject_jax_unsupported("ref_2d_modes", eps_xy, k=k, return_vecs=return_vecs)
    import jax.numpy as jnp
    L = jnp.asarray(_frozen_helmholtz_L(Nx, Ny, Lx, Ly, kx0, ky0), jnp.complex128)
    eps = jnp.asarray(eps_xy, jnp.complex128).reshape(Nx * Ny)
    k0c = jnp.asarray(k0, jnp.complex128)
    A = L + (k0c ** 2) * eps[:, None] * jnp.eye(Nx * Ny, dtype=jnp.complex128)
    lam, _ = _jax_eig_stable()(A)
    return jnp.sort(lam.real)[::-1]


def _ref_2d_modes_vector_jax(eps_xy, Lx, Ly, Nx, Ny, k0, kx0=0.0, ky0=0.0,
                             return_vecs=False, k=None, sigma=None,
                             return_complex=False, mu_xy=None):
    """Differentiable full-vector twin of ``eme_2d_vector.ref_2d_modes_vector``.
    Rebuilds the ``4N x 4N`` Yee generator in jnp from the traced ``eps`` (Yee
    ops frozen) and returns ``qz^2`` (descending real part), differentiable
    w.r.t. ``eps`` and ``k0``.  ``return_complex`` keeps the complex ``qz^2``."""
    _require_jax_x64("ref_2d_modes_vector")
    _reject_jax_unsupported("ref_2d_modes_vector", eps_xy, k=k,
                            return_vecs=return_vecs, mu_xy=mu_xy)
    import jax.numpy as jnp
    DxF, DxB, DyF, DyB = (jnp.asarray(op, jnp.complex128) for op in
                          _frozen_yee_dense(Nx, Ny, Lx, Ly, kx0, ky0))
    N = Nx * Ny
    eps = jnp.asarray(eps_xy, jnp.complex128).reshape(N)
    k0c = jnp.asarray(k0, jnp.complex128)
    invk = 1.0 / (k0c * eps)                                   # diag(1/(k0 eps))
    Ieye = jnp.eye(N, dtype=jnp.complex128)
    Z = jnp.zeros((N, N), dtype=jnp.complex128)
    Eps = eps[:, None] * Ieye                                  # diag(eps)
    # Mirror _build_generator (mu = None) BLOCK-FOR-BLOCK so the dense generator
    # equals the NumPy sparse .toarray() (and hence the spectra match).
    Ez_hx = -1j * (invk[:, None] * DyF)
    Ez_hy = 1j * (invk[:, None] * DxF)
    hz_Ex = (1j / k0c) * DyB
    hz_Ey = -(1j / k0c) * DxB
    rEx = [Z, Z, DxB @ Ez_hx, 1j * k0c * Ieye + DxB @ Ez_hy]
    rEy = [Z, Z, -1j * k0c * Ieye + DyB @ Ez_hx, DyB @ Ez_hy]
    rhx = [DxF @ hz_Ex, -1j * k0c * Eps + DxF @ hz_Ey, Z, Z]
    rhy = [1j * k0c * Eps + DyF @ hz_Ex, DyF @ hz_Ey, Z, Z]
    G = jnp.block([rEx, rEy, rhx, rhy])
    gam, _ = _jax_eig_stable()(G)
    qz2 = -(gam ** 2)
    qz2 = qz2[jnp.argsort(qz2.real)[::-1]]
    return qz2 if return_complex else qz2.real
