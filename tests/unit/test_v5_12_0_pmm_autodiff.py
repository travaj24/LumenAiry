"""v5.12.0 PMM 1-D differentiability (Phase 0 + Phase 1 de-risking spike).

``pmm_efficiency_1d`` gains a JAX (differentiable) twin for the SIMPLEST
surface: a BINARY grating at NORMAL incidence, real lossless permittivity, TE
and TM, fixed geometry (``duty_cycle`` held fixed -- the moving-boundary
gradient is a later phase).  The path is opt-in: it is invoked ONLY when the
index / geometry inputs are JAX arrays (mirroring rcwa's backend detection), so
the NumPy path stays byte-identical.

This pins:

* the JAX path reproduces the NumPy path's per-order R, T to eig precision
  (the generalized modal pencil ``A x = q^2 B x`` is folded to the standard
  ``eig(B^-1 A)`` so the differentiable custom-VJP eig from rcwa applies);
* ``jax.grad`` of ``sum(T)`` / ``sum(R)`` matches a central finite difference
  for ``eps_ridge`` / ``eps_groove`` (real), ``depth`` and ``wavelength``,
  TE and TM.

REQUIRES JAX double precision: the modal eigenproblem is ill-conditioned in
single precision and JAX silently truncates complex128 -> complex64 unless
``jax_enable_x64`` is set (the solver warns; see ``_warn_if_jax_f32``).  The
JAX path requires ``stabilize=False`` (the degree-scan returns a discrete
degree via host control flow, which is non-differentiable).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import pmm_efficiency_1d  # noqa: E402

_CJ = jnp.complex128

# Three binary / normal-incidence cells, each clean (R+T == 1) at its degree
# with stabilize=False (no resonance at that fixed degree -- the JAX path takes
# a single fixed degree, no degree-scan).
CELLS = {
    "cell1": dict(period=0.8e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.45,
                  n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.6,
                  wavelength=0.55e-6, degree=16),
    "cell2": dict(period=1.5e-6, n_ridge=1.8, n_groove=1.3, n_substrate=1.0,
                  n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.4,
                  wavelength=0.7e-6, degree=16),
    "cell0": dict(period=1.2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
                  n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5,
                  wavelength=0.633e-6, degree=12),
}


def _solve_numpy(c, pol):
    return pmm_efficiency_1d(
        c["period"], c["n_ridge"], c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=0.0, polarization=pol, degree=c["degree"], stabilize=False)


def _solve_jax(c, pol, *, n_ridge=None, n_groove=None, depth=None,
               wavelength=None):
    nr = c["n_ridge"] if n_ridge is None else n_ridge
    ng = c["n_groove"] if n_groove is None else n_groove
    dep = c["depth"] if depth is None else depth
    wl = c["wavelength"] if wavelength is None else wavelength
    return pmm_efficiency_1d(
        c["period"], jnp.asarray(nr, _CJ), jnp.asarray(ng, _CJ),
        jnp.asarray(c["n_substrate"], _CJ), jnp.asarray(c["n_superstrate"], _CJ),
        jnp.asarray(dep), c["duty_cycle"], jnp.asarray(wl),
        angle=0.0, polarization=pol, degree=c["degree"], stabilize=False)


# ---------------------------------------------------------------------------
# GATE-FWD: the JAX twin reproduces the NumPy solve to eig precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cname", list(CELLS))
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_jax_forward_matches_numpy(cname, pol):
    c = CELLS[cname]
    o_np, R_np, T_np = _solve_numpy(c, pol)
    o_j, R_j, T_j = _solve_jax(c, pol)
    assert np.array_equal(np.asarray(o_j), o_np)
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-9
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-9
    # lossless: power is conserved on the JAX path too
    assert abs(float(np.sum(R_j) + np.sum(T_j)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# GATE-GRAD: jax.grad of sum(T) / sum(R) vs central finite difference
# ---------------------------------------------------------------------------

def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_eps_ridge_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(eps):
        _, _, T = _solve_jax(c, pol, n_ridge=jnp.sqrt(eps))
        return jnp.sum(T)

    def sumR(eps):
        _, R, _ = _solve_jax(c, pol, n_ridge=jnp.sqrt(eps))
        return jnp.sum(R)

    x0 = c["n_ridge"] ** 2
    h = 1e-5
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_grad_wrt_eps_groove_matches_fd(pol):
    c = CELLS["cell1"]

    def sumT(eps):
        _, _, T = _solve_jax(c, pol, n_groove=jnp.sqrt(eps))
        return jnp.sum(T)

    x0 = c["n_groove"] ** 2
    h = 1e-5
    gT = float(jax.grad(sumT)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    assert np.isfinite(gT)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_depth_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(d):
        _, _, T = _solve_jax(c, pol, depth=d)
        return jnp.sum(T)

    def sumR(d):
        _, R, _ = _solve_jax(c, pol, depth=d)
        return jnp.sum(R)

    x0 = c["depth"]
    # h small enough that the central-FD truncation error (the depth gradient
    # magnitude is ~1e6 per metre) is below the rtol; verified to converge
    # quadratically toward the autodiff value as h shrinks.
    h = 1e-10
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-3
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1e-3


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_wavelength_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(wl):
        _, _, T = _solve_jax(c, pol, wavelength=wl)
        return jnp.sum(T)

    def sumR(wl):
        _, R, _ = _solve_jax(c, pol, wavelength=wl)
        return jnp.sum(R)

    x0 = c["wavelength"]
    h = 1e-12
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1.0
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1.0


# ---------------------------------------------------------------------------
# Guard rails: the JAX path raises a precise error outside its validated scope
# ---------------------------------------------------------------------------

def test_jax_oblique_raises():
    c = CELLS["cell1"]
    with pytest.raises(NotImplementedError):
        pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ), c["n_groove"],
            c["n_substrate"], c["n_superstrate"], c["depth"], c["duty_cycle"],
            c["wavelength"], angle=0.1, polarization="te", degree=c["degree"],
            stabilize=False)


def test_jax_stabilize_true_raises():
    c = CELLS["cell1"]
    with pytest.raises(ValueError):
        pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ), c["n_groove"],
            c["n_substrate"], c["n_superstrate"], c["depth"], c["duty_cycle"],
            c["wavelength"], angle=0.0, polarization="te", degree=c["degree"],
            stabilize=True)


def test_jax_multi_element_raises():
    c = CELLS["cell1"]
    with pytest.raises(NotImplementedError):
        pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ), c["n_groove"],
            c["n_substrate"], c["n_superstrate"], c["depth"], c["duty_cycle"],
            c["wavelength"], angle=0.0, polarization="te", degree=c["degree"],
            elements_per_region=2, stabilize=False)


def test_jax_grad_is_jittable():
    """A real inverse-design loop wraps the solve in jax.jit; the single-level
    jax.jit(jax.grad(...)) path must compile and run."""
    c = CELLS["cell1"]

    def loss(n_ridge, depth):
        _, _, T = pmm_efficiency_1d(
            c["period"], n_ridge, jnp.asarray(c["n_groove"], _CJ),
            jnp.asarray(c["n_substrate"], _CJ),
            jnp.asarray(c["n_superstrate"], _CJ), depth, c["duty_cycle"],
            jnp.asarray(c["wavelength"]), angle=0.0, polarization="te",
            degree=c["degree"], stabilize=False)
        return jnp.sum(T)

    g = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        jnp.asarray(c["n_ridge"], _CJ), jnp.asarray(c["depth"]))
    assert np.isfinite(complex(g[0]))
    assert np.isfinite(float(jnp.real(g[1])))
