"""Differentiable (JAX) twin of the axisymmetric BOR-PMM stack solve.

``BORStack.solve()`` now runs under a trace when any layer permittivity, ring
index or thickness is a JAX array (the half-spaces stay concrete): the staggered
``K x = q^2 B x`` eigensolve (gauge-stable custom-VJP eig on the equilibrated
fold), the modal field/flux reconstruction, and the Redheffer cascade are all
differentiable.  Gradients flow w.r.t. a concentric-ring ridge index, a uniform
layer eps (incl. Im for a lossy layer) and a layer thickness -- the natural
knobs for an axisymmetric grating / VCSEL-aperture design loop.

Like the RCWA/PMM twins the propagating ORDER SET cannot be materialized under a
trace, so ``R``/``T`` come back as full-``2N`` per-mode arrays masked to 0 off
the propagating set; the TOTAL ``sum(R)``/``sum(T)`` match the NumPy solve
(order-/gauge-invariant) and a scalar design loss differentiates.  These tests
pin the forward total-parity vs NumPy, the gradient vs central FD for each
traced input form (ring index / thickness / lossy eps), the lossless per-order
energy closure of the masked arrays, and that the concrete NumPy path is
unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.bor.bor_stack import BORStack

_WL = 2 * np.pi / 2.0                       # k0 = 2.0 (validated BOR scale)


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _solve_ring(nr, thk=0.5, ng=1.414, N=56, Rbig=3.0):
    """One concentric-ring grating layer between matched n=1.4142 half-spaces."""
    s = BORStack(Rbig=Rbig, m=1, N=N, n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(thk, rings=(0.8, 0.5, nr, ng))
    return s.set_source(wavelength=_WL).solve()


def test_bor_jax_forward_parity_total():
    """Traced total sum(R)/sum(T) reproduce the NumPy solve (order-invariant)."""
    _jax()
    import jax.numpy as jnp
    rc, rj = _solve_ring(2.449), _solve_ring(jnp.asarray(2.449))
    Rc, Tc = float(np.sum(rc["R"])), float(np.sum(rc["T"]))
    Rj = float(np.sum(np.asarray(rj["R"])))
    Tj = float(np.sum(np.asarray(rj["T"])))
    assert abs(Rj - Rc) < 1e-9, f"|dR|={abs(Rj - Rc):.2e}"
    assert abs(Tj - Tc) < 1e-9, f"|dT|={abs(Tj - Tc):.2e}"


def test_bor_jax_ring_index_gradient_matches_fd():
    """d sum(R) / d(ring ridge index) vs central FD."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(nr):
        return jnp.sum(jnp.asarray(_solve_ring(nr)["R"]))

    g = float(jax.grad(loss)(jnp.asarray(2.449)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(2.449 + h)))
          - float(loss(jnp.asarray(2.449 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_bor_jax_thickness_gradient_matches_fd():
    """d sum(T) / d(layer thickness) vs central FD (traced propagation phase)."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(thk):
        return jnp.sum(jnp.asarray(_solve_ring(2.449, thk)["T"]))

    g = float(jax.grad(loss)(jnp.asarray(0.5)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.5 + h)))
          - float(loss(jnp.asarray(0.5 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_bor_jax_lossy_eps_gradient_matches_fd():
    """d sum(T) / d(Im eps) of a uniform lossy layer vs central FD (eps= path)."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(ei):
        s = BORStack(Rbig=3.0, m=1, N=56,
                     n_superstrate=1.4142, n_substrate=1.4142)
        s.add_layer(0.5, eps=2.0 + 1j * ei)            # lossy uniform layer
        return jnp.sum(jnp.asarray(s.set_source(wavelength=_WL).solve()["T"]))

    g = float(jax.grad(loss)(jnp.asarray(0.1)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.1 + h)))
          - float(loss(jnp.asarray(0.1 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_bor_jax_energy_closure_lossless():
    """The masked full-2N arrays close energy per incident order (lossless)."""
    _jax()
    import jax.numpy as jnp
    rj = _solve_ring(jnp.asarray(2.449))               # all-real -> lossless
    energy = np.asarray(rj["energy"])
    inc = np.asarray(rj["inc_mask"])
    per_order = energy[inc > 0.5]
    assert per_order.size >= 3                          # a real multi-order set
    assert np.max(np.abs(per_order - 1.0)) < 1e-6, \
        f"max|R+T-1|={np.max(np.abs(per_order - 1.0)):.2e}"


def test_concrete_bor_unchanged():
    """The plain-NumPy path returns the documented dict and closes energy."""
    r = _solve_ring(2.449)                              # plain python floats
    assert set(r) >= {"q", "gamma", "angles", "R", "T", "energy",
                      "inc", "out", "S"}
    assert r["R"].shape == r["T"].shape == r["inc"].shape
    assert r["inc"].size >= 3
    assert np.max(np.abs(r["energy"] - 1.0)) < 1e-6
