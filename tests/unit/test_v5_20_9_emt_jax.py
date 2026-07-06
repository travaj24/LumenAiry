"""Differentiable (JAX) EMT mixing rules -- rytov / maxwell_garnett / bruggeman.

The closed-form effective-medium rules in ``emt.py`` are now backend-generic, so
a traced constituent eps / index or fill flows through them (and onward through
``BerremanStack.add_effective_grating`` -> the Berreman jnp far-field twin),
enabling gradient-based homogenized-grating design loops.  These tests pin the
gradient vs central finite difference and that the concrete NumPy path is
unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.emt import bruggeman, maxwell_garnett, rytov_tensor


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _fd(f, x, h=1e-6):
    import jax.numpy as jnp
    return (float(f(jnp.asarray(x + h))) - float(f(jnp.asarray(x - h)))) / (2 * h)


def test_rytov_through_berreman_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    from lumenairy.elements.berreman import BerremanStack

    def loss(nr):
        eff = rytov_tensor(nr ** 2, 1.0, 0.4)          # traced ridge eps
        st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
        st.add_layer(0.15e-6, eps=eff)
        R, _T, _J = st.set_source(0.55e-6, theta=0.2).solve()
        return jnp.sum(jnp.asarray(R))

    g = float(jax.grad(loss)(jnp.asarray(1.8)))
    assert abs(g - _fd(loss, 1.8)) <= 1e-5 * max(abs(_fd(loss, 1.8)), 1.0) + 1e-9


def test_maxwell_garnett_and_bruggeman_gradients_match_fd():
    jax = _jax()
    import jax.numpy as jnp

    def lmg(f):
        e = maxwell_garnett(2.1, -8 + 1.2j, f, geometry="cylinder")
        return jnp.abs(e) ** 2 + jnp.real(e)

    def lbr(f):
        e = bruggeman(-8 + 1.2j, 2.1, f, geometry="cylinder")
        return jnp.abs(e) ** 2 + jnp.real(e)

    for loss in (lmg, lbr):
        g = float(jax.grad(loss)(jnp.asarray(0.35)))
        fd = _fd(loss, 0.35)
        assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-8


def test_concrete_path_unchanged():
    # a concrete call stays NumPy and returns a plain python complex
    t = rytov_tensor(2.5, 1.0, 0.4)
    assert t.shape == (3, 3) and t.dtype == np.complex128
    # harmonic (perp, x) < arithmetic (par, y) for a dielectric lamellar cell
    assert t[0, 0].real < t[1, 1].real
    mg = maxwell_garnett(2.1, 6.0, 0.3, geometry="cylinder")
    assert isinstance(mg, complex)
    br = bruggeman(-8 + 1.2j, 2.1, 0.35, geometry="cylinder")
    assert isinstance(br, complex) and br.imag >= 0
