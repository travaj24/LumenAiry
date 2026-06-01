"""v5.5.3 C10 -- thin-film coatings modernization.

A clean differentiable companion to :func:`coating_reflectance`:

- ``coating_reflectance_jax`` reproduces the NumPy real-Snell TMM to machine
  precision across polarization / angle / multilayer stacks, and is
  differentiable w.r.t. layer thickness (autodiff == finite-difference).
- the ``'coating'`` element in ``propagate_through_system`` applies the
  coating's specular scalar response (``sqrt(T)`` for the transmission port,
  ``sqrt(R) e^{i phi_r}`` for reflection), falling back to the system
  wavelength, with clear errors on a missing stack or bad port.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.coatings import coating_reflectance
from lumenairy.propagators.system import propagate_through_system

WL = 0.633e-6


def _R(val):
    """First scalar of a possibly-vectorized coating_reflectance return."""
    return float(np.atleast_1d(val)[0])


# ----------------------------------------------------------------------------
# 'coating' system element (NumPy path -- always available)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("port", ["transmission", "reflection"])
@pytest.mark.parametrize("layers", [
    [(1.38, 0.115e-6)],                                   # MgF2 QW AR
    [(2.30, 0.07e-6), (1.38, 0.11e-6), (2.30, 0.07e-6)],  # 3-layer stack
])
def test_coating_system_element_matches_tmm(port, layers):
    E0 = np.ones((16, 16), complex)
    out = propagate_through_system(
        E0, [{"type": "coating", "layers": layers, "port": port}],
        WL, dx=2e-6, verbose=False)
    E = out[0] if isinstance(out, tuple) else out
    R, T, phase_r = coating_reflectance(
        layers, WL, angle=0.0, n_substrate=1.52, polarization="avg")
    if port == "transmission":
        expect = np.sqrt(_R(T))
    else:
        expect = np.sqrt(_R(R)) * np.exp(1j * _R(phase_r))
    assert np.allclose(E, expect)


def test_coating_element_wavelength_falls_back_to_system():
    """Omitting 'wavelength' in the element uses the system wavelength."""
    E0 = np.ones((8, 8), complex)
    layers = [(1.38, 0.115e-6)]
    out = propagate_through_system(
        E0, [{"type": "coating", "layers": layers, "port": "transmission"}],
        WL, dx=2e-6, verbose=False)
    E = out[0] if isinstance(out, tuple) else out
    _, T, _ = coating_reflectance(layers, WL, angle=0.0, polarization="avg")
    assert np.allclose(E, np.sqrt(_R(T)))


def test_coating_element_composes_with_propagation():
    E0 = np.ones((32, 32), complex)
    out = propagate_through_system(
        E0, [{"type": "coating", "layers": [(1.38, 0.115e-6)]},
             {"type": "propagate", "z": 5e-3}], WL, dx=2e-6, verbose=False)
    E = out[0] if isinstance(out, tuple) else out
    assert np.all(np.isfinite(E))


def test_coating_element_errors_are_clear():
    E0 = np.ones((8, 8), complex)
    with pytest.raises(ValueError, match="needs 'layers'"):
        propagate_through_system(E0, [{"type": "coating"}], WL, dx=2e-6,
                                 verbose=False)
    with pytest.raises(ValueError, match="port must be"):
        propagate_through_system(
            E0, [{"type": "coating", "layers": [(1.38, 0.1e-6)],
                  "port": "bogus"}], WL, dx=2e-6, verbose=False)


# ----------------------------------------------------------------------------
# coating_reflectance_jax -- differentiable companion (JAX-gated)
# ----------------------------------------------------------------------------

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.coatings import coating_reflectance_jax  # noqa: E402


@pytest.mark.parametrize("pol", ["s", "p", "avg"])
@pytest.mark.parametrize("ang_deg", [0.0, 20.0, 45.0])
@pytest.mark.parametrize("layers", [
    [(1.38, 0.115e-6)],
    [(2.30, 0.07e-6), (1.38, 0.11e-6), (2.30, 0.07e-6)],
])
def test_jax_matches_numpy_tmm(pol, ang_deg, layers):
    ang = np.deg2rad(ang_deg)
    Rn = _R(coating_reflectance(
        layers, WL, angle=ang, n_substrate=1.52, polarization=pol)[0])
    Rj = float(coating_reflectance_jax(
        [(n, jnp.asarray(d)) for n, d in layers], WL, angle=ang,
        n_substrate=1.52, polarization=pol))
    assert abs(Rn - Rj) < 1e-12


def test_jax_grad_matches_finite_difference():
    """d R / d(thickness) from autodiff matches a central difference."""
    n0, d0 = 2.30, 0.07e-6
    other = (1.38, jnp.asarray(0.11e-6))

    def loss(d):
        return coating_reflectance_jax(
            [(n0, d), other], WL, angle=np.deg2rad(15), polarization="avg")

    g = float(jax.grad(loss)(jnp.asarray(d0)))
    h = 1e-12
    fd = (float(loss(jnp.asarray(d0 + h))) - float(loss(jnp.asarray(d0 - h)))) \
        / (2 * h)
    assert abs(g - fd) / max(abs(fd), 1e-30) < 1e-6


def test_jax_grad_drives_ar_thickness_toward_quarter_wave():
    """A gradient step on a single-layer AR reduces reflectance -- the
    differentiable path is usable for inverse design, not just evaluation."""
    n_film = 1.38
    d = jnp.asarray(0.08e-6)        # off the quarter-wave optimum

    def R_of(d):
        return coating_reflectance_jax([(n_film, d)], WL, angle=0.0,
                                       n_substrate=1.52, polarization="avg")

    R0 = float(R_of(d))
    for _ in range(40):
        d = d - 2e-19 * jax.grad(R_of)(d)
    R1 = float(R_of(d))
    assert R1 < R0
