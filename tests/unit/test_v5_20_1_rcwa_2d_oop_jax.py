"""Differentiable (JAX) OUT-OF-PLANE 2-D RCWA -- rcwa_jones_2d + RCWAStack.

An out-of-plane permittivity tensor (``eps_xz/eps_yz != 0``) at oblique/conical
incidence routes through the generalized (Li 2003) 6-tuple generator whose
forward/backward split was a host ``argsort`` -- non-differentiable.  v5.20.1
adds a trace-safe ``jnp.argsort`` twin (``_select_forward_flux_jax``) and, since
a TRACED tensor cannot be inspected for out-of-plane coupling, routes a traced
jax tensor to the general cascade (exact for in-plane tensors too), so the
forward and the gradient stay on the SAME branch.

The regression the fix closes: a concrete forward correctly took the OOP branch
(so it looked fine), while ``jax.grad`` silently took the IN-PLANE branch and
dropped the z-coupling -- returning a ~30%-wrong gradient with no error.  These
tests pin the gradient against central finite difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa._core import uniaxial_tensor
from lumenairy.elements.rcwa.stack import RCWAStack

_TH, _PH = np.deg2rad(20.0), np.deg2rad(25.0)
_WL, _DEP, _P = 0.55e-6, 0.4e-6, 1e-6


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _sv(J):
    return np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))


def _uni_jax(jnp, ne):
    """Differentiable tilted uniaxial (OUT-OF-PLANE: director in the x-z plane)."""
    eo, ee = 1.5 ** 2, ne ** 2
    D = jnp.diag(jnp.array([ee, eo, eo], dtype=jnp.complex128))
    ct, st = jnp.cos(jnp.deg2rad(35.0)), jnp.sin(jnp.deg2rad(35.0))
    ca, sa = jnp.cos(jnp.deg2rad(20.0)), jnp.sin(jnp.deg2rad(20.0))
    Ry = jnp.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]], dtype=jnp.complex128)
    Rz = jnp.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=jnp.complex128)
    R = Rz @ Ry
    return R @ D @ R.T


def _inpl_jax(jnp, ne):
    """Differentiable IN-PLANE anisotropic tensor (eps_xz = eps_yz = 0)."""
    eo, ee = 1.5 ** 2, ne ** 2
    D = jnp.diag(jnp.array([ee, eo, eo], dtype=jnp.complex128))
    ca, sa = jnp.cos(jnp.deg2rad(25.0)), jnp.sin(jnp.deg2rad(25.0))
    Rz = jnp.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=jnp.complex128)
    return Rz @ D @ Rz.T


def test_rcwa_jones_2d_oop_forward_matches_numpy():
    _jax()
    import jax.numpy as jnp
    OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(20.0))
    tile = np.broadcast_to(OOP, (9, 9, 3, 3)).copy()
    Jn = rcwa_jones_2d(_P, _P, tile, 1.5, 1.0, _DEP, _WL, theta=_TH, phi=_PH,
                       n_orders_x=2, n_orders_y=2)[3]
    Jj = rcwa_jones_2d(_P, _P, jnp.asarray(tile, jnp.complex128), 1.5, 1.0, _DEP,
                       _WL, theta=_TH, phi=_PH, n_orders_x=2, n_orders_y=2)[3]
    assert np.max(np.abs(_sv(np.asarray(Jn)) - _sv(np.asarray(Jj)))) < 1e-11


@pytest.mark.parametrize("kind", ["oop", "inplane"])
def test_rcwa_jones_2d_jax_gradient_matches_fd(kind):
    jax = _jax()
    import jax.numpy as jnp
    build = _uni_jax if kind == "oop" else _inpl_jax

    def loss(ne):
        e = build(jnp, ne)
        out = rcwa_jones_2d(_P, _P, jnp.broadcast_to(e, (9, 9, 3, 3)), 1.5, 1.0,
                            _DEP, _WL, theta=_TH, phi=_PH,
                            n_orders_x=2, n_orders_y=2)
        return jnp.sum(jnp.abs(jnp.asarray(out[3])) ** 2)

    ne0, h = 1.7, 1e-6
    g = float(jax.grad(loss)(jnp.asarray(ne0)))
    fd = (float(loss(jnp.asarray(ne0 + h)))
          - float(loss(jnp.asarray(ne0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9
    # jit must trace (no host argsort severing the graph)
    assert np.isfinite(float(jax.jit(jax.grad(loss))(jnp.asarray(ne0))))


def test_rcwastack_oop_jax_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    def loss(ne):
        st = RCWAStack(_P, period_y=_P, n_superstrate=1.0, n_substrate=1.5,
                       n_orders=2, n_orders_y=2)
        st.add_layer(_DEP, eps_tensor_cell=jnp.broadcast_to(
            _uni_jax(jnp, ne), (9, 9, 3, 3)))
        st.set_source(_WL, theta=_TH, phi=_PH)
        return jnp.sum(jnp.abs(jnp.asarray(st.solve().jones_reflection())) ** 2)

    ne0, h = 1.7, 1e-6
    g = float(jax.grad(loss)(jnp.asarray(ne0)))
    fd = (float(loss(jnp.asarray(ne0 + h)))
          - float(loss(jnp.asarray(ne0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9
