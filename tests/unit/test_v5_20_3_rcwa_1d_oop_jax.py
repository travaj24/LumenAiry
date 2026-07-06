"""Differentiable (JAX) 1-D OUT-OF-PLANE RCWA -- rcwa_jones_1d(_segments).

The 1-D full-3x3 (out-of-plane) tensor path used to reject the JAX backend
(``_reject_jax_offplane``), on the now-stale grounds that its forward/backward
flux split was a host ``argsort``.  That split has a trace-safe twin
(``_select_forward_flux_jax``, wired into the shared ``_layer_eigenmodes_tensor``
for the 2-D OOP twin), so the 1-D path is unblocked: a concrete off-plane jax
tensor routes to the general full-3x3 solver, and a TRACED tensor routes there
too (exact for in-plane -- the off-plane blocks vanish) so the forward and the
gradient stay on ONE branch.  A concrete in-plane tensor keeps the fast 2N path.

Pins: OOP forward parity vs NumPy, OOP gradient vs central finite difference
(binary + multi-segment), a finite gradient under ``jit``, and that a concrete
in-plane jax tensor still matches NumPy on the fast path (no regression).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import rcwa_jones_1d, rcwa_jones_1d_segments

_P = 0.5e-6
_WL = 0.55e-6
_DEP = 0.40e-6
_ANG = np.deg2rad(15.0)
_DUTY = 0.5
_NO = 8


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _oop(xp, ne):
    """Tilted uniaxial with OUT-OF-PLANE coupling (director in the x-z plane)."""
    D = xp.diag(xp.asarray([ne ** 2, 1.5 ** 2, 1.5 ** 2], dtype=xp.complex128))
    ct, st = xp.cos(xp.deg2rad(35.0)), xp.sin(xp.deg2rad(35.0))
    Ry = xp.asarray([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]],
                    dtype=xp.complex128)
    return Ry @ D @ Ry.T


def _inpl(xp, ne):
    D = xp.diag(xp.asarray([ne ** 2, 1.5 ** 2, 1.5 ** 2], dtype=xp.complex128))
    ca, sa = xp.cos(xp.deg2rad(25.0)), xp.sin(xp.deg2rad(25.0))
    Rz = xp.asarray([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
                    dtype=xp.complex128)
    return Rz @ D @ Rz.T


_GROOVE = np.eye(3) * 1.5 ** 2


def test_rcwa_jones_1d_oop_forward_matches_numpy():
    _jax()
    import jax.numpy as jnp
    Jn = rcwa_jones_1d(_P, _oop(np, 1.8), _GROOVE, 1.5, 1.0, _DEP, _DUTY, _WL,
                       angle=_ANG, n_orders=_NO)
    Jj = rcwa_jones_1d(_P, jnp.asarray(_oop(np, 1.8)), jnp.asarray(_GROOVE),
                       1.5, 1.0, _DEP, _DUTY, _WL, angle=_ANG, n_orders=_NO)
    assert np.max(np.abs(np.asarray(Jj[1]) - Jn[1])) < 1e-11    # R
    assert np.max(np.abs(np.asarray(Jj[2]) - Jn[2])) < 1e-11    # T
    assert np.max(np.abs(np.asarray(Jj[3]) - Jn[3])) < 1e-11    # Jones


def test_rcwa_jones_1d_oop_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    def loss(ne):
        o = rcwa_jones_1d(_P, _oop(jnp, ne), jnp.asarray(_GROOVE), 1.5, 1.0,
                          _DEP, _DUTY, _WL, angle=_ANG, n_orders=_NO)
        return (jnp.sum(jnp.abs(jnp.asarray(o[3])) ** 2)
                + jnp.sum(jnp.asarray(o[1])))

    ne0, h = 1.8, 1e-6
    g = float(jax.grad(loss)(jnp.asarray(ne0)))
    fd = (float(loss(jnp.asarray(ne0 + h)))
          - float(loss(jnp.asarray(ne0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9
    assert np.isfinite(float(jax.jit(jax.grad(loss))(jnp.asarray(ne0))))


def test_rcwa_jones_1d_segments_oop_gradient_matches_fd():
    jax = _jax()
    import jax.numpy as jnp

    def loss(ne):
        segs = [(0.3, jnp.asarray(_GROOVE)),
                (0.4, _oop(jnp, ne)),
                (0.3, jnp.asarray(np.eye(3) * 2.1 ** 2))]
        o = rcwa_jones_1d_segments(_P, segs, 1.5, 1.0, _DEP, _WL,
                                   angle=_ANG, n_orders=_NO)
        return jnp.sum(jnp.abs(jnp.asarray(o[3])) ** 2)

    ne0, h = 1.8, 1e-6
    g = float(jax.grad(loss)(jnp.asarray(ne0)))
    fd = (float(loss(jnp.asarray(ne0 + h)))
          - float(loss(jnp.asarray(ne0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9


def test_rcwa_jones_1d_inplane_concrete_jax_keeps_fast_path():
    """A concrete IN-PLANE jax tensor must still match NumPy on the fast 2N
    path (the unblock only reroutes off-plane / traced tensors)."""
    _jax()
    import jax.numpy as jnp
    Jn = rcwa_jones_1d(_P, _inpl(np, 1.8), _GROOVE, 1.5, 1.0, _DEP, _DUTY, _WL,
                       angle=_ANG, n_orders=_NO)
    Jj = rcwa_jones_1d(_P, jnp.asarray(_inpl(np, 1.8)), jnp.asarray(_GROOVE),
                       1.5, 1.0, _DEP, _DUTY, _WL, angle=_ANG, n_orders=_NO)
    assert np.max(np.abs(np.asarray(Jj[3]) - Jn[3])) < 1e-11
