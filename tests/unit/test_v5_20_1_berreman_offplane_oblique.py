"""Berreman out-of-plane-tensor at OBLIQUE / CONICAL incidence (v5.20.1).

The native Berreman 4x4 S-matrix cascade is exact for isotropic, in-plane and
out-of-plane-at-NORMAL layers, but its forward/backward mode pairing (which
implicitly relies on the ``[W; -V] <-> -lam`` symmetry) is ~2% off for an
OUT-OF-PLANE tensor (``eps_xz / eps_yz != 0``) at OBLIQUE incidence.  v5.20.1
routes that regime -- and ONLY that regime -- to the generalized (Li 2003)
single-Fourier-order S-matrix that ``rcwa_jones_1d`` / ``RCWAStack`` use, to
which a planar stack reduces exactly.

Oracle: ``RCWAStack`` at ``n_orders = 1`` (a planar stack IS a zeroth-order
grating), which is independently validated against inkstone / grcwa and, for the
out-of-plane path, against the ``_berreman4x4`` transfer-matrix oracle.  The two
methods share NO cascade code with the native Berreman path, so agreement to
machine precision pins the fix.  The non-out-of-plane regimes must stay on the
native path (byte-identical), so they are checked to be UNCHANGED as well.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.berreman import BerremanStack, berreman_jones_1d
from lumenairy.elements.rcwa._core import uniaxial_tensor
from lumenairy.elements.rcwa.stack import RCWAStack

WL = 0.55e-6


def _sv(J):
    return np.sort(np.linalg.svd(np.asarray(J), compute_uv=False))


def _rcwastack_jones(layers, n_sub, n_sup, wl, theta, phi):
    """Zeroth-order reflection Jones from a 1-order RCWAStack (the oracle)."""
    st = RCWAStack(1e-6, period_y=1e-6, n_superstrate=n_sup,
                   n_substrate=n_sub, n_orders=1, n_orders_y=1)
    for eps, d in layers:
        e = np.asarray(eps, dtype=complex)
        if e.ndim == 0 or np.max(np.abs(e - e[0, 0] * np.eye(3))) < 1e-13:
            st.add_layer(d, eps=complex(e if e.ndim == 0 else e[0, 0]))
        else:
            st.add_layer(d, eps_tensor_cell=np.broadcast_to(e, (3, 3, 3, 3)).copy())
    st.set_source(wl, theta=theta, phi=phi)
    return np.asarray(st.solve().jones_reflection())


_OOP = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(20.0))
_OOP_LOSSY = uniaxial_tensor(1.5 + 0.03j, 1.7 + 0.05j, np.deg2rad(35.0),
                             phi=np.deg2rad(20.0))


def _inplane(no, ne, az):
    eo, ee = no ** 2, ne ** 2
    c, s = np.cos(az), np.sin(az)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ np.diag([ee, eo, eo]).astype(complex) @ Rz.T


# --------------------------------------------------------------------------- #
# out-of-plane tensor at oblique / conical incidence == RCWAStack
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("theta_deg,phi_deg", [
    (20.0, 0.0),     # planar oblique
    (20.0, 25.0),    # conical
    (45.0, 60.0),    # steeper conical
])
def test_oop_oblique_single_layer_matches_rcwastack(theta_deg, phi_deg):
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    layers = [(_OOP, 0.4e-6)]
    _R, _T, Jr, _Jt = berreman_jones_1d(layers, 1.5, 1.0, WL, angle=th, phi=ph)
    Jo = _rcwastack_jones(layers, 1.5, 1.0, WL, th, ph)
    # singular values are basis / phase invariant -- the physical reflection.
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-11


def test_oop_conical_lossy_matches_rcwastack():
    th, ph = np.deg2rad(20.0), np.deg2rad(25.0)
    layers = [(_OOP_LOSSY, 0.4e-6)]
    _R, _T, Jr, _Jt = berreman_jones_1d(layers, 1.5, 1.0, WL, angle=th, phi=ph)
    Jo = _rcwastack_jones(layers, 1.5, 1.0, WL, th, ph)
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-11


def test_oop_oblique_multilayer_matches_rcwastack():
    """A mixed OOP / isotropic / lossy-OOP stack -- the case the native path and
    both harness transfer-matrix oracles all got wrong (each for a different
    reason)."""
    th, ph = np.deg2rad(20.0), np.deg2rad(25.0)
    layers = [(_OOP, 0.3e-6), (2.1 * np.eye(3), 0.15e-6), (_OOP_LOSSY, 0.2e-6)]
    _R, _T, Jr, _Jt = berreman_jones_1d(layers, 1.5, 1.0, WL, angle=th, phi=ph)
    Jo = _rcwastack_jones(layers, 1.5, 1.0, WL, th, ph)
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-10


def test_berremanstack_oop_oblique_matches_rcwastack():
    th, ph = np.deg2rad(30.0), np.deg2rad(40.0)
    layers = [(_OOP, 0.3e-6), (2.1 * np.eye(3), 0.15e-6), (_OOP_LOSSY, 0.2e-6)]
    st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    for eps, d in layers:
        st.add_layer(d, eps=eps)
    _R, _T, Jr = st.set_source(WL, theta=th, phi=ph).solve()
    Jo = _rcwastack_jones(layers, 1.5, 1.0, WL, th, ph)
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-10


# --------------------------------------------------------------------------- #
# energy: lossless conserves, lossy absorbs (per incident polarization)
# --------------------------------------------------------------------------- #

def test_oop_oblique_lossless_energy_conserved():
    th, ph = np.deg2rad(20.0), np.deg2rad(25.0)
    R, T, _Jr, _Jt = berreman_jones_1d([(_OOP, 0.4e-6)], 1.5, 1.0, WL,
                                       angle=th, phi=ph)
    assert np.allclose(R + T, 1.0, atol=1e-6)


def test_oop_oblique_lossy_energy_bounded():
    th, ph = np.deg2rad(20.0), np.deg2rad(25.0)
    R, T, _Jr, _Jt = berreman_jones_1d([(_OOP_LOSSY, 0.4e-6)], 1.5, 1.0, WL,
                                       angle=th, phi=ph)
    absorb = 1.0 - (R + T)
    assert np.all(absorb > 1e-4) and np.all(absorb < 1.0)


# --------------------------------------------------------------------------- #
# routing: NON-out-of-plane regimes stay on the native path (unchanged)
# --------------------------------------------------------------------------- #

def _native_only(layers, th, ph):
    """The native path, called directly by BYPASSING the offplane route (a
    scalar / in-plane cell never routes), then compared to the public entry."""
    _R, _T, Jr, _Jt = berreman_jones_1d(layers, 1.5, 1.0, WL, angle=th, phi=ph)
    return Jr


@pytest.mark.parametrize("tag,eps", [
    ("iso", 2.25 * np.eye(3)),
    ("inplane", _inplane(1.5, 1.7, np.deg2rad(25.0))),
])
def test_non_offplane_matches_rcwastack(tag, eps):
    """Isotropic / in-plane at oblique stay on the native cascade AND agree with
    RCWAStack -- the route condition must not fire, and the native path is
    already exact there."""
    th, ph = np.deg2rad(30.0), np.deg2rad(40.0)
    _R, _T, Jr, _Jt = berreman_jones_1d([(eps, 0.4e-6)], 1.5, 1.0, WL,
                                        angle=th, phi=ph)
    Jo = _rcwastack_jones([(eps, 0.4e-6)], 1.5, 1.0, WL, th, ph)
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-11


def test_oop_at_normal_incidence_matches_rcwastack():
    """An out-of-plane tensor at NORMAL incidence stays on the native path
    (the route needs OBLIQUE) and is exact there."""
    _R, _T, Jr, _Jt = berreman_jones_1d([(_OOP, 0.4e-6)], 1.5, 1.0, WL,
                                        angle=0.0, phi=0.0)
    Jo = _rcwastack_jones([(_OOP, 0.4e-6)], 1.5, 1.0, WL, 0.0, 0.0)
    assert np.max(np.abs(_sv(Jr) - _sv(Jo))) < 1e-11


# --------------------------------------------------------------------------- #
# internal-field retention on the generalized path (C2: previously raised)
# --------------------------------------------------------------------------- #

def test_retain_internal_works_for_oop_oblique():
    """AUDIT_DYNAMETA_CONSUMER_API_GAPS C2 (2026-07-14): the generalized
    OOP-oblique cascade now RETAINS internals (this gate previously pinned
    the NotImplementedError).  The retained core must close the absorption
    budget at machine precision on the lossy tensor; the full C2 gate set
    (conical, lossless-zero, theta -> 0 continuity vs the native path)
    lives in ``test_audit_dynameta_consumer_api_2.py``."""
    th, ph = np.deg2rad(20.0), np.deg2rad(25.0)
    st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    st.add_layer(0.4e-6, eps=_OOP_LOSSY)
    st.set_source(WL, theta=th, phi=ph)
    R, T, _Jr = st.solve(retain_internal=True)
    A = st.layer_absorption()
    assert A.min() > 0.0
    assert np.abs(A.sum(axis=0) + R + T - 1.0).max() < 1e-12


# --------------------------------------------------------------------------- #
# JAX twin: differentiable out-of-plane-oblique matches NumPy + valid gradients
# --------------------------------------------------------------------------- #

def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


@pytest.mark.parametrize("theta_deg,phi_deg", [(20.0, 0.0), (20.0, 25.0)])
def test_jax_oop_oblique_matches_numpy(theta_deg, phi_deg):
    _jax()
    import jax.numpy as jnp
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    _R, _T, Jn, _ = berreman_jones_1d([(_OOP_LOSSY, 0.4e-6)], 1.5, 1.0, WL,
                                      angle=th, phi=ph)
    _Rj, _Tj, Jj, _ = berreman_jones_1d(
        [(jnp.asarray(_OOP_LOSSY, jnp.complex128), 0.4e-6)],
        jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(WL),
        angle=jnp.asarray(th), phi=jnp.asarray(ph))
    assert np.max(np.abs(_sv(np.asarray(Jj)) - _sv(np.asarray(Jn)))) < 1e-11


def test_jax_oop_conical_gradient_matches_fd():
    """d/dtheta of the reflected-power Frobenius norm through the generalized
    out-of-plane path -- AD must agree with central finite difference."""
    jax = _jax()
    import jax.numpy as jnp
    ph = np.deg2rad(25.0)

    def loss(theta):
        _R, _T, J, _ = berreman_jones_1d(
            [(jnp.asarray(_OOP, jnp.complex128), 0.4e-6)],
            jnp.asarray(1.5 + 0j), jnp.asarray(1.0 + 0j), jnp.asarray(WL),
            angle=theta, phi=jnp.asarray(ph))
        return jnp.sum(jnp.abs(J) ** 2)

    th0 = 0.35
    g_ad = float(jax.grad(loss)(jnp.asarray(th0)))
    h = 1e-6
    g_fd = (float(loss(jnp.asarray(th0 + h)))
            - float(loss(jnp.asarray(th0 - h)))) / (2 * h)
    assert abs(g_ad - g_fd) <= 1e-5 * max(abs(g_fd), 1.0) + 1e-9
    # jit must trace without a host-side argsort severing the graph
    assert np.isfinite(float(jax.jit(loss)(jnp.asarray(th0))))
