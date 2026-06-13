"""Berreman 4x4 anisotropic planar multilayer solver (v5.14.4).

The fast, exact planar-anisotropic member of the solver family -- the
generalization of the scalar transfer-matrix coating model to fully
anisotropic layers, with full 2x2 Jones, internal fields, and per-layer
absorption.  Pinned by independent oracles:

* the ISOTROPIC limit reproduces ``coating_reflectance`` (the validated
  complex-angle scalar TMM) in R, T AND reflection phase, for lossy and
  multilayer stacks at oblique incidence -- to machine precision;
* a UNIFORM TENSOR slab reproduces the independent single-slab Berreman
  ``_berreman4x4`` oracle (which the RCWA out-of-plane path is validated
  against) to machine precision;
* energy is conserved on lossless stacks (incl. conical mounts);
* the in-structure field is tangentially continuous across interfaces and
  equals incident + reflected Jones at the top plane; the volume-integral
  absorption ``k0 Im(E^dagger eps E)`` closes against the flux-based
  ``layer_absorption`` and against ``1 - R - T``.

CONVENTION NOTE pinned here: the solver takes PUBLIC ``eps`` raw (no
conjugation) so the forward/backward mode split -- and hence the flux-based
power -- is PHYSICAL on lossy stacks.  The standalone oracle conjugates eps
in and the Jones out (an equivalent pair for the Jones only); it never
validated loss.  ``test_lossy_power_matches_scalar_tmm`` is the guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.berreman import BerremanStack, berreman_jones_1d
from lumenairy.elements.coatings import coating_reflectance

from ._berreman4x4 import berreman_jones as _oracle

WL = 0.633e-6


def _lc_tensor(tilt_deg=35.0, no=1.5, ne=1.75, loss=(0.0, 0.0)):
    th = np.deg2rad(tilt_deg)
    no2, ne2 = no ** 2, ne ** 2
    c, s = np.cos(th), np.sin(th)
    M = np.array([[ne2 * c * c + no2 * s * s, (ne2 - no2) * c * s, 0],
                  [(ne2 - no2) * c * s, ne2 * s * s + no2 * c * c, 0],
                  [0, 0, no2]], complex)
    M[0, 0] += 1j * loss[0]
    M[1, 1] += 1j * loss[1]
    return M


# =========================================================================== #
# isotropic limit == scalar TMM (R, T, phase)
# =========================================================================== #

@pytest.mark.parametrize("ang", [0.0, 0.25, 0.45])
def test_isotropic_single_film_matches_scalar_tmm(ang):
    nf, tf = 2.3, 0.18e-6
    R, T, Jr, Jt = berreman_jones_1d([(nf ** 2, tf)], 1.5, 1.0, WL, angle=ang)
    Rs, Ts, phs = coating_reflectance([(nf, tf)], WL, angle=ang,
                                      n_substrate=1.5, polarization='s')
    Rp, Tp, php = coating_reflectance([(nf, tf)], WL, angle=ang,
                                      n_substrate=1.5, polarization='p')
    # s-pol == incident Ey (col 1); p-pol == incident Ex (col 0)
    assert abs(R[1] - float(Rs)) < 1e-13 and abs(T[1] - float(Ts)) < 1e-13
    assert abs(R[0] - float(Rp)) < 1e-13 and abs(T[0] - float(Tp)) < 1e-13
    dphi = abs((np.angle(Jr[1, 1]) - float(phs) + np.pi) % (2 * np.pi) - np.pi)
    assert dphi < 1e-12


def test_isotropic_multilayer_matches_scalar_tmm():
    layers = [(2.1, 0.12e-6), (1.46, 0.09e-6), (2.1, 0.07e-6)]
    eL = [(n ** 2, t) for n, t in layers]
    for ang in (0.0, 0.3):
        R, T, Jr, Jt = berreman_jones_1d(eL, 1.52, 1.0, WL, angle=ang)
        Rs, Ts, _ = coating_reflectance(layers, WL, angle=ang,
                                        n_substrate=1.52, polarization='s')
        Rp, Tp, _ = coating_reflectance(layers, WL, angle=ang,
                                        n_substrate=1.52, polarization='p')
        assert abs(R[1] - float(Rs)) < 1e-12 and abs(R[0] - float(Rp)) < 1e-12
        # phi = 0 -> no s-p mixing
        assert abs(Jr[0, 1]) < 1e-12 and abs(Jr[1, 0]) < 1e-12


def test_lossy_power_matches_scalar_tmm():
    """REGRESSION guard: lossy stacks must give PHYSICAL R/T/absorption
    (the raw-eps convention).  A conjugated-eps split returns T > 1 and
    negative absorption here."""
    n = 2.3 + 0.15j
    for ang in (0.0, 0.3):
        R, T, Jr, Jt = berreman_jones_1d([(n ** 2, 0.2e-6)], 1.5, 1.0, WL,
                                         angle=ang)
        Rs, Ts, _ = coating_reflectance([(n, 0.2e-6)], WL, angle=ang,
                                        n_substrate=1.5, polarization='s')
        Rp, Tp, _ = coating_reflectance([(n, 0.2e-6)], WL, angle=ang,
                                        n_substrate=1.5, polarization='p')
        assert abs(R[1] - float(Rs)) < 1e-12 and abs(T[1] - float(Ts)) < 1e-12
        assert abs(R[0] - float(Rp)) < 1e-12 and abs(T[0] - float(Tp)) < 1e-12
        assert (1 - R[0] - T[0]) > 0 and T[0] < 1.0       # physical


# =========================================================================== #
# tensor == independent Berreman oracle
# =========================================================================== #

@pytest.mark.parametrize("ang", [0.0, 0.3])
@pytest.mark.parametrize("loss", [(0.0, 0.0), (0.3, 0.2)])
def test_uniform_tensor_matches_oracle(ang, loss):
    eps = _lc_tensor(loss=loss)
    R, T, Jr, Jt = berreman_jones_1d([(eps, 0.22e-6)], 1.5, 1.0, WL, angle=ang)
    Jr_o, Jt_o = _oracle(eps, 1.0, 1.5, 0.22e-6, WL, ang)
    assert np.max(np.abs(Jr - Jr_o)) < 1e-12
    assert np.max(np.abs(Jt - Jt_o)) < 1e-12


def test_scalar_tensor_equals_isotropic():
    """A scalar-on-diagonal tensor is the isotropic layer exactly."""
    R1, T1, Jr1, _ = berreman_jones_1d([(2.25, 1e-7)], 1.5, 1.0, WL, angle=0.3)
    R2, T2, Jr2, _ = berreman_jones_1d(
        [(2.25 * np.eye(3, dtype=complex), 1e-7)], 1.5, 1.0, WL, angle=0.3)
    assert np.max(np.abs(Jr1 - Jr2)) < 1e-14 and np.max(np.abs(R1 - R2)) < 1e-14


# =========================================================================== #
# energy
# =========================================================================== #

def test_energy_conserved_lossless_conical():
    eps = _lc_tensor()
    R, T, Jr, Jt = berreman_jones_1d([(eps, 0.2e-6), (2.1, 0.1e-6)],
                                     1.52, 1.0, WL, angle=0.4, phi=0.6)
    assert np.max(np.abs(R + T - 1.0)) < 1e-12


# =========================================================================== #
# class API + internal observables
# =========================================================================== #

def _device():
    st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    st.add_layer(180e-9, eps=_lc_tensor(loss=(0.08, 0.05)))
    st.add_layer(120e-9, eps=2.1)
    st.add_layer(60e-9, eps=4.0 + 0.1j)
    return st


def test_internal_field_continuity_and_boundary():
    st = _device()
    R, T, J = st.set_source(WL, angle=0.3).solve(retain_internal=True)
    # tangential continuity across both internal interfaces (probe-limited)
    for zint in (180e-9, 300e-9):
        fa = st.internal_field(np.array([zint - 1e-13]), incident=(1, 0))
        fb = st.internal_field(np.array([zint + 1e-13]), incident=(1, 0))
        sc = max(abs(fa['Ex'][0]), abs(fa['Ey'][0]), 1e-12)
        for comp in ('Ex', 'Ey', 'Hx', 'Hy'):
            assert abs(fa[comp][0] - fb[comp][0]) / sc < 1e-4
    # top plane == incident + reflected Jones
    ft = st.internal_field(np.array([1e-13]), incident=(1, 0))
    assert abs(ft['Ex'][0] - (1.0 + J[0, 0])) < 1e-4
    assert abs(ft['Ey'][0] - J[1, 0]) < 1e-4


def test_absorption_volume_identity_and_closure():
    st = _device()
    R, T, J = st.set_source(WL, angle=0.3).solve(retain_internal=True)
    A_flux = st.layer_absorption()
    k0 = 2 * np.pi / WL
    Finc = st._incident_flux(0)
    NZ = 600
    for li, (t, eps) in enumerate(st._layers):
        z0 = sum(L[0] for L in st._layers[:li])
        zq = z0 + (np.arange(NZ) + 0.5) / NZ * t
        f = st.internal_field(zq, incident=(1, 0))
        E = np.array([f['Ex'], f['Ey'], f['Ez']])
        A_vol = k0 * np.sum(
            np.imag(np.einsum('an,ab,bn->n', np.conj(E), eps, E))) \
            * (t / NZ) / Finc
        assert abs(A_flux[li, 0] - A_vol) < 5e-4
    assert abs(A_flux[:, 0].sum() - (1 - R[0] - T[0])) < 1e-9
    assert abs(A_flux[1].sum()) < 1e-12          # lossless spacer


def test_solve_guards_and_api():
    st = BerremanStack(n_substrate=1.5)
    with pytest.raises(ValueError, match="set_source"):
        st.solve()
    st.set_source(WL)
    with pytest.raises(ValueError, match="at least one layer"):
        st.solve()
    st.add_layer(1e-7, eps=2.25)
    with pytest.raises(ValueError, match="thickness must be > 0"):
        st.add_layer(-1.0, eps=2.0)
    R, T, J = st.solve()
    assert R.shape == (2,) and J.shape == (2, 2)
    with pytest.raises(ValueError, match="retain_internal"):
        st.internal_field(5e-8)


def test_theta_alias_and_phi():
    a = berreman_jones_1d([(_lc_tensor(), 1e-7)], 1.5, 1.0, WL, angle=0.3)
    b = berreman_jones_1d([(_lc_tensor(), 1e-7)], 1.5, 1.0, WL, theta=0.3)
    assert np.max(np.abs(a[2] - b[2])) < 1e-14
