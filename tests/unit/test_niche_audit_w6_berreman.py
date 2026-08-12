"""W6 audit (2026-07-26): numerical validation of the Berreman 4x4 interiors.

The 2026-07-25 adversarial audit never numerically exercised
``lumenairy/elements/berreman.py`` or its JAX twin ``_berreman_jax.py`` -- only
one cross-family retardance check existed.  This module is the oracle suite that
territory was missing, plus the pins for the four defects it found.

ORACLES (each independent of the solver under test)
  1. ISOTROPIC LIMIT -- an isotropic stack through Berreman equals the
     ``coatings.coating_reflectance`` complex-angle TMM (itself verified against
     two from-scratch oracles at 1e-15 in the audit): s and p, R and T, normal
     through 1.4 rad, lossless / lossy film / lossy substrate / metal film /
     7-layer DBR / immersed superstrate past TIR.  Measured worst 1.9e-15.
  1b. BARE INTERFACE vs the closed-form Fresnel amplitudes ACROSS the critical
     angle -- 3.9e-16, and Berreman is the CORRECT side where the coatings TMM's
     legacy ``min(sin_t, 0.9999)`` cap makes it 2.2e-2 wrong at ``theta_c - 1e-5``.
  1c. CONICAL isotropic -- the per-lab-polarization R/T are the FLUX-weighted
     (``eta_p = n/cos``, ``eta_s = n cos``) s/p mixture, NOT the naive amplitude
     average (which is off by 3.5e-4).  A contract fact, pinned.
  2. UNIAXIAL closed form for all three c-axis orientations (x / y / z) vs the
     exact single-film Airy amplitude with the analytic extraordinary
     ``(n_eq, cos_eq)``; plus the transmitted-Jones retardance -> 2 pi d dn / wl.
  3. ENERGY R+T=1 for lossless anisotropic stacks at oblique AND conical
     incidence, and -- the flux-projection stress case -- R+T=1 for lossless
     layers on an ABSORBING substrate (this is where W6 F-1 lived).
  4. RECIPROCITY: stack reversal + half-space swap at the Snell-conjugate angle.
  5. TWIST -> CONTINUUM: N thin rotated slabs converge 2nd order.
  5b. ROTATION COVARIANCE: conical(phi) == planar on the ``R_z(-phi)`` stack --
     the strongest available gate on the conical ``Kx*Ky`` Delta entries.
  6. NumPy vs ``_berreman_jax`` twin parity over the whole case matrix.
  7. Cache interiors: cold == warm bit-identically, LRU eviction is
     result-preserving, the mode-sorting criterion under evanescent / gain /
     degenerate-partition edges, and NumPy-vs-JAX partition agreement.

DEFECTS FIXED AND PINNED HERE
  F-1  CRITICAL-physics.  ``_offplane_oblique_solve`` (NumPy) and
       ``_offplane_solve_jax`` passed the PUBLIC eps to
       ``rcwa._core._forward_flux_kz``, which UN-conjugates its argument (every
       rcwa call site feeds it the INTERNAL gauge).  Double-conjugated, an
       absorbing half-space came back with ``Re(kz) < 0`` and the propagating
       mask SILENTLY ZEROED T.  Measured: tilted-director slab on
       ``n_sub = 1.5+0.3j`` at theta = 0.3 gave T = 0.000000 vs
       ``rcwa_jones_1d`` T = 0.930316.  The JAX twin reached it WIDER (it routes
       out-of-plane tensors at EVERY incidence, so T = 0 even at normal, a 0.988
       twin divergence).
  F-2  HIGH-bug.  No incidence / geometry / energy guards at all, where every
       sibling engine has them: a metallic superstrate returned T = 30.8 (3000%
       energy violation) silently; a back-side ``angle = 2.0`` rad aliased onto
       the supplementary front-side angle; ``angle = pi/2`` and ``angle = nan``
       raised bare ``LinAlgError``; a gain layer returned R = 16.62 with no
       tripwire; the FUNCTIONAL entry accepted ``thickness <= 0`` that
       ``BerremanStack.add_layer`` already rejected.
  F-3  MEDIUM-doc, REFUTATION.  The documented "the native cascade is ~2% off
       for an out-of-plane tensor at oblique incidence" does NOT reproduce: the
       two paths agree to ~1e-15 on R, T and BOTH Jones.  Docs corrected; the
       two-path agreement is pinned as a standing gate.
  F-4  LOW-hygiene.  ``_farfield(core, eps_sup, eps_sub, Kx, Ky)`` referenced
       none of the last four (AST-verified inert params).  Signature trimmed.
  F-5  LOW-hygiene.  Both LRUs handed out WRITEABLE views of their stored
       arrays despite docstrings asserting read-only, so one stray write
       poisoned every later solve.  Frozen (the ``pmm/_core`` precedent).

Tolerances are the MEASURED worst on this box with cross-platform headroom; the
measured figure is stated at each site.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from lumenairy.elements import berreman as B
from lumenairy.elements.berreman import BerremanStack, berreman_jones_1d
from lumenairy.elements.coatings import coating_reflectance
from lumenairy.elements.rcwa import rcwa_jones_1d

WL = 0.633e-6
_I3 = np.eye(3, dtype=complex)
BIAX = np.diag([2.1, 2.45, 2.9]).astype(complex)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _rot3(axis, t):
    c, s = np.cos(t), np.sin(t)
    if axis == 'x':
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rot(eps, axis, t):
    R = _rot3(axis, t)
    return R @ np.asarray(eps, dtype=complex) @ R.T


def _t3(e):
    e = np.asarray(e, dtype=complex)
    return e * _I3 if e.ndim == 0 else e


def _tmm(layers_n, n_sub, n_amb, angle, pol):
    R, T, _ph = coating_reflectance(layers_n, WL, angle=angle,
                                    n_substrate=n_sub, n_ambient=n_amb,
                                    polarization=pol)
    return float(np.atleast_1d(R)[0]), float(np.atleast_1d(T)[0])


def _film_rt(n0, n1, n2, cos0, cos1, cos2, d, pol):
    """Exact single-film Airy R/T with EXPLICIT (possibly non-Snell) cos_t --
    the closed form the uniaxial extraordinary channel maps onto."""
    if pol == 's':
        e0, e1, e2 = n0 * cos0, n1 * cos1, n2 * cos2
    else:
        e0, e1, e2 = n0 / cos0, n1 / cos1, n2 / cos2
    r01, r12 = (e0 - e1) / (e0 + e1), (e1 - e2) / (e1 + e2)
    t01, t12 = 2 * e0 / (e0 + e1), 2 * e1 / (e1 + e2)
    dl = 2 * np.pi * n1 * d * cos1 / WL
    ph = np.exp(2j * dl)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    t = t01 * t12 * np.exp(1j * dl) / (1 + r01 * r12 * ph)
    return abs(r) ** 2, float(np.real(e2 / e0)) * abs(t) ** 2


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


# =========================================================================== #
# ORACLE 1 -- isotropic limit == the verified coatings TMM
# =========================================================================== #

_ISO_CASES = [
    ("bare interface", [], 1.52, 1.0),
    ("single lossless film", [(1.38, 120e-9)], 1.52, 1.0),
    ("3-layer lossless", [(1.38, 120e-9), (2.30, 80e-9), (1.46, 200e-9)],
     1.52, 1.0),
    ("7-layer DBR", [(2.3, 68.8e-9), (1.46, 108.4e-9)] * 3 + [(2.3, 68.8e-9)],
     1.52, 1.0),
    ("metal film", [(0.15 + 3.5j, 40e-9)], 1.52, 1.0),
    ("lossy substrate", [(1.38, 120e-9)], 1.5 + 0.02j, 1.0),
    ("metal-dielectric hybrid", [(1.46, 90e-9), (0.15 + 3.5j, 25e-9),
                                 (1.46, 90e-9)], 1.52, 1.0),
    ("immersed sup n=1.5 (past TIR)", [(1.38, 120e-9)], 1.0, 1.5),
]


@pytest.mark.parametrize("name,layers_n,n_sub,n_amb", _ISO_CASES,
                         ids=[c[0] for c in _ISO_CASES])
@pytest.mark.parametrize("angle", [0.0, 0.1, 0.35, 0.7, 1.1, 1.4])
def test_o1_isotropic_limit_matches_coatings_tmm(name, layers_n, n_sub, n_amb,
                                                 angle):
    """ORACLE 1.  Lab ``E_y`` is s, lab ``E_x`` is p at ``phi = 0``; R AND T
    must match the audit-verified complex-angle TMM.  Measured worst over the
    full 8x6 matrix: 1.9e-15 (tol 2e-13 for cross-platform LAPACK spread)."""
    layers = [(complex(n) ** 2, d) for n, d in layers_n]
    R, T, _Jr, _Jt = berreman_jones_1d(layers, n_sub, n_amb, WL, angle=angle)
    Rs, Ts = _tmm(layers_n, n_sub, n_amb, angle, 's')
    Rp, Tp = _tmm(layers_n, n_sub, n_amb, angle, 'p')
    assert abs(R[1] - Rs) < 2e-13, f"{name}: R_s"
    assert abs(T[1] - Ts) < 2e-13, f"{name}: T_s"
    assert abs(R[0] - Rp) < 2e-13, f"{name}: R_p"
    assert abs(T[0] - Tp) < 2e-13, f"{name}: T_p"


def test_o1b_bare_interface_exact_fresnel_across_critical():
    """ORACLE 1b.  Closed-form Fresnel, INCLUDING the immediate neighbourhood of
    the critical angle -- an oracle independent of both engines.  Measured worst
    3.9e-16 (tol 1e-13).

    RECORDED CROSS-ENGINE FACT: at ``theta_c - 1e-5`` the coatings TMM takes its
    ``_real_subcritical`` branch and its legacy ``min(sin_t, 0.9999)`` cap fires,
    so IT is 2.2e-2 wrong there while Berreman is exact.  Berreman is the
    correct side of that seam; do not "fix" Berreman towards coatings near
    TIR."""
    nsup, nsub = 1.5, 1.0
    crit = float(np.arcsin(nsub / nsup))
    for a in (0.0, 0.3, 0.6, 0.72, crit - 1e-5, crit - 1e-9,
              crit + 1e-9, 0.8, 1.2):
        ct1 = np.cos(a)
        st2 = nsup * np.sin(a) / nsub
        ct2 = np.sqrt(1 - st2 ** 2 + 0j)
        if ct2.imag < 0:
            ct2 = -ct2
        rs = (nsup * ct1 - nsub * ct2) / (nsup * ct1 + nsub * ct2)
        rp = (nsup / ct1 - nsub / ct2) / (nsup / ct1 + nsub / ct2)
        ts = 2 * nsup * ct1 / (nsup * ct1 + nsub * ct2)
        Ts = float(np.real(nsub * ct2 / (nsup * ct1))) * abs(ts) ** 2
        R, T, _Jr, _Jt = berreman_jones_1d([], nsub, nsup, WL, angle=a)
        assert abs(R[1] - abs(rs) ** 2) < 1e-13, f"R_s at {a}"
        assert abs(R[0] - abs(rp) ** 2) < 1e-13, f"R_p at {a}"
        assert abs(T[1] - Ts) < 1e-13, f"T_s at {a}"

    # the coatings-side cap, recorded so the seam is not mistaken for a bug
    a = crit - 1e-5
    R, _T, _Jr, _Jt = berreman_jones_1d([], nsub, nsup, WL, angle=a)
    Rs_tmm, _ = _tmm([], nsub, nsup, a, 's')
    assert abs(R[1] - Rs_tmm) > 1e-3, (
        "the coatings 0.9999 sin-cap divergence just below the critical angle "
        "has disappeared -- re-derive which side is now exact")


@pytest.mark.parametrize("angle", [0.3, 0.7, 1.2])
@pytest.mark.parametrize("phi", [0.0, 0.35, np.pi / 4, 1.2, np.pi / 2])
def test_o1c_conical_isotropic_is_the_flux_weighted_sp_mixture(angle, phi):
    """ORACLE 1c + CONTRACT.  For an isotropic stack at azimuth ``phi`` the
    per-lab-polarization R/T are the s/p values weighted by the modal
    ADMITTANCES (``eta_p = n/cos(theta)``, ``eta_s = n cos(theta)``), because R/T
    are flux ratios -- not the naive ``cos^2 phi R_p + sin^2 phi R_s`` amplitude
    average, which is wrong by up to 3.5e-4 here.  Measured worst 1.6e-15
    (tol 2e-13)."""
    layers_n = [(1.38, 120e-9), (2.30, 80e-9)]
    layers = [(complex(n) ** 2, d) for n, d in layers_n]
    R, T, _Jr, _Jt = berreman_jones_1d(layers, 1.52, 1.0, WL,
                                       angle=angle, phi=phi)
    Rp, Tp = _tmm(layers_n, 1.52, 1.0, angle, 'p')
    Rs, Ts = _tmm(layers_n, 1.52, 1.0, angle, 's')
    c2, s2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    ct2 = np.cos(angle) ** 2
    for col, (wp, ws) in enumerate(((c2 / ct2, s2), (s2 / ct2, c2))):
        n = wp + ws
        assert abs(R[col] - (wp * Rp + ws * Rs) / n) < 2e-13
        assert abs(T[col] - (wp * Tp + ws * Ts) / n) < 2e-13


# =========================================================================== #
# ORACLE 2 -- uniaxial closed forms
# =========================================================================== #

@pytest.mark.parametrize("angle", [0.0, 0.4, 0.9])
@pytest.mark.parametrize("axis", ['x', 'y', 'z'])
def test_o2_uniaxial_closed_form_both_axis_orientations(axis, angle):
    """ORACLE 2.  A c-axis-ALIGNED uniaxial slab decouples into two scalar
    channels with CLOSED-FORM (n_eq, cos_eq):

      c || y : s sees ``ne``, p sees ``no``            (both plain Snell)
      c || x : s sees ``no``; p is ``n_eq = ne`` with
               ``cos_eq = sqrt(1 - Kx^2/no^2)``
      c || z : s sees ``no``; p is ``n_eq = no`` with
               ``cos_eq = sqrt(1 - Kx^2/ne^2)``

    (``n_eq^2 = Kz_e * eta_p`` from the extraordinary dispersion
    ``Kx^2/eps_perp + Kz^2/eps_par = 1`` and ``eta_p = eps_xx / Kz``.)
    Measured worst 8.9e-16 (tol 1e-13)."""
    no, ne, d = 1.5, 1.7, 300e-9
    n_sup, n_sub = 1.0, 1.52
    Kx = n_sup * np.sin(angle)
    cos0 = np.sqrt(1 - (Kx / n_sup) ** 2 + 0j)
    cos2 = np.sqrt(1 - (Kx / n_sub) ** 2 + 0j)
    table = {
        'y': (np.diag([no ** 2, ne ** 2, no ** 2]).astype(complex),
              (ne, np.sqrt(1 - (Kx / ne) ** 2 + 0j)),
              (no, np.sqrt(1 - (Kx / no) ** 2 + 0j))),
        'x': (np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex),
              (no, np.sqrt(1 - (Kx / no) ** 2 + 0j)),
              (ne, np.sqrt(1 - Kx ** 2 / no ** 2 + 0j))),
        'z': (np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex),
              (no, np.sqrt(1 - (Kx / no) ** 2 + 0j)),
              (no, np.sqrt(1 - Kx ** 2 / ne ** 2 + 0j))),
    }
    eps, (ns_, cs_), (npp, cp_) = table[axis]
    R, T, _Jr, _Jt = berreman_jones_1d([(eps, d)], n_sub, n_sup, WL, angle=angle)
    Rs, Ts = _film_rt(n_sup, ns_, n_sub, cos0, cs_, cos2, d, 's')
    Rp, Tp = _film_rt(n_sup, npp, n_sub, cos0, cp_, cos2, d, 'p')
    assert abs(R[1] - Rs) < 1e-13 and abs(T[1] - Ts) < 1e-13
    assert abs(R[0] - Rp) < 1e-13 and abs(T[0] - Tp) < 1e-13


def test_o2b_inplane_uniaxial_retardance_is_2pi_d_dn_over_lambda():
    """ORACLE 2 (retardance).  The transmitted-Jones diagonal phase difference
    of an in-plane uniaxial slab in air is the geometric retardance plus the
    (exactly computable) Fabry-Perot correction: pin against the same scalar
    Airy oracle for both eigen-channels, then check the residual against the
    ideal ``2 pi d (ne - no) / lambda`` stays inside the Fresnel-etalon
    envelope."""
    no, ne = 1.5, 1.7
    eps = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    for d in (100e-9, 500e-9, 1200e-9):
        _R, _T, _Jr, Jt = berreman_jones_1d([(eps, d)], 1.0, 1.0, WL, angle=0.0)
        one = np.array([1.0 + 0j])
        # exact per-channel transmission amplitude at normal incidence
        amps = []
        for n1 in (ne, no):
            r01 = (1.0 - n1) / (1.0 + n1)
            t01, t12 = 2.0 / (1.0 + n1), 2 * n1 / (n1 + 1.0)
            dl = 2 * np.pi * n1 * d / WL
            amps.append(t01 * t12 * np.exp(1j * dl)
                        / (1 - r01 * r01 * np.exp(2j * dl)))
        assert abs(abs(Jt[0, 0]) - abs(amps[0])) < 1e-12
        assert abs(abs(Jt[1, 1]) - abs(amps[1])) < 1e-12
        meas = float(np.angle(Jt[0, 0] / Jt[1, 1]))
        exact = float(np.angle(amps[0] / amps[1]))
        assert abs(np.angle(np.exp(1j * (meas - exact)))) < 1e-12
        ideal = 2 * np.pi * d * (ne - no) / WL
        # etalon correction is bounded by the single-surface Fresnel phase
        assert abs(np.angle(np.exp(1j * (meas - ideal)))) < 0.12, one.size


# =========================================================================== #
# ORACLE 3 -- energy conservation, incl. the absorbing-substrate flux
# =========================================================================== #

_ENERGY_STACKS = {
    'in-plane rot 30': [(_rot(BIAX, 'z', 0.5), 400e-9)],
    'in-plane 2-layer': [(_rot(BIAX, 'z', 0.5), 400e-9),
                         (_rot(BIAX, 'z', -0.9), 250e-9)],
    'OOP tilt x': [(_rot(BIAX, 'x', 0.61), 400e-9)],
    'OOP tilt y + iso cap': [(2.1, 90e-9), (_rot(BIAX, 'y', 0.61), 400e-9)],
    'twist 3': [(_rot(BIAX, 'z', 0.0), 150e-9), (_rot(BIAX, 'z', 0.6), 150e-9),
                (_rot(BIAX, 'z', 1.2), 150e-9)],
}


@pytest.mark.parametrize("name", list(_ENERGY_STACKS))
@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.7),
                                       (1.1, 0.42), (1.3, np.pi / 2)])
def test_o3_energy_lossless_anisotropic_oblique_and_conical(name, angle, phi):
    """ORACLE 3.  Measured worst 5.3e-15 (tol 1e-12)."""
    R, T, _Jr, _Jt = berreman_jones_1d(_ENERGY_STACKS[name], 1.52, 1.0, WL,
                                       angle=angle, phi=phi)
    assert np.max(np.abs(R + T - 1.0)) < 1e-12


@pytest.mark.parametrize("n_sub", [1.5 + 0.3j, 0.15 + 3.5j, 3.0 + 0.001j])
@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.6, 0.0), (0.6, 0.5)])
@pytest.mark.parametrize("name", ['iso film', 'in-plane aniso', 'OOP aniso'])
def test_o3b_absorbing_substrate_still_closes_R_plus_T(name, n_sub, angle, phi):
    """ORACLE 3 (flux projection) + **F-1 PIN**.  With LOSSLESS layers all the
    absorption is in the exit half-space, so the flux crossing the last
    interface must still give ``R + T = 1`` EXACTLY -- the tightest available
    check on the transmitted-side Poynting projection.

    PRE-FIX this FAILED for the 'OOP aniso' rows at oblique incidence with
    ``|R+T-1|`` up to 9.59e-01 (T was masked to exactly 0.0), because
    ``_forward_flux_kz`` was fed the PUBLIC instead of the INTERNAL eps.
    Measured post-fix worst 4.7e-15 (tol 1e-12)."""
    stacks = {
        'iso film': [(1.38 ** 2, 120e-9)],
        'in-plane aniso': [(_rot(BIAX, 'z', 0.5), 400e-9)],
        'OOP aniso': [(_rot(BIAX, 'x', 0.61), 400e-9)],
    }
    R, T, _Jr, _Jt = berreman_jones_1d(stacks[name], n_sub, 1.0, WL,
                                       angle=angle, phi=phi)
    assert np.all(T > 0.0), f"{name}: transmission masked to zero"
    assert np.max(np.abs(R + T - 1.0)) < 1e-12


@pytest.mark.parametrize("n_sub", [1.52, 1.5 + 0.3j, 0.15 + 3.5j, 3.0 + 0.001j])
@pytest.mark.parametrize("angle", [0.0, 0.3, 0.6, 1.2])
@pytest.mark.parametrize("axis", ['x', 'y'])
def test_f1_offplane_path_matches_rcwa_on_absorbing_substrate(axis, n_sub,
                                                              angle):
    """**F-1 PIN** against the path's OWN stated reference.  A uniform
    (ridge == groove) tensor slab through ``rcwa_jones_1d`` is the same physical
    problem; the berreman docstring claims machine-precision agreement.

    PRE-FIX: ``n_sub = 1.5+0.3j``, ``axis='x'``, ``angle = 0.3`` gave berreman
    T = 0.000000 vs rcwa T = 0.930316 (dT = 9.30e-01).  Post-fix measured worst
    dR 4.1e-15 / dT 5.0e-15 over this 2x4x4 matrix (tol 5e-12)."""
    eps = _rot(BIAX, axis, 0.61)
    R, T, _Jr, _Jt = berreman_jones_1d([(eps, 400e-9)], n_sub, 1.0, WL,
                                       angle=angle)
    _o, Re_, Te_, _J = rcwa_jones_1d(1e-6, eps, eps, n_sub, 1.0, 400e-9, 0.5,
                                     WL, angle=angle, n_orders=3)
    rR = np.array([Re_[0].sum(), Re_[1].sum()])
    rT = np.array([Te_[0].sum(), Te_[1].sum()])
    assert np.max(np.abs(R - rR)) < 5e-12
    assert np.max(np.abs(T - rT)) < 5e-12


def test_f1_lossy_substrate_retain_internal_absorption_closure():
    """**F-1 PIN** on the retained-internals consumer: with T masked to 0 the
    ``sum(A_i) == 1 - R - T`` closure was off by the whole transmitted power."""
    st = BerremanStack(n_substrate=1.5 + 0.3j, n_superstrate=1.0)
    st.add_layer(300e-9, eps=_rot(np.diag([2.1 + 0.05j, 2.45, 2.9])
                                 .astype(complex), 'y', 0.5))
    st.add_layer(80e-9, eps=2.1)
    R, T, _Jr = st.set_source(WL, angle=0.5).solve(retain_internal=True)
    A = st.layer_absorption()
    assert np.all(T > 0.05)
    assert np.max(np.abs(A.sum(axis=0) - (1 - R - T))) < 1e-11


# =========================================================================== #
# ORACLE 4 -- reciprocity
# =========================================================================== #

_RECIP_STACKS = {
    'iso 3-layer': [(1.38 ** 2, 120e-9), (2.3 ** 2, 80e-9), (1.46 ** 2, 200e-9)],
    'in-plane twist': [(_rot(BIAX, 'z', 0.0), 200e-9),
                       (_rot(BIAX, 'z', 0.7), 200e-9),
                       (_rot(BIAX, 'z', 1.4), 200e-9)],
    'OOP tilt + cap': [(2.1, 90e-9), (_rot(BIAX, 'x', 0.61), 300e-9)],
    'lossy in-plane': [(_rot(np.diag([2.1 + 0.05j, 2.45, 2.9]).astype(complex),
                             'z', 0.5), 300e-9), (2.1, 100e-9)],
}


@pytest.mark.parametrize("name", list(_RECIP_STACKS))
@pytest.mark.parametrize("angle", [0.0, 0.55, 1.05])
def test_o4_reciprocity_stack_reversal_planar_mount(name, angle):
    """ORACLE 4.  Reverse the layer order, swap the half-spaces, and enter at
    the Snell-conjugate angle: the TOTAL transmittance is reversal-invariant for
    reciprocal (symmetric-eps) media.  Measured worst 2.4e-15 (tol 1e-12).

    Deliberately restricted to the PLANAR mount: in the flux-weighted lab-pol
    basis (test_o1c) the two lab columns mix s and p with ``theta``-dependent
    weights, and the reversed problem has a DIFFERENT theta, so the lab-basis
    sum is not reversal-invariant at conical -- conical reciprocity is covered
    instead by rotation covariance (test_o5b) composed with this."""
    lay = _RECIP_STACKS[name]
    _Rf, Tf, _a, _b = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=angle)
    a2 = float(np.arcsin(min(1.0 * np.sin(angle) / 1.52, 1.0)))
    _Rr, Tr, _c, _d = berreman_jones_1d(lay[::-1], 1.0, 1.52, WL, angle=a2)
    assert abs(Tf.sum() - Tr.sum()) < 1e-12


# =========================================================================== #
# ORACLE 5 -- twist -> continuum, and rotation covariance
# =========================================================================== #

@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.4, 0.0), (0.4, 0.5)])
def test_o5_twisted_biaxial_converges_to_the_continuum(angle, phi):
    """ORACLE 5.  A 90-deg twist over 1.2 um, sliced N ways: the answer must
    converge MONOTONICALLY and at 2nd order (measured step ratio 4.00), with
    energy closing at every N (measured 3.9e-14 at N = 256)."""
    tot, twist = 1.2e-6, np.pi / 2
    prev, prev_step = None, None
    for N in (16, 32, 64, 128, 256):
        lay = [(_rot(BIAX, 'z', twist * (i + 0.5) / N), tot / N)
               for i in range(N)]
        R, T, _Jr, Jt = berreman_jones_1d(lay, 1.52, 1.0, WL,
                                          angle=angle, phi=phi)
        assert np.max(np.abs(R + T - 1.0)) < 1e-11
        cur = np.concatenate([R, T, Jt.ravel().real, Jt.ravel().imag])
        if prev is not None:
            step = float(np.max(np.abs(cur - prev)))
            if prev_step is not None:
                assert step < prev_step, "twist slicing stopped converging"
                assert 3.0 < prev_step / step < 5.5, (
                    f"twist convergence order left [3, 5.5]: "
                    f"{prev_step / step:.2f}")
            prev_step = step
        prev = cur
    assert prev_step is not None and prev_step < 1e-5


_COVAR_STACKS = {
    'iso 2-layer': [(1.38 ** 2, 120e-9), (2.3 ** 2, 80e-9)],
    'in-plane biax': [(_rot(BIAX, 'z', 0.4), 300e-9)],
    'twist 3': [(_rot(BIAX, 'z', 0.0), 150e-9), (_rot(BIAX, 'z', 0.6), 150e-9),
                (_rot(BIAX, 'z', 1.2), 150e-9)],
    'OOP rotx': [(_rot(BIAX, 'x', 0.61), 300e-9)],
    'OOP roty': [(_rot(BIAX, 'y', 0.61), 300e-9)],
    'gyrotropic': [(np.array([[2.5, 0.2j, 0], [-0.2j, 2.5, 0], [0, 0, 2.5]],
                             dtype=complex), 400e-9)],
    'lossy OOP + cap': [(2.1, 80e-9),
                        (_rot(np.diag([2.1 + 0.05j, 2.45, 2.9]).astype(complex),
                              'y', 0.5), 300e-9)],
}


@pytest.mark.parametrize("name", list(_COVAR_STACKS))
@pytest.mark.parametrize("angle", [0.3, 0.7, 1.1])
@pytest.mark.parametrize("phi", [0.35, 0.9, np.pi / 2, 2.4])
def test_o5b_rotation_covariance_conical_equals_rotated_planar(name, angle,
                                                              phi):
    """ORACLE 5b.  A conical mount at azimuth ``phi`` IS the planar mount on the
    ``R_z(-phi)``-rotated stack, with the Jones conjugated by the same 2-D
    rotation: ``J(phi) == R2(phi) J_rot(0) R2(-phi)``.  This is the strongest
    available gate on the conical ``Kx*Ky`` entries of the Berreman Delta (the
    audit-F1 fix site) and it also carries reciprocity into the conical mount.
    Measured worst 7.7e-15 across iso / in-plane / twist / OOP-x / OOP-y /
    gyrotropic / lossy-OOP (tol 1e-12)."""
    lay = _COVAR_STACKS[name]
    c, s = np.cos(phi), np.sin(phi)
    R2 = np.array([[c, -s], [s, c]])
    _R, _T, Jr, Jt = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=angle, phi=phi)
    Rm = _rot3('z', -phi)
    lay2 = [((Rm @ _t3(e) @ Rm.T), t) for e, t in lay]
    _R2, _T2, Jr2, Jt2 = berreman_jones_1d(lay2, 1.52, 1.0, WL,
                                           angle=angle, phi=0.0)
    assert np.max(np.abs(Jr - R2 @ Jr2 @ R2.T)) < 1e-12
    assert np.max(np.abs(Jt - R2 @ Jt2 @ R2.T)) < 1e-12


def test_o5c_gyrotropic_hermitian_conserves_energy():
    """A Hermitian gyrotropic (magneto-optic) tensor is LOSSLESS even though eps
    is complex; the module advertises gyrotropic support.  Measured worst
    5.6e-15 (tol 1e-12)."""
    for g in (0.02, 0.2):
        eg = np.array([[2.5, 1j * g, 0], [-1j * g, 2.5, 0], [0, 0, 2.5]],
                      dtype=complex)
        assert np.allclose(eg, eg.conj().T)
        for angle, phi in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.6)):
            R, T, _Jr, _Jt = berreman_jones_1d([(eg, 500e-9)], 1.52, 1.0, WL,
                                               angle=angle, phi=phi)
            assert np.max(np.abs(R + T - 1.0)) < 1e-12


# =========================================================================== #
# F-3 -- the two internal cascades agree (the "~2% off" claim, refuted)
# =========================================================================== #

@pytest.mark.parametrize("axis,tilt", [('x', 0.61), ('y', 0.61), ('y', 0.96)])
@pytest.mark.parametrize("angle,phi", [(0.2, 0.0), (0.5, 0.0), (0.5, 0.6),
                                       (0.9, 0.6), (1.2, 2.4)])
def test_f3_native_and_generalized_cascades_agree_on_offplane_oblique(
        axis, tilt, angle, phi):
    """**F-3 PIN (refutation).**  The router comment used to assert the NATIVE
    cascade is "~2% off" for an out-of-plane tensor at oblique incidence.  It is
    not: R, T, jones_r AND jones_t agree to 3.8e-15 (tol 1e-11, loosened for the
    near-grazing conditioning seen in a 4000-config randomized sweep whose worst
    was 1.0e-11).  If this ever regresses, the router is load-bearing again and
    the docstring correction must be revisited."""
    eps = _rot(BIAX, axis, tilt)
    Kx = np.sin(angle) * np.cos(phi)
    Ky = np.sin(angle) * np.sin(phi)
    core = B._solve_core([eps], [400e-9], 1.0, 1.52 ** 2, WL, Kx, Ky)
    Jr_n, Jt_n, R_n, T_n = B._farfield(core)
    R_g, T_g, Jr_g, Jt_g = B._offplane_oblique_solve(
        [eps], [400e-9], 1.0, 1.52 ** 2, WL, Kx, Ky)
    assert np.max(np.abs(R_n - R_g)) < 1e-11
    assert np.max(np.abs(T_n - T_g)) < 1e-11
    assert np.max(np.abs(Jr_n - Jr_g)) < 1e-11
    assert np.max(np.abs(Jt_n - Jt_g)) < 1e-11


@pytest.mark.parametrize("eps_yz", [0.0, 1e-14, 1e-13, 1e-11, 1e-9])
def test_f3b_router_threshold_is_continuous(eps_yz):
    """The ``1e-12 * diag`` off-plane router threshold: crossing it must not
    move the answer.  Measured spread across the threshold 1.4e-14 (tol
    1e-11)."""
    e = np.diag([2.1, 2.45, 2.9]).astype(complex)
    e[1, 2] = e[2, 1] = eps_yz
    R, T, _Jr, _Jt = berreman_jones_1d([(e, 400e-9)], 1.52, 1.0, WL, angle=0.5)
    e0 = np.diag([2.1, 2.45, 2.9]).astype(complex)
    R0, T0, _a, _b = berreman_jones_1d([(e0, 400e-9)], 1.52, 1.0, WL, angle=0.5)
    assert np.max(np.abs(R - R0)) < 1e-11 and np.max(np.abs(T - T0)) < 1e-11


# =========================================================================== #
# ORACLE 6 -- NumPy vs the JAX twin
# =========================================================================== #

_JAX_STACKS = {
    'iso 1': [(2.1, 120e-9)],
    'iso multi': [(1.38 ** 2, 120e-9), (2.3 ** 2, 80e-9), (1.46 ** 2, 200e-9)],
    'in-plane biax': [(_rot(BIAX, 'z', 0.4), 300e-9)],
    'twist 3': [(_rot(BIAX, 'z', 0.0), 150e-9), (_rot(BIAX, 'z', 0.6), 150e-9),
                (_rot(BIAX, 'z', 1.2), 150e-9)],
    'gyrotropic': [(np.array([[2.5, 0.2j, 0], [-0.2j, 2.5, 0], [0, 0, 2.5]],
                             dtype=complex), 400e-9)],
    'lossy in-plane': [(_rot(np.diag([2.1 + 0.05j, 2.45, 2.9]).astype(complex),
                             'z', 0.5), 300e-9), (2.1, 80e-9)],
    'OOP rotx': [(_rot(BIAX, 'x', 0.61), 300e-9)],
    'OOP roty + cap': [(2.1, 80e-9), (_rot(BIAX, 'y', 0.61), 300e-9)],
}


@pytest.mark.parametrize("name", list(_JAX_STACKS))
@pytest.mark.parametrize("n_sub,n_sup", [(1.52, 1.0), (1.5 + 0.3j, 1.0),
                                         (1.0, 1.5)])
@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.6),
                                       (1.1, 2.4)])
def test_o6_numpy_vs_jax_twin_parity(name, n_sub, n_sup, angle, phi):
    """ORACLE 6 + **F-1 twin PIN**.  Measured worst 1.0e-14 over this
    8 x 3 x 4 matrix (tol 1e-11).

    PRE-FIX the 'OOP rotx' / 'OOP roty + cap' rows at ``n_sub = 1.5+0.3j``,
    ``angle = 0.0`` diverged by 9.88e-01 and 9.39e-01: the JAX twin routes a
    concrete out-of-plane tensor to the generalized cascade at EVERY incidence
    (the NumPy router also demands oblique), so the flux-gauge bug zeroed T at
    NORMAL incidence on JAX while NumPy's native cascade was correct."""
    _jax()
    import jax.numpy as jnp
    lay = _JAX_STACKS[name]
    Rn, Tn, Jrn, Jtn = berreman_jones_1d(lay, n_sub, n_sup, WL,
                                         angle=angle, phi=phi)
    layj = [(jnp.asarray(_t3(e), jnp.complex128), t) for e, t in lay]
    Rj, Tj, Jrj, Jtj = berreman_jones_1d(layj, n_sub, n_sup, WL,
                                         angle=angle, phi=phi)
    for a, b in ((Rn, Rj), (Tn, Tj), (Jrn, Jrj), (Jtn, Jtj)):
        assert np.max(np.abs(np.asarray(b) - a)) < 1e-11


# =========================================================================== #
# ORACLE 7 -- mode sorting + cache interiors
# =========================================================================== #

def _n_forward(eps, Kx, Ky):
    gam, _P = np.linalg.eig(B._berreman_delta(np.asarray(eps, dtype=complex),
                                             Kx, Ky))
    tol = 1e-9 * max(1.0, float(np.max(np.abs(gam))))
    return int(np.where(gam.real < -tol, True,
                        np.where(gam.real > tol, False, gam.imag > 0.0)).sum())


def _o7_draws(n):
    """The O7 tensor family: triple-rotated biaxial, half lossy, every fifth
    gyrotropic, ``|Kx|, |Ky| <= 1.4`` so the evanescent regime is reached."""
    rng = np.random.default_rng(7)
    for it in range(n):
        e = np.diag(rng.uniform(1.5, 4.0, 3)).astype(complex)
        for axis in ('x', 'y', 'z'):
            e = _rot(e, axis, rng.uniform(0, np.pi))
        if it % 2:
            e = e + 1j * np.diag(rng.uniform(0, 0.3, 3))
        if it % 5 == 0:
            e = e + np.array([[0, 0.3j, 0], [-0.3j, 0, 0], [0, 0, 0]])
        yield e, rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4)


def _o7_partition_invariants(W, V, lam):
    """``(sum lam, sum lam^2, orthonormal projector of the 4x2 modal block)``
    for one half of a Berreman partition.

    Every entry is invariant under any invertible RIGHT-mixing of the two
    columns -- permutation, scaling, in-subspace rotation -- which is the whole
    gauge freedom two ``eig`` backends can differ by at a FIXED partition.  The
    two power sums determine the forward eigenvalue MULTISET uniquely without
    sorting it (the sort is what tied at ``Re(gam) = +-0`` and produced the
    spurious 0.74 drift the docstring below records), and the projector fixes
    the SUBSPACE those modes span.  Together they say "the same modes are on
    the same side", which is the entire physical content of a partition: the
    cascade contracts over the column order and never sees it.
    """
    Mb = np.vstack([np.asarray(W), np.asarray(V)])           # 4x2
    lam = np.asarray(lam)
    Q, _R = np.linalg.qr(Mb)
    return complex(lam.sum()), complex((lam ** 2).sum()), Q @ Q.conj().T


def _o7_score(eig, n=400):
    """``(worst elementwise, worst invariant, correspondence count, n)`` over
    the O7 family, comparing ``_split_fwd_bwd`` against the JAX twin.

    The element-wise comparison is CONDITIONED on the two backends returning
    the same eigenproblem in the same RAW ORDER; where they do not, only the
    partition-invariant observables are meaningful.  See the test below.
    """
    import jax.numpy as jnp

    from lumenairy.elements._berreman_jax import _layer_modes_jax
    worst_elem, worst_inv, corr = 0.0, 0.0, 0
    total = 0
    for e, Kx, Ky in _o7_draws(n):
        total += 1
        D = B._berreman_delta(np.asarray(e, dtype=complex), Kx, Ky)
        gn, _Pn = np.linalg.eig(D)
        gj = np.asarray(eig(jnp.asarray(D, jnp.complex128))[0])
        scale = max(1.0, float(np.max(np.abs(gn))))
        # PRECONDITION: both backends solved the same eigenproblem.  Without
        # this the invariants below compare two different spectra and prove
        # nothing, so it is asserted per draw rather than assumed.  The bar is
        # deliberately looser than the claims (measured 0.0 on both mounts):
        # its job is only to separate "same spectrum, different bookkeeping"
        # from "different spectrum", and the second is an O(1) event.  The
        # multiset is compared by POWER SUMS, not by sorting: this family puts
        # several modes at ``Re(gam) = +-0``, and a lexicographic sort of that
        # ties -- which is the very artefact this file's 0.74 drift came from.
        set_gap = max(abs(complex(np.sum(gn ** k) - np.sum(gj ** k)))
                      / scale ** k for k in (1, 2, 3, 4))
        assert set_gap < 1e-7, (
            f"numpy and JAX returned DIFFERENT Berreman spectra (worst "
            f"relative power-sum gap {set_gap:.3e}): this is not a partition "
            f"question at all, one of the two eigensolves is wrong")
        mn = B._layer_modes(e, Kx, Ky)
        mj = _layer_modes_jax(jnp.asarray(e, jnp.complex128), Kx, Ky, jnp, eig)
        for half in (0, 3):
            an = _o7_partition_invariants(*mn[half:half + 3])
            aj = _o7_partition_invariants(*(np.asarray(b)
                                            for b in mj[half:half + 3]))
            worst_inv = max(worst_inv,
                            abs(an[0] - aj[0]) / scale,
                            abs(an[1] - aj[1]) / scale ** 2,
                            float(np.max(np.abs(an[2] - aj[2]))))
        if float(np.max(np.abs(gn - gj))) / scale < 1e-11:
            corr += 1
            for a, b in zip(mn, mj):
                worst_elem = max(worst_elem,
                                 float(np.max(np.abs(a - np.asarray(b)))))
    return worst_elem, worst_inv, corr, total


def test_o7_split_fwd_bwd_matches_the_jax_twin_on_physical_tensors():
    """ORACLE 7 (S1-13 re-verified, physical media).  ``_split_fwd_bwd`` and
    ``_berreman_jax._layer_modes_jax`` must put the SAME modes on the SAME
    side -- and, where the question is well posed, in the same column order
    (not merely the same SET: an earlier SORTED comparison of ours reported a
    spurious 0.74 drift purely from ordering ties at ``Re(gam) = +-0``).

    2026-08-12 (``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S4).  The
    element-wise form of this alone read 1.146 on the ubuntu py3.12 JAX job of
    main while measuring 1.1e-14 on both mounts.  ADJUDICATED, and it is not
    the silent-wrong class it looks like: both partition rules are the same
    stable flag-argsort, which keeps each group in the RAW ORDER ``eig``
    returned it in, so the two agree element-wise only while the two ``eig``
    backends also agree on that order.  On both mounts they do -- numpy's and
    jaxlib's LAPACK return the Berreman 4x4 spectrum BIT-IDENTICALLY, raw-order
    gap exactly 0.0 over all 400 draws -- but nothing makes that portable, and
    an eigenvalue ORDER is not physics: the cascade contracts over the column
    order and never sees it.

    So the claim is split at the point where it stops being well posed:

    * the PRECONDITION, per draw -- both backends solved the same eigenproblem
      (the spectra agree as MULTISETS, compared by power sums rather than by
      sorting, since sorting is what tied in the first place; measured 0.0);
    * the PARTITION, on every draw -- the partition-INVARIANT observables of
      each half agree: the two eigenvalue power sums, which fix the forward
      multiset without sorting it, and the orthonormal projector, which fixes
      the subspace those modes span (measured worst 5.6e-15, tol 1e-11).
      This is the whole physical content of "same partition";
    * the COLUMN ORDER, on the draws where the two backends returned the raw
      spectrum in the same order -- the original element-wise comparison,
      verbatim (measured worst 1.2e-14, tol 1e-11, 400 of 400 draws on both
      mounts).

    The fail-before is the test below, which shows the restructured claim still
    fires on a real partition disagreement and does NOT fire on a mere
    re-ordering."""
    _jax()
    from lumenairy.elements.rcwa import _jax_eig_stable
    worst_elem, worst_inv, corr, total = _o7_score(_jax_eig_stable())
    assert worst_inv < 1e-11, (
        f"numpy/JAX PARTITION drift {worst_inv:.3e}: the two twins put "
        f"different modes on the same side, or span different subspaces with "
        f"them.  This is a physics disagreement, not a gauge one")
    print(f"\nO7 partition: {corr} of {total} draws in raw-order "
          f"correspondence; worst invariant {worst_inv:.3e}, worst "
          f"elementwise over the correspondence class {worst_elem:.3e}")
    assert worst_elem < 1e-11, (
        f"numpy/JAX mode partition drift {worst_elem:.3e} on draws where both "
        f"eig backends returned the raw spectrum in the SAME order -- there "
        f"the stable flag-argsort must reproduce the column order too")


def test_o7_partition_claim_survives_a_reorder_and_still_catches_a_side_flip():
    """THE FAIL-BEFORE for the O7 partition claim, driven both ways.

    ``FIX_RUNNER_PINS_2026_08_12`` S4.  Two injectors on the ``eig`` the JAX
    twin is HANDED -- ``_layer_modes_jax`` takes it as a parameter, so nothing
    is monkeypatched and both arms are the shipped partition rule:

    (a) a raw-order PERMUTATION, which is what a different LAPACK build is
        entitled to return and what the CI job's 1.146 was.  It must knock the
        element-wise comparison over (measured 1.566, the same O(1) the runner
        read) and leave the partition invariants where they were (5.618e-15
        injected against 5.618e-15 shipped -- not merely small, the same
        reading, because the invariants cannot see a column swap);

    (b) a genuine SIDE FLIP -- one eigenvalue nudged across the classifier cut,
        so a mode really does change half-space.  The restructured claim MUST
        fail on it (measured invariant drift 1.826), or the restructure traded
        a runner-fragile pin for a vacuous one.

    Only the cut position and the column order are injected; the geometry, the
    materials, the operator and the eigenvalues are the shipped ones."""
    _jax()
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import _jax_eig_stable
    eig = _jax_eig_stable()
    base_elem, base_inv, corr, total = _o7_score(eig, n=80)
    assert corr == total, (
        f"only {corr} of {total} draws have the two eig backends in raw-order "
        f"correspondence on this build, so injector (a) has nothing to "
        f"disturb and this fail-before is vacuous")

    def eig_reordered(A):
        lam, V = eig(A)
        p = jnp.asarray([1, 0, 3, 2])
        return lam[p], V[:, p]

    def eig_side_flip(A):
        """Push the FORWARD mode nearest the cut across it: a REAL
        disagreement about which half-space a mode belongs to, not a gauge
        difference.  The nudge is 3e-9 of the spectrum scale -- above the
        classifier's own ``1e-9`` cut, so the flag really flips, and 33x below
        the same-eigenproblem precondition, so the two backends are still
        solving the same problem."""
        lam, V = eig(A)
        sc = jnp.maximum(jnp.max(jnp.abs(lam)), 1.0)
        re, im = jnp.real(lam), jnp.imag(lam)
        tol = 1e-9 * sc
        is_fwd = jnp.where(re < -tol, True, jnp.where(re > tol, False, im > 0))
        k = jnp.argmin(jnp.where(is_fwd, jnp.abs(re), jnp.inf))
        return lam.at[k].add(3e-9 * sc), V

    # (a) a re-ordering breaks the ELEMENT-WISE claim ...
    from lumenairy.elements._berreman_jax import _layer_modes_jax
    ra_elem, ra_inv, ra_corr, _ = _o7_score(eig_reordered, n=80)
    assert ra_corr < total, (
        f"the reorder injector left all {ra_corr} draws in raw-order "
        f"correspondence, so it did not reproduce the CI condition")
    assert ra_elem < 1e-11, (
        f"element-wise drift {ra_elem:.3e} on draws the reorder injector left "
        f"IN correspondence -- those are draws it could not disturb (an "
        f"eigenvalue tie), so they must still agree column by column")
    with pytest.raises(AssertionError, match="mode partition drift"):
        # the original claim, verbatim, on the injected arm
        worst = 0.0
        for e, Kx, Ky in _o7_draws(80):
            mn = B._layer_modes(e, Kx, Ky)
            mj = _layer_modes_jax(jnp.asarray(e, jnp.complex128), Kx, Ky,
                                  jnp, eig_reordered)
            for a, b in zip(mn, mj):
                worst = max(worst, float(np.max(np.abs(a - np.asarray(b)))))
        assert worst < 1e-11, f"numpy/JAX mode partition drift {worst:.3e}"
    # ... and leaves the PARTITION invariants where they were.  Not merely
    # "still small": the same reading, to the bar the claim itself uses.  (It
    # is bit-identical on both mounts -- 5.618e-15 either way -- but the QR
    # behind the projector is entitled to an ulp under a column swap, so the
    # assertion is a bar rather than an ``==``.)
    assert abs(ra_inv - base_inv) < 1e-11, (
        f"a pure column re-ordering moved the partition invariants "
        f"{base_inv:.4e} -> {ra_inv:.4e}.  They are supposed to be blind to "
        f"it by construction, so one of them is not actually invariant")

    # (b) a real side flip still breaks the RESTRUCTURED claim.
    _fe, sf_inv, _c, _t = _o7_score(eig_side_flip, n=80)
    assert sf_inv > 1e-2, (
        f"pushing a mode across the classifier cut moved the partition "
        f"invariants by only {sf_inv:.3e} (measured 1.826): the restructured "
        f"claim would not catch a real partition disagreement")


def test_o7_split_fwd_bwd_matches_the_jax_twin_in_the_degenerate_fallback():
    """ORACLE 7 (S1-13, the DEGENERATE branch).  The S1-13 fix aligned the two
    backends specifically for inputs where NOT exactly two modes flag forward --
    numpy used to rank by decay while JAX kept flag-then-index order.  Physical
    symmetric media almost always give 2 + 2, so this drives the branch with
    GENERAL complex ``Delta`` inputs (the unreachable-but-latent bianisotropic
    class the fix was written for), where ~21% of draws land there
    (measured 8493 / 40000).  Both the hit count and element-wise agreement are
    pinned: measured worst 0.0 over the hits."""
    _jax()
    import jax.numpy as jnp

    from lumenairy.elements._berreman_jax import _layer_modes_jax
    from lumenairy.elements.rcwa import _jax_eig_stable
    eig = _jax_eig_stable()
    rng = np.random.default_rng(7)
    hits, worst = 0, 0.0
    for it in range(300):
        e = (rng.normal(size=(3, 3)) * 3
             + 1j * rng.normal(size=(3, 3)) * 3).astype(complex)
        if it % 3 == 0:
            e = e + e.T
        Kx, Ky = rng.uniform(-2, 2), rng.uniform(-2, 2)
        if _n_forward(e, Kx, Ky) == 2:
            continue
        hits += 1
        mn = B._layer_modes(e, Kx, Ky)
        mj = _layer_modes_jax(jnp.asarray(e, jnp.complex128), Kx, Ky, jnp, eig)
        for a, b in zip(mn, mj):
            worst = max(worst, float(np.max(np.abs(a - np.asarray(b)))))
    assert hits >= 20, (
        f"the degenerate n_forward != 2 branch was reached only {hits} times -- "
        f"the S1-13 twin-agreement claim is no longer exercised")
    assert worst < 1e-11, (
        f"numpy/JAX partition drift in the degenerate branch: {worst:.3e}")


def test_o7b_gain_layer_matches_the_scalar_tmm_exactly():
    """ORACLE 7 (gain edge, REFUTATION).  The native split ranks modes by DECAY,
    so for a GAIN layer it labels the physically-BACKWARD mode 'forward'
    (measured: ``gam`` forward set has ``Im(gam) < 0``).  Under the
    ``[W; -V] <-> -lam`` pairing symmetry of the native cascade that is an EXACT
    relabelling -- and it is the numerically stable one, so nothing overflows.
    Pinned against the scalar TMM at two thicknesses (measured 0.0 difference)
    so a future "fix" to the sort cannot silently change the answer.

    F-2 / ACTIVE MEDIA -- DECLINED WITH MEASUREMENT.  ``rcwa_jones_1d`` raises
    ``_EnergyError`` on the very same gain slab (measured ``sum R+T = 3.253``);
    berreman deliberately does NOT, because
    ``test_audit_v5_24_2_g01_conventions.py::
    test_s1_6_berreman_public_path_raw_eps_absorbs_with_positive_imag``
    contract-locks the gain-slab return as this module's raw-eps
    sign-convention oracle, and the value is exact.  The W6 energy tripwire is
    therefore gated on ``_is_passive``; the divergence is recorded, not
    "fixed"."""
    n_gain, angle = 1.5 - 0.3j, 0.3
    gam, _P = np.linalg.eig(B._berreman_delta(
        (n_gain ** 2) * _I3, np.sin(angle), 0.0))
    fwd, _bwd = B._split_fwd_bwd(gam)
    assert np.all(gam[fwd].real < 0.0)
    assert np.all(gam[fwd].imag < 0.0), (
        "the gain-layer mode sort no longer selects by decay -- re-verify the "
        "TMM agreement below before trusting it")
    for d in (2e-6, 5e-6):
        core = B._solve_core([(n_gain ** 2) * _I3], [d], 1.0, 1.52 ** 2, WL,
                             np.sin(angle), 0.0)
        _Jr, _Jt, R, T = B._farfield(core)
        Rt, Tt, _ph = coating_reflectance([(n_gain, d)], WL, angle=angle,
                                          n_substrate=1.52, n_ambient=1.0,
                                          polarization='s')
        assert abs(R[1] - float(np.atleast_1d(Rt)[0])) < 1e-9 * max(1.0, R[1])
        assert abs(T[1] - float(np.atleast_1d(Tt)[0])) < 1e-9 * max(1.0, T[1])
        assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    # the public entry must NOT raise on a gain LAYER (S1-6 contract) ...
    Rp, Tp, _a, _b = berreman_jones_1d([((n_gain ** 2) * _I3, 2e-6)], 1.52, 1.0,
                                       WL, angle=angle)
    assert Rp[1] + Tp[1] > 1.0
    assert not B._is_passive([(n_gain ** 2) * _I3], 1.0, 1.52 ** 2)
    # ... while a PASSIVE stack keeps the tripwire, and a gain SUPERSTRATE
    # (where the flux normalisation itself inverts) is still rejected.
    assert B._is_passive([np.diag([2.1 + 0.05j, 2.45, 2.9]).astype(complex)],
                         1.0, 1.52 ** 2)
    gyro = np.array([[2.5, 0.2j, 0], [-0.2j, 2.5, 0], [0, 0, 2.5]],
                    dtype=complex)
    assert B._is_passive([gyro], 1.0, 1.52 ** 2), (
        "a Hermitian gyrotropic tensor is LOSSLESS -- classifying it active "
        "would silently disable the tripwire for magneto-optic stacks")
    with pytest.raises(ValueError, match="gain incidence medium"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, n_gain, WL, angle=angle)


def test_o7c_cache_cold_equals_warm_and_survives_lru_eviction():
    """ORACLE 7 (cache).  Byte-identical cold vs warm, and a full LRU turnover
    (256 mode entries / 512 interface entries) leaves the result unchanged."""
    lay = [(_rot(BIAX, 'z', 0.4), 300e-9), (2.1, 80e-9)]
    B._clear_berreman_mode_cache()
    cold = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=0.5, phi=0.3)
    warm = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=0.5, phi=0.3)
    assert all(np.array_equal(a, b) for a, b in zip(cold, warm))
    for k in range(400):                       # force eviction of both LRUs
        berreman_jones_1d([(2.0 + 0.001 * k, 100e-9)], 1.52, 1.0, WL,
                          angle=0.5, phi=0.3)
    assert len(B._MODE_CACHE) == B._MODE_CACHE_SIZE
    assert len(B._IFACE_CACHE) == B._IFACE_CACHE_SIZE
    after = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=0.5, phi=0.3)
    assert all(np.array_equal(a, b) for a, b in zip(cold, after))
    B._clear_berreman_mode_cache()
    assert len(B._MODE_CACHE) == 0 and len(B._IFACE_CACHE) == 0
    again = berreman_jones_1d(lay, 1.52, 1.0, WL, angle=0.5, phi=0.3)
    assert all(np.array_equal(a, b) for a, b in zip(cold, again))


def test_f5_cached_arrays_are_frozen_read_only():
    """**F-5 PIN.**  Both LRUs hand out the STORED arrays and their docstrings
    assert read-only use.  PRE-FIX nothing enforced it:
    ``_layer_modes_cached(...)[0][0, 0] = 999`` made the very next call return
    999 -- a silent cross-solve poisoning.  The sibling ``pmm/_core`` freezes its
    cached arrays the same way."""
    B._clear_berreman_mode_cache()
    eps = _rot(BIAX, 'z', 0.4)
    modes = B._layer_modes_cached(eps, 0.5, 0.1)
    for blk in modes:
        assert isinstance(blk, np.ndarray) and not blk.flags.writeable
    with pytest.raises(ValueError):
        modes[0][0, 0] = 999.0
    M = B._modes_to_M(*[np.asarray(b) for b in
                        (modes[0], modes[1], modes[3], modes[4])])
    iface = B._interface_smatrix_cached(M, M)
    for blk in iface:
        assert isinstance(blk, np.ndarray) and not blk.flags.writeable
    tri = B._offplane_condensed_M_cached(
        eps, np.array([[0.5]], dtype=complex), np.array([[0.1]], dtype=complex))
    assert not tri[0].flags.writeable


def test_f5b_interface_cache_key_carries_shape():
    """**F-5b.**  ``_interface_smatrix_cached`` keyed on raw bytes only, while
    its sibling ``_layer_modes_cached`` also keys on shape.  Unreachable at the
    fixed 4x4 Berreman block size, but keying on bytes alone is the latent form
    of the bug -- pinned so the key cannot silently lose the shape again."""
    src = inspect.getsource(B._interface_smatrix_cached)
    tree = ast.parse(src.strip()).body[0]
    keys = [n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "key" for t in n.targets)]
    assert keys, "the interface cache no longer builds a `key` tuple"
    attrs = {n.attr for n in ast.walk(keys[0]) if isinstance(n, ast.Attribute)}
    assert "shape" in attrs, "interface cache key dropped the shape component"


# =========================================================================== #
# F-2 -- guards, sibling-identical to rcwa / pmm
# =========================================================================== #

@pytest.mark.parametrize("angle", [np.pi / 2, 2.0, 10.0, -2.0, float('nan')])
def test_f2_backside_and_nonfinite_angle_raise(angle):
    """**F-2 PIN.**  ``|angle| >= pi/2`` (and NaN) must raise with the family's
    message.  PRE-FIX: ``angle = 2.0`` returned R = [0.01771, 0.19647] -- the
    ``pi - 2.0`` front-side answer for the WRONG geometry -- ``angle = 10.0``
    was accepted too, and ``pi/2`` / NaN raised a bare ``LinAlgError``."""
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, 1.0, WL, angle=angle)
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        (BerremanStack(n_substrate=1.52).add_layer(100e-9, eps=2.1)
         .set_source(WL, angle=angle))
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, 1.0, WL, theta=angle)


def test_f2_nonpropagating_and_gain_superstrate_raise():
    """**F-2 PIN.**  PRE-FIX a metallic superstrate ``n_sup = 0.15+3.5j`` at
    ``theta = 0.5`` returned ``T = [30.78, 30.73]`` -- a 3000% energy violation
    -- silently, and a GAIN superstrate went unchecked.  ``rcwa_jones_1d``
    raises on both; berreman now routes through the same
    ``_require_propagating_incidence``."""
    with pytest.raises(ValueError, match="non-propagating"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, 0.15 + 3.5j, WL, angle=0.5)
    with pytest.raises(ValueError, match="gain incidence medium"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, 1.5 - 0.01j, WL, angle=0.5)
    st = BerremanStack(n_substrate=1.52, n_superstrate=0.15 + 3.5j)
    st.add_layer(100e-9, eps=2.1).set_source(WL, angle=0.5)
    with pytest.raises(ValueError, match="non-propagating"):
        st.solve()


def test_f2_energy_tripwire_on_a_lossy_incidence_medium():
    """**F-2 PIN.**  A LOSSY superstrate makes the per-wave flux normalisation
    non-physical (the family-wide caveat on ``_forward_flux_kz``).  PRE-FIX
    berreman returned ``T = [1.055, 1.142]``, ``R + T = 1.086`` per pol,
    silently; ``rcwa_jones_1d`` raises ``_EnergyError``."""
    with pytest.raises(ValueError, match="energy non-conservation"):
        berreman_jones_1d([(2.1, 100e-9)], 1.0, 1.5 + 0.5j, WL, angle=0.5)


@pytest.mark.parametrize("thickness", [0.0, -100e-9, float('nan')])
def test_f2_nonpositive_thickness_raises_in_the_functional_entry(thickness):
    """**F-2 PIN.**  ``BerremanStack.add_layer`` has always rejected
    ``thickness <= 0``; the FUNCTIONAL entry silently accepted it (measured
    pre-fix: ``t = -100 nm`` returned R = [0.02268, 0.02979])."""
    with pytest.raises(ValueError, match="thickness must be > 0"):
        berreman_jones_1d([(2.1, thickness)], 1.52, 1.0, WL, angle=0.3)


@pytest.mark.parametrize("wavelength", [0.0, -1e-6, float('nan')])
def test_f2_nonpositive_wavelength_raises(wavelength):
    """**F-2 PIN.**  PRE-FIX ``wavelength = 0`` raised a bare
    ``ZeroDivisionError`` and a NEGATIVE wavelength was silently accepted."""
    with pytest.raises(ValueError, match="wavelength must be"):
        berreman_jones_1d([(2.1, 100e-9)], 1.52, 1.0, wavelength, angle=0.3)


@pytest.mark.parametrize("bad", [float('nan'), float('inf')])
def test_f2_nonfinite_material_values_raise(bad):
    """**F-2 PIN.**  A NaN/inf index or permittivity used to surface as a bare
    ``LinAlgError`` from deep inside ``np.linalg.eig`` (audit P3's class)."""
    with pytest.raises(ValueError, match="not finite"):
        berreman_jones_1d([(bad, 100e-9)], 1.52, 1.0, WL, angle=0.3)
    with pytest.raises(ValueError, match="not finite"):
        berreman_jones_1d([(2.1, 100e-9)], bad, 1.0, WL, angle=0.3)


def test_f2_zero_eps_zz_raises():
    """**F-2 PIN.**  The Berreman Delta eliminates ``Ez``/``Hz`` through
    ``1/eps_zz``; ``eps_zz = 0`` used to reach ``eig`` as inf and raise a bare
    ``LinAlgError``."""
    e = np.diag([2.1, 2.1, 0.0]).astype(complex)
    with pytest.raises(ValueError, match="eps_zz = 0"):
        berreman_jones_1d([(e, 100e-9)], 1.52, 1.0, WL, angle=0.3)


def test_f2_jax_path_shares_the_guards():
    """**F-2 PIN (twin).**  The differentiable entry gets the same concrete
    guards -- a tracer skips them, a CONCRETE value does not."""
    _jax()
    import jax.numpy as jnp
    with pytest.raises(ValueError, match=r"\|angle\| < pi/2"):
        berreman_jones_1d([(jnp.asarray(2.1 + 0j), 100e-9)], 1.52, 1.0, WL,
                          angle=2.0)
    with pytest.raises(ValueError, match="non-propagating"):
        berreman_jones_1d([(jnp.asarray(2.1 + 0j), 100e-9)], 1.52,
                          0.15 + 3.5j, WL, angle=0.5)
    with pytest.raises(ValueError, match="thickness must be > 0"):
        berreman_jones_1d([(jnp.asarray(2.1 + 0j), -100e-9)], 1.52, 1.0, WL,
                          angle=0.3)


def test_f2_guards_do_not_disturb_the_traced_solve():
    """The guards must be tracer-transparent: a jitted / differentiated solve
    still runs and matches the concrete value."""
    jax = _jax()
    import jax.numpy as jnp
    eps = jnp.asarray(_rot(BIAX, 'z', 0.4), jnp.complex128)

    def f(t):
        R, _T, _Jr, _Jt = berreman_jones_1d([(eps, t)], 1.52, 1.0, WL,
                                            angle=jnp.asarray(0.4))
        return R[0]
    val = float(jax.jit(f)(jnp.asarray(300e-9)))
    g = float(jax.grad(f)(jnp.asarray(300e-9)))
    Rc, _T, _Jr, _Jt = berreman_jones_1d([(_rot(BIAX, 'z', 0.4), 300e-9)],
                                         1.52, 1.0, WL, angle=0.4)
    assert abs(val - Rc[0]) < 1e-11
    assert np.isfinite(g) and abs(g) > 0.0


# =========================================================================== #
# F-4 -- inert parameters
# =========================================================================== #

def test_f4_farfield_has_no_inert_parameters():
    """**F-4 PIN.**  ``_farfield`` used to take ``(core, eps_sup, eps_sub, Kx,
    Ky)`` and reference NONE of the last four -- the audit's recurring
    inert-parameter class.  This AST check keeps every parameter of the
    module's core operators live."""
    for fn in (B._farfield, B._solve_core, B._offplane_oblique_solve,
               B._offplane_condensed_M, B._flux, B._berreman_delta,
               B._check_inputs, B._check_energy_pols, B._checked_angle,
               B._is_passive, B._split_fwd_bwd, B._layer_modes):
        tree = ast.parse(inspect.getsource(fn).strip()).body[0]
        params = [a.arg for a in tree.args.args]
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        dead = [p for p in params if p not in used]
        assert not dead, f"{fn.__name__} has inert parameters {dead}"


# =========================================================================== #
# class / functional consistency and the internal observables
# =========================================================================== #

@pytest.mark.parametrize("name", ['iso', 'in-plane', 'OOP'])
@pytest.mark.parametrize("n_sub", [1.52, 1.5 + 0.3j])
@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.6)])
def test_stack_is_bit_identical_to_the_functional_entry(name, n_sub, angle,
                                                       phi):
    """Measured 0.0 (bit-identical) across the matrix, including
    ``jones_transmission`` (the A1 retained transmission Jones)."""
    lay = {'iso': [(1.38 ** 2, 120e-9), (2.3 ** 2, 80e-9)],
           'in-plane': [(_rot(BIAX, 'z', 0.4), 300e-9)],
           'OOP': [(_rot(BIAX, 'y', 0.61), 300e-9), (2.1, 80e-9)]}[name]
    R1, T1, Jr1, Jt1 = berreman_jones_1d(lay, n_sub, 1.0, WL,
                                         angle=angle, phi=phi)
    st = BerremanStack(n_substrate=n_sub, n_superstrate=1.0)
    for e, t in lay:
        st.add_layer(t, eps=e)
    R2, T2, Jr2 = st.set_source(WL, angle=angle, phi=phi).solve()
    assert np.array_equal(R1, R2) and np.array_equal(T1, T2)
    assert np.array_equal(Jr1, Jr2)
    assert np.array_equal(Jt1, st.jones_transmission())


@pytest.mark.parametrize("name", ['lossy iso', 'lossy in-plane', 'lossy OOP'])
@pytest.mark.parametrize("angle,phi", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.6)])
def test_layer_absorption_closes_against_the_far_field(name, angle, phi):
    """``sum_i A_i == 1 - R - T`` -- internal amplitudes vs the half-space far
    field, which also exercises the conjugated-gauge retain path of the
    generalized cascade.  Measured worst 1.6e-15 (tol 1e-12)."""
    lossy = np.diag([2.1 + 0.05j, 2.45, 2.9]).astype(complex)
    lay = {'lossy iso': [(1.38 ** 2, 120e-9), ((0.15 + 3.5j) ** 2, 25e-9),
                         (1.46 ** 2, 90e-9)],
           'lossy in-plane': [(_rot(lossy, 'z', 0.5), 300e-9), (2.1, 80e-9)],
           'lossy OOP': [(_rot(lossy, 'y', 0.5), 300e-9), (2.1, 80e-9)]}[name]
    st = BerremanStack(n_substrate=1.52, n_superstrate=1.0)
    for e, t in lay:
        st.add_layer(t, eps=e)
    R, T, _Jr = st.set_source(WL, angle=angle, phi=phi).solve(
        retain_internal=True)
    A = st.layer_absorption()
    assert np.max(np.abs(A.sum(axis=0) - (1 - R - T))) < 1e-12


def test_internal_field_is_continuous_across_an_internal_interface():
    """The tangential E/H must be continuous at a layer boundary; the
    reconstruction bridges two different modal bases there.  Measured relative
    jump at +-1 pm: 3e-5 (i.e. the smooth-field slope, not a discontinuity)."""
    st = BerremanStack(n_substrate=1.52, n_superstrate=1.0)
    st.add_layer(300e-9, eps=_rot(BIAX, 'z', 0.4))
    st.add_layer(150e-9, eps=2.1)
    st.set_source(WL, angle=0.5, phi=0.6).solve(retain_internal=True)
    lo = st.internal_field(300e-9 - 1e-12)
    hi = st.internal_field(300e-9 + 1e-12)
    assert lo["layer"] == 0 and hi["layer"] == 1
    for c in ("Ex", "Ey", "Hx", "Hy"):
        scale = max(abs(lo[c]), 1e-30)
        assert abs(lo[c] - hi[c]) / scale < 1e-4, c
