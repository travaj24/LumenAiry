"""AUDIT: out-of-plane generator factor-i fix (loose-ends round 2026-07-14).

THE BUG (multi-release, v5.11.0..v5.21.5): the off-plane cross-blocks of the
generalized layer generator -- the ``E <- E`` block ``A`` (from ``ezx, ezy``)
and the ``H <- H`` block ``B`` (from ``exz, eyz``) -- were written with REAL
coefficients in the modal-u state convention whose in-plane ``P``/``Q``
blocks demand relative ``-/+i`` factors there.  Consequences, all confined to
OUT-OF-PLANE tensors at OBLIQUE incidence (``A = B = 0`` at normal incidence
and for in-plane tensors, which is why every other regime stayed exact):

* the extraordinary-wave propagation constants inside the layer came out
  artificially ``+/-`` SYMMETRIC (``kz_e/k0 = +/-1.5646`` on the tilted-35deg
  uniaxial probe) instead of the exact asymmetric det-condition roots
  (``{-1.5214, +1.6090}``) -- a 3-5% kz error;
* the mode fields violated Maxwell (no constant re-scaling could fix them --
  probed exhaustively), so internal fields broke the local Poynting theorem
  by ~7% and the density- vs flux-based absorption attributions disagreed at
  3e-3 while each budget still closed (telescoping hides it).

WHY NO GATE CAUGHT IT: the ``_berreman4x4`` test oracle shared the same
prototype ancestry and carried the SAME real-coefficient blocks -- every
"1e-10" agreement between solver and oracle was CIRCULAR.  Lossless energy
closure is insensitive (the lossless-trap rule), and the ordinary wave (which
dominates many observables) was exact.

THE ANCHORS HERE (independent of all shared code):

1. the EXACT dispersion relation ``det(k x (k x .) + eps) = 0`` -- closed-form
   physics, no solver machinery: ``eig(G)`` must reproduce its four ``kz``
   roots to machine precision (and the extraordinary pair must be
   ASYMMETRIC);
2. per-mode MAXWELL residuals: every generalized eigenmode, read in the
   public gauge, must satisfy all six curl equations to machine precision;
3. the local POYNTING theorem inside OOP layers at oblique:
   ``-dSz/dz = k0 * Im(E* . eps . E)`` pointwise (the diagnostic that
   uncovered the bug).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements import berreman as bb

_C = complex
WL = 1.55e-6
K0 = 2.0 * np.pi / WL


def _tilted(tilt_deg, loss=0.08, no=1.5, ne=1.7):
    no2, ne2 = (no ** 2 + loss * 1j), (ne ** 2 + loss * 1j)
    th = np.deg2rad(tilt_deg)
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Rm @ np.diag([ne2, no2, no2]) @ Rm.T


def _exact_kz_roots(eps, Kx, Ky):
    """The four exact ``kz/k0`` roots of ``det(kkT - |k|^2 I + eps) = 0``
    (closed-form physics; quartic sampled exactly and rooted)."""
    e = np.asarray(eps, dtype=_C)

    def detM(kz):
        k = np.array([Kx, Ky, kz], dtype=_C)
        return np.linalg.det(np.outer(k, k) - (k @ k) * np.eye(3) + e)

    zs = np.array([0.0, 0.7, -0.9, 1.3, -1.6, 2.1])
    coeffs = np.polyfit(zs, np.array([detM(z) for z in zs]), 4)
    return np.sort_complex(np.roots(coeffs))


def _gen_modes(eps_pub, Kx, Ky):
    """Generalized modes via the production path (berreman's condensed M ->
    rcwa._core._layer_eigenmodes_tensor), mapped to the PUBLIC gauge."""
    o = lambda v: np.array([[v]], dtype=_C)     # noqa: E731
    Ml, lf, lb = bb._offplane_condensed_M(np.conj(eps_pub), o(Kx), o(Ky))
    n = Ml.shape[0] // 2
    return (np.conj(Ml[:n, :n]), -np.conj(Ml[n:, :n]), np.conj(lf),
            np.conj(Ml[:n, n:]), -np.conj(Ml[n:, n:]), np.conj(lb))


@pytest.mark.parametrize("tilt,theta,phi", [
    (35.0, 0.45, 0.6),          # the discovery probe (conical)
    (35.0, 0.45, 0.0),          # planar oblique
    (-20.0, 0.30, 1.1),         # opposite tilt, other azimuth
])
def test_generator_eigenvalues_match_exact_dispersion(tilt, theta, phi):
    """eig(G) == the exact det-condition kz roots (machine precision), and
    the extraordinary pair is ASYMMETRIC (the legacy blocks forced +/-
    symmetry -- the smoking gun)."""
    eps = _tilted(tilt)
    Kx, Ky = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    _Wf, _Vf, lamf, _Wb, _Vb, lamb = _gen_modes(eps, Kx, Ky)
    # forward decay exp(-lam k0 z): kz = +i*lam maps eigenvalues to kz/k0
    kz_solver = np.sort_complex(np.concatenate([1j * lamf, 1j * lamb]))
    kz_exact = _exact_kz_roots(eps, Kx, Ky)
    assert np.max(np.abs(kz_solver - kz_exact)) < 1e-9
    # the EXTRAORDINARY pair is asymmetric (the o-pair stays symmetric --
    # the ordinary wave does not see the tilt).  The legacy blocks forced
    # ALL FOUR roots into +/- pairs; the exact set has exactly one such
    # pair, so the count of roots whose negative is also a root must be 2.
    n_paired = sum(1 for r in kz_exact
                   if np.min(np.abs(kz_exact + r)) < 1e-6)
    assert n_paired == 2, f"{n_paired} +/- paired roots (legacy bug gives 4)"


@pytest.mark.parametrize("tilt,theta,phi", [(35.0, 0.45, 0.6),
                                            (-20.0, 0.30, 1.1)])
def test_generalized_modes_satisfy_maxwell(tilt, theta, phi):
    """Every generalized eigenmode (public gauge) satisfies all six Maxwell
    rows to machine precision -- the legacy blocks failed curlH-x / curlE-x /
    curlE-y at 1e-2..5e-2 under EVERY constant re-scaling."""
    e = np.asarray(_tilted(tilt))
    Kx, Ky = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    Wf, Vf, lamf, Wb, Vb, lamb = _gen_modes(e, Kx, Ky)
    for (W, V, lam) in ((Wf, Vf, lamf), (Wb, Vb, lamb)):
        for j in range(W.shape[1]):
            Ex, Ey = W[0, j], W[1, j]
            Hx, Hy = -1j * V[0, j], -1j * V[1, j]
            dz = -lam[j] * K0
            Hz = Kx * Ey - Ky * Ex
            Ez = (-(Kx * Hy - Ky * Hx) - e[2, 0] * Ex - e[2, 1] * Ey) / e[2, 2]
            rows = (
                1j * K0 * Ky * Hz - dz * Hy
                + 1j * K0 * (e[0, 0] * Ex + e[0, 1] * Ey + e[0, 2] * Ez),
                dz * Hx - 1j * K0 * Kx * Hz
                + 1j * K0 * (e[1, 0] * Ex + e[1, 1] * Ey + e[1, 2] * Ez),
                1j * K0 * Ky * Ez - dz * Ey - 1j * K0 * Hx,
                dz * Ex - 1j * K0 * Kx * Ez - 1j * K0 * Hy,
            )
            assert max(abs(r) for r in rows) / K0 < 1e-12


def test_oracle_delta_matches_exact_dispersion():
    """The (fixed) _berreman4x4 test oracle's Delta now also reproduces the
    exact dispersion -- it shared the defect with the solver (circular
    validation), so it is pinned to the independent anchor here."""
    from tests.unit._berreman4x4 import _berreman_delta
    eps = _tilted(35.0)
    Kx, Ky = np.sin(0.45) * np.cos(0.6), np.sin(0.45) * np.sin(0.6)
    gam = np.linalg.eig(_berreman_delta(np.conj(eps), Kx, Ky))[0]
    # oracle convention exp(+gam k0 z): kz = -i*gam, INTERNAL gauge roots
    kz = np.sort_complex(-1j * gam)
    kz_exact = _exact_kz_roots(np.conj(eps), Kx, Ky)
    assert np.max(np.abs(kz - kz_exact)) < 1e-9


def test_internal_poynting_theorem_oop_oblique():
    """Local Poynting inside OOP layers at oblique: -dSz/dz = k0 * dens,
    layer-by-layer, via the retained internals (C/k0 was 1.072 / 0.953
    pre-fix)."""
    L = [_tilted(35.0), 2.25 + 0.04j, _tilted(-20.0)]
    T = [400e-9, 150e-9, 300e-9]
    st = bb.BerremanStack(n_substrate=1.45)
    for Ti, Li in zip(T, L):
        st.add_layer(Ti, eps=Li)
    st.set_source(WL, angle=0.45, phi=0.6)
    st.solve(retain_internal=True)
    z_top = np.concatenate([[0.0], np.cumsum(T)])
    tensors = [np.asarray(L[0]), L[1] * np.eye(3), np.asarray(L[2])]
    for i in range(3):
        zs = z_top[i] + np.linspace(0.15, 0.85, 41) * T[i]
        f = st.internal_field(zs, incident=(1.0, 0.0))
        Sz = np.real(f["Ex"] * np.conj(f["Hy"]) - f["Ey"] * np.conj(f["Hx"]))
        E = np.stack([f["Ex"], f["Ey"], f["Ez"]], axis=-1)
        dens = np.imag(np.einsum("za,ab,zb->z", np.conj(E), tensors[i], E))
        C = float(np.mean(-np.gradient(Sz, zs) / dens))
        assert abs(C / K0 - 1.0) < 2e-3, f"layer {i}: C/k0 = {C / K0:.5f}"


def test_attribution_methods_agree_oop_oblique():
    """Density-based (RCWA) and flux-difference (Berreman) per-layer
    absorption attributions agree on identical OOP-oblique physics -- the
    Poynting-forced consistency that flagged the bug (pre-fix: 3.4e-3)."""
    from lumenairy.elements.rcwa import RCWAStack
    L = [_tilted(35.0), 2.25 + 0.04j, _tilted(-20.0)]
    T = [400e-9, 150e-9, 300e-9]
    rs = RCWAStack(1.0e-6, period_y=1.0e-6, n_substrate=1.45, n_orders=1,
                   n_orders_y=1)
    rs.add_layer(T[0], eps_tensor_cell=np.broadcast_to(L[0], (8, 8, 3, 3)).copy())
    rs.add_layer(T[1], eps=L[1])
    rs.add_layer(T[2], eps_tensor_cell=np.broadcast_to(L[2], (8, 8, 3, 3)).copy())
    rs.set_source(WL, theta=0.45, phi=0.6)
    A_r = rs.solve(retain_internal=True).layer_absorption(nx=8)
    bs = bb.BerremanStack(n_substrate=1.45)
    for Ti, Li in zip(T, L):
        bs.add_layer(Ti, eps=Li)
    bs.set_source(WL, angle=0.45, phi=0.6)
    bs.solve(retain_internal=True)
    A_b = bs.layer_absorption()
    assert np.abs(A_r - A_b.T).max() < 1e-4


# ===================================================================
# PMM 1-D generator dispersion gates (metric + covariant) -- the same
# closed-form anchor applied to the spectral-element generators.  For a
# UNIFORM medium every Galerkin coefficient mass is exactly a scalar multiple
# of the unit mass S0, so each generator is (up to the un-gated
# div-conforming Ez closure, which only touches unresolved modes) an exact
# matrix polynomial in the derivative operator Dop -- eig(L) must land on the
# exact det-condition kz roots at every alpha in the operator's OWN alpha
# spectrum {kx0 - i*d : d in eig(Dop)}.  These pin the off-plane cross-block
# factor-i signs of BOTH PMM generators: each in-file assignment is the
# UNIQUE combo of the 256 per-block {+-1, +-i} choices that closes
# (metric 1.9e-10 vs next-best 1.2e-2; covariant 4e-12 vs next-best 1.6e-2;
# a regressed sign lands >=1e-2 here and fails hard).  The tensors are fed
# to the internal builders directly and compared against the SAME tensor, so
# the public/internal conjugation convention cancels.
# ===================================================================
_T_GEN = np.array([                     # generic non-symmetric complex tensor:
    [4.0 + 0.05j, 0.30 + 0.02j, 0.70 - 0.10j],   # all four off-plane channels
    [0.20 - 0.03j, 5.0 + 0.02j, -0.55 + 0.08j],  # distinct + breaks the
    [0.45 + 0.06j, 0.65 - 0.04j, 6.0 + 0.03j]],  # normal-incidence mirror tie
    dtype=complex)


def _exact_kz_roots_scaled(eps, Kx):
    """Four exact kz/k0 roots at (possibly complex, possibly LARGE) Kx --
    sampling scaled to the root magnitude so the high-|alpha| spectral-element
    modes stay well-conditioned (the fixed-point sampler above is only for
    |Kx| ~ 1)."""
    e = np.asarray(eps, dtype=complex)
    s = 2.0 * max(1.0, abs(Kx), float(np.sqrt(np.max(np.abs(e)))))

    def detM(kz):
        k = np.array([Kx, 0.0, kz], dtype=complex)
        return np.linalg.det(np.outer(k, k) - (k @ k) * np.eye(3) + e)

    ws = np.array([0.0, 0.35, -0.45, 0.65, -0.8, 1.05, -1.2, 1.4, -1.6])
    coeffs = np.polyfit(s * ws, np.array([detM(z) for z in s * ws]), 4)
    return np.roots(coeffs)


def _pmm_uniform_mats(T, degree=12, nel=3):
    from lumenairy.elements.pmm import _core as pc
    t = pc._t3_slant(np.asarray(T, dtype=complex))
    return pc._build_nodal_metric(1.0e-6, 0.5e-6, t, t, degree, nel, nel, 1.0)


def _resolved_expected(pc, mats, T, k0, kx0, slant, sigma, cap=3.0):
    """Expected generator eigenvalues over RESOLVED alphas (|alpha|/k0 <= cap;
    the div-conforming Ez closure differs from the modal one only on
    unresolved modes): beta = kz*k0*cos(slant) + sigma*alpha*sin(slant)."""
    dj = np.linalg.eigvals(pc._safe_solve(mats["S0"], mats["C"]))
    cos, sin = np.cos(slant), np.sin(slant)
    out = []
    for d in dj:
        alpha = kx0 - 1j * d
        if abs(alpha) / k0 > cap:
            continue
        for kz in _exact_kz_roots_scaled(T, alpha / k0):
            out.append(kz * k0 * cos + sigma * alpha * sin)
    return np.array(out)


def _one_sided(eigs, expected, k0):
    a = np.asarray(eigs) / k0
    return max(np.min(np.abs(a - y)) for y in np.asarray(expected) / k0)


def test_pmm_metric_generator_matches_exact_dispersion():
    """eig(L) of the METRIC generator (the pmm_jones_1d vertical OOP path) on
    a uniform full-3x3 slab reproduces the exact det-condition roots at
    oblique incidence: expected mu = -i*k0*kz (L psi = mu psi, q = i mu/k0).
    Measured 1.9e-10; ANY single cross-block sign regression >= 1.2e-2."""
    from lumenairy.elements.pmm import _core as pc
    k0 = 2.0 * np.pi / 0.633e-6
    kx0 = 0.5 * k0
    mats = _pmm_uniform_mats(_T_GEN)
    L, _n = pc._build_generator_metric(mats, k0, 0.0, kx0)
    exp_mu = -1j * _resolved_expected(pc, mats, _T_GEN, k0, kx0, 0.0, 0.0)
    assert _one_sided(np.linalg.eigvals(L), exp_mu, k0) < 1e-8


@pytest.mark.parametrize("slant_deg", [30.0, 45.0])
def test_pmm_covariant_generator_matches_exact_dispersion(slant_deg):
    """eig(M) of the COVARIANT generator on a uniform full-3x3 slab
    reproduces the exact roots through the oblique-frame eigenvalue map
    beta = kz*k0*cos(phi) + alpha*sin(phi) (calibrated on the validated
    in-plane path; the docstring's `beta*sec(phi)` read-out is the internal
    frame-phase device, not the lab kz).  Both Ez closures: the modal
    (divconf=False) generator is an exact Dop-polynomial (full-spectrum
    4e-12); the production div-conforming closure matches on resolved alphas
    (2e-10).  ANY single cross-block sign regression >= 1.6e-2."""
    from lumenairy.elements.pmm import _core as pc
    k0 = 2.0 * np.pi / 0.633e-6
    kx0 = 0.5 * k0
    phi = np.deg2rad(slant_deg)
    mats = _pmm_uniform_mats(_T_GEN)
    exp_b = _resolved_expected(pc, mats, _T_GEN, k0, kx0, phi, 1.0)
    for divconf in (False, True):
        M, _n = pc._cov_generator_4n(mats, k0, phi, kx0, divconf=divconf)
        assert _one_sided(np.linalg.eigvals(M), exp_b, k0) < 1e-8, (
            f"divconf={divconf}")
