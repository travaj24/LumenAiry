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
