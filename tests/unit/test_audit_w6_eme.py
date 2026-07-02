"""Wave-6 audit fixes, EME cluster (P3-17 .. P3-23, AUDIT_V5_17_0_2026_07_01_DEEP).

Discriminating regression tests for:
  P3-17  scalar strip solver / ref_2d_modes no longer silently discard Im(eps)
  P3-18  layer_modes skips a scan sample landing exactly on a strip band edge
  P3-19  strip heights must sum to Ly (scalar + vector layer finders)
  P3-20  _global_lateral_nullspace KEPT two-call (accuracy-load-bearing; the
         test below pins the sigma/null-vector contract either way)
  P3-21  ONE strips_to_eps_xy implementation (scalar module delegates)
  P3-22  magnetic (eps, h, mu) 3-tuple strips accepted by mode_field_vec /
         strips_to_eps_xy / layer_vector_modes(verify=True), mu-consistent oracle
  P3-23  diffraction_eme returns the bare mode_match dict (qz2 folded in)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pytest

from lumenairy.elements.eme import eme_2d, eme_2d_vector, eme_diffraction
from lumenairy.elements.eme import strips_to_eps_xy as pkg_strips_to_eps_xy

PI = np.pi


# --------------------------------------------------------------------------- #
#  P3-17: lossy eps is not silently truncated                                  #
# --------------------------------------------------------------------------- #
def test_strip_x_modes_lossy_kx0_zero_keeps_imag():
    """kx0=0 + Im(eps)!=0 used to hit eigh (LAPACK drops the imaginary
    diagonal); now it routes to the general eig path like kx0!=0."""
    k0 = 2 * PI
    eps = np.full(16, 2.0 + 0.5j)
    lam0, _ = eme_2d.strip_x_modes(eps, 1.0, 16, k0, kx0=0.0)
    lam1, _ = eme_2d.strip_x_modes(eps, 1.0, 16, k0, kx0=0.1)
    want = 0.5 * k0 ** 2                    # Im(lam) = Im(eps) * k0^2 (uniform)
    assert np.max(np.abs(np.asarray(lam0).imag)) == pytest.approx(want, rel=1e-12)
    assert np.max(np.abs(np.asarray(lam1).imag)) == pytest.approx(want, rel=1e-12)


def test_strip_x_modes_lossless_path_unchanged():
    """Real eps at kx0=0 still takes the Hermitian eigh branch (real lam)."""
    lam, Phi = eme_2d.strip_x_modes(np.full(16, 2.0), 1.0, 16, 2 * PI, kx0=0.0)
    assert not np.iscomplexobj(lam)
    assert lam.max() == pytest.approx(2.0 * (2 * PI) ** 2, rel=1e-12)


def test_ref_2d_modes_lossy_warns_and_return_complex():
    eps = np.full((6, 6), 2.0 + 0.5j)
    k0 = 2 * PI
    with pytest.warns(UserWarning, match="DISCARDED"):
        w_real = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, k0)
    assert not np.iscomplexobj(w_real)
    w_c = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, k0, return_complex=True)
    assert np.iscomplexobj(w_c)
    assert np.max(np.abs(w_c.imag)) == pytest.approx(0.5 * k0 ** 2, rel=1e-9)


def test_ref_2d_modes_lossless_return_complex_matches_legacy():
    eps = np.full((6, 6), 2.0)
    wa = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, 2 * PI)
    wb = eme_2d.ref_2d_modes(eps, 1.0, 1.0, 6, 6, 2 * PI, return_complex=True)
    assert np.allclose(wa, wb.real, rtol=0, atol=1e-10)
    assert np.max(np.abs(wb.imag)) < 1e-10


# --------------------------------------------------------------------------- #
#  P3-18: a band-edge scan sample no longer raises LinAlgError                 #
# --------------------------------------------------------------------------- #
def test_layer_modes_band_edge_window_no_crash():
    """hi = max(eps)*k0^2 exactly (an x-uniform strip's discrete band top) used
    to raise LinAlgError('Singular matrix') from the singular _interface solve;
    the band-edge sample is now skipped (not a mode candidate)."""
    k0 = 2 * PI
    eps = np.full(12, 2.0)
    strips = [(eps, 0.5), (eps, 0.5)]
    qz2 = eme_2d.layer_modes(strips, 1.0, 12, 1.0, k0, (1.0, 2.0 * k0 ** 2))
    # the interior mode qz2 = 2 k0^2 - (2 pi)^2 (kx=0, ky=2pi branch) survives
    assert np.any(np.abs(qz2 - (2.0 * k0 ** 2 - (2 * PI) ** 2)) < 1e-3)


# --------------------------------------------------------------------------- #
#  P3-19: strip heights must sum to Ly                                          #
# --------------------------------------------------------------------------- #
def test_layer_modes_rejects_bad_heights():
    eps = np.full(12, 2.0)
    with pytest.raises(ValueError, match="heights sum to"):
        eme_2d.layer_modes([(eps, 0.3), (eps, 0.3)], 1.0, 12, 1.0, 2 * PI,
                           (1.0, 70.0), ky0=PI)


def test_layer_vector_modes_rejects_bad_heights():
    eps = np.full(8, 2.0)
    with pytest.raises(ValueError, match="heights sum to"):
        eme_2d_vector.layer_vector_modes([(eps, 0.3), (eps, 0.3)], 1.0, 8, 1.0,
                                         8.0, (130.0, 256.0), ky0=PI, n_scan=10)


def test_layer_modes_consistent_heights_still_pass():
    """The analytic uniform-cell mode at ky0=pi is unchanged by the new guard."""
    k0 = 2 * PI
    eps = np.full(12, 2.0)
    qz2 = eme_2d.layer_modes([(eps, 0.5), (eps, 0.5)], 1.0, 12, 1.0, k0,
                             (1.0, 70.0), ky0=PI)
    assert qz2[0] == pytest.approx(2.0 * k0 ** 2 - PI ** 2, abs=1e-3)


# --------------------------------------------------------------------------- #
#  P3-20: nullspace sigma/null-vector contract (the two-SVD structure was       #
#  investigated and KEPT -- see the NOTE in _global_lateral_nullspace)          #
# --------------------------------------------------------------------------- #
def test_global_lateral_nullspace_sigma_and_null_vector():
    k0 = 8.0
    eps = np.full(12, 2.0)
    sm = [(eme_2d.strip_x_modes(eps, 1.0, 12, k0), 0.5)] * 2
    qz2 = 2.0 * k0 ** 2 - PI ** 2                 # true mode at ky0 = pi
    t = np.exp(1j * PI * 1.0)
    c, sigma = eme_2d._global_lateral_nullspace(sm, qz2, t)
    assert sigma < 1e-8                           # a true mode -> singular G
    assert np.linalg.norm(c) == pytest.approx(1.0, rel=1e-12)
    psi, sig2 = eme_2d.mode_field(sm, qz2, PI, 1.0, 32)
    assert sig2 == sigma
    # uniform cell + kx=0 mode -> |psi| is x-uniform (a broken null vector mixes
    # kx != 0 strip modes and destroys this)
    mag = np.abs(psi)
    assert np.max(np.ptp(mag, axis=0)) < 1e-8 * np.max(mag)


def test_nullspace_sigma_accurate_on_ill_scaled_G():
    """Regression for the REJECTED P3-20 single-SVD variant: on the structured
    reference cell the unequilibrated G carries exp(+|ky|h) ~ 1e12 columns and
    the with-vectors gesdd SVD reports s[-1] ~14x off (4.97e-2 > the 1e-2 mode
    diagnostic); sigma must stay values-only (svdvals) accurate."""
    k0, Nx = 8.0, 28
    xg = (np.arange(Nx) + 0.5) / Nx
    grat = np.where(xg < 0.5, 4.0, 1.0).astype(float)
    strips = [(grat, 0.5), (np.full(Nx, 2.0), 0.5)]
    sm = [(eme_2d.strip_x_modes(e, 1.0, Nx, k0, 0.0), h) for e, h in strips]
    qs = eme_2d.layer_modes(strips, 1.0, Nx, 1.0, k0, (120, 256), ky0=PI)[:3]
    assert len(qs) == 3
    for q in qs:
        _, sig = eme_2d._global_lateral_nullspace(sm, q, np.exp(1j * PI))
        assert sig < 1e-2                         # the documented mode residual


# --------------------------------------------------------------------------- #
#  P3-21: one rasterizer -- the scalar module delegates to the general one      #
# --------------------------------------------------------------------------- #
def test_strips_to_eps_xy_single_implementation():
    eps = np.full(8, 4.0)
    strips = [(eps, 0.4), (np.full(8, 1.0), 0.6)]
    a = eme_2d.strips_to_eps_xy(strips, 1.0, 8, 1.0, 10)
    b = eme_2d_vector.strips_to_eps_xy(strips, 1.0, 8, 1.0, 10)
    c = pkg_strips_to_eps_xy(strips, 1.0, 8, 1.0, 10)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    # the scalar-module name now handles tensor strips too (used to ValueError)
    t = np.tile(np.eye(3) * 2.0, (8, 1, 1))
    out = eme_2d.strips_to_eps_xy([(t, 0.5), (t, 0.5)], 1.0, 8, 1.0, 10)
    assert out.shape == (8, 10, 3, 3)


# --------------------------------------------------------------------------- #
#  P3-22: magnetic (eps, h, mu) 3-tuple strips                                  #
# --------------------------------------------------------------------------- #
def test_magnetic_3tuple_strips_no_crash():
    eps = np.full(8, 2.0)
    mag = [(eps, 0.5, 1.0), (eps, 0.5)]           # used to ValueError (unpack)
    grid = eme_2d_vector.strips_to_eps_xy(mag, 1.0, 8, 1.0, 10)
    assert grid.shape == (8, 10)
    Ex, Ez, sigma = eme_2d_vector.mode_field_vec(mag, 1.0, 8, 1.0, 8.0, 200.0,
                                                 PI, 10)
    assert Ex.shape == (8, 10) and Ez.shape == (8, 10)


def test_magnetic_3tuple_verify_matches_2tuple():
    """verify=True used to crash on 3-tuple strips; with mu=1 it must equal the
    2-tuple result exactly."""
    grat = np.where((np.arange(16) + 0.5) / 16 < 0.5, 4.0, 1.0)
    uni = np.full(16, 2.0)
    kw = dict(ky0=PI, verify=True, n_scan=30)
    q3 = eme_2d_vector.layer_vector_modes(
        [(grat, 0.5, 1.0), (uni, 0.5, 1.0)], 1.0, 16, 1.0, 8.0, (140.0, 160.0),
        **kw)
    q2 = eme_2d_vector.layer_vector_modes(
        [(grat, 0.5), (uni, 0.5)], 1.0, 16, 1.0, 8.0, (140.0, 160.0), **kw)
    assert np.array_equal(q3, q2)
    assert len(q2) > 0                            # verify kept the genuine modes


def test_fd_eig_dist_uses_magnetic_oracle():
    """The verify oracle must include mu: a uniform magnetic cell (eps*mu = 2)
    has its analytic mode ACCEPTED, while a mu-blind oracle (eps alone) would
    put the nearest FD eigenvalue ~ (2 - 2/1.5) k0^2 = 21 away."""
    k0, mu = 8.0, 1.5
    eps = np.full(8, 2.0 / mu)
    qz2_true = 2.0 * k0 ** 2 - PI ** 2            # eps*mu*k0^2 - ky^2 (ky0=pi)
    d_mag = eme_2d_vector._fd_eig_dist(
        [(eps, 0.5, mu), (eps, 0.5, mu)], 1.0, 8, 1.0, k0, 0.0, PI, qz2_true, 48)
    assert d_mag < 0.5                            # mu-consistent oracle: nearby
    d_blind = eme_2d_vector._fd_eig_dist(
        [(eps, 0.5), (eps, 0.5)], 1.0, 8, 1.0, k0, 0.0, PI, qz2_true, 48)
    assert d_blind > 5.0                          # the non-magnetic cell is far


# --------------------------------------------------------------------------- #
#  P3-23: diffraction_eme returns the bare dict (qz2 folded in)                 #
# --------------------------------------------------------------------------- #
def test_diffraction_eme_returns_dict_like_fd():
    k0 = 2 * PI / 0.8
    eps = np.full(12, 2.0)
    hi = 2.0 * k0 ** 2
    out = eme_diffraction.diffraction_eme(
        [(eps, 0.4), (eps, 0.4)], 0.8, 12, 0.8, k0, 1.0, 1.0, 0.3, 1, 1,
        qz2_window=(-8.0 * hi, hi), n_scan=300)
    assert isinstance(out, dict)                  # was an undocumented tuple
    for key in ("orders", "r", "t", "R", "T", "energy", "qz2"):
        assert key in out
    assert len(out["qz2"]) == 9                   # K = N_pw retained eigenvalues
