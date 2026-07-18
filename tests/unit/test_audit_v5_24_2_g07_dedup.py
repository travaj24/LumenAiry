"""Regression tests for the v5.24.2 exhaustive-audit G07 dedup refactors.

Group G07-dedup-refactors consolidated several copy-pasted blocks into single
shared helpers.  Every dedup here is a BYTE-IDENTICAL refactor: the shared
helper must reproduce the former inline copy's output to machine precision.
Each test therefore carries an INDEPENDENT oracle -- the former formula written
out by hand -- and asserts exact equality (``np.array_equal`` or ``<= 1e-15``),
so a future drift in the helper (the exact failure mode that bred the six-copy
factor-i bug, audit S1-8) fails loudly.

Findings covered:
  * S1-8  _forward_branch_flip          (elements/pmm/_core.py)
  * S1-9  _project_efficiency           (elements/rcwa/_core.py)
  * S1-10 _pmm2d_project_orders/_order_kz (elements/pmm/twod_staggered.py)
  * S2-14 _freespace_tensor_moebius_np  (propagators/gbd.py)
  * S2-14 _tukey_taper                  (elements/lenses_maslov.py)
  * S2-14 _maslov_newton_saddle_cpu     (elements/lenses_maslov.py)
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# S1-8: PMM forward-mode branch selector (the factor-i multi-copy pattern)
# --------------------------------------------------------------------------- #
def _former_branch_flip_np(q):
    """The exact former inline copy (1e-8 scalar-vertical selector)."""
    tol = 1e-8 * max(float(np.max(np.abs(q))), 1.0)
    flip = (q.imag < -tol) | ((np.abs(q.imag) <= tol) & (q.real < 0.0))
    return np.where(flip, -q, q)


def test_s1_8_forward_branch_flip_matches_former_numpy():
    from lumenairy.elements.pmm._core import _forward_branch_flip

    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        assert np.array_equal(_forward_branch_flip(q),
                              _former_branch_flip_np(q))


def test_s1_8_forward_branch_flip_crafted_band_cases():
    """Independent hand-computed expectations exercising every branch:
    real-positive keep, real-negative flip, backward-evanescent flip,
    forward-evanescent keep, and the near-real tie-break band."""
    from lumenairy.elements.pmm._core import _forward_branch_flip

    # max|q| = 4 -> tol = 4e-8, so 1e-9 sits inside the near-real band.
    q = np.array([3 + 0j, -2 + 0j, -4j, 4j, 0.5 + 1e-9j, -0.5 + 1e-9j])
    expected = np.array([3 + 0j, 2 + 0j, 4j, 4j, 0.5 + 1e-9j, 0.5 - 1e-9j])
    assert np.array_equal(_forward_branch_flip(q), expected)


def test_s1_8_forward_branch_flip_jax_parity():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    jax.config.update("jax_enable_x64", True)
    from lumenairy.elements.pmm._core import _forward_branch_flip

    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(1, 10))
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        out = np.asarray(_forward_branch_flip(jnp.asarray(q), jnp))
        assert np.allclose(out, _former_branch_flip_np(q), rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- #
# S1-9: RCWA Poynting-flux efficiency projection
# --------------------------------------------------------------------------- #
def _former_project_efficiency(kz_ref, kz_trn, kz_inc, rx, ry, rz,
                               tx, ty, tz, einc_sq):
    R = np.real(kz_ref / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2
                                    + np.abs(rz) ** 2) / einc_sq
    T = np.real(kz_trn / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                    + np.abs(tz) ** 2) / einc_sq
    R = np.where(np.real(kz_ref) > 0, np.real(R), 0.0)
    T = np.where(np.real(kz_trn) > 0, np.real(T), 0.0)
    return R, T


def test_s1_9_project_efficiency_matches_former():
    from lumenairy.elements.rcwa._core import _project_efficiency

    rng = np.random.default_rng(2)

    def cplx(n):
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    for _ in range(300):
        n = int(rng.integers(1, 9))
        args = (cplx(n), cplx(n), abs(rng.standard_normal()) + 0.2,
                cplx(n), cplx(n), cplx(n), cplx(n), cplx(n), cplx(n),
                abs(rng.standard_normal()) + 0.1)
        R, T = _project_efficiency(np, *args)
        Ro, To = _former_project_efficiency(*args)
        assert np.array_equal(R, Ro) and np.array_equal(T, To)


def test_s1_9_rcwa_entry_points_still_energy_conserve():
    """End-to-end: routing every RCWA site through the helper preserves the
    lossless closure sum(R)+sum(T)=1 (independent physical oracle)."""
    from lumenairy.elements.rcwa import rcwa_efficiency_1d

    for pol in ("te", "tm"):
        o, R, T = rcwa_efficiency_1d(
            period=1.2, n_ridge=2.0, n_groove=1.0, n_substrate=1.0,
            n_superstrate=1.0, depth=0.4, duty_cycle=0.5, wavelength=0.633,
            angle=0.2, polarization=pol, n_orders=7)
        assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# S1-10: PMM-2D far-field projection + kz block
# --------------------------------------------------------------------------- #
def test_s1_10_pmm2d_project_orders_matches_former():
    from lumenairy.elements.pmm.twod_staggered import _pmm2d_project_orders

    rng = np.random.default_rng(3)
    for _ in range(100):
        qq = int(rng.integers(1, 8))
        nfo = int(rng.integers(1, 6))
        ncols = int(rng.integers(1, 4))
        P1 = rng.standard_normal((nfo, qq)) + 1j * rng.standard_normal((nfo, qq))
        P2 = rng.standard_normal((nfo, qq)) + 1j * rng.standard_normal((nfo, qq))
        W = (rng.standard_normal((2 * qq, ncols))
             + 1j * rng.standard_normal((2 * qq, ncols)))
        # former closure
        top = P1 @ W[:qq, :]
        bot = P2 @ W[qq:, :]
        former = np.concatenate([top, bot], axis=0)
        assert np.array_equal(_pmm2d_project_orders(P1, P2, W, qq), former)


def test_s1_10_pmm2d_order_kz_matches_former():
    from lumenairy.elements.pmm.twod_staggered import (
        _kz_forward2,
        _pmm2d_order_kz,
    )

    rng = np.random.default_rng(4)
    for _ in range(100):
        n = int(rng.integers(1, 8))
        eps_sup = abs(rng.standard_normal()) + 1.0
        eps_sub = abs(rng.standard_normal()) + 1.0
        kxv = rng.standard_normal(n)
        kyv = rng.standard_normal(n)
        kx0 = float(rng.standard_normal())
        ky0 = float(rng.standard_normal())
        # former inline block
        kz_ref = _kz_forward2(eps_sup, kxv, kyv)
        kz_trn = _kz_forward2(eps_sub, kxv, kyv)
        kz_inc = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
        safe_r = np.where(np.abs(kz_ref) < 1e-12, 1.0, kz_ref)
        safe_t = np.where(np.abs(kz_trn) < 1e-12, 1.0, kz_trn)
        got = _pmm2d_order_kz(eps_sup, eps_sub, kxv, kyv, kx0, ky0)
        assert np.array_equal(got[0], kz_ref)
        assert np.array_equal(got[1], kz_trn)
        assert got[2] == kz_inc
        assert np.array_equal(got[3], safe_r)
        assert np.array_equal(got[4], safe_t)


# --------------------------------------------------------------------------- #
# S2-14: GBD free-space tensor Moebius + amplitude
# --------------------------------------------------------------------------- #
def test_s2_14_gbd_freespace_moebius_matches_former():
    from lumenairy.propagators.gbd import (
        _eigvals2x2,
        _freespace_tensor_moebius_np,
        _guard_tensor_freespace_branch,
        _inv2x2,
    )

    def former(Q, amp, t, k0):
        I2 = np.eye(2)[None, :, :]
        lam = _eigvals2x2(Q, np)
        _guard_tensor_freespace_branch(lam, t, np)
        amp = amp * np.prod(1.0 / np.sqrt(1.0 + t[:, None] * lam), axis=1)
        amp = amp * np.exp(1j * k0 * t)
        Q = Q @ _inv2x2(I2 + t[:, None, None] * Q, np)
        Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
        return Q, amp

    rng = np.random.default_rng(5)
    import warnings
    for _ in range(80):
        n = int(rng.integers(2, 20))
        A = rng.standard_normal((n, 2, 2)) + 1j * rng.standard_normal((n, 2, 2))
        Q = A + np.transpose(A, (0, 2, 1))          # symmetric
        # physical beam: Im(lambda) < 0
        Q = Q - 1j * (np.abs(rng.standard_normal((n, 1, 1))) + 0.1) * np.eye(2)[None]
        amp = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        t = rng.standard_normal(n)
        k0 = 2 * np.pi / 0.633
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Qa, aa = _freespace_tensor_moebius_np(Q.copy(), amp.copy(),
                                                  t.copy(), k0)
            Qb, ab = former(Q.copy(), amp.copy(), t.copy(), k0)
        assert np.array_equal(Qa, Qb) and np.array_equal(aa, ab)


# --------------------------------------------------------------------------- #
# S2-14: Maslov Tukey window (defined three times)
# --------------------------------------------------------------------------- #
def test_s2_14_tukey_taper_matches_both_former_defs():
    from lumenairy.elements.lenses_maslov import _tukey_taper

    def former_tukey_n(n, alpha=0.2):           # the tukey(n) helper
        u = np.linspace(-1, 1, n)
        abs_u = np.abs(u)
        w = np.ones_like(u)
        ts = 1.0 - alpha
        tmask = abs_u > ts
        w[tmask] = 0.5 * (1 + np.cos(np.pi * (abs_u[tmask] - ts) / alpha))
        return w

    def former_tuk_u(u, alpha=0.2):             # the _tuk(u) closure
        au = np.abs(u)
        w = np.ones_like(u)
        m = au > 1.0 - alpha
        w[m] = 0.5 * (1.0 + np.cos(np.pi * (au[m] - (1.0 - alpha)) / alpha))
        return w

    for n in (2, 4, 7, 16, 33, 64, 129):
        assert np.array_equal(_tukey_taper(np.linspace(-1, 1, n)),
                              former_tukey_n(n))
    rng = np.random.default_rng(6)
    for _ in range(200):
        u = rng.uniform(-1.2, 1.2, size=int(rng.integers(1, 50)))
        for alpha in (0.1, 0.2, 0.3):
            assert np.array_equal(_tukey_taper(u, alpha),
                                  former_tuk_u(u, alpha))


# --------------------------------------------------------------------------- #
# S2-14: Maslov CPU active-subset Newton saddle solver
# --------------------------------------------------------------------------- #
def _former_cpu_newton(opd_eval, coef_opd, u_s2x_flat, u_s2y_flat, inbox_flat,
                       newton_iter, newton_tol, lin_v3, lin_v4):
    N_px = u_s2x_flat.shape[0]
    u_v2x = np.zeros(N_px, dtype=np.float64)
    u_v2y = np.zeros(N_px, dtype=np.float64)
    converged = np.zeros(N_px, dtype=bool)
    converged[~inbox_flat] = True
    for _it in range(newton_iter):
        if converged.all():
            break
        active = ~converged
        u1 = u_s2x_flat[active]
        u2 = u_s2y_flat[active]
        u3 = u_v2x[active]
        u4 = u_v2y[active]
        _, g3, g4, H33, H34, H44 = opd_eval(coef_opd, u1, u2, u3, u4)
        g3 = g3 + lin_v3
        g4 = g4 + lin_v4
        det_H = H33 * H44 - H34 * H34
        det_safe = np.where(np.abs(det_H) < 1e-30,
                            np.where(det_H < 0, -1e-30, 1e-30), det_H)
        dv3 = -(H44 * g3 - H34 * g4) / det_safe
        dv4 = -(-H34 * g3 + H33 * g4) / det_safe
        step_size = np.sqrt(dv3 ** 2 + dv4 ** 2)
        damp = np.where(step_size > 0.5,
                        0.5 / np.maximum(step_size, 1e-30), 1.0)
        dv3 = dv3 * damp
        dv4 = dv4 * damp
        u_v2x[active] = np.clip(u_v2x[active] + dv3, -1.0, 1.0)
        u_v2y[active] = np.clip(u_v2y[active] + dv4, -1.0, 1.0)
        grad_mag = np.sqrt(g3 ** 2 + g4 ** 2)
        newly = np.zeros(N_px, dtype=bool)
        newly[active] = grad_mag < newton_tol
        converged |= newly
    return u_v2x, u_v2y, converged


def test_s2_14_maslov_cpu_newton_matches_former():
    from lumenairy.elements.lenses_maslov import _maslov_newton_saddle_cpu

    def opd_eval(coef, u1, u2, u3, u4):
        f = np.sin(u1 + u3) + np.cos(u2 + u4)
        g3 = np.cos(u1 + u3) * 0.7 + 0.3 * u3 - 0.1
        g4 = -np.sin(u2 + u4) * 0.7 + 0.3 * u4 + 0.05
        H33 = -0.7 * np.sin(u1 + u3) + 0.3
        H34 = 0.02 * np.ones_like(u3)
        H44 = -0.7 * np.cos(u2 + u4) + 0.3
        return f, g3, g4, H33, H34, H44

    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(5, 40))
        u_s2x = rng.uniform(-1, 1, n)
        u_s2y = rng.uniform(-1, 1, n)
        inbox = rng.random(n) > 0.2
        lin_v3 = float(rng.standard_normal())
        lin_v4 = float(rng.standard_normal())
        a = _maslov_newton_saddle_cpu(opd_eval, None, u_s2x, u_s2y, inbox,
                                      12, 1e-10, lin_v3, lin_v4)
        b = _former_cpu_newton(opd_eval, None, u_s2x, u_s2y, inbox,
                               12, 1e-10, lin_v3, lin_v4)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert np.array_equal(a[2], b[2])
