"""Phase P6 (niche N7) -- carrier-referenced ASM: astigmatic carriers +
mid-chain apertures.

Task 1 (astigmatic): a separable Sziklas-Siegman transform with per-axis
magnification ``m_x``, ``m_y`` and PER-AXIS focus crossing (the x and y foci
fall at different z).  ``carrier=(R_x, R_y)`` alongside the scalar form.

Task 2 (apertures): a hard aperture on the co-moving envelope grid clips the
envelope in place, REMOVES the clipped power (never renormalises), and leaves
the wavefront carrier ``R`` unchanged by default (opt-in re-fit / re-reference).

Independent oracles (all lumenairy-free):

* **Exact band-limited ASM on a FINE grid** (N=8192): the ground truth the
  task specifies for the astigmatic line foci / circle-of-least-confusion and
  for the apertured downstream field.  The modest carrier grid (N<=2048) must
  match it to < 1 % windowed r2m while using orders of magnitude less memory.
* **Analytic astigmatic Gaussian ABCD (q-parameter)** -- closed-form, grid-
  free -- locates the two line foci and the circle of least confusion.
* **Energy conservation** -- the co-moving area element ``dx_out*dy_out``.

The astigmatic transform reduces to the scalar path when ``R_x == R_y``:
``carrier=(R, R)`` routes to the scalar path BYTE-IDENTICAL (pinned with
``array_equal``), and the raw separable machinery reproduces it to ~1e-11.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy import (
    carrier_referenced_aperture,
    carrier_referenced_envelope,
    carrier_referenced_fit_radius,
    carrier_referenced_reconstruct,
    propagate_carrier_referenced,
)
from lumenairy.propagators.carrier import _propagate_carrier_astigmatic
from lumenairy.propagators.propagation import angular_spectrum_propagate

WL = 1.31e-6
K = 2.0 * np.pi / WL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gauss_env(N, dx, wx, dy=None, wy=None):
    """Real elliptical Gaussian amplitude envelope, axis 1 == x, axis 0 == y."""
    dy = dx if dy is None else dy
    wy = wx if wy is None else wy
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')          # axis 1 == x, axis 0 == y
    return np.exp(-(X ** 2) / wx ** 2 - (Y ** 2) / wy ** 2).astype(np.complex128)


def _q_at(R, z, w_in):
    q0 = 1.0 / (1.0 / R - 1j * WL / (np.pi * w_in ** 2))
    return q0 + z


def _w_of_q(q):
    return np.sqrt(-WL / (np.pi * (1.0 / q).imag))


def _r2m_axis(I, dx, dy, win_x, win_y):
    """Per-axis windowed second-moment widths (r2m_x from the x-marginal,
    r2m_y from the y-marginal) in a rectangular window, centred on the grid."""
    Ny, Nx = I.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    mask = (np.abs(X) <= win_x) & (np.abs(Y) <= win_y)
    Iw = np.where(mask, I, 0.0)
    tot = Iw.sum()
    r2m_x = np.sqrt((Iw * X * X).sum() / tot)
    r2m_y = np.sqrt((Iw * Y * Y).sum() / tot)
    return r2m_x, r2m_y


def _r2m_radial(I, dx, dy, win):
    Ny, Nx = I.shape
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    r = np.sqrt(X * X + Y * Y)
    Iw = np.where(r <= win, I, 0.0)
    tot = Iw.sum()
    return np.sqrt((Iw * r * r).sum() / tot)


# ===========================================================================
# Task 1 -- astigmatic carrier
# ===========================================================================
def test_astig_routing_equal_radii_is_byte_identical():
    """CRITICAL regression: an isotropic input through the astigmatic API
    (``carrier=(R, R)``) MUST reproduce the scalar path byte-identical -- the
    equal-radii tuple routes to the untouched scalar path."""
    N, dx = 512, 4e-6
    env = _gauss_env(N, dx, 40e-6)
    for R, z in [(30e-3, 40e-3), (-80e-3, 20e-3), (np.inf, 5e-3)]:
        out_scalar = propagate_carrier_referenced(env, R, z, WL, dx)
        out_tuple = propagate_carrier_referenced(env, (R, R), z, WL, dx)
        assert np.array_equal(np.asarray(out_tuple.env),
                              np.asarray(out_scalar.env)), f"R={R} z={z}"
        assert out_tuple.R == out_scalar.R
        assert out_tuple.dx == out_scalar.dx


def test_astig_machinery_reduces_to_scalar():
    """The raw separable astigmatic machinery (NOT short-circuited) with
    ``R_x == R_y`` reproduces the scalar path to ~1e-11 -- proving the
    coordinate-wise transform reduces correctly, not merely that equal radii
    are routed away."""
    N, dx = 512, 4e-6
    env = _gauss_env(N, dx, 40e-6)
    R, z = 30e-3, 40e-3
    ref = propagate_carrier_referenced(env, R, z, WL, dx)
    out = _propagate_carrier_astigmatic(env, R, R, z, WL, dx, dx)
    scale = np.max(np.abs(np.asarray(ref.env)))
    assert np.max(np.abs(np.asarray(out.env) - np.asarray(ref.env))) / scale < 1e-9
    assert out.R == (R + z, R + z)
    assert out.dx[0] == pytest.approx(ref.dx) and out.dx[1] == pytest.approx(ref.dx)


@pytest.mark.slow
def test_astig_matches_fine_asm_at_line_foci_and_lc():
    """The core Task-1 claim: a cylindrical-lens astigmatic Gaussian (distinct
    R_x, R_y) transported on a MODEST N=2048 grid matches the exact fine-grid
    ASM (N=8192) to < 1 % windowed r2m on BOTH axes at the x line focus, the y
    line focus, AND the circle-of-least-confusion plane -- exercising the
    per-axis focus-crossing bridge (the two foci fall at different z)."""
    R_x, R_y, w_in = -20e-3, -30e-3, 0.5e-3

    # analytic astigmatic Gaussian: line foci (Re q == 0) and circle of least
    # confusion (w_x == w_y).
    z_fx = -_q_at(R_x, 0, w_in).real
    z_fy = -_q_at(R_y, 0, w_in).real
    zz = np.linspace(0.5 * min(z_fx, z_fy), 1.3 * max(z_fx, z_fy), 20001)
    wx = np.array([_w_of_q(_q_at(R_x, z, w_in)) for z in zz])
    wy = np.array([_w_of_q(_q_at(R_y, z, w_in)) for z in zz])
    z_lc = zz[int(np.argmin(np.abs(wx - wy)))]

    # fine-grid ASM ground truth (axis 1 == x -> R_x, axis 0 == y -> R_y)
    Nf, ext = 8192, 10e-3
    dxf = ext / Nf
    xf = (np.arange(Nf) - Nf / 2) * dxf
    Yf, Xf = np.meshgrid(xf, xf, indexing='ij')
    E0 = (np.exp(-(Xf ** 2 + Yf ** 2) / w_in ** 2)
          * np.exp(1j * K * Xf ** 2 / (2 * R_x))
          * np.exp(1j * K * Yf ** 2 / (2 * R_y))).astype(np.complex128)
    del Xf, Yf

    # modest carrier grid
    Nm = 2048
    dxm = (8 * w_in) / Nm
    env0 = _gauss_env(Nm, dxm, w_in)

    try:
        for label, zt in [('x-focus', z_fx), ('lc', z_lc), ('y-focus', z_fy)]:
            wx_t = _w_of_q(_q_at(R_x, zt, w_in))
            wy_t = _w_of_q(_q_at(R_y, zt, w_in))
            win_x, win_y = 3.0 * wx_t, 3.0 * wy_t

            Ef = angular_spectrum_propagate(E0, zt, WL, dxf)
            r2mx_f, r2my_f = _r2m_axis(np.abs(Ef) ** 2, dxf, dxf, win_x, win_y)
            del Ef

            env, R_out, d_out = propagate_carrier_referenced(
                env0, (R_x, R_y), zt, WL, dxm)
            dx_out, dy_out = d_out
            r2mx_c, r2my_c = _r2m_axis(
                np.abs(np.asarray(env)) ** 2, dx_out, dy_out, win_x, win_y)

            assert abs(r2mx_c - r2mx_f) / r2mx_f < 0.01, (
                f"{label}: r2m_x {r2mx_c*1e6:.3f} vs ASM {r2mx_f*1e6:.3f}")
            assert abs(r2my_c - r2my_f) / r2my_f < 0.01, (
                f"{label}: r2m_y {r2my_c*1e6:.3f} vs ASM {r2my_f*1e6:.3f}")
    finally:
        la.clear_asm_caches()


@pytest.mark.slow
def test_astig_collimated_axis_cylindrical():
    """A cylindrical carrier ``(inf, R_y)`` -- collimated in x, converging in y
    -- keeps the x pitch and x carrier unchanged (collimated), magnifies only
    y, and matches a fine-grid ASM to < 1 % on both axes."""
    R_y, w_in = -40e-3, 0.5e-3
    zt = 20e-3

    Nf, ext = 8192, 10e-3
    dxf = ext / Nf
    xf = (np.arange(Nf) - Nf / 2) * dxf
    Yf, Xf = np.meshgrid(xf, xf, indexing='ij')
    E0 = (np.exp(-(Xf ** 2 + Yf ** 2) / w_in ** 2)
          * np.exp(1j * K * Yf ** 2 / (2 * R_y))).astype(np.complex128)
    del Xf, Yf

    Nm = 2048
    dxm = (8 * w_in) / Nm
    env0 = _gauss_env(Nm, dxm, w_in)
    try:
        Ef = angular_spectrum_propagate(E0, zt, WL, dxf)
        wx_t = _w_of_q(_q_at(np.inf, zt, w_in))
        wy_t = _w_of_q(_q_at(R_y, zt, w_in))
        win_x, win_y = 3.0 * wx_t, 3.0 * wy_t
        r2mx_f, r2my_f = _r2m_axis(np.abs(Ef) ** 2, dxf, dxf, win_x, win_y)
        del Ef

        env, R_out, d_out = propagate_carrier_referenced(
            env0, (np.inf, R_y), zt, WL, dxm)
        # x is collimated: pitch and carrier unchanged; y magnified.
        assert R_out[0] == np.inf
        assert R_out[1] == pytest.approx(R_y + zt, rel=1e-12)
        assert d_out[0] == pytest.approx(dxm, rel=1e-12)
        assert d_out[1] == pytest.approx(dxm * (R_y + zt) / R_y, rel=1e-9)

        r2mx_c, r2my_c = _r2m_axis(
            np.abs(np.asarray(env)) ** 2, d_out[0], d_out[1], win_x, win_y)
        assert abs(r2mx_c - r2mx_f) / r2mx_f < 0.01
        assert abs(r2my_c - r2my_f) / r2my_f < 0.01
    finally:
        la.clear_asm_caches()


def test_astig_power_conserved_fast_step():
    """Total power (|env|^2 over the anisotropic co-moving area element
    ``dx_out*dy_out``) is invariant across a non-crossing astigmatic step."""
    N, dx = 768, 6e-6
    env = _gauss_env(N, dx, 80e-6, wy=110e-6)
    p_in = (np.abs(env) ** 2).sum() * dx * dx
    env_out, R_out, d_out = propagate_carrier_referenced(
        env, (30e-3, 45e-3), 12e-3, WL, dx)
    dx_out, dy_out = d_out
    p_out = (np.abs(np.asarray(env_out)) ** 2).sum() * dx_out * dy_out
    assert p_out == pytest.approx(p_in, rel=2e-4)
    assert R_out == (30e-3 + 12e-3, 45e-3 + 12e-3)


def test_astig_reconstruct_envelope_roundtrip():
    """The astigmatic carrier reconstruct/extract are exact inverses, and the
    reconstructed full field carries exactly the separable +k carrier."""
    N = 256
    dx, dy = 5e-6, 6e-6
    env = _gauss_env(N, dx, 60e-6, dy=dy, wy=70e-6) * np.exp(0.3j)
    Rx, Ry = 42e-3, 55e-3
    full = carrier_referenced_reconstruct(env, (Rx, Ry), WL, dx, dy)
    back = carrier_referenced_envelope(full, (Rx, Ry), WL, dx, dy)
    assert np.max(np.abs(back - env)) < 1e-13
    # explicit separable carrier
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    Y, X = np.meshgrid(y, x, indexing='ij')
    ref = env * np.exp(1j * K * (X ** 2 / (2 * Rx) + Y ** 2 / (2 * Ry)))
    assert np.max(np.abs(full - ref)) < 1e-12
    # a collimated axis drops out
    full_c = carrier_referenced_reconstruct(env, (np.inf, Ry), WL, dx, dy)
    ref_c = env * np.exp(1j * K * (Y ** 2 / (2 * Ry)))
    assert np.max(np.abs(full_c - ref_c)) < 1e-12


def test_astig_zero_axis_radius_raises():
    N, dx = 128, 5e-6
    env = _gauss_env(N, dx, 40e-6)
    with pytest.raises(ValueError):
        propagate_carrier_referenced(env, (0.0, 30e-3), 5e-3, WL, dx)
    with pytest.raises(ValueError):
        propagate_carrier_referenced(env, (30e-3, 0.0), 5e-3, WL, dx)


def test_astig_bad_tuple_length_raises():
    N, dx = 128, 5e-6
    env = _gauss_env(N, dx, 40e-6)
    with pytest.raises(ValueError):
        propagate_carrier_referenced(env, (30e-3, 40e-3, 50e-3), 5e-3, WL, dx)


def test_astig_z0_identity():
    N, dx = 128, 5e-6
    env = _gauss_env(N, dx, 40e-6)
    env_out, R_out, d_out = propagate_carrier_referenced(
        env, (30e-3, 45e-3), 0.0, WL, dx)
    assert np.array_equal(np.asarray(env_out), np.asarray(env))
    assert R_out == (30e-3, 45e-3)
    assert d_out == (dx, dx)


def test_astig_per_axis_focus_crossing_finite_and_flipped():
    """A converging astigmatic beam stepped through BOTH line foci returns a
    finite field with per-axis flipped diverging carriers (x and y cross at
    different z, each handled by its own bridge)."""
    N = 2048
    w_in = 0.5e-3
    dx = (8 * w_in) / N
    env = _gauss_env(N, dx, w_in)
    # foci near 20 mm (x) and 30 mm (y); step to 40 mm crosses both.
    env_out, R_out, d_out = propagate_carrier_referenced(
        env, (-20e-3, -30e-3), 40e-3, WL, dx)
    assert np.all(np.isfinite(np.asarray(env_out)))
    assert R_out[0] == pytest.approx(20e-3, rel=1e-9)     # -20 + 40, diverging
    assert R_out[1] == pytest.approx(10e-3, rel=1e-9)     # -30 + 40, diverging
    assert d_out[0] > 0 and d_out[1] > 0


# ===========================================================================
# Task 2 -- apertures
# ===========================================================================
def test_aperture_energy_removed_not_renormalised():
    """A hard aperture removes the clipped power exactly and does NOT
    renormalise: the returned transmission equals the measured power ratio,
    and the surviving power is strictly less than the input power."""
    N, dx = 512, 4e-6
    env = _gauss_env(N, dx, 200e-6)
    p_in = (np.abs(env) ** 2).sum()
    field, tr = carrier_referenced_aperture(
        env, -60e-3, WL, dx, radius=250e-6, return_transmission=True)
    p_out = (np.abs(np.asarray(field.env)) ** 2).sum()
    # exact accounting, no spurious renormalisation
    assert tr == pytest.approx(p_out / p_in, rel=1e-12)
    assert 0.0 < tr < 1.0
    assert p_out == pytest.approx(tr * p_in, rel=1e-12)
    assert p_out < p_in                    # power genuinely gone
    # carrier unchanged by an amplitude clip (default)
    assert field.R == -60e-3
    assert field.dx == dx


def test_aperture_rect_and_mask_forms():
    """Rectangular (slit) and explicit-mask apertures clip exactly and account
    energy correctly."""
    N, dx = 256, 5e-6
    env = _gauss_env(N, dx, 150e-6)
    p_in = (np.abs(env) ** 2).sum()
    # rectangular slit in x only
    field, tr = carrier_referenced_aperture(
        env, np.inf, WL, dx, half_x=120e-6, return_transmission=True)
    p_out = (np.abs(np.asarray(field.env)) ** 2).sum()
    assert tr == pytest.approx(p_out / p_in, rel=1e-12) and 0.0 < tr < 1.0
    # explicit mask == radius form
    x = (np.arange(N) - N / 2) * dx
    Y, X = np.meshgrid(x, x, indexing='ij')
    m = (X ** 2 + Y ** 2 <= 200e-6 ** 2).astype(float)
    fa = carrier_referenced_aperture(env, np.inf, WL, dx, mask=m)
    fb = carrier_referenced_aperture(env, np.inf, WL, dx, radius=200e-6)
    assert np.array_equal(np.asarray(fa.env), np.asarray(fb.env))
    with pytest.raises(ValueError):
        carrier_referenced_aperture(env, np.inf, WL, dx)      # no spec
    with pytest.raises(ValueError):
        carrier_referenced_aperture(env, np.inf, WL, dx, radius=1e-4,
                                    refit_carrier=True, new_carrier=1.0)


@pytest.mark.slow
def test_aperture_converging_leg_matches_fine_asm():
    """The core Task-2 claim: a converging leg apertured mid-chain and
    continued on a MODEST grid matches the exact fine-grid ASM (N=8192) to
    < 1 % windowed r2m PAST the aperture, with the clipped power correctly
    removed (transmission matched, no renormalisation).

    Measured envelope (grid-converged, N in {2048,4096,6144} agree): the
    carrier path's post-aperture r2m error is set by the paraxial-envelope /
    geometric-carrier treatment of the hard clip edge, NOT by sampling -- it
    is ~0.65 % for a clip at ~1.2 w and grows for more aggressive clips
    (~1.3 % at ~1.05 w).  This case clips at ~1.2 w."""
    R0, w_in = -60e-3, 0.4e-3
    z_ap, z_t, r_ap = 20e-3, 30e-3, 320e-6      # ~1.2 w at the aperture plane
    w_t = _w_of_q(_q_at(R0, z_t, w_in))
    win = 4.0 * w_t

    # fine-grid ASM ground truth
    Nf, ext = 8192, 6e-3
    dxf = ext / Nf
    xf = (np.arange(Nf) - Nf / 2) * dxf
    Xf, Yf = np.meshgrid(xf, xf, indexing='ij')
    E0 = (np.exp(-(Xf ** 2 + Yf ** 2) / w_in ** 2)
          * np.exp(1j * K * (Xf ** 2 + Yf ** 2) / (2 * R0))).astype(np.complex128)
    Rf = np.sqrt(Xf ** 2 + Yf ** 2)
    del Xf, Yf
    try:
        E = angular_spectrum_propagate(E0, z_ap, WL, dxf)
        p_before = (np.abs(E) ** 2).sum()
        E = E * (Rf <= r_ap)
        tr_f = (np.abs(E) ** 2).sum() / p_before
        E = angular_spectrum_propagate(E, z_t - z_ap, WL, dxf)
        r2m_f = _r2m_radial(np.abs(E) ** 2, dxf, dxf, win)
        del E, Rf

        # carrier-referenced path on a modest grid
        Nm = 2048
        dxm = (8 * w_in) / Nm
        env0 = _gauss_env(Nm, dxm, w_in)
        env, R, dx = propagate_carrier_referenced(env0, R0, z_ap, WL, dxm)
        p_before_m = (np.abs(np.asarray(env)) ** 2).sum() * dx * dx
        field, tr_m = carrier_referenced_aperture(
            env, R, WL, dx, radius=r_ap, return_transmission=True)
        p_after_m = (np.abs(np.asarray(field.env)) ** 2).sum() * field.dx ** 2
        env2, R2, dx2 = propagate_carrier_referenced(
            field.env, field.R, z_t - z_ap, WL, field.dx)
        r2m_m = _r2m_radial(np.abs(np.asarray(env2)) ** 2, dx2, dx2, win)

        # < 1 % downstream spatial agreement
        assert abs(r2m_m - r2m_f) / r2m_f < 0.01, (
            f"r2m carrier {r2m_m*1e6:.4f} vs ASM {r2m_f*1e6:.4f}")
        # transmission matches the fine ASM clip fraction
        assert abs(tr_m - tr_f) / tr_f < 0.005
        # no renormalisation: surviving power == transmission * pre-clip power
        assert p_after_m == pytest.approx(tr_m * p_before_m, rel=1e-12)
        assert tr_m < 0.98                       # a genuine (hard) clip
    finally:
        la.clear_asm_caches()


def test_fit_radius_recovers_known_carrier():
    """The carrier-fit estimator recovers a known spherical / astigmatic
    carrier from a clean full field, and returns +inf for a collimated one."""
    N, dx = 512, 3e-6
    w = 120e-6
    x = (np.arange(N) - N / 2) * dx
    Y, X = np.meshgrid(x, x, indexing='ij')
    amp = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    # isotropic
    R_known = 80e-3
    E = (amp * np.exp(1j * K * (X ** 2 + Y ** 2) / (2 * R_known))).astype(
        np.complex128)
    assert carrier_referenced_fit_radius(E, WL, dx) == pytest.approx(
        R_known, rel=5e-3)
    # collimated
    assert np.isinf(carrier_referenced_fit_radius(amp.astype(np.complex128),
                                                  WL, dx))
    # astigmatic
    Rx, Ry = 70e-3, -110e-3
    Ea = (amp * np.exp(1j * K * (X ** 2 / (2 * Rx) + Y ** 2 / (2 * Ry)))).astype(
        np.complex128)
    fx, fy = carrier_referenced_fit_radius(Ea, WL, dx, astigmatic=True)
    assert fx == pytest.approx(Rx, rel=5e-3)
    assert fy == pytest.approx(Ry, rel=5e-3)


def test_aperture_refit_flattens_envelope_and_reports_true_R():
    """``refit_carrier=True`` absorbs residual envelope curvature: after a
    clip on a beam whose true wavefront differs from the reference carrier by
    a known amount, the re-fitted R equals the true wavefront R and the
    re-referenced envelope is flatter (smaller residual phase curvature)."""
    N, dx = 512, 4e-6
    w = 160e-6
    x = (np.arange(N) - N / 2) * dx
    Y, X = np.meshgrid(x, x, indexing='ij')
    r2 = X ** 2 + Y ** 2
    R_true = 50e-3
    R_ref = 65e-3                       # carrier deliberately off the true R
    # envelope referenced by R_ref of a field whose true wavefront is R_true
    env = (np.exp(-r2 / w ** 2)
           * np.exp(1j * K * r2 * (1.0 / (2 * R_true) - 1.0 / (2 * R_ref)))
           ).astype(np.complex128)
    field = carrier_referenced_aperture(env, R_ref, WL, dx, radius=3 * w,
                                        refit_carrier=True)
    assert field.R == pytest.approx(R_true, rel=1e-2)
    # residual curvature of the re-referenced envelope is ~0 (flat)
    inv_resid = abs(1.0 / carrier_referenced_fit_radius(
        np.asarray(field.env), WL, dx))
    inv_before = abs(1.0 / R_true - 1.0 / R_ref)
    assert inv_resid < 0.05 * inv_before


def test_aperture_new_carrier_rereferences():
    """``new_carrier=R2`` re-references the apertured envelope to R2: the full
    field is unchanged (a pure phase relabel), only the carrier split moves."""
    N, dx = 256, 5e-6
    env = _gauss_env(N, dx, 120e-6)
    R1, R2 = 40e-3, 25e-3
    full_before = carrier_referenced_reconstruct(env, R1, WL, dx)
    field = carrier_referenced_aperture(env, R1, WL, dx, radius=3e-4,
                                        new_carrier=R2)
    assert field.R == R2
    # reconstruct with the NEW carrier -> same physical full field (clipped)
    full_after = carrier_referenced_reconstruct(field.env, R2, WL, dx)
    x = (np.arange(N) - N / 2) * dx
    Y, X = np.meshgrid(x, x, indexing='ij')
    clip = (X ** 2 + Y ** 2 <= (3e-4) ** 2)
    assert np.max(np.abs(full_after - full_before * clip)) < 1e-11


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
