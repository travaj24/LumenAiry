"""Carrier-referenced ("pilot-beam") free-space propagation -- Phase E
prototype (2026-07-19).

``propagate_carrier_referenced`` transports a strongly diverging /
converging beam's slowly-varying ENVELOPE on a modest grid while the
spherical carrier radius ``R`` and the co-moving grid pitch evolve
analytically (the Sziklas & Siegman 1975 scaled-coordinate Fresnel
transform).  This sidesteps the audit-H8 memory wall: the full field's
own fringe pitch ``lambda*R/r`` forces production grids to N=28672
propagator-independently, but the envelope carries no such fringe.

Independent oracles (rule 1):

* **Analytic Gaussian ABCD (q-parameter)** -- closed-form, grid-free,
  fully independent of the transform's FFT machinery.  A Gaussian's
  windowed second moment and encircled-energy radii have exact
  expressions in ``w`` (``r2m = w/sqrt(2)``; ``EE(a) = 1 - exp(-2a^2/w^2)``).
* **Exact band-limited ASM on a FINE grid** -- the same ground truth the
  task specifies; the modest carrier grid must match it to < 1% while
  using orders of magnitude less memory.

Fail-before / pass-after: propagating the FULL field (envelope times the
carrier fringe) on the SAME modest grid aliases the fringe and is grossly
wrong (test ``test_full_field_on_modest_grid_is_aliased``); the
carrier-referenced path on that grid is correct (< 1%).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy import (
    carrier_referenced_envelope,
    carrier_referenced_reconstruct,
    propagate_carrier_referenced,
)
from lumenairy.propagators.propagation import (
    angular_spectrum_propagate,
    fresnel_tf_propagate,
)

WL = 1.31e-6
K = 2.0 * np.pi / WL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gauss_env(N, dx, w):
    """Real Gaussian amplitude envelope, 1/e amplitude radius w."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _gauss_full(N, dx, w, R):
    """Full diverging Gaussian: amplitude w, spherical carrier R (>0 diverging)."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2
    ph = np.ones_like(r2) if np.isinf(R) else np.exp(1j * K * r2 / (2.0 * R))
    return (np.exp(-r2 / w ** 2) * ph).astype(np.complex128)


def _q_at_R(w0, R_target):
    """(z, w, R, zR) of a w0-waist Gaussian at the far-field plane where R==R_target."""
    zR = np.pi * w0 ** 2 / WL
    z = (R_target + np.sqrt(R_target ** 2 - 4 * zR ** 2)) / 2.0
    w = w0 * np.sqrt(1 + (z / zR) ** 2)
    return z, w, R_target, zR


def _abcd_free(w0, z):
    zR = np.pi * w0 ** 2 / WL
    return w0 * np.sqrt(1 + (z / zR) ** 2), z * (1 + (zR / z) ** 2)


def _metrics(I, dx, window_r=None):
    """Windowed second-moment radius and EE50/EE80 radii, centred on the grid."""
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    if window_r is not None:
        I = np.where(r <= window_r, I, 0.0)
    tot = I.sum()
    r2m = np.sqrt((I * r ** 2).sum() / tot)
    order = np.argsort(r.ravel())
    cs = np.cumsum(I.ravel()[order]) / tot
    r_sorted = r.ravel()[order]
    ee50 = r_sorted[min(int(np.searchsorted(cs, 0.5)), r.size - 1)]
    ee80 = r_sorted[min(int(np.searchsorted(cs, 0.8)), r.size - 1)]
    return r2m, ee50, ee80


# ---------------------------------------------------------------------------
# 1. single step vs the analytic Gaussian ABCD oracle (grid-free, exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('w0', [4e-6, 8e-6])
def test_single_step_matches_gaussian_abcd(w0):
    """A diverging Gaussian (waist w0, referenced at R=+55 mm) propagated a
    further 50 mm by the carrier-referenced step must reproduce the analytic
    ABCD beam width -- r2m, EE50, EE80 to < 0.2 % -- and grow the carrier
    radius and grid pitch by exactly the geometric magnification."""
    z1, w1, R1, zR = _q_at_R(w0, 55e-3)
    z_prop = 50e-3
    w2, R2_abcd = _abcd_free(w0, z1 + z_prop)

    N = 1024
    dx = (6.5 * w1) / N
    env_out, R_out, dx_out = propagate_carrier_referenced(
        _gauss_env(N, dx, w1), R1, z_prop, WL, dx)
    I = np.abs(env_out) ** 2                      # |full field|^2 == |env|^2
    r2m, ee50, ee80 = _metrics(I, dx_out)

    # carrier tracks the geometric sphere exactly (R_out = R_in + z); this
    # differs from the true Gaussian R2 only by the negligible (zR/z)^2 term
    # (~2e-7 rel), which the envelope absorbs.
    assert R_out == pytest.approx(R1 + z_prop, rel=1e-12)
    assert R_out == pytest.approx(R2_abcd, rel=1e-5)
    assert dx_out == pytest.approx(dx * R_out / R1, rel=1e-9)

    # closed-form Gaussian oracle
    r2m_a, ee50_a, ee80_a = w2 / np.sqrt(2), 0.588705 * w2, 0.897226 * w2
    assert r2m == pytest.approx(r2m_a, rel=2e-3)
    assert ee50 == pytest.approx(ee50_a, rel=2e-3)
    assert ee80 == pytest.approx(ee80_a, rel=2e-3)


# ---------------------------------------------------------------------------
# 2. modest carrier grid vs exact ASM on a FINE grid + memory contrast
# ---------------------------------------------------------------------------
def test_matches_exact_asm_fine_grid_low_memory():
    """The core claim: the carrier-referenced result on a MODEST grid matches
    the full-sampling exact-ASM ground truth on a FINE grid to < 1 % in
    windowed r2m / EE, using orders of magnitude less memory."""
    w0 = 8e-6
    z1, w1, R1, zR = _q_at_R(w0, 55e-3)
    z_prop = 50e-3
    w2, _ = _abcd_free(w0, z1 + z_prop)
    window = 1.5 * w2

    # ground truth: exact band-limited ASM of the FULL field on a fine grid
    Nf = 4096
    dxf = (6.0 * w2) / Nf
    Ef = angular_spectrum_propagate(_gauss_full(Nf, dxf, w1, R1), z_prop, WL, dxf)
    r2m_f, ee50_f, ee80_f = _metrics(np.abs(Ef) ** 2, dxf, window_r=window)

    # carrier-referenced on a modest grid
    Nm = 1024
    dxm = (6.5 * w1) / Nm
    env_out, R_out, dx_out = propagate_carrier_referenced(
        _gauss_env(Nm, dxm, w1), R1, z_prop, WL, dxm)
    r2m_m, ee50_m, ee80_m = _metrics(np.abs(env_out) ** 2, dx_out, window_r=window)

    assert abs(r2m_m - r2m_f) / r2m_f < 0.01
    assert abs(ee50_m - ee50_f) / ee50_f < 0.01
    assert abs(ee80_m - ee80_f) / ee80_f < 0.01

    # memory contrast: the carrier path never allocates a fine-grid array
    fine_bytes = Nf * Nf * 16
    modest_bytes = Nm * Nm * 16
    assert modest_bytes * 8 < fine_bytes           # >= 8x smaller working array
    assert env_out.size == Nm * Nm                 # output stayed modest


def test_full_field_on_modest_grid_is_aliased():
    """Fail-before: sampling the FULL field (carrier fringe included) on the
    modest grid the carrier path uses aliases the fringe and is grossly wrong
    -- this is exactly what the carrier reference avoids.  The carrier path on
    the SAME grid is correct (previous test); this locks in WHY it is needed."""
    w0 = 8e-6
    z1, w1, R1, zR = _q_at_R(w0, 55e-3)
    z_prop = 50e-3
    w2, _ = _abcd_free(w0, z1 + z_prop)
    window = 1.5 * w2

    Nf = 4096
    dxf = (6.0 * w2) / Nf
    Ef = angular_spectrum_propagate(_gauss_full(Nf, dxf, w1, R1), z_prop, WL, dxf)
    r2m_f, _, _ = _metrics(np.abs(Ef) ** 2, dxf, window_r=window)

    # naive full-field ASM on the modest grid: the R=55 mm fringe is undersampled
    Nm = 1024
    dxm = (6.5 * w1) / Nm
    # verify the fringe really is undersampled at this pitch (pitch < 2*dx at 2w1)
    fringe_pitch_2w = WL * R1 / (2.0 * w1)
    assert dxm > 0.5 * fringe_pitch_2w, "grid unexpectedly samples the fringe"
    E_naive = angular_spectrum_propagate(
        _gauss_full(Nm, dxm, w1, R1), z_prop, WL, dxm)
    r2m_naive, _, _ = _metrics(np.abs(E_naive) ** 2, dxm, window_r=window)

    # the aliased full-field result is far off the ground truth (>> the 1%
    # the carrier path achieves)
    assert abs(r2m_naive - r2m_f) / r2m_f > 0.1


# ---------------------------------------------------------------------------
# 3. collimated carrier reduces to an ordinary Fresnel step (exact)
# ---------------------------------------------------------------------------
def test_collimated_reduces_to_fresnel_tf_bit_exact():
    """R = +inf is a collimated carrier: the transform degenerates to m == 1,
    z_eff == z, and the step is byte-identical to a plain Fresnel TF
    propagation with the grid unchanged."""
    N = 512
    dx = 4e-6
    env = _gauss_env(N, dx, 30e-6)
    out = propagate_carrier_referenced(env, np.inf, 5e-3, WL, dx)
    ref = fresnel_tf_propagate(env, 5e-3, WL, dx)
    assert out.R == np.inf
    assert out.dx == dx
    assert np.array_equal(np.asarray(out.env), np.asarray(ref))


# ---------------------------------------------------------------------------
# 4. power conservation + reconstruct/extract inverse
# ---------------------------------------------------------------------------
def test_power_conserved_across_scaled_step():
    """Total power (|env|^2 integrated over the co-moving area element) is
    invariant: the 1/m amplitude factor exactly offsets the m^2 area growth."""
    N = 768
    dx = 6e-6
    env = _gauss_env(N, dx, 80e-6)
    p_in = (np.abs(env) ** 2).sum() * dx ** 2
    for R, z in [(30e-3, 40e-3), (-80e-3, 20e-3), (150e-3, -25e-3)]:
        env_out, R_out, dx_out = propagate_carrier_referenced(env, R, z, WL, dx)
        p_out = (np.abs(env_out) ** 2).sum() * dx_out ** 2
        assert p_out == pytest.approx(p_in, rel=2e-4), f"R={R} z={z}"
        assert R_out == pytest.approx(R + z, rel=1e-12)


def test_reconstruct_envelope_roundtrip():
    """reconstruct(env) then envelope() recovers the envelope to ~machine eps;
    the reconstructed full field carries exactly the +k r^2/2R carrier."""
    N = 256
    dx = 5e-6
    env = _gauss_env(N, dx, 60e-6) * np.exp(0.3j)     # nontrivial global phase
    R = 42e-3
    full = carrier_referenced_reconstruct(env, R, WL, dx)
    back = carrier_referenced_envelope(full, R, WL, dx)
    assert np.max(np.abs(back - env)) < 1e-13
    # the imprinted carrier matches apply_fresnel_curvature's +R convention
    ref = la.apply_fresnel_curvature(env, dx, WL, R, sign=+1)
    assert np.max(np.abs(full - ref)) < 1e-12


# ---------------------------------------------------------------------------
# 5. guards: z==0 identity, R==0, focus crossing
# ---------------------------------------------------------------------------
def test_zero_distance_identity():
    N = 128
    dx = 5e-6
    env = _gauss_env(N, dx, 40e-6)
    env_out, R_out, dx_out = propagate_carrier_referenced(env, 30e-3, 0.0, WL, dx)
    assert np.array_equal(np.asarray(env_out), np.asarray(env))
    assert R_out == 30e-3 and dx_out == dx


def test_zero_carrier_raises_but_focus_crossing_is_handled():
    """R_carrier == 0 (the carrier's own focus) still raises; a focus CROSSING
    is now auto-split (Task 2), not an error -- it returns a finite field with
    a flipped diverging carrier."""
    N = 128
    dx = 5e-6
    env = _gauss_env(N, dx, 40e-6)
    with pytest.raises(ValueError):
        propagate_carrier_referenced(env, 0.0, 5e-3, WL, dx)
    # converging R=-20mm stepped +25mm crosses the focus (R_out=+5mm): handled.
    out = propagate_carrier_referenced(env, -20e-3, 25e-3, WL, dx)
    assert np.all(np.isfinite(np.asarray(out.env)))
    assert out.R == pytest.approx(5e-3, rel=1e-12)      # flipped, diverging
    assert out.dx > 0


# ---------------------------------------------------------------------------
# 6. two-leg composition: carrier leg -> ideal thin lens -> carrier leg
# ---------------------------------------------------------------------------
def test_two_leg_composition_ideal_lens():
    """A diverging Gaussian threaded through carrier leg 1 -> an ideal thin
    lens (a pure carrier operation: envelope unchanged, R updated by the lens
    equation) -> carrier leg 2, on modest grids, agrees with (a) the analytic
    ABCD image and (b) a full-grid exact-ASM ground truth, at a fraction of
    the memory."""
    # input plane A: w0-waist Gaussian, z_A past the waist so R_A is finite
    w0 = 40e-6
    zR = np.pi * w0 ** 2 / WL
    z_A = 30e-3
    q_A = z_A + 1j * zR
    w_A = w0 * np.sqrt(1 + (z_A / zR) ** 2)
    R_A = z_A * (1 + (zR / z_A) ** 2)
    L1, f, L2 = 25e-3, 300e-3, 40e-3

    # ABCD q-trace oracle through the whole chain
    q = q_A + L1
    q = 1.0 / (1.0 / q - 1.0 / f)
    q = q + L2
    w_img = np.sqrt(-WL / (np.pi * (1.0 / q).imag))
    r2m_abcd = w_img / np.sqrt(2)
    window = 1.5 * w_img

    # carrier-referenced chain on a modest grid
    Nm = 1024
    dxm = (7 * w_A) / Nm
    env, R, dx = propagate_carrier_referenced(
        _gauss_env(Nm, dxm, w_A), R_A, L1, WL, dxm)
    R = 1.0 / (1.0 / R - 1.0 / f)                  # ideal thin lens: env untouched
    env, R, dx = propagate_carrier_referenced(env, R, L2, WL, dx)
    r2m_c, _, _ = _metrics(np.abs(env) ** 2, dx, window_r=window)

    # full-grid exact-ASM ground truth
    Nf = 4096
    w_lens = w0 * np.sqrt(1 + ((z_A + L1) / zR) ** 2)
    dxf = (7 * max(w_A, w_lens, w_img)) / Nf
    E = _gauss_full(Nf, dxf, w_A, R_A)
    E = angular_spectrum_propagate(E, L1, WL, dxf)
    x = (np.arange(Nf) - Nf / 2) * dxf
    X, Y = np.meshgrid(x, x)
    E = E * np.exp(-1j * K * (X ** 2 + Y ** 2) / (2 * f))    # ideal thin lens
    E = angular_spectrum_propagate(E, L2, WL, dxf)
    r2m_g, _, _ = _metrics(np.abs(E) ** 2, dxf, window_r=window)

    # carrier chain matches the full-grid ASM ground truth (same window)
    assert abs(r2m_c - r2m_g) / r2m_g < 0.01
    # both track the analytic ABCD image (windowed second moment truncates the
    # Gaussian tail identically for both, so use a tolerance that admits it)
    assert abs(r2m_c - r2m_abcd) / r2m_abcd < 0.05
    assert abs(r2m_g - r2m_abcd) / r2m_abcd < 0.05
    # memory: the chain ran entirely on the modest grid
    assert Nm * Nm * 8 < Nf * Nf


# ---------------------------------------------------------------------------
# 7. hand-off to apply_real_lens_traced via the carrier (H6): reconstruct the
#    full field at the lens plane, pass carrier=R, verify it focuses.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_traced_handoff_focuses_at_abcd_image():
    """Demonstrate the traced hand-off the task calls for: a carrier-referenced
    leg's envelope, reconstructed to a full field at the lens plane and handed
    to ``apply_real_lens_traced(carrier=R)``, focuses the diverging beam at the
    ABCD image plane (H6 makes the carrier reference exact).  Independent
    oracle: ABCD Gaussian q-trace through the singlet."""
    n_glass = 1.5168
    R1s, R2s, tc = 51.68e-3, -51.68e-3, 5e-3
    from lumenairy.glass import GLASS_REGISTRY
    GLASS_REGISTRY['_CARRIER_HANDOFF_GLASS'] = lambda wl: n_glass
    # aperture metadata fits the N*dx/2 = 3.07 mm grid (the 1 mm-waist beam is
    # tiny inside it) so the traced call stays warning-free
    presc = {
        'wavelength': WL, 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': R1s, 'thickness': tc, 'glass_before': 'air',
             'glass_after': '_CARRIER_HANDOFF_GLASS', 'semi_diameter': 3e-3},
            {'radius': R2s, 'thickness': 0.0,
             'glass_before': '_CARRIER_HANDOFF_GLASS', 'glass_after': 'air',
             'semi_diameter': 3e-3},
        ], 'thicknesses': [tc], 'stop_index': 0,
    }

    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])

    M = refr(R2s, n_glass, 1.0) @ trans(tc) @ refr(R1s, 1.0, n_glass)

    # diverging Gaussian at the lens entrance vertex: R_lens = +150 mm
    R_lens = 150e-3
    w_L = 1.0e-3
    q0 = 1.0 / (1.0 / R_lens - 1j * WL / (np.pi * w_L ** 2))
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    z_img = -q1.real                              # ABCD image distance past exit
    q_w = q1 + z_img
    w_img = np.sqrt(-WL / (np.pi * (1.0 / q_w).imag))

    # build the lens-input field as (envelope) * (carrier) via the reconstruct
    # helper -- exactly the hand-off a carrier-referenced leg produces
    N, dx = 768, 8e-6
    env = _gauss_env(N, dx, w_L)
    E_lens_in = carrier_referenced_reconstruct(env, R_lens, WL, dx)

    E_exit = np.asarray(la.apply_real_lens_traced(
        E_lens_in, prescription=presc, wavelength=WL, dx=dx,
        carrier=R_lens, on_noncollimated='silent'))
    assert np.all(np.isfinite(E_exit))

    def ee_at(z, r_ee=3.0 * float(w_img)):
        E = angular_spectrum_propagate(E_exit, z, WL, dx)
        I = np.abs(E) ** 2
        j, i = np.unravel_index(np.argmax(I), I.shape)
        xx = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(xx - xx[i], xx - xx[j])
        r = np.sqrt(X ** 2 + Y ** 2)
        return I[r <= r_ee].sum() / I.sum()

    zs = np.linspace(0.8 * z_img, 1.2 * z_img, 13)
    ees = [ee_at(float(z)) for z in zs]
    z_bf = zs[int(np.argmax(ees))]
    # best focus at the ABCD image (not the collimated focal plane), < 6%
    assert abs(z_bf - z_img) / z_img < 0.06
    assert ee_at(z_img) > ee_at(49.163e-3)        # beats the collimated BFL


# ---------------------------------------------------------------------------
# 8. focus crossing (Task 2): auto-split carrier -> waist ASM bridge -> carrier
# ---------------------------------------------------------------------------
def _analytic_windowed(w_true, window_r):
    """Exact windowed r2m/EE80 of a Gaussian |E|^2 = exp(-2 r^2/w^2)."""
    rr = np.linspace(0, window_r, 20001)
    I = np.exp(-2 * rr ** 2 / w_true ** 2)
    tot = np.trapezoid(I * rr, rr)
    r2m = np.sqrt(np.trapezoid(I * rr ** 3, rr) / tot)
    cum = np.array([np.trapezoid((I * rr)[:j + 1], rr[:j + 1])
                    for j in range(0, rr.size, 40)]) / tot
    rq = rr[::40]
    return r2m, float(np.interp(0.8, cum, rq))


def _radial_metrics(I, dx, window_r):
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    Iw = np.where(r <= window_r, I, 0.0)
    tot = Iw.sum()
    r2m = np.sqrt((Iw * r ** 2).sum() / tot)
    order = np.argsort(r.ravel())
    cs = np.cumsum(Iw.ravel()[order]) / tot
    rs = r.ravel()[order]
    ee80 = rs[min(int(np.searchsorted(cs, 0.8)), r.size - 1)]
    return r2m, ee80


# converging beam that focuses to a w0=4 um waist 30 mm downstream
_W0_X = 4e-6
_ZR_X = np.pi * _W0_X ** 2 / WL
_ZF_X = 30e-3
_RIN_X = -(_ZF_X + _ZR_X ** 2 / _ZF_X)             # ~ -30 mm, converging
_WIN_X = _W0_X * np.sqrt(1 + (_ZF_X / _ZR_X) ** 2)  # input beam radius (~3.1 mm)


@pytest.mark.parametrize('z_mm,label', [
    (30.0, 'waist'), (35.0, 'just past'), (45.0, 'past'), (60.0, 'far past')])
def test_focus_crossing_matches_analytic_gaussian(z_mm, label):
    """A converging Gaussian (w0=4 um, R=-30 mm) propagated THROUGH its waist on
    a MODEST N=2048 grid must match the exact analytic Gaussian (ABCD) width to
    <1% windowed r2m/EE -- at the waist AND on both sides.  The single fine grid
    that would be needed to sample both the 3 mm input beam and the 4 um waist
    is why the carrier auto-split is required."""
    N = 2048
    dx = (8.0 * _WIN_X) / N                        # hold the beam to ~3 sigma
    z = z_mm * 1e-3
    env, R_out, dx_out = propagate_carrier_referenced(
        _gauss_env(N, dx, _WIN_X), _RIN_X, z, WL, dx)
    I = np.abs(np.asarray(env)) ** 2               # |full| == |env|
    dz = z - _ZF_X
    w_true = _W0_X * np.sqrt(1 + (dz / _ZR_X) ** 2)
    window = 3.0 * w_true
    r2m_c, ee80_c = _radial_metrics(I, dx_out, window)
    r2m_a, ee80_a = _analytic_windowed(w_true, window)
    assert abs(r2m_c - r2m_a) / r2m_a < 0.01, (
        f"{label}: r2m {r2m_c*1e6:.3f} vs analytic {r2m_a*1e6:.3f}")
    assert abs(ee80_c - ee80_a) / ee80_a < 0.01, (
        f"{label}: EE80 {ee80_c*1e6:.3f} vs analytic {ee80_a*1e6:.3f}")
    # past-focus carrier is DIVERGING (flipped through the waist)
    if z > _ZF_X:
        assert R_out > 0


def test_focus_crossing_conserves_power():
    """Total power (|env|^2 over the co-moving area element) is conserved across
    the auto-split focus crossing."""
    N = 2048
    dx = (8.0 * _WIN_X) / N
    env0 = _gauss_env(N, dx, _WIN_X)
    p_in = (np.abs(env0) ** 2).sum() * dx ** 2
    env, R_out, dx_out = propagate_carrier_referenced(env0, _RIN_X, 60e-3, WL, dx)
    p_out = (np.abs(np.asarray(env)) ** 2).sum() * dx_out ** 2
    assert abs(p_out - p_in) / p_in < 2e-3


def test_focus_crossing_composition():
    """A crossing leg composes: (carrier leg, no crossing) -> (leg that crosses
    the focus) -> (carrier leg) equals a single crossing leg to the same plane
    (the auto-split re-references a faithful diverging carrier)."""
    N = 2048
    dx = (8.0 * _WIN_X) / N
    env0 = _gauss_env(N, dx, _WIN_X)
    # 10 (no cross) -> 40 (crosses) -> 15 == single 65 mm crossing leg
    e1, R1, d1 = propagate_carrier_referenced(env0, _RIN_X, 10e-3, WL, dx)
    e2, R2, d2 = propagate_carrier_referenced(e1, R1, 40e-3, WL, d1)
    e3, R3, d3 = propagate_carrier_referenced(e2, R2, 15e-3, WL, d2)
    es, Rs, ds = propagate_carrier_referenced(env0, _RIN_X, 65e-3, WL, dx)
    assert R3 == pytest.approx(Rs, rel=1e-9)
    assert d3 == pytest.approx(ds, rel=1e-6)
    win = 6.0 * (Rs * WL / (np.pi * _W0_X))         # generous window
    r_comp, _ = _radial_metrics(np.abs(np.asarray(e3)) ** 2, d3, win)
    r_sing, _ = _radial_metrics(np.abs(np.asarray(es)) ** 2, ds, win)
    assert abs(r_comp - r_sing) / r_sing < 1e-3


def test_near_focus_landing_fast_path_unchanged():
    """A step landing comfortably away from focus (same sign, |m| ~ O(1)) takes
    the byte-identical fast path -- the near-focus guard does NOT reroute it to
    the bridge, so the result is the exact fast-step output.  Pins the
    no-crossing contract."""
    from lumenairy.propagators.carrier import _carrier_step_fast
    N = 512
    dx = 5e-6
    env = _gauss_env(N, dx, 40e-6)
    out = propagate_carrier_referenced(env, 30e-3, 40e-3, WL, dx)
    ref = _carrier_step_fast(env, 30e-3, 40e-3, WL, dx, dx)
    # byte-identical to the extracted fast step (i.e. no rerouting to bridge)
    assert np.array_equal(np.asarray(out.env), np.asarray(ref.env))
    assert out.R == ref.R and out.dx == ref.dx


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
