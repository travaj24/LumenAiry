"""Regression tests for the 2026-07 wave-lens-models audit remediation.

Covers (see docs/audits/AUDIT_WAVE_LENS_MODELS_2026_07_02*.md):
  N1  Maslov G design matrix hoisted -> non-quadrature integrators run at
      output_subsample=1 without the (N_out^2, M) allocation.
  N4  fitted linear OPD term re-applied -> extract_linear_phase True == False.
  N3  pupil chart sized from input angular content + input_na coverage guard.
  N2  under-resolved uniform quadrature warns.
  F3  Maslov progress uses the suite (stage, frac, msg) signature.
  F4  tilt_aware_rays recommendation gated on a wrapping-safe coherence ratio.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_traced import apply_real_lens_traced

LAM = 1.31e-6


def _singlet():
    return {
        'name': 'singlet', 'aperture_diameter': 12e-3,
        'surfaces': [
            {'radius': 103e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 6e-3},
            {'radius': -103e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 6e-3},
        ],
        'thicknesses': [4e-3],
    }


def _gauss(N, dx, w=3e-3, tilt=0.0):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    if tilt:
        E = E * np.exp(1j * (2 * np.pi / LAM) * np.sin(tilt) * X)
    return E.astype(np.complex64)


_MASLOV_KW = dict(output_subsample=1, ray_field_samples=14,
                  ray_pupil_samples=14, poly_order=4, n_v2=24)


@pytest.mark.parametrize('method',
                         ['quadrature', 'stationary_phase', 'local_quadrature'])
def test_n1_all_integrators_run_full_resolution(method):
    """All three integrators run at output_subsample=1 and conserve power.
    Post-N1 the non-quadrature integrators no longer build G at all."""
    N, dx = 192, 70e-6
    E = _gauss(N, dx)
    out = la.apply_real_lens_maslov(
        E, prescription=_singlet(), wavelength=LAM, dx=dx,
        integration_method=method, **_MASLOV_KW)
    assert out.shape == (N, N)
    assert np.isfinite(out).all()
    ratio = float(np.sum(np.abs(out) ** 2) / np.sum(np.abs(E) ** 2))
    assert 0.9 < ratio < 1.1


@pytest.mark.parametrize('method',
                         ['quadrature', 'stationary_phase', 'local_quadrature'])
def test_n4_linear_phase_reapplied_offaxis(method):
    """extract_linear_phase True and False must agree on an OFF-AXIS
    (tilted) input now that the fitted linear OPD term is re-applied."""
    N, dx = 256, 60e-6
    E = _gauss(N, dx, tilt=0.02)  # off-axis -> non-zero linear OPD term
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method=method, **_MASLOV_KW)
    a = la.apply_real_lens_maslov(E, extract_linear_phase=True, **kw)
    b = la.apply_real_lens_maslov(E, extract_linear_phase=False, **kw)
    denom = float(np.abs(a).max())
    rel = float(np.abs(a - b).max()) / denom
    assert rel < 5e-3, f"{method}: True/False diverge ({rel:.2e})"


def test_n3_input_na_widens_chart():
    """A divergent input contributes a positive input-NA term to the chart."""
    import contextlib
    import io
    N, dx = 256, 60e-6
    E = _gauss(N, dx, w=1.0e-3)  # tight waist -> divergent
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx,
            integration_method='stationary_phase', verbose=True, **_MASLOV_KW)
    na_lines = [ln for ln in buf.getvalue().splitlines() if 'NA_proxy' in ln]
    assert na_lines, "no NA_proxy verbose line"
    assert 'input' in na_lines[0]


def test_n3_coverage_warning_on_undersized_input_na():
    N, dx = 256, 60e-6
    E = _gauss(N, dx, w=1.0e-3)
    with pytest.warns(RuntimeWarning, match='may not cover'):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx,
            integration_method='stationary_phase', input_na=1e-5, **_MASLOV_KW)


def test_n2_under_resolved_quadrature_warns():
    N, dx = 256, 60e-6
    E = _gauss(N, dx, w=1.0e-3)
    with pytest.warns(RuntimeWarning, match='under-resolved'):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx,
            integration_method='quadrature', output_subsample=2,
            ray_field_samples=14, ray_pupil_samples=14, poly_order=4, n_v2=6)


def test_f3_suite_style_progress_callback_does_not_crash():
    """A strict 3-arg (stage, frac, msg) suite callback must work; before F3
    the bespoke keyword call raised TypeError mid-lens."""
    N, dx = 160, 90e-6
    E = _gauss(N, dx)
    seen = []

    def cb(stage, frac, msg=''):
        seen.append((str(stage), float(frac)))

    out = la.apply_real_lens_maslov(
        E, prescription=_singlet(), wavelength=LAM, dx=dx,
        integration_method='stationary_phase', progress=cb, **_MASLOV_KW)
    assert np.isfinite(out).all()
    assert len(seen) > 0
    assert all(0.0 <= f <= 1.0 for _, f in seen)


def _f4_prescription():
    return {
        'name': 't', 'aperture_diameter': 2e-3,
        'surfaces': [
            {'radius': 12e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7'},
            {'radius': -12e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air'},
        ],
        'thicknesses': [2e-3],
    }


def _f4_warn_text(E):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        apply_real_lens_traced(
            E, prescription=_f4_prescription(), wavelength=LAM, dx=6e-6,
            ray_subsample=4, parallel_amp=False)
    return ' '.join(str(x.message) for x in w
                    if isinstance(x.message, RuntimeWarning))


def test_f4_single_tilt_recommends_tilt_aware():
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    g = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2)
    # strong single tilt (many 2*pi wraps) -- the case the gradient-based
    # discriminator mishandled
    E = (g * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.08) * X)).astype(np.complex64)
    txt = _f4_warn_text(E)
    # the recommend-branch is the only one that says "for tilt-sensitive
    # analyses" (the incoherent branch also mentions tilt_aware_rays=True,
    # negated -- so match the unambiguous phrase)
    assert 'for tilt-sensitive analyses' in txt
    assert 'INCOHERENT' not in txt


def test_f4_two_beam_fringe_does_not_recommend_tilt_aware():
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    g = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2)
    kx = (2 * np.pi / LAM) * np.sin(0.05)
    E = (g * (np.exp(1j * kx * X) + np.exp(-1j * kx * X))).astype(np.complex64)
    txt = _f4_warn_text(E)
    assert 'INCOHERENT' in txt
    assert 'for tilt-sensitive analyses' not in txt


def test_f4_collimated_no_warning():
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2).astype(np.complex64)
    assert _f4_warn_text(E).strip() == ''


# --------------------------------------------------------------------------
# S5.1 carrier-referenced traced + F1 collimation guard
# --------------------------------------------------------------------------
from lumenairy.elements._lens_real import apply_real_lens  # noqa: E402
from lumenairy.elements._lens_traced import _carrier_residual_rms, _compute_carrier  # noqa: E402


def _diverging(N, dx, s, w=0.6e-3):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    k0 = 2 * np.pi / LAM
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * k0 * (X ** 2 + Y ** 2) / (2 * s)))
    return E.astype(np.complex64), X, Y


def test_carrier_residual_removes_divergence():
    """The matching carrier drives the residual angular spread to ~0."""
    N, dx, s = 512, 6e-6, 30e-3
    E, X, Y = _diverging(N, dx, s)
    raw = _carrier_residual_rms(E, None, LAM, dx)
    W = (X ** 2 + Y ** 2) / (2 * s)
    resid = _carrier_residual_rms(E, W, LAM, dx)
    assert raw > 0.01
    assert resid < 0.1 * raw


def test_carrier_auto_recovers_conjugate():
    """carrier='auto' fits a wavefront whose edge slope matches the known
    diverging conjugate (grad W ~ x/s)."""
    N, dx, s = 512, 6e-6, 30e-3
    E, X, Y = _diverging(N, dx, s)
    _W, grad_fn = _compute_carrier('auto', E, LAM, dx, X, Y)
    xq = np.array([0.8e-3, -0.8e-3, 0.0])
    yq = np.array([0.0, 0.0, 0.8e-3])
    L, M = grad_fn(xq, yq)
    # expected direction cosines x/s, y/s
    assert np.allclose(L, xq / s, atol=0.15 * abs(0.8e-3 / s) + 1e-4)
    assert np.allclose(M, yq / s, atol=0.15 * abs(0.8e-3 / s) + 1e-4)


def _carrier_lens():
    return {
        'name': 's', 'aperture_diameter': 3e-3,
        'surfaces': [
            {'radius': 60e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 1.5e-3},
            {'radius': -60e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 1.5e-3},
        ],
        'thicknesses': [3e-3],
    }


def test_f1_guard_warns_on_unreferenced_divergent_input():
    N, dx, s = 1024, 4e-6, 25e-3
    E, _, _ = _diverging(N, dx, s)
    with pytest.warns(RuntimeWarning, match='collimated-reference'):
        apply_real_lens_traced(E, prescription=_carrier_lens(),
                               wavelength=LAM, dx=dx, ray_subsample=2,
                               parallel_amp=False)


def test_f1_guard_silent_when_carrier_matches():
    """With a matching carrier the residual is small -> no F1 warning."""
    N, dx, s = 1024, 4e-6, 25e-3
    E, _, _ = _diverging(N, dx, s)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        apply_real_lens_traced(E, prescription=_carrier_lens(),
                               wavelength=LAM, dx=dx, ray_subsample=2,
                               parallel_amp=False, carrier=s)
    assert not any('collimated-reference' in str(x.message) for x in w)


def test_f1_delegate_returns_analytic():
    """on_noncollimated='delegate' falls back to apply_real_lens exactly."""
    N, dx, s = 1024, 4e-6, 25e-3
    E, _, _ = _diverging(N, dx, s)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = apply_real_lens_traced(
            E, prescription=_carrier_lens(), wavelength=LAM, dx=dx,
            ray_subsample=2, parallel_amp=False, on_noncollimated='delegate')
        ref = apply_real_lens(E, prescription=_carrier_lens(),
                              wavelength=LAM, dx=dx)
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------------
# Adversarial-review P2 fixes (2026-07-03)
#   N4  s2-tilt post-multiply moved to the FINE (post-upsample) grid.
#   N3  NA proxy clamped to < 1 so a broadband input cannot grazing-kill
#       the whole pupil chart.
# --------------------------------------------------------------------------
def _weak_lens(R=2000e-3, ap=1.6e-3):
    """A weak, small-aperture lens whose output is well-resolved on a
    coarse output grid -> isolates the linear-OPD post-multiply from the
    quadratic (focusing) phase that N2 under-resolution is about."""
    return {
        'name': 'weak', 'aperture_diameter': ap,
        'surfaces': [
            {'radius': R, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': ap / 2},
            {'radius': -R, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': ap / 2},
        ],
        'thicknesses': [1.5e-3],
    }


def _spectral_centroid_fx(E, dx):
    N = E.shape[0]
    P = np.abs(np.fft.fft2(E)) ** 2
    fx = np.fft.fftfreq(N, d=dx)
    FX, _ = np.meshgrid(fx, fx, indexing='xy')
    return float((FX * P).sum() / P.sum())


def test_n4_wellresolved_output_is_subsample_invariant():
    """A well-resolved output tilt must be recovered with the SAME sign and
    ~same magnitude at output_subsample=1 and >1.  Before the P2 fix the
    linear-OPD post-multiply was applied on the coarse grid before the cubic
    upsample; a coarse-grid multiply that aliases would flip the recovered
    tilt sign here.  (The reachable regime keeps the tilt below the coarse
    Nyquist; the fix guarantees the post-multiply itself never corrupts it.)"""
    N, dx, w = 384, 5e-6, 0.45e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.012) * X)).astype(np.complex64)
    kw = dict(prescription=_weak_lens(), wavelength=LAM, dx=dx,
              integration_method='quadrature', ray_field_samples=14,
              ray_pupil_samples=14, poly_order=4, n_v2=20,
              extract_linear_phase=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c1 = _spectral_centroid_fx(
            la.apply_real_lens_maslov(E, output_subsample=1, **kw), dx)
        c6 = _spectral_centroid_fx(
            la.apply_real_lens_maslov(E, output_subsample=6, **kw), dx)
    assert np.sign(c1) == np.sign(c6), f"tilt sign flipped: {c1:.1f} vs {c6:.1f}"
    assert abs(c6 - c1) / (abs(c1) + 1e-9) < 0.05, f"{c1:.1f} vs {c6:.1f}"


def test_n4_piston_reapplied_as_global_phase():
    """The fitted piston (_lin[0], ~10^3 waves) is re-applied but is a global
    phase: it must not touch the intensity (|E| identical whether or not the
    linear term is extracted)."""
    N, dx, w = 256, 6e-6, 0.5e-3
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex64)
    kw = dict(prescription=_weak_lens(), wavelength=LAM, dx=dx,
              integration_method='quadrature', ray_field_samples=14,
              ray_pupil_samples=14, poly_order=4, n_v2=20, output_subsample=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = la.apply_real_lens_maslov(E, extract_linear_phase=True, **kw)
        b = la.apply_real_lens_maslov(E, extract_linear_phase=False, **kw)
    ia, ib = np.abs(a), np.abs(b)
    rel = float(np.abs(ia - ib).max()) / (float(ia.max()) + 1e-30)
    assert rel < 5e-3, f"piston leaked into intensity: {rel:.2e}"


def test_n3_broadband_input_clamps_na_proxy():
    """A broadband (speckle) input at fine dx pushes the 3-sigma input-NA
    estimate above 1.  Without the clamp na_proxy>1 makes every pupil ray
    grazing (N_dir=0) and the chart is empty; the clamp caps it at 0.999,
    warns, and still produces a finite non-zero field."""
    N, dx = 96, 0.6e-6
    lens = {
        'name': 'micro', 'aperture_diameter': 60e-6,
        'surfaces': [
            {'radius': 40e-6, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 30e-6},
            {'radius': -40e-6, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 30e-6},
        ],
        'thicknesses': [5e-6],
    }
    rng = np.random.default_rng(0)
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (N * dx / 4) ** 2)
    E = (amp * np.exp(1j * rng.uniform(-np.pi, np.pi, (N, N)))).astype(np.complex64)
    with pytest.warns(RuntimeWarning, match='exceeds 1'):
        out = la.apply_real_lens_maslov(
            E, prescription=lens, wavelength=LAM, dx=dx,
            integration_method='stationary_phase', output_subsample=1,
            ray_field_samples=12, ray_pupil_samples=12, poly_order=4, n_v2=16)
    assert np.isfinite(out).all()
    assert float(np.sum(np.abs(out) ** 2)) > 0.0


def test_n3_explicit_input_na_not_clamped_when_physical():
    """A physical explicit input_na (<1) must pass through unclamped -- the
    clamp only fires on the auto-estimate blowing past the horizon."""
    N, dx = 128, 60e-6
    E = _gauss(N, dx, w=1.0e-3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx,
            integration_method='stationary_phase', input_na=0.3, **_MASLOV_KW)
    assert not any('exceeds 1' in str(x.message) for x in w)


def test_carrier_none_regression_wellbehaved():
    """carrier=None (default) keeps the plane-wave reference: a collimated
    input is unaffected and the output is finite/power-reasonable."""
    N, dx = 512, 8e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(E, prescription=_carrier_lens(),
                                     wavelength=LAM, dx=dx, ray_subsample=2,
                                     parallel_amp=False, carrier=None)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0
