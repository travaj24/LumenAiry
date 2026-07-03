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
