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


def test_a1_auto_n_v2_resolves_demanding_default_quadrature():
    """A1 (v5.20): n_v2=None (the new default) auto-resolves the uniform
    quadrature from the fitted v2-oscillation estimate, so the *default*
    ``apply_real_lens_maslov`` call on a demanding tight-focus chart matches
    the well-resolved local_quadrature truth instead of speckling at a fixed
    n_v2=32.  This is the corrected A1 fix: the '67% gap' was never a
    local_quadrature bug -- it was the default quadrature being under-resolved.
    """
    N, dx = 128, 90e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
              poly_order=6)

    def il2(a, b):
        A, B = np.abs(a) ** 2, np.abs(b) ** 2
        m = B > 0.02 * B.max()
        return float(np.linalg.norm((A - B)[m])) / float(np.linalg.norm(B[m]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        truth = la.apply_real_lens_maslov(
            E, integration_method='local_quadrature', n_v2=24,
            local_n_samples=8, local_window_sigma=3.0, **kw)
        auto = la.apply_real_lens_maslov(E, **kw)          # default: n_v2=None
        old32 = la.apply_real_lens_maslov(E, n_v2=32, **kw)
    # The auto default now tracks the truth; the old fixed default did not.
    assert il2(auto, truth) < 5e-3, il2(auto, truth)
    assert il2(old32, truth) > 0.5, il2(old32, truth)


def test_a1_auto_n_v2_floor_is_byte_identical_to_32_on_weak_chart():
    """A low-NA / weakly-focusing chart needs << 32 v2 samples, so auto-
    resolution clamps to the floor (_N_V2_AUTO_MIN=32) and the default call is
    *byte-identical* to the historical explicit n_v2=32 -- no silent change for
    the configs the old default already handled."""
    N, dx = 96, 150e-6
    weak = {
        'name': 'weak', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 400e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -400e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [4e-3],
    }
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (6e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=weak, wavelength=LAM, dx=dx, output_subsample=1,
              ray_field_samples=12, ray_pupil_samples=12, poly_order=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        auto = la.apply_real_lens_maslov(E, **kw)
        pinned = la.apply_real_lens_maslov(E, n_v2=32, **kw)
    assert np.array_equal(auto, pinned)


def test_a1_explicit_n_v2_is_respected_not_auto_overridden():
    """Passing an explicit n_v2 pins the sampling exactly (reproducibility);
    auto-resolution only engages when n_v2 is left None."""
    N, dx = 96, 120e-6
    E = _gauss(N, dx, w=4e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=12, ray_pupil_samples=12,
              poly_order=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = la.apply_real_lens_maslov(E, n_v2=20, **kw)
        b = la.apply_real_lens_maslov(E, n_v2=20, **kw)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# v5.20 anamorphic (dy != dx) support
# --------------------------------------------------------------------------
def _rms_px(a):
    """Intensity-weighted rms width in pixels, per axis (sx, sy)."""
    I = np.abs(a) ** 2
    I = I / I.sum()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    cy = (I * yy).sum()
    cx = (I * xx).sum()
    return (np.sqrt((I * (xx - cx) ** 2).sum()),
            np.sqrt((I * (yy - cy) ** 2).sum()))


def test_anamorphic_square_pixels_byte_identical_to_before():
    """dy=None and dy=dx must be byte-identical (the anamorphic plumbing
    collapses exactly to the legacy square path when dx == dy -- no regression
    for the overwhelmingly common square-pixel call)."""
    N, dx = 128, 90e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
              poly_order=6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_none = la.apply_real_lens_maslov(E, **kw)
        m_eq = la.apply_real_lens_maslov(E, dy=dx, **kw)
    assert np.array_equal(m_none, m_eq)


def test_anamorphic_isotropic_beam_pixel_ellipticity_matches_analytic():
    """A physically circular beam through the symmetric singlet, rendered on an
    anamorphic grid (dy = 2*dx), must render with pixel-space y-rms = 1/2 the
    x-rms (each y-pixel spans 2x the physical distance).  Both apply_real_lens
    (independently anamorphic) and apply_real_lens_maslov must reproduce that
    exact 0.5 ratio and agree -- the decisive proof the Maslov dy threading is
    geometrically correct (not merely finite)."""
    N, ratio = 160, 2.0
    dx, dy = 40e-6, 40e-6 * ratio
    xa = (np.arange(N) - N // 2) * dx
    ya = (np.arange(N) - N // 2) * dy
    Xa, Ya = np.meshgrid(xa, ya)
    Eiso = np.exp(-(Xa ** 2 + Ya ** 2) / (1.5e-3) ** 2).astype(np.complex128)
    p = {
        'name': 'singlet', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 140e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -140e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [4e-3],
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ana = la.apply_real_lens(Eiso, prescription=p, wavelength=LAM,
                                 dx=dx, dy=dy)
        mas = la.apply_real_lens_maslov(
            Eiso, prescription=p, wavelength=LAM, dx=dx, dy=dy,
            output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
            poly_order=6)
    sxa, sya = _rms_px(ana)
    sxm, sym = _rms_px(mas)
    assert mas.shape == (N, N) and np.isfinite(mas).all()
    # both reproduce the geometric 1/ratio pixel-ellipticity
    assert abs((sya / sxa) - 1.0 / ratio) < 0.02
    assert abs((sym / sxm) - 1.0 / ratio) < 0.02
    # and maslov tracks analytic's ellipticity tightly
    assert abs((sym / sxm) - (sya / sxa)) / (sya / sxa) < 5e-3


def test_anamorphic_dy_actually_threaded_changes_output():
    """Sanity floor: propagating with dy=2*dx must NOT equal propagating the
    same array with dy=dx -- if dy were silently ignored the two would match."""
    N = 160
    dx, dy = 40e-6, 80e-6
    xa = (np.arange(N) - N // 2) * dx
    ya = (np.arange(N) - N // 2) * dy
    Xa, Ya = np.meshgrid(xa, ya)
    E = np.exp(-(Xa ** 2) / (1.2e-3) ** 2
               - (Ya ** 2) / (2.4e-3) ** 2).astype(np.complex128)
    p = {
        'name': 'singlet', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 140e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -140e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [4e-3],
    }
    kw = dict(prescription=p, wavelength=LAM, dx=dx, output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m_ana = la.apply_real_lens_maslov(E, dy=dy, **kw)
        m_sq = la.apply_real_lens_maslov(E, dy=dx, **kw)
    assert not np.allclose(m_ana, m_sq)


def test_anamorphic_rectangular_array_rejected():
    """A rectangular *array* (Ny != Nx) is still apply_real_lens territory and
    must raise the square-2D guard."""
    E = np.ones((64, 128), dtype=np.complex128)
    with pytest.raises(ValueError, match='square'):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=40e-6, dy=80e-6,
            output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
            poly_order=4)


def test_anamorphic_roi_rejected():
    """roi= with anamorphic pixels maps a square physical window to a
    rectangular pixel grid the square integrators can't take -> clean raise."""
    N, dx, dy = 128, 40e-6, 80e-6
    E = _gauss(N, dx, w=1.5e-3).astype(np.complex128)
    with pytest.raises(NotImplementedError, match='anamorphic'):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx, dy=dy,
            roi=(0.0, 0.0, 1e-3), ray_field_samples=14, ray_pupil_samples=14,
            poly_order=4)


# --------------------------------------------------------------------------
# v5.20 GPU (CuPy) quadrature integrator
# --------------------------------------------------------------------------
def _gpu_available():
    """True only when CuPy imports AND a GEMM (cublas) actually runs -- a bare
    ``import cupy`` succeeds on hosts whose cublas DLL is missing, so probe a
    real matmul (the quadrature path's core op)."""
    try:
        import cupy as cp
        _ = cp.asnumpy(cp.arange(4.0) @ cp.arange(4.0))
        return True
    except Exception:
        return False


_GPU = _gpu_available()
_gpu_skip = pytest.mark.skipif(not _GPU, reason='CuPy+GPU (cublas) unavailable')


def _il2c(a, b):
    m = np.abs(b) > 0.02 * np.abs(b).max()
    return float(np.linalg.norm((a - b)[m])) / (float(np.linalg.norm(b[m]))
                                                 + 1e-30)


@_gpu_skip
def test_gpu_quadrature_matches_cpu():
    """use_gpu=True runs the phase-space quadrature on the device and matches
    the CPU integrator (device BLAS/reduction order -> ~1e-5 or better), and
    returns a CuPy device array."""
    import cupy as cp
    N, dx = 160, 70e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
              poly_order=6, integration_method='quadrature')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu = la.apply_real_lens_maslov(E, **kw)
        gpu = la.apply_real_lens_maslov(E, use_gpu=True, **kw)
    assert isinstance(gpu, cp.ndarray)
    assert gpu.dtype == E.dtype and gpu.shape == (N, N)
    assert _il2c(cp.asnumpy(gpu), cpu) < 1e-5


@_gpu_skip
def test_gpu_cupy_input_array_triggers_gpu():
    """Passing a CuPy input array (no use_gpu flag) routes to the GPU path and
    returns a device array matching the CPU result."""
    import cupy as cp
    N, dx = 128, 80e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=12, ray_pupil_samples=12,
              poly_order=6, integration_method='quadrature')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu = la.apply_real_lens_maslov(E, **kw)
        gpu = la.apply_real_lens_maslov(cp.asarray(E), **kw)
    assert isinstance(gpu, cp.ndarray)
    assert _il2c(cp.asnumpy(gpu), cpu) < 1e-5


@_gpu_skip
def test_gpu_anamorphic_matches_cpu():
    """The GPU quadrature honours anamorphic pixels (dy != dx) and matches the
    CPU anamorphic result."""
    import cupy as cp
    N, dx, dy = 160, 40e-6, 80e-6
    xa = (np.arange(N) - N // 2) * dx
    ya = (np.arange(N) - N // 2) * dy
    Xa, Ya = np.meshgrid(xa, ya)
    E = np.exp(-(Xa ** 2 + Ya ** 2) / (1.5e-3) ** 2).astype(np.complex128)
    p = {
        'name': 'singlet', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 140e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -140e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [4e-3],
    }
    kw = dict(prescription=p, wavelength=LAM, dx=dx, dy=dy, output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=6)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu = la.apply_real_lens_maslov(E, **kw)
        gpu = la.apply_real_lens_maslov(E, use_gpu=True, **kw)
    assert _il2c(cp.asnumpy(gpu), cpu) < 1e-5


@_gpu_skip
@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_gpu_asymptotic_evaluators_match_cpu(method):
    """The asymptotic evaluators run on the GPU (fused CuPy RawKernel for the
    per-pixel Chebyshev value+derivs) and match the CPU integrator; they return
    a CuPy device array.  (Earlier these raised under use_gpu; now supported.)"""
    import cupy as cp
    N, dx = 160, 70e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex128)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
              poly_order=6, integration_method=method,
              local_n_samples=8, local_window_sigma=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu = la.apply_real_lens_maslov(E, **kw)
        gpu = la.apply_real_lens_maslov(E, use_gpu=True, **kw)
    assert isinstance(gpu, cp.ndarray)
    assert gpu.dtype == E.dtype and gpu.shape == (N, N)
    assert _il2c(cp.asnumpy(gpu), cpu) < 1e-6


@_gpu_skip
@pytest.mark.parametrize('method', ['stationary_phase', 'local_quadrature'])
def test_gpu_asymptotic_complex64_and_anamorphic(method):
    """The GPU asymptotic evaluators preserve complex64 and honour anamorphic
    pixels, matching the CPU integrator in both cases."""
    import cupy as cp
    N = 160
    # complex64, square
    dx = 70e-6
    E = _gauss(N, dx, w=3e-3).astype(np.complex64)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
              poly_order=6, integration_method=method,
              local_n_samples=8, local_window_sigma=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cpu64 = la.apply_real_lens_maslov(E, **kw)
        gpu64 = la.apply_real_lens_maslov(cp.asarray(E), **kw)
    assert gpu64.dtype == np.complex64
    assert _il2c(cp.asnumpy(gpu64), cpu64) < 1e-4
    # anamorphic dy = 2*dx
    dxa, dya = 40e-6, 80e-6
    xa = (np.arange(N) - N // 2) * dxa
    ya = (np.arange(N) - N // 2) * dya
    Xa, Ya = np.meshgrid(xa, ya)
    Ean = np.exp(-(Xa ** 2 + Ya ** 2) / (1.5e-3) ** 2).astype(np.complex128)
    p = {
        'name': 'singlet', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 140e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -140e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [4e-3],
    }
    akw = dict(prescription=p, wavelength=LAM, dx=dxa, dy=dya,
               output_subsample=1, ray_field_samples=14, ray_pupil_samples=14,
               poly_order=6, integration_method=method,
               local_n_samples=8, local_window_sigma=3.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        an_c = la.apply_real_lens_maslov(Ean, **akw)
        an_g = la.apply_real_lens_maslov(Ean, use_gpu=True, **akw)
    assert _il2c(cp.asnumpy(an_g), an_c) < 1e-6


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
    """Collimated input must trigger NO F4-class (tilt-discriminator)
    warning.  v5.25.0: assert the F4 phrases are absent rather than that
    the text is EMPTY -- the hammer-audit H3 exit-NA Nyquist guard now
    legitimately fires on this configuration (NA_exit = 0.126 needs
    dx <= 5.2 um; this grid is 6 um), and pinning "no warnings at all"
    would wrongly forbid unrelated future guards."""
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2).astype(np.complex64)
    txt = _f4_warn_text(E)
    assert 'for tilt-sensitive analyses' not in txt
    assert 'INCOHERENT' not in txt


# --------------------------------------------------------------------------
# T-P1 (audit perf follow-up): prepared traced lens caches the input-
# independent screen; each call is one apply_real_lens + one complex multiply.
# --------------------------------------------------------------------------
from lumenairy.elements._lens_traced import (  # noqa: E402
    apply_real_lens_traced_multi,
    prepare_real_lens_traced,
)


def _prep_lens():
    return {
        'name': 'p', 'aperture_diameter': 6e-3,
        'surfaces': [
            {'radius': 60e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
            {'radius': -60e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 3e-3},
        ],
        'thicknesses': [3e-3],
    }


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_tp1_prepared_matches_direct(dtype):
    """prepare_real_lens_traced()(E) must equal a direct apply_real_lens_traced
    with the same prepared-mode settings (full-grid Newton, no tilt-aware,
    sequential amp): exact at complex128, float32-ULP at complex64 (the
    cached screen is complex128 so the prepared path is marginally MORE
    accurate)."""
    N, dx = 192, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = (np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2)
         * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.01) * X)).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = apply_real_lens_traced(
            E, prescription=_prep_lens(), wavelength=LAM, dx=dx,
            ray_subsample=4, parallel_amp=False, newton_amp_mask_rel=0.0,
            tilt_aware_rays=False)
        prep = prepare_real_lens_traced(
            prescription=_prep_lens(), wavelength=LAM, dx=dx, N=N,
            ray_subsample=4)
        out = prep(E)
    assert out.shape == direct.shape and out.dtype == E.dtype
    m = np.abs(direct) > 0.02 * np.abs(direct).max()
    rel = float(np.linalg.norm((out - direct)[m])) / (float(np.linalg.norm(direct[m])) + 1e-30)
    tol = 1e-12 if dtype == np.complex128 else 1e-5
    assert rel < tol, f"{dtype.__name__}: prepared vs direct {rel:.2e}"


def test_tp1_reuse_is_input_independent():
    """One prepared object applied to two DIFFERENT inputs must equal two
    direct calls -- i.e. the cached screen is genuinely input-independent."""
    N, dx = 160, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E1 = np.exp(-(X ** 2 + Y ** 2) / (1.0e-3) ** 2).astype(np.complex128)
    E2 = (np.exp(-((X - 0.4e-3) ** 2 + Y ** 2) / (0.7e-3) ** 2)
          * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.015) * Y)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep = prepare_real_lens_traced(prescription=_prep_lens(),
                                        wavelength=LAM, dx=dx, N=N, ray_subsample=4)
        kw = dict(prescription=_prep_lens(), wavelength=LAM, dx=dx,
                  ray_subsample=4, parallel_amp=False, newton_amp_mask_rel=0.0,
                  tilt_aware_rays=False)
        for E in (E1, E2):
            d = apply_real_lens_traced(E, **kw)
            p = prep(E)
            rel = float(np.linalg.norm(p - d)) / (float(np.linalg.norm(d)) + 1e-30)
            assert rel < 1e-12, f"reuse mismatch {rel:.2e}"


def _two_emitters(N, dx):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E1 = np.exp(-((X + 0.5e-3) ** 2 + Y ** 2) / (0.8e-3) ** 2).astype(np.complex128)
    E2 = np.exp(-((X - 0.5e-3) ** 2 + Y ** 2) / (0.8e-3) ** 2).astype(np.complex128)
    return E1, E2


def test_multi_single_emitter_is_exact():
    """multi([E]) with one emitter must equal a single traced pass with the
    same forced settings (full-grid Newton, no tilt-aware, sequential amp)."""
    N, dx = 160, 6e-6
    E1, _ = _two_emitters(N, dx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = apply_real_lens_traced_multi(
            [E1], prescription=_prep_lens(), wavelength=LAM, dx=dx,
            carriers=30e-3, ray_subsample=4)
        d = apply_real_lens_traced(
            E1, prescription=_prep_lens(), wavelength=LAM, dx=dx, carrier=30e-3,
            ray_subsample=4, newton_amp_mask_rel=0.0, tilt_aware_rays=False,
            preserve_input_phase=True, parallel_amp=False)
    rel = float(np.linalg.norm(m - d)) / (float(np.linalg.norm(d)) + 1e-30)
    assert rel < 1e-12, f"single-emitter multi not exact: {rel:.2e}"


def test_multi_reuse_matches_noreuse_for_shared_carrier():
    """With a shared explicit carrier the prepared-screen reuse path must equal
    the full per-emitter path (byte-identical) and equal the sum of two direct
    traced calls."""
    N, dx = 160, 6e-6
    E1, E2 = _two_emitters(N, dx)
    kw = dict(prescription=_prep_lens(), wavelength=LAM, dx=dx,
              carriers=30e-3, ray_subsample=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reuse = apply_real_lens_traced_multi([E1, E2], reuse_prepared=True, **kw)
        noreuse = apply_real_lens_traced_multi([E1, E2], reuse_prepared=False, **kw)
        dkw = dict(prescription=_prep_lens(), wavelength=LAM, dx=dx,
                   carrier=30e-3, ray_subsample=4, newton_amp_mask_rel=0.0,
                   tilt_aware_rays=False, preserve_input_phase=True,
                   parallel_amp=False)
        direct = (apply_real_lens_traced(E1, **dkw)
                  + apply_real_lens_traced(E2, **dkw))
    assert np.array_equal(reuse, noreuse), "reuse != no-reuse"
    rel = float(np.linalg.norm(reuse - direct)) / (float(np.linalg.norm(direct)) + 1e-30)
    assert rel < 1e-12, f"multi != sum of direct calls: {rel:.2e}"


def test_multi_captures_traced_nonlinearity():
    """On an aberrated lens with DIVERGENT, OVERLAPPING emitter congruences the
    traced model is strongly non-linear, so the per-emitter coherent sum must
    DIFFER substantially from feeding the combined field to a single traced pass
    -- the effect the mode exists to handle.  The emitters are point sources
    free-space-propagated to the lens plane so their beams overlap there (a
    collimated, non-overlapping pair would show no effect -- separate regime)."""
    from lumenairy.propagators.asm import angular_spectrum_propagate as _asm
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)

    def _src(x0):
        s = np.exp(-((X - x0) ** 2 + Y ** 2) / (0.2e-3) ** 2).astype(np.complex128)
        return _asm(s, 18e-3, LAM, dx)     # diverge to the lens plane
    E1, E2 = _src(-1.0e-3), _src(+1.0e-3)
    aber = {  # strong plano-convex -> large ray aberration -> traced != analytic
        'name': 'pc', 'aperture_diameter': 9e-3,
        'surfaces': [
            {'radius': 12e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 4.5e-3},
            {'radius': -1e9, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 4.5e-3},
        ],
        'thicknesses': [3e-3],
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        multi = apply_real_lens_traced_multi(
            [E1, E2], prescription=aber, wavelength=LAM, dx=dx,
            carriers='auto', ray_subsample=4)
        naive = apply_real_lens_traced(
            E1 + E2, prescription=aber, wavelength=LAM, dx=dx, carrier='auto',
            ray_subsample=4, newton_amp_mask_rel=0.0, tilt_aware_rays=False,
            parallel_amp=False)
    m = np.abs(multi) > 0.02 * np.abs(multi).max()
    rel = float(np.linalg.norm((naive - multi)[m])) / (float(np.linalg.norm(multi[m])) + 1e-30)
    # v5.25.1 (hammer H6): the carrier entrance-eikonal fix corrected BOTH
    # the multi and naive paths, shifting the measured discrepancy from
    # ~0.2+ to ~0.19.  The regression this test guards -- multi collapsing
    # onto naive (rel ~ 0) -- is still decisively excluded at 0.1.
    assert rel > 0.1, f"expected large traced non-linearity, got {rel:.2e}"


def test_multi_input_validation():
    with pytest.raises(ValueError, match='empty'):
        apply_real_lens_traced_multi([], prescription=_prep_lens(),
                                     wavelength=LAM, dx=6e-6)
    E = np.ones((64, 64), dtype=np.complex128)
    Ebad = np.ones((32, 32), dtype=np.complex128)
    with pytest.raises(ValueError, match='shape'):
        apply_real_lens_traced_multi([E, Ebad], prescription=_prep_lens(),
                                     wavelength=LAM, dx=6e-6)
    with pytest.raises(ValueError, match='length'):
        apply_real_lens_traced_multi([E, E], prescription=_prep_lens(),
                                     wavelength=LAM, dx=6e-6,
                                     carriers=[1e-3, 2e-3, 3e-3])


def test_tp2_fit_inversion_matches_newton():
    """T-P2: inversion_method='fit' (scattered Chebyshev inverse-map fit,
    hull-masked) must reproduce the Newton inversion to a small relative error
    over the illuminated region, at both full-res and subsampled output.  It
    is opt-in; 'newton' remains the default.

    SCORED WITH ``inverse_map=False`` (2026-08-13), because that is what makes
    the ``newton`` arm a Newton inversion.  The inverse-characteristic
    per-pixel evaluator (``TRACED_INVERSE_MAP``, shipped ``True`` since
    ``FIX_G8_PROBE_2026_08_12``) is gated on ``inversion_method == 'newton'``
    BY DESIGN -- ``_lens_traced.py`` :8466 and the gate's own comment at
    :10308, "the 'fit' path is already a per-pixel exit polynomial" -- so at
    the shipped default this comparison is model-vs-'fit', two DIFFERENT
    per-pixel exit representations, and not the fit-vs-Newton comparison the
    test names.  Measured on this fixture:

        sub   inverse_map=False   SHIPPED default   map built?
          1        3.1849e-06        3.1849e-06     neither (gate needs sub>1)
          4        3.1833e-06        6.1318e-02     'newton' yes, 'fit' no

    Nothing here is wrong: at ``sub = 4`` the model supplies the exact
    per-pixel OPL where the coarse Newton had to upsample a 4x-subsampled
    lattice, so the 6.1e-02 is the accuracy the model ADDS on that arm and the
    'fit' arm not following it is the gate working as documented.  The
    assertion and its 1e-3 bar are unchanged, and ``sub = 1`` -- where the gate
    refuses on both arms -- reads the same number under both flags, which is
    the control that says the scoping changed nothing else."""
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = (np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2)
         * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.008) * X)).astype(np.complex128)
    lens = {
        'name': 's', 'aperture_diameter': 8e-3,
        'surfaces': [
            {'radius': 60e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 4e-3},
            {'radius': -60e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 4e-3},
        ],
        'thicknesses': [3e-3],
    }
    for sub in (1, 4):
        kw = dict(prescription=lens, wavelength=LAM, dx=dx, ray_subsample=sub,
                  parallel_amp=False, newton_amp_mask_rel=0.0,
                  inverse_map=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            newton = apply_real_lens_traced(E, inversion_method='newton', **kw)
            fit = apply_real_lens_traced(E, inversion_method='fit',
                                         newton_poly_order=8, **kw)
        msk = np.abs(newton) > 0.02 * np.abs(newton).max()
        rel = float(np.linalg.norm((fit - newton)[msk])) / (float(np.linalg.norm(newton[msk])) + 1e-30)
        assert rel < 1e-3, f"sub={sub}: fit vs newton {rel:.2e}"


from lumenairy.elements._lens_real import prepare_real_lens  # noqa: E402


def _multi_lens(nsurf=6, ap=8e-3):
    surfs = []
    for i in range(nsurf):
        R = 60e-3 if i % 2 == 0 else -60e-3
        surfs.append({
            'radius': R, 'conic': 0.0,
            'glass_before': 'air' if i % 2 == 0 else 'N-BK7',
            'glass_after': 'N-BK7' if i % 2 == 0 else 'air',
            'semi_diameter': ap / 2})
    return {'name': 'm', 'aperture_diameter': ap, 'surfaces': surfs,
            'thicknesses': [2e-3] * (nsurf - 1)}


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_ap1_prepared_analytic_matches_direct(dtype):
    """A-P1: prepare_real_lens()(E) must equal apply_real_lens(E) to machine
    precision on the default ASM / conic path (the cached screens are the same
    exp(-i k0 opd) the direct path recomputes; only exp reassociation differs
    -> ~3e-15 at complex128, float32-ULP at complex64)."""
    N, dx = 384, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = (np.exp(-(X ** 2 + Y ** 2) / (2e-3) ** 2)
         * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.006) * X)).astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        direct = la.apply_real_lens(E, prescription=_multi_lens(6),
                                    wavelength=LAM, dx=dx)
        prep = prepare_real_lens(prescription=_multi_lens(6), wavelength=LAM,
                                 dx=dx, N=N)
        out = prep(E)
    assert out.shape == direct.shape and out.dtype == E.dtype
    rel = float(np.abs(out - direct).max()) / (float(np.abs(direct).max()) + 1e-30)
    tol = 1e-12 if dtype == np.complex128 else 1e-5
    assert rel < tol, f"{dtype.__name__}: prepared vs direct {rel:.2e}"


def test_ap1_reuse_is_input_independent():
    N, dx = 256, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E1 = np.exp(-(X ** 2 + Y ** 2) / (1.5e-3) ** 2).astype(np.complex128)
    E2 = (np.exp(-((X - 0.5e-3) ** 2 + Y ** 2) / (1.0e-3) ** 2)
          * np.exp(1j * (2 * np.pi / LAM) * np.sin(0.01) * Y)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep = prepare_real_lens(prescription=_multi_lens(4), wavelength=LAM,
                                 dx=dx, N=N)
        for E in (E1, E2):
            d = la.apply_real_lens(E, prescription=_multi_lens(4),
                                   wavelength=LAM, dx=dx)
            rel = float(np.abs(prep(E) - d).max()) / (float(np.abs(d).max()) + 1e-30)
            assert rel < 1e-12, f"reuse mismatch {rel:.2e}"


def test_ap1_rejects_unsupported_configs():
    for bad in ({'decenter': (1e-4, 0.0)}, {'tilt': (1e-3, 0.0)},
                {'freeform_type': 'xy_polynomial'}, {'radius_y': 50e-3},
                {'clear_aperture': 3e-3}):
        p = _multi_lens(2)
        p['surfaces'][0].update(bad)
        with pytest.raises(NotImplementedError):
            prepare_real_lens(prescription=p, wavelength=LAM, dx=6e-6, N=64)
    p = _multi_lens(3)
    p['stop_index'] = 1
    with pytest.raises(NotImplementedError):
        prepare_real_lens(prescription=p, wavelength=LAM, dx=6e-6, N=64)


def test_ap1_shape_mismatch_raises():
    prep = prepare_real_lens(prescription=_multi_lens(2), wavelength=LAM,
                             dx=6e-6, N=128)
    with pytest.raises(ValueError, match='shape'):
        prep(np.ones((64, 64), dtype=np.complex128))


def test_tp1_rejects_auto_carrier():
    with pytest.raises(ValueError, match="auto"):
        prepare_real_lens_traced(prescription=_prep_lens(), wavelength=LAM,
                                 dx=6e-6, N=128, carrier='auto')


def test_tp1_shape_mismatch_raises():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep = prepare_real_lens_traced(prescription=_prep_lens(),
                                        wavelength=LAM, dx=6e-6, N=128, ray_subsample=4)
    with pytest.raises(ValueError, match='shape'):
        prep(np.ones((64, 64), dtype=np.complex128))


# --------------------------------------------------------------------------
# C1 (Phase-C perf): prepared traced screen + EXPLICIT carrier.
#   The screen is built on a flat ``ones`` placeholder, so the F1
#   collimation guard must not judge it against a carrier (a scalar/ndarray
#   carrier makes ``ones`` look strongly non-collimated even though the real
#   reuse fields carry exactly that congruence).  Two failure modes fixed:
#   (1) a spurious noncollimated warning at the default ``'warn'``; and
#   (2) the latent ``'delegate'`` bug -- the guard would hand off to
#   apply_real_lens (which ignores return_screen) and cache a GARBAGE screen.
# --------------------------------------------------------------------------
def _c1_carrier_lens():
    # small-aperture biconvex whose ``ones``-placeholder carrier residual
    # exceeds the F1 threshold at this grid (so the guard WOULD fire pre-fix).
    return {
        'name': 'c1', 'aperture_diameter': 3e-3,
        'surfaces': [
            {'radius': 40e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 1.5e-3},
            {'radius': -40e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 1.5e-3},
        ],
        'thicknesses': [3e-3],
    }


def test_c1_prepared_scalar_carrier_no_spurious_warning():
    """Preparing with an explicit scalar conjugate must NOT emit the F1
    noncollimated warning: the ``ones`` placeholder is not the beam, so the
    guard cannot judge it against the carrier (the real reuse fields carry
    that divergence).  Pre-fix this warned ('residual angular spread ...
    exceeds the collimated-reference validity threshold')."""
    N, dx, s = 512, 6e-6, 25e-3
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        prepare_real_lens_traced(prescription=_c1_carrier_lens(),
                                 wavelength=LAM, dx=dx, N=N, carrier=s,
                                 ray_subsample=8)
    spurious = [str(w.message) for w in wlist
                if 'collimated-reference' in str(w.message)]
    assert not spurious, f"spurious F1 warning(s) during prepare: {spurious}"


def test_c1_prepared_scalar_carrier_delegate_screen_not_garbage():
    """With an explicit carrier the internal guard is forced 'off', so even
    ``on_noncollimated='delegate'`` caches a valid UNIT-MODULUS traced screen
    -- not the delegated ``apply_real_lens(ones)`` field (|.| up to ~1.4),
    which ignores return_screen.  Byte-identical to the default-guard prepare."""
    N, dx, s = 512, 6e-6, 25e-3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prep_default = prepare_real_lens_traced(
            prescription=_c1_carrier_lens(), wavelength=LAM, dx=dx, N=N,
            carrier=s, ray_subsample=8)
        prep_delegate = prepare_real_lens_traced(
            prescription=_c1_carrier_lens(), wavelength=LAM, dx=dx, N=N,
            carrier=s, ray_subsample=8, on_noncollimated='delegate')
    # a genuine traced screen is a pure phase (|screen| == 1 in-domain, 0
    # outside); the garbage delegated screen has |.| ~ 1.4 somewhere.
    assert float(np.abs(prep_delegate.screen).max()) < 1.0 + 1e-9
    # both now force the guard 'off' internally, so the screens match to the
    # traced path's own ~1e-14 FFT-threading nondeterminism (not garbage).
    rel = float(np.abs(prep_default.screen - prep_delegate.screen).max())
    assert rel < 1e-10, f"delegate screen != default-guard screen: {rel:.2e}"


# Constant-index glass for the C1 diverging-input ABCD oracle (independent
# of the traced implementation; same singlet as the H6 hammer test).
_C1_N_GLASS = 1.5168
_C1_R1, _C1_R2, _C1_TC = 51.68e-3, -51.68e-3, 5e-3
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_C1_ABCD_GLASS': lambda wl: _C1_N_GLASS}


def _c1_singlet():
    return {
        'wavelength': LAM, 'aperture_diameter': 20e-3,
        'surfaces': [
            {'radius': _C1_R1, 'thickness': _C1_TC, 'glass_before': 'air',
             'glass_after': '_C1_ABCD_GLASS', 'semi_diameter': 10e-3},
            {'radius': _C1_R2, 'thickness': 0.0,
             'glass_before': '_C1_ABCD_GLASS', 'glass_after': 'air',
             'semi_diameter': 10e-3},
        ],
        'thicknesses': [_C1_TC], 'stop_index': 0,
    }


def _c1_q_trace_image(R_in, w_L):
    """ABCD Gaussian q-trace: entry vertex -> waist location past the exit
    vertex.  Fully independent of the traced implementation."""
    def refr(R, n1, n2):
        return np.array([[1.0, 0.0], [-(n2 - n1) / (R * n2), n1 / n2]])

    def trans(d):
        return np.array([[1.0, d], [0.0, 1.0]])
    M = (refr(_C1_R2, _C1_N_GLASS, 1.0) @ trans(_C1_TC)
         @ refr(_C1_R1, 1.0, _C1_N_GLASS))
    q_inv = 1.0 / R_in - 1j * LAM / (np.pi * w_L ** 2)
    q0 = 1.0 / q_inv
    q1 = (M[0, 0] * q0 + M[0, 1]) / (M[1, 0] * q0 + M[1, 1])
    return float(-q1.real)


def _c1_ee100(E_exit, dx, z):
    E = la.angular_spectrum_propagate(E_exit, z, LAM, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    return float(I[r <= 100e-6].sum() / I.sum())


def test_c1_prepared_scalar_carrier_focuses_diverging_input():
    """The prepared screen for an EXPLICIT scalar conjugate carries the H6
    entrance eikonal, so it focuses a diverging Gaussian AT the ABCD image
    plane (EE(100um) > 0.9), NOT at the collimated f (~0.02) -- and is
    byte-close to the direct carrier-referenced traced call.  This is the
    121-class per-group reuse (one KNOWN conjugate, many fields)."""
    N, dx, w_L, R_in = 2048, 5e-6, 3e-3, 150e-3
    z_img = _c1_q_trace_image(R_in, w_L)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    k = 2 * np.pi / LAM
    E0 = (np.exp(-r_sq / w_L ** 2)
          * np.exp(1j * k * r_sq / (2.0 * R_in))).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')      # H3 guard may advise finer dx
        prep = prepare_real_lens_traced(
            prescription=_c1_singlet(), wavelength=LAM, dx=dx, N=N,
            carrier=R_in)
        E_prep = prep(E0)
        E_direct = apply_real_lens_traced(
            E0, prescription=_c1_singlet(), wavelength=LAM, dx=dx,
            carrier=R_in, newton_amp_mask_rel=0.0, tilt_aware_rays=False,
            preserve_input_phase=True, parallel_amp=False)
    ee_img = _c1_ee100(E_prep, dx, z_img)
    ee_f = _c1_ee100(E_prep, dx, 49.163e-3)      # collimated BFL
    assert ee_img > 0.9, (
        f"prepared+carrier EE(100um)@z_img={z_img*1e3:.2f}mm is {ee_img:.3f} "
        f"-- the H6 entrance eikonal must flow through the prepared screen")
    assert ee_img > ee_f, (
        f"focus must sit at z_img ({ee_img:.3f}) not the collimated f "
        f"({ee_f:.3f})")
    rel = float(np.linalg.norm(E_prep - E_direct)) / (
        float(np.linalg.norm(E_direct)) + 1e-30)
    assert rel < 1e-12, f"prepared vs direct (carrier): {rel:.2e}"


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
    # v5.25.1 (hammer H6): _compute_carrier now also returns the
    # eikonal evaluator w_fn (the entrance-plane W the ray OPL must add).
    _W, grad_fn, _w_fn = _compute_carrier('auto', E, LAM, dx, X, Y)
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


def test_c4_input_tilt_stats_matches_both_former_sites():
    """C4 perf dedup: the single shared _input_tilt_stats pass reproduces BOTH
    former duplicate computations byte-for-byte -- its tilt_rms equals the
    carrier=None noncollimated-guard residual (_carrier_residual_rms(E, None)),
    and its coherence_ratio is the F4 single-beam-vs-incoherent discriminator.
    This is what makes removing the second full-grid phase-gradient pass
    (~8-9% of the traced runtime at N=4k) byte-identical."""
    from lumenairy.elements._lens_traced import (
        _carrier_residual_rms,
        _input_tilt_stats,
    )
    N, dx = 384, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    k0 = 2 * np.pi / LAM
    g = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2)
    cases = {
        'collimated': g,
        'coherent': g * np.exp(1j * k0 * np.sin(0.01) * X),
        'incoherent': (g * np.exp(1j * k0 * np.sin(0.02) * X)
                       + g * np.exp(-1j * k0 * np.sin(0.02) * X)),
        'diverging': g * np.exp(1j * k0 * (X ** 2 + Y ** 2) / (2 * 20e-3)),
    }
    for name, E in cases.items():
        E = E.astype(np.complex128)
        st = _input_tilt_stats(E, LAM, dx)
        resid = _carrier_residual_rms(E, None, LAM, dx)
        assert st is not None, name
        # BYTE-identical to the value the noncollimated guard used to compute
        # separately (both are pure -- no FFT nondeterminism).
        assert st[0] == resid, f"{name}: tilt_rms {st[0]!r} != residual {resid!r}"
        assert 0.0 <= st[1] <= 1.0 + 1e-12, name
    # coherence discriminates a single-beam tilt (~1) from an opposed-tilt
    # (multi-beam) field (<<1) -- the F4 branch selector the tilt warning uses.
    coh = _input_tilt_stats(cases['coherent'].astype(np.complex128), LAM, dx)[1]
    inc = _input_tilt_stats(cases['incoherent'].astype(np.complex128), LAM, dx)[1]
    assert coh >= 0.5 > inc, (coh, inc)
    # degenerate fields yield None (silently skip the warning, as before)
    assert _input_tilt_stats(np.zeros((8, 8), np.complex128), LAM, dx) is None


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


def _flat_prism(amp, ap=24e-3, R_flat=1e9):
    """A wedge/prism: FLAT surfaces (no power -> coarse field stays well-
    resolved) with an xy_polynomial (1,0) tilt term (honored by the trace)
    that deviates the beam.  This drives a large REAL s2 linear-OPD slope
    (_lin[1]) while keeping the coarse integral resolvable -- isolating the
    fine-grid post-multiply from N2 under-resolution."""
    return {
        'name': 'prism', 'aperture_diameter': ap,
        'surfaces': [
            {'radius': R_flat, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': ap / 2,
             'freeform_type': 'xy_polynomial', 'xy_coeffs': {(1, 0): amp}},
            {'radius': R_flat, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': ap / 2},
        ],
        'thicknesses': [4e-3],
    }


def test_n4_freeform_large_slope_is_subsample_invariant():
    """A freeform PRISM produces a real _lin[1] ~ 15.6 waves -- nearly 2x the
    coarse Nyquist at output_subsample=6 -- so the s2 post-multiply MUST run
    on the fine grid.  Applied there, the recovered output tilt is sign- and
    magnitude-invariant between output_subsample 1 and 6; a coarse-grid
    multiply would alias/flip it (adversarial-review verification, Claim C:
    fine grid recovers 0.42 cyc/pix vs coarse-then-zoom 0.10).  This is the
    reachable large-real-slope regime that refuted the earlier
    'slope always ~0' characterisation."""
    N = 192
    dx = 24e-3 / N
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (7e-3) ** 2).astype(np.complex64)
    kw = dict(prescription=_flat_prism(4e-3), wavelength=LAM, dx=dx,
              integration_method='quadrature', ray_field_samples=12,
              ray_pupil_samples=12, poly_order=4, n_v2=16,
              collimated_input=True, extract_linear_phase=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c1 = _spectral_centroid_fx(
            la.apply_real_lens_maslov(E, output_subsample=1, **kw), dx)
        c6 = _spectral_centroid_fx(
            la.apply_real_lens_maslov(E, output_subsample=6, **kw), dx)
    # a genuine, large output tilt (not a near-DC artefact)
    assert abs(c1) > 500.0, f"prism tilt too weak to exercise the path: {c1:.1f}"
    assert np.sign(c1) == np.sign(c6), f"tilt sign flipped: {c1:.1f} vs {c6:.1f}"
    assert abs(c6 - c1) / abs(c1) < 0.06, f"{c1:.1f} vs {c6:.1f}"


def test_n3_nan_input_na_raises_clear_error():
    """input_na=NaN must fail fast with a clear ValueError, not slip past the
    (NaN-blind) na_proxy>=1 clamp and die later with a misleading
    '0 rays survived' TIR message (adversarial-review Claim B)."""
    N, dx = 128, 60e-6
    E = _gauss(N, dx, w=1.0e-3)
    with pytest.raises(ValueError, match='finite'):
        la.apply_real_lens_maslov(
            E, prescription=_singlet(), wavelength=LAM, dx=dx,
            integration_method='stationary_phase', input_na=float('nan'),
            **_MASLOV_KW)


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


# --------------------------------------------------------------------------
# Deferred follow-ups (audit remediation): integrator memory-banding.
#   stationary_phase: _opd_and_derivs pixel-banded (was not pixel-chunked ->
#     OOM at scale).  F2: quadrature builds only a per-output-row-band G
#     instead of the full (N_out^2, M) design matrix.
# Both must be numerically equivalent to the unbanded path.
# --------------------------------------------------------------------------
import lumenairy.elements.lenses_maslov as _lm  # noqa: E402


def test_mp4_numba_kernel_matches_numpy_reference():
    """M-P4: the Numba 4-var Chebyshev value+derivative kernel must reproduce
    the NumPy reference for all six outputs (f, df3, df4, d2_33, d2_34, d2_44)
    to ULP -- same 3-term recurrences, only the term-reduction order differs."""
    kern = _lm._get_cheb4d_numba()
    if kern is None:
        import pytest as _pt
        _pt.skip("numba not available")
    rng = np.random.default_rng(1)
    P = 6
    mi = _lm._multi_indices_total_degree(4, P)
    K = [np.array([t[j] for t in mi], dtype=np.int64) for j in range(4)]
    coef = rng.standard_normal(len(mi))
    us = [rng.uniform(-1.0, 1.0, 3000) for _ in range(4)]
    ref = _lm._opd6_numpy(coef, *K, *us, P)
    old = _lm._MASLOV_USE_NUMBA
    try:
        _lm._MASLOV_USE_NUMBA = True
        got = _lm._opd6(coef, *K, *us, P)
    finally:
        _lm._MASLOV_USE_NUMBA = old
    for r, g in zip(ref, got):
        rel = float(np.abs(r - g).max()) / (float(np.abs(r).max()) + 1e-30)
        assert rel < 1e-12, f"kernel output diverges: {rel:.2e}"


def test_mp4_end_to_end_numba_equals_numpy():
    """apply_real_lens_maslov must give the same field with the Numba kernel
    (default) and the NumPy reference path."""
    if _lm._get_cheb4d_numba() is None:
        import pytest as _pt
        _pt.skip("numba not available")
    N, dx = 192, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='stationary_phase', output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=6, n_v2=24)
    old = _lm._MASLOV_USE_NUMBA
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _lm._MASLOV_USE_NUMBA = True
            nb = la.apply_real_lens_maslov(E, **kw)
            _lm._MASLOV_USE_NUMBA = False
            npx = la.apply_real_lens_maslov(E, **kw)
    finally:
        _lm._MASLOV_USE_NUMBA = old
    rel = float(np.abs(nb - npx).max()) / (float(np.abs(npx).max()) + 1e-30)
    assert rel < 1e-5, f"numba vs numpy end-to-end {rel:.2e}"


def test_mp6_roi_matches_full_slice():
    """M-P6: a ROI evaluation must return exactly the corresponding sub-window
    of the full-grid field (byte-identical -- each output pixel is integrated
    independently), for both an on-axis and an off-axis window, while
    computing only roi_n^2 pixels."""
    N, dx = 192, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='stationary_phase', output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=6, n_v2=24)
    roi_n = 32
    hw = roi_n * dx / 2
    off = (N - roi_n) // 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full = la.apply_real_lens_maslov(E, normalize_output='none', **kw)
        r_on = la.apply_real_lens_maslov(E, roi=(0.0, 0.0, hw), **kw)
        k0x, k0y = 24, -16
        r_off = la.apply_real_lens_maslov(E, roi=(k0x * dx, k0y * dx, hw), **kw)
    assert r_on.shape == (roi_n, roi_n)
    assert np.array_equal(r_on, full[off:off + roi_n, off:off + roi_n]), "on-axis ROI"
    assert np.array_equal(
        r_off, full[k0y + off:k0y + off + roi_n, k0x + off:k0x + off + roi_n]
    ), "off-axis ROI"


def test_stationary_phase_pixel_banding_matches_unbanded():
    """Banding _opd_and_derivs by pixel must reproduce the unbanded result:
    byte-identical for a realistic band (reduction shape preserved), and
    within float32 ULP for the degenerate 1-pixel band (np.sum reduces a
    different array shape -> ULP reordering only, not a logic change)."""
    N, dx = 160, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='stationary_phase', output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=4, n_v2=24)
    old = _lm._SP_PIXEL_CHUNK
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _lm._SP_PIXEL_CHUNK = None
            ref = la.apply_real_lens_maslov(E, **kw)
            _lm._SP_PIXEL_CHUNK = 997   # forces many bands, keeps reduce shape
            banded = la.apply_real_lens_maslov(E, **kw)
            _lm._SP_PIXEL_CHUNK = 1     # maximal banding (degenerate reduce)
            band1 = la.apply_real_lens_maslov(E, **kw)
    finally:
        _lm._SP_PIXEL_CHUNK = old
    assert np.array_equal(ref, banded), "realistic band not byte-identical"
    rel = float(np.abs(ref - band1).max()) / (float(np.abs(ref).max()) + 1e-30)
    assert rel < 1e-6, f"1-pixel band exceeds ULP: {rel:.2e}"


@pytest.mark.parametrize('use_numexpr', [False, True])
def test_quadrature_factorization_matches_explicit_G(use_numexpr):
    """M-P2: the Kronecker-factorized quadrature (no G materialized) must
    match the explicit per-row-band G @ H reference to float32 ULP, for both
    integrand kernels.  Uses poly_order=6 (M=210, P=7) so the factorization
    path is meaningfully different from the GEMM path."""
    N, dx = 160, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='quadrature', output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=6, n_v2=24,
              use_numexpr=use_numexpr)
    old = _lm._QUAD_FACTORIZE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _lm._QUAD_FACTORIZE = False
            ref = la.apply_real_lens_maslov(E, **kw)
            _lm._QUAD_FACTORIZE = True
            fac = la.apply_real_lens_maslov(E, **kw)
    finally:
        _lm._QUAD_FACTORIZE = old
    rel = float(np.abs(fac - ref).max()) / (float(np.abs(ref).max()) + 1e-30)
    assert rel < 1e-6, f"factorized vs G @ H exceeds ULP: {rel:.2e}"


@pytest.mark.parametrize('use_numexpr', [False, True])
def test_quadrature_output_row_banding_matches_unbanded(use_numexpr):
    """Output-row-banding must not change the result across band sizes.  On
    the explicit-G reference path (`_QUAD_FACTORIZE=False`) the banding is
    BYTE-identical (the GEMM contraction is band-independent).  On the default
    Kronecker-factorized path the per-band `einsum` picks a contraction order
    that depends on the band's row count -- notably a degenerate 1-row band
    differs at ULP -- so there the invariant is ULP, not byte-identical."""
    N, dx = 160, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='quadrature', output_subsample=1,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=4, n_v2=20,
              use_numexpr=use_numexpr)
    old_band = _lm._QUAD_ROW_BAND
    old_fac = _lm._QUAD_FACTORIZE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # (a) reference GEMM path: byte-identical across band sizes.
            _lm._QUAD_FACTORIZE = False
            _lm._QUAD_ROW_BAND = None
            ref = la.apply_real_lens_maslov(E, **kw)
            _lm._QUAD_ROW_BAND = 7       # non-divisor of N -> ragged last band
            band7 = la.apply_real_lens_maslov(E, **kw)
            _lm._QUAD_ROW_BAND = 1       # one output row per band
            band1 = la.apply_real_lens_maslov(E, **kw)
            assert np.array_equal(ref, band7), "GEMM 7-row band not byte-identical"
            assert np.array_equal(ref, band1), "GEMM 1-row band not byte-identical"
            # (b) default factorized path: band-invariant to ULP.
            _lm._QUAD_FACTORIZE = True
            _lm._QUAD_ROW_BAND = None
            fref = la.apply_real_lens_maslov(E, **kw)
            _lm._QUAD_ROW_BAND = 1
            fband1 = la.apply_real_lens_maslov(E, **kw)
    finally:
        _lm._QUAD_ROW_BAND = old_band
        _lm._QUAD_FACTORIZE = old_fac
    rel = float(np.abs(fref - fband1).max()) / (float(np.abs(fref).max()) + 1e-30)
    assert rel < 1e-6, f"factorized band-variance exceeds ULP: {rel:.2e}"


def test_non_divisible_subsample_roundtrips():
    """output_subsample that does not divide N (N=200, ss=3 -> N_out=66,
    66*3=198) must still return an (N, N) field, finite, with True/False
    linear-phase extraction agreeing.  scipy.zoom returns exactly N here
    (verified over N=16..4096 x ss=2..16), so the pad guards do not fire --
    this pins the realistic non-divisible path end to end."""
    N, dx = 200, 60e-6
    E = _gauss(N, dx, w=3e-3)
    kw = dict(prescription=_singlet(), wavelength=LAM, dx=dx,
              integration_method='quadrature', output_subsample=3,
              ray_field_samples=14, ray_pupil_samples=14, poly_order=4, n_v2=20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = la.apply_real_lens_maslov(E, extract_linear_phase=True, **kw)
        b = la.apply_real_lens_maslov(E, extract_linear_phase=False, **kw)
    assert a.shape == (N, N) and b.shape == (N, N)
    assert np.isfinite(a).all()
    rel = float(np.abs(np.abs(a) - np.abs(b)).max()) / (float(np.abs(a).max()) + 1e-30)
    assert rel < 0.05


def test_fine_grid_pad_guard_recovers_short_zoom():
    """The out_axis_f pad guard (`if out_axis_f.shape[0] != N`) is defensive
    -- scipy.zoom empirically always returns exactly N, so it never fires in
    practice.  Force it by shortening ONLY the 1-D axis zoom (ndim==1, leaving
    the 2-D amp/phase zooms intact) and confirm the field is still (N, N),
    finite and non-zero (the slope branch is live: a freeform prism gives a
    large _lin[1])."""
    import scipy.ndimage as _ndi
    N, dx = 192, 24e-3 / 192
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (7e-3) ** 2).astype(np.complex64)
    _orig = _ndi.zoom

    def _short1d(arr, *a, **k):
        out = _orig(arr, *a, **k)
        if arr.ndim == 1 and out.shape[0] == N:
            return out[:-1]                     # force shape != N -> pad guard
        return out

    _ndi.zoom = _short1d
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = la.apply_real_lens_maslov(
                E, prescription=_flat_prism(4e-3), wavelength=LAM, dx=dx,
                integration_method='quadrature', output_subsample=6,
                extract_linear_phase=True, collimated_input=True,
                ray_field_samples=12, ray_pupil_samples=12, poly_order=4, n_v2=16)
    finally:
        _ndi.zoom = _orig
    assert out.shape == (N, N)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0.0


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
