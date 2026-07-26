"""Territory-E regression pins for ``lumenairy.elements.lenses_maslov``
(audit ``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md``).

E-H1 (HIGH) -- ``output_subsample > 1`` upsampled the coarse Maslov field with
``scipy.ndimage.zoom(grid_mode=False)``, which is EDGE-anchored (fine i ->
coarse ``i*(Nc-1)/(N-1)``), while the coarse output grid is a STRIDE subsample
of the standard lattice (coarse j sits at fine index ``j*sub + (N-Nc*sub)/2``).
The exact sibling of the ``ii*Ns/N`` traced-upsample bug fixed at 0a743a6.
Measured on a rotationally-symmetric on-axis element (true SPATIAL intensity
centroid = 0), N=96:

    sub     pre-fix centroid    lattice prediction    post-fix
    2       +0.501849 px        +0.5106 px            +0.002392 px
    4       +1.535862 px        +1.5652 px            +0.019073 px
    8       +3.701055 px        +3.8182 px            +0.106073 px

with a second-moment width (magnification) error of x1.010073 at sub=2 ->
x1.000502 post-fix.  The pre-existing pins were blind to all of this: they
score the SPECTRAL centroid (invariant to a spatial shift) and the dtype.

E-H5 (HIGH) -- ``roi=`` silently overwrote ``normalize_output``, so a roi call
at the *default* ``normalize_output='power'`` returned the raw 'none' scale
(measured median ``|roi|/|full patch| = 0.000000``), and an explicit scalar
factor was dropped outright (measured ratio 0.500).

Also pinned here: E-M14 (``integration_method`` error message omitted the
default 'auto'), E-L20 (``fold_split=True`` silently dropped
``output_plane_distance`` / ``output_plane_n`` / ``roi`` and never forwarded
``progress``), E-L9 (dead args on the four Maslov integrators).
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from lumenairy.elements import lenses_maslov as _mz
from lumenairy.elements.lenses_maslov import apply_real_lens_maslov

LAM = 632.8e-9


# ---------------------------------------------------------------------------
# E-H1
# ---------------------------------------------------------------------------
def _weak_symmetric(R=2000e-3, ap=1.6e-3):
    """A weak, small-aperture, rotationally-symmetric singlet: its output is
    well-resolved on a coarse output grid, so the coarse->fine upsample is the
    ONLY thing that can move the spatial centroid."""
    return {
        'name': 'weak-symmetric', 'aperture_diameter': ap,
        'surfaces': [
            {'radius': R, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': ap / 2},
            {'radius': -R, 'conic': 0.0, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': ap / 2},
        ],
        'thicknesses': [1.5e-3],
    }


_N, _DX, _W = 96, 5e-6, 0.15e-3


def _axis():
    return (np.arange(_N) - _N / 2) * _DX


def _on_axis_input():
    x = _axis()
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / _W ** 2).astype(np.complex128)


def _maslov_kw():
    return dict(prescription=_weak_symmetric(), wavelength=LAM, dx=_DX,
                integration_method='quadrature', ray_field_samples=12,
                ray_pupil_samples=12, poly_order=4, n_v2=16,
                normalize_output='none', verbose=False)


@pytest.fixture(scope='module')
def _subsample_fields():
    """One Maslov run per output_subsample, shared across the E-H1 pins."""
    E = _on_axis_input()
    kw = _maslov_kw()
    out = {}
    for sub in (1, 2, 4):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[sub] = apply_real_lens_maslov(E, output_subsample=sub, **kw)
    return out


def _intensity_moments(A):
    x = _axis()
    I = np.abs(A) ** 2
    s = I.sum()
    cx = float((I.sum(0) * x).sum() / s)
    cy = float((I.sum(1) * x).sum() / s)
    wx = float(np.sqrt((I.sum(0) * (x - cx) ** 2).sum() / s))
    wy = float(np.sqrt((I.sum(1) * (x - cy) ** 2).sum() / s))
    return cx, cy, wx, wy


def test_eh1_spatial_centroid_is_subsample_invariant(_subsample_fields):
    """The SPATIAL intensity centroid of a rotationally-symmetric on-axis
    element is 0 by symmetry and must stay there at every output_subsample.

    Pre-fix: +0.5018 fine px at sub=2 and +1.5359 at sub=4 (matching the
    closed-form edge-anchored-zoom prediction +0.5106 / +1.5652); post-fix
    +0.0024 / +0.0191.  The 0.10-px gate is 5x the worst post-fix residual
    (which is the coarse-sampling / cos-sin phase-interpolation floor, not a
    lattice error) and 5x BELOW the pre-fix sub=2 walk."""
    for sub, F in sorted(_subsample_fields.items()):
        cx, cy, _, _ = _intensity_moments(F)
        assert abs(cx) / _DX < 0.10, (
            f"output_subsample={sub}: x centroid {cx / _DX:+.6f} fine px, "
            f"expected 0 by symmetry (edge-anchored upsample lattice?)")
        assert abs(cy) / _DX < 0.10, (
            f"output_subsample={sub}: y centroid {cy / _DX:+.6f} fine px, "
            f"expected 0 by symmetry")


def test_eh1_second_moment_width_is_not_magnified(_subsample_fields):
    """The edge-anchored zoom also MAGNIFIES the field by
    ``(N-1)/(sub*(Nc-1))``: measured second-moment width ratio (sub=2 vs
    sub=1) 1.010073 pre-fix vs the 1.01064 closed form, 1.000502 post-fix."""
    _, _, wx1, wy1 = _intensity_moments(_subsample_fields[1])
    _, _, wx2, wy2 = _intensity_moments(_subsample_fields[2])
    assert abs(wx2 / wx1 - 1.0) < 4e-3, f"x width ratio {wx2 / wx1:.6f}"
    assert abs(wy2 / wy1 - 1.0) < 4e-3, f"y width ratio {wy2 / wy1:.6f}"


def test_eh1_upsample_is_interpolatory_on_the_stride_lattice(
        _subsample_fields):
    """The sharpest statement of the lattice contract: every Maslov output
    pixel is integrated independently and the coarse output axis is exactly
    ``x[::sub]``, so the coarse samples must come back out of the upsample
    UNCHANGED at their own fine indices ``i = sub*j``.

    Pre-fix the edge-anchored zoom put them somewhere else entirely: measured
    max deviation 2.7e-2 of the field peak at sub=2 on this well-resolved
    chart (34% on a strong singlet), vs <=6e-16 post-fix."""
    ref = _subsample_fields[1]
    peak = float(np.abs(ref).max())
    for sub in (2, 4):
        F = _subsample_fields[sub]
        n_ok = (_N // sub) * sub
        got = F[:n_ok:sub, :n_ok:sub]
        want = ref[:n_ok:sub, :n_ok:sub]
        rel = float(np.abs(got - want).max()) / peak
        assert rel < 1e-10, (
            f"output_subsample={sub}: the coarse samples do not land on their "
            f"own fine indices -- max deviation {rel:.3e} of the field peak")


def _flat_prism(amp=4e-3, ap=24e-3, R_flat=1e9):
    """Flat surfaces + an xy_polynomial (1,0) tilt: a large REAL output-OPD
    slope on a coarse-resolvable field."""
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


@pytest.mark.parametrize('dy_factor', [1.0, 1.4])
def test_eh1_prism_tilt_magnitude_is_subsample_invariant(dy_factor):
    """The lattice error is a MAGNIFICATION as well as a shift, so it rescales
    a recovered output tilt.  A flat prism (large real ``_lin[1]``, re-applied
    on the fine grid via the resampled output axis) at N=96 measures the
    spectral tilt 744.04 cyc/m at sub=1; pre-fix sub=2 read 778.30 (+4.61%)
    and sub=6 read 915.34 (+23.02%) -- exactly the
    ``(N-1)/(sub*(Nc-1))`` stretch.  Post-fix: 743.91 (0.02%) and 742.82
    (0.16%).  Run at dy=dx and dy=1.4*dx so the anamorphic two-axis resample
    is covered as well (the existing subsample pins are square-pixel only)."""
    N = 96
    dx = 24e-3 / N
    dy = dy_factor * dx
    xs = (np.arange(N) - N // 2) * dx
    ys = (np.arange(N) - N // 2) * dy
    X, Y = np.meshgrid(xs, ys)
    E = np.exp(-(X ** 2 + Y ** 2) / (7e-3) ** 2).astype(np.complex64)
    kw = dict(prescription=_flat_prism(), wavelength=LAM, dx=dx, dy=dy,
              integration_method='quadrature', ray_field_samples=12,
              ray_pupil_samples=12, poly_order=4, n_v2=16,
              collimated_input=True, extract_linear_phase=True)

    def _spectral_cx(F):
        P = np.abs(np.fft.fft2(F)) ** 2
        fx = np.fft.fftfreq(N, d=dx)
        FX, _ = np.meshgrid(fx, fx, indexing='xy')
        return float((FX * P).sum() / P.sum())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c1 = _spectral_cx(apply_real_lens_maslov(E, output_subsample=1, **kw))
        c2 = _spectral_cx(apply_real_lens_maslov(E, output_subsample=2, **kw))
        c6 = _spectral_cx(apply_real_lens_maslov(E, output_subsample=6, **kw))
    assert abs(c1) > 500.0, f"prism tilt too weak to exercise the path: {c1}"
    assert abs(c2 - c1) / abs(c1) < 0.01, f"sub=2: {c2:.2f} vs {c1:.2f}"
    assert abs(c6 - c1) / abs(c1) < 0.01, f"sub=6: {c6:.2f} vs {c1:.2f}"


def test_eh1_subsample_1_does_no_resampling():
    """Invariant pin (holds pre- AND post-fix): ``output_subsample=1`` must
    never touch an ``ndimage`` resampler, so the fix cannot perturb the
    default path.  (Verified separately: the sub=1 output is bit-identical
    pre/post fix, sha256 1dac8c1d4da5bc18 on this chart.)"""
    import scipy.ndimage as ndi

    calls = {'n': 0}
    orig_zoom, orig_affine = ndi.zoom, ndi.affine_transform
    orig_mc = ndi.map_coordinates

    def _count(fn):
        def _wrapped(*a, **k):
            calls['n'] += 1
            return fn(*a, **k)
        return _wrapped

    E = _on_axis_input()
    kw = _maslov_kw()
    ndi.zoom = _count(orig_zoom)
    ndi.affine_transform = _count(orig_affine)
    ndi.map_coordinates = _count(orig_mc)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_maslov(E, output_subsample=1, **kw)
        assert calls['n'] == 0, (
            f"output_subsample=1 made {calls['n']} ndimage resample call(s)")
        calls['n'] = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_maslov(E, output_subsample=2, **kw)
        assert calls['n'] >= 3, (
            "output_subsample=2 must resample amplitude + cos + sin")
    finally:
        ndi.zoom, ndi.affine_transform = orig_zoom, orig_affine
        ndi.map_coordinates = orig_mc


# ---------------------------------------------------------------------------
# E-H5
# ---------------------------------------------------------------------------
_NR, _DXR = 64, 30e-6


def _roi_kw():
    from lumenairy.io.prescriptions_builders import make_singlet
    return dict(
        prescription=make_singlet(R1=40e-3, R2=-40e-3, d=3e-3, glass='N-BK7',
                                  aperture=1.8e-3),
        wavelength=LAM, dx=_DXR, verbose=False, ray_field_samples=10,
        ray_pupil_samples=10, poly_order=4, n_v2=16,
        integration_method='quadrature')


def _roi_input():
    xr = (np.arange(_NR) - _NR / 2) * _DXR
    X, Y = np.meshgrid(xr, xr)
    return np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)


def _centre_patch(full, n):
    c = _NR // 2
    sl = slice(c - n // 2, c - n // 2 + n)
    return full[sl, sl]


@pytest.mark.parametrize('norm', [2.0, 0.25, (3.0 + 1j)])
def test_eh5_roi_honours_a_scalar_normalize_output(norm):
    """A scalar ``normalize_output`` is window-INDEPENDENT, so the ROI patch
    must equal the corresponding full-grid patch exactly.  Pre-fix the roi
    branch overwrote normalize_output='none' and the factor was dropped
    (measured median ratio 0.500 for normalize_output=2.0)."""
    E = _roi_input()
    kw = _roi_kw()
    hw = 10 * _DXR
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full = apply_real_lens_maslov(E, normalize_output=norm, **kw)
        roi = apply_real_lens_maslov(E, normalize_output=norm,
                                     roi=(0.0, 0.0, hw), **kw)
    patch = _centre_patch(full, roi.shape[0])
    scale = float(np.abs(patch).max())
    rel = float(np.abs(roi - patch).max()) / scale
    assert rel < 1e-12, (
        f"normalize_output={norm!r}: roi patch differs from the full-grid "
        f"patch by {rel:.3e} (relative) -- the factor was dropped")


@pytest.mark.parametrize('norm', ['power', 'peak'])
def test_eh5_roi_warns_when_it_cannot_honour_normalize_output(norm):
    """'power' / 'peak' are global reductions over the full output grid, which
    the ROI path never evaluates.  Falling back to the raw scale is allowed
    (it is what keeps the ROI byte-identical to a 'none' full-grid slice) --
    doing it SILENTLY is not.  Pre-fix: no warning at all."""
    E = _roi_input()
    kw = _roi_kw()
    with pytest.warns(UserWarning, match='cannot honour normalize_output'):
        apply_real_lens_maslov(E, normalize_output=norm,
                               roi=(0.0, 0.0, 10 * _DXR), **kw)


def test_eh5_roi_none_scale_still_matches_the_full_grid_slice():
    """Invariant pin: the documented ROI contract (identical to the slice of a
    normalize_output='none' full-grid run) is preserved by the E-H5 fix, and
    an explicit 'none' does not warn."""
    E = _roi_input()
    kw = _roi_kw()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        full = apply_real_lens_maslov(E, normalize_output='none', **kw)
        roi = apply_real_lens_maslov(E, normalize_output='none',
                                     roi=(0.0, 0.0, 10 * _DXR), **kw)
    assert not [w for w in rec
                if 'normalize_output' in str(w.message)], \
        "explicit normalize_output='none' must not warn"
    patch = _centre_patch(full, roi.shape[0])
    assert np.array_equal(roi, patch)


def test_eh5_roi_does_not_swallow_an_invalid_normalize_output():
    """The unconditional ``normalize_output = 'none'`` overwrite also swallowed
    a typo'd value; with roi= it now reaches the same ValueError the full-grid
    path raises."""
    E = _roi_input()
    kw = _roi_kw()
    with pytest.raises(ValueError, match='normalize_output'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_maslov(E, normalize_output='powerr',
                                   roi=(0.0, 0.0, 10 * _DXR), **kw)


def test_eh5_normalize_output_is_documented():
    """``normalize_output`` was entirely undocumented (audit E-H5), including
    its ROI restriction."""
    doc = apply_real_lens_maslov.__doc__
    assert '``normalize_output``' in doc
    for token in ("'power'", "'peak'", "'none'", 'roi='):
        assert token in doc, f"normalize_output docs omit {token}"


# ---------------------------------------------------------------------------
# E-M14 / E-L20 / E-L9
# ---------------------------------------------------------------------------
def test_em14_integration_method_error_names_the_default():
    """The rejection message listed every method EXCEPT 'auto' -- the
    default."""
    E = _roi_input()
    kw = {k: v for k, v in _roi_kw().items() if k != 'integration_method'}
    with pytest.raises(ValueError, match="'auto'"):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_maslov(E, integration_method='qadrature', **kw)


def _folded_rx():
    els = [
        {'element_type': 'surface', 'radius': 60e-3, 'conic': 0.,
         'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'element_type': 'surface', 'radius': float('inf'), 'conic': 0.,
         'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'element_type': 'mirror', 'radius': float('inf'), 'conic': 0.},
        {'element_type': 'surface', 'radius': float('inf'), 'conic': 0.,
         'glass_before': 'air', 'glass_after': 'air'}]
    return {'elements': els,
            'surfaces': [e for e in els if e['element_type'] == 'surface'],
            'thicknesses': [3e-3, 8e-3, 0.0],
            'all_thicknesses': [3e-3, 4e-3, 4e-3],
            'aperture_diameter': 4e-3, 'name': 'fold'}


@pytest.mark.parametrize('extra,name', [
    (dict(output_plane_distance=5e-3), 'output_plane_distance'),
    (dict(roi=(0.0, 0.0, 10 * _DXR)), 'roi'),
])
def test_el20_fold_split_refuses_to_drop_the_observation_plane(extra, name):
    """``fold_split=True`` built its per-leg kwargs without
    output_plane_distance / output_plane_n / roi, so the requested observation
    plane / ROI window was silently DROPPED (pre-fix: a full (96, 96) field at
    the last lens vertex came back with no diagnostic)."""
    E = _roi_input()
    with pytest.raises(NotImplementedError, match=name):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_maslov(
                E, fold_split=True, prescription=_folded_rx(), wavelength=LAM,
                dx=_DXR, ray_field_samples=8, ray_pupil_samples=8,
                poly_order=3, n_v2=8, integration_method='quadrature',
                **extra)


def test_el20_fold_split_forwards_progress():
    """``progress`` was dropped from the per-leg kwargs too: a folded run
    reported nothing.  It is a pure callback, so it is forwarded."""
    seen = []
    E = _roi_input()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_maslov(
            E, fold_split=True, prescription=_folded_rx(), wavelength=LAM,
            dx=_DXR, ray_field_samples=8, ray_pupil_samples=8, poly_order=3,
            n_v2=8, integration_method='quadrature',
            progress=lambda *a, **k: seen.append(a))
    assert out.shape == (_NR, _NR)
    assert seen, "fold_split=True never invoked the progress callback"


@pytest.mark.parametrize('fn,dead', [
    ('_integrate_quadrature', ('K1_arr', 'K2_arr')),
    ('_integrate_stationary_phase', ('v2x_c', 'v2y_c')),
    ('_integrate_levin', ('mi', 'v2x_h', 'v2y_h')),
    ('_integrate_local_quadrature', ('mi', 'v2x_c', 'v2y_c')),
])
def test_el9_maslov_integrators_have_no_dead_args(fn, dead):
    """10 never-read parameters across the four Maslov integrators (ruff
    ARG001).  All four are module-private with exactly one call site, so
    dropping them is not an API change -- and it CONVERGES each one onto its
    CuPy twin, which never took them (the audit's NumPy/CuPy twin drift)."""
    params = set(inspect.signature(getattr(_mz, fn)).parameters)
    assert not (params & set(dead)), (
        f"{fn} still takes never-read arg(s) {sorted(params & set(dead))}")
