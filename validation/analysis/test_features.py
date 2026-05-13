"""Tests for the specialty-feature subsystems.

Covers coatings, interferometry, freeform surfaces, ghost analysis,
RCWA, multi-configuration / afocal.  Content taken wholesale from
``new_features_deep_test.py`` (minus the tests that are more naturally
homed in test_sources.py, test_optimize.py, or test_integration.py).
"""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import lumenairy as la
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd,
)


H = Harness('features')
lam = 1.31e-6


# ---------------------------------------------------------------------
H.section('Thin-film coatings')


def t_coating_qw_ar_zero():
    layers = la.quarter_wave_ar(1.52, lam)
    R, T, _ = la.coating_reflectance(layers, [lam], n_substrate=1.52)
    return R[0] < 1e-10, f'R = {R[0]:.2e}'


H.run('Coating: QW AR gives R=0 at design wavelength',
      t_coating_qw_ar_zero)


def t_coating_uncoated_fresnel():
    n = 1.52
    R, _, _ = la.coating_reflectance([], [lam], n_substrate=n)
    expected = ((n - 1) / (n + 1)) ** 2
    return abs(R[0] - expected) < 1e-4, \
        f'R={R[0]:.5f}, expected={expected:.5f}'


H.run('Coating: uncoated Fresnel matches formula',
      t_coating_uncoated_fresnel)


def t_coating_energy_conservation():
    layers = la.broadband_ar_v_coat(1.52, lam)
    R, T, _ = la.coating_reflectance(layers, [lam], n_substrate=1.52)
    return abs(R[0] + T[0] - 1.0) < 0.01, f'R+T = {R[0]+T[0]:.6f}'


H.run('Coating: R + T = 1 (energy conservation)',
      t_coating_energy_conservation)


def t_coating_spectral_shape():
    wvs = np.linspace(1.0e-6, 1.6e-6, 51)
    layers = la.quarter_wave_ar(1.52, 1.31e-6)
    R, _, _ = la.coating_reflectance(layers, wvs, n_substrate=1.52)
    i_min = np.argmin(R)
    wv_min = wvs[i_min]
    return abs(wv_min - 1.31e-6) < 0.03e-6, \
        f'R minimum at {wv_min*1e6:.3f} um'


H.run('Coating: QW AR minimum at design wavelength',
      t_coating_spectral_shape)


# ---------------------------------------------------------------------
H.section('Interferometry / phase-shift extraction')


def t_interferogram_flat_opd():
    opd = np.zeros((64, 64))
    fringe = la.simulate_interferogram(opd, lam)
    std = np.std(fringe)
    return std < 1e-10, f'fringe std = {std:.2e}'


H.run('Interferogram: flat OPD gives uniform field',
      t_interferogram_flat_opd)


def t_interferogram_tilt_fringes():
    N = 256; dx = 4e-6
    opd = np.zeros((N, N))
    tilt = 1.0 / (20 * dx)
    fringe = la.simulate_interferogram(opd, lam, tilt_x=tilt, dx=dx)
    row = fringe[N//2, :]
    peaks = []
    for i in range(1, len(row)-1):
        if row[i] > row[i-1] and row[i] > row[i+1]:
            peaks.append(i)
    if len(peaks) >= 2:
        period = np.mean(np.diff(peaks))
        return abs(period - 20) < 3, \
            f'fringe period = {period:.1f} pixels'
    return False, f'found {len(peaks)} peaks'


H.run('Interferogram: tilt produces correct fringe period',
      t_interferogram_tilt_fringes)


def t_psi_roundtrip():
    N = 64
    opd = np.random.default_rng(0).standard_normal((N, N)) * 50e-9
    shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
    frames = []
    for s in shifts:
        phase = 2 * np.pi * opd / lam + s
        frames.append(0.5 + 0.5 * np.cos(phase))
    extracted, mod = la.phase_shift_extract(
        frames, shifts, convention='library')
    input_phase = 2 * np.pi * opd / lam
    diff = np.angle(np.exp(1j * (extracted - input_phase)))
    rms = np.sqrt(np.mean(diff**2))
    return rms < 0.1, f'phase extraction RMS = {rms:.4f} rad'


H.run('PSI: round-trip phase extraction', t_psi_roundtrip)


# ---------------------------------------------------------------------
H.section('Freeform surfaces')


def t_freeform_xy_astigmatism():
    x = np.linspace(-1e-3, 1e-3, 50)
    X, Y = np.meshgrid(x, x)
    sag_base = la.surface_sag_general(X**2 + Y**2, R=np.inf)
    sag_astig = la.surface_sag_xy_polynomial(
        X, Y, R=np.inf,
        xy_coeffs={(2, 0): 1e-6, (0, 2): -1e-6})
    diff = sag_astig - sag_base
    diff_x = diff[25, 49]
    diff_y = diff[49, 25]
    return diff_x > 0 and diff_y < 0, \
        f'diff_x={diff_x:.2e}, diff_y={diff_y:.2e}'


H.run('Freeform: XY poly adds astigmatism correctly',
      t_freeform_xy_astigmatism)


def t_freeform_zernike_defocus():
    x = np.linspace(-1e-3, 1e-3, 50)
    X, Y = np.meshgrid(x, x)
    sag = la.surface_sag_zernike_freeform(
        X, Y, R=np.inf, zernike_coeffs={4: 100e-9},
        norm_radius=1e-3)
    center = sag[25, 25]
    edge = sag[0, 25]
    return edge > center, \
        f'edge={edge:.2e}, center={center:.2e}'


H.run('Freeform: Zernike defocus adds r^2',
      t_freeform_zernike_defocus)


def t_freeform_chebyshev_nonzero():
    x = np.linspace(-1e-3, 1e-3, 20)
    X, Y = np.meshgrid(x, x)
    sag = la.surface_sag_chebyshev(
        X, Y, R=np.inf,
        cheb_coeffs={(2, 0): 1e-7, (0, 2): 5e-8},
        norm_x=1e-3, norm_y=1e-3)
    return np.std(sag) > 0, f'sag std = {np.std(sag):.2e}'


H.run('Freeform: Chebyshev produces nonzero sag',
      t_freeform_chebyshev_nonzero)


def t_freeform_dispatch():
    x = np.linspace(-1e-3, 1e-3, 10)
    X, Y = np.meshgrid(x, x)
    sd = {'radius': np.inf, 'freeform_type': 'xy_polynomial',
          'xy_coeffs': {(2, 0): 1e-6}}
    sag = la.surface_sag_freeform(X, Y, sd)
    return sag.shape == (10, 10) and np.std(sag) > 0, 'dispatch OK'


H.run('Freeform: dispatcher routes correctly', t_freeform_dispatch)


def t_freeform_on_prescription_surface():
    N = 128; dx = 16e-6
    pres = la.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    pres['surfaces'][0]['freeform_type'] = 'xy_polynomial'
    pres['surfaces'][0]['xy_coeffs'] = {(4, 0): 1e-12}
    x = np.linspace(-1e-3, 1e-3, 10)
    X, Y = np.meshgrid(x, x)
    sag = la.surface_sag_freeform(X, Y, pres['surfaces'][0])
    return sag.shape == (10, 10), 'freeform sag computed'


H.run('Freeform: dispatch works on prescription surface dict',
      t_freeform_on_prescription_surface)


# ---------------------------------------------------------------------
H.section('Ghost analysis')


def t_ghost_path_count():
    paths = la.enumerate_ghost_paths(4)
    expected = 4 * 3 // 2
    return len(paths) == expected, \
        f'{len(paths)} paths (expect {expected})'


H.run('Ghost: correct path count for 4 surfaces', t_ghost_path_count)


def t_ghost_intensity_ordering():
    pres = la.thorlabs_lens('AC254-100-C')
    ghosts = la.ghost_analysis(pres, lam, verbose=False)
    intensities = [g['intensity'] for g in ghosts]
    is_sorted = all(intensities[i] >= intensities[i+1]
                    for i in range(len(intensities)-1))
    return is_sorted, \
        f'sorted = {is_sorted}, brightest = {intensities[0]:.2e}'


H.run('Ghost: intensities sorted brightest-first',
      t_ghost_intensity_ordering)


def t_ghost_fresnel_consistent():
    pres = la.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=10e-3)
    ghosts = la.ghost_analysis(pres, lam, verbose=False)
    g = ghosts[0]
    expected = g['R_i'] * g['R_j']
    return abs(g['intensity'] - expected) < 1e-10, \
        f'I={g["intensity"]:.6e}, R_i*R_j={expected:.6e}'


H.run('Ghost: intensity = R_i * R_j', t_ghost_fresnel_consistent)


def t_ghost_of_thorlabs():
    pres = la.thorlabs_lens('AC254-100-C')
    ghosts = la.ghost_analysis(pres, lam, verbose=False)
    return len(ghosts) == 3, f'{len(ghosts)} ghost paths'


H.run('Ghost analysis of Thorlabs doublet (3 paths)',
      t_ghost_of_thorlabs)


def t_ghost_focus_positions_finite_and_distinct():
    """Each ghost path's focus_z_estimate should be finite and the
    different ghost paths should produce DIFFERENT focal positions
    (a real lens system has multiple ghost focuses, not one).
    """
    pres = la.thorlabs_lens('AC254-100-C')
    ghosts = la.ghost_analysis(pres, lam, verbose=False)
    z_focuses = [g['focus_z_estimate'] for g in ghosts
                 if 'focus_z_estimate' in g
                 and g['focus_z_estimate'] is not None]
    n_finite = sum(1 for z in z_focuses if np.isfinite(z))
    distinct = len(set(round(z * 1e6) for z in z_focuses
                       if np.isfinite(z))) >= 2
    return n_finite >= 2 and distinct, (
        f'finite focus_z_estimates: {n_finite}/{len(ghosts)}; '
        f'distinct: {distinct}; values (mm): '
        f'{[f"{z*1e3:.2f}" for z in z_focuses if np.isfinite(z)][:5]}')


def t_ghost_intensity_ordering_by_path():
    """Ghosts traversing two reflection bounces should always have
    intensity below 1: R_i * R_j is the product of two Fresnel
    reflectances, each < 1.  Catches sign-bug or normalization-bug
    in the intensity computation.
    """
    pres = la.thorlabs_lens('AC254-100-C')
    ghosts = la.ghost_analysis(pres, lam, verbose=False)
    intensities = [g['intensity'] for g in ghosts]
    all_below_one = all(0 <= I < 1.0 for I in intensities)
    sorted_descending = intensities == sorted(intensities, reverse=True)
    return all_below_one and sorted_descending, (
        f'intensities={intensities[:5]}, sorted={sorted_descending}')


H.run('Ghost focus_z positions finite + distinct',
      t_ghost_focus_positions_finite_and_distinct)
H.run('Ghost intensities < 1 and sorted brightest-first',
      t_ghost_intensity_ordering_by_path)


# ---------------------------------------------------------------------
H.section('Thin-grating diffraction efficiency')


def t_rcwa_energy_conservation():
    orders, _, T = la.thin_grating_efficiency_1d(1e-6, 1.5, 1.0, 1.52, 1.0, 0.5e-6,
                              0.5, lam)
    total = T.sum()
    return 0.5 < total <= 1.01, f'total T = {total:.4f}'


H.run('RCWA: transmitted efficiency sum <= 1',
      t_rcwa_energy_conservation)


def t_rcwa_zeroth_order_dominates():
    orders, _, T = la.thin_grating_efficiency_1d(10e-6, 1.5, 1.0, 1.52, 1.0, 0.01e-6,
                              0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] > 0.8, f'T_0 = {T[i0]:.4f}'


H.run('RCWA: zeroth order dominates for shallow grating',
      t_rcwa_zeroth_order_dominates)


def t_rcwa_deep_grating_splits():
    d_pi = lam / (2 * (1.5 - 1.0))
    orders, _, T = la.thin_grating_efficiency_1d(5e-6, 1.5, 1.0, 1.52, 1.0, d_pi, 0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] < 0.2, f'T_0 = {T[i0]:.4f} at pi depth'


H.run('RCWA: pi-depth suppresses zeroth order',
      t_rcwa_deep_grating_splits)


def t_rcwa_large_period_scalar():
    orders, _, T = la.thin_grating_efficiency_1d(100e-6, 1.5, 1.0, 1.52, 1.0, 0.1e-6,
                              0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] > 0.5, \
        f'T_0 = {T[i0]:.4f} for large-period grating'


H.run('RCWA vs scalar: large period -> zeroth order dominates',
      t_rcwa_large_period_scalar)


# ---------------------------------------------------------------------
H.section('Multi-configuration / afocal')


def t_keplerian_afocal():
    pres = la.keplerian_telescope(200e-3, 50e-3)
    mag, _ = la.afocal_angular_magnification(pres, lam)
    surfs = surfaces_from_prescription(pres)
    M_abcd, _, _, _ = system_abcd(surfs, lam)
    B = abs(M_abcd[0, 1])
    return B < 0.5, f'B = {B:.4f}, mag = {mag:.3f}'


H.run('Afocal: Keplerian telescope has B ~ 0', t_keplerian_afocal)


def t_beam_expander_magnification():
    pres = la.beam_expander_prescription(3.0, 100e-3)
    mag, _ = la.afocal_angular_magnification(pres, lam)
    expected = 1.0 / 3.0
    err = abs(abs(mag) - expected) / expected
    return err < 0.5, \
        f'mag = {mag:.3f}, expected = {expected:.3f}'


H.run('Beam expander: angular mag ~ 1/M',
      t_beam_expander_magnification)


def t_multi_config_weighted():
    from lumenairy.optimize.multiconfig import (
        Configuration, multi_config_merit)
    pres = la.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=10e-3)
    cfgs = [
        Configuration('a', pres, weight=1.0),
        Configuration('b', pres, weight=3.0),
    ]
    def const_merit(p, w, f):
        return 1.0
    total, per = multi_config_merit(cfgs, const_merit)
    return total == 4.0, f'total = {total} (expect 4)'


H.run('Multi-config: weighted sum correct', t_multi_config_weighted)


def t_multi_config_field_angle_changes_merit():
    """Configurations with different field_angle values should
    produce different per-config merit values when the merit_fn
    consumes field_angle (catches the case where Configuration's
    field_angle isn't actually plumbed through to the merit).
    """
    from lumenairy.optimize.multiconfig import (
        Configuration, multi_config_merit)
    pres = la.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=10e-3)
    cfgs = [
        Configuration('on_axis', pres, field_angle=0.0, weight=1.0),
        Configuration('off_axis', pres, field_angle=0.05, weight=1.0),
    ]

    # Merit that depends on field_angle directly via the prescription
    # (which Configuration carries unchanged) -- since field_angle
    # itself isn't part of the prescription dict, we read it via the
    # merit_fn signature `(prescription, wavelength, field_angle)`.
    def field_dependent_merit(p, w, f):
        return float(f) ** 2

    total, per = multi_config_merit(cfgs, field_dependent_merit)
    differ = abs(per[0] - per[1]) > 1e-12
    return differ, f'per-config merits = {per}'


H.run('Multi-config: field_angle propagates to merit',
      t_multi_config_field_angle_changes_merit)


# ---------------------------------------------------------------------
H.section('Freeform sag continuity')


def t_freeform_xy_polynomial_C0_continuous():
    """surface_sag_xy_polynomial should be C^0 continuous: a small
    grid shift produces a small sag change.  Catches stair-step or
    interpolation bugs.
    """
    from lumenairy.elements.freeform import surface_sag_xy_polynomial
    N = 64
    x = np.linspace(-1e-3, 1e-3, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    coeffs = {(2, 0): 1e-3, (0, 2): -5e-4, (4, 0): 1e2}
    sag1 = surface_sag_xy_polynomial(X, Y, R=np.inf, conic=0.0,
                                      xy_coeffs=coeffs)
    # Shift the grid by 1% of a pixel.
    eps = 1e-2 * (x[1] - x[0])
    sag2 = surface_sag_xy_polynomial(X + eps, Y, R=np.inf, conic=0.0,
                                      xy_coeffs=coeffs)
    diff = float(np.max(np.abs(sag2 - sag1)))
    sag_scale = float(np.max(np.abs(sag1)))
    return diff < 0.05 * sag_scale + 1e-12, (
        f'sub-pixel shift produced max-abs-diff = {diff:.2e} '
        f'(sag scale {sag_scale:.2e})')


def t_freeform_zernike_derivative_finite():
    """surface_sag_zernike_freeform's first derivative computed by
    central differences should be finite and bounded everywhere
    inside the unit disk (no infinity / NaN at any pixel).
    """
    from lumenairy.elements.freeform import surface_sag_zernike_freeform
    N = 96
    x = np.linspace(-0.95, 0.95, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    # Mix of low- and mid-order Zernikes with realistic magnitudes.
    zern = {4: 0.5e-6, 11: 0.2e-6, 22: 0.1e-6}
    sag = surface_sag_zernike_freeform(
        X, Y, R=np.inf, conic=0.0,
        zernike_coeffs=zern, norm_radius=1.0)
    finite = bool(np.all(np.isfinite(sag)))
    inside = (X**2 + Y**2) < 0.81
    dz_dx = np.gradient(sag, axis=1) / (x[1] - x[0])
    dz_dy = np.gradient(sag, axis=0) / (x[1] - x[0])
    grad_finite_inside = bool(
        np.all(np.isfinite(dz_dx[inside]))
        and np.all(np.isfinite(dz_dy[inside])))
    return finite and grad_finite_inside, (
        f'sag finite={finite}, grad finite (inside)={grad_finite_inside}')


def t_freeform_dispatch_recovers_underlying_sag():
    """surface_sag_freeform routing dispatch should produce the same
    sag as calling the underlying surface_sag_xy_polynomial directly.
    Catches any wiring drift between the high-level dispatch helper
    and the per-type implementation.
    """
    from lumenairy.elements.freeform import (
        surface_sag_xy_polynomial, surface_sag_freeform,
    )
    N = 32
    x = np.linspace(-1e-3, 1e-3, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    coeffs = {(2, 0): 5e-4, (0, 2): 5e-4}
    direct = surface_sag_xy_polynomial(X, Y, R=np.inf, conic=0.0,
                                        xy_coeffs=coeffs)
    via_dispatch = surface_sag_freeform(X, Y, {
        'freeform_type': 'xy_polynomial',
        'xy_coeffs': coeffs,
        'radius': np.inf, 'conic': 0.0,
    })
    err = float(np.max(np.abs(direct - via_dispatch)))
    return err < 1e-15, f'direct vs dispatch max-err = {err:.2e}'


H.run('Freeform XY-polynomial sag is sub-pixel C^0 continuous',
      t_freeform_xy_polynomial_C0_continuous)
H.run('Freeform Zernike sag has finite derivative inside unit disk',
      t_freeform_zernike_derivative_finite)
H.run('surface_sag_freeform dispatch matches direct call',
      t_freeform_dispatch_recovers_underlying_sag)


# ---------------------------------------------------------------------
H.section('BSDF surface scatter')


def t_lambertian_tis():
    bsdf = la.LambertianBSDF(rho=0.3)
    tis = bsdf.total_integrated_scatter()
    return abs(tis - 0.3) < 1e-12, f'TIS = {tis}'


H.run('Lambertian BSDF: TIS == rho', t_lambertian_tis)


def t_lambertian_sample_hemisphere():
    bsdf = la.LambertianBSDF(rho=1.0)
    samples = bsdf.sample(np.array([0.0, 0.0, -1.0]), 500, rng=42)
    in_hemi = np.all(samples[:, 2] > 0)
    unit = np.allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-10)
    return in_hemi and unit, \
        f'in_hemi={in_hemi}, unit_vectors={unit}'


H.run('Lambertian BSDF: samples are unit vectors in +z hemisphere',
      t_lambertian_sample_hemisphere)


def t_gaussian_tis():
    bsdf = la.GaussianBSDF(sigma_rad=0.005, scattered_fraction=0.01)
    return abs(bsdf.total_integrated_scatter() - 0.01) < 1e-12, \
        f'TIS = {bsdf.total_integrated_scatter()}'


H.run('Gaussian BSDF: TIS == scattered_fraction', t_gaussian_tis)


def t_gaussian_sample_concentrates_near_specular():
    bsdf = la.GaussianBSDF(sigma_rad=0.005, scattered_fraction=0.01)
    inc = np.array([0.0, 0.0, -1.0])
    samples = bsdf.sample(inc, 1000, rng=0)
    specular = np.array([0.0, 0.0, 1.0])
    cos_theta = np.sum(samples * specular, axis=1)
    mean_angle = np.mean(np.arccos(np.clip(cos_theta, -1, 1)))
    # Expect mean angle ~ sigma (roughly)
    return mean_angle < 10 * bsdf.sigma_rad, \
        f'mean scatter angle = {np.degrees(mean_angle):.3f} deg'


H.run('Gaussian BSDF: samples cluster near specular direction',
      t_gaussian_sample_concentrates_near_specular)


def t_harvey_shack_onaxis_greater_than_offaxis():
    bsdf = la.HarveyShackBSDF(b0=1.0, l=0.01, s=2.0)
    inc = np.array([0.0, 0.0, -1.0])
    on = bsdf.evaluate(inc, np.array([0.0, 0.0, 1.0]))
    off = bsdf.evaluate(inc, np.array([
        np.sin(np.radians(5)), 0, np.cos(np.radians(5))]))
    return on > off > 0, f'on-axis={on:.3e}, 5deg={off:.3e}'


H.run('Harvey-Shack: on-axis BSDF exceeds off-axis',
      t_harvey_shack_onaxis_greater_than_offaxis)


def t_make_bsdf_dispatch():
    b1 = la.make_bsdf({'kind': 'lambertian', 'rho': 0.5})
    b2 = la.make_bsdf({'kind': 'gaussian', 'sigma_rad': 0.001,
                       'scattered_fraction': 0.005})
    b3 = la.make_bsdf({'kind': 'harvey_shack', 'b0': 2.0,
                       'l': 0.005, 's': 1.8})
    b4 = la.make_bsdf(None)
    ok = (isinstance(b1, la.LambertianBSDF) and b1.rho == 0.5
          and isinstance(b2, la.GaussianBSDF)
          and isinstance(b3, la.HarveyShackBSDF)
          and b4 is None)
    return ok, 'all four dispatch cases handled'


H.run('make_bsdf dispatches by kind', t_make_bsdf_dispatch)


def t_make_bsdf_unknown_kind_raises():
    try:
        la.make_bsdf({'kind': 'does_not_exist'})
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('make_bsdf raises on unknown kind',
      t_make_bsdf_unknown_kind_raises)


def t_sample_scatter_rays_spawns_correct_count():
    from lumenairy.raytrace import Surface, _make_bundle
    surf = Surface(radius=np.inf, semi_diameter=10e-3,
                   bsdf=la.LambertianBSDF(rho=1.0))
    incident = _make_bundle(
        x=np.array([0, 1e-3, -1e-3, 2e-3, -2e-3]),
        y=np.array([0, 0, 0, 0, 0]),
        L=np.array([0, 0, 0, 0, 0]),
        M=np.array([0, 0, 0, 0, 0]),
        wavelength=1.31e-6)
    scatt = la.sample_scatter_rays(surf, incident, n_per_ray=3, rng=0)
    return (scatt.x.size == 15 and np.all(scatt.N > 0)
            and np.all(scatt.alive)), \
        f'n_rays={scatt.x.size}, alive={scatt.alive.sum()}'


H.run('sample_scatter_rays: n_per_ray * n_incident rays in +z hemisphere',
      t_sample_scatter_rays_spawns_correct_count)


# ---------------------------------------------------------------------
# Additional feature-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('Specialty features: cross-checks')


def t_quarter_wave_ar_design_lambda_minimizes_R():
    """A quarter-wave AR coating layer at its design wavelength has
    near-zero reflectance for normal incidence."""
    layers = la.quarter_wave_ar(1.52, lam)
    R, _, _ = la.coating_reflectance(layers, [lam], n_substrate=1.52)
    return R[0] < 1e-6, f'R(design) = {R[0]:.2e}'


H.run('Quarter-wave AR: near-zero R at design wavelength',
      t_quarter_wave_ar_design_lambda_minimizes_R)


def t_quarter_wave_ar_R_grows_off_design():
    """Off the AR design wavelength reflectance grows."""
    layers = la.quarter_wave_ar(1.52, lam)
    R_on, _, _ = la.coating_reflectance(layers, [lam], n_substrate=1.52)
    R_off, _, _ = la.coating_reflectance(layers, [lam * 1.5], n_substrate=1.52)
    return R_off[0] > R_on[0], f'R(on)={R_on[0]:.4e}, R(off)={R_off[0]:.4e}'


H.run('Quarter-wave AR: R grows off design wavelength',
      t_quarter_wave_ar_R_grows_off_design)


def t_simulate_interferogram_fringes_full_contrast():
    """Tilted-reference interferogram has fringes with full contrast
    (intensity reaches near 0 and near 4 for unit-amplitude fields)."""
    N, dx = 256, 4e-6
    E_ref = np.ones((N, N), dtype=np.complex128)
    x = (np.arange(N) - N/2) * dx
    X, _ = np.meshgrid(x, x)
    E_obj = np.exp(1j * 2 * np.pi * X / 100e-6).astype(np.complex128)
    out = la.simulate_interferogram(E_ref, E_obj)
    I = out[0] if isinstance(out, tuple) else out
    return float(I.max()) > 3.5 and float(I.min()) < 0.5, \
        f'I min/max = ({float(I.min()):.3f}, {float(I.max()):.3f})'


H.run('Interferogram: full contrast for unit-amp tilted reference',
      t_simulate_interferogram_fringes_full_contrast)


def t_keplerian_telescope_angular_magnification():
    """A Keplerian telescope (f1=200mm, f2=50mm) has |M| = f1 / f2."""
    rx = la.keplerian_telescope(f_objective=200e-3, f_eyepiece=50e-3,
                                  glass='N-BK7', wavelength=lam)
    out = la.afocal_angular_magnification(rx, lam)
    M = out[0] if isinstance(out, tuple) else out
    return abs(abs(M) - 200e-3 / 50e-3) < 0.05, \
        f'|M|={abs(M):.4f}, expected=4.0'


H.run('Keplerian telescope: |M| ~ f1/f2',
      t_keplerian_telescope_angular_magnification)


def t_beam_expander_prescription_returns_dict():
    """A 5x beam-expander prescription is built without error."""
    rx = la.beam_expander_prescription(
        M=5, f_objective=20e-3, glass='N-BK7',
        aperture=20e-3, wavelength=lam)
    has_keys = isinstance(rx, dict) and ('elements' in rx or 'surfaces' in rx)
    return has_keys, f'rx keys = {sorted(rx.keys())[:8]}'


H.run('beam_expander_prescription: returns valid prescription',
      t_beam_expander_prescription_returns_dict)


def t_freeform_xy_polynomial_zero_coefficients_returns_zero_sag():
    """A zero-coefficient XY polynomial with R=inf gives zero sag."""
    N = 32
    coords = np.linspace(-1e-3, 1e-3, N)
    X, Y = np.meshgrid(coords, coords)
    sag = la.surface_sag_xy_polynomial(X, Y, R=np.inf, conic=0.0,
                                         xy_coeffs={})
    return float(np.max(np.abs(sag))) < 1e-15, \
        f'max |sag| = {float(np.max(np.abs(sag))):.2e}'


H.run('Freeform XY-poly: zero coefficients give zero sag (flat input)',
      t_freeform_xy_polynomial_zero_coefficients_returns_zero_sag)


def t_make_bsdf_lambertian_returns_lambertian_instance():
    """make_bsdf({'type':'lambertian',...}) returns a LambertianBSDF."""
    bsdf = la.make_bsdf({'kind': 'lambertian', 'rho': 1.0})
    return isinstance(bsdf, la.LambertianBSDF), \
        f'type = {type(bsdf).__name__}'


H.run('make_bsdf: lambertian factory returns LambertianBSDF',
      t_make_bsdf_lambertian_returns_lambertian_instance)


def t_lambertian_bsdf_evaluates_to_constant_rho_over_pi():
    """Lambertian BRDF is rho/pi for ALL incident/scattered direction
    pairs in the upper hemisphere -- the defining property.  Catches
    bugs where a directional weighting is applied to the value
    (the cosine weighting belongs in the radiance integral, not in
    the BRDF itself).
    """
    rho = 0.4
    bsdf = la.LambertianBSDF(rho=rho)
    expected = rho / np.pi
    inc = np.array([0.0, 0.0, -1.0])
    samples = [
        np.array([0.0, 0.0, 1.0]),                  # straight up
        np.array([0.5, 0.0, np.sqrt(0.75)]),        # 30 deg in x
        np.array([0.0, 0.7071, 0.7071]),            # 45 deg in y
        np.array([0.6, 0.6, np.sqrt(1 - 0.72)]),    # diagonal
    ]
    vals = [float(bsdf.evaluate(inc, s)) for s in samples]
    err = max(abs(v - expected) for v in vals)
    return err < 1e-12, (
        f'expected {expected:.6e}; got {[f"{v:.6e}" for v in vals]}; '
        f'max-err={err:.2e}')


def t_lambertian_bsdf_sample_distribution_matches_cos_law():
    """Lambertian sampling: scattered ray polar angle should follow
    p(theta) = sin(theta) * cos(theta) / pi (the projected-area-
    weighted distribution).  Verify that <cos(theta)> = 2/3 (the
    closed-form mean of the cosine-weighted distribution over the
    upper hemisphere).
    """
    rng = np.random.default_rng(0)
    bsdf = la.LambertianBSDF(rho=1.0)
    inc = np.array([0.0, 0.0, -1.0])
    n = 50000
    dirs = bsdf.sample(inc, n_samples=n, rng=rng)
    # cos(theta) = z-component of a unit-length scattered direction.
    cos_theta = np.asarray(dirs)[:, 2]
    in_upper = cos_theta > 0
    mean_cos = float(np.mean(cos_theta[in_upper]))
    # 2/3 is the closed-form mean for cos-weighted hemisphere sampling.
    err = abs(mean_cos - 2.0 / 3.0)
    return err < 0.02, (
        f'mean(cos theta) = {mean_cos:.4f}, expected 0.6667, err={err:.2e}')


def t_gaussian_bsdf_concentrates_within_sigma():
    """Most Gaussian-BSDF samples should land within ~3 sigma of the
    specular direction.  Catches sigma-units-bug or wrong sampling
    width.
    """
    sigma = 0.01  # rad
    rng = np.random.default_rng(7)
    bsdf = la.GaussianBSDF(sigma_rad=sigma, scattered_fraction=1.0)
    inc = np.array([0.0, 0.0, -1.0])  # specular -> +z
    n = 20000
    dirs = np.asarray(bsdf.sample(inc, n_samples=n, rng=rng))
    # Angle between each sample and the specular direction (+z):
    cos_theta = dirs[:, 2]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    frac_within_3sigma = float(np.mean(theta < 3 * sigma))
    return frac_within_3sigma > 0.99, (
        f'fraction of samples within 3*sigma = {frac_within_3sigma:.4f}')


H.run('Lambertian BSDF.evaluate is constant rho/pi',
      t_lambertian_bsdf_evaluates_to_constant_rho_over_pi)
H.run('Lambertian BSDF.sample: <cos(theta)> = 2/3 (cosine law)',
      t_lambertian_bsdf_sample_distribution_matches_cos_law)
H.run('Gaussian BSDF.sample: > 99 percent within 3-sigma',
      t_gaussian_bsdf_concentrates_within_sigma)


# =====================================================================
# Regression: BSDFModel abstract base (4.0.1)
# =====================================================================

H.section('Regression: BSDFModel is an explicit ABC (4.0.1)')


def t_bsdf_base_class_raises_typeerror_at_instantiation():
    """In 4.0 and earlier, ``BSDFModel`` was a regular class whose
    ``evaluate``/``sample`` methods raised NotImplementedError only
    when called -- bad UX for users who accidentally instantiated
    the base.  4.0.1 promotes it to ``abc.ABC`` so direct
    instantiation fails immediately."""
    try:
        la.BSDFModel()
        return False, 'no exception raised'
    except TypeError as e:
        return 'abstract' in str(e).lower(), str(e)[:120]


H.run('BSDFModel: direct instantiation raises TypeError',
      t_bsdf_base_class_raises_typeerror_at_instantiation)


def t_bsdf_concrete_subclasses_still_instantiate():
    """The ABC change must not break the existing concrete classes."""
    classes_ok = []
    classes_ok.append(la.LambertianBSDF(rho=0.5))
    classes_ok.append(la.GaussianBSDF(sigma_rad=0.01))
    classes_ok.append(la.HarveyShackBSDF(b0=1.0, l=0.01, s=2.0))
    return all(c is not None for c in classes_ok), \
        f'all {len(classes_ok)} concrete BSDFs instantiated OK'


H.run('BSDF concrete subclasses still instantiable after ABC promotion',
      t_bsdf_concrete_subclasses_still_instantiate)


if __name__ == '__main__':
    sys.exit(H.summary())
