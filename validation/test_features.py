"""Tests for the specialty-feature subsystems.

Covers coatings, interferometry, freeform surfaces, ghost analysis,
RCWA, multi-configuration / afocal.  Content taken wholesale from
``new_features_deep_test.py`` (minus the tests that are more naturally
homed in test_sources.py, test_optimize.py, or test_integration.py).
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd,
)


H = Harness('features')
lam = 1.31e-6


# ---------------------------------------------------------------------
H.section('Thin-film coatings')


def t_coating_qw_ar_zero():
    layers = op.quarter_wave_ar(1.52, lam)
    R, T, _ = op.coating_reflectance(layers, [lam], n_substrate=1.52)
    return R[0] < 1e-10, f'R = {R[0]:.2e}'


H.run('Coating: QW AR gives R=0 at design wavelength',
      t_coating_qw_ar_zero)


def t_coating_uncoated_fresnel():
    n = 1.52
    R, _, _ = op.coating_reflectance([], [lam], n_substrate=n)
    expected = ((n - 1) / (n + 1)) ** 2
    return abs(R[0] - expected) < 1e-4, \
        f'R={R[0]:.5f}, expected={expected:.5f}'


H.run('Coating: uncoated Fresnel matches formula',
      t_coating_uncoated_fresnel)


def t_coating_energy_conservation():
    layers = op.broadband_ar_v_coat(1.52, lam)
    R, T, _ = op.coating_reflectance(layers, [lam], n_substrate=1.52)
    return abs(R[0] + T[0] - 1.0) < 0.01, f'R+T = {R[0]+T[0]:.6f}'


H.run('Coating: R + T = 1 (energy conservation)',
      t_coating_energy_conservation)


def t_coating_spectral_shape():
    wvs = np.linspace(1.0e-6, 1.6e-6, 51)
    layers = op.quarter_wave_ar(1.52, 1.31e-6)
    R, _, _ = op.coating_reflectance(layers, wvs, n_substrate=1.52)
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
    fringe = op.simulate_interferogram(opd, lam)
    std = np.std(fringe)
    return std < 1e-10, f'fringe std = {std:.2e}'


H.run('Interferogram: flat OPD gives uniform field',
      t_interferogram_flat_opd)


def t_interferogram_tilt_fringes():
    N = 256; dx = 4e-6
    opd = np.zeros((N, N))
    tilt = 1.0 / (20 * dx)
    fringe = op.simulate_interferogram(opd, lam, tilt_x=tilt, dx=dx)
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
    extracted, mod = op.phase_shift_extract(
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
    sag_base = op.surface_sag_general(X**2 + Y**2, R=np.inf)
    sag_astig = op.surface_sag_xy_polynomial(
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
    sag = op.surface_sag_zernike_freeform(
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
    sag = op.surface_sag_chebyshev(
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
    sag = op.surface_sag_freeform(X, Y, sd)
    return sag.shape == (10, 10) and np.std(sag) > 0, 'dispatch OK'


H.run('Freeform: dispatcher routes correctly', t_freeform_dispatch)


def t_freeform_on_prescription_surface():
    N = 128; dx = 16e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    pres['surfaces'][0]['freeform_type'] = 'xy_polynomial'
    pres['surfaces'][0]['xy_coeffs'] = {(4, 0): 1e-12}
    x = np.linspace(-1e-3, 1e-3, 10)
    X, Y = np.meshgrid(x, x)
    sag = op.surface_sag_freeform(X, Y, pres['surfaces'][0])
    return sag.shape == (10, 10), 'freeform sag computed'


H.run('Freeform: dispatch works on prescription surface dict',
      t_freeform_on_prescription_surface)


# ---------------------------------------------------------------------
H.section('Ghost analysis')


def t_ghost_path_count():
    paths = op.enumerate_ghost_paths(4)
    expected = 4 * 3 // 2
    return len(paths) == expected, \
        f'{len(paths)} paths (expect {expected})'


H.run('Ghost: correct path count for 4 surfaces', t_ghost_path_count)


def t_ghost_intensity_ordering():
    pres = op.thorlabs_lens('AC254-100-C')
    ghosts = op.ghost_analysis(pres, lam, verbose=False)
    intensities = [g['intensity'] for g in ghosts]
    is_sorted = all(intensities[i] >= intensities[i+1]
                    for i in range(len(intensities)-1))
    return is_sorted, \
        f'sorted = {is_sorted}, brightest = {intensities[0]:.2e}'


H.run('Ghost: intensities sorted brightest-first',
      t_ghost_intensity_ordering)


def t_ghost_fresnel_consistent():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=10e-3)
    ghosts = op.ghost_analysis(pres, lam, verbose=False)
    g = ghosts[0]
    expected = g['R_i'] * g['R_j']
    return abs(g['intensity'] - expected) < 1e-10, \
        f'I={g["intensity"]:.6e}, R_i*R_j={expected:.6e}'


H.run('Ghost: intensity = R_i * R_j', t_ghost_fresnel_consistent)


def t_ghost_of_thorlabs():
    pres = op.thorlabs_lens('AC254-100-C')
    ghosts = op.ghost_analysis(pres, lam, verbose=False)
    return len(ghosts) == 3, f'{len(ghosts)} ghost paths'


H.run('Ghost analysis of Thorlabs doublet (3 paths)',
      t_ghost_of_thorlabs)


# ---------------------------------------------------------------------
H.section('RCWA')


def t_rcwa_energy_conservation():
    orders, _, T = op.rcwa_1d(1e-6, 1.5, 1.0, 1.52, 1.0, 0.5e-6,
                              0.5, lam)
    total = T.sum()
    return 0.5 < total <= 1.01, f'total T = {total:.4f}'


H.run('RCWA: transmitted efficiency sum <= 1',
      t_rcwa_energy_conservation)


def t_rcwa_zeroth_order_dominates():
    orders, _, T = op.rcwa_1d(10e-6, 1.5, 1.0, 1.52, 1.0, 0.01e-6,
                              0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] > 0.8, f'T_0 = {T[i0]:.4f}'


H.run('RCWA: zeroth order dominates for shallow grating',
      t_rcwa_zeroth_order_dominates)


def t_rcwa_deep_grating_splits():
    d_pi = lam / (2 * (1.5 - 1.0))
    orders, _, T = op.rcwa_1d(5e-6, 1.5, 1.0, 1.52, 1.0, d_pi, 0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] < 0.2, f'T_0 = {T[i0]:.4f} at pi depth'


H.run('RCWA: pi-depth suppresses zeroth order',
      t_rcwa_deep_grating_splits)


def t_rcwa_large_period_scalar():
    orders, _, T = op.rcwa_1d(100e-6, 1.5, 1.0, 1.52, 1.0, 0.1e-6,
                              0.5, lam)
    i0 = np.argmin(np.abs(orders))
    return T[i0] > 0.5, \
        f'T_0 = {T[i0]:.4f} for large-period grating'


H.run('RCWA vs scalar: large period -> zeroth order dominates',
      t_rcwa_large_period_scalar)


# ---------------------------------------------------------------------
H.section('Multi-configuration / afocal')


def t_keplerian_afocal():
    pres = op.keplerian_telescope(200e-3, 50e-3)
    mag, _ = op.afocal_angular_magnification(pres, lam)
    surfs = surfaces_from_prescription(pres)
    M_abcd, _, _, _ = system_abcd(surfs, lam)
    B = abs(M_abcd[0, 1])
    return B < 0.5, f'B = {B:.4f}, mag = {mag:.3f}'


H.run('Afocal: Keplerian telescope has B ~ 0', t_keplerian_afocal)


def t_beam_expander_magnification():
    pres = op.beam_expander_prescription(3.0, 100e-3)
    mag, _ = op.afocal_angular_magnification(pres, lam)
    expected = 1.0 / 3.0
    err = abs(abs(mag) - expected) / expected
    return err < 0.5, \
        f'mag = {mag:.3f}, expected = {expected:.3f}'


H.run('Beam expander: angular mag ~ 1/M',
      t_beam_expander_magnification)


def t_multi_config_weighted():
    from lumenairy.multiconfig import (
        Configuration, multi_config_merit)
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
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


# ---------------------------------------------------------------------
H.section('BSDF surface scatter')


def t_lambertian_tis():
    bsdf = op.LambertianBSDF(rho=0.3)
    tis = bsdf.total_integrated_scatter()
    return abs(tis - 0.3) < 1e-12, f'TIS = {tis}'


H.run('Lambertian BSDF: TIS == rho', t_lambertian_tis)


def t_lambertian_sample_hemisphere():
    bsdf = op.LambertianBSDF(rho=1.0)
    samples = bsdf.sample(np.array([0.0, 0.0, -1.0]), 500, rng=42)
    in_hemi = np.all(samples[:, 2] > 0)
    unit = np.allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-10)
    return in_hemi and unit, \
        f'in_hemi={in_hemi}, unit_vectors={unit}'


H.run('Lambertian BSDF: samples are unit vectors in +z hemisphere',
      t_lambertian_sample_hemisphere)


def t_gaussian_tis():
    bsdf = op.GaussianBSDF(sigma_rad=0.005, scattered_fraction=0.01)
    return abs(bsdf.total_integrated_scatter() - 0.01) < 1e-12, \
        f'TIS = {bsdf.total_integrated_scatter()}'


H.run('Gaussian BSDF: TIS == scattered_fraction', t_gaussian_tis)


def t_gaussian_sample_concentrates_near_specular():
    bsdf = op.GaussianBSDF(sigma_rad=0.005, scattered_fraction=0.01)
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
    bsdf = op.HarveyShackBSDF(b0=1.0, l=0.01, s=2.0)
    inc = np.array([0.0, 0.0, -1.0])
    on = bsdf.evaluate(inc, np.array([0.0, 0.0, 1.0]))
    off = bsdf.evaluate(inc, np.array([
        np.sin(np.radians(5)), 0, np.cos(np.radians(5))]))
    return on > off > 0, f'on-axis={on:.3e}, 5deg={off:.3e}'


H.run('Harvey-Shack: on-axis BSDF exceeds off-axis',
      t_harvey_shack_onaxis_greater_than_offaxis)


def t_make_bsdf_dispatch():
    b1 = op.make_bsdf({'kind': 'lambertian', 'rho': 0.5})
    b2 = op.make_bsdf({'kind': 'gaussian', 'sigma_rad': 0.001,
                       'scattered_fraction': 0.005})
    b3 = op.make_bsdf({'kind': 'harvey_shack', 'b0': 2.0,
                       'l': 0.005, 's': 1.8})
    b4 = op.make_bsdf(None)
    ok = (isinstance(b1, op.LambertianBSDF) and b1.rho == 0.5
          and isinstance(b2, op.GaussianBSDF)
          and isinstance(b3, op.HarveyShackBSDF)
          and b4 is None)
    return ok, 'all four dispatch cases handled'


H.run('make_bsdf dispatches by kind', t_make_bsdf_dispatch)


def t_make_bsdf_unknown_kind_raises():
    try:
        op.make_bsdf({'kind': 'does_not_exist'})
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('make_bsdf raises on unknown kind',
      t_make_bsdf_unknown_kind_raises)


def t_sample_scatter_rays_spawns_correct_count():
    from lumenairy.raytrace import Surface, _make_bundle
    surf = Surface(radius=np.inf, semi_diameter=10e-3,
                   bsdf=op.LambertianBSDF(rho=1.0))
    incident = _make_bundle(
        x=np.array([0, 1e-3, -1e-3, 2e-3, -2e-3]),
        y=np.array([0, 0, 0, 0, 0]),
        L=np.array([0, 0, 0, 0, 0]),
        M=np.array([0, 0, 0, 0, 0]),
        wavelength=1.31e-6)
    scatt = op.sample_scatter_rays(surf, incident, n_per_ray=3, rng=0)
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
    layers = op.quarter_wave_ar(1.52, lam)
    R, _, _ = op.coating_reflectance(layers, [lam], n_substrate=1.52)
    return R[0] < 1e-6, f'R(design) = {R[0]:.2e}'


H.run('Quarter-wave AR: near-zero R at design wavelength',
      t_quarter_wave_ar_design_lambda_minimizes_R)


def t_quarter_wave_ar_R_grows_off_design():
    """Off the AR design wavelength reflectance grows."""
    layers = op.quarter_wave_ar(1.52, lam)
    R_on, _, _ = op.coating_reflectance(layers, [lam], n_substrate=1.52)
    R_off, _, _ = op.coating_reflectance(layers, [lam * 1.5], n_substrate=1.52)
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
    out = op.simulate_interferogram(E_ref, E_obj)
    I = out[0] if isinstance(out, tuple) else out
    return float(I.max()) > 3.5 and float(I.min()) < 0.5, \
        f'I min/max = ({float(I.min()):.3f}, {float(I.max()):.3f})'


H.run('Interferogram: full contrast for unit-amp tilted reference',
      t_simulate_interferogram_fringes_full_contrast)


def t_keplerian_telescope_angular_magnification():
    """A Keplerian telescope (f1=200mm, f2=50mm) has |M| = f1 / f2."""
    rx = op.keplerian_telescope(f_objective=200e-3, f_eyepiece=50e-3,
                                  glass='N-BK7', wavelength=lam)
    out = op.afocal_angular_magnification(rx, lam)
    M = out[0] if isinstance(out, tuple) else out
    return abs(abs(M) - 200e-3 / 50e-3) < 0.05, \
        f'|M|={abs(M):.4f}, expected=4.0'


H.run('Keplerian telescope: |M| ~ f1/f2',
      t_keplerian_telescope_angular_magnification)


def t_beam_expander_prescription_returns_dict():
    """A 5x beam-expander prescription is built without error."""
    rx = op.beam_expander_prescription(
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
    sag = op.surface_sag_xy_polynomial(X, Y, R=np.inf, conic=0.0,
                                         xy_coeffs={})
    return float(np.max(np.abs(sag))) < 1e-15, \
        f'max |sag| = {float(np.max(np.abs(sag))):.2e}'


H.run('Freeform XY-poly: zero coefficients give zero sag (flat input)',
      t_freeform_xy_polynomial_zero_coefficients_returns_zero_sag)


def t_make_bsdf_lambertian_returns_lambertian_instance():
    """make_bsdf({'type':'lambertian',...}) returns a LambertianBSDF."""
    bsdf = op.make_bsdf({'kind': 'lambertian', 'rho': 1.0})
    return isinstance(bsdf, op.LambertianBSDF), \
        f'type = {type(bsdf).__name__}'


H.run('make_bsdf: lambertian factory returns LambertianBSDF',
      t_make_bsdf_lambertian_returns_lambertian_instance)


if __name__ == '__main__':
    sys.exit(H.summary())
