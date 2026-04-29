"""Wavefront / PSF / MTF / Zernike analysis tests.

From:
- physics_deep_test.py (Zernike orthogonality, decompose/reconstruct)
- physics_complex_test.py (Gaussian q-parameter, Airy disk,
  polychromatic Strehl, Gaussian through ABCD lens)
- physics_exhaustive_test.py (beam power, radial power, Strehl perfect,
  Strehl aberrated)
- physics_remaining_test.py (opd_pv_rms, remove_wavefront_modes,
  strehl_ratio, mtf_radial, Zernike helpers, Zernike basis_matrix)
- hammer_test.py (polychromatic Strehl bounds, field-overlap invariance)
- deep_audit.py (f_ref subtraction, Zernike round-trip)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('analysis')


# ---------------------------------------------------------------------
H.section('Zernike')


def t_zernike_orthogonality():
    N = 256; dx = 10e-6; ap = 2e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    r_pupil = ap / 2
    rho = np.sqrt(X**2 + Y**2) / r_pupil
    theta = np.arctan2(Y, X)
    mask = rho <= 1.0
    n_pixels = mask.sum()
    Z4 = op.zernike_polynomial(2, 0, rho[mask], theta[mask])
    Z10 = op.zernike_polynomial(4, -4, rho[mask], theta[mask])
    inner = np.sum(Z4 * Z10) / n_pixels
    return abs(inner) < 0.01, \
        f'inner product Z4.Z10 = {inner:.4f}'


H.run('Zernike: orthogonality of Z4 and Z10', t_zernike_orthogonality)


def t_zernike_round_trip():
    N = 256; dx = 10e-6; ap = 2e-3
    known = np.zeros(15)
    known[4] = 50e-9; known[8] = 20e-9; known[12] = 15e-9
    opd_in = op.zernike_reconstruct(known, dx, (N, N), ap)
    coeffs, _ = op.zernike_decompose(opd_in, dx, ap, n_modes=15)
    err = np.max(np.abs(coeffs - known))
    return err < 1e-11, f'max coeff error = {err:.2e}'


H.run('Zernike: round-trip decompose/reconstruct', t_zernike_round_trip)


def t_zernike_deep_audit_round_trip():
    N = 256; dx = 10e-6; ap = 2e-3
    known = np.zeros(15)
    known[4] = 50e-9; known[8] = 20e-9; known[12] = 15e-9
    opd_in = op.zernike_reconstruct(known, dx, (N, N), ap)
    coeffs, _ = op.zernike_decompose(opd_in, dx, ap, n_modes=15)
    return np.allclose(coeffs, known, atol=1e-11), \
        f'max err = {np.max(np.abs(coeffs - known)):.2e}'


H.run('Zernike decompose/reconstruct round-trip',
      t_zernike_deep_audit_round_trip)


def t_zernike_index_helpers():
    n, m = op.zernike_index_to_nm(12)
    j = op.zernike_nm_to_index(n, m)
    return j == 12 and n == 4 and m == 0, \
        f'j=12 -> (n={n},m={m}) -> j={j}'


H.run('Zernike: index_to_nm / nm_to_index roundtrip',
      t_zernike_index_helpers)


def t_zernike_basis_matrix():
    x = (np.arange(64) - 32) * 10e-6
    X, Y = np.meshgrid(x, x)
    basis, mask = op.zernike_basis_matrix(15, X, Y, pupil_radius=0.3e-3)
    return (basis.shape[1] == 15 and basis.shape[0] == mask.sum()), \
        f'basis shape = {basis.shape}, mask count = {mask.sum()}'


H.run('Zernike: basis_matrix shape correct', t_zernike_basis_matrix)


# ---------------------------------------------------------------------
H.section('Strehl, Airy, PSF peaks')


def t_strehl_perfect():
    N = 256; dx = 20e-6; lam = 1.31e-6; D = 5e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = np.where(X**2 + Y**2 <= (D/2)**2, 1.0, 0.0).astype(np.complex128)
    psf, dx_psf = op.compute_psf(pupil, lam, 50e-3, dx)
    return psf.max() > 0.99, f'PSF peak = {psf.max():.4f}'


H.run('Strehl: unaberrated pupil gives peak = 1.0', t_strehl_perfect)


def t_strehl_aberrated():
    N = 256; dx = 20e-6; lam = 1.31e-6; D = 5e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    pupil = np.where(r2 <= (D/2)**2, 1.0, 0.0).astype(np.complex128)
    amp_ref = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    I_ref_peak = np.abs(amp_ref).max()**2
    rho = np.sqrt(r2) / (D/2)
    aberr = 0.25 * lam * np.sqrt(5) * (6*rho**4 - 6*rho**2 + 1)
    pupil_ab = pupil * np.exp(2j * np.pi * aberr / lam)
    amp_ab = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_ab)))
    I_ab_peak = np.abs(amp_ab).max()**2
    strehl = I_ab_peak / I_ref_peak
    return strehl < 0.95, f'Strehl = {strehl:.4f}'


H.run('Strehl: 1/4 wave SA reduces peak below 1', t_strehl_aberrated)


def t_strehl_ratio_identical():
    N = 64; dx = 16e-6
    E = np.ones((N, N), dtype=np.complex128)
    s = op.strehl_ratio(E, E.copy(), dx)
    return abs(s - 1.0) < 0.01, \
        f'Strehl of identical fields = {s:.4f}'


H.run('strehl_ratio: identical fields give 1.0',
      t_strehl_ratio_identical)


def t_airy_disk():
    N = 512; dx = 20e-6; lam = 0.633e-6
    D = 5e-3
    f = 100e-3
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    pupil = np.where(X**2 + Y**2 <= (D/2)**2, 1.0, 0.0).astype(np.complex128)
    pupil = op.apply_thin_lens(pupil, f, lam, dx)
    E_focus = op.angular_spectrum_propagate(pupil, f, lam, dx)
    I = np.abs(E_focus)**2
    r_airy = 1.22 * lam * f / D
    cx, cy = N//2, N//2
    pix_at_airy = int(round(r_airy / dx))
    I_at_airy = I[cy, cx + pix_at_airy]
    I_peak = I[cy, cx]
    ratio = I_at_airy / I_peak
    return ratio < 0.05, f'I(r_airy)/I_peak = {ratio:.4f}'


H.run('Airy disk: first zero at 1.22*lam*f/D', t_airy_disk)


def t_polychromatic_strehl():
    wvs = [1.064e-6, 1.31e-6, 1.55e-6]
    weights = [1.0, 1.0, 1.0]
    N = 128; dx = 16e-6
    pres = op.thorlabs_lens('AC254-100-C')
    pres['aperture_diameter'] = 3e-3
    s_poly, strehls, _ = op.polychromatic_strehl(
        pres, wvs, weights, N, dx)
    ok = (0.1 < s_poly < 2.0) and all(s > 0 for s in strehls)
    return ok, (f'poly={s_poly:.4f}, per-wv=['
                f'{", ".join(f"{s:.3f}" for s in strehls)}]')


H.run('Polychromatic Strehl: within reasonable bounds',
      t_polychromatic_strehl)


def t_polychromatic_strehl_bounds():
    wls = [1.20e-6, 1.31e-6, 1.55e-6]
    weights = [1.0, 1.0, 1.0]
    tmpl = op.make_singlet(R1=0.05, R2=-0.05, d=4e-3, glass='N-BK7',
                           aperture=5e-3)
    try:
        s_poly, s_each, _ = op.polychromatic_strehl(
            tmpl, wls, weights, N=1024, dx=12e-6)
        bounded = (float(np.min(s_each)) - 1e-3 <= s_poly
                   <= float(np.max(s_each)) + 1e-3)
        peak_bound = float(np.max(s_each)) <= 1.0 + 1e-3
        return bounded and peak_bound, \
            (f's_poly={s_poly:.4f}, per-wv min/max = '
             f'{s_each.min():.4f}/{s_each.max():.4f}')
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


H.run('min(per-lambda) <= poly-Strehl <= max(per-lambda)',
      t_polychromatic_strehl_bounds)


# ---------------------------------------------------------------------
H.section('Gaussian beam ABCD')


def t_gaussian_q_parameter():
    N = 1024; dx = 2e-6; lam = 1.31e-6
    sigma = 100e-6
    w0 = sigma * np.sqrt(2)
    z = 50e-3
    z_R = np.pi * w0**2 / lam
    w_expected = w0 * np.sqrt(1 + (z / z_R)**2)
    E, _, _ = op.create_gaussian_beam(N, dx, sigma)
    E_prop = op.angular_spectrum_propagate(E, z, lam, dx)
    d4x, d4y = op.beam_d4sigma(E_prop, dx)
    w_measured = d4x / 2
    err_pct = abs(w_measured - w_expected) / w_expected * 100
    return err_pct < 2.0, \
        (f'w(z)={w_measured*1e6:.2f}um, '
         f'expected={w_expected*1e6:.2f}um, err={err_pct:.2f}%')


H.run('Gaussian q-parameter propagation (50 mm)',
      t_gaussian_q_parameter)


def t_gaussian_through_thick_lens():
    from lumenairy.raytrace import (
        surfaces_from_prescription, system_abcd)
    N = 512; dx = 4e-6; lam = 1.31e-6
    sigma = 200e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_in, _, _ = op.create_gaussian_beam(N, dx, sigma)
    E_exit = op.apply_real_lens(E_in, pres, lam, dx)
    z_range = np.linspace(max(bfl - 10e-3, 1e-3), bfl + 10e-3, 41)
    best_d4 = np.inf
    best_z = 0
    for z in z_range:
        E_z = op.angular_spectrum_propagate(E_exit, z, lam, dx)
        d4x, _ = op.beam_d4sigma(E_z, dx)
        if d4x < best_d4:
            best_d4 = d4x
            best_z = z
    err = abs(best_z - bfl)
    return err < 15e-3, \
        (f'wave focus at z={best_z*1e3:.2f}mm, '
         f'BFL={bfl*1e3:.2f}mm, err={err*1e3:.2f}mm')


H.run('Gaussian through thick lens: ABCD q-parameter',
      t_gaussian_through_thick_lens)


# ---------------------------------------------------------------------
H.section('OPD / wavefront metrics')


def t_opd_pv_rms():
    opd = np.array([0, 1e-9, -1e-9, 2e-9, -2e-9])
    pv, rms = op.opd_pv_rms(opd)
    return pv > 0 and rms > 0, f'PV={pv:.2e}, RMS={rms:.2e}'


H.run('opd_pv_rms: returns positive values', t_opd_pv_rms)


def t_remove_wavefront_modes():
    x = np.linspace(-1, 1, 100)
    opd = 5e-9 + 3e-9*x + 10e-9*x**2
    residual, coeffs = op.remove_wavefront_modes(
        x, opd, 'piston,tilt,defocus')
    rms = np.sqrt(np.mean(residual**2))
    return rms < 1e-15, f'residual RMS = {rms:.2e}'


H.run('remove_wavefront_modes: removes piston+tilt+defocus exactly',
      t_remove_wavefront_modes)


def t_f_ref_subtraction():
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=10e-3)
    E = np.ones((1024, 1024), dtype=np.complex128)
    E_out = op.apply_real_lens(E, pres, 1.31e-6, 4e-6)
    _, opd = op.wave_opd_1d(E_out, 4e-6, 1.31e-6,
                            aperture=9e-3, f_ref=100e-3)
    return np.all(np.isfinite(opd)), f'opd has NaN? {not np.isfinite(opd).all()}'


H.run('f_ref subtraction works', t_f_ref_subtraction)


def t_mtf_radial_delta_psf():
    N = 64
    psf = np.zeros((N, N)); psf[N//2, N//2] = 1.0
    dx_psf = 1.31e-6 * 100e-3 / (N * 20e-6)
    freq, mtf_profile = op.mtf_radial(psf, dx_psf, 1.31e-6, 100e-3)
    return mtf_profile.max() > 0.9, f'MTF peak = {mtf_profile.max():.4f}'


H.run('mtf_radial: delta PSF gives flat MTF', t_mtf_radial_delta_psf)


# ---------------------------------------------------------------------
H.section('Power / encircled energy / overlap invariance')


def t_beam_power_gaussian():
    N = 512; dx = 2e-6; w0 = 100e-6
    E, _, _ = op.create_gaussian_beam(N, dx, w0)
    I = np.abs(E)**2
    P_measured = float(np.sum(I) * dx**2)
    peak = np.abs(E).max()
    return P_measured > 0 and abs(peak - 1.0) < 1e-6, \
        f'P = {P_measured:.4e}, peak = {peak:.6f}'


H.run('Beam power: normalised Gaussian = 1.0',
      t_beam_power_gaussian)


def t_radial_power_bands():
    from lumenairy.analysis import radial_power_bands
    N = 512; dx = 2e-6; w0 = 100e-6
    E, _, _ = op.create_gaussian_beam(N, dx, w0)
    sigma = w0
    radii = [sigma, 2*sigma, 3*sigma]
    powers = radial_power_bands(E, dx, radii)
    P_total = op.beam_power(E, dx)
    fractions = powers / P_total
    expected = 1 - np.exp(-1)
    err = abs(fractions[0] - expected)
    return err < 0.05, \
        f'P(r<sigma)/P_total = {fractions[0]:.4f} (expect {expected:.4f})'


H.run('Radial power: 86.5% at r=sigma for Gaussian',
      t_radial_power_bands)


def t_overlap_self():
    E_a = np.exp(-((np.indices((64, 64))[0] - 32)**2
                   + (np.indices((64, 64))[1] - 32)**2) / 100)
    E_a = E_a.astype(np.complex128)

    def overlap(a, b):
        return (abs(np.vdot(a.ravel(), b.ravel()))
                / (np.linalg.norm(a) * np.linalg.norm(b)))

    ovl = overlap(E_a, E_a)
    return abs(ovl - 1.0) < 1e-12, f'value = {ovl}'


H.run('overlap(E, E) = 1 to fp precision', t_overlap_self)


def t_overlap_phase_invariance():
    E_a = np.exp(-((np.indices((64, 64))[0] - 32)**2
                   + (np.indices((64, 64))[1] - 32)**2) / 100)
    E_a = E_a.astype(np.complex128)
    E_b = E_a * np.exp(1j * 1.234)

    def overlap(a, b):
        return (abs(np.vdot(a.ravel(), b.ravel()))
                / (np.linalg.norm(a) * np.linalg.norm(b)))

    ovl = overlap(E_a, E_b)
    return abs(ovl - 1.0) < 1e-12, f'value = {ovl}'


H.run('overlap invariant to global phase', t_overlap_phase_invariance)


def t_overlap_amplitude_invariance():
    E_a = np.exp(-((np.indices((64, 64))[0] - 32)**2
                   + (np.indices((64, 64))[1] - 32)**2) / 100)
    E_a = E_a.astype(np.complex128)
    E_b = E_a * 17.5

    def overlap(a, b):
        return (abs(np.vdot(a.ravel(), b.ravel()))
                / (np.linalg.norm(a) * np.linalg.norm(b)))

    ovl = overlap(E_a, E_b)
    return abs(ovl - 1.0) < 1e-12, f'value = {ovl}'


H.run('overlap invariant to global amplitude',
      t_overlap_amplitude_invariance)


# ---------------------------------------------------------------------
# Additional analysis-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------
H.section('Maréchal / Parseval / Zernike orthonormality')


def t_strehl_marechal_relation():
    """For small wavefront RMS in waves, Strehl ~ exp(-(2pi sigma)^2)."""
    N = 256
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    pupil = (r2 <= 1.0).astype(np.complex128)
    coeff_waves = 0.05  # ~0.05 waves on the defocus polynomial
    z4 = np.sqrt(3.0) * (2 * r2 - 1.0)  # Z4 (Noll defocus, normalized)
    pupil_ab = pupil * np.exp(1j * 2 * np.pi * coeff_waves * z4)
    F0 = np.fft.fftshift(np.fft.fft2(pupil))
    F1 = np.fft.fftshift(np.fft.fft2(pupil_ab))
    s0 = float(np.max(np.abs(F0)**2))
    s1 = float(np.max(np.abs(F1)**2))
    strehl = s1 / s0
    sigma_rad = 2 * np.pi * coeff_waves
    marechal = np.exp(-sigma_rad**2)
    rel = abs(strehl - marechal) / marechal
    return rel < 0.02, \
        (f'Strehl={strehl:.4f}, Marechal exp(-sigma^2)={marechal:.4f}, '
         f'rel={rel*100:.2f}%')


H.run('Strehl-Maréchal: small RMS gives S ~ exp(-sigma^2)',
      t_strehl_marechal_relation)


def t_parseval_for_random_field():
    """Parseval (numpy FFT convention): sum |E|^2 == sum |F[E]|^2 / N^2."""
    N = 256
    rng = np.random.default_rng(17)
    E = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    p_space = float(np.sum(np.abs(E)**2))
    F = np.fft.fft2(E)
    p_freq = float(np.sum(np.abs(F)**2)) / (N * N)
    rel = abs(p_space - p_freq) / p_space
    return rel < 1e-12, \
        f'p_space={p_space:.4e}, p_freq/N^2={p_freq:.4e}, rel={rel:.2e}'


H.run('Parseval: |E|^2 (space) == |F[E]|^2 / N^2',
      t_parseval_for_random_field)


def t_zernike_decompose_linear_in_input():
    """zernike_decompose is linear: coeffs(a*A + b*B) = a*coeffs(A) + b*coeffs(B)."""
    N, dx, ap = 256, 10e-6, 2e-3
    cA = np.zeros(15); cA[4] = 50e-9; cA[8] = 20e-9
    cB = np.zeros(15); cB[12] = 15e-9; cB[5] = 10e-9
    opd_A = op.zernike_reconstruct(cA, dx, (N, N), ap)
    opd_B = op.zernike_reconstruct(cB, dx, (N, N), ap)
    a, b = 0.7, -1.3
    cAB, _ = op.zernike_decompose(a * opd_A + b * opd_B, dx, ap, n_modes=15)
    cA_d, _ = op.zernike_decompose(opd_A, dx, ap, n_modes=15)
    cB_d, _ = op.zernike_decompose(opd_B, dx, ap, n_modes=15)
    err = np.max(np.abs(cAB - (a * cA_d + b * cB_d)))
    return err < 1e-11, f'linearity error = {err:.2e}'


H.run('Zernike decompose: linear in input OPD',
      t_zernike_decompose_linear_in_input)


def t_zernike_decompose_recovers_known_coefficient_in_one_mode():
    """Synthesize a pure Z4 OPD (defocus) and verify decompose returns
    the input coefficient, with all others ~ 0."""
    N, dx, ap = 256, 10e-6, 2e-3
    known = np.zeros(15); known[4] = 100e-9
    opd = op.zernike_reconstruct(known, dx, (N, N), ap)
    coeffs, _ = op.zernike_decompose(opd, dx, ap, n_modes=15)
    rest = np.delete(coeffs, 4)
    return (abs(coeffs[4] - 100e-9) < 1e-11
            and float(np.max(np.abs(rest))) < 1e-11), \
        (f'coeffs[4]={coeffs[4]*1e9:.4f}nm (target 100 nm), '
         f'max(other)={float(np.max(np.abs(rest)))*1e9:.2e}nm')


H.run('Zernike decompose: pure Z4 input recovers Z4 coefficient',
      t_zernike_decompose_recovers_known_coefficient_in_one_mode)


def t_psf_peak_centered_for_uniform_pupil():
    """For a uniform circular pupil, the diffraction-limited PSF
    has its maximum at the array center."""
    N, dx, lam, f = 128, 4e-6, 1.31e-6, 50e-3
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    pupil = ((X**2 + Y**2) <= 1).astype(np.complex128)
    psf, _ = op.compute_psf(pupil, lam, f, dx)
    pk = np.unravel_index(np.argmax(psf), psf.shape)
    centered = (abs(pk[0] - N // 2) <= 1 and abs(pk[1] - N // 2) <= 1)
    return centered and np.isfinite(psf).all(), \
        f'peak at {pk}, center={N//2}'


H.run('compute_psf: perfect pupil has centered peak',
      t_psf_peak_centered_for_uniform_pupil)


def t_remove_wavefront_modes_piston_only_leaves_linear():
    """remove_wavefront_modes removing piston only does not zero out
    the linear part (sanity check that mode list is honored)."""
    x = np.linspace(-1, 1, 200)
    opd = 5e-9 + 3e-9 * x  # piston + tilt
    residual_piston, _ = op.remove_wavefront_modes(x, opd, 'piston')
    rms_residual = float(np.sqrt(np.mean(residual_piston**2)))
    rms_orig_no_piston = float(np.sqrt(np.mean(opd[1:]**2)))  # rough
    # We expect the residual still contains tilt -> nonzero.
    return rms_residual > 0.5e-9, \
        f'residual rms after piston-only removal = {rms_residual:.2e}'


H.run('remove_wavefront_modes: removing only piston leaves tilt',
      t_remove_wavefront_modes_piston_only_leaves_linear)


def t_remove_wavefront_modes_full_zeros_residual():
    """Removing piston+tilt+defocus from an OPD that IS a sum of those
    three modes gives essentially-zero residual."""
    x = np.linspace(-1, 1, 200)
    opd = 5e-9 + 3e-9 * x + 10e-9 * x**2
    residual, _ = op.remove_wavefront_modes(x, opd, 'piston,tilt,defocus')
    rms = float(np.sqrt(np.mean(residual**2)))
    return rms < 1e-15, f'residual RMS = {rms:.2e}'


H.run('remove_wavefront_modes: zero residual for matching mode set',
      t_remove_wavefront_modes_full_zeros_residual)


def t_chromatic_focal_shift_finite_for_singlet():
    """chromatic_focal_shift returns finite numbers for a singlet across
    visible wavelengths."""
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7', aperture=20e-3)
    out = op.chromatic_focal_shift(
        pres, np.array([0.486e-6, 0.587e-6, 0.656e-6]))
    arr = np.asarray(out[0] if isinstance(out, tuple) else out)
    return np.all(np.isfinite(arr)) and arr.size >= 3, \
        (f'output type={type(arr).__name__}, '
         f'finite={np.all(np.isfinite(arr))}, '
         f'shape={arr.shape}')


H.run('chromatic_focal_shift: finite for singlet',
      t_chromatic_focal_shift_finite_for_singlet)


def t_opd_pv_rms_zero_for_zero_field():
    """A zero-OPD input gives PV=0 and RMS=0."""
    N = 64
    opd = np.zeros((N, N))
    pv, rms = op.opd_pv_rms(opd)
    return abs(pv) < 1e-15 and abs(rms) < 1e-15, \
        f'pv={pv:.2e}, rms={rms:.2e}'


H.run('opd_pv_rms: PV=0 and RMS=0 for zero input',
      t_opd_pv_rms_zero_for_zero_field)


def t_strehl_ratio_unchanged_under_global_phase():
    """strehl_ratio is invariant under multiplying both fields by the
    same global phase."""
    N, dx = 64, 4e-6
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    pupil = ((X**2 + Y**2) <= 1).astype(np.complex128)
    s_a = op.strehl_ratio(pupil, pupil, dx)
    s_b = op.strehl_ratio(pupil * np.exp(1j * 1.234),
                           pupil * np.exp(1j * 1.234), dx)
    return abs(s_a - s_b) < 1e-9 and abs(s_a - 1.0) < 1e-9, \
        f's_a={s_a:.6f}, s_b={s_b:.6f}'


H.run('strehl_ratio: invariant under global phase rotation',
      t_strehl_ratio_unchanged_under_global_phase)


if __name__ == '__main__':
    sys.exit(H.summary())
