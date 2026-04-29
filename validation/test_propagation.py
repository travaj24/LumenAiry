"""Free-space propagation tests (ASM, Fresnel, Fraunhofer, R-S).

From:
- physics_deep_test.py (ASM phase, ASM energy, sampling check)
- physics_complex_test.py (Talbot, ASM vs Fresnel)
- physics_exhaustive_test.py (Tilted ASM, Fraunhofer Gaussian)
- physics_extended_test.py (Rayleigh-Sommerfeld near-field)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('propagation')


def t_asm_phase_accumulation():
    N = 256; dx = 4e-6; lam = 1.31e-6; z = 10e-3
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.angular_spectrum_propagate(E_in, z, lam, dx)
    k = 2 * np.pi / lam
    expected_phase = (k * z) % (2 * np.pi)
    actual_phase = np.angle(E_out[N//2, N//2]) % (2 * np.pi)
    err = abs(actual_phase - expected_phase)
    if err > np.pi:
        err = 2 * np.pi - err
    return err < 0.01, f'phase err = {err:.4f} rad (expect < 0.01)'


H.run('ASM: plane-wave phase accumulation', t_asm_phase_accumulation)


def t_asm_energy_conservation():
    N = 512; dx = 2e-6; lam = 1.31e-6
    E_in = np.exp(-(np.arange(N)[:, None] - N//2)**2 / 100**2
                  - (np.arange(N)[None, :] - N//2)**2 / 100**2).astype(np.complex128)
    P_in = float(np.sum(np.abs(E_in)**2) * dx**2)
    E_out = op.angular_spectrum_propagate(E_in, 5e-3, lam, dx)
    P_out = float(np.sum(np.abs(E_out)**2) * dx**2)
    ratio = P_out / P_in
    return abs(ratio - 1.0) < 1e-6, f'P_out/P_in = {ratio:.8f}'


H.run('ASM: energy conservation', t_asm_energy_conservation)


def t_tilted_beam_propagation():
    N = 512; dx = 2e-6; lam = 1.31e-6; z = 5e-3; angle = 0.01
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    k = 2 * np.pi / lam
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    E_tilted = E_in * np.exp(1j * k * np.sin(angle) * X)
    E_out = op.angular_spectrum_propagate(E_tilted, z, lam, dx)
    cx_out, _ = op.beam_centroid(E_out, dx)
    expected = z * np.sin(angle)
    err = abs(cx_out - expected)
    return err < 5e-6, \
        f'shift={cx_out*1e6:.2f}um, expected={expected*1e6:.2f}um'


H.run('Tilted beam: drifts by z*sin(angle) under ASM',
      t_tilted_beam_propagation)


def t_fraunhofer_gaussian():
    N = 256; dx = 4e-6; lam = 1.31e-6; z = 1.0
    sigma = 50e-6
    E_in, _, _ = op.create_gaussian_beam(N, dx, sigma)
    result = op.fraunhofer_propagate(E_in, z, lam, dx)
    E_out = result[0] if isinstance(result, tuple) else result
    I = np.abs(E_out)**2
    peak_idx = np.unravel_index(np.argmax(I), I.shape)
    err = max(abs(peak_idx[0] - N//2), abs(peak_idx[1] - N//2))
    return err <= 1, f'Fraunhofer peak at {peak_idx}, center={N//2}'


H.run('Fraunhofer: Gaussian remains centered', t_fraunhofer_gaussian)


def t_talbot_self_imaging():
    d = 20e-6; dx = 0.5e-6
    N = int(round(40 * d / dx))
    lam = 0.633e-6
    z_T = 2 * d**2 / lam
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    grating = np.where(np.mod(X, d) < d/2, 1.0, 0.3)
    E_in = grating.astype(np.complex128)
    E_out = op.angular_spectrum_propagate(E_in, z_T, lam, dx)
    I_in = np.abs(E_in[N//2, :])**2
    I_out = np.abs(E_out[N//2, :])**2
    I_in = I_in / I_in.max()
    I_out = I_out / I_out.max()
    corr = np.corrcoef(I_in, I_out)[0, 1]
    return corr > 0.85, f'Talbot correlation = {corr:.4f}'


H.run('Talbot self-imaging at z_T = 2d^2/lam', t_talbot_self_imaging)


def t_asm_vs_fresnel():
    N = 512; dx = 4e-6; lam = 1.31e-6; z = 1e-3
    a = N * dx / 2
    F_num = a**2 / (lam * z)
    E_in, _, _ = op.create_gaussian_beam(N, dx, 100e-6)
    E_asm = op.angular_spectrum_propagate(E_in, z, lam, dx)
    fres_result = op.fresnel_propagate(E_in, z, lam, dx)
    if isinstance(fres_result, tuple):
        E_fresnel = fres_result[0]
    else:
        E_fresnel = fres_result
    I_asm_peak = np.abs(E_asm[N//2, N//2])**2
    I_fres_peak = np.abs(E_fresnel[N//2, N//2])**2
    if I_asm_peak > 0:
        ratio = I_fres_peak / I_asm_peak
        err = abs(ratio - 1.0)
    else:
        ratio = 0.0
        err = 1.0
    return err < 0.5, \
        f'Fresnel number={F_num:.0f}, peak ratio ASM/Fres={ratio:.4f}'


H.run('ASM vs Fresnel: agree at high Fresnel number', t_asm_vs_fresnel)


def t_rs_vs_asm_near_field():
    N = 256; dx = 1e-6; lam = 1.31e-6; z = 20e-6
    E_in, _, _ = op.create_gaussian_beam(N, dx, 20e-6)
    E_asm = op.angular_spectrum_propagate(E_in, z, lam, dx)
    from lumenairy.propagation import rayleigh_sommerfeld_propagate
    E_rs = rayleigh_sommerfeld_propagate(E_in, z, lam, dx)
    I_asm = np.abs(E_asm[N//2, N//2])**2
    I_rs = np.abs(E_rs[N//2, N//2])**2
    ratio = I_rs / I_asm if I_asm > 0 else 0
    return 0.9 < ratio < 1.1, f'I_RS/I_ASM = {ratio:.4f}'


H.run('Rayleigh-Sommerfeld vs ASM: near-field agreement',
      t_rs_vs_asm_near_field)


def t_sampling_check_values():
    r = op.check_opd_sampling(4e-6, 1.31e-6, 12e-3, 45e-3, verbose=False)
    marginal_ok = not r['ok'] and r['margin'] > 1.0
    r2 = op.check_opd_sampling(1e-6, 1.31e-6, 10e-3, 100e-3, verbose=False)
    safe_ok = r2['ok'] and r2['margin'] > 5
    return marginal_ok and safe_ok, \
        f'marginal margin={r["margin"]:.2f}, safe margin={r2["margin"]:.2f}'


H.run('Sampling check: correct margin classification',
      t_sampling_check_values)


def t_sas_power_conservation():
    N = 512; dx = 2e-6; lam = 1.31e-6; z = 100e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    P_in = float(np.sum(np.abs(E_in)**2) * dx**2)
    E_sas, dx_sas, dy_sas = op.scalable_angular_spectrum_propagate(
        E_in, z, lam, dx)
    P_sas = float(np.sum(np.abs(E_sas)**2) * dx_sas * dy_sas)
    ratio = P_sas / P_in
    return abs(ratio - 1.0) < 1e-6, f'P_SAS/P_in = {ratio:.8f}'


H.run('SAS: power conservation (Gaussian, z=100mm)',
      t_sas_power_conservation)


def t_sas_output_pitch_scaling():
    N = 256; dx = 2e-6; lam = 1.31e-6; z = 500e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    _, dx_sas, _ = op.scalable_angular_spectrum_propagate(
        E_in, z, lam, dx)
    expected = lam * z / (2 * N * dx)  # pad=2 default
    err = abs(dx_sas - expected) / expected
    return err < 1e-12, \
        f'dx_sas={dx_sas*1e6:.3f}um, expected={expected*1e6:.3f}um'


H.run('SAS: output pitch = lam*z/(pad*N*dx)',
      t_sas_output_pitch_scaling)


def t_sas_matches_fresnel_peak_at_moderate_z():
    N = 512; dx = 2e-6; lam = 1.31e-6; z = 500e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_fres, _, _ = op.fresnel_propagate(E_in, z, lam, dx)
    E_sas, _, _ = op.scalable_angular_spectrum_propagate(
        E_in, z, lam, dx)
    # SAS has pad=2 so pitch is Fresnel/2 -- peak amplitudes should still
    # match because the peak is independent of grid pitch for a smooth
    # Gaussian far-enough-field.
    peak_f = np.abs(E_fres).max()
    peak_s = np.abs(E_sas).max()
    rel = abs(peak_f - peak_s) / peak_f
    return rel < 1e-3, \
        f'|E|_max: Fresnel={peak_f:.3e}, SAS={peak_s:.3e}, rel={rel:.2e}'


H.run('SAS matches Fresnel peak at moderate z',
      t_sas_matches_fresnel_peak_at_moderate_z)


def t_sas_skip_final_phase_intensity_match():
    N = 256; dx = 2e-6; lam = 1.31e-6; z = 200e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_full, _, _ = op.scalable_angular_spectrum_propagate(
        E_in, z, lam, dx)
    E_skip, _, _ = op.scalable_angular_spectrum_propagate(
        E_in, z, lam, dx, skip_final_phase=True)
    I_full = np.abs(E_full)**2
    I_skip = np.abs(E_skip)**2
    rel = np.max(np.abs(I_full - I_skip)) / np.max(I_full)
    return rel < 1e-10, \
        f'max relative I diff = {rel:.2e}'


H.run('SAS: skip_final_phase preserves intensity',
      t_sas_skip_final_phase_intensity_match)


def t_sas_rejects_nonsquare_input():
    E = np.ones((64, 128), dtype=np.complex128)
    try:
        op.scalable_angular_spectrum_propagate(E, 0.1, 1e-6, 4e-6)
        return False, 'should have raised'
    except ValueError:
        return True, 'ValueError raised'


H.run('SAS: rejects non-square input', t_sas_rejects_nonsquare_input)


def t_sas_inside_propagate_through_system():
    N, dx, lam = 256, 4e-6, 1.31e-6
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    elements = [
        {'type': 'propagate', 'z': 200e-3, 'method': 'sas'},
        {'type': 'aperture', 'shape': 'circular',
         'params': {'diameter': 1e-3}},
        {'type': 'propagate', 'z': 10e-3, 'method': 'asm'},
    ]
    E_out, _ = op.propagate_through_system(E_in, elements, lam, dx)
    ok = (E_out.shape == (N, N) and np.all(np.isfinite(E_out))
          and np.abs(E_out).max() > 0)
    return ok, (f'shape={E_out.shape}, '
                f'|E|_max={np.abs(E_out).max():.3e}')


H.run('SAS works as method in propagate_through_system',
      t_sas_inside_propagate_through_system)


def t_nyquist_warning_fires():
    import warnings
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=10e-3)
    E = np.ones((512, 512), dtype=np.complex128)
    E_out = op.apply_real_lens(E, pres, 1.31e-6, 16e-6)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        op.wave_opd_1d(E_out, 16e-6, 1.31e-6,
                       aperture=10e-3, focal_length=100e-3)
    fired = any('Nyquist' in str(w.message) for w in ws)
    return fired, f'fired={fired}'


H.run('Nyquist warning fires on under-sampled grid',
      t_nyquist_warning_fires)


# ---------------------------------------------------------------------
# Additional physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------

def t_asm_linearity():
    """ASM is a linear operator: ASM(a*A + b*B) == a*ASM(A) + b*ASM(B)."""
    N, dx, lam, z = 256, 4e-6, 1.31e-6, 5e-3
    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    a, b = 0.7 + 0.3j, -1.2 - 0.4j
    lhs = op.angular_spectrum_propagate(a * A + b * B, z, lam, dx)
    rhs = a * op.angular_spectrum_propagate(A, z, lam, dx) \
        + b * op.angular_spectrum_propagate(B, z, lam, dx)
    err = np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs))
    return err < 1e-12, f'max relative error {err:.2e}'


H.run('ASM: linearity over complex superposition', t_asm_linearity)


def t_asm_round_trip_reversibility():
    """Forward then backward ASM (z then -z) should reproduce the input."""
    N, dx, lam, z = 256, 4e-6, 1.31e-6, 8e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 50e-6)
    E_fwd = op.angular_spectrum_propagate(E_in, z, lam, dx)
    E_back = op.angular_spectrum_propagate(E_fwd, -z, lam, dx)
    err = np.max(np.abs(E_back - E_in)) / np.max(np.abs(E_in))
    return err < 1e-10, f'max relative round-trip error {err:.2e}'


H.run('ASM: round-trip (forward+backward) recovers input',
      t_asm_round_trip_reversibility)


def t_propagation_preserves_centroid_under_symmetric_grid():
    """For a centered Gaussian on a symmetric grid, the centroid stays
    at the origin after free-space propagation."""
    N, dx, lam, z = 256, 4e-6, 1.31e-6, 4e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 80e-6)
    E_out = op.angular_spectrum_propagate(E_in, z, lam, dx)
    cx, cy = op.beam_centroid(E_out, dx)
    return abs(cx) < 1e-7 and abs(cy) < 1e-7, \
        f'centroid=({cx*1e6:.3f}um, {cy*1e6:.3f}um)'


H.run('ASM: centered Gaussian centroid stays at origin',
      t_propagation_preserves_centroid_under_symmetric_grid)


def t_propagation_d4sigma_grows_in_far_field():
    """A focused Gaussian's D4-sigma must grow when propagated past
    its waist into the far field."""
    N, dx, lam = 512, 2e-6, 1.31e-6
    E_in, _, _ = op.create_gaussian_beam(N, dx, 30e-6)
    d4_in = op.beam_d4sigma(np.abs(E_in)**2, dx)
    E_far = op.angular_spectrum_propagate(E_in, 50e-3, lam, dx)
    d4_far = op.beam_d4sigma(np.abs(E_far)**2, dx)
    grew = d4_far[0] > d4_in[0] and d4_far[1] > d4_in[1]
    return grew, (f'D4_in={d4_in[0]*1e6:.1f}um, '
                  f'D4_far={d4_far[0]*1e6:.1f}um')


H.run('ASM: D4-sigma grows in far field for focused Gaussian',
      t_propagation_d4sigma_grows_in_far_field)


def t_zero_distance_propagation_is_identity():
    """Propagating by z=0 must return exactly the input field."""
    N, dx, lam = 128, 4e-6, 1.31e-6
    E_in, _, _ = op.create_gaussian_beam(N, dx, 40e-6)
    E_out = op.angular_spectrum_propagate(E_in, 0.0, lam, dx)
    err = np.max(np.abs(E_out - E_in))
    return err < 1e-12, f'|E_out - E_in|_max = {err:.2e}'


H.run('ASM: z=0 returns input exactly',
      t_zero_distance_propagation_is_identity)


def t_check_sampling_conditions_returns_diagnostics():
    """check_sampling_conditions runs without error on a typical setup
    and returns a dict with at least an 'ok' key."""
    rep = op.check_sampling_conditions(256, 4e-6, 10e-3, 1.31e-6,
                                        verbose=False)
    ok = (isinstance(rep, dict)
          and 'nyquist_ok' in rep and 'fresnel_ok' in rep)
    return ok, f'returned keys = {list(rep.keys())[:6]}'


H.run('check_sampling_conditions returns dict with ok key',
      t_check_sampling_conditions_returns_diagnostics)


if __name__ == '__main__':
    sys.exit(H.summary())
