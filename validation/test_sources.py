"""Tests for beam/source generators.

From:
- physics_deep_test.py (tilted plane wave, point source)
- physics_exhaustive_test.py (HG, LG, Gaussian power)
- new_features_deep_test.py (top-hat, annular, Bessel, LED, fiber)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('sources')


def t_gaussian_beam_power():
    N = 512; dx = 2e-6; w0 = 100e-6
    E, _, _ = op.create_gaussian_beam(N, dx, w0)
    I = np.abs(E)**2
    P_measured = float(np.sum(I) * dx**2)
    peak = np.abs(E).max()
    return P_measured > 0 and abs(peak - 1.0) < 1e-6, \
        f'P = {P_measured:.4e}, peak = {peak:.6f}'


H.run('Gaussian: peak-normalised to 1.0', t_gaussian_beam_power)


def t_hg_mode_tem01():
    E, _, _ = op.create_hermite_gauss(256, 4e-6, 50e-6,
                                      wavelength=1.31e-6, m=0, n=1)
    I = np.abs(E)**2
    col = I[:, 128]
    center_val = col[128]
    off_center = max(col[128+15], col[128-15])
    return center_val < off_center * 0.1, \
        f'center={center_val:.4e}, off-center(y)={off_center:.4e}'


H.run('HG mode: TEM01 has null at center', t_hg_mode_tem01)


def t_lg_mode_oam():
    E, _, _ = op.create_laguerre_gauss(256, 4e-6, 50e-6,
                                       wavelength=1.31e-6, p=0, l=1)
    amp_center = np.abs(E[128, 128])
    amp_ring = np.abs(E[128, 128+15])
    return amp_center < amp_ring * 0.1, \
        f'center={amp_center:.4e}, ring={amp_ring:.4e}'


H.run('LG mode: l=1 has vortex (zero at center)', t_lg_mode_oam)


def t_tilted_plane_wave_phase():
    N = 256; dx = 4e-6; lam = 1.31e-6; angle = 0.02
    E, _, _ = op.create_tilted_plane_wave(N, dx, lam, angle_x=angle)
    row = E[N//2, :]
    phase = np.unwrap(np.angle(row))
    k = 2 * np.pi / lam
    expected_gradient = k * np.sin(angle)
    actual_gradient = np.mean(np.diff(phase) / dx)
    err_pct = abs(actual_gradient - expected_gradient) / expected_gradient * 100
    return err_pct < 0.1, f'gradient err = {err_pct:.4f}%'


H.run('Tilted plane wave: correct phase gradient',
      t_tilted_plane_wave_phase)


def t_point_source_divergence():
    N = 256; dx = 4e-6; lam = 1.31e-6
    E, _, _ = op.create_point_source(N, dx, lam, z0=-10e-3)
    I = np.abs(E)**2
    I_center = I[N//2, N//2]
    I_edge = I[N//2, -1]
    return I_center > I_edge, \
        f'I_center={I_center:.3e} > I_edge={I_edge:.3e}'


H.run('Point source: center brighter than edge',
      t_point_source_divergence)


def t_tophat_profile():
    E, _, _ = op.create_top_hat_beam(128, 4e-6, 0.2e-3)
    I = np.abs(E)**2
    center = I[64, 64]
    edge = I[64, 0]
    return center > 0 and edge == 0, \
        f'center={center:.3e}, edge={edge:.3e}'


H.run('Top-hat: uniform inside, zero outside', t_tophat_profile)


def t_annular_hole():
    E, _, _ = op.create_annular_beam(128, 4e-6, 0.4e-3, 0.2e-3)
    I = np.abs(E)**2
    center = I[64, 64]
    ring = I[64, 64 + 38]
    return center == 0 and ring > 0, \
        f'center={center:.3e}, ring(150um)={ring:.3e}'


H.run('Annular: zero at center, nonzero in ring', t_annular_hole)


def t_bessel_central_peak():
    E, _, _ = op.create_bessel_beam(128, 4e-6, 1.31e-6, cone_angle=0.05)
    I = np.abs(E)**2
    return I[64, 64] == I.max(), \
        f'center is max: {I[64,64]==I.max()}'


H.run('Bessel: J0 peaks at center', t_bessel_central_peak)


def t_led_source_angles():
    E, angles, _, _ = op.create_led_source(64, 16e-6, 100e-6, 0.3,
                                           1.31e-6)
    max_angle = max(np.sqrt(a[0]**2 + a[1]**2) for a in angles)
    return max_angle <= 0.31 and len(angles) > 10, \
        f'{len(angles)} angles, max = {max_angle:.4f} rad'


H.run('LED: angles within divergence cone', t_led_source_angles)


def t_fiber_mode_gaussian():
    E, _, _ = op.create_fiber_mode(256, 2e-6, 10e-6, 1.31e-6)
    I = np.abs(E)**2
    return I.max() > 0 and I[128, 128] == I.max(), \
        'fiber mode peaked at center'


H.run('Fiber: peaked Gaussian at center', t_fiber_mode_gaussian)


# ---------------------------------------------------------------------
# Additional sources-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------

def t_gaussian_beam_d4sigma_matches_w0():
    """A Gaussian beam created with waist w0 has D4-sigma diameter = 2*w0
    (the 1/e^2 diameter is 2*w0)."""
    N, dx, w0 = 512, 2e-6, 100e-6
    E, _, _ = op.create_gaussian_beam(N, dx, w0)
    d4x, d4y = op.beam_d4sigma(np.abs(E)**2, dx)
    # D4-sigma equals 2 * w0 for Gaussian.  Sampling pushes a small bias.
    rel = abs(d4x - 2 * w0) / (2 * w0)
    return rel < 0.02, \
        (f'D4_x={d4x*1e6:.3f}um, expect 2*w0={2*w0*1e6:.3f}um, '
         f'rel={rel*100:.2f}%')


H.run('Gaussian beam: D4-sigma matches 2*w0',
      t_gaussian_beam_d4sigma_matches_w0)


def t_top_hat_intensity_uniform_inside():
    """A top-hat beam has near-uniform intensity inside its aperture."""
    N, dx, R = 256, 4e-6, 200e-6
    E, _, _ = op.create_top_hat_beam(N, dx, R)
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    inside = (X**2 + Y**2) < (0.5 * R)**2
    inside_vals = np.abs(E[inside])
    rel_std = float(np.std(inside_vals)) / max(float(np.mean(inside_vals)), 1e-30)
    return rel_std < 0.05, f'inside rel std = {rel_std:.4f}'


H.run('Top-hat: uniform intensity well inside the aperture',
      t_top_hat_intensity_uniform_inside)


def t_annular_beam_has_central_obscuration():
    """An annular beam has zero intensity at the center but nonzero
    in the annulus."""
    N, dx = 512, 2e-6
    D_outer, D_inner = 400e-6, 200e-6
    out = op.create_annular_beam(N, dx, D_outer, D_inner)
    E = out[0] if isinstance(out, tuple) else out
    center = float(np.abs(E[N//2, N//2]))
    # Sample at outer-radius position along x.
    R_avg = 0.25 * (D_outer + D_inner)
    annulus_idx = (N // 2) + int(R_avg / dx)
    annulus = float(np.abs(E[N//2, annulus_idx]))
    return center < 1e-6 and annulus > 0.1, \
        f'center={center:.2e}, annulus(at R_avg)={annulus:.4f}'


H.run('Annular beam: zero center, nonzero annulus',
      t_annular_beam_has_central_obscuration)


def t_tilted_plane_wave_intensity_uniform():
    """A tilted plane wave has uniform amplitude (only phase varies)."""
    N, dx, lam = 128, 4e-6, 1.31e-6
    out = op.create_tilted_plane_wave(N, dx, lam,
                                       angle_x=np.radians(0.5),
                                       angle_y=np.radians(0.0))
    E = out[0] if isinstance(out, tuple) else out
    amps = np.abs(E)
    return float(np.std(amps)) < 1e-12, \
        f'amplitude std = {float(np.std(amps)):.2e}'


H.run('Tilted plane wave: uniform |E|',
      t_tilted_plane_wave_intensity_uniform)


def t_point_source_returns_field_with_finite_amplitude():
    """A point source generator returns a field with finite,
    positive amplitude at the array center."""
    N, dx, lam = 128, 4e-6, 1.31e-6
    out = op.create_point_source(N, dx, lam, x0=0, y0=0, z0=-10.0)
    E = out[0] if isinstance(out, tuple) else out
    amp_center = float(np.abs(E[N//2, N//2]))
    return amp_center > 0 and np.all(np.isfinite(E)), \
        f'|E|_center = {amp_center:.4f}'


H.run('Point source: returns finite, positive |E|',
      t_point_source_returns_field_with_finite_amplitude)


def t_fiber_mode_normalized_total_power_finite():
    """Fiber mode generator returns a field with finite total power."""
    out = op.create_fiber_mode(N=128, dx=2e-6,
                                mode_field_diameter=10e-6,
                                wavelength=1.31e-6)
    E = out[0] if isinstance(out, tuple) else out
    P = float(np.sum(np.abs(E)**2) * (2e-6)**2)
    return np.isfinite(P) and P > 0, f'P = {P:.3e}'


H.run('Fiber mode: finite, positive total power',
      t_fiber_mode_normalized_total_power_finite)


if __name__ == '__main__':
    sys.exit(H.summary())
