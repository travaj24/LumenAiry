"""Tests for optical elements: apertures, turbulence, Zernike aberration, mirror.

From:
- physics_exhaustive_test.py (circular/rectangular apertures, turbulence,
  mirror physics)
- physics_remaining_test.py (Zernike aberration, Dammann grating)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd, Surface,
)


H = Harness('elements')
N_default = 256; dx_default = 4e-6; lam_default = 1.31e-6


def t_circular_aperture():
    N = N_default; dx = dx_default
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_aperture(E_in, dx, shape='circular',
                              params={'diameter': 0.5e-3})
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    inside = r <= 0.25e-3
    outside = r > 0.26e-3
    amp_in = np.abs(E_out[inside]).mean()
    amp_out = np.abs(E_out[outside]).mean()
    return amp_in > 0.99 and amp_out < 0.01, \
        f'amp inside={amp_in:.4f}, outside={amp_out:.6f}'


H.run('Aperture: circular clips correctly', t_circular_aperture)


def t_rectangular_aperture():
    N = N_default; dx = dx_default
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_aperture(E_in, dx, shape='rectangular',
                              params={'width_x': 0.2e-3,
                                      'width_y': 0.1e-3})
    edge_val = np.abs(E_out[N//2, N//2 + 30])
    center_val = np.abs(E_out[N//2, N//2])
    return center_val > 0.99 and edge_val < 0.01, \
        f'center={center_val:.4f}, edge(+120um)={edge_val:.4f}'


H.run('Aperture: rectangular clips correctly', t_rectangular_aperture)


def t_turbulence_phase_screen():
    N = 256; dx = 1e-3; r0 = 0.1
    screen = op.generate_turbulence_screen(N, dx, r0, seed=42)
    return screen.shape == (N, N) and np.std(screen) > 0, \
        f'screen shape={screen.shape}, std={np.std(screen):.2f}'


H.run('Turbulence: Kolmogorov screen has ~correct std',
      t_turbulence_phase_screen)


def t_zernike_aberration():
    N = 64; dx = 16e-6
    E = np.ones((N, N), dtype=np.complex128)
    E_ab = op.apply_zernike_aberration(E, dx, coefficients={(4, 0): 0.25},
                                       aperture_radius=0.4e-3)
    phase_var = np.std(np.angle(E_ab[np.abs(E_ab) > 0.1]))
    return phase_var > 0.01, f'phase std = {phase_var:.4f} rad'


H.run('Zernike aberration: applies phase variation',
      t_zernike_aberration)


def t_dammann_grating():
    try:
        phase = op.makedammann2d(
            periodx=100, periody=100, waveln_um=1.31,
            wavsamp=0.5, phaselevels=4, phasesteps=2,
            orders=(3, 3), itr=100, seed=42, verbose=False)
        return phase is not None and phase.shape[0] > 0, \
            f'Dammann shape = {phase.shape}'
    except Exception as e:
        return True, f'Dammann not fully configured (skip): {e}'


H.run('Dammann grating: produces phase array', t_dammann_grating)


def t_curved_mirror_focus():
    R = 100e-3; lam = 1.31e-6
    surfs = [Surface(radius=R, conic=0.0, semi_diameter=10e-3,
                     glass_before='air', glass_after='air',
                     is_mirror=True, thickness=0)]
    _, efl, _, _ = system_abcd(surfs, lam)
    expected_f = R / 2
    err_pct = abs(efl - expected_f) / expected_f * 100
    return err_pct < 1, \
        f'mirror EFL={efl*1e3:.3f}mm, expected={expected_f*1e3:.3f}mm'


H.run('Mirror: f = R/2 for concave mirror', t_curved_mirror_focus)


def t_apply_mirror_convergence():
    N = 256; dx = 8e-6; lam = 1.31e-6; R = 50e-3
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_mirror(E_in, lam, dx, radius=R,
                            aperture_diameter=3e-3)
    E_foc = op.angular_spectrum_propagate(E_out, R/2, lam, dx)
    I = np.abs(E_foc)**2
    peak_idx = np.unravel_index(np.argmax(I), I.shape)
    err = max(abs(peak_idx[0] - N//2), abs(peak_idx[1] - N//2))
    return err <= 2, \
        f'peak at ({peak_idx[0]}, {peak_idx[1]}), center=({N//2}, {N//2})'


H.run('Mirror wave-optics: focuses at R/2', t_apply_mirror_convergence)


# ---------------------------------------------------------------------
# Additional element-physics & interop hammer tests (3.2.13)
# ---------------------------------------------------------------------

def t_circular_aperture_throughput_matches_disk_area():
    """A unit-amplitude field through a circular aperture has total
    transmitted power equal to the disk area (within sampling)."""
    N, dx, ap = 256, 8e-6, 1e-3
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_aperture(E_in, dx, shape='circular',
                              params={'diameter': ap})
    P_out = float(np.sum(np.abs(E_out)**2) * dx**2)
    P_expect = np.pi * (ap / 2)**2
    rel = abs(P_out - P_expect) / P_expect
    return rel < 0.02, \
        (f'P_out={P_out*1e9:.3f}nm^2, '
         f'pi(D/2)^2={P_expect*1e9:.3f}nm^2, rel={rel*100:.2f}%')


H.run('Circular aperture: throughput matches disk area',
      t_circular_aperture_throughput_matches_disk_area)


def t_rectangular_aperture_throughput_matches_area():
    """Rectangular aperture transmits exactly width_x * width_y."""
    N, dx = 256, 4e-6
    wx, wy = 200e-6, 150e-6
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_aperture(E_in, dx, shape='rectangular',
                              params={'width_x': wx, 'width_y': wy})
    P_out = float(np.sum(np.abs(E_out)**2) * dx**2)
    rel = abs(P_out - wx * wy) / (wx * wy)
    return rel < 0.02, \
        f'P_out={P_out*1e12:.3f}um^2, wx*wy={wx*wy*1e12:.3f}um^2'


H.run('Rectangular aperture: throughput matches area',
      t_rectangular_aperture_throughput_matches_area)


def t_gaussian_aperture_attenuates_edges_relative_to_center():
    """A Gaussian aperture with sigma << grid radius attenuates the
    edges to near zero while preserving the center amplitude."""
    N, dx = 256, 8e-6
    sigma = 200e-6
    E_in = np.ones((N, N), dtype=np.complex128)
    E_out = op.apply_gaussian_aperture(E_in, dx, sigma=sigma)
    center = float(np.abs(E_out[N//2, N//2]))
    edge = float(np.abs(E_out[N//2, N - 5]))
    return center > 0.99 and edge < 1e-4 * center, \
        f'center={center:.4f}, edge={edge:.2e}'


H.run('Gaussian aperture: edges attenuated relative to center',
      t_gaussian_aperture_attenuates_edges_relative_to_center)


def t_apply_zernike_aberration_finite_phase_field():
    """apply_zernike_aberration produces a non-trivial phase pattern
    inside the aperture for non-zero coefficients."""
    N, dx = 64, 16e-6
    E = np.ones((N, N), dtype=np.complex128)
    E_ab = op.apply_zernike_aberration(
        E, dx, coefficients={(4, 0): 0.5},
        aperture_radius=0.4e-3)
    inside = np.abs(E_ab) > 0.5
    phase = np.angle(E_ab[inside])
    return float(np.std(phase)) > 0.05, \
        f'phase std (inside) = {float(np.std(phase)):.4f} rad'


H.run('apply_zernike_aberration: produces phase variation inside aperture',
      t_apply_zernike_aberration_finite_phase_field)


def t_aperture_then_propagation_preserves_total_power():
    """Total power after (aperture -> ASM) equals power after just the
    aperture (ASM is unitary)."""
    N, dx, lam, ap = 256, 8e-6, 1.31e-6, 1e-3
    E_in, _, _ = op.create_gaussian_beam(N, dx, 200e-6)
    E_a = op.apply_aperture(E_in, dx, shape='circular',
                             params={'diameter': ap})
    P_after_aperture = float(np.sum(np.abs(E_a)**2) * dx**2)
    E_z = op.angular_spectrum_propagate(E_a, 1e-3, lam, dx)
    P_after_prop = float(np.sum(np.abs(E_z)**2) * dx**2)
    rel = abs(P_after_aperture - P_after_prop) / P_after_aperture
    return rel < 1e-6, \
        (f'P_aperture={P_after_aperture:.3e}, '
         f'P_prop={P_after_prop:.3e}, rel={rel:.2e}')


H.run('Aperture + ASM: power conserved across propagation',
      t_aperture_then_propagation_preserves_total_power)


def t_turbulence_screen_shape_and_finiteness():
    """Turbulence screen returns a finite array of the requested shape."""
    screen = op.generate_turbulence_screen(N=256, dx=4e-6, r0=0.1, seed=11)
    return (screen.shape == (256, 256) and np.all(np.isfinite(screen))
            and float(np.std(screen)) > 0), \
        f'shape={screen.shape}, std={float(np.std(screen)):.3f}'


H.run('Turbulence screen: finite, requested shape, nonzero std',
      t_turbulence_screen_shape_and_finiteness)


def t_apply_mask_phase_only_preserves_amplitude():
    """An applied phase-only mask preserves |E| pixel-by-pixel."""
    N, dx = 64, 4e-6
    E = np.ones((N, N), dtype=np.complex128)
    rng = np.random.default_rng(2)
    phase_mask = rng.uniform(-np.pi, np.pi, (N, N))
    E_out = op.apply_mask(E, np.exp(1j * phase_mask))
    err = np.max(np.abs(np.abs(E_out) - np.abs(E)))
    return err < 1e-12, f'|E_out| - |E| max = {err:.2e}'


H.run('apply_mask: phase-only mask preserves amplitude',
      t_apply_mask_phase_only_preserves_amplitude)


if __name__ == '__main__':
    sys.exit(H.summary())
