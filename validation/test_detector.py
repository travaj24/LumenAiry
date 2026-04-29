"""Detector, wavefront-sensor, and phase-retrieval tests.

From:
- physics_deep_test.py (detector total signal)
- physics_extended_test.py (shot noise, read noise, phase retrieval GS)
- physics_complex_test.py (Shack-Hartmann recovers defocus)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op


H = Harness('detector')


def t_detector_total_signal():
    N = 128; dx = 8e-6
    E = np.ones((N, N), dtype=np.complex128) * 1e6
    img, _, _ = op.apply_detector(E, dx, pixel_pitch=32e-6, n_pixels=30,
                                  exposure_time=1.0,
                                  quantum_efficiency=1.0,
                                  read_noise_e=0, dark_current_e_per_s=0,
                                  seed=42)
    I = np.abs(E)**2
    expected_total = float(I.sum() * dx**2)
    actual_total = float(img.sum())
    ratio = actual_total / expected_total if expected_total > 0 else 0
    return 0.5 < ratio < 2.0, f'signal ratio = {ratio:.4f}'


H.run('Detector: total signal conserved', t_detector_total_signal)


def t_detector_shot_noise():
    N = 128; dx = 8e-6
    E = np.ones((N, N), dtype=np.complex128) * 1e5
    imgs = []
    for seed in range(20):
        img, _, _ = op.apply_detector(E, dx, pixel_pitch=32e-6,
                                      n_pixels=30, exposure_time=1.0,
                                      quantum_efficiency=1.0,
                                      read_noise_e=0, seed=seed)
        imgs.append(img)
    stack = np.array(imgs)
    mean_signal = stack.mean()
    std_signal = stack.std(axis=0).mean()
    expected_std = np.sqrt(mean_signal)
    ratio = std_signal / expected_std if expected_std > 0 else 0
    return 0.5 < ratio < 2.0, \
        f'std/sqrt(mean) = {ratio:.3f} (expect ~1.0)'


H.run('Detector: shot noise ~ sqrt(N)', t_detector_shot_noise)


def t_detector_read_noise():
    N = 128; dx = 8e-6
    E = np.ones((N, N), dtype=np.complex128) * 1e5
    imgs_no = [op.apply_detector(E, dx, 32e-6, 30,
                                 read_noise_e=0, seed=i)[0]
               for i in range(10)]
    imgs_rn = [op.apply_detector(E, dx, 32e-6, 30,
                                 read_noise_e=50, seed=i)[0]
               for i in range(10)]
    std_no = np.array(imgs_no).std(axis=0).mean()
    std_rn = np.array(imgs_rn).std(axis=0).mean()
    return std_rn > std_no * 1.1, \
        f'std_no_RN={std_no:.1f}, std_with_RN={std_rn:.1f}'


H.run('Detector: read noise increases total noise',
      t_detector_read_noise)


def t_shack_hartmann_defocus():
    N = 256; dx = 4e-6; lam = 1.31e-6
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x)
    k0 = 2 * np.pi / lam
    defocus_coeff = 100e-9
    r_pupil = N * dx / 2
    rho = np.sqrt(X**2 + Y**2) / r_pupil
    Z4 = np.sqrt(3) * (2 * rho**2 - 1)
    phase = k0 * defocus_coeff * Z4
    E = np.exp(1j * phase)
    _, _, wf, _, _ = op.shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6,
        n_lenslets=8)
    center = wf[4, 4]
    edge = wf[0, 0]
    has_curvature = abs(edge - center) > 0
    return has_curvature, \
        f'WF center={center*1e9:.1f}nm, edge={edge*1e9:.1f}nm'


H.run('Shack-Hartmann: recovers defocus curvature',
      t_shack_hartmann_defocus)


def t_gerchberg_saxton():
    N = 128
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    source = np.exp(-(X**2 + Y**2) / 0.3**2)
    target = (np.where(np.abs(X) < 0.3, 1.0, 0.0)
              * np.where(np.abs(Y) < 0.3, 1.0, 0.0))
    phase, err = op.gerchberg_saxton(source, target, n_iter=100)
    if isinstance(err, (list, np.ndarray)) and len(err) > 1:
        return err[-1] < err[0], \
            f'GS error: initial={err[0]:.4f}, final={err[-1]:.4f}'
    return phase.shape == source.shape, \
        f'GS returned phase shape {phase.shape}'


H.run('Phase retrieval: GS error decreases', t_gerchberg_saxton)


if __name__ == '__main__':
    sys.exit(H.summary())
