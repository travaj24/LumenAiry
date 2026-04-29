"""Glass dispersion and tolerancing / Monte Carlo tests.

From:
- physics_extended_test.py (BK7 Sellmeier, achromat EFL variation,
  tolerance tilt, form error, Monte Carlo Strehl stats)
- deep_audit.py (tolerancing + biconic, Monte Carlo tolerancing)
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

import lumenairy as op
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd,
)


H = Harness('glass_tolerancing')


# ---------------------------------------------------------------------
H.section('Glass dispersion')


def t_bk7_dispersion():
    reference = {
        0.5876e-6: 1.51680,
        0.6563e-6: 1.51432,
        0.4861e-6: 1.52238,
        1.0140e-6: 1.50731,
    }
    max_err = 0.0
    for wl, n_ref in reference.items():
        n_calc = op.get_glass_index('N-BK7', wl)
        max_err = max(max_err, abs(n_calc - n_ref))
    return max_err < 0.002, f'max index error = {max_err:.5f}'


H.run('Glass: N-BK7 Sellmeier dispersion accuracy',
      t_bk7_dispersion)


def t_achromat_correction():
    pres = op.thorlabs_lens('AC254-100-C')
    wvs = np.linspace(1.0e-6, 1.6e-6, 7)
    efls, _, _ = op.chromatic_focal_shift(pres, wvs)
    efl_pv = efls.max() - efls.min()
    efl_mean = efls.mean()
    variation_pct = efl_pv / efl_mean * 100
    return variation_pct < 5, \
        f'EFL PV = {efl_pv*1e3:.3f}mm ({variation_pct:.2f}%)'


H.run('Achromat: EFL variation < 5% across 1.0-1.6 um',
      t_achromat_correction)


# ---------------------------------------------------------------------
H.section('Tolerancing: tilt / form error')


def t_tolerance_tilt_shifts_centroid():
    N = 256; dx = 8e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=3e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_nom = op.apply_real_lens(E_in, pres, lam, dx)
    E_focus_nom = op.angular_spectrum_propagate(E_nom, bfl, lam, dx)
    cx_nom, _ = op.beam_centroid(E_focus_nom, dx)
    pres_pert = op.apply_perturbations(pres, [
        op.Perturbation(surface_index=0, tilt=(1e-3, 0), name='tilt')])
    E_pert = op.apply_real_lens(E_in, pres_pert, lam, dx)
    E_focus_pert = op.angular_spectrum_propagate(E_pert, bfl, lam, dx)
    cx_pert, _ = op.beam_centroid(E_focus_pert, dx)
    shift = abs(cx_pert - cx_nom)
    return shift > 1e-6, \
        f'centroid shift = {shift*1e6:.2f} um'


H.run('Tolerance: tilt shifts focal-plane centroid',
      t_tolerance_tilt_shifts_centroid)


def t_tolerance_form_error_degrades():
    N = 256; dx = 8e-6; lam = 1.31e-6
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, lam)
    E_in = np.ones((N, N), dtype=np.complex128)
    E_nom = op.apply_real_lens(E_in, pres, lam, dx)
    ideal = op.diffraction_limited_peak(E_nom, lam, bfl, dx)
    E_focus_nom = op.angular_spectrum_propagate(E_nom, bfl, lam, dx)
    strehl_nom = float(np.abs(E_focus_nom).max()**2 / ideal)
    pres_pert = op.apply_perturbations(pres, [
        op.Perturbation(surface_index=0, form_error_rms=200e-9,
                        random_seed=42, name='form')],
        N=N, dx=dx)
    E_pert = op.apply_real_lens(E_in, pres_pert, lam, dx)
    E_focus_pert = op.angular_spectrum_propagate(E_pert, bfl, lam, dx)
    strehl_pert = float(np.abs(E_focus_pert).max()**2 / ideal)
    return strehl_pert < strehl_nom, \
        f'nominal={strehl_nom:.4f}, perturbed={strehl_pert:.4f}'


H.run('Tolerance: form error degrades Strehl',
      t_tolerance_form_error_degrades)


def t_tolerancing_with_biconic():
    pres = op.make_biconic(50e-3, 60e-3, -60e-3, -70e-3, 3e-3,
                           'N-BK7', aperture=3e-3)
    E = np.ones((128, 128), dtype=np.complex128)
    perts = [op.Perturbation(surface_index=0, tilt=(1e-4, 0),
                             name='p1')]
    results = op.tolerancing_sweep(
        pres, 1.31e-6, 128, 16e-6, E, perts,
        focal_length=50e-3, aperture=3e-3, verbose=False)
    return len(results) == 2, f'{len(results)} sweep points'


H.run('Tolerancing sweep with biconic lens',
      t_tolerancing_with_biconic)


# ---------------------------------------------------------------------
H.section('Monte Carlo tolerancing')


def t_monte_carlo_strehl_stats():
    pres = op.make_singlet(50e-3, np.inf, 4e-3, 'N-BK7',
                           aperture=3e-3)
    surfs = surfaces_from_prescription(pres)
    _, _, bfl, _ = system_abcd(surfs, 1.31e-6)
    E_in = np.ones((64, 64), dtype=np.complex128)
    stats = op.monte_carlo_tolerancing(
        pres, 1.31e-6, 64, 32e-6, E_in,
        perturbation_spec=[{'surface_index': 0, 'tilt_std': 5e-3}],
        focal_length=bfl, aperture=3e-3,
        n_trials=5, seed=0, verbose=False)
    mean = stats['strehl_peak_mean']
    std = stats['strehl_peak_std']
    return (0 < mean <= 1.5 and std > 0), \
        f'Strehl mean={mean:.4f}, std={std:.4f}'


H.run('Monte Carlo: Strehl stats are physically reasonable',
      t_monte_carlo_strehl_stats)


def t_monte_carlo_tolerancing_basic():
    pres = op.make_singlet(50e-3, float('inf'), 4e-3, 'N-BK7',
                           aperture=3e-3)
    E = np.ones((128, 128), dtype=np.complex128)
    stats = op.monte_carlo_tolerancing(
        pres, 1.31e-6, 128, 16e-6, E,
        perturbation_spec=[{'surface_index': 0,
                            'tilt_std': 1e-4,
                            'decenter_std': 1e-5}],
        focal_length=100e-3, aperture=3e-3,
        n_trials=3, seed=1, verbose=False)
    return 'strehl_peak_mean' in stats, \
        f'keys: {sorted(stats.keys())[:6]}'


H.run('monte_carlo_tolerancing returns expected stats',
      t_monte_carlo_tolerancing_basic)


if __name__ == '__main__':
    sys.exit(H.summary())
