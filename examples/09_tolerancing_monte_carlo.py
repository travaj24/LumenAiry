"""
Lumenairy example 9 -- Monte Carlo tolerancing of a singlet.

Demonstrates the v5.2 ``lumenairy.monte_carlo_tolerancing`` API by
perturbing a singlet's surface decenter and tilt by small tolerances
and running a few dozen trials to gather Strehl-ratio statistics.

The example uses a small (N=128) pupil and a 9-step through-focus
scan per trial to keep total runtime under ~30 s on a laptop.  For
a real tolerancing run with the same prescription you would raise
``n_trials`` to ~500 and ``z_scan_n`` to ~21 to tighten the
confidence interval on the Strehl-drop distribution.

Run::

    python examples/09_tolerancing_monte_carlo.py

A histogram of the trial Strehl values is written to
``examples/output/09_tolerancing_monte_carlo.png``.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import os as _os

import numpy as np

import lumenairy as la


def main():
    # --- 1. Build a singlet prescription ------------------------------
    # ~100 mm-EFL N-BK7 plano-convex singlet at 12 mm clear aperture.
    # The two refracting surfaces are indices 0 and 1; we'll tolerance
    # both.
    wavelength = 587.56e-9  # d-line
    aperture = 12e-3
    prescription = la.make_singlet(
        R1=51.5e-3, R2=float('inf'), d=4e-3,
        glass='N-BK7', aperture=aperture,
    )

    # v5.2.5 (AUDIT_V5_2_3 P3-F1-1 Strehl normalization fix):
    # Use the singlet's paraxial back focal length (computed from the
    # prescription) as ``focal_length`` rather than a hand-picked
    # round number.  The reference Strehl denominator inside
    # ``monte_carlo_tolerancing`` is evaluated at this z; if it
    # disagrees with the actual focal plane the through-focus scan
    # peak can exceed the reference and yield non-physical Strehl > 1.
    surfs = la.surfaces_from_prescription(prescription)
    _, efl, bfl, _ = la.system_abcd(surfs, wavelength=wavelength)
    f_target = bfl
    print(f'  Singlet paraxial: EFL = {efl*1e3:.3f} mm,  '
          f'BFL = {bfl*1e3:.3f} mm  (used as focal_length)')

    # --- 2. Build the source pupil ------------------------------------
    # A circular clear pupil on a 128x128 grid.  Pupil semi-diameter
    # ~ 0.45 * aperture so there's some guard band for the perturbation
    # kernels.
    N = 128
    dx = aperture / 64
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    semi_ap = aperture / 2
    E_source = np.where(np.sqrt(X * X + Y * Y) <= semi_ap, 1.0, 0.0).astype(
        np.complex128)

    # --- 3. Define the tolerance specification ------------------------
    # Each surface (indices 0, 1) takes a 1-sigma lateral decenter of
    # 50 um and a 1-sigma tilt of 0.5 mrad.  No form-error term --
    # set form_error_rms=0 to skip Zernike-noise sag perturbations.
    perturbation_spec = [
        {'surface_index': 0, 'decenter_std': 50e-6,
         'tilt_std': 0.5e-3, 'form_error_rms': 0.0},
        {'surface_index': 1, 'decenter_std': 50e-6,
         'tilt_std': 0.5e-3, 'form_error_rms': 0.0},
    ]

    # --- 4. Run the Monte Carlo trials --------------------------------
    # n_trials=24 + z_scan_n=9 keeps the example under a minute.
    # For a real tolerancing report you'd raise both substantially.
    n_trials = 24
    print(f'  Running {n_trials} Monte-Carlo tolerancing trials '
          f'(this takes ~20-40 s)...')
    stats = la.monte_carlo_tolerancing(
        prescription=prescription, wavelength=wavelength,
        N=N, dx=dx, E_source=E_source,
        perturbation_spec=perturbation_spec,
        focal_length=f_target, aperture=aperture,
        n_trials=n_trials, seed=0,
        z_scan_n=9, verbose=False,
    )

    strehls = np.asarray(stats['strehl_array'])
    mean = stats['strehl_peak_mean']
    std = stats['strehl_peak_std']
    p05 = stats['strehl_peak_p05']
    p95 = stats['strehl_peak_p95']

    # --- 5. Plot the Strehl distribution histogram -------------------
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping histogram figure.')
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(strehls, bins=12, edgecolor='black', alpha=0.7)
        ax.axvline(mean, color='red', linestyle='--',
                   label=f'mean = {mean:.3f}')
        ax.axvline(p05, color='orange', linestyle=':',
                   label=f'5th pct = {p05:.3f}')
        ax.axvline(p95, color='orange', linestyle=':',
                   label=f'95th pct = {p95:.3f}')
        ax.set_xlabel('Strehl ratio (best focus)')
        ax.set_ylabel('Trial count')
        ax.set_title(
            f'09_tolerancing_monte_carlo.py -- {n_trials} trials')
        ax.legend()
        plt.tight_layout()
        out_path = _os.path.join(out_dir, '09_tolerancing_monte_carlo.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  Figure saved to {out_path}')

    # --- 6. Summary --------------------------------------------------
    print()
    print(f'  Tolerancing summary ({n_trials} trials):')
    print(f'    Strehl mean +/- std:   {mean:.3f} +/- {std:.3f}')
    print(f'    Strehl 5th percentile: {p05:.3f}')
    print(f'    Strehl 95th percentile: {p95:.3f}')
    print(f'    Strehl range:          '
          f'[{strehls.min():.3f}, {strehls.max():.3f}]')


if __name__ == '__main__':
    main()
