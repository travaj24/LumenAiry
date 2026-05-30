"""v5.4.6 Wave 7 regression pins: sources/coherence cluster.

- F-38 : create_gaussian_beam(normalize='peak') is unit-peak even off-grid.
- F-39 : create_gaussian_beam rejects sigma <= 0 / non-finite.
- F-40 : create_tilted_plane_wave rejects the evanescent tilt regime.
- P3-10: Schell realizations use a deterministic mean-intensity norm
         (ensemble mean intensity ~ 1).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


def test_gaussian_beam_rejects_nonphysical_sigma():
    for s in (0.0, -3e-6, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            la.create_gaussian_beam(32, 2e-6, 633e-9, sigma=s)


def test_gaussian_beam_peak_unit_even_off_grid():
    # Peak placed far outside the +/-32 um grid; normalize='peak' must
    # still divide by the actual max so |E|max == 1.
    E, _, _ = la.create_gaussian_beam(32, 2e-6, 633e-9, sigma=4e-6,
                                      x0=100e-6, normalize='peak')
    assert np.isclose(float(np.abs(E).max()), 1.0)


def test_tilted_plane_wave_rejects_evanescent():
    with pytest.raises(ValueError, match='evanescent'):
        la.create_tilted_plane_wave(
            32, 2e-6, 633e-9,
            angle_x=np.radians(80), angle_y=np.radians(80))
    # A single-axis grazing tilt (<=90 deg) is still allowed.
    la.create_tilted_plane_wave(32, 2e-6, 633e-9, angle_x=np.radians(40))


def test_schell_realizations_deterministic_mean_intensity():
    from lumenairy.sources.core import _schell_phase_realizations
    phi = _schell_phase_realizations(
        N=64, Ny=64, Nx=64, dx=2e-6, dy=2e-6,
        coherence_length=15e-6, n_realizations=96,
        rng=np.random.default_rng(0))
    ens_mean = float(np.mean(np.abs(phi) ** 2))
    # Unit mean intensity in expectation (averaged over the ensemble).
    assert abs(ens_mean - 1.0) < 0.05, f"ensemble mean intensity {ens_mean}"
