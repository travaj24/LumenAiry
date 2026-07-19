"""Hammer-audit H3 (2026-07-19): traced exit-NA Nyquist guard.

``apply_real_lens_traced`` documents the critical-sampling rule
``dx <= lambda*f/aperture`` but never enforced it.  Violating it is
SILENT and insidious: the exit converging wavefront exceeds grid
Nyquist beyond a radius, the aliased annulus folds far-halo energy to
wrong positions, and r^2-weighted metrics (r2m) read LOW while
EE50/EE80 stay plausible.  Dual-oracle f/5 case (R=+/-51.68mm, t=5mm,
n=1.5168, lambda=1.31um, w0=5mm): traced r2m 40.9 vs oracle 65.0 um at
dx=6um (2.24x the limit), fully recovered to 64.77 (99.7%) at dx=3um.

The v5.25.0 guard computes the exit NA from the already-traced ray
direction cosines (amplitude-aware: rays with >= e^-4 of peak input
amplitude) and warns when dx > lambda/(2*NA_exit).  It deliberately
never raises (core metrics remain valid); ``on_undersample='silent'``
suppresses it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.glass import GLASS_REGISTRY

_WL = 1.31e-6

GLASS_REGISTRY['_H3_FIX_GLASS'] = lambda wl: 1.5168


def _singlet_f5():
    return {
        'wavelength': _WL,
        'aperture_diameter': 24e-3,
        'surfaces': [
            {'radius': 51.68e-3, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': '_H3_FIX_GLASS',
             'semi_diameter': 12e-3},
            {'radius': -51.68e-3, 'thickness': 0.0,
             'glass_before': '_H3_FIX_GLASS', 'glass_after': 'air',
             'semi_diameter': 12e-3},
        ],
        'thicknesses': [5e-3],
        'stop_index': 0,
    }


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def test_h3_guard_fires_on_undersampled_fast_beam():
    """f/5 beam (w0=5mm through the f~50mm singlet, NA_exit ~ 0.2) on a
    dx=12um grid violates dx <= lambda/(2*NA_exit) ~ 3um by 4x -> the
    guard must warn, naming NA_exit and the required dx."""
    E0 = _gauss(1024, 12e-6, 5e-3)
    with pytest.warns(RuntimeWarning, match='NA_exit'):
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=12e-6)


def test_h3_guard_silent_on_benign_slow_beam():
    """Benign w0=0.5mm beam through the same lens: significant rays only
    reach h ~ 1.4w0 = 0.7mm -> NA_exit ~ 0.014 -> dx_need ~ 47um >> the
    8um grid.  The guard must stay silent (amplitude-aware: the huge
    zero-energy aperture must NOT over-fire it)."""
    E0 = _gauss(1024, 8e-6, 0.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=8e-6)


def test_h3_guard_suppressed_by_silent_policy():
    E0 = _gauss(1024, 12e-6, 5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=12e-6,
                                  on_undersample='silent')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
