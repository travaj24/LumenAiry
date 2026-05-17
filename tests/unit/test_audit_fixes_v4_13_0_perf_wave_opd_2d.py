"""Correctness pins for the v4.13.0 ``wave_opd_2d`` axis-unwrap
vectorisation (audit Phase-3 Group beta task beta.2).

Pre-4.13 the 2-D OPD unwrap walked the phase grid with two Python
for-loops:

.. code-block:: python

    for j in range(Ny):
        phase_unwrapped[j, :] = np.unwrap(phase[j, :])
    for i in range(Nx):
        phase_unwrapped[:, i] = np.unwrap(phase_unwrapped[:, i])

Each ``np.unwrap`` call on a 1-D slice spends most of its time on
Python-level overhead (slice creation, function call, ufunc dispatch);
the actual unwrap is O(N) compiled C work.  ``np.unwrap`` accepts
``axis=``, so the two passes collapse to two compiled calls operating
on the full 2-D array:

.. code-block:: python

    phase_unwrapped = np.unwrap(phase, axis=1)
    phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)

What this test pins
-------------------

1. **Equivalence on smooth wavefronts** -- a quadratic OPD with no
   2*pi wraps gives the same map (to numerical round-off) before and
   after vectorisation.
2. **Wrap recovery** -- a steep tilt that wraps multiple times across
   the pupil is unwrapped correctly, recovering the underlying OPD.
3. **Aperture masking** -- the ``aperture`` parameter still masks
   samples outside the clear aperture to NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis.core import wave_opd_2d


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _reference_unwrap_2d(phase: np.ndarray) -> np.ndarray:
    """Pre-vectorisation row-then-column unwrap reference.

    Reproduces the pre-4.13 Python double-loop so we can verify the
    vectorised path matches bit-by-bit on smooth inputs.
    """
    Ny, Nx = phase.shape
    out = np.empty_like(phase)
    for j in range(Ny):
        out[j, :] = np.unwrap(phase[j, :])
    for i in range(Nx):
        out[:, i] = np.unwrap(out[:, i])
    return out


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_vectorised_unwrap_matches_loop_on_smooth_wavefront():
    """A smooth quadratic OPD has no phase wraps; the vectorised
    unwrap should give an output identical to the legacy row-then-
    column loop to numerical round-off."""
    N = 64
    dx = 5e-6
    lam = 1.31e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k0 = 2 * np.pi / lam
    # Gentle quadratic, < 1 cycle of phase across the grid.
    opd_true = 30e-9 * (X**2 + Y**2) / ((N * dx / 2)**2)
    E = np.exp(1j * k0 * opd_true)

    _, _, opd_vec = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)

    # Reference: do the unwrap the legacy way and convert back to OPD.
    phase = np.angle(E)
    phase_ref = _reference_unwrap_2d(phase)
    opd_ref_full = phase_ref / k0
    # Apply the same aperture mask wave_opd_2d does.
    valid = (X**2 + Y**2 <= (0.5 * 0.9 * N * dx)**2) & (np.abs(E) > 0)
    opd_ref = np.where(valid, opd_ref_full, np.nan)

    # Bit-identity on the valid region.
    mask = np.isfinite(opd_vec) & np.isfinite(opd_ref)
    assert mask.sum() > 0
    diff = np.max(np.abs(opd_vec[mask] - opd_ref[mask]))
    assert diff < 1e-12, f"Max OPD diff vs reference = {diff:.3e}"


def test_unwrap_recovers_steep_tilt():
    """A wavefront tilt steep enough to wrap multiple times across
    the pupil should be unwrapped back to the smooth linear OPD."""
    N = 128
    dx = 5e-6
    lam = 1.31e-6
    x = (np.arange(N) - N / 2) * dx
    X, _ = np.meshgrid(x, x)
    k0 = 2 * np.pi / lam
    # Strong tilt: 4 waves of OPD across the grid (-> 8*pi of phase).
    opd_true = 4.0 * lam * (X / (N * dx))
    E = np.exp(1j * k0 * opd_true)

    _, _, opd_vec = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)

    # Recovered OPD should differ from analytical only by a constant
    # offset (unwrap doesn't constrain the integration constant).
    mask = np.isfinite(opd_vec)
    diff = opd_vec[mask] - opd_true[mask]
    offset = float(np.mean(diff))
    residual = float(np.max(np.abs(diff - offset)))
    assert residual < 1e-12, f"Max unwrap residual = {residual:.3e}"


def test_aperture_masks_outside_to_nan():
    """Samples outside the requested clear aperture must be NaN."""
    N = 32
    dx = 5e-6
    lam = 1.31e-6
    aperture = 0.5 * N * dx
    E = np.ones((N, N), dtype=complex)
    X_, Y_, opd = wave_opd_2d(E, dx, lam, aperture=aperture)
    r2 = X_**2 + Y_**2
    outside = r2 > (0.5 * aperture)**2
    assert np.all(np.isnan(opd[outside]))
    # And inside the aperture is NOT NaN (flat -> 0 OPD).
    inside = r2 <= (0.5 * aperture)**2
    assert np.all(np.isfinite(opd[inside]))


def test_unwrap_on_zero_field_propagates_nan():
    """A sample with ``|E| == 0`` must still be masked to NaN regardless
    of aperture (the unwrap is meaningless where amplitude is zero)."""
    N = 32
    dx = 5e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    # Punch a few zeros.
    E[5, 7] = 0.0
    E[10, 12] = 0.0
    _, _, opd = wave_opd_2d(E, dx, lam, aperture=0.9 * N * dx)
    assert np.isnan(opd[5, 7])
    assert np.isnan(opd[10, 12])
