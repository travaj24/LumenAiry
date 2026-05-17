"""Correctness pins for the v4.13.0 Shack-Hartmann FFT batching
(audit Phase-3 Group beta task beta.1).

Pre-4.13 ``lumenairy.analysis.detector.shack_hartmann`` walked the
lenslet array with two nested Python loops (one for reference centroids
on a flat-wavefront calibration field, one for the actual measurement
on the user's field).  Each iteration extracted a sub-aperture, applied
the lenslet quadratic-phase mask, and propagated it to the focal plane
via either a raw ``np.fft.fft2`` (reference pass) or the full ASM
machinery (measurement pass), one call per lenslet.

The v4.13.0 vectorisation:

* Recognises that every sub-aperture of the flat reference field is
  identical, so the reference centroid is computed ONCE and broadcast
  to every valid lenslet (zero FFTs in the inner loop, one in total).
* Stacks the measurement sub-apertures into a single
  ``(K, sa_pixels, sa_pixels)`` ndarray, applies the shared lenslet
  phase, and propagates the whole stack with a single
  ``np.fft.fft2(..., axes=(-2, -1))`` + ``ifft2(...)`` pair, multiplied
  through one pre-built ASM transfer function that is geometry-only
  and therefore identical for every lenslet.

What this test pins
-------------------

1. **Tilt recovery** -- a known linear-tilt wavefront gives a slope_x
   map close to the analytical gradient (sign + magnitude), echoing
   the existing validation test in ``validation/analysis/test_detector.py``.
2. **Determinism** -- same input twice returns bit-identical output.
3. **OOB sentinels** -- with a lenslet pitch chosen so the array
   overhangs the field, the out-of-bounds lenslets stay at NaN and
   the in-bounds ones produce finite slopes.
4. **Reference-centroid invariance** -- a flat input produces slopes
   that are unaffected by the per-lenslet centring bias (i.e. the
   reference subtraction still cancels it).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.analysis.detector import shack_hartmann


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _tilt_field(N: int, dx: float, wavelength: float,
                waves_total: float = 5.0) -> np.ndarray:
    """Build a complex field with `waves_total` waves of tilt across
    the grid (left-to-right linear phase ramp)."""
    x = (np.arange(N) - N / 2) * dx
    X, _ = np.meshgrid(x, x)
    phase = (2 * np.pi * waves_total / (N * dx)) * X
    return np.exp(1j * phase)


# ---------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------

def test_shack_hartmann_recovers_known_tilt():
    """Linear tilt across the pupil should produce a near-uniform
    positive slope_x and ~zero slope_y.  Mirrors the existing
    validation harness test but pins it to pytest so the unit suite
    catches regressions of the batched path on every run."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = _tilt_field(N, dx, lam, waves_total=5.0)
    slope_x, slope_y, wf, cx, cy = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6,
        n_lenslets=8)
    # slope_x: positive, near-uniform across valid lenslets
    valid_sx = slope_x[np.isfinite(slope_x)]
    assert valid_sx.size > 0
    assert float(np.mean(valid_sx)) > 0
    assert float(np.std(valid_sx)) < 1e-3 * abs(float(np.mean(valid_sx))) + 1e-12
    # slope_y: small (no tilt in y).  There is a small ASM-vs-FFT
    # residual (~2e-3 rad) on every lenslet that is a pre-existing
    # artefact of the reference-vs-measurement convention mismatch
    # in the legacy code -- compare against slope_x magnitude.
    valid_sy = slope_y[np.isfinite(slope_y)]
    assert float(np.max(np.abs(valid_sy))) < float(np.mean(np.abs(valid_sx)))


def test_shack_hartmann_deterministic_repeat():
    """Running the same input twice through the (vectorised) path
    should produce bit-identical output."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = _tilt_field(N, dx, lam, waves_total=3.0)
    sx1, sy1, wf1, cx1, cy1 = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    sx2, sy2, wf2, cx2, cy2 = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    np.testing.assert_array_equal(sx1, sx2)
    np.testing.assert_array_equal(sy1, sy2)
    np.testing.assert_array_equal(wf1, wf2)
    np.testing.assert_array_equal(cx1, cx2)
    np.testing.assert_array_equal(cy1, cy2)


def test_shack_hartmann_oob_lenslets_are_nan():
    """When n_lenslets * lenslet_pitch overhangs the grid, the
    overhanging lenslets must stay at NaN (the 4.10 OOB-sentinel
    contract -- pre-4.10 they got 0 slopes which masqueraded as real
    measurements in the cumsum integrator)."""
    # Choose a grid so the lenslet array deliberately overhangs.
    # N=32, dx=4e-6 -> 128 um field; lenslet_pitch=20e-6, n_lenslets=12
    # -> 240 um requested footprint, so the array hangs off both sides
    # and only the central few are in-bounds.
    N = 32
    dx = 4e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    slope_x, slope_y, wf, cx, cy = shack_hartmann(
        E, dx, lam, lenslet_pitch=20e-6, lenslet_focal=200e-6,
        n_lenslets=12)
    # Most lenslets out of bounds: at least the four corners must be NaN.
    assert np.isnan(slope_x[0, 0])
    assert np.isnan(slope_x[0, -1])
    assert np.isnan(slope_x[-1, 0])
    assert np.isnan(slope_x[-1, -1])
    # Some lenslets in bounds: at least one finite value somewhere.
    assert np.any(np.isfinite(slope_x))


def test_shack_hartmann_flat_input_zero_slope_after_reference():
    """The 4.10 reference-centroid subtraction guarantees a flat
    wavefront produces ~zero slope (modulo small ASM-vs-FFT residuals
    from the reference using a raw fft2 and the measurement using
    band-limited ASM -- this is a pre-existing approximation, not a
    v4.13.0 regression).  Pin the order of magnitude so a future
    refactor that drops the reference subtraction is caught."""
    N = 128
    dx = 4e-6
    lam = 1.31e-6
    E = np.ones((N, N), dtype=complex)
    slope_x, slope_y, _, _, _ = shack_hartmann(
        E, dx, lam, lenslet_pitch=50e-6, lenslet_focal=200e-6, n_lenslets=8)
    valid_sx = slope_x[np.isfinite(slope_x)]
    valid_sy = slope_y[np.isfinite(slope_y)]
    # Flat input: residual slope should be tiny (< 0.01 rad effective).
    # Reference-subtraction cancels the per-lenslet centring bias.
    assert float(np.max(np.abs(valid_sx))) < 1e-2
    assert float(np.max(np.abs(valid_sy))) < 1e-2
