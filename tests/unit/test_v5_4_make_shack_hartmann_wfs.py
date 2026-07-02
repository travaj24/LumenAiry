"""Tests for the v5.4 ``make_shack_hartmann_wfs`` factory.

Scope: pin the public surface, side-effect-free closure semantics,
defocus round-trip, noise behaviour, and modal-basis validation for
the canonical Shack-Hartmann WFS-callable adapter shipped at v5.4
(AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A).

The adapter wraps the existing ``shack_hartmann`` spot simulator and
``slope_to_modal`` reconstructor into a single closure with the
signature expected by ``ao_closed_loop(wfs=...)``.

Author: Andrew Traverso -- v5.4 / SH-WFS adapter.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis import ao as _ao_module
from lumenairy.analysis.ao import make_shack_hartmann_wfs

# --------------------------------------------------------------------------- #
# API smoke pins
# --------------------------------------------------------------------------- #


def test_factory_exists_at_top_level():
    """Pin: ``lumenairy.make_shack_hartmann_wfs`` is exported."""
    assert hasattr(la, 'make_shack_hartmann_wfs'), (
        "lumenairy.make_shack_hartmann_wfs is missing -- v5.4 should "
        "have wired the factory into lumenairy/__init__.py.")
    assert callable(la.make_shack_hartmann_wfs)
    assert la.make_shack_hartmann_wfs is _ao_module.make_shack_hartmann_wfs


def test_factory_in_analysis_ao_all():
    """Pin: the factory is listed in ``analysis.ao.__all__``."""
    assert 'make_shack_hartmann_wfs' in _ao_module.__all__


def test_factory_returns_callable():
    """The factory returns a callable closure."""
    wfs = make_shack_hartmann_wfs()
    assert callable(wfs), "factory must return a callable closure"


# --------------------------------------------------------------------------- #
# Behaviour pins
# --------------------------------------------------------------------------- #


def _disk_grid(N: int = 128):
    """Build a normalised-pupil (X, Y, rho^2, mask) tuple on an
    (N, N) grid sized so dx = 2 / N and the pupil fills the grid."""
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    rho2 = X ** 2 + Y ** 2
    mask = rho2 <= 1.0
    return X, Y, rho2, mask


def test_zero_residual_yields_zero_measured():
    """A zero residual produces an approximately-zero measured phase.

    The library SH simulator's reference and measurement paths use
    differing FFT conventions, so the factory captures a one-time
    flat-field calibration on first call and subtracts it from every
    subsequent measurement.  After that calibration, a zero residual
    must reconstruct to a phase whose RMS is small relative to typical
    AO loop residuals (well below the SH slope-quantisation noise).
    """
    wfs = make_shack_hartmann_wfs(subaperture_grid=8, n_modes=15)
    N = 64
    out = wfs(np.zeros((N, N), dtype=np.float64))
    rms = float(np.sqrt(np.mean(out * out)))
    assert rms < 1e-6, (
        f"Zero residual reconstructed with RMS={rms:.3e} rad; expected "
        f"<1e-6 after flat-field calibration.")


def test_pure_defocus_round_trips():
    """A known defocus phase round-trips through the closure with
    bounded RMS error (< 0.1 rad as per the spec).
    """
    wfs = make_shack_hartmann_wfs(subaperture_grid=8, n_modes=15)
    _X, _Y, rho2, mask = _disk_grid(128)
    # Pure Z_2^0 (defocus) in the un-normalised polynomial.  The
    # amplitude is small enough to stay inside the SH simulator's
    # linear PSF-centroid regime.
    phase = 0.3 * (2.0 * rho2 - 1.0) * mask
    out = wfs(phase)
    # Compare on the disk only (outside the pupil the closure returns 0
    # but the input is also 0 because of the mask, so the difference
    # there is identically 0).
    diff = out[mask] - phase[mask]
    rms = float(np.sqrt(np.mean(diff * diff)))
    assert rms < 0.1, (
        f"Defocus round-trip RMS={rms:.4f} rad > 0.1 tolerance.  "
        "Closure failed to recover the injected Zernike mode.")


def test_noise_sigma_affects_output():
    """With ``rng_seed=None`` and noise > 0, repeated calls on the
    same input produce DIFFERENT outputs (per-call default_rng()).
    """
    wfs = make_shack_hartmann_wfs(
        subaperture_grid=8, noise_sigma_pixels=1.0, n_modes=15)
    N = 64
    X, _Y, _rho2, mask = _disk_grid(N)
    phase = 0.05 * X * mask  # mild tilt
    a = wfs(phase)
    b = wfs(phase)
    diff = float(np.sqrt(np.mean((a - b) ** 2)))
    assert diff > 0.0, (
        "Per-call noise RNG should produce DIFFERENT outputs on "
        "repeated calls with the same input; got identical outputs.")


def test_seeded_rng_is_reproducible_sequence():
    """v5.17 (audit P3-01): with ``rng_seed`` set the noise SEQUENCE is
    reproducible -- two closures built with the same seed produce
    identical per-call outputs -- but successive calls draw FRESH
    noise (no frozen per-frame realisation)."""
    kw = dict(subaperture_grid=8, noise_sigma_pixels=1.0, rng_seed=42,
              n_modes=15)
    wfs1 = make_shack_hartmann_wfs(**kw)
    wfs2 = make_shack_hartmann_wfs(**kw)
    N = 64
    X, _Y, _rho2, mask = _disk_grid(N)
    phase = 0.05 * X * mask
    a1, b1 = wfs1(phase), wfs1(phase)
    a2, b2 = wfs2(phase), wfs2(phase)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2), (
        "Same rng_seed must reproduce the same noise SEQUENCE across "
        "closures; got differing outputs.")
    assert not np.array_equal(a1, b1), (
        "Successive calls with rng_seed set must draw FRESH noise "
        "(reproducible sequence, not a frozen per-frame realisation).")


def test_modal_basis_choices():
    """Zernike is the only canonical basis at v5.4; non-zernike
    choices raise ``ValueError`` rather than silently mis-behaving."""
    # Zernike works.
    wfs_z = make_shack_hartmann_wfs(modal_basis='zernike')
    assert callable(wfs_z)
    # KL (placeholder) and free_actuator (reserved) raise.
    with pytest.raises(ValueError, match='modal_basis'):
        make_shack_hartmann_wfs(modal_basis='karhunen_loeve')
    with pytest.raises(ValueError, match='modal_basis'):
        make_shack_hartmann_wfs(modal_basis='bogus')


# --------------------------------------------------------------------------- #
# Composition with ao_closed_loop (smoke pin)
# --------------------------------------------------------------------------- #


def test_closure_compatible_with_ao_closed_loop():
    """The closure has the ``residual -> measured`` signature
    expected by :func:`ao_closed_loop`; the loop accepts it and
    runs at least one iteration without raising.
    """
    wfs = make_shack_hartmann_wfs(subaperture_grid=8, n_modes=15)
    N = 64
    X, _Y, _rho2, mask = _disk_grid(N)
    phase = 0.05 * X * mask
    # Minimal DM matching the (N, dx) pair the closure expects when
    # dx_pupil is auto-derived (dx = 2 / N).
    dx = 2.0 / N
    dm = la.DeformableMirror(
        n_actuators=5, pitch=(N * dx) / 6, dx=dx, N=N,
        inter_actuator_coupling=0.2,
    )
    # One iteration is enough to confirm the wiring round-trips.
    residual = la.ao_closed_loop(
        phase, dm=dm, n_iterations=1, gain=0.3, dx=dx, wfs=wfs)
    assert residual.shape == (N, N)
    assert np.all(np.isfinite(residual))


# --------------------------------------------------------------------------- #
# Input-validation pins
# --------------------------------------------------------------------------- #


def test_invalid_subaperture_grid_raises():
    """``subaperture_grid <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='subaperture_grid'):
        make_shack_hartmann_wfs(subaperture_grid=0)


def test_invalid_n_modes_raises():
    """``n_modes <= 0`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='n_modes'):
        make_shack_hartmann_wfs(n_modes=0)


def test_negative_noise_raises():
    """Negative noise sigma is rejected."""
    with pytest.raises(ValueError, match='noise_sigma_pixels'):
        make_shack_hartmann_wfs(noise_sigma_pixels=-0.1)


def test_non_2d_residual_raises():
    """The closure rejects a non-2-D residual."""
    wfs = make_shack_hartmann_wfs()
    with pytest.raises(ValueError, match='residual'):
        wfs(np.zeros(64))
