"""Tests for the v5.2.3 ROADMAP ``ao_closed_loop`` helper.

Scope: pin the public surface, convergence behaviour, zero-disturbance
trivial path, history-dict shape, and noisy-WFS regression of the
canonical closed-loop AO helper added in v5.2.3.

Mirrors the v5.2 ROADMAP test idioms (small grids, fixed seeds, tight
tolerances, parametrised diagnostics).

Author: Andrew Traverso -- v5.2.3 / ao_closed_loop helper.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis import ao as _ao_module


# --------------------------------------------------------------------------- #
# API smoke pin
# --------------------------------------------------------------------------- #


def test_ao_closed_loop_exists_at_top_level():
    """``ao_closed_loop`` is exported at ``lumenairy.ao_closed_loop``."""
    assert hasattr(la, 'ao_closed_loop'), (
        "lumenairy.ao_closed_loop is missing -- v5.2.3 ROADMAP should "
        "have wired the helper into lumenairy/__init__.py.")
    assert callable(la.ao_closed_loop)


def test_ao_closed_loop_exists_in_analysis_ao():
    """``ao_closed_loop`` is exported at ``lumenairy.analysis.ao``."""
    assert hasattr(_ao_module, 'ao_closed_loop')
    assert callable(_ao_module.ao_closed_loop)
    assert 'ao_closed_loop' in _ao_module.__all__


def test_ao_closed_loop_top_level_matches_submodule():
    """The top-level and submodule symbols are the same object."""
    assert la.ao_closed_loop is _ao_module.ao_closed_loop


# --------------------------------------------------------------------------- #
# Convergence pin
# --------------------------------------------------------------------------- #


def _build_dm_and_grid(N=64, extent=20e-3, n_act=9, aperture=8e-3):
    """Helper: build a (DM, dx, grid coords) triple matching the
    canonical ``examples/11_ao_closed_loop.py`` geometry but at the
    smaller N=64 grid the unit tests use.
    """
    dx = extent / N
    pitch = aperture / (n_act - 1)
    dm = la.DeformableMirror(
        n_actuators=n_act, pitch=pitch, dx=dx, N=N,
        inter_actuator_coupling=0.2,
    )
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return dm, dx, X, Y


def _rms(a):
    """RMS of a flat array."""
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


def test_ao_closed_loop_converges_on_low_order_disturbance():
    """Pin: starting from a known DM-representable disturbance (a
    direct linear combination of the DM's Gaussian influence functions
    at unit amplitude), 10 iterations at gain 0.5 drive the global
    residual RMS below 1% of the initial RMS.

    Why an IF-built disturbance and not a sinusoid: a sinusoidal phase
    drawn on the FULL grid is not representable by a DM whose
    actuators only span a sub-region of that grid (the IFs decay to
    zero outside the actuator footprint, so the off-DM portion of the
    sinusoid is forever uncorrectable).  Sampling the disturbance from
    the DM's own basis sidesteps this geometric mismatch and pins the
    pure-controller convergence the test cares about.
    """
    dm, dx, _, _ = _build_dm_and_grid()
    # Build a DM-representable disturbance.  Use a smooth low-order
    # pattern across the actuator grid (one period of a sinusoid in
    # actuator-index space) so the integrator can converge fully.
    n = dm.n_actuators
    ai = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    AX, _ = np.meshgrid(ai, ai)
    truth_cmd = np.sin(np.pi * AX)  # 1-period sinusoid across actuators
    # Use a scratch DM with the SAME geometry to render the
    # disturbance onto the pupil grid.
    scratch = la.DeformableMirror(
        n_actuators=dm.n_actuators, pitch=dm.pitch, dx=dm.dx, N=dm.N,
        inter_actuator_coupling=dm.inter_actuator_coupling)
    scratch.command = truth_cmd
    phase_disturbance = scratch.phase()
    rms_initial = _rms(phase_disturbance)
    assert rms_initial > 0  # sanity

    residual, history = la.ao_closed_loop(
        phase_disturbance,
        dm=dm,
        n_iterations=10,
        gain=0.5,
        dx=dx,
        return_history=True,
    )
    rms_final = _rms(residual)

    # Pin: < 1% of initial -- the loop must converge.
    ratio = rms_final / max(rms_initial, 1e-30)
    assert ratio < 0.01, (
        f"ao_closed_loop failed to converge: rms_final/rms_initial = "
        f"{ratio:.4g} (expected < 0.01).  "
        f"rms_initial={rms_initial:.4g}, rms_final={rms_final:.4g}")

    # Pin: history's last entry should match the final residual RMS.
    assert np.isclose(history['rms_per_iter'][-1], rms_final, rtol=1e-9), (
        f"history['rms_per_iter'][-1]={history['rms_per_iter'][-1]:.4g} "
        f"!= rms_final={rms_final:.4g}")


# --------------------------------------------------------------------------- #
# Zero-disturbance pin
# --------------------------------------------------------------------------- #


def test_ao_closed_loop_zero_disturbance():
    """Pin: with ``phase_disturbance = zeros(...)`` the residual is
    identically zero (within fp epsilon) at every iteration."""
    dm, dx, _, _ = _build_dm_and_grid()
    phase_zero = np.zeros((dm.N, dm.N), dtype=np.float64)

    residual, history = la.ao_closed_loop(
        phase_zero,
        dm=dm,
        n_iterations=5,
        gain=0.5,
        dx=dx,
        return_history=True,
    )

    assert np.allclose(residual, 0.0, atol=1e-12), (
        f"ao_closed_loop on zero disturbance produced nonzero residual: "
        f"max |residual| = {np.max(np.abs(residual)):.4g}")
    assert np.allclose(history['rms_per_iter'], 0.0, atol=1e-12), (
        f"Zero-disturbance loop should leave rms_per_iter at 0; got "
        f"{history['rms_per_iter']}")
    # DM should remain at its zero command (no spurious actuation).
    assert np.allclose(dm.command, 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# History pin
# --------------------------------------------------------------------------- #


def test_ao_closed_loop_history_dict_shape_and_keys():
    """Pin: ``return_history=True`` returns a dict with the documented
    keys + array shapes."""
    dm, dx, X, _ = _build_dm_and_grid()
    L = 8e-3
    phase = 0.5 * np.sin(2.0 * np.pi * X / L)
    n_iter = 7
    wavelength = 633e-9

    residual, history = la.ao_closed_loop(
        phase,
        dm=dm,
        n_iterations=n_iter,
        gain=0.5,
        wavelength=wavelength,
        dx=dx,
        return_history=True,
    )

    assert isinstance(history, dict)
    assert set(history.keys()) >= {'rms_per_iter', 'wfe_per_iter'}, (
        f"history keys missing -- got {sorted(history.keys())}")
    rms_arr = history['rms_per_iter']
    wfe_arr = history['wfe_per_iter']
    assert rms_arr.shape == (n_iter + 1,), (
        f"rms_per_iter shape {rms_arr.shape} != ({n_iter + 1},)")
    assert wfe_arr.shape == (n_iter + 1,), (
        f"wfe_per_iter shape {wfe_arr.shape} != ({n_iter + 1},)")
    # WFE conversion: rms [rad] / (2 pi) * wavelength [m].
    expected_wfe = rms_arr / (2.0 * np.pi) * wavelength
    assert np.allclose(wfe_arr, expected_wfe, rtol=1e-12), (
        "wfe_per_iter is not rms_per_iter / (2 pi) * wavelength.")

    # Final residual RMS = last entry of history.
    assert np.isclose(rms_arr[-1], _rms(residual), rtol=1e-9)


def test_ao_closed_loop_default_return_is_array_only():
    """Pin: with ``return_history=False`` (default), the return is a
    bare ndarray, not a tuple."""
    dm, dx, X, _ = _build_dm_and_grid()
    phase = 0.5 * np.sin(2.0 * np.pi * X / 8e-3)
    out = la.ao_closed_loop(
        phase, dm=dm, n_iterations=3, gain=0.5, dx=dx)
    assert isinstance(out, np.ndarray), (
        f"Default return should be an ndarray; got {type(out).__name__}")
    assert out.shape == phase.shape


# --------------------------------------------------------------------------- #
# WFS noise pin
# --------------------------------------------------------------------------- #


def _noisy_wfs_factory(sigma, seed):
    """Build a noisy ideal-WFS: returns ``residual + N(0, sigma)``."""
    rng = np.random.default_rng(seed)
    def _wfs(residual):
        return residual + sigma * rng.standard_normal(residual.shape)
    return _wfs


def test_ao_closed_loop_lower_gain_rejects_noise_better():
    """Pin: with a noisy WFS, reducing gain from 0.5 to 0.3 lowers the
    steady-state residual RMS.  Standard AO control-theory result --
    the integrator's noise transfer function has gain ~ g / (2 - g) at
    DC, so smaller g rejects measurement noise better.
    """
    dm_hi, dx, X, _ = _build_dm_and_grid()
    dm_lo, _, _, _ = _build_dm_and_grid()
    # Mid-amplitude low-frequency disturbance.
    L = 8e-3
    phase = 0.5 * np.sin(2.0 * np.pi * X / L)

    # Same WFS noise realisation (fixed seed) for both gains so the
    # comparison reflects the integrator only, not the noise draw.
    sigma = 0.1  # rad of measurement noise

    n_iter = 30
    wfs_hi = _noisy_wfs_factory(sigma=sigma, seed=12345)
    wfs_lo = _noisy_wfs_factory(sigma=sigma, seed=12345)

    _, hist_hi = la.ao_closed_loop(
        phase, dm=dm_hi, n_iterations=n_iter, gain=0.5, dx=dx,
        wfs=wfs_hi, return_history=True)
    _, hist_lo = la.ao_closed_loop(
        phase, dm=dm_lo, n_iterations=n_iter, gain=0.3, dx=dx,
        wfs=wfs_lo, return_history=True)

    # Average the last 10 iterations to remove single-step noise spikes
    # from the steady-state estimate.
    tail = 10
    steady_hi = float(np.mean(hist_hi['rms_per_iter'][-tail:]))
    steady_lo = float(np.mean(hist_lo['rms_per_iter'][-tail:]))

    # Pin: lower gain -> lower steady-state residual under WFS noise.
    assert steady_lo < steady_hi, (
        f"Expected gain=0.3 noise rejection < gain=0.5: "
        f"steady_lo={steady_lo:.4g} should be < steady_hi={steady_hi:.4g}")
    # Sanity: both should still converge below the initial RMS (the
    # loop is doing useful work, not just noise).
    rms_initial = _rms(phase)
    assert steady_hi < rms_initial, (
        f"gain=0.5 loop did not converge below initial RMS: "
        f"steady_hi={steady_hi:.4g} vs rms_initial={rms_initial:.4g}")


def test_ao_closed_loop_ideal_wfs_matches_default():
    """Pin: passing an identity callable as ``wfs`` gives the same
    result as the default ``wfs=None`` path."""
    dm_a, dx, X, _ = _build_dm_and_grid()
    dm_b, _, _, _ = _build_dm_and_grid()
    phase = 0.5 * np.sin(2.0 * np.pi * X / 8e-3)

    res_a = la.ao_closed_loop(
        phase, dm=dm_a, n_iterations=5, gain=0.5, dx=dx, wfs=None)
    res_b = la.ao_closed_loop(
        phase, dm=dm_b, n_iterations=5, gain=0.5, dx=dx,
        wfs=lambda r: r)

    assert np.allclose(res_a, res_b, atol=1e-12), (
        "Default WFS path differs from explicit identity callable.")


# --------------------------------------------------------------------------- #
# Input-validation pins
# --------------------------------------------------------------------------- #


def test_ao_closed_loop_rejects_bad_gain():
    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): gain validation loosened
    # from ``0.0 < gain <= 1.0`` to ``0.0 <= gain <= 2.0``.  The
    # boundary ``gain=0.0`` is now an open-loop fallback (useful
    # for noise-only characterization); ``gain=1.5`` is a valid
    # high-bandwidth tuning.  Out-of-range values still raise.
    dm, dx, _, _ = _build_dm_and_grid()
    phase = np.zeros((dm.N, dm.N))
    with pytest.raises(ValueError, match='gain'):
        la.ao_closed_loop(phase, dm=dm, n_iterations=1, gain=-0.1, dx=dx)
    with pytest.raises(ValueError, match='gain'):
        la.ao_closed_loop(phase, dm=dm, n_iterations=1, gain=2.5, dx=dx)


def test_ao_closed_loop_rejects_bad_dx():
    dm, _, _, _ = _build_dm_and_grid()
    phase = np.zeros((dm.N, dm.N))
    with pytest.raises(ValueError, match='dx'):
        la.ao_closed_loop(phase, dm=dm, n_iterations=1, gain=0.5, dx=-1.0)


def test_ao_closed_loop_rejects_non_dm():
    phase = np.zeros((16, 16))
    with pytest.raises(TypeError, match='DeformableMirror'):
        la.ao_closed_loop(
            phase, dm=object(), n_iterations=1, gain=0.5, dx=1e-3)


def test_ao_closed_loop_rejects_non_2d_phase():
    dm, dx, _, _ = _build_dm_and_grid()
    bad = np.zeros((4, dm.N, dm.N))
    with pytest.raises(ValueError, match='2-D'):
        la.ao_closed_loop(bad, dm=dm, n_iterations=1, gain=0.5, dx=dx)
