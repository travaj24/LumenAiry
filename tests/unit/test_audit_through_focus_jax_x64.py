"""Regression tests for audit finding S3-2 (v5.24.4).

``through_focus_scan_jax`` / ``monte_carlo_tolerancing_jax`` used to call
``jnp.asarray(E_exit)`` with no precision guard.  When the caller passes a
double-precision (``complex128``) field but ``jax_enable_x64`` is off (JAX's
default, which lumenairy never flips), JAX SILENTLY downcasts to
``complex64``.  The through-focus ASM kernel then evaluates its phase
``kz * z`` (~1e5-1e6 rad near focus) in float32, losing all sub-radian
accuracy -- peak intensity drifts ~1-15% and best-focus Strehl can exceed 1,
so ``through_focus_scan(backend='jax')`` and the Monte-Carlo tolerancer
silently mis-rank designs and corrupt Strehl statistics.

The fix routes the target dtype through the shared
``fft_infra._resolve_jax_complex_dtype`` helper (the same contract every
other JAX entry point already uses): when double precision is requested but
disabled it auto-enables x64 with a one-shot ``RuntimeWarning``, so the scan
runs in the requested precision instead of a silent float32 truncation.  A
caller that explicitly wants single precision passes a ``complex64`` field
and is respected silently.

Independent oracle: the pure-NumPy ``through_focus_scan`` (a different code
path, always float64) is used as ground truth.  Pre-fix the JAX peak_I / Strehl
diverged from it by ~1.6e-2 for the config below; post-fix they match to the
FFT-backend noise floor (~1e-16 measured).

Measured (WSL venv, x64 off):
    quantity           pre-fix (float32)   post-fix (x64 auto-enabled)
    peak_I rel err     1.6e-2              6.8e-17
    best_focus Strehl  1.07095 (JAX)       1.080860 == NumPy oracle
"""
from __future__ import annotations

import numpy as np
import pytest


def _jax_or_skip():
    """Import jax (skip if absent) and skip if x64 is already on
    process-wide -- the silent complex64 truncation (audit S3-2) only
    reproduces with x64 OFF, the JAX default and what a fresh test process
    uses.  Mirrors ``test_audit_jax_c64_propagator_precision._jnp_or_skip``.
    """
    jax = pytest.importorskip("jax")
    if jax.config.jax_enable_x64:
        pytest.skip(
            "jax_enable_x64 is on process-wide; the silent complex64 "
            "truncation (audit S3-2) only reproduces with x64 OFF "
            "(JAX default)")
    return jax


def _make_focusing_field(N, dx, wl, f, sigma, dtype=np.complex128):
    """Gaussian apodised by a perfect converging-lens phase -- a tight
    focal spot near ``z = f`` whose |E(z)|^2 gradient is steep, so any
    kernel-phase precision loss shows up as a large peak-intensity drift.
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k = 2.0 * np.pi / wl
    amp = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    phase = np.exp(-1j * k * (X ** 2 + Y ** 2) / (2.0 * f))
    return (amp * phase).astype(dtype)


def _relerr(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    return float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))


def test_through_focus_scan_jax_matches_numpy_oracle_no_silent_c64():
    """A double-precision through-focus scan on the JAX backend must match
    the pure-NumPy oracle instead of silently truncating to float32.

    Fail-before/pass-after: pre-fix the JAX peak_I / Strehl diverged from the
    NumPy oracle by ~1.6e-2 and NO warning was emitted with x64 left off;
    post-fix a ``RuntimeWarning`` fires, x64 is auto-enabled, and the scan
    matches the oracle to the FFT-backend noise floor.
    """
    jax = _jax_or_skip()
    import lumenairy as lm
    from lumenairy.analysis.through_focus import (
        through_focus_scan,
        through_focus_scan_jax,
    )

    orig_x64 = jax.config.jax_enable_x64
    try:
        N, dx, wl, f, sigma = 128, 4e-6, 1.31e-6, 100e-3, 120e-6
        E = _make_focusing_field(N, dx, wl, f, sigma)  # complex128
        z = np.linspace(96e-3, 104e-3, 21)
        ideal = lm.diffraction_limited_peak(E, wl, f, dx, bandlimit=True)

        # Independent oracle: pure-NumPy, always float64.
        ref = through_focus_scan(
            E, dx, wl, z, ideal_peak=ideal, bandlimit=True, verbose=False)

        # JAX backend with x64 off at entry -- the fix must remove the
        # "silently" by warning + auto-enabling double precision.
        with pytest.warns(RuntimeWarning, match="jax_enable_x64"):
            got = through_focus_scan_jax(
                E, dx, wl, z, ideal_peak=ideal, bandlimit=True)

        assert jax.config.jax_enable_x64, (
            "through_focus_scan_jax should auto-enable jax_enable_x64 when a "
            "complex128 field is passed with x64 off (audit S3-2 guard).")

        peak_rel = _relerr(got.peak_I, ref.peak_I)
        strehl_rel = _relerr(got.strehl, ref.strehl)
        assert peak_rel < 1e-6, (
            f"JAX peak_I diverges from the NumPy oracle by {peak_rel:.2e} "
            f"(pre-fix ~1.6e-2 float32 truncation); the S3-2 guard should "
            f"run the scan in double precision.")
        assert strehl_rel < 1e-6, (
            f"JAX Strehl diverges from the NumPy oracle by {strehl_rel:.2e} "
            f"(pre-fix float32 truncation).")
    finally:
        # Restore the process-wide config so this test does not leak x64=on
        # to sibling tests (and so a second S3-2 test in this file still
        # reproduces the x64-off condition).
        jax.config.update("jax_enable_x64", orig_x64)


def test_through_focus_scan_jax_respects_explicit_complex64():
    """A caller that explicitly wants single precision (``complex64`` field)
    must be honoured silently: no warning, no x64 mutation -- only the
    silent complex128->complex64 downcast is guarded, not a deliberate
    single-precision request.
    """
    jax = _jax_or_skip()
    import warnings

    from lumenairy.analysis.through_focus import through_focus_scan_jax

    orig_x64 = jax.config.jax_enable_x64
    try:
        E32 = _make_focusing_field(
            64, 8e-6, 1.55e-6, 50e-3, 80e-6, dtype=np.complex64)
        z = np.linspace(45e-3, 55e-3, 9)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scan = through_focus_scan_jax(E32, 8e-6, 1.55e-6, z, bandlimit=True)
        x64_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "jax_enable_x64" in str(w.message)]
        assert not x64_warnings, (
            "an explicit complex64 field must not trigger the S3-2 x64 "
            "warning -- a deliberate single-precision request is honoured.")
        assert not jax.config.jax_enable_x64, (
            "an explicit complex64 field must not flip jax_enable_x64 on.")
        assert np.all(np.isfinite(scan.peak_I))
    finally:
        jax.config.update("jax_enable_x64", orig_x64)


def test_monte_carlo_tolerancing_jax_guards_x64():
    """The Monte-Carlo tolerancer shares the silent-truncation path and must
    apply the same S3-2 guard up front: a complex128 source with x64 off
    warns, auto-enables double precision, and returns physically bounded
    Strehl values.
    """
    jax = _jax_or_skip()
    import lumenairy as la

    orig_x64 = jax.config.jax_enable_x64
    try:
        wavelength = 587.56e-9
        aperture = 12e-3
        prescription = la.make_singlet(
            R1=51.5e-3, R2=float("inf"), d=4e-3,
            glass="N-BK7", aperture=aperture)
        surfs = la.surfaces_from_prescription(prescription)
        _, _efl, bfl, _ = la.system_abcd(surfs, wavelength=wavelength)

        N = 64
        dx = aperture / 32
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E_source = np.where(
            np.sqrt(X * X + Y * Y) <= aperture / 2, 1.0, 0.0
        ).astype(np.complex128)
        perturbation_spec = [
            {"surface_index": 0, "decenter_std": 20e-6,
             "tilt_std": 0.2e-3, "form_error_rms": 0.0},
        ]

        with pytest.warns(RuntimeWarning, match="jax_enable_x64"):
            stats = la.monte_carlo_tolerancing_jax(
                prescription=prescription, wavelength=wavelength, N=N, dx=dx,
                E_source=E_source, perturbation_spec=perturbation_spec,
                focal_length=bfl, aperture=aperture, n_trials=4, seed=0,
                z_scan_n=5, wave_propagator="real_lens", verbose=False)

        assert jax.config.jax_enable_x64, (
            "monte_carlo_tolerancing_jax should auto-enable jax_enable_x64 "
            "for a complex128 source with x64 off (audit S3-2 guard).")
        strehls = np.asarray(stats["strehl_array"])
        assert strehls.size == 4
        assert np.all(np.isfinite(strehls))
        # Physical bound (with the P3-7 focus-shift slack test_v5_3 documents).
        assert np.all(strehls >= 0.0) and np.all(strehls <= 1.05), (
            f"MC Strehl out of physical bounds: "
            f"min={strehls.min():.4f}, max={strehls.max():.4f}.")
    finally:
        jax.config.update("jax_enable_x64", orig_x64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
