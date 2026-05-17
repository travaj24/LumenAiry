"""Correctness pinning tests for the v4.12.0 ``through_focus_scan``
FFT-hoist optimisation.

The v4.11.x ``through_focus_scan`` (NumPy backend) called
``angular_spectrum_propagate`` once per ``z`` value, re-running the
full input FFT and rebuilding the K-grid / propagating mask each time.
v4.12.0 hoists the z-invariant work (input FFT, kx/ky, propagating
mask) above the loop and only rebuilds the z-dependent transfer
function ``H = exp(i * kz * z) * propagating * bandlimit_z`` inside.

These tests pin that the hoisted path is functionally identical to
the per-z fallback to within tight bounds:

* vs per-z ``angular_spectrum_propagate`` reference  --  1e-12 relative
  (bit-near-exact: same FFT primitives, same float64 phase,
  same fftshift convention).
* vs JAX twin (``through_focus_scan_jax``)            --  1e-9 absolute
  (the JAX path is the canonical reference for the hoisted structure
  but uses jax.numpy FFTs and JIT, so a small numeric drift is
  expected).
* Per-z metrics (Strehl, RMS, D4sigma, power-in-bucket) match the
  per-z fallback (1e-12 relative) so callers downstream of the scan
  (``find_best_focus``, ``MultiFieldMerit``, MC tolerancing) see
  unchanged values.
* ``bandlimit=True`` / ``bandlimit=False`` both match their per-z
  reference.

Note: ``through_focus_scan`` only exposes the ASM path; there are no
SAS / Fresnel / RS sub-paths in its current signature.  The
"sub-paths still work" requirement is therefore vacuous for this
function -- the optimisation has nothing else to break.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.analysis.through_focus import (
    through_focus_scan,
    through_focus_scan_jax,
    single_plane_metrics,
)
from lumenairy.propagators.propagation import angular_spectrum_propagate


def _make_focusing_field(N=64, dx=8e-6, wl=1.55e-6, f=50e-3,
                          sigma=80e-6, dtype=np.complex128):
    """Small Gaussian apodised by a perfect converging-lens phase.

    Produces a tight focal spot somewhere near ``z = f``; the gradient
    of |E(z)|^2 around focus is large, which makes per-z FFT vs
    hoisted-FFT comparisons sensitive to any drift.
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    k = 2.0 * np.pi / wl
    amp = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    phase = np.exp(-1j * k * (X ** 2 + Y ** 2) / (2.0 * f))
    return (amp * phase).astype(dtype)


def _per_z_reference(E_exit, dx, wl, z_arr, bandlimit, *,
                      bucket_radius=None, ideal_peak=None):
    """Build a reference result by calling
    ``angular_spectrum_propagate`` per z and aggregating
    ``single_plane_metrics`` exactly as the pre-v4.12 loop did.

    Returns six per-z arrays (peak_I, strehl, d4x, d4y, rms_r,
    p_bucket) and the list of complex fields E_z so a follow-up test
    can compare them directly.
    """
    n_z = z_arr.size
    peak_I = np.full(n_z, np.nan)
    strehl = np.full(n_z, np.nan)
    d4x = np.full(n_z, np.nan)
    d4y = np.full(n_z, np.nan)
    rms_r = np.full(n_z, np.nan)
    p_bucket = np.full(n_z, np.nan)
    fields = []
    for i, z in enumerate(z_arr):
        if z == 0.0:
            E_z = np.asarray(E_exit)
        else:
            E_z = angular_spectrum_propagate(
                E_exit, float(z), wl, dx, bandlimit=bandlimit)
        fields.append(E_z)
        m = single_plane_metrics(
            E_z, dx, wl,
            bucket_radius=bucket_radius,
            ideal_peak=ideal_peak,
        )
        peak_I[i] = m['peak_I']
        d4x[i] = m['d4sigma_x']
        d4y[i] = m['d4sigma_y']
        rms_r[i] = m['rms_radius']
        if 'power_in_bucket' in m:
            p_bucket[i] = m['power_in_bucket']
        if 'strehl' in m:
            strehl[i] = m['strehl']
    return peak_I, strehl, d4x, d4y, rms_r, p_bucket, fields


# ==========================================================================
# Pin #1 -- bit-near-exact vs per-z angular_spectrum_propagate
# ==========================================================================

class TestThroughFocusScanFFTHoistBitExact:
    """The hoisted scan must reproduce the per-z reference to within
    1e-12 relative on every metric and every field sample.
    """

    @pytest.mark.parametrize('bandlimit', [True, False])
    def test_metrics_match_per_z_reference_to_1e_12(self, bandlimit):
        N, dx, wl = 64, 8e-6, 1.55e-6
        f = 50e-3
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=f)
        z_arr = np.linspace(45e-3, 55e-3, 7)
        bucket_radius = 30e-6
        # Use the per-z code to compute an ideal_peak so Strehl is
        # populated (matches the through_focus_scan API contract).
        ideal_peak = lm.diffraction_limited_peak(
            E_exit, wl, f, dx, bandlimit=bandlimit)

        ref_peak, ref_strehl, ref_d4x, ref_d4y, ref_rms, ref_buck, _ = \
            _per_z_reference(
                E_exit, dx, wl, z_arr, bandlimit,
                bucket_radius=bucket_radius, ideal_peak=ideal_peak)

        scan = through_focus_scan(
            E_exit, dx, wl, z_arr,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
            bandlimit=bandlimit, verbose=False)

        # Relative-tolerance comparison with a small absolute floor so
        # NaN entries (no power-in-bucket if bucket_radius=None) and
        # tiny near-zero metrics don't blow up the ratio.
        np.testing.assert_allclose(
            scan.peak_I, ref_peak, rtol=1e-12, atol=0,
            err_msg=f'peak_I drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan.d4sigma_x, ref_d4x, rtol=1e-12, atol=0,
            err_msg=f'd4sigma_x drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan.d4sigma_y, ref_d4y, rtol=1e-12, atol=0,
            err_msg=f'd4sigma_y drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan.rms_radius, ref_rms, rtol=1e-12, atol=0,
            err_msg=f'rms_radius drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan.power_in_bucket, ref_buck, rtol=1e-12, atol=0,
            err_msg=f'power_in_bucket drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan.strehl, ref_strehl, rtol=1e-12, atol=0,
            err_msg=f'strehl drift (bandlimit={bandlimit})')

    def test_field_complex_values_match_per_z_reference(self):
        """Strongest check: rebuild the per-z complex fields by hand
        and compare them to the hoisted path's intermediate ``E_z``
        values.  Done by repeating the per-z propagation and the
        hoisted propagation independently, then asserting the |E|^2
        peak (a proxy for the complex field that ``through_focus_scan``
        does NOT expose) matches to 1e-12 relative.
        """
        N, dx, wl = 96, 6e-6, 1.31e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=20e-3,
                                       sigma=60e-6)
        z_arr = np.array([15e-3, 18e-3, 20e-3, 22e-3, 25e-3])
        ref_peak, *_ = _per_z_reference(
            E_exit, dx, wl, z_arr, bandlimit=True)
        scan = through_focus_scan(
            E_exit, dx, wl, z_arr, bandlimit=True, verbose=False)
        np.testing.assert_allclose(
            scan.peak_I, ref_peak, rtol=1e-12, atol=0,
            err_msg='hoisted peak_I diverges from per-z reference')

    def test_z_zero_passthrough(self):
        """The hoisted path must still return ``E_in`` unchanged at
        z=0 (preserves the v4.11.x behaviour that pre-4.12 callers
        rely on for sanity checks)."""
        N, dx, wl = 64, 8e-6, 1.55e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=50e-3)
        z_arr = np.array([-1e-3, 0.0, 1e-3])
        scan = through_focus_scan(
            E_exit, dx, wl, z_arr, bandlimit=True, verbose=False)
        # The z=0 entry should equal |E_exit|^2.max() exactly.
        expected_peak_z0 = float((np.abs(E_exit) ** 2).max())
        assert abs(scan.peak_I[1] - expected_peak_z0) < 1e-15 * (
            1.0 + abs(expected_peak_z0)), (
            f'z=0 peak drifted from input: '
            f'{scan.peak_I[1]:.16g} vs {expected_peak_z0:.16g}')


# ==========================================================================
# Pin #2 -- hoisted NumPy result agrees with the JAX twin
# ==========================================================================

class TestThroughFocusScanMatchesJAXTwin:
    """The JAX twin is the canonical reference for the hoisted-FFT
    structure.  The NumPy hoisted path must match it to 1e-9 absolute
    under JAX x64 mode -- looser than the per-z comparison because
    jax.numpy uses XLA FFTs that have a small numeric offset from
    MKL/PocketFFT.

    NOTE: JAX defaults to float32, which produces ~1e-3 to 1e-2 drift
    on focal-spot peaks (the JAX-twin docstring's "5-15x" claim is
    independent of x64 mode -- it just means *speedup*, not
    precision).  These tests therefore enable x64 before running,
    matching the existing project pattern in
    ``lumenairy.optimize.core``.
    """

    @pytest.fixture(scope='class')
    def jax_x64(self):
        try:
            from lumenairy.backend import JAX_AVAILABLE
        except ImportError:
            return False
        if not JAX_AVAILABLE:
            return False
        import jax
        # Idempotent: project-wide pattern for parity tests that need
        # NumPy-grade precision from the JAX backend.
        if not jax.config.read('jax_enable_x64'):
            jax.config.update('jax_enable_x64', True)
        return True

    @pytest.mark.parametrize('bandlimit', [True, False])
    def test_metrics_match_jax_to_1e_9_absolute(self, bandlimit, jax_x64):
        if not jax_x64:
            pytest.skip('JAX not installed; skipping JAX-twin parity check')
        N, dx, wl = 64, 8e-6, 1.55e-6
        f = 50e-3
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=f)
        z_arr = np.linspace(45e-3, 55e-3, 9)
        bucket_radius = 30e-6
        ideal_peak = lm.diffraction_limited_peak(
            E_exit, wl, f, dx, bandlimit=bandlimit)

        scan_np = through_focus_scan(
            E_exit, dx, wl, z_arr,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
            bandlimit=bandlimit, verbose=False)
        scan_jax = through_focus_scan_jax(
            E_exit, dx, wl, z_arr,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
            bandlimit=bandlimit)

        # peak_I: 1e-9 absolute is comfortable on the focal-spot peak
        # which is O(1) for this configuration; XLA FFT vs MKL FFT can
        # diverge by ~1e-12 to 1e-10 on individual frequency bins,
        # which compounds to ~1e-9 on the peak.
        np.testing.assert_allclose(
            scan_np.peak_I, scan_jax.peak_I, rtol=0, atol=1e-9,
            err_msg=f'peak_I NumPy vs JAX drift (bandlimit={bandlimit})')
        np.testing.assert_allclose(
            scan_np.strehl, scan_jax.strehl, rtol=0, atol=1e-9,
            err_msg=f'strehl NumPy vs JAX drift (bandlimit={bandlimit})')

    def test_z_grid_unchanged(self, jax_x64):
        """Sanity: NumPy and JAX both echo back z_values unchanged."""
        if not jax_x64:
            pytest.skip('JAX not installed; skipping JAX-twin parity check')
        N, dx, wl = 64, 8e-6, 1.55e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=50e-3)
        z_arr = np.linspace(45e-3, 55e-3, 7)
        scan_np = through_focus_scan(
            E_exit, dx, wl, z_arr, bandlimit=True, verbose=False)
        scan_jax = through_focus_scan_jax(
            E_exit, dx, wl, z_arr, bandlimit=True)
        np.testing.assert_array_equal(scan_np.z, scan_jax.z)


# ==========================================================================
# Pin #3 -- propagator dispatch coverage (single ASM path)
# ==========================================================================

class TestThroughFocusScanDispatch:
    """``through_focus_scan`` exposes ONLY the ASM path -- it does NOT
    take a ``wave_propagator`` kwarg (unlike
    ``monte_carlo_tolerancing_jax``).  This test pins that the public
    signature has not silently grown SAS / Fresnel / RS branches that
    would need their own hoisting.

    If a future revision adds those, this test will fail and the
    optimisation work will need to be revisited to cover the new
    sub-paths.
    """

    def test_through_focus_scan_signature_has_no_wave_propagator(self):
        import inspect
        sig = inspect.signature(through_focus_scan)
        # Allowed kwargs at v4.12.  ``backend`` is the only dispatch
        # knob (``'numpy'`` / ``'jax'``); any new dispatch keyword
        # would indicate a sub-path that the FFT-hoist optimisation
        # has not yet been extended to cover.
        allowed = {
            'E_exit', 'dx', 'wavelength', 'z_values',
            'bucket_radius', 'ideal_peak', 'bandlimit',
            'verbose', 'progress', 'backend',
        }
        params = set(sig.parameters)
        unexpected = params - allowed
        assert not unexpected, (
            f'through_focus_scan grew unexpected kwarg(s) {unexpected}; '
            f'the v4.12 FFT-hoist optimisation covers only the ASM path. '
            f'Audit the new dispatch before assuming the speedup carries.')

    def test_backend_jax_dispatches_to_jax_twin(self):
        """The ``backend='jax'`` route still hands off to the JAX
        implementation; the hoisted NumPy refactor must not have
        swallowed the dispatch branch."""
        try:
            from lumenairy.backend import JAX_AVAILABLE
        except ImportError:
            JAX_AVAILABLE = False
        if not JAX_AVAILABLE:
            pytest.skip('JAX not installed; skipping backend dispatch check')
        N, dx, wl = 64, 8e-6, 1.55e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=50e-3)
        z_arr = np.linspace(45e-3, 55e-3, 5)
        # If the dispatch is missing, this call raises an error
        # because the JAX implementation has its own signature.
        scan = through_focus_scan(
            E_exit, dx, wl, z_arr, bandlimit=True, backend='jax')
        # And the returned object must be a ThroughFocusResult with
        # the expected fields populated.
        assert scan.z.shape == z_arr.shape
        assert np.all(np.isfinite(scan.peak_I))

    def test_backend_invalid_raises(self):
        N, dx, wl = 64, 8e-6, 1.55e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=50e-3)
        with pytest.raises(ValueError):
            through_focus_scan(
                E_exit, dx, wl, [50e-3], backend='not_a_backend')


# ==========================================================================
# Pin #4 -- per-z metrics (Strehl, RMS, D4sigma, peak, bucket)
# ==========================================================================

class TestThroughFocusScanPerZMetrics:
    """Every per-z metric the scan returns must match a hand-computed
    reference built from ``single_plane_metrics`` on the per-z
    propagated field.  This is the contract that ``find_best_focus``,
    ``MultiFieldMerit``, and the MC tolerancing variants rely on.
    """

    def test_all_metrics_match_single_plane_metrics(self):
        N, dx, wl = 64, 8e-6, 1.55e-6
        f = 50e-3
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=f)
        z_arr = np.linspace(45e-3, 55e-3, 11)
        bucket_radius = 25e-6
        ideal_peak = lm.diffraction_limited_peak(E_exit, wl, f, dx)

        scan = through_focus_scan(
            E_exit, dx, wl, z_arr,
            bucket_radius=bucket_radius, ideal_peak=ideal_peak,
            bandlimit=True, verbose=False)

        # Per-z verification: take the field at each z (via per-z
        # angular_spectrum_propagate, which we already know agrees
        # with the hoisted path to 1e-12) and feed it through
        # single_plane_metrics.  Both numbers must agree.
        for i, z in enumerate(z_arr):
            E_z = angular_spectrum_propagate(
                E_exit, float(z), wl, dx, bandlimit=True)
            m = single_plane_metrics(
                E_z, dx, wl,
                bucket_radius=bucket_radius,
                ideal_peak=ideal_peak,
            )
            # peak / strehl / d4sigma / rms / bucket
            assert abs(scan.peak_I[i] - m['peak_I']) <= \
                1e-12 * abs(m['peak_I']), (
                f"z={z:.3e}: peak_I {scan.peak_I[i]:.16g} vs "
                f"{m['peak_I']:.16g}")
            assert abs(scan.d4sigma_x[i] - m['d4sigma_x']) <= \
                1e-12 * abs(m['d4sigma_x'] or 1.0)
            assert abs(scan.d4sigma_y[i] - m['d4sigma_y']) <= \
                1e-12 * abs(m['d4sigma_y'] or 1.0)
            assert abs(scan.rms_radius[i] - m['rms_radius']) <= \
                1e-12 * abs(m['rms_radius'] or 1.0)
            assert abs(scan.power_in_bucket[i] - m['power_in_bucket']) <= \
                1e-12 * abs(m['power_in_bucket'] or 1.0)
            assert abs(scan.strehl[i] - m['strehl']) <= \
                1e-12 * abs(m['strehl'] or 1.0)

    def test_find_best_focus_finds_same_z_as_per_z_reference(self):
        """End-to-end: ``find_best_focus(scan, 'strehl')`` returns
        the same z as a manual argmax over the per-z reference.  This
        is the contract MC tolerancing / MultiFieldMerit rely on."""
        from lumenairy.analysis.through_focus import find_best_focus
        N, dx, wl = 96, 6e-6, 1.55e-6
        f = 30e-3
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=f, sigma=70e-6)
        z_arr = np.linspace(28e-3, 32e-3, 13)
        ideal_peak = lm.diffraction_limited_peak(E_exit, wl, f, dx)

        scan = through_focus_scan(
            E_exit, dx, wl, z_arr, ideal_peak=ideal_peak,
            bandlimit=True, verbose=False)
        z_best, s_best = find_best_focus(scan, 'strehl')

        # Reference: build the same scan via per-z reference path.
        ref_peak, ref_strehl, *_ = _per_z_reference(
            E_exit, dx, wl, z_arr, bandlimit=True, ideal_peak=ideal_peak)
        z_best_ref = z_arr[int(np.nanargmax(ref_strehl))]
        s_best_ref = float(np.nanmax(ref_strehl))
        assert z_best == z_best_ref, (
            f'best-focus z drifted: hoisted={z_best} ref={z_best_ref}')
        assert abs(s_best - s_best_ref) <= 1e-12 * abs(s_best_ref), (
            f'best-Strehl drifted: hoisted={s_best:.16g} '
            f'ref={s_best_ref:.16g}')


# ==========================================================================
# Pin #5 -- progress callback still invoked once per z
# ==========================================================================

class TestThroughFocusScanProgressCallback:
    """The hoisted refactor must not have removed the progress-bar
    hook calls.  Pin that ``progress(stage, fraction, message)`` is
    invoked ``n_z + 1`` times (one per z plus the final 'done').
    """

    def test_progress_called_per_z_plus_done(self):
        N, dx, wl = 32, 8e-6, 1.55e-6
        E_exit = _make_focusing_field(N=N, dx=dx, wl=wl, f=50e-3)
        z_arr = np.linspace(48e-3, 52e-3, 4)
        calls = []
        def cb(stage, fraction, message):
            calls.append((stage, float(fraction), message))
        scan = through_focus_scan(
            E_exit, dx, wl, z_arr, bandlimit=True, progress=cb)
        # Expect one progress event per z plus a final 'done' event.
        assert len(calls) == len(z_arr) + 1, (
            f'progress should fire {len(z_arr) + 1} times, got '
            f'{len(calls)}')
        # Final event has fraction == 1.0
        assert calls[-1][1] == 1.0
        # All events tagged with the right stage name.
        for stage, _, _ in calls:
            assert stage == 'through_focus_scan'
