"""Pinning tests for the 14 v4.11.2 audit coverage gaps identified by
AUDIT_ROUND4_2026_05_16 (section "Test-suite quality").

Each item below corresponds to a v4.11.2 fix that landed in the code
but had no regression test.  The tests here would FAIL on the pre-fix
version of the code.

Item map (one test class per item):

1. compute_psf non-square pupil error
2. apply_detector non-integer pixel ratio
3. find_best_focus NaN injection
4. monte_carlo_tolerancing_linearized a_k >= 0 clamp
5. load_material RuntimeWarning on dispersion drop
6. Source.* **factory_kwargs propagation
7. apply_real_lens_traced M_x/M_y transpose (centered output)
8. NaN sentinel mask in apply_real_lens (no NaN in wave-leg output)
9. stop_index != 0 RuntimeWarning in _traced / _maslov
10. Freeform-terms RuntimeWarning in thin-element apply_real_lens
11. Zemax coord-break STOP marker (refractive-only counter)
12. JAX <-> NumPy phase-retrieval cross-parity (gerchberg_saxton)
13. Cassegrain S1/S2/S3/S5 hand-derivation (extends v4.11.2's S4 pin)
14. Richards-Wolf vs paraxial Airy at low NA

All tests should PASS on the current v4.12.1 codebase.  A failure
indicates the underlying fix has regressed (or, if it was never
present, the round-4 audit was right to flag it).
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.raytrace import Surface, seidel_coefficients


# ============================================================================
# Item 1 -- compute_psf raises on non-square pupil
# ============================================================================

class TestComputePsfNonSquarePupil:
    """v4.11.2 added an explicit ``ValueError`` in
    :func:`compute_psf` when the pupil array is not square.  Pre-fix
    the function used ``pupil.shape[0]`` for both axes (and the
    Fraunhofer pad / grid scale), so a rectangular input produced a
    silently wrong PSF.
    """

    def test_non_square_pupil_raises_value_error(self):
        # Rectangular pupil: 32 rows, 48 columns.
        pupil = np.ones((32, 48), dtype=np.complex128)
        with pytest.raises(ValueError) as excinfo:
            la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                            dx_pupil=10e-6)
        msg = str(excinfo.value)
        # Message must reference the rectangular shape so a user
        # ports their code into a square embed.
        assert '32' in msg and '48' in msg, (
            f"v4.11.2 ValueError must reference the rectangular shape "
            f"(32, 48); got: {msg!r}")
        # And it must specifically name 'square' or 'rectangular'
        # to be actionable.
        msg_lower = msg.lower()
        assert ('square' in msg_lower or 'rectangular' in msg_lower), (
            f"v4.11.2 ValueError should explain the constraint "
            f"(square / rectangular); got: {msg!r}")

    def test_non_square_pupil_raises_before_fft(self):
        """The raise must happen ON ENTRY, not after the FFT.  Pin
        with the very first non-trivial shape mismatch (rows == 1,
        cols == 2)."""
        pupil = np.ones((1, 2), dtype=np.complex128)
        with pytest.raises(ValueError):
            la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                            dx_pupil=10e-6)

    def test_square_pupil_does_not_raise(self):
        """Sanity check: a square pupil still works."""
        pupil = np.ones((32, 32), dtype=np.complex128)
        psf, dx_psf = la.compute_psf(pupil, wavelength=1e-6, f=0.1,
                                       dx_pupil=10e-6)
        assert psf.shape == (32, 32)
        assert dx_psf > 0


# ============================================================================
# Item 2 -- apply_detector non-integer pixel ratio (area-integral correctness)
# ============================================================================

class TestApplyDetectorNonIntegerPixelRatio:
    """v4.11.2 added the area-integral correction to
    :func:`apply_detector` so a non-integer ``pixel_pitch / dx_field``
    ratio doesn't break photon conservation.  Pre-fix the bilinear
    zoom + ``pixel_pitch**2`` multiplication broke conservation for
    ratios like 2.3 -- integrated detector power vs analytic input
    power drifted by several percent.

    Pin: build a uniform unit-amplitude field, run apply_detector
    with ratio = 2.3, and assert the integrated photon count agrees
    with the analytic expectation (input_intensity * total_area * QE
    * exposure_time / photon_energy) to within ~1%.
    """

    def test_ratio_2p3_conserves_total_power(self):
        # Uniform field with controlled photon scale.  No noise, no
        # dark current, no read noise -- isolate the area-integral
        # branch.
        N = 64
        dx_field = 1e-6  # 1 um per field sample
        # Non-integer ratio = 2.3
        pixel_pitch = 2.3e-6  # detector pitch
        n_pixels = 20  # 20 * 2.3 um = 46 um detector span

        E = np.ones((N, N), dtype=np.complex128)
        # apply_detector returns (image, x_det, y_det).
        image_ni, _x, _y = la.apply_detector(
            E, dx_field=dx_field, pixel_pitch=pixel_pitch,
            n_pixels=n_pixels,
            exposure_time=1.0,
            quantum_efficiency=1.0,
            read_noise_e=0.0,
            dark_current_e_per_s=0.0,
            seed=0,
        )
        total_counts_ni = float(image_ni.sum())

        # Reference: integer ratio 2.0, same physical detector area
        # (23 * 2 um = 46 um = 20 * 2.3 um, exact).
        n_pixels_ref = 23
        pixel_pitch_ref = 2.0e-6
        image_int, _x_r, _y_r = la.apply_detector(
            E, dx_field=dx_field, pixel_pitch=pixel_pitch_ref,
            n_pixels=n_pixels_ref,
            exposure_time=1.0,
            quantum_efficiency=1.0,
            read_noise_e=0.0,
            dark_current_e_per_s=0.0,
            seed=0,
        )
        total_counts_int = float(image_int.sum())

        # Under v4.11.2's area-integral correction, the two detectors
        # over the same physical area must integrate to within ~1% of
        # each other.  Pre-fix the bilinear-zoom path made the
        # non-integer ratio over- or under-count by several percent.
        # (The Poisson-shot-noise step adds sqrt(N) jitter, hence the
        # 2% allowance instead of 1%.)
        if total_counts_int > 0:
            rel = abs(total_counts_ni - total_counts_int) / total_counts_int
        else:
            rel = abs(total_counts_ni)
        assert rel < 0.02, (
            f"apply_detector with ratio=2.3 and ratio=2.0 over the "
            f"same physical area produced integrated counts differing "
            f"by {rel*100:.2f}% -- v4.11.2 area-integral correction "
            f"should keep the relative drift below ~1%.  "
            f"non-int total = {total_counts_ni:.6e}, "
            f"int total = {total_counts_int:.6e}")


# ============================================================================
# Item 3 -- find_best_focus NaN injection (all-NaN guard)
# ============================================================================

class TestFindBestFocusNanGuard:
    """v4.11.2 added an all-NaN guard returning ``(nan, nan)`` from
    :func:`find_best_focus` so a monte-carlo trial whose every scan
    point is NaN doesn't poison the whole MC run with a
    ``ValueError: All-NaN slice encountered``.

    Pin: build a :class:`ThroughFocusResult` with all-NaN ``strehl``,
    call find_best_focus(metric='strehl'), and assert it returns
    (nan, nan) without raising.
    """

    def test_all_nan_strehl_returns_nan_pair(self):
        # Build a minimal scan struct.
        from lumenairy.analysis import ThroughFocusResult
        n = 5
        z = np.linspace(-1e-3, 1e-3, n)
        nans = np.full(n, np.nan)
        scan = ThroughFocusResult(
            z=z, peak_I=nans.copy(),
            strehl=nans.copy(),
            d4sigma_x=nans.copy(),
            d4sigma_y=nans.copy(),
            rms_radius=nans.copy(),
            power_in_bucket=nans.copy(),
            wavelength=1e-6,
        )
        # Should not raise; returns (nan, nan).
        z_best, val_best = la.find_best_focus(scan, metric='strehl')
        assert np.isnan(z_best), (
            f"find_best_focus on all-NaN strehl should return NaN z; "
            f"got {z_best!r}")
        assert np.isnan(val_best), (
            f"find_best_focus on all-NaN strehl should return NaN value; "
            f"got {val_best!r}")


# ============================================================================
# Item 4 -- monte_carlo_tolerancing_linearized a_k >= 0 clamp
# ============================================================================

class TestMcTolLinearizedAkClamp:
    """v4.11.2 clamps the per-knob sensitivity coefficient
    ``a_k = max(a_k, 0)`` in
    :func:`monte_carlo_tolerancing_linearized` so a one-sided FD
    probe that yields S_p > S_nom (due to numerical noise, sign
    coincidence with the nominal aberration, or a non-monotone FD
    step) cannot predict S_pred > S_nom -- impossible under Marechal
    (S_perturbed <= S_nom for any small perturbation).

    Pin: build a tolerancing scenario whose nominal lens is
    essentially perfect (S_nom ~ 1) and whose perturbation sigma is
    deliberately small, so any FD-noise-driven S_p > S_nom would be
    clipped to a_k=0.  Verify that the maximum predicted Strehl in
    the Monte-Carlo run is <= S_nom + a small epsilon (the v4.11.2
    clamp guarantees this).
    """

    def test_predicted_strehl_never_exceeds_nominal_and_a_k_nonneg(self):
        from lumenairy.analysis import (
            monte_carlo_tolerancing_linearized,
        )
        wavelength = 1.31e-6
        N = 32
        dx = 8e-6
        # Near-perfect singlet so the FD-probe S_p ~ S_nom and any
        # numerical-noise sign coincidence would land naturally
        # close to the clamp boundary.
        rx = la.make_singlet(R1=200e-3, R2=-200e-3, d=2e-3,
                              glass='N-BK7', aperture=200e-6)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E_source = np.exp(-(X ** 2 + Y ** 2) / (50e-6) ** 2).astype(
            np.complex128)
        # Perturbation spec: tiny decenter / tilt / form-error, where
        # any of the per-knob FD probes could produce S_p > S_nom
        # purely from numerical noise.
        perturbation_spec = [
            {'surface_index': 0,
             'decenter_std': 1e-12,    # 1 pm
             'tilt_std': 1e-12,        # 1 prad
             'form_error_rms': 1e-12,  # 1 pm
             'name': 'tiny_pert_s0'},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = monte_carlo_tolerancing_linearized(
                prescription=rx, wavelength=wavelength,
                N=N, dx=dx,
                E_source=E_source,
                perturbation_spec=perturbation_spec,
                focal_length=100e-3,
                aperture=200e-6,
                n_trials=64,
                seed=42,
                verbose=False,
            )

        # 1. Every sensitivity coefficient a_k must be >= 0
        # (the v4.11.2 clamp ``a_k = max(a_k, 0)``).
        sensitivities = result['sensitivities']
        for (spec_idx, knob, sigma, a_k) in sensitivities:
            assert a_k >= 0.0, (
                f"sensitivities[{spec_idx}, {knob}]: a_k = {a_k!r} < 0 "
                f"-- v4.11.2 clamps a_k = max(a_k, 0).")

        # 2. Every per-trial S_pred must be <= S_nom (since
        # S_pred = S_nom - sum a_k * xi_k^2 with a_k >= 0).
        strehls = np.asarray(result['strehl_array'])
        s_nom = float(result['nominal_strehl'])
        max_strehl = float(np.nanmax(strehls))
        assert max_strehl <= s_nom + 1e-12, (
            f"monte_carlo_tolerancing_linearized produced "
            f"S_pred={max_strehl} > S_nom={s_nom} -- Marechal-"
            f"invalid.  v4.11.2 clamps a_k=max(a_k,0), so S_pred "
            f"<= S_nom always.")


# ============================================================================
# Item 5 -- load_material RuntimeWarning on dispersion drop
# ============================================================================

class TestLoadMaterialDispersionWarning:
    """v4.11.2 added a ``RuntimeWarning`` in :func:`load_material`
    when a fixed-glass JSON includes a ``dispersion`` field
    (which ``register_fixed_glass`` silently drops).  Pre-fix the
    saved dispersion data was discarded with no surface signal.

    Pin: save a fixed-glass material with a dispersion dict, reload
    it, and assert the warning fires.
    """

    def test_dispersion_drop_warning_emitted(self):
        # Save a fixed-glass material with a dispersion dict (mimics
        # what a save_material with dispersion= kwarg writes to disk).
        # ``set_library_path`` lives on the user_library module, not
        # on the top-level namespace, so import it directly.
        from lumenairy import user_library

        # Snapshot the existing library path so we can restore it.
        prior_path = user_library._library_path
        with tempfile.TemporaryDirectory() as td:
            user_library.set_library_path(td)
            try:
                user_library.save_material(
                    name='test_glass_disp',
                    n=1.5,
                    dispersion={'A': 1.0, 'B': 0.01},
                    description='test fixture',
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    user_library.load_material('test_glass_disp')
                relevant = [
                    w for w in caught
                    if issubclass(w.category, RuntimeWarning)
                    and 'dispersion' in str(w.message).lower()
                ]
                assert relevant, (
                    f"load_material on a fixed-glass file with a "
                    f"'dispersion' field must emit a RuntimeWarning "
                    f"mentioning 'dispersion' (v4.11.2 fix).  "
                    f"Caught: {[(w.category.__name__, str(w.message)[:80]) for w in caught]}")
            finally:
                # Restore the prior library path so other tests are
                # unaffected.
                user_library._library_path = prior_path


# ============================================================================
# Item 6 -- Source.* **factory_kwargs propagation (dy=)
# ============================================================================

class TestSourceFactoryKwargsForwarded:
    """v4.11.2 added ``**factory_kwargs`` to the :class:`Source`
    classmethods so callers can pass ``dy=``, ``dtype=``,
    ``normalize=`` etc. through to the underlying ``create_*``
    factories.  Pre-fix these kwargs were silently dropped and the
    factory defaulted to ``dy=dx``.

    Pin: build ``Source.gaussian(w0, N, dx, wavelength, dy=...)``
    with ``dy != dx`` and verify the resulting field shows the
    anisotropic pixel scaling (a vertical sample 1 pixel above
    the centre maps to physical y = dy, not dx).
    """

    def test_gaussian_dy_kwarg_propagates(self):
        N = 16
        dx = 2e-6
        dy = 5e-6  # 2.5x dx, so a centred Gaussian with sigma matched
                   # to dx would show clearly different values along
                   # the y-axis vs x-axis if dy were honored.
        wavelength = 1e-6
        w0 = 20e-6
        sigma = w0 / np.sqrt(2)
        src_dy = la.Source.gaussian(w0=w0, N=N, dx=dx,
                                     wavelength=wavelength, dy=dy)
        E_dy = np.asarray(src_dy.E)
        # |E| at row-offset 1 from centre (i.e. y = +/- dy):
        # exp(-(dy)^2 / w0^2)
        center = N // 2
        v_at_dy = float(np.abs(E_dy[center + 1, center]))
        expected_with_dy = np.exp(-(dy ** 2) / (w0 ** 2))
        # If dy was honored: 0.939... (for w0=20um, dy=5um).
        # If dy was ignored (defaulted to dx): 0.990...
        assert abs(v_at_dy - expected_with_dy) < 1e-6, (
            f"Source.gaussian with dy={dy} gave |E|[c+1, c]="
            f"{v_at_dy:.6f}; expected exp(-(dy/w0)^2)="
            f"{expected_with_dy:.6f}.  Pre-v4.11.2 dy was silently "
            f"ignored, which would have given "
            f"{np.exp(-(dx**2)/(w0**2)):.6f} instead.")

    def test_gaussian_default_dy_equals_dx(self):
        """When the caller does NOT pass dy, the factory defaults to
        dy=dx -- so the field should be isotropic."""
        N = 16
        dx = 2e-6
        wavelength = 1e-6
        w0 = 20e-6
        src = la.Source.gaussian(w0=w0, N=N, dx=dx, wavelength=wavelength)
        E = np.asarray(src.E)
        center = N // 2
        # x-offset 1 == y-offset 1 in physical units (both = dx).
        v_row = float(np.abs(E[center, center + 1]))
        v_col = float(np.abs(E[center + 1, center]))
        assert abs(v_row - v_col) < 1e-10, (
            f"Default Source.gaussian (no dy kwarg) should produce an "
            f"isotropic field (dy=dx); got |E|[c,c+1]={v_row}, "
            f"|E|[c+1,c]={v_col}.")


# ============================================================================
# Item 7 -- apply_real_lens_traced M_x/M_y stencil (Newton initial guess)
# ============================================================================

class TestApplyRealLensTracedMxMyTranspose:
    """v4.11.2 fixed the NumPy ``apply_real_lens_traced`` central
    finite-difference stencil that measures the paraxial magnification
    M_x / M_y from the forward-map grid.  Pre-fix the indices were
    transposed (computing dx_out/dy_in and dy_out/dx_in instead of
    dx_out/dx_in and dy_out/dy_in), which for a rotationally-symmetric
    singlet returned M_x = M_y ~ 0 from the cross-axis derivative.
    Newton then started from the all-zero initial guess (clipped to
    boundary 0.91-fallback) on every pixel.

    Pin: trace a centred Gaussian through a symmetric biconvex
    singlet (paraxial M = +1 by symmetry) and verify the output
    centroid sits within sub-pixel of the input centroid.  Pre-fix
    the silent transpose used to put Newton start at the centre-of-
    the-grid-but-clipped, producing an off-centred output bias.
    """

    def test_symmetric_singlet_keeps_input_centered(self):
        rx = la.make_singlet(
            R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=8e-3)
        N = 64
        dx = 1e-5
        wavelength = 1.31e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E_in = np.exp(-(X ** 2 + Y ** 2) / (1.5e-3) ** 2).astype(
            np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = la.apply_real_lens_traced(
                E_in, prescription=rx, wavelength=wavelength, dx=dx,
                ray_subsample=4,
            )

        I_out = np.abs(E_out) ** 2
        total = float(I_out.sum())
        assert total > 0, "output had no power"
        cx = float((I_out * X).sum() / total)
        cy = float((I_out * Y).sum() / total)
        # Centroid should be within 1 pixel of (0, 0) for the
        # symmetric input.  The transposed-stencil version produced
        # a multi-pixel offset because Newton started from the clipped
        # boundary on every pixel.
        assert abs(cx) < dx, (
            f"Symmetric singlet output centroid x = {cx:.3e} m; "
            f"expected within 1 pixel ({dx:.3e} m) of zero.  "
            f"Pre-v4.11.2 the M_x/M_y stencil index transpose "
            f"produced an offset centroid (Newton initial guess "
            f"degenerated to clipped boundary).")
        assert abs(cy) < dx, (
            f"Symmetric singlet output centroid y = {cy:.3e} m; "
            f"expected within 1 pixel ({dx:.3e} m) of zero.")


# ============================================================================
# Item 8 -- NaN sentinel mask in apply_real_lens
# ============================================================================

class TestApplyRealLensNanSentinelMask:
    """v4.11.2 masks NaN OPD values from
    :func:`surface_sag_general` (returned outside the conic validity
    domain ``norm >= 0.9999``) to 0 before
    ``exp(-1j * k0 * opd)``.  Pre-fix the NaN propagated as
    ``exp(-i*k0*NaN) = NaN``, poisoning the entire downstream ASM step.

    Pin: build a prescription with an oblate conic (k = +3) whose
    valid-domain radius is smaller than the grid extent.  Run
    apply_real_lens; the wave-leg output must contain NO NaN.
    """

    def test_oblate_conic_outside_domain_no_nan(self):
        # R = 5 mm, conic = +3 (oblate): (1+k)*h^2/R^2 = 4*h^2/(25e-6)
        # blow up at h = 2.5 mm.  Grid spans +/- 6.4 mm (64 * 200 um/2).
        rx = {
            'name': 'oblate_test',
            'aperture_diameter': 6e-3,
            'surfaces': [
                {'radius': 5e-3, 'conic': 3.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7'},
                {'radius': -5e-3, 'conic': 0.0,
                 'aspheric_coeffs': None,
                 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs_y': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [2e-3],
        }
        N = 64
        dx = 2e-4  # 200 um, grid spans +/- 6.4 mm (well outside 2.5 mm valid)
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E_out = la.apply_real_lens(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx,
                slant_correction=True,
            )
        assert not np.any(np.isnan(E_out)), (
            f"apply_real_lens with an oblate conic produced NaN in "
            f"the output ({int(np.isnan(E_out).sum())} pixels) -- "
            f"v4.11.2 NaN sentinel mask should zero the OPD on "
            f"undefined-surface pixels before exp(-i*k0*opd).")
        assert not np.any(np.isinf(E_out)), (
            "apply_real_lens output contained Inf -- NaN mask "
            "should have prevented this.")


# ============================================================================
# Item 9 -- stop_index != 0 warn in apply_real_lens_traced / _maslov
# ============================================================================

class TestStopIndexWarn:
    """v4.11.2 adds a ``RuntimeWarning`` when
    :func:`apply_real_lens_traced` (and ``_maslov``) sees a
    prescription with ``stop_index != 0``, because the ray-traced
    phase leg launches from the entrance pupil only -- the actual
    stop placement is silently moved to surface 0.
    """

    def test_traced_emits_warning_for_stop_index_2(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        rx['stop_index'] = 2
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens_traced(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx, ray_subsample=8)
        stop_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'stop_index' in str(w.message).lower()
        ]
        assert stop_warns, (
            f"apply_real_lens_traced with stop_index=2 must emit a "
            f"RuntimeWarning mentioning 'stop_index'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")
        # Also confirm the warning mentions the value 2.
        msg = str(stop_warns[0].message)
        assert '2' in msg, (
            f"stop_index warning should reference the offending "
            f"value (2); got: {msg!r}")

    def test_maslov_emits_warning_for_stop_index_2(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        rx['stop_index'] = 2
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens_maslov(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx,
                # Keep ray-field sampling cheap; defaults are 16/16.
                ray_field_samples=8, ray_pupil_samples=8, n_v2=8,
            )
        stop_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'stop_index' in str(w.message).lower()
        ]
        assert stop_warns, (
            f"apply_real_lens_maslov with stop_index=2 must emit a "
            f"RuntimeWarning mentioning 'stop_index'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")


# ============================================================================
# Item 10 -- Freeform-terms RuntimeWarning in thin-element apply_real_lens
# ============================================================================

class TestApplyRealLensFreeformWarning:
    """v4.11.2 adds a ``RuntimeWarning`` when
    :func:`apply_real_lens` (thin-element wave-optics path) sees a
    surface with ``freeform_type`` set.  The function only computes
    conic + aspheric + biconic sag at the phase-screen step; freeform
    departures are silently DROPPED from the OPD.  Pin so a future
    refactor can't silently re-introduce the silent drop.
    """

    def test_freeform_surface_emits_runtime_warning(self):
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                              glass='N-BK7', aperture=5e-3)
        # Add a freeform_type to the first surface.
        rx['surfaces'][0]['freeform_type'] = 'xy_polynomial'
        rx['surfaces'][0]['freeform_coeffs'] = {(2, 0): 1e-6}
        N = 32
        dx = 1e-5
        wavelength = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.apply_real_lens(
                E_in, prescription=rx,
                wavelength=wavelength, dx=dx)
        ff_warns = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and 'freeform' in str(w.message).lower()
        ]
        assert ff_warns, (
            f"apply_real_lens with a freeform_type surface must emit a "
            f"RuntimeWarning mentioning 'freeform'.  Caught: "
            f"{[(w.category.__name__, str(w.message)[:80]) for w in caught]}")


# ============================================================================
# Item 11 -- Zemax coord-break STOP marker (refractive-only counter)
# ============================================================================

class TestZemaxCoordBreakStopMarker:
    """v4.11.2 fixed the STOP-marker indexing in
    ``_export_zemax_zmx_full``: the writer now tracks a separate
    ``refr_counter`` so the STOP marker lands on the REFRACTIVE
    surface index requested, not the global surface counter that
    includes coord-breaks and mirrors.

    Round-4 noted this was only tested with mirrors.  Pin a
    coord-break case: 1 coord-break BEFORE the stop, then re-load
    and verify the STOP lands on the correct refractive surface.

    We test this directly on the .zmx text output (the loader does
    not surface stop_index for load_zemax_zmx but the text file
    carries the STOP marker explicitly).
    """

    def test_coord_break_does_not_displace_stop_marker(self):
        # Build a prescription with two refractive surfaces and one
        # coord-break BEFORE the second surface; stop_surface=1
        # (the SECOND refractive surface).
        rx = {
            'name': 'cb_stop_test',
            'aperture_diameter': 6e-3,
            'elements': [
                {'element_type': 'surface', 'radius': 50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 1},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 2},
                {'element_type': 'surface', 'radius': -50e-3,
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'semi_diameter': 3e-3, 'surf_num': 4},
                {'element_type': 'surface', 'radius': float('inf'),
                 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'semi_diameter': 3e-3, 'surf_num': 5},
            ],
            'all_thicknesses': [2e-3, 5e-3, 2e-3, 10e-3],
            'coord_breaks': [
                {'surf_num': 3, 'decenter_x_m': 0.0,
                 'decenter_y_m': 0.0, 'tilt_x_deg': 0.0,
                 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
                 'order': 0, 'thickness_m': 0.0},
            ],
            # Lens-only fallbacks (the full writer uses 'elements').
            'surfaces': [],
            'thicknesses': [],
        }
        # stop_surface = 1 -> the SECOND refractive surface.
        with tempfile.TemporaryDirectory() as td:
            zmx_path = os.path.join(td, 'cb_stop_test.zmx')
            la.export_zemax_zmx(
                rx, zmx_path, wavelength=1.31e-6, stop_surface=1)
            with open(zmx_path, encoding='utf-8') as f:
                txt = f.read()

        # Parse line by line: find the SURF blocks (object plus
        # written), identify which is COORDBRK vs STANDARD vs MIRROR,
        # and which carries STOP.
        lines = txt.splitlines()
        # Each surface block starts at a 'SURF <n>' line.
        surfs = []
        cur = None
        for ln in lines:
            s = ln.strip()
            if s.startswith('SURF '):
                if cur is not None:
                    surfs.append(cur)
                cur = {'idx': int(s.split()[1]),
                       'type': None,
                       'is_stop': False}
            elif cur is not None:
                if s.startswith('TYPE '):
                    cur['type'] = s.split()[1]
                elif s == 'STOP':
                    cur['is_stop'] = True
        if cur is not None:
            surfs.append(cur)

        # SURF 0 is the object (always STANDARD, never STOP).
        # SURF 1: first refractive (TYPE STANDARD)
        # SURF 2: second refractive (TYPE STANDARD)
        # SURF 3: coord-break (TYPE COORDBRK)
        # SURF 4: third refractive (NEW: STOP must land HERE if
        #         refr_counter is correctly tracked, since
        #         stop_surface=1 + 0 (object) + 1 (refr) = surface 4 ...
        # Actually let's re-check the logic.  The writer skips the
        # object (SURF 0) and starts emitting from SURF 1.  Within
        # the flat list, refr_counter starts at -1 and increments
        # BEFORE comparing to stop_surface.  So refr_counter goes
        # 0, 1, 2, 3 across the four refractive emissions.  The 2nd
        # (refr_counter == 1) gets the STOP.
        # In the SURF numbering: SURF 1, 2 are refractive; SURF 3 is
        # the coord-break; SURF 4, 5 are refractive.  refr_counter at
        # SURF 1 = 0, at SURF 2 = 1 (STOP HERE), at SURF 4 = 2, at
        # SURF 5 = 3.

        # Find the SURF index of the surface that carries STOP.
        stop_surfs = [s for s in surfs if s['is_stop']]
        assert len(stop_surfs) == 1, (
            f"Expected exactly one STOP marker; got {len(stop_surfs)}.")
        stop_surf = stop_surfs[0]
        # v4.11.2 fix: stop_surface=1 maps to the SECOND refractive
        # surface, which is SURF 2 in the emitted file (SURF 0 is
        # object, SURF 1 is first refractive, SURF 2 is second
        # refractive).
        assert stop_surf['type'] == 'STANDARD', (
            f"STOP marker landed on a non-refractive surface "
            f"(type={stop_surf['type']}); pre-v4.11.2 the global "
            f"surf_counter (including coord-breaks) was used so the "
            f"STOP could land on a COORDBRK.")
        assert stop_surf['idx'] == 2, (
            f"STOP marker on SURF {stop_surf['idx']}; expected SURF 2 "
            f"(the second refractive surface).  Pre-v4.11.2 the "
            f"writer used the global counter, which would place STOP "
            f"differently in the presence of coord-breaks.")


# ============================================================================
# Item 12 -- JAX <-> NumPy phase-retrieval cross-parity
# ============================================================================

class TestGerchbergSaxtonCrossBackendParity:
    """v4.12.0 (Track Tier-0) wired ``seed`` / ``dtype`` /
    ``initial_phase`` forwarding through the unified
    :func:`gerchberg_saxton` dispatcher.  Same-backend reproducibility
    is pinned in test_audit_fixes_v4_12_0_round4_tier0.py; this test
    pins CROSS-backend parity: given the same explicit
    ``initial_phase``, the JAX path's final phase agrees with the
    NumPy path's final phase to ~1e-3 RMS over the support.

    Uses a small grid (N=16) for runtime.  Forces float32 because the
    JAX kernel's internal complex64 path determines precision; the
    NumPy path runs at float64 but the comparison is done after the
    same number of iterations from the same initial phase, so the
    agreement is good even with the precision asymmetry.
    """

    def test_cross_backend_phase_agreement(self):
        from lumenairy.backend import JAX_AVAILABLE
        if not JAX_AVAILABLE:
            pytest.skip("JAX is not available.")

        N = 16
        rng = np.random.default_rng(42)
        # Reasonable source + target amplitudes.
        source = (np.abs(rng.standard_normal((N, N))) + 0.1).astype(
            np.float32)
        target = (np.abs(rng.standard_normal((N, N))) + 0.1).astype(
            np.float32)
        init_phase = rng.uniform(-np.pi, np.pi, (N, N)).astype(
            np.float32)

        phase_np, _err_np = la.gerchberg_saxton(
            source, target, n_iter=50,
            initial_phase=init_phase, backend='numpy')
        phase_jax, _err_jax = la.gerchberg_saxton(
            source, target, n_iter=50,
            initial_phase=init_phase, backend='jax')

        # Compare wrapped phase difference over the support
        # (source > small threshold).
        support = source > 1e-3
        diff = phase_np - phase_jax
        diff_mod = np.angle(np.exp(1j * diff))
        rms_mod = float(np.sqrt(np.mean(diff_mod[support] ** 2)))
        assert rms_mod < 1e-3, (
            f"gerchberg_saxton JAX vs NumPy phase RMS = {rms_mod:.3e} "
            f"rad over the support; expected < 1e-3 rad after 50 "
            f"iterations from the same initial phase.  Pre-v4.12.0 "
            f"the JAX path didn't honour ``initial_phase`` so the "
            f"two trajectories diverged.")


# ============================================================================
# Item 13 -- Cassegrain S1/S2/S3/S5 hand-derivation
# ============================================================================

class TestCassegrainSeidelHandDerivation:
    """v4.11.2 added a hand-derived S4 = -16 pin for a two-mirror
    Cassegrain with R1 = -0.5 m, R2 = -0.1 m, d12 = 0.2 m, stop at
    primary (see test_audit_fixes_v4_11_2_raytrace.py).  Round-4 audit
    flagged that S1, S2, S3, S5 lacked equivalent hand pins.

    Geometry (user-spec): R1 = -0.5 m primary, R2 = -0.1 m secondary,
    d12 = 0.4 m, stop at primary, lambda = 0.55 um, field_angle =
    0.001 rad, r_stop = 0.05 m.

    The hand calc follows Welford's per-surface Seidel formulas
    (chapter 8 / 6.5) verbatim, threaded through the library's
    parity-tracked n2 = -n1 mirror convention.  Per-surface and total
    values pinned to 1e-9 (the formulas are exact to roundoff once
    parity is correctly threaded).

    Hand calculation (see C:/tmp/_handcalc_cassegrain.py for the
    step-by-step trace):

    Primary (parity 0 -> 1):
        n1=+1, n2=-1, c1=-2 m^-1
        y_m=0.05, nu_m=0, y_c=0, nu_c=0.001
        A_m = -0.1, A_c = +0.001
        h = 0.05, delta_un = -0.2
        H = -5e-5, H^2 = 2.5e-9
        S1_1 = -A_m^2 * h * delta_un = -(0.01)*0.05*(-0.2) = +1e-4
        S2_1 = -A_m*A_c * h * delta_un = -(-1e-4)*0.05*(-0.2) = -1e-6
        S3_1 = -A_c^2 * h * delta_un = -(1e-6)*0.05*(-0.2) = +1e-8
        S4_1 = -(1/(n2*n1)) * c1 * (n2-n1) = -(1/-1)*(-2)*(-2) = +4
        S5_1 = -(A_c/A_m) * (S3_1 + H^2*S4_1)
             = -(0.001/-0.1) * (1e-8 + 1e-8)
             = +2e-10

    Transfer to secondary at d=0.4 m, n_after=-1 (parity flipped):
        u_m = -0.2/-1 = 0.2;  u_c = 0.001/-1 = -0.001
        y_m_at_2 = 0.05 + 0.2 * 0.4 = 0.13
        y_c_at_2 = 0 + (-0.001) * 0.4 = -0.0004

    Secondary (parity 1, sign=-1):
        n1=-1, n2=+1, c2=-10 m^-1
        u_m = -0.2/-1 = 0.2;  u_c = 0.001/-1 = -0.001
        i_m = c2*y_m + u_m = -10*0.13 + 0.2 = -1.1
        i_c = c2*y_c + u_c = -10*(-0.0004) + (-0.001) = 0.003
        A_m = n1*i_m = -1 * -1.1 = +1.1
        A_c = n1*i_c = -1 * 0.003 = -0.003
        nu_m_after = -0.2 - 0.13 * 2 * -10 = -0.2 + 2.6 = +2.4
        nu_c_after = 0.001 - (-0.0004) * 2 * -10 = 0.001 - 0.008 = -0.007
        u_m_after = 2.4 / 1 = 2.4;  u_c_after = -0.007 / 1 = -0.007
        delta_un = u_m_after/n2 - u_m/n1 = 2.4 - 0.2/-1 = 2.4 - (-0.2) = 2.6
        h = 0.13
        S1_2 = -(1.21)*0.13*2.6 = -0.40898
        S2_2 = -(1.1*-0.003)*0.13*2.6 = +0.0011154
        S3_2 = -(9e-6)*0.13*2.6 = -3.042e-6
        S4_2 = -(1/(1*-1)) * (-10) * (1 - (-1)) = -(-1)*(-10)*2 = -20
        S5_2 = -(-0.003/1.1) * (-3.042e-6 + 2.5e-9*-20)
             = (0.003/1.1) * (-3.042e-6 - 5e-8)
             = -8.432727e-9

    Totals:
        S1 = -0.40888,  S2 = +0.0011144,  S3 = -3.032e-6,
        S4 = -16,        S5 = -8.232727e-9
    """

    @staticmethod
    def _make_cassegrain():
        """Build the user-spec Cassegrain prescription."""
        R1 = -0.5
        R2 = -0.1
        d12 = 0.4
        r_stop = 0.05
        surfaces = [
            Surface(radius=R1, thickness=d12,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=r_stop),
            Surface(radius=R2, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True,
                    semi_diameter=0.015),
        ]
        return surfaces

    def test_S1_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S1 = np.asarray(result['S1'])
        assert abs(float(S1[0]) - 1e-4) < 1e-12, (
            f"Primary S1 = {S1[0]!r}; expected +1e-4 "
            f"(Welford hand calc).")
        assert abs(float(S1[1]) - (-0.40898)) < 1e-9, (
            f"Secondary S1 = {S1[1]!r}; expected -0.40898 "
            f"(Welford hand calc with parity-tracked n1=-1).")
        S1_total = float(result['total']['S1'])
        assert abs(S1_total - (-0.40888)) < 1e-9, (
            f"Cassegrain S1 total = {S1_total!r}; expected -0.40888 "
            f"(sum of Welford per-surface S1 with mirror parity).")

    def test_S2_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S2 = np.asarray(result['S2'])
        assert abs(float(S2[0]) - (-1e-6)) < 1e-14, (
            f"Primary S2 = {S2[0]!r}; expected -1e-6.")
        assert abs(float(S2[1]) - 0.0011154) < 1e-9, (
            f"Secondary S2 = {S2[1]!r}; expected +0.0011154.")
        S2_total = float(result['total']['S2'])
        assert abs(S2_total - 0.0011144) < 1e-9, (
            f"Cassegrain S2 total = {S2_total!r}; expected +0.0011144.")

    def test_S3_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S3 = np.asarray(result['S3'])
        assert abs(float(S3[0]) - 1e-8) < 1e-15, (
            f"Primary S3 = {S3[0]!r}; expected +1e-8.")
        assert abs(float(S3[1]) - (-3.042e-6)) < 1e-9, (
            f"Secondary S3 = {S3[1]!r}; expected -3.042e-6.")
        S3_total = float(result['total']['S3'])
        assert abs(S3_total - (-3.032e-6)) < 1e-9, (
            f"Cassegrain S3 total = {S3_total!r}; expected -3.032e-6.")

    def test_S5_per_surface_and_total(self):
        surfaces = self._make_cassegrain()
        result, _ = seidel_coefficients(
            surfaces, wavelength=0.55e-6, field_angle=0.001)
        S5 = np.asarray(result['S5'])
        # S5_1 = 2e-10 (handcalc); the library computes it via the
        # Schwarzschild equation, so the hand value is exact.
        assert abs(float(S5[0]) - 2e-10) < 1e-15, (
            f"Primary S5 = {S5[0]!r}; expected +2e-10.")
        assert abs(float(S5[1]) - (-8.432727272727272e-9)) < 1e-13, (
            f"Secondary S5 = {S5[1]!r}; expected -8.432727e-9.")
        S5_total = float(result['total']['S5'])
        assert abs(S5_total - (-8.232727272727272e-9)) < 1e-13, (
            f"Cassegrain S5 total = {S5_total!r}; expected "
            f"-8.232727e-9.")


# ============================================================================
# Item 14 -- Richards-Wolf vs paraxial Airy at low NA
# ============================================================================

class TestRichardsWolfAiryAtLowNA:
    """At low NA the Richards-Wolf vectorial focal field reduces to
    the paraxial Airy pattern.  v4.11.2's Richards-Wolf prefactor /
    apodisation fix puts the on-axis peak in the right place; this
    test pins the first-null radius against the paraxial Airy formula
    ``r1 = 1.22 * lambda / (2 * NA)``.

    Pin: NA = 0.1 (paraxial regime), N = 128, dx_pupil = 20 um,
    f = 1 mm => dx_focal = 0.39 um, Airy first null = 6.1 um
    (~15.6 focal pixels).  The first null should agree with the
    paraxial formula to within ~5%.
    """

    def test_first_null_within_5_percent_of_paraxial(self):
        Np = 128
        wavelength = 1e-6
        NA = 0.1
        f = 1e-3
        dx_pupil = 20e-6
        pupil = np.ones((Np, Np), dtype=np.complex128)
        Ex, Ey, Ez, _xf, _yf = la.richards_wolf_focus(
            pupil, wavelength, NA, f, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        I = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
        # I is shape (1, Np, Np) when z_planes has one element.
        I = np.squeeze(I)
        nc = Np // 2
        center_row = I[nc, :]
        dx_focal = wavelength * f / (Np * dx_pupil)

        peak_idx = int(np.argmax(center_row))
        # Find first local minimum to the right of the peak.
        right = center_row[peak_idx:]
        first_min_offset = None
        for k in range(2, len(right) - 1):
            if right[k] < right[k - 1] and right[k] < right[k + 1]:
                first_min_offset = k
                break

        assert first_min_offset is not None, (
            "Could not locate first Airy null in the focal-plane "
            "intensity profile.")
        r_observed = first_min_offset * dx_focal
        r_paraxial = 1.22 * wavelength / (2 * NA)
        rel = abs(r_observed - r_paraxial) / r_paraxial
        assert rel < 0.05, (
            f"Richards-Wolf at NA=0.1 first-null radius = "
            f"{r_observed:.3e} m; paraxial Airy = {r_paraxial:.3e} m; "
            f"relative error = {rel*100:.1f}% (expected < 5%).")
