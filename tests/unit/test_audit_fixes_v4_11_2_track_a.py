"""Regression tests for the 4.11.2 Track-A audit-residual patch.

Three findings pinned here, each a v4.10/v4.11.1 fix that turned out to
be wrong on closer inspection:

* **C-LR-1 reversal** -- ``apply_real_lens(seidel_correction=True)`` had
  its sign flipped by the v4.10 audit-response wave based on an
  incorrect physics-reasoning step in the round-1 audit.  The pre-v4.10
  negation was correct; the v4.10 patch produced a correction that
  approximately *tripled* the lens's analytic OPD at the rim
  (millimetre-scale rather than tens-of-nm residual).  Round-3 audit
  caught this; v4.11.2 restores the original sign.  Pin against ground
  truth from ``apply_real_lens_traced`` (which ray-traces every pixel
  and does not need this correction at all).

* **GBD axial_opl dead-on-arrival** -- v4.11.1 added a
  ``surfaces_from_prescription`` loop calling ``.get('thickness', ...)``
  on each element, but the function returns ``List[Surface]`` and
  ``Surface`` is a ``@dataclass`` with no ``.get`` method.  Every call
  raised ``AttributeError``, swallowed by a bare ``except Exception``,
  so ``axial_opl`` was always ``None``.  v4.11.2 switches to
  ``getattr`` and emits a ``RuntimeWarning`` on any other failure.

* **S-LAH64 / S-LAH79 Sellmeier coefficients wrong** -- the in-code
  coefficients gave ``n_d = 1.846`` (LAH64) and ``1.885`` (LAH79) vs
  the Ohara catalog values of ``1.78800`` and ``2.00330`` -- off by
  +5.8% and -5.9% respectively.  Appears to be misattributed
  coefficients.  v4.11.2 removes the in-code entries and routes both
  glasses through the ``__sellmeier__`` sentinel
  (``refractiveindex.info`` lookup, requires ``pip install
  refractiveindex``).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import lumenairy as lm


# ============================================================================
# C-LR-1 sign reversal -- correction matches ground truth (traced path)
# ============================================================================

class TestSeidelCorrectionSignAgainstGroundTruth:
    """``apply_real_lens(seidel_correction=True)`` adds a small (tens-
    of-nm) residual correction on top of the analytic phase screen.
    ``apply_real_lens_traced`` ray-traces every pixel and inherently
    contains the same correction.  Both should agree to within tens of
    nm RMS at the exit pupil for a paraxial singlet.

    Pre-v4.11.2 (v4.10 / v4.11.0 / v4.11.1) ``correction`` had the
    wrong sign and approximately tripled the analytic OPD, producing a
    field whose phase disagreed with ``apply_real_lens_traced`` by mm-
    scale OPD -- ~ 10^4 waves at visible wavelengths.

    The assertion is intentionally loose (``50 lambda`` RMS at the
    exit pupil) because:
      - ``apply_real_lens_traced`` includes the full per-pixel OPL
        whereas the analytic-screen path approximates each interface
        as a thin element, so even with the correct sign there's an
        irreducible ASM / interface-slant residual at the % level.
      - The point of the test is to lock in the SIGN, not to pin a
        precise numerical value.  Pre-v4.11.2 the disagreement was
        ~10^4 waves; post-fix it should be << 100.  50 lambda
        comfortably distinguishes a sign error from a small physics
        residual.
    """

    def test_seidel_correction_field_matches_traced_within_few_waves(self):
        # 100 mm-EFL plano-convex BK7 singlet -- a textbook case where
        # the analytic thin-element model should be accurate to a
        # fraction of a wave on-axis and the seidel-correction
        # contribution is genuinely small.
        wavelength = 0.5876e-6  # d-line
        N = 96
        dx = 60e-6  # 5.76 mm half-width, well outside the lens stop
        aperture = 5e-3  # 5 mm-diameter clear aperture
        prescription = lm.make_singlet(
            R1=51.5e-3, R2=float('inf'),
            d=2e-3, glass='N-BK7', aperture=aperture)

        # Flat field at the entrance pupil; both paths should converge
        # to the same exit-pupil phase (modulo a constant piston).
        E_in = np.ones((N, N), dtype=np.complex128)

        E_corr = lm.apply_real_lens(
            E_in, prescription=prescription, wavelength=wavelength,
            dx=dx, seidel_correction=True)

        # Ground truth: ray-traced per-pixel OPD.  Returns the wave at
        # the exit-pupil plane.  ``ray_subsample=1`` is critical here:
        # the default subsample factor wants a much larger grid to stay
        # above the alias gate, but we don't need a high-fidelity PSF
        # -- just an exit-pupil phase to compare against.
        E_traced = lm.apply_real_lens_traced(
            E_in, prescription=prescription, wavelength=wavelength,
            dx=dx, ray_subsample=1)

        # Compare phase inside the lens aperture only -- outside is
        # zeroed by both paths so any "disagreement" there is trivial.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        ap = (X * X + Y * Y) <= (aperture / 2.0) ** 2

        # Remove the piston (mean phase difference) before comparing.
        if not np.any(ap & (np.abs(E_corr) > 0) & (np.abs(E_traced) > 0)):
            pytest.skip(
                "no overlapping non-zero pixels in the aperture; "
                "grid / aperture geometry chosen too tightly.")

        # Element-wise phase difference inside the aperture.
        phase_diff = np.angle(E_corr / np.where(
            np.abs(E_traced) > 1e-30, E_traced, 1e-30))
        # Unwrap-ish: piston-subtract using the median to avoid 2-pi
        # wraparound dominating the RMS.
        phase_diff = phase_diff - np.median(phase_diff[ap])
        # Snap phase wraps back to [-pi, pi]:
        phase_diff = np.angle(np.exp(1j * phase_diff))

        rms_waves = float(np.sqrt(np.mean(phase_diff[ap] ** 2))) / (2 * np.pi)
        # Pre-v4.11.2: rms_waves ~ thousands.  Post-fix: should be O(1)
        # for a paraxial singlet.  Use 50-wave gate to lock in the
        # sign without being fragile to small physics-residual changes.
        assert rms_waves < 50.0, (
            f"apply_real_lens(seidel_correction=True) phase disagrees "
            f"with apply_real_lens_traced by {rms_waves:.1f} waves "
            f"RMS inside the aperture.  Pre-v4.11.2 the sign of "
            f"opl_wave_rel was flipped (v4.10 'C-LR-1 fix' was wrong); "
            f"the disagreement should be << 50 waves now.")


# ============================================================================
# GBD axial_opl -- dataclass access actually populates the value
# ============================================================================

class TestGbdAxialOplPopulated:
    """``propagate_gbd_through_prescription`` is supposed to compute
    ``axial_opl = sum_k n_k * t_k`` over the prescription's surfaces
    and pass it as a kwarg to ``apply_abcd_to_beamlets`` so the
    reconstructed field carries the system's absolute axial-phase
    reference.

    Pre-v4.11.2 (v4.11.1 work) the loop called ``_s.get('thickness',
    0.0)`` on each element of ``surfaces_from_prescription``, which
    returns ``List[Surface]`` -- a list of @dataclass instances with
    no ``.get`` method.  Every iteration raised ``AttributeError``,
    silently swallowed by a bare ``except Exception``, and
    ``axial_opl`` was always set to ``None``.

    Pin: the v4.11.2 path now emits a ``RuntimeWarning`` on any axial-
    OPL failure (the bare-except → warn conversion is part of the
    fix).  If the loop is broken in any future refactor, this warning
    will fire and the test will fail.
    """

    def test_axial_opl_path_does_not_emit_failure_warning(self):
        wavelength = 1.0e-6
        N = 32
        dx = 8e-6
        prescription = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=100e-6)
        # Plane wave on the source grid.
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ = lm.propagate_gbd_through_prescription(
                E_in, dx=dx, wavelength=wavelength,
                prescription=prescription)
        failure_warns = [
            w for w in caught
            if 'axial-OPL computation failed' in str(w.message)
        ]
        assert not failure_warns, (
            "propagate_gbd_through_prescription emitted "
            "'axial-OPL computation failed' RuntimeWarning -- the "
            "v4.11.2 fix to use attribute access on the Surface "
            "dataclass has regressed.  See AUDIT_ROUND3_2026_05_16.md "
            "CRIT-8.")

    def test_axial_opl_is_actually_non_zero(self):
        """v4.12.1 (audit round-4): the warning-absence test above can
        pass even if ``axial_opl`` silently fell through to None (the
        warn is suppressed inside the try/except but the actual OPL
        wasn't computed).  Pin a stronger condition by monkey-patching
        ``apply_abcd_to_beamlets`` and capturing the ``axial_opl``
        kwarg, then asserting the captured value is non-trivially
        non-zero (it should equal n_glass * t_glass for the N-BK7
        singlet, ~1.51 * 2e-3 = ~3e-3 m).
        """
        from lumenairy.propagators import gbd as _gbd_mod
        wavelength = 1.0e-6
        N = 32
        dx = 8e-6
        prescription = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=100e-6)
        E_in = np.ones((N, N), dtype=np.complex128)

        captured = {'axial_opl': '__unset__'}
        original = _gbd_mod.apply_abcd_to_beamlets

        def _spy(beamlets, A, B, C, D, *, wavelength, axial_opl=None,
                 **kw):
            captured['axial_opl'] = axial_opl
            return original(
                beamlets, A, B, C, D, wavelength=wavelength,
                axial_opl=axial_opl, **kw)

        _gbd_mod.apply_abcd_to_beamlets = _spy
        try:
            _ = lm.propagate_gbd_through_prescription(
                E_in, dx=dx, wavelength=wavelength,
                prescription=prescription)
        finally:
            _gbd_mod.apply_abcd_to_beamlets = original

        assert captured['axial_opl'] != '__unset__', (
            "apply_abcd_to_beamlets was not called -- monkey-patch "
            "spy missed the call site.")
        opl = captured['axial_opl']
        assert opl is not None, (
            "propagate_gbd_through_prescription passed axial_opl=None "
            "to apply_abcd_to_beamlets -- the v4.11.1 silent fallthrough "
            "regressed.  v4.11.2 expects a finite numeric value "
            "(sum n_k * t_k).")
        assert np.isfinite(float(opl)), (
            f"axial_opl was non-finite: {opl!r}.")
        # For the BK7 singlet (n_d ~ 1.51 at 1 um) of thickness 2 mm
        # the OPL is approximately 1.51 * 2e-3 = 3.02e-3 m.  Pin
        # ``axial_opl > 1e-3`` so a regression to silent-zero (or
        # n*t = 1*0 = 0) is caught loudly without being fragile to
        # the precise glass-index value.
        assert float(opl) > 1e-3, (
            f"axial_opl = {opl!r} m, expected > 1e-3 m for an N-BK7 "
            f"singlet of thickness 2 mm (~3e-3 m).  Pre-v4.11.2 the "
            f"bare-except in propagate_gbd_through_prescription "
            f"silently fell through to axial_opl=0 (or None which "
            f"the kernel treats as no piston).")


# ============================================================================
# S-LAH64 / S-LAH79 -- coefficients re-bundled in v4.15 (P1-GL-1)
# ============================================================================

class TestSLahGlassesRoutedViaSentinel:
    """v4.11.2 history: the original in-code Sellmeier coefficients for
    S-LAH64 and S-LAH79 were wrong by 5-6% in n_d (misattributed from
    a different glass).  v4.11.2 *removed* the in-code entries so that
    lookups would route through ``refractiveindex.info`` exclusively.

    v4.15 (P1-GL-1) re-bundles correct OHARA coefficients (sourced
    verbatim from the ``refractiveindex.info-database`` YAML for the
    OHARA Zemax 2017-11-30 catalog) so that minimal installs without
    the ``refractiveindex`` Python package can still resolve these
    glasses via the dispatcher's ``__sellmeier__`` fallback path.
    The new coefficients agree with refractiveindex.info to ~1e-10
    (S-LAH64) and ~4e-7 (S-LAH79) at n_d.

    This test class was originally written to pin the v4.11.2 absence;
    v4.15 inverts the contract: the entries are present AND accurate.
    """

    def test_s_lah64_in_table_with_correct_n_d(self):
        """v4.15 (P1-GL-1): S-LAH64 has bundled Sellmeier coefficients
        matching the Ohara catalog n_d=1.78800 within 5e-5."""
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'S-LAH64' in SELLMEIER_COEFFICIENTS, (
            "v4.15 (P1-GL-1): S-LAH64 should be bundled now to support "
            "minimal installs without the ``refractiveindex`` package."
        )
        n_d = _sellmeier_index(
            wavelength_m=0.58756e-6,
            coeffs=SELLMEIER_COEFFICIENTS['S-LAH64'],
            glass_name='S-LAH64')
        assert abs(n_d - 1.788001) < 5e-5, (
            f"S-LAH64 bundled Sellmeier n_d={n_d:.6f}; Ohara catalog "
            f"is 1.788001.  Coefficients may be misattributed -- the "
            f"v4.11.2 audit (CRIT-1) caught this exact problem with "
            f"the previous in-code values.")

    def test_s_lah79_in_table_with_correct_n_d(self):
        """v4.15 (P1-GL-1): S-LAH79 has bundled Sellmeier coefficients
        matching the Ohara catalog n_d=2.00330 within 5e-5."""
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'S-LAH79' in SELLMEIER_COEFFICIENTS, (
            "v4.15 (P1-GL-1): S-LAH79 should be bundled now to support "
            "minimal installs without the ``refractiveindex`` package."
        )
        n_d = _sellmeier_index(
            wavelength_m=0.58756e-6,
            coeffs=SELLMEIER_COEFFICIENTS['S-LAH79'],
            glass_name='S-LAH79')
        assert abs(n_d - 2.003300) < 5e-5, (
            f"S-LAH79 bundled Sellmeier n_d={n_d:.6f}; Ohara catalog "
            f"is 2.003300.")

    def test_in_code_sellmeier_n_d_within_1e3_of_catalog_for_a_known_good_glass(
            self):
        """Sanity check that the rest of the Sellmeier table is sane:
        a known-good in-code entry (N-BK7) produces n_d within 1e-3 of
        the well-established catalog value (n_d = 1.5168).

        Pin this so that if someone re-introduces miscalibrated
        coefficients for any other glass, this check at least flags
        N-BK7 as a canary.
        """
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'N-BK7' in SELLMEIER_COEFFICIENTS, (
            "N-BK7 missing from in-code Sellmeier table; this is the "
            "canary glass for the rest of the table.  Restore it.")
        n_d = _sellmeier_index(
            wavelength_m=0.5876e-6,
            coeffs=SELLMEIER_COEFFICIENTS['N-BK7'],
            glass_name='N-BK7')
        assert abs(n_d - 1.5168) < 1e-3, (
            f"N-BK7 in-code Sellmeier gave n_d = {n_d:.5f}; "
            f"catalog value is 1.5168.  If this fails another glass "
            f"may also be miscalibrated -- audit the entire table.")
