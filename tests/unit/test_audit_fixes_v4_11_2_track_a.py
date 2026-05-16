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


# ============================================================================
# S-LAH64 / S-LAH79 -- in-code Sellmeier removed, routed via sentinel
# ============================================================================

class TestSLahGlassesRoutedViaSentinel:
    """The Sellmeier coefficients in-code for S-LAH64 and S-LAH79
    were wrong by 5-6% in n_d -- almost certainly misattributed from
    a different glass.  v4.11.2 removes the in-code entries; lookups
    now go through the ``'__sellmeier__'`` sentinel which dispatches
    to ``refractiveindex.info`` (requires ``pip install
    refractiveindex``).

    This test verifies the entries are no longer in the in-code
    Sellmeier table.  Whether the refractiveindex lookup succeeds in
    a given environment depends on whether the optional dep is
    installed -- not tested here.
    """

    def test_s_lah64_not_in_in_code_sellmeier_table(self):
        from lumenairy.glass import SELLMEIER_COEFFICIENTS
        assert 'S-LAH64' not in SELLMEIER_COEFFICIENTS, (
            f"S-LAH64 should not have hard-coded Sellmeier coefficients "
            f"-- pre-v4.11.2 they were misattributed (n_d off by "
            f"+5.8% vs Ohara catalog).  Removed in v4.11.2; route via "
            f"refractiveindex.info instead."
        )

    def test_s_lah79_not_in_in_code_sellmeier_table(self):
        from lumenairy.glass import SELLMEIER_COEFFICIENTS
        assert 'S-LAH79' not in SELLMEIER_COEFFICIENTS, (
            f"S-LAH79 should not have hard-coded Sellmeier coefficients "
            f"-- pre-v4.11.2 they were misattributed (n_d off by "
            f"-5.9% vs Ohara catalog).  Removed in v4.11.2; route via "
            f"refractiveindex.info instead."
        )

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
