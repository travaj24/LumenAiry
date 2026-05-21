"""Consolidated audit-fix tests for the **glass** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 2 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_2_track_a.py``
* ``test_audit_fixes_v4_14_2_agent_a.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_2_track_a.py
# Audit version: V4_11_2  scope: track_a
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the 4.11.2 Track-A audit-residual patch.
#   
#   Three findings pinned here, each a v4.10/v4.11.1 fix that turned out to
#   be wrong on closer inspection:
#   
#   * **C-LR-1 reversal** -- ``apply_real_lens(seidel_correction=True)`` had
#     its sign flipped by the v4.10 audit-response wave based on an
#     incorrect physics-reasoning step in the round-1 audit.  The pre-v4.10
#     negation was correct; the v4.10 patch produced a correction that
#     approximately *tripled* the lens's analytic OPD at the rim
#     (millimetre-scale rather than tens-of-nm residual).  Round-3 audit
#     caught this; v4.11.2 restores the original sign.  Pin against ground
#     truth from ``apply_real_lens_traced`` (which ray-traces every pixel
#     and does not need this correction at all).
#   
#   * **GBD axial_opl dead-on-arrival** -- v4.11.1 added a
#     ``surfaces_from_prescription`` loop calling ``.get('thickness', ...)``
#     on each element, but the function returns ``List[Surface]`` and
#     ``Surface`` is a ``@dataclass`` with no ``.get`` method.  Every call
#     raised ``AttributeError``, swallowed by a bare ``except Exception``,
#     so ``axial_opl`` was always ``None``.  v4.11.2 switches to
#     ``getattr`` and emits a ``RuntimeWarning`` on any other failure.
#   
#   * **S-LAH64 / S-LAH79 Sellmeier coefficients wrong** -- the in-code
#     coefficients gave ``n_d = 1.846`` (LAH64) and ``1.885`` (LAH79) vs
#     the Ohara catalog values of ``1.78800`` and ``2.00330`` -- off by
#     +5.8% and -5.9% respectively.  Appears to be misattributed
#     coefficients.  v4.11.2 removes the in-code entries and routes both
#     glasses through the ``__sellmeier__`` sentinel
#     (``refractiveindex.info`` lookup, requires ``pip install
#     refractiveindex``).
# ============================================================================
import math
import warnings

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# C-LR-1 sign reversal -- correction matches ground truth (traced path)
# ============================================================================

class TestAuditFixesV4_11_2_track_a_SeidelCorrectionSignAgainstGroundTruth:
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

class TestAuditFixesV4_11_2_track_a_GbdAxialOplPopulated:
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

class TestAuditFixesV4_11_2_track_a_SLahGlassesRoutedViaSentinel:
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


# ============================================================================
# Source: test_audit_fixes_v4_14_2_agent_a.py
# Audit version: V4_14_2  scope: agent_a
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.2 audit (Agent A scope --
#   ``lumenairy.glass``, ``lumenairy.elements.freeform``,
#   ``lumenairy.elements.polarization``).
#   
#   Four audit items are pinned:
#   
#   * **P0-NEW-1** -- ``GLASS_REGISTRY['S-LAH64']`` /
#     ``GLASS_REGISTRY['S-LAH79']`` were stranded with the
#     ``'__sellmeier__'`` sentinel after v4.11.2 removed their Sellmeier
#     rows from :data:`SELLMEIER_COEFFICIENTS`.  Every lookup for these
#     two glasses raised ``ValueError`` from the dispatcher's
#     consistency-check branch -- the v4.11.2 fix forgot to re-route them
#     to a refractiveindex.info tuple.  v4.14.2 re-routes to
#     ``('specs', 'OHARA-optical', 'S-LAH64')`` /
#     ``('specs', 'OHARA-optical', 'S-LAH79')``, restoring the Ohara
#     catalogue n_d values (1.788 / 2.003).  A new module-load
#     consistency check (``_check_glass_registry_consistency``) converts
#     the same class-of-bug into a fail-fast at import time so a future
#     drift cannot re-surface as a silent ``ValueError`` at first call.
#   
#   * **P1-NEW-5** -- ``surface_sag_xy_polynomial`` evaluated
#     ``c * X**i * Y**j`` for every pixel of the input grid, with no
#     out-of-domain guard.  A high-order coefficient on a large grid
#     produced a discontinuous step at the aperture rim (e.g.
#     ``(2, 0): 1e3`` on a 50 mm half-grid produces 2.5 m of corner sag
#     applied to pixels outside the physical aperture).  v4.14.2 adds
#     ``norm_x``/``norm_y`` kwargs (default 1.0 = unit box) matching the
#     Chebyshev branch's ``np.where(outside, 0.0, departure)`` pattern.
#   
#   * **P1-NEW-7** -- ``apply_rotator`` was the only angle-taking
#     polarization helper without an ``angle_deg=`` kwarg.  Idiomatic
#     ``apply_rotator(field, angle_deg=45)`` calls raised TypeError.
#     v4.14.2 closes the v4.7 sibling-gap; passing both with
#     conflicting values raises ``ValueError`` (consistent-value pairs
#     are accepted).
#   
#   * **P1-NEW-8** -- ``JonesField.__init__`` validated
#     ``Ex.shape == Ey.shape`` but not ``Ex.ndim == 2``, ``dx > 0``,
#     ``dy > 0``.  Invalid inputs propagated all the way to the FFT in
#     :meth:`propagate` where an opaque shape / value error was raised
#     far from the construction site.  v4.14.2 validates at construction.
#   
#   Author:  Agent A -- v4.14.2.
# ============================================================================

import importlib
import inspect

import numpy as np
import pytest

import lumenairy as lm
from lumenairy import glass as _glass
from lumenairy.elements.freeform import surface_sag_xy_polynomial
from lumenairy.elements.polarization import (
    JonesField,
    apply_rotator,
    create_linear_polarized,
)

# ===========================================================================
# A.1 (P0-NEW-1) -- S-LAH64 / S-LAH79 dispatch
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_SLahDispatch:
    """Pin that ``GLASS_REGISTRY['S-LAH64']`` /
    ``GLASS_REGISTRY['S-LAH79']`` resolve to a real refractive index.

    The pre-v4.14.2 bug:  both glasses were flagged
    ``'__sellmeier__'`` but their Sellmeier rows were removed in
    v4.11.2, so the dispatcher's consistency-check raised
    ``ValueError`` on every lookup.
    """

    def test_s_lah64_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH64', d-line)`` returns a finite
        refractive index in the [1.5, 2.0] range characteristic of
        Ohara lanthanum-flint glasses.  Pre-v4.14.2 raised
        ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH64', 587.6e-9)
        assert np.isfinite(n), 'S-LAH64 must return a finite n'
        assert 1.5 < n < 2.0, (
            f'S-LAH64 d-line n = {n:.4f} outside the expected '
            f'lanthanum-flint range [1.5, 2.0]; Ohara catalogue '
            f'value is 1.788.')

    def test_s_lah79_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH79', d-line)`` returns a finite
        refractive index near the Ohara catalogue value 2.003.
        Pre-v4.14.2 raised ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH79', 587.6e-9)
        assert np.isfinite(n), 'S-LAH79 must return a finite n'
        # S-LAH79 n_d = 2.00330 per Ohara catalogue
        assert 1.5 < n < 2.1, (
            f'S-LAH79 d-line n = {n:.4f} outside the expected '
            f'high-index lanthanum-flint range [1.5, 2.1]; Ohara '
            f'catalogue value is 2.003.')

    def test_glass_registry_consistency_check_present(self):
        """The new module-load consistency check function exists
        and contains the drift-detection logic.  Its source must
        reference both ``GLASS_REGISTRY`` and
        ``SELLMEIER_COEFFICIENTS`` and contain a ``RuntimeError``
        path -- this is the structural counter-measure that prevents
        the same class-of-bug from re-surfacing silently.
        """
        assert hasattr(_glass, '_check_glass_registry_consistency'), (
            'v4.14.2 module-load consistency check '
            '`_check_glass_registry_consistency` is missing from '
            'lumenairy.glass.')
        # TODO(v5.2.1): replace with behavioral pin -- inspect.getsource proxy-test pattern (per AUDIT_V4_13_1 Part 6.1)
        src = inspect.getsource(_glass._check_glass_registry_consistency)
        assert 'GLASS_REGISTRY' in src, (
            'consistency check must iterate GLASS_REGISTRY')
        assert 'SELLMEIER_COEFFICIENTS' in src, (
            'consistency check must verify membership in '
            'SELLMEIER_COEFFICIENTS')
        assert 'RuntimeError' in src, (
            'consistency check must fail-fast with RuntimeError on '
            'detected drift')
        assert "'__sellmeier__'" in src or '"__sellmeier__"' in src, (
            'consistency check must filter on the __sellmeier__ '
            'sentinel')

    def test_glass_registry_consistency_check_rejects_drift(self):
        """Inject a synthetic ``'__test_sentinel_v4_14_2__'`` entry
        flagged ``'__sellmeier__'`` but absent from
        ``SELLMEIER_COEFFICIENTS``, then re-invoke the check.  Must
        raise ``RuntimeError`` naming the drift.  Restore state on
        exit so subsequent tests are unaffected.
        """
        sentinel_name = '__test_sentinel_v4_14_2__'
        assert sentinel_name not in _glass.GLASS_REGISTRY
        assert sentinel_name not in _glass.SELLMEIER_COEFFICIENTS
        _glass.GLASS_REGISTRY[sentinel_name] = '__sellmeier__'
        try:
            with pytest.raises(RuntimeError, match='GLASS_REGISTRY drift'):
                _glass._check_glass_registry_consistency()
        finally:
            del _glass.GLASS_REGISTRY[sentinel_name]
        # Re-run the real check to confirm no real-entry drift was
        # introduced.
        _glass._check_glass_registry_consistency()


# ===========================================================================
# A.2 (P1-NEW-5) -- surface_sag_xy_polynomial out-of-domain guard
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_XyPolynomialDomainGuard:
    """Pin that ``surface_sag_xy_polynomial`` zeros the polynomial
    departure outside the ``(norm_x, norm_y)`` rectangular box.

    The pre-v4.14.2 bug:  no guard, so a high-order coefficient on a
    large grid produced a discontinuous step at the aperture rim
    where the polynomial diverged but the raytracer saw no
    aperture-aware clip.
    """

    def test_xy_polynomial_zero_outside_unit_box_nonzero_inside(self):
        """Build a (2, 0): 1.0 (pure X^2) XY-polynomial surface,
        evaluate on a 64x64 grid spanning [-2, +2] (twice the
        unit-box half-extent in each axis).  The freeform departure
        must be zero outside the unit box and equal to X^2 inside.
        """
        N = 64
        half = 2.0  # twice the default unit-box half-extent
        x = np.linspace(-half, half, N)
        y = np.linspace(-half, half, N)
        X, Y = np.meshgrid(x, y, indexing='xy')
        # Pure X^2 freeform with flat base (R = inf, no conic).
        sag = surface_sag_xy_polynomial(
            X, Y, R=np.inf, conic=0.0,
            xy_coeffs={(2, 0): 1.0},
            norm_x=1.0, norm_y=1.0)
        # Outside the unit box: sag must be 0 (flat base + zeroed
        # freeform).
        outside = (np.abs(X) > 1.0) | (np.abs(Y) > 1.0)
        np.testing.assert_allclose(
            sag[outside], 0.0, atol=1e-15,
            err_msg='sag must be zero outside the unit box')
        # Inside the unit box: sag must equal X^2 (the pure
        # polynomial term, no base sag).
        inside = ~outside
        np.testing.assert_allclose(
            sag[inside], (X ** 2)[inside], atol=1e-15,
            err_msg='sag must equal X^2 inside the unit box')
        # Sanity: must have BOTH inside and outside pixels.
        assert outside.any(), 'test grid must extend beyond unit box'
        assert inside.any(), 'test grid must include unit-box pixels'


# ===========================================================================
# A.3 (P1-NEW-7) -- apply_rotator angle_deg kwarg
# ===========================================================================


def _x_polarized_field(N: int = 32, dx: float = 1e-6) -> JonesField:
    """A unit-amplitude x-polarized JonesField for rotator tests."""
    scalar = np.ones((N, N), dtype=complex)
    return create_linear_polarized(scalar, dx, angle=0.0)


class TestAuditFixesV4_14_2_agent_a_ApplyRotatorAngleDeg:
    """Pin that ``apply_rotator`` accepts ``angle_deg=`` matching the
    v4.7 convention used by ``apply_polarizer``, ``apply_waveplate``,
    and the half/quarter-wave-plate convenience wrappers.
    """

    def test_apply_rotator_angle_deg_equivalent_to_angle_rad(self):
        """``apply_rotator(field, angle_deg=45)`` must produce
        bit-equal output to ``apply_rotator(field, angle=pi/4)``.
        """
        f_rad = _x_polarized_field()
        f_deg = _x_polarized_field()
        apply_rotator(f_rad, angle=np.pi / 4)
        apply_rotator(f_deg, angle_deg=45.0)
        np.testing.assert_allclose(
            f_rad.Ex, f_deg.Ex, atol=1e-15,
            err_msg='Ex from angle_deg=45 must match angle=pi/4')
        np.testing.assert_allclose(
            f_rad.Ey, f_deg.Ey, atol=1e-15,
            err_msg='Ey from angle_deg=45 must match angle=pi/4')

    def test_apply_rotator_consistent_angle_and_angle_deg_accepted(self):
        """Supplying both ``angle`` and ``angle_deg`` with consistent
        values must be accepted (silently use ``angle_deg``)."""
        f = _x_polarized_field()
        # pi/4 == radians(45) -- consistent
        apply_rotator(f, angle=np.pi / 4, angle_deg=45.0)
        # Should not raise; Ey should be sin(pi/4) = 1/sqrt(2)
        np.testing.assert_allclose(
            f.Ey[0, 0], 1.0 / np.sqrt(2), atol=1e-12)

    def test_apply_rotator_conflicting_angle_raises(self):
        """Supplying ``angle`` and ``angle_deg`` with disagreeing
        values must raise ``ValueError``."""
        f = _x_polarized_field()
        with pytest.raises(ValueError, match='conflicting'):
            apply_rotator(f, angle=np.pi / 3, angle_deg=45.0)


# ===========================================================================
# A.4 (P1-NEW-8) -- JonesField input validation
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_JonesFieldInputValidation:
    """Pin that ``JonesField.__init__`` validates ``dx > 0``,
    ``dy > 0``, and ``Ex.ndim == 2`` at construction time, rather
    than letting invalid inputs propagate to the FFT in
    :meth:`JonesField.propagate`.
    """

    def test_jones_field_rejects_zero_dx(self):
        """``dx = 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=0.0)

    def test_jones_field_rejects_negative_dx(self):
        """``dx < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=-1e-6)

    def test_jones_field_rejects_negative_dy(self):
        """``dy < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dy must be'):
            JonesField(Ex, Ey, dx=1e-6, dy=-1e-6)

    def test_jones_field_rejects_1d_input(self):
        """A 1-D ``Ex`` / ``Ey`` must raise ``ValueError`` at
        construction.  Pre-v4.14.2 this propagated to the FFT in
        ``propagate`` and raised an opaque error far from the
        construction site.
        """
        Ex = np.ones(64, dtype=complex)
        Ey = np.ones(64, dtype=complex)
        with pytest.raises(ValueError, match='2-D'):
            JonesField(Ex, Ey, dx=1e-6)

    def test_jones_field_valid_inputs_construct_cleanly(self):
        """Valid 2-D inputs with positive ``dx``, ``dy`` construct
        without error and preserve the supplied pitch."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        f = JonesField(Ex, Ey, dx=1e-6, dy=2e-6)
        assert f.dx == 1e-6
        assert f.dy == 2e-6
        assert f.Ex.shape == (N, N)
        # Default dy -> dx
        f2 = JonesField(Ex, Ey, dx=1e-6)
        assert f2.dy == 1e-6
