"""Pinning tests for the v4.14.2 Agent-B audit fixes.

Audit reference
---------------

``AUDIT_V4_14_1_2026_05_17.md`` P1-NEW-1 / P1-NEW-4 (Tier 0 row 2 / row 5).
v4.14.1 introduced ``_ZeroApertureMaskSentinel`` to handle
``aperture_diameter=0`` correctly but the CHANGELOG only documented
"3 callers updated"; the audit found a **fourth** consumer of
``_get_wrapper_merit_cache`` -- ``ToleranceAwareMerit.evaluate`` -- and a
pre-existing semantically-identical bug in ``MatchIdealSystemMerit._make_source``.
Both produced a grid-filling plane wave when ``aperture_diameter`` was
explicitly zero, which then propagated through ``apply_real_lens`` as a
bright on-axis "source" and silently mis-scored the merit.

Agent B fixes
-------------

* **B.1 / P1-NEW-1 (ToleranceAware leg)** -- Add the canonical
  ``_cache['mask'] is _ZERO_APERTURE_MASK -> zero E_in`` branch to
  ``ToleranceAwareMerit.evaluate`` so the per-trial wave-leg source is
  zero when the (perturbed) prescription's ``aperture_diameter`` is
  explicitly zero or negative.  Mirrors the canonical branch at
  ``MultiWavelengthMerit.evaluate`` / ``MultiFieldMerit.evaluate``.

* **B.2 / P1-NEW-1 (MatchIdealSystem leg)** -- Pre-v4.14.2
  ``_make_source`` had an ``if ap is not None and np.isfinite(ap) and
  ap > 0:`` guard that fell through to "full grid plane wave" when
  ``aperture_diameter`` was zero.  v4.14.2 adds an explicit
  ``ap <= 0 -> zero field`` branch matching the
  ``_ZERO_APERTURE_MASK`` semantics.

* **B.2-residual / P1-NEW-4 (0+0j sweep)** -- The ``np.where(mask, E,
  0.0 + 0.0j)`` literal at the same site silently upcast complex64
  fields to complex128.  v4.14.2 replaces the literal with the
  dtype-aware-zero pattern ``np.where(mask, E, np.zeros((), dtype=cdtype))``
  matching the v4.13.2 sweep across apply_aperture / apply_mirror /
  _lens_thin / _lens_real.

* **B.3 (validation gap)** -- ``apply_perturbations`` does NOT call
  ``validate_prescription``, but since perturbations only modify
  per-surface ``decenter`` / ``tilt`` / ``form_error`` fields (not the
  prescription-level ``aperture_diameter``), the perturbed
  ``aperture_diameter`` is always identical to the nominal.  The B.1
  fix already handles the case where the nominal aperture is zero;
  this test pins the contract that ``apply_perturbations`` does not
  itself zero/negativise the aperture.

Author: Andrew Traverso -- v4.14.2 / Agent B
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.optimize import (
    EvaluationContext,
    MatchIdealSystemMerit,
    StrehlMerit,
    ToleranceAwareMerit,
)
from lumenairy.optimize.core import (
    _ZERO_APERTURE_MASK,
    _get_wrapper_merit_cache,
    _clear_wrapper_merit_cache,
)


# ============================================================================
# Helpers
# ============================================================================

def _zero_aperture_prescription():
    """Build a singlet prescription, then override aperture_diameter to 0.

    ``make_singlet`` requires a positive aperture argument (its argparse
    accepts any float but validate_prescription would reject 0); we
    bypass by constructing the dict normally and setting the field
    directly.  This is the only way to exercise the production code path
    where ``aperture_diameter <= 0`` reaches the merit-eval code.
    """
    rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                          glass='N-BK7', aperture=5e-3)
    rx['aperture_diameter'] = 0.0
    return rx


# ============================================================================
# B.1 -- ToleranceAwareMerit honours _ZERO_APERTURE_MASK sentinel
# ============================================================================

class TestB1ToleranceAwareApertureZero:
    """``ToleranceAwareMerit.evaluate`` produces a zero E_in (not a
    full-grid plane wave) when the prescription's aperture_diameter
    is explicitly zero.

    Pre-fix the cached ``E_ones`` array was always full-ones; the
    ``_ZERO_APERTURE_MASK`` sentinel was placed in ``_cache['mask']``
    but ``ToleranceAwareMerit`` did NOT check it, so a deliberate-zero
    aperture silently propagated a grid-filling plane wave through
    ``apply_real_lens``.
    """

    def test_tolerance_aware_aperture_zero_produces_zero_field(self):
        """Capture E_in by monkey-patching apply_real_lens and assert
        all zeros for an aperture_diameter=0 prescription.

        Pins **P1-NEW-1 (ToleranceAware leg)**.
        """
        rx = _zero_aperture_prescription()
        # Clear the module cache so this test sees a fresh build.
        _clear_wrapper_merit_cache()

        captured = {}

        from lumenairy.elements import lenses as lens_mod

        original = lens_mod.apply_real_lens

        def capturing_apply_real_lens(E_in, **kw):
            captured['E_in'] = np.asarray(E_in).copy()
            # Return zeros so downstream through-focus and Strehl
            # computations are well-defined.
            return np.zeros_like(E_in)

        try:
            # Patch the binding inside optimize.core, since that's the
            # ``from ..elements.lenses import apply_real_lens`` import
            # site ToleranceAwareMerit actually uses.
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                # A single-trial ToleranceAware over an empty spec
                # exercises the perturbed wave-leg path without
                # actually perturbing anything; the perturbed
                # prescription is a deep copy of the original (with
                # aperture_diameter = 0 preserved).
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[], n_trials=1, seed=0,
                    weight=1.0)
                # Suppress benign warnings from downstream through_focus
                # on a zero field.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        assert 'E_in' in captured, (
            'ToleranceAwareMerit.evaluate did not reach apply_real_lens')
        E_in = captured['E_in']
        assert E_in.shape == (32, 32)
        # The whole point: every grid point must be zero so the
        # downstream apply_real_lens sees "no light," matching the
        # physical meaning of aperture_diameter=0.
        assert np.all(E_in == 0), (
            f'ToleranceAware aperture=0 produced non-zero E_in '
            f'(max|E|={np.max(np.abs(E_in)):.3e}); '
            f'sentinel branch was not honoured.')

    def test_tolerance_aware_aperture_positive_unchanged(self):
        """Sanity check: ``aperture_diameter > 0`` produces the
        cached full-ones template (the v4.14.1 invariant).
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        _clear_wrapper_merit_cache()

        captured = {}

        def capturing_apply_real_lens(E_in, **kw):
            captured['E_in'] = np.asarray(E_in).copy()
            return np.zeros_like(E_in)

        try:
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[], n_trials=1, seed=0,
                    weight=1.0)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        E_in = captured['E_in']
        # Positive aperture -> cached template = np.ones.  Pin against
        # the contract documented in _get_wrapper_merit_cache.
        assert np.all(E_in == 1.0), (
            f'ToleranceAware aperture>0 did not produce the cached '
            f'np.ones template (mean|E|={np.mean(np.abs(E_in)):.3e}).')


# ============================================================================
# B.2 -- MatchIdealSystemMerit._make_source honours zero-aperture
# ============================================================================

class TestB2MatchIdealApertureZero:
    """``MatchIdealSystemMerit._make_source`` with an
    aperture_diameter=0 prescription produces a zero field.

    Pre-fix the ``if ap > 0`` guard fell through to the bare
    ``np.ones`` branch, producing a grid-filling plane wave.
    """

    def test_match_ideal_aperture_zero_produces_zero_field(self):
        """Pins **P1-NEW-1 (MatchIdealSystem leg)**.

        Builds a MatchIdealSystemMerit instance and calls
        ``_make_source`` directly with a zero-aperture context.
        """
        rx = _zero_aperture_prescription()
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)

        # Minimal ideal-system spec; the merit only needs _make_source
        # to be callable, not the full evaluate path.
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025},
                            {'type': 'propagate', 'z': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        assert E.shape == (32, 32)
        assert np.all(E == 0), (
            f'MatchIdealSystem aperture=0 produced non-zero source '
            f'field (max|E|={np.max(np.abs(E)):.3e}); '
            f'the ap <= 0 branch was not taken.')

    def test_match_ideal_aperture_positive_produces_circular_mask(self):
        """Sanity check: ``aperture > 0`` produces the standard
        circular boolean mask, unchanged by the B.2 fix.
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=120e-6)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        # Some pixels lit, some zero -- the circular mask.
        nonzero = np.count_nonzero(np.abs(E))
        assert 0 < nonzero < 32 * 32, (
            f'MatchIdealSystem aperture>0 did not produce a partial '
            f'mask (nonzero={nonzero}/{32 * 32}).')

    def test_match_ideal_aperture_none_produces_full_grid(self):
        """``aperture_diameter`` missing -> full-grid plane wave
        (unchanged pre-existing behaviour)."""
        # Construct a singlet, then strip aperture_diameter.
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        rx.pop('aperture_diameter', None)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)
        E = merit._make_source(ctx, wavelength=1.31e-6,
                                field_angle=(0.0, 0.0))
        # No aperture specified -> full-grid plane wave.
        assert np.all(np.abs(E) == 1.0), (
            'MatchIdealSystem with no aperture should produce a '
            'full-grid plane wave.')


# ============================================================================
# B.2-residual -- complex64 dtype preserved through _make_source
# ============================================================================

class TestB2MatchIdealComplex64DtypePreserved:
    """The v4.14.2 dtype-aware-zero fix at ``optimize/core.py:966`` --
    formerly ``np.where(mask, E, 0.0 + 0.0j)`` -- must not upcast a
    complex64 cdtype back to complex128.

    Pins **P1-NEW-4** at this specific site.
    """

    def test_match_ideal_complex64_dtype_preserved(self):
        """Drive ``get_default_complex_dtype`` to complex64 via the
        precision knob and assert ``_make_source`` returns complex64.
        """
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=120e-6)
        ctx = EvaluationContext(
            prescription=rx, wavelength=1.31e-6,
            N=32, dx=10e-6, efl=0.025, bfl=0.025)
        merit = MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.025}],
            weight=1.0)

        # Flip the runtime precision knob to single.
        from lumenairy.propagators.propagation import (
            get_default_complex_dtype, set_default_complex_dtype,
        )
        saved = get_default_complex_dtype()
        try:
            set_default_complex_dtype(np.complex64)
            E = merit._make_source(ctx, wavelength=1.31e-6,
                                    field_angle=(0.0, 0.0))
            assert E.dtype == np.complex64, (
                f'MatchIdealSystem._make_source upcast complex64 -> '
                f'{E.dtype} (P1-NEW-4 regression); the 0+0j literal '
                f'must be np.zeros((), dtype=cdtype).')
        finally:
            set_default_complex_dtype(saved)


# ============================================================================
# B.3 -- apply_perturbations does not silently zero/negativise aperture
# ============================================================================

class TestB3PerturbedApertureInvariant:
    """``apply_perturbations`` modifies per-surface ``decenter`` /
    ``tilt`` / ``form_error`` fields but DOES NOT touch the
    prescription-level ``aperture_diameter``.  This pin guards against
    a future regression where perturbations start mutating the aperture
    (which would then escape ``validate_prescription`` since
    ``apply_perturbations`` does not invoke it).
    """

    def test_apply_perturbations_does_not_modify_aperture(self):
        """Apply a typical Monte-Carlo perturbation set and assert the
        perturbed prescription's ``aperture_diameter`` is identical to
        the nominal.  Pins the contract that B.1's sentinel guard
        already covers the perturbed wave leg.
        """
        from lumenairy.analysis.through_focus import (
            apply_perturbations, Perturbation,
        )
        rx = la.make_singlet(R1=25e-3, R2=-25e-3, d=2.5e-3,
                              glass='N-BK7', aperture=5e-3)
        nominal_ap = rx['aperture_diameter']

        perts = [Perturbation(
            surface_index=0,
            decenter=(10e-6, 5e-6),
            tilt=(1e-4, -1e-4),
            form_error_rms=20e-9,
            random_seed=12345,
            name='unit_test_pert')]
        perturbed = apply_perturbations(rx, perts, N=32, dx=10e-6)
        assert perturbed['aperture_diameter'] == nominal_ap, (
            'apply_perturbations should not modify the prescription-'
            'level aperture_diameter; if this fails, the B.1 sentinel '
            'guard is insufficient -- additional validate_prescription '
            'guard required at ToleranceAwareMerit.evaluate entry.')

    def test_tolerance_aware_with_perturbation_aperture_zero(self):
        """Pin the end-to-end contract: a *perturbed* trial whose
        nominal aperture is zero produces a zero E_in (and thus a
        well-defined merit, not a garbage plane-wave score).
        """
        from lumenairy.analysis.through_focus import Perturbation

        rx = _zero_aperture_prescription()
        _clear_wrapper_merit_cache()

        captured = []

        def capturing_apply_real_lens(E_in, **kw):
            captured.append(np.asarray(E_in).copy())
            return np.zeros_like(E_in)

        try:
            from lumenairy.optimize import core as opt_core
            saved = opt_core.apply_real_lens
            opt_core.apply_real_lens = capturing_apply_real_lens
            try:
                ctx = EvaluationContext(
                    prescription=rx, wavelength=1.31e-6,
                    N=32, dx=10e-6, efl=0.025, bfl=0.025)
                # 3 trials, one perturbation each -- exercises the
                # full _evaluate_perturbed loop.
                merit = ToleranceAwareMerit(
                    sub_merit=StrehlMerit(min_strehl=0.5, weight=1.0),
                    perturbation_spec=[{
                        'surface_index': 0,
                        'decenter_std': 5e-6,
                        'tilt_std': 1e-4,
                        'form_error_rms': 0.0,
                    }],
                    n_trials=3, seed=42, weight=1.0)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _ = merit.evaluate(ctx)
            finally:
                opt_core.apply_real_lens = saved
        finally:
            _clear_wrapper_merit_cache()

        assert len(captured) == 3, (
            f'Expected 3 captures (one per trial), got {len(captured)}')
        for i, E in enumerate(captured):
            assert np.all(E == 0), (
                f'Trial {i}: perturbed E_in was not zero '
                f'(max|E|={np.max(np.abs(E)):.3e}); sentinel was not '
                f'honoured under perturbation.')


# ============================================================================
# Cache invariance pin -- sentinel still in place after these fixes
# ============================================================================

class TestSentinelStillCached:
    """Sanity check: the v4.14.1 contract that
    ``_get_wrapper_merit_cache`` returns ``_cache['mask'] is
    _ZERO_APERTURE_MASK`` for aperture_diameter=0 is still honoured.
    """

    def test_zero_aperture_cache_mask_is_sentinel(self):
        """Pins the v4.14.1 P1-NEW-1 invariant the B.1/B.2 fixes
        depend on."""
        _clear_wrapper_merit_cache()
        try:
            entry = _get_wrapper_merit_cache(
                32, 10e-6, 0.0, np.complex128)
            assert entry['mask'] is _ZERO_APERTURE_MASK
        finally:
            _clear_wrapper_merit_cache()

    def test_negative_aperture_cache_mask_is_sentinel(self):
        """Negative aperture (could arise if a perturbed
        aperture_diameter ever went below zero) also routes through
        the sentinel branch.
        """
        _clear_wrapper_merit_cache()
        try:
            entry = _get_wrapper_merit_cache(
                32, 10e-6, -1e-3, np.complex128)
            assert entry['mask'] is _ZERO_APERTURE_MASK
        finally:
            _clear_wrapper_merit_cache()
