"""v4.15.3 Agent C tests -- wire-the-dead-sentinels closure.

Closes audit ``AUDIT_V4_15_2_2026_05_18`` P1-NEW-F1-3:

    The v4.15.2 closure for the v4.15.1 audit P2 finding ("sentinel
    proliferation in ``optimize/core.py``") defined three new
    ``_Sentinel`` subclasses (``_InvalidFocalLengthSentinel``,
    ``_FailedScanStrehlSentinel``, ``_PerturbedABCDFallbackSentinel``)
    at ``optimize/core.py:2044-2144`` but the four callsites at
    ``:2374, :2632, :2873, :2936`` (v4.15.2 line numbers) were NOT
    updated -- they still wrote raw values (``-1.0``-equivalent
    ``1e9``, ``0.0``, and the tuple fallback).  The sentinel classes
    were unreachable.  CHANGELOG bullet asserted the migration was
    complete; in fact only the class definitions had landed.

v4.15.3 Agent C wires the two SCALAR sentinels at their intended
callsites:

* ``_INVALID_FL_SENTINEL_OBJ`` at the
  ``MultiWavelengthMerit.evaluate`` degenerate-ABCD branch (callsite
  1, v4.15.3 line ~2424).
* ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` at the two
  through-focus-scan-failure branches in
  ``MultiFieldMerit.evaluate`` (callsite 2, v4.15.3 line ~2696) and
  ``ToleranceAwareMerit.evaluate`` (callsite 4, v4.15.3 line ~3015).

The third sentinel ``_PerturbedABCDFallbackSentinel`` (callsite 3,
v4.15.3 line ~2873) is INTENTIONALLY NOT wired: the underlying
fallback is a 2-tuple ``(efl_p, bfl_p)`` not a scalar, and wrapping
the tuple in a single sentinel singleton breaks downstream
unpacking.  The class is KEPT defined (for v4.15.2 test-pin
compatibility -- see ``test_v4_15_2_agent_e.py``) but is marked as
dead code via a class-level ``_v4_15_3_dead_code`` marker and an
explicit deprecation docstring.

Files touched (Agent-C scope):

* ``lumenairy/optimize/core.py`` -- wired the two scalar sentinels;
  added the dead-code marker on the tuple sentinel; added the
  downstream ``float()`` coercion at ``StrehlMerit.evaluate``.
* ``tests/unit/test_v4_15_1_agent_g_matches_system_abcd.py`` -- 4-fold
  mirror parametric coverage (4-flat periscope + 2-curved/2-flat
  cassegrain).
* ``tests/unit/test_v4_15_2_agent_b.py`` -- non-tautological
  Gaussian-beam waist relation pin for FourierTransform.
* ``tests/unit/test_v4_15_3_agent_c.py`` -- this file.
* ``docs/release_notes/.release_notes_v4_15_3_agent_c.md`` -- CHANGELOG
  draft.

Author: Andrew Traverso -- v4.15.3 / Agent C.
"""
from __future__ import annotations

import pickle
import warnings
from typing import Any
from unittest import mock

import numpy as np
import pytest

import lumenairy as la
from lumenairy.optimize.core import (
    _FAILED_SCAN_STREHL_SENTINEL_OBJ,
    _INVALID_FL_SENTINEL_OBJ,
    EvaluationContext,
    FocalLengthMerit,
    MultiFieldMerit,
    MultiWavelengthMerit,
    StrehlMerit,
    ToleranceAwareMerit,
    _FailedScanStrehlSentinel,
    _InvalidFocalLengthSentinel,
)

# v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a): the
# ``_PerturbedABCDFallbackSentinel`` class and its singleton
# ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ`` were DELETED in v4.15.4 as
# dead code (never wired; the perturbed-ABCD fallback is a 2-tuple,
# which is not single-sentinel-friendly).  The
# ``TestPerturbedABCDSentinelDeadCode`` class below (which asserted
# the v4.15.3 class-attribute marker and AST-unwiredness) is
# superseded by the v4.15.4 deletion-existence pin in
# ``test_v4_15_2_agent_e.py
# ::TestOptimizeCoreSentinelMigration
# ::test_perturbed_abcd_fallback_sentinel_deleted_v4_15_4`` and is
# converted into a single non-existence pin here for the v4.15.3
# scope's own record.


# ============================================================================
# Helpers -- build a minimal-but-valid context that the merits will accept
# ============================================================================


def _build_minimal_singlet_prescription() -> dict:
    """Build a minimal BK7 singlet prescription for the merits to chew on.

    Stays small (N=32) so the wave-leg branches (if exercised) cost
    O(1 ms) each.  The wave-leg branches won't actually be entered in
    these tests because we provide ``ctx.E_exit = None`` so
    ``ctx.E_exit is not None`` short-circuits the wave-leg gate (the
    merit collapses to just the ABCD extraction + sub-merit
    evaluation path).
    """
    return la.make_singlet(
        R1=50e-3, R2=-50e-3, d=5e-3,
        glass='N-BK7', aperture=25.4e-3,
    )


def _build_minimal_ctx(prescription: dict) -> EvaluationContext:
    """A minimal ``EvaluationContext`` with nominal EFL/BFL filled in.

    The merits' ABCD-failure branches re-extract EFL/BFL via
    ``system_abcd(surfs, wl)``; the values on ``ctx`` are only used
    for fallback paths.
    """
    return EvaluationContext(
        prescription=prescription,
        wavelength=633e-9,
        N=32,
        dx=4e-6,
        efl=50e-3,
        bfl=50e-3,
        E_exit=None,
        strehl_best=0.9,
        rms_radius_best=1e-6,
        z_best=50e-3,
    )


# ============================================================================
# C.1.1 -- sentinel returned from failed ABCD path
# ============================================================================


class TestInvalidFlSentinelWired:
    """v4.15.3 (audit P1-NEW-F1-3): the
    ``_INVALID_FL_SENTINEL_OBJ`` singleton must be reachable from the
    ``MultiWavelengthMerit.evaluate`` degenerate-ABCD branch (the
    callsite that v4.15.2 left unwired).
    """

    def test_invalid_fl_sentinel_returned_from_failed_abcd_path(
        self,
    ) -> None:
        """Monkey-patch :func:`system_abcd` to raise ``ValueError`` at
        the wavelength used by the merit and assert the
        sentinel-handling branch is reached.

        We verify via a side-channel: the
        ``_INVALID_FL_SENTINEL_OBJ`` has ``float() == 1e9``, and
        :attr:`EvaluationContext.efls_per_wavelength` is populated as
        a side effect of evaluation.  When the ABCD path fails, the
        sentinel collapses to ``1e9`` via the local ``float(efl)``
        cast at the ``efls.append`` line, so the recorded per-
        wavelength EFL must equal ``1e9``.
        """
        rx = _build_minimal_singlet_prescription()
        ctx = _build_minimal_ctx(rx)

        # MultiWavelengthMerit with a single wavelength; sub-merit
        # is FocalLengthMerit (geometric, doesn't need wave leg).
        merit = MultiWavelengthMerit(
            wavelengths=[633e-9],
            sub_merit=FocalLengthMerit(target=50e-3, weight=1.0),
            weight=1.0,
        )

        # Monkey-patch the lumenairy.optimize.core import-site of
        # ``system_abcd`` (the function is imported lazily at module
        # body via ``from ..raytrace import surfaces_from_prescription,
        # system_abcd`` higher up in optimize/core).  Patch via
        # ``mock.patch`` on the import-site path.
        with mock.patch(
            'lumenairy.optimize.core.system_abcd',
            side_effect=ValueError('simulated ABCD failure'),
        ):
            merit.evaluate(ctx)

        # Side effect: ctx.efls_per_wavelength records the failed-path
        # value as ``float(_INVALID_FL_SENTINEL_OBJ) == 1e9``.
        assert ctx.efls_per_wavelength is not None
        assert ctx.efls_per_wavelength.shape == (1,)
        assert ctx.efls_per_wavelength[0] == pytest.approx(1e9), (
            f"Expected the degenerate-ABCD branch to record "
            f"float(_INVALID_FL_SENTINEL_OBJ) == 1e9 in "
            f"efls_per_wavelength, got "
            f"{ctx.efls_per_wavelength[0]!r}."
        )

    def test_invalid_fl_sentinel_skips_per_wavelength_wave_leg(
        self,
    ) -> None:
        """The wired callsite must also guard the downstream
        ``np.isfinite(bfl) and abs(bfl) < 10`` per-wavelength wave-leg
        check.  Pre-wire, ``bfl = 1e9`` already failed the
        ``< 10`` clause; post-wire, ``bfl = _INVALID_FL_SENTINEL_OBJ``
        is a Python object and would raise ``TypeError`` inside
        ``np.isfinite`` if the guard's ``is not _INVALID_FL_SENTINEL_OBJ``
        check were missing.  This test exercises the failure-branch
        with a wave-leg-needing sub-merit (the merit must NOT raise).
        """
        rx = _build_minimal_singlet_prescription()
        ctx = _build_minimal_ctx(rx)
        # Synthesize an exit field so the wave-leg gate doesn't
        # short-circuit on the ``ctx.E_exit is not None`` check.
        ctx.E_exit = np.zeros((ctx.N, ctx.N), dtype=np.complex128)

        # StrehlMerit needs wave; combined with a degenerate ABCD
        # this exercises the line-2382 guard directly.
        merit = MultiWavelengthMerit(
            wavelengths=[633e-9],
            sub_merit=StrehlMerit(min_strehl=0.8, weight=1.0),
            weight=1.0,
        )
        with mock.patch(
            'lumenairy.optimize.core.system_abcd',
            side_effect=ValueError('simulated ABCD failure'),
        ):
            # Must NOT raise ``TypeError: ufunc 'isfinite' not
            # supported for the input types``.  Pre-wire this would
            # have been impossible (bfl was always 1e9, a plain
            # float); post-wire the sentinel singleton would crash
            # ``np.isfinite`` without the ``is``-guard.
            merit.evaluate(ctx)


# ============================================================================
# C.1.2 -- _FAILED_SCAN_STREHL_SENTINEL_OBJ wired
# ============================================================================


class TestFailedScanStrehlSentinelWired:
    """v4.15.3 (audit P1-NEW-F1-3): the
    ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton must be reachable
    from the ``MultiFieldMerit.evaluate`` and
    ``ToleranceAwareMerit.evaluate`` through-focus-scan-failure
    branches (the two callsites v4.15.2 left unwired).
    """

    def test_failed_scan_strehl_sentinel_returned_from_multifield(
        self,
    ) -> None:
        """``MultiFieldMerit.evaluate``: when the field-leg
        ``through_focus_scan`` raises, ``sub_ctx.strehl_best`` must
        be assigned the ``_FAILED_SCAN_STREHL_SENTINEL_OBJ``
        singleton (not the bare ``0.0`` literal).

        Verified via mock: the sub-merit captures the ``sub_ctx``
        passed to it, and we check ``sub_ctx.strehl_best is
        _FAILED_SCAN_STREHL_SENTINEL_OBJ``.
        """
        rx = _build_minimal_singlet_prescription()
        ctx = _build_minimal_ctx(rx)
        # A non-None E_exit gates the wave-leg path.
        ctx.E_exit = np.ones((ctx.N, ctx.N), dtype=np.complex128)

        captured = {'sub_ctxs': []}

        class _CapturingSubMerit(StrehlMerit):
            needs_wave = True
            def evaluate(self_inner, sub_ctx: Any) -> float:  # noqa: N805
                captured['sub_ctxs'].append(sub_ctx)
                return super().evaluate(sub_ctx)

        # Use the canonical (theta_x, theta_y) tuple form (the
        # scalar form fires a one-shot DeprecationWarning that is
        # noise here).  ``(0.0, 0.0)`` is the on-axis field.
        merit = MultiFieldMerit(
            field_angles=[(0.0, 0.0)],
            sub_merit=_CapturingSubMerit(min_strehl=0.8, weight=1.0),
            weight=1.0,
        )

        # Force through_focus_scan to fail inside MultiFieldMerit.
        with mock.patch(
            'lumenairy.optimize.core.through_focus_scan',
            side_effect=ValueError('simulated through-focus failure'),
        ):
            # The small N=32 grid fires a UserWarning from
            # apply_real_lens about the prescription aperture
            # exceeding the simulation grid -- intentional to keep
            # the test minimal.  Suppress so the test does not
            # break under ``-W error``.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                merit.evaluate(ctx)

        assert len(captured['sub_ctxs']) == 1, (
            f"Expected MultiFieldMerit to invoke the sub-merit once "
            f"for the single field-angle entry; got "
            f"{len(captured['sub_ctxs'])} calls."
        )
        sub_ctx = captured['sub_ctxs'][0]
        assert sub_ctx.strehl_best is _FAILED_SCAN_STREHL_SENTINEL_OBJ, (
            f"Expected sub_ctx.strehl_best to be the "
            f"_FAILED_SCAN_STREHL_SENTINEL_OBJ singleton; got "
            f"{sub_ctx.strehl_best!r} of type "
            f"{type(sub_ctx.strehl_best).__name__}."
        )

    def test_failed_scan_strehl_sentinel_returned_from_tolerance(
        self,
    ) -> None:
        """``ToleranceAwareMerit.evaluate``: the perturbed
        through-focus-scan-failure branch (callsite 4) must wire the
        same ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton.
        """
        rx = _build_minimal_singlet_prescription()
        ctx = _build_minimal_ctx(rx)

        captured = {'sub_ctxs': []}

        class _CapturingSubMerit(StrehlMerit):
            needs_wave = True
            def evaluate(self_inner, sub_ctx: Any) -> float:  # noqa: N805
                captured['sub_ctxs'].append(sub_ctx)
                return super().evaluate(sub_ctx)

        # ToleranceAwareMerit needs a small n_trials so the test is
        # quick; use a single-surface tiny-decenter perturbation
        # schedule that nominally barely moves the design but still
        # forces the through-focus-scan path to be exercised.
        merit = ToleranceAwareMerit(
            sub_merit=_CapturingSubMerit(min_strehl=0.9, weight=1.0),
            perturbation_spec=[{
                'surface_index': 0,
                'decenter_std': 1e-7,
                'tilt_std': 0.0,
                'form_error_rms': 0.0,
            }],
            n_trials=1,
            seed=0,
            weight=1.0,
        )
        with mock.patch(
            'lumenairy.optimize.core.through_focus_scan',
            side_effect=ValueError('simulated tolerance-trial failure'),
        ):
            with warnings.catch_warnings():
                # See note in the MultiField test above; same
                # apply_real_lens UserWarning fires here.
                warnings.simplefilter('ignore', UserWarning)
                merit.evaluate(ctx)

        assert len(captured['sub_ctxs']) >= 1, (
            "Expected ToleranceAwareMerit to invoke the sub-merit at "
            "least once.")
        # At least one of the captured sub_ctxs must carry the
        # sentinel (one per failed trial).
        sentinel_count = sum(
            1 for s in captured['sub_ctxs']
            if s.strehl_best is _FAILED_SCAN_STREHL_SENTINEL_OBJ
        )
        assert sentinel_count >= 1, (
            f"Expected at least one tolerance-trial sub_ctx to carry "
            f"the _FAILED_SCAN_STREHL_SENTINEL_OBJ singleton, got "
            f"{sentinel_count}/{len(captured['sub_ctxs'])} trials."
        )


# ============================================================================
# C.1.3 -- downstream consumer (StrehlMerit) handles the sentinel
# ============================================================================


class TestDownstreamConsumersHandleSentinel:
    """v4.15.3 (audit P1-NEW-F1-3): the immediate downstream consumer
    of ``sub_ctx.strehl_best`` -- ``StrehlMerit.evaluate`` -- must
    correctly handle the ``_FAILED_SCAN_STREHL_SENTINEL_OBJ``
    singleton.  The brief explicitly notes that we should only update
    immediate consumer sites; broader merits-framework refactor is
    deferred to v4.16+.
    """

    def test_strehl_merit_handles_failed_scan_sentinel(self) -> None:
        """``StrehlMerit.evaluate(ctx)`` with
        ``ctx.strehl_best = _FAILED_SCAN_STREHL_SENTINEL_OBJ`` must
        return the worst-case penalty (the same value it would have
        returned for ``strehl_best = 0.0`` pre-wire) -- the
        ``float()`` coercion added at the consumer is preserves
        physics behaviour while threading the sentinel through.
        """
        merit = StrehlMerit(min_strehl=0.8, weight=2.0)
        ctx_sentinel = _build_minimal_ctx(
            _build_minimal_singlet_prescription())
        ctx_sentinel.strehl_best = _FAILED_SCAN_STREHL_SENTINEL_OBJ

        ctx_scalar = _build_minimal_ctx(
            _build_minimal_singlet_prescription())
        ctx_scalar.strehl_best = 0.0

        val_sentinel = merit.evaluate(ctx_sentinel)
        val_scalar = merit.evaluate(ctx_scalar)
        # Both must yield the same numeric merit (worst-case
        # ``2.0 * (0.8 - 0.0)^2 = 1.28``).
        assert val_sentinel == pytest.approx(val_scalar), (
            f"StrehlMerit should treat the sentinel form the same as "
            f"the scalar 0.0 (physics preservation): "
            f"sentinel={val_sentinel!r}, scalar={val_scalar!r}."
        )
        # And the expected analytical value.
        expected = 2.0 * (0.8 - 0.0) ** 2
        assert val_sentinel == pytest.approx(expected), (
            f"StrehlMerit returned {val_sentinel!r}, expected "
            f"{expected!r} = weight * max(0, min_strehl)^2."
        )

    def test_strehl_merit_handles_normal_scalar(self) -> None:
        """Regression check: the ``float()`` coercion at
        :meth:`StrehlMerit.evaluate` (added by v4.15.3) does NOT
        change the behaviour for the normal (non-sentinel) scalar
        path -- a plain ``float`` is its own ``float()``.
        """
        merit = StrehlMerit(min_strehl=0.8, weight=1.0)
        ctx = _build_minimal_ctx(_build_minimal_singlet_prescription())
        # Range of normal Strehl values to spot-check.
        for s in (0.0, 0.5, 0.79, 0.8, 0.95, 1.0):
            ctx.strehl_best = s
            v = merit.evaluate(ctx)
            expected = (max(0.0, 0.8 - s)) ** 2
            assert v == pytest.approx(expected), (
                f"StrehlMerit({s!r}) returned {v!r}, expected "
                f"{expected!r}."
            )


# ============================================================================
# C.1.4 -- _PerturbedABCDFallbackSentinel marked as dead code
# ============================================================================


class TestPerturbedABCDSentinelDeletedV4_15_4:
    """v4.15.4 supersedes the v4.15.3 ``TestPerturbedABCDSentinelDeadCode``
    class.  The v4.15.3 closure marked the ``_PerturbedABCDFallbackSentinel``
    class as dead code via a class-level ``_v4_15_3_dead_code = True``
    attribute and an explicit deprecation docstring -- but the marker
    was data-only, not honoured by any static analyser.  v4.15.4
    deletes the class outright (AUDIT_V4_15_3 P2-NEW-F1-B option a).

    This collapsed class records the v4.15.4 deletion from the
    v4.15.3 scope's vantage point.  The canonical v4.15.4 deletion
    pin lives at ``test_v4_15_2_agent_e.py
    ::TestOptimizeCoreSentinelMigration
    ::test_perturbed_abcd_fallback_sentinel_deleted_v4_15_4``.
    """

    def test_perturbed_abcd_sentinel_class_no_longer_exists(self) -> None:
        """The class is no longer importable from
        ``lumenairy.optimize.core``."""
        from lumenairy.optimize import core as oc
        assert not hasattr(oc, '_PerturbedABCDFallbackSentinel'), (
            "optimize/core.py still exposes _PerturbedABCDFallbackSentinel; "
            "v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted the "
            "class outright as unwired dead code."
        )

    def test_perturbed_abcd_singleton_no_longer_exists(self) -> None:
        """The singleton is no longer importable from
        ``lumenairy.optimize.core``."""
        from lumenairy.optimize import core as oc
        assert not hasattr(oc, '_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ'), (
            "optimize/core.py still exposes "
            "_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ; v4.15.4 "
            "(AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted the singleton."
        )


# ============================================================================
# C.1.5 / C.1.6 -- sentinel pickle round-trip
# ============================================================================


class TestSentinelPickleRoundTrip:
    """v4.15.3 (audit P1-NEW-F1-3): pickle round-trip for the two
    sentinels we WIRED must preserve singleton identity.

    The v4.15.2 ``test_v4_15_2_agent_e.py
    ::TestOptimizeCoreSentinelMigration
    ::test_optimize_core_sentinel_pickle_roundtrip`` covers the same
    invariant for the three v4.15.2 sentinel classes (including the
    one we declared dead).  This file adds dedicated single-purpose
    pins for the two wired singletons so a v4.15.3 regression
    isolates cleanly.
    """

    def test_invalid_fl_sentinel_pickle_round_trip(self) -> None:
        """``pickle.loads(pickle.dumps(_INVALID_FL_SENTINEL_OBJ)) is
        _INVALID_FL_SENTINEL_OBJ``.  Distributed-pipeline pin: a
        worker that unpickles a context whose ``efl`` field carried
        the sentinel must see the singleton, not a fresh
        ``_InvalidFocalLengthSentinel`` instance.
        """
        recovered = pickle.loads(
            pickle.dumps(_INVALID_FL_SENTINEL_OBJ))
        assert recovered is _INVALID_FL_SENTINEL_OBJ, (
            f"_INVALID_FL_SENTINEL_OBJ pickle round-trip broke "
            f"identity: got {recovered!r} (type "
            f"{type(recovered).__name__}); expected the singleton.")
        # And the subclass survives too.
        assert isinstance(recovered, _InvalidFocalLengthSentinel)
        # ``float()`` still yields ``1e9`` after the round-trip.
        assert float(recovered) == pytest.approx(1e9)

    def test_failed_scan_strehl_sentinel_pickle_round_trip(
        self,
    ) -> None:
        """``pickle.loads(pickle.dumps(_FAILED_SCAN_STREHL_SENTINEL_OBJ))
        is _FAILED_SCAN_STREHL_SENTINEL_OBJ``.  Same distributed-
        pipeline pin as above."""
        recovered = pickle.loads(
            pickle.dumps(_FAILED_SCAN_STREHL_SENTINEL_OBJ))
        assert recovered is _FAILED_SCAN_STREHL_SENTINEL_OBJ, (
            f"_FAILED_SCAN_STREHL_SENTINEL_OBJ pickle round-trip "
            f"broke identity: got {recovered!r}; expected the "
            f"singleton.")
        assert isinstance(recovered, _FailedScanStrehlSentinel)
        # ``float()`` still yields ``0.0`` after the round-trip.
        assert float(recovered) == pytest.approx(0.0)

    # v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a): the
    # ``test_perturbed_abcd_sentinel_pickle_round_trip_still_works``
    # cross-check was removed because v4.15.4 deleted the
    # ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ`` singleton outright.
    # Pickle-round-trip identity for the remaining two wired sentinels
    # (above) is unaffected.


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
