"""S4-5 (AUDIT_V5_24_2) -- optimizer failure-direction regressions.

Three independent failure-handling defects let a FAILED design read as a
GOOD one (or crash / silently mis-optimise) inside the ``optimize``
driver and its merit terms:

(a) A degenerate wave leg (fully-vignetted / dark exit field) makes
    ``find_best_focus`` return ``nan``.  The driver wrote
    ``ctx.strehl_best = float(nan)`` and ``StrehlMerit`` then evaluated
    ``max(0.0, min_strehl - nan) == 0.0`` -- a PERFECT score for a
    failed design (the main wave leg disagreed in the SIGN of its
    failure handling with the wrapper merits, which write the 0.0
    failed-scan sentinel).  Fix: coerce a non-finite best Strehl to 0.0
    at the driver write site.

(b) ``SpotSizeMerit`` reads the default ``rms_radius_best == np.inf``
    when the through-focus scan is skipped (|BFL| out of range
    early-return) and injected ``inf`` into the scipy merit sum,
    stalling the L-BFGS-B line search.  Fix: a non-finite radius yields
    a large but FINITE quadratic penalty.

(c) ``method='lm'`` silently clamps negative merit contributions
    (``sqrt(max(m, 0))``), so a reward-style ``CallableMerit`` term that
    returns a negative value is invisible under 'lm' only.  Fix: warn
    (exactly once per run) when a term is negative under 'lm'.

These tests are pure-NumPy (no JAX / GUI); each pins the observable
contract against an independent pre-fix oracle rather than a tautology.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy
from lumenairy.optimize.context import EvaluationContext
from lumenairy.optimize.core import (
    CallableMerit,
    DesignParameterization,
    SpotSizeMerit,
    StrehlMerit,
    design_optimize,
)


def _singlet_template():
    """A plano-convex N-BK7 singlet with a well-defined finite BFL."""
    return lumenairy.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7', aperture=12e-3)


def _bare_ctx():
    """An EvaluationContext with all defaults (strehl_best=0.0,
    rms_radius_best=np.inf) and a throwaway prescription the failure-leg
    merits under test never read."""
    return EvaluationContext(prescription={}, wavelength=1.30e-6,
                             N=16, dx=10e-6)


# =========================================================================
# (a) A degenerate wave leg -> nan best-Strehl must PENALISE, not reward.
# =========================================================================

class TestS4_5NaNStrehlPenalises:

    def test_pre_fix_oracle_nan_strehl_scores_a_perfect_zero(self):
        """Independent oracle proving the defect is real: the raw ``nan``
        the PRE-fix driver wrote makes ``StrehlMerit`` score a *perfect*
        0.0 (``max(0.0, 0.8 - nan) == 0.0``)."""
        ctx = _bare_ctx()
        ctx.strehl_best = float('nan')
        assert StrehlMerit(min_strehl=0.8, weight=1.0).evaluate(ctx) == 0.0

    def test_coerced_zero_strehl_scores_the_full_deficit(self):
        """The value the POST-fix driver writes (0.0) scores the full
        deficit penalty ``weight * (min_strehl)^2``."""
        ctx = _bare_ctx()
        ctx.strehl_best = 0.0
        got = StrehlMerit(min_strehl=0.8, weight=1.0).evaluate(ctx)
        assert got == pytest.approx(0.8 * 0.8)

    def test_driver_coerces_nan_best_focus_to_zero(self, monkeypatch):
        """End-to-end pin of the driver fix: with ``find_best_focus``
        forced to the degenerate ``(nan, nan)`` return the audit
        describes, the driver writes ``strehl_best == 0.0`` and the
        StrehlMerit wave leg reports a POSITIVE penalty -- not a perfect
        0.0."""
        import lumenairy.optimize.core as core
        monkeypatch.setattr(
            core, 'find_best_focus',
            lambda scan, metric: (float('nan'), float('nan')))
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(40e-3, 90e-3)])
        merit = StrehlMerit(min_strehl=0.8, weight=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = design_optimize(
                parameterization=param, merit_terms=[merit],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        # The driver coerced the degenerate nan to the 0.0 failed-scan
        # value (this is the exact fixed write site).
        assert res.context_final.strehl_best == 0.0
        assert np.isfinite(res.merit)
        # A failed design must READ AS a failure (full deficit), not a
        # perfect score.  Pre-fix this would be 0.0.
        assert res.merit == pytest.approx(0.8 * 0.8)


# =========================================================================
# (b) SpotSizeMerit on a skipped scan (rms_radius_best == inf) -> FINITE.
# =========================================================================

class TestS4_5SpotSizeFiniteOnFailedScan:

    def test_pre_fix_oracle_inf_radius_is_infinite(self):
        """Oracle: the PRE-fix quadratic on the default inf radius is
        ``inf`` -- the value that stalled the scipy line search."""
        ctx = _bare_ctx()
        assert not np.isfinite(ctx.rms_radius_best)  # dataclass default inf
        max_rms = 5e-6
        pre_fix = 1.0 * max(0.0, ctx.rms_radius_best - max_rms) ** 2
        assert not np.isfinite(pre_fix)

    def test_inf_radius_yields_finite_saturation_penalty(self):
        """POST-fix: a non-finite radius maps to the finite saturation
        penalty ``weight * (10 * max_rms_radius)^2``."""
        ctx = _bare_ctx()  # rms_radius_best == inf
        m = SpotSizeMerit(max_rms_radius=5e-6, weight=1.0)
        got = m.evaluate(ctx)
        assert np.isfinite(got)
        assert got == pytest.approx((10.0 * 5e-6) ** 2)

    def test_finite_radius_below_target_still_zero(self):
        """The guard must not perturb the clean path: a radius below
        target scores exactly 0 (no false penalty)."""
        ctx = _bare_ctx()
        ctx.rms_radius_best = 2e-6
        assert SpotSizeMerit(max_rms_radius=5e-6, weight=1.0).evaluate(ctx) == 0.0

    def test_finite_radius_above_target_unchanged_quadratic(self):
        """The guard must not perturb the clean path: a radius above
        target scores the original ``(r - max)^2`` quadratic."""
        ctx = _bare_ctx()
        ctx.rms_radius_best = 9e-6
        got = SpotSizeMerit(max_rms_radius=5e-6, weight=1.0).evaluate(ctx)
        assert got == pytest.approx((9e-6 - 5e-6) ** 2)

    def test_driver_early_return_hands_scipy_a_finite_merit(self, monkeypatch):
        """End-to-end: force the |BFL|>10 early-return (which leaves
        ``rms_radius_best`` at its default ``inf``) and confirm
        SpotSizeMerit still hands scipy a FINITE merit -- i.e. the merit
        guard is what saves the run, since the driver never wrote a
        finite radius."""
        import lumenairy.optimize.core as core
        # A huge finite BFL routes evaluate() into the |BFL|>10
        # early-return that skips the through-focus scan.
        monkeypatch.setattr(
            core, 'system_abcd',
            lambda surfs, wl: (np.eye(2), 0.05, 1e12, 0.05))
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(40e-3, 90e-3)])
        merit = SpotSizeMerit(max_rms_radius=5e-6, weight=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = design_optimize(
                parameterization=param, merit_terms=[merit],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        # The driver left rms_radius_best untouched at its inf default on
        # the early-return path -- the finite merit is the guard's doing.
        assert not np.isfinite(res.context_final.rms_radius_best)
        assert np.isfinite(res.merit)
        assert res.merit == pytest.approx((10.0 * 5e-6) ** 2)


# =========================================================================
# (c) method='lm' silently clamps negative merit terms -> warn once.
# =========================================================================

class TestS4_5LMNegativeTermWarns:

    @staticmethod
    def _reward_term():
        # A reward-style term returning a constant NEGATIVE contribution.
        return CallableMerit(lambda ctx: -1.0, weight=1.0, needs_wave=False)

    def test_lm_negative_term_warns_exactly_once(self):
        """``method='lm'`` forms residuals ``sqrt(max(m, 0))`` which
        silently zeroes negative (reward-style) contributions; the driver
        must warn -- and, because ``least_squares`` calls the residual
        function several times (initial value + FD Jacobian), the warning
        must be one-shot per run."""
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=None)  # bounds=None so real 'lm' runs (no 'trf' override)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            design_optimize(
                parameterization=param, merit_terms=[self._reward_term()],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='lm', max_iter=3, verbose=False)
        hits = [w for w in ws
                if issubclass(w.category, RuntimeWarning)
                and 'reward-style' in str(w.message)]
        assert len(hits) == 1, (
            f"the lm negative-term warning must fire exactly once per "
            f"run; got {len(hits)} from messages="
            f"{[str(w.message) for w in ws]}")

    def test_non_lm_method_does_not_emit_the_clamp_warning(self):
        """The clamp -- and therefore its warning -- is 'lm'-only.  A
        non-'lm' method sums the raw (negative) value and must NOT emit
        the reward-style warning."""
        param = DesignParameterization(
            template=_singlet_template(),
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(40e-3, 90e-3)])
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            design_optimize(
                parameterization=param, merit_terms=[self._reward_term()],
                wavelength=1.30e-6, N=16, dx=10e-6,
                method='L-BFGS-B', max_iter=1, verbose=False)
        hits = [w for w in ws if 'reward-style' in str(w.message)]
        assert not hits, (
            f"non-'lm' methods must not emit the lm-clamp warning; got "
            f"{[str(w.message) for w in hits]}")
