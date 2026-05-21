"""Unit tests for merit-term semantics in lumenairy.optimize.

Every merit term must respect its docstring contract: the merit
landscape's gradient must point toward the documented optimum.  These
tests are regressions for past bugs where the formula and the
docstring disagreed.
"""
from __future__ import annotations

import numpy as np

from lumenairy.optimize.core import (
    BackFocalLengthMerit,
    EvaluationContext,
    FocalLengthMerit,
)


def _ctx(efl: float = 0.0, bfl: float = 0.0) -> EvaluationContext:
    """Minimal EvaluationContext with the fields these merits read."""
    return EvaluationContext(
        prescription={},
        wavelength=1.31e-6,
        N=64,
        dx=10e-6,
        efl=efl,
        bfl=bfl,
    )


class TestFocalLengthMeritAfocalTarget:
    """Regression: when target=0 ('afocal / collimator'), pushing EFL
    toward infinity must drive the merit toward zero -- and pushing it
    toward zero must explode the merit.  The 4.7.x implementation had
    these backwards (formula returned weight*efl^2, which grows as
    efl -> infinity, contrary to the docstring).
    """

    def test_zero_target_decreases_as_efl_grows(self) -> None:
        m = FocalLengthMerit(target=0.0, weight=1.0)
        v_short = m.evaluate(_ctx(efl=0.01))     # 10 mm
        v_med = m.evaluate(_ctx(efl=1.0))         # 1 m
        v_long = m.evaluate(_ctx(efl=100.0))      # 100 m
        v_huge = m.evaluate(_ctx(efl=1e6))        # 1000 km (afocal-ish)
        assert v_short > v_med > v_long > v_huge, (
            f"merit should decrease as EFL grows for the afocal "
            f"target; got {v_short=}, {v_med=}, {v_long=}, {v_huge=}"
        )
        # And the asymptote should be near zero (1e-12 dioptre^2).
        assert v_huge < 1e-10

    def test_zero_target_finite_at_efl_near_zero(self) -> None:
        """At efl = 0 exactly, the merit must stay finite (clamped) so
        the optimiser can recover from a degenerate trial point."""
        m = FocalLengthMerit(target=0.0, weight=1.0)
        v = m.evaluate(_ctx(efl=0.0))
        assert np.isfinite(v)
        # And the gradient direction is "push EFL away from zero":
        v_tiny = m.evaluate(_ctx(efl=1e-9))
        v_small = m.evaluate(_ctx(efl=1e-3))
        assert v_tiny > v_small

    def test_finite_target_minimum_at_target(self) -> None:
        m = FocalLengthMerit(target=0.1, weight=1.0)  # 100 mm
        v_at = m.evaluate(_ctx(efl=0.1))
        v_below = m.evaluate(_ctx(efl=0.05))
        v_above = m.evaluate(_ctx(efl=0.15))
        assert v_at < v_below
        assert v_at < v_above
        assert np.isclose(v_at, 0.0, atol=1e-15)


class TestBackFocalLengthMeritAfocalTarget:
    """Same regression as FocalLengthMerit, applied to BFL."""

    def test_zero_target_decreases_as_bfl_grows(self) -> None:
        m = BackFocalLengthMerit(target=0.0, weight=1.0)
        v_short = m.evaluate(_ctx(bfl=0.01))
        v_long = m.evaluate(_ctx(bfl=100.0))
        v_huge = m.evaluate(_ctx(bfl=1e6))
        assert v_short > v_long > v_huge
        assert v_huge < 1e-10

    def test_finite_target_minimum_at_target(self) -> None:
        m = BackFocalLengthMerit(target=0.05, weight=1.0)
        v_at = m.evaluate(_ctx(bfl=0.05))
        v_off = m.evaluate(_ctx(bfl=0.10))
        assert v_at < v_off
        assert np.isclose(v_at, 0.0, atol=1e-15)
