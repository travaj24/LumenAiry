"""PMMStack at an exact Rayleigh anomaly (2026-11-14): the classical solve
announces and nudges the wavelength, like every other 1-D entry point.

Before: a period equal to the wavelength at normal incidence (orders +/-1
grazing in air) solved silently, with an ill-conditioned answer that
scattered across element sizes and degrees.  Fixture: the strip layer of the
strip / spacer / mirror stack (a generic fixture).
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import WoodNudgeWarning

WL = 1500e-9


def _st(period, degree=10):
    st = PMMStack(period, n_substrate=1.5, degree=degree)
    st.add_layer(40e-9, segments=[(0.36, -90 + 7j), (0.64, 1.0)])
    st.set_source(WL, theta=0.0)
    return st


def test_exact_anomaly_is_announced():
    with pytest.warns(WoodNudgeWarning, match="PMMStack.solve"):
        o, R, T, J = _st(WL).solve()
    assert np.all(np.isfinite(J))


def test_off_the_anomaly_nothing_changes():
    with warnings.catch_warnings():
        warnings.simplefilter("error", WoodNudgeWarning)
        _st(1.001 * WL).solve()


def test_sweep_and_prepared_nudge_the_same_way():
    """A sweep point and a prepared solve at the anomaly equal the pointwise
    solve (all three nudge identically); before, solve() nudged and the
    sweep did not, and they differed by 3.3e-4 on a tapered grating."""
    with pytest.warns(WoodNudgeWarning):
        J0 = np.asarray(_st(WL).solve()[3])
    with pytest.warns(WoodNudgeWarning, match="solve_vs_wavelength"):
        o, R, T, J = _st(WL).solve_vs_wavelength([0.9 * WL, WL], jones=True)
    assert np.array_equal(np.asarray(J)[1], J0)
    with pytest.warns(WoodNudgeWarning, match="prepare"):
        Jp = np.asarray(_st(WL).prepare().solve(wavelength=WL)[3])
    assert np.array_equal(Jp, J0)
