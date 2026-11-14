"""PMMStack at an exact Rayleigh anomaly (2026-11-14): the classical solve
announces and nudges the wavelength, like every other 1-D entry point.

Before: a period equal to the wavelength at normal incidence (orders +/-1
grazing in air) solved silently, with an ill-conditioned answer that
scattered by 1e-5 .. 5e-4 across element sizes and degrees.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import WoodNudgeWarning

WL = 1300e-9


def _st(period, degree=10):
    st = PMMStack(period, n_substrate=1.447, degree=degree)
    st.add_layer(50e-9, segments=[(0.47, -74 + 6j), (0.53, 1.0)])
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
