"""The small-element guard of Chebyshev / uniform layouts, re-derived.

Two layers whose walls are slightly more than ``min_feature`` apart leave a
sub-nanometre union cell; a Chebyshev split of it into 16 or 20 elements per
region puts elements at a few 1e-6 of the wavelength.  The Chebyshev /
uniform refusal bar was 3e-6 of the wavelength and refused those solves,
although they are as accurate as the ones it accepted.  Measured on thin and
wide cells alike (the round-off scale is ``h / wavelength`` whatever made the
element small), a Chebyshev / uniform layout keeps its 1e-5 accuracy level
down to 2e-6 of the wavelength (8.2e-6 worst), is case-dependent between
1e-6 and 2e-6 (2.4e-6 .. 1.5e-5), and is refused below 1e-6; the band
between solves with a ``UserWarning``.
"""
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm.stack as S
from lumenairy.elements.pmm import PMMStack

P, WL, W = 500e-9, 1500e-9, 0.4


def _two_walls(n_el, degree=10, gap=0.6e-9, theta=0.0, **kw):
    st = PMMStack(P, n_substrate=1.5, degree=degree, elements_per_region=n_el,
                  **kw)
    st.add_layer(40e-9, segments=[(W, -50 + 4j), (1 - W, 1.0)])
    st.add_layer(20e-9, segments=[(W + gap / P, 2.5),
                                  (1 - W - gap / P, 1.0)])
    return st.set_source(WL, theta=theta)


def _J(st):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return np.asarray(st.solve()[3])


@pytest.mark.parametrize("n_el", [16, 20])
@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_near_coincident_walls_solve_silently_and_stay_accurate(n_el, theta):
    """Wall elements 3.8e-6 / 2.5e-6 of the wavelength in a 0.6 nm cell:
    refused before, now solved without a warning; measured within 1.8e-6 of
    the degree-20 x 6 reference."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.asarray(_two_walls(6, degree=20, theta=theta).solve()[3])
    assert np.abs(_J(_two_walls(n_el, theta=theta)) - ref).max() < 1e-5


def test_the_band_below_2e6_warns_and_solves():
    """28 elements in the 0.6 nm cell: a 1.26e-6 x wavelength wall element,
    inside the warning band -- solved, with the band's UserWarning."""
    with pytest.warns(UserWarning, match="round-off"):
        J = np.asarray(_two_walls(28, degree=4).solve()[3])
    assert np.all(np.isfinite(J))


def test_below_1e6_is_refused():
    """64 Chebyshev elements in the 0.6 nm cell: a 2.4e-7 x wavelength wall
    element -- refused (before solving), with the Chebyshev advice."""
    with pytest.raises(ValueError, match="fewer elements_per_region"):
        _two_walls(64).solve()


def test_bars_apply_to_thin_and_wide_cells_alike():
    for wc in (1e-6, 1e-10):                      # a wide and a 0.1 nm cell
        hit = PMMStack._refused_element([(5e-13, wc, True)], True, WL)
        assert hit is not None and hit[2] == S._SMALL_ELEMENT_FRAC_PLAIN
        assert PMMStack._refused_element([(2e-12, wc, True)], False,
                                         WL) is None
    assert S._SMALL_ELEMENT_FRAC_PLAIN == 1e-6
    assert S._SMALL_ELEMENT_WARN_FRAC_PLAIN == 2e-6


def test_an_element_on_the_bar_is_not_below_it():
    size = S._SMALL_ELEMENT_FRAC * WL
    cells = [(size * (1 - 1e-12), 1e-7, True)]
    assert PMMStack._layout_made_below(cells, size) is None
