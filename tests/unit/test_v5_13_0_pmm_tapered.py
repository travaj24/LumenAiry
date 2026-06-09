"""v5.13.0 -- PMMStack.add_tapered_grating: trapezoidal / slanted-sidewall 1-D
grating as a z-staircase of thin VERTICAL PMM layers (lateral-exact, no Fourier
floor in x), the spectral-element counterpart of RCWAStack.add_tapered_grating.

Pins: the vertical limit (equal duties) reproduces a single vertical layer
EXACTLY; a true taper conserves energy and converges monotonically in n_slices;
the new PMMStack energy tripwire warns on the many-interface cascade instability
(it cannot return non-physical gain silently); input validation.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm.stack import _warn_stack_energy

_P = 0.8e-6
_WL = 0.55e-6
_TH = 0.3e-6
_ER, _EG = 6.0 + 0j, 1.0 + 0j


def _tapered(duty_b, duty_t, n_slices, degree=12):
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=degree)
    st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                           duty_bottom=duty_b, duty_top=duty_t, n_slices=n_slices)
    return st.set_source(_WL, angle=0.0).solve()


def test_vertical_limit_matches_single_layer():
    """duty_top == duty_bottom is a vertical binary grating -> the staircase of
    identical slices must reproduce a single vertical layer EXACTLY."""
    o_t, R_t, T_t, _j = _tapered(0.5, 0.5, n_slices=5, degree=12)
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(_TH, segments=[(0.25, _EG), (0.5, _ER), (0.25, _EG)])
    o_v, R_v, T_v, _ = st.set_source(_WL, angle=0.0).solve()
    assert np.array_equal(o_t, o_v)
    assert np.max(np.abs(R_t - R_v)) < 1e-11
    assert np.max(np.abs(T_t - T_v)) < 1e-11


def test_taper_converges_and_conserves():
    """A true trapezoid (0.6 -> 0.3) conserves energy and the zeroth-order
    transmission converges monotonically with n_slices (no plateau)."""
    t0s = []
    for ns in (2, 4, 8, 14):
        o, R, T, _ = _tapered(0.6, 0.3, n_slices=ns)
        for pol in (0, 1):                       # both incident polarizations
            assert abs(float(R[pol].sum() + T[pol].sum()) - 1.0) < 1e-4
        t0s.append(float(T[1][o == 0][0]))       # TE (Ey) zeroth order
    d = np.diff(t0s)
    assert np.all(d > 0)                          # monotone increasing
    assert d[-1] < d[0]                           # and converging (shrinking steps)


def test_energy_tripwire_warns_on_gain():
    """The guard warns when a solve returns non-physical R+T > 1 (the cascade
    instability), and stays silent on a conserving result."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # conserving -> no warning
        _warn_stack_energy(np.array([[0.3, 0.1]]), np.array([[0.4, 0.2]]))
    with pytest.warns(UserWarning, match="energy not conserved"):
        _warn_stack_energy(np.array([[0.9, 0.5]]), np.array([[0.8, 0.2]]))


def test_validation():
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=8)
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=1.4, n_slices=4)     # duty > 1
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.5, n_slices=0)     # n_slices < 1
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.5, n_slices=4, rule="bogus")
