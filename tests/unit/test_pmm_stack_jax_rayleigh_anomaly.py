"""The JAX twin of ``PMMStack.solve`` nudges an exact Rayleigh anomaly as the
NumPy path does.

At a diffraction order exactly at cut-off the NumPy solve runs a wavelength
nudged by ~1e-7 relative (with a ``WoodNudgeWarning``); the twin used to solve
the exact wavelength, so the two disagreed by 8.6e-4 there (period equal to
the wavelength at normal incidence) against 6e-13 one part in 1e4 away.  The
nudge reads only CONCRETE media: a traced permittivity's own anomalies cannot
be located without severing the trace (documented on ``PMMStack.solve``).
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import WoodNudgeWarning

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)


def _solve(e, period, wl=1000e-9):
    st = la.PMMStack(period, n_substrate=1.45, degree=12)
    st.add_layer(60e-9, segments=[(0.4, -20 + 2j), (0.6, 1.0)])
    st.add_layer(30e-9, eps=e)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        J = np.asarray(st.set_source(wl, theta=0.0).solve()[3])
    return J, sum(issubclass(x.category, WoodNudgeWarning) for x in w)


def test_twin_nudges_at_an_exact_anomaly():
    Jn, nn = _solve(2.1 + 0j, 1000e-9)
    Jj, nj = _solve(jnp.asarray(2.1 + 0j), 1000e-9)
    assert nn == 1 and nj == 1
    assert np.abs(Jn - Jj).max() < 1e-10


def test_twin_unchanged_off_the_anomaly():
    Jn, nn = _solve(2.1 + 0j, 1000e-9 * (1 + 1e-4))
    Jj, nj = _solve(jnp.asarray(2.1 + 0j), 1000e-9 * (1 + 1e-4))
    assert nn == 0 and nj == 0
    assert np.abs(Jn - Jj).max() < 1e-10


def test_twin_gradient_at_the_anomaly_is_finite():
    def f(e):
        st = la.PMMStack(1000e-9, n_substrate=1.45, degree=10)
        st.add_layer(60e-9, segments=[(0.4, -20 + 2j), (0.6, 1.0)])
        st.add_layer(30e-9, eps=e)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.abs(st.set_source(1000e-9, theta=0.0).solve()[3][0, 0])
    g = jax.grad(lambda e: f(e + 0j))(2.1)
    assert np.isfinite(float(g))
