"""The prepared solve's caches are keyed on everything the cached modes read.

``PMMStack.prepare()`` caches the half-space modes per solve key and every
layer's modes per layer key.  The half-space modes are built from
``n_superstrate`` / ``n_substrate`` and the layer modes from the Bloch
wavenumber ``kx0 = Re(n_superstrate) sin(angle) k0``, while both are plain
public attributes of the stack.  The keys used to omit them, so a solve
after ``st.n_sub = ...`` mixed the old half-space modes with the new far
field (measured 0.315 off the answer of the changed stack), and one after
``st.n_sup = ...`` at oblique incidence reused layer modes of the old
``kx0``.  A layer added after ``prepare()`` reached the cascade with no
cached grid (an ``IndexError``); it is now refused as a layout change.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la

P, WL = 500e-9, 1500e-9
M = -20.0 + 2.0j


def _stack(n_sub=1.5, n_sup=1.0, key=True):
    st = la.PMMStack(P, n_substrate=n_sub, n_superstrate=n_sup, degree=12)
    st.add_layer(40e-9, segments=[(0.4, "M" if key else M), (0.6, 1.0)])
    st.add_layer(30e-9, eps=2.1)
    return st


def _fresh(n_sub, n_sup, theta):
    st = _stack(n_sub, n_sup, key=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(st.set_source(WL, theta=theta).solve()[3])


def _prep_solve(pr, theta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(pr.solve(wavelength=WL, materials={"M": M},
                                   theta=theta)[3])


def test_substrate_index_change_after_prepare():
    st = _stack()
    pr = st.prepare()
    J1 = _prep_solve(pr, 0.0)
    assert np.abs(J1 - _fresh(1.5, 1.0, 0.0)).max() < 1e-12
    st.n_sub = 2.5
    J2 = _prep_solve(pr, 0.0)
    assert np.abs(J2 - _fresh(2.5, 1.0, 0.0)).max() < 1e-12


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_superstrate_index_change_after_prepare(theta):
    """At oblique incidence the superstrate index also moves ``kx0``, which
    every LAYER's modes depend on."""
    st = _stack()
    pr = st.prepare()
    _prep_solve(pr, theta)
    st.n_sup = 1.33
    J = _prep_solve(pr, theta)
    assert np.abs(J - _fresh(1.5, 1.33, theta)).max() < 1e-12


def test_layer_added_after_prepare_is_refused():
    st = _stack()
    pr = st.prepare()
    _prep_solve(pr, 0.0)
    st.add_layer(20e-9, eps=3.0)
    with pytest.raises(ValueError, match="prepare"):
        _prep_solve(pr, 0.0)
