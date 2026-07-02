"""Audit 2026-07-01 P1-05 -- ``pmm_efficiency_2d_staggered`` incidence guard.

Pre-fix, the staggered entry point (unlike every other efficiency entry point
since the v5.14.1 suite-wide fix) never called
``rcwa._core._require_propagating_incidence``: a gain superstrate
(``Im(n_sup) < 0``, e.g. a -1e-9 rounding artifact from a fitted index)
flipped the forward ``kz_inc`` root and silently NEGATED every transmitted
efficiency (measured ``sum(T) = -144.6``), and an evanescent/metallic
superstrate silently returned ``tot = 41.2``.  These tests FAIL on the
pre-fix code (no ValueError was raised; verified by A/B on the unfixed
tree) and pass with the guard in place.  The baseline pins prove the guard
is inert for valid (lossless AND lossy-superstrate) inputs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered

_G = dict(period_x=1.2e-6, period_y=1.2e-6, depth=0.3e-6, wavelength=1.0e-6,
          degree=5, n_orders=3)
_PILLAR = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)


def test_gain_superstrate_raises():
    # Im(n_sup) < 0 (public convention) flips the forward-root kz_inc; must
    # raise, matching pmm_efficiency_2d_cell, instead of returning sum(T)<0.
    with pytest.raises(ValueError, match="gain incidence medium"):
        pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.5, n_superstrate=1.0 - 1e-6j, **_G)


def test_evanescent_superstrate_raises():
    # Metallic superstrate: Re(eps_sup) < 0 -> the incident wave is evanescent
    # and kz_inc ~ 0 divides the flux normalization; must raise.
    with pytest.raises(ValueError, match="non-propagating"):
        pmm_efficiency_2d_staggered(
            eps_cell=_PILLAR, n_substrate=1.5, n_superstrate=0.1 + 3.0j, **_G)


def test_lossless_baseline_unchanged():
    # Valid input path must be untouched by the guard (A/B verified
    # byte-identical pre/post fix); pin the energy total.
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.5, n_superstrate=1.0, **_G)
    tot = float(np.sum(R) + np.sum(T))
    assert abs(tot - 1.0) < 1e-3
    np.testing.assert_allclose(tot, 0.999776592219514, rtol=1e-9)


def test_lossy_superstrate_still_allowed():
    # Im(n_sup) > 0 (loss, public convention) remains supported -- the conj
    # into the guard's internal convention must NOT misread loss as gain.
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=_PILLAR, n_substrate=1.5, n_superstrate=1.0 + 0.01j,
        polarization="te", theta=np.deg2rad(10.0), **_G)
    tot = float(np.sum(R) + np.sum(T))
    assert 0.0 < tot <= 1.0
