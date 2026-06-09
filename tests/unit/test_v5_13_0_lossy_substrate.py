"""v5.13.0 -- P1-A fix: transmittance into a LOSSY EXIT SUBSTRATE.

RCWA (and the hybrid PMM-2D) ran the modal solve in the internal exp(+iwt)
convention (conjugated region eps); ``_sqrt_forward`` (Im(kz) >= 0) then returned
``Re(kz) < 0`` for a lossy substrate, and the ``Re(kz) > 0`` propagating mask
SILENTLY ZEROED the transmittance into ANY absorbing exit medium (R was fine).
The fix computes the z-flux weight, the propagating mask, and the longitudinal
field from the PUBLIC-convention forward kz (``rcwa._core._forward_flux_kz`` /
un-conjugating the region eps); lossless eps is real so the fix is byte-identical.

These tests pin that the transmittance into a lossy substrate is now physical
(matches analytic Fresnel for a bare interface, NOT zero) across the affected
solvers, and that energy is conserved.  The staggered PMM (public convention) was
already correct and is included as an independent cross-check.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_efficiency_2d
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
)

_P = 0.5e-6
_WL = 0.55e-6
_DEP = 0.02e-6   # thin near-vacuum layer -> ~bare interface


def _fresnel(n2, n1=1.0, theta=0.0, pol="te"):
    c1 = np.cos(theta)
    c2 = np.sqrt(1.0 - (n1 * np.sin(theta) / n2) ** 2)
    if pol == "te":                                   # s-pol
        r = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        t = 2.0 * n1 * c1 / (n1 * c1 + n2 * c2)
        flux = np.real(n2 * c2) / (n1 * c1)
    else:                                             # p-pol (TM)
        r = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        t = 2.0 * n1 * c1 / (n2 * c1 + n1 * c2)
        flux = np.real(np.conj(n2) * c2) / (n1 * c1)
    return float(abs(r) ** 2), float(flux * abs(t) ** 2)


@pytest.mark.parametrize("n2", [1.5 + 0.001j, 1.5 + 0.1j, 1.5 + 0.5j, 4.0 + 2.0j])
def test_eff_1d_lossy_substrate_matches_fresnel(n2):
    """rcwa_efficiency_1d into a lossy substrate: T is the physical Fresnel value
    (NOT zeroed), R+T = 1 for the (effectively) bare interface."""
    o, R, T = rcwa_efficiency_1d(_P, 1.0, 1.0, n2, 1.0, _DEP, 0.5, _WL,
                                 n_orders=5, polarization="te")
    Rf, Tf = _fresnel(n2)
    assert float(T.sum()) > 0.1                       # the bug returned 0
    assert abs(float(T.sum()) - Tf) < 5e-3
    assert abs(float(R.sum() + T.sum()) - 1.0) < 5e-3


def test_eff_1d_oblique_tm_lossy_substrate():
    n2 = 1.5 + 0.2j
    o, R, T = rcwa_efficiency_1d(_P, 1.0, 1.0, n2, 1.0, _DEP, 0.5, _WL,
                                 n_orders=8, angle=0.5, polarization="tm")
    Rf, Tf = _fresnel(n2, theta=0.5, pol="tm")
    assert abs(float(T.sum()) - Tf) < 1e-2
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-2


def test_eff_2d_lossy_substrate_matches_fresnel():
    n2 = 1.5 + 0.3j
    o, R, T = rcwa_efficiency_2d(_P, _P, np.full((8, 8), 1.0 + 0j), n2, 1.0,
                                 _DEP, _WL, n_orders_x=3, n_orders_y=3)[:3]
    Rf, Tf = _fresnel(n2)
    assert float(np.sum(T)) > 0.1
    assert abs(float(np.sum(T)) - Tf) < 5e-3
    assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 5e-3


def test_jones_1d_lossy_substrate_nonzero_and_conserves():
    n2 = 1.5 + 0.2j
    I3 = np.eye(3)
    o, R, T, _j = rcwa_jones_1d(_P, 1.0 * I3, 1.0 * I3, n2, 1.0, _DEP, 0.5, _WL,
                                n_orders=5)
    # R, T are (2, M) -- one row per incident polarization, each conserves
    for pol in (0, 1):
        assert float(np.sum(T[pol])) > 0.1
        assert abs(float(np.sum(R[pol]) + np.sum(T[pol])) - 1.0) < 5e-3


def test_stack_lossy_substrate_nonzero_and_conserves():
    n2 = 1.5 + 0.3j
    st = RCWAStack(period=_P, n_superstrate=1.0, n_substrate=n2, n_orders=5)
    st.add_layer(_DEP, eps=1.0)
    o, R, T = st.set_source(_WL, theta=0.0).solve().efficiencies()
    for pol in (0, 1):
        assert float(np.sum(T[pol])) > 0.1
        assert abs(float(np.sum(R[pol]) + np.sum(T[pol])) - 1.0) < 5e-3


def test_pmm_hybrid_lossy_substrate_nonzero_and_conserves():
    """The hybrid PMM-2D shared the bug (it conjugates eps); fixed too."""
    n2 = 1.5 + 0.3j
    o, R, T = pmm_efficiency_2d(_P, _P, 2.25, 2.25, (0.2 * _P, 0.6 * _P),
                                (0.2 * _P, 0.6 * _P), n2, 1.0, _DEP, _WL,
                                degree=9, n_orders=3)[:3]
    # uniform 2.25 layer into a lossy substrate -- T must be nonzero, R+T ~ 1
    assert float(np.sum(T)) > 0.1
    assert float(np.sum(R) + np.sum(T)) < 1.05
