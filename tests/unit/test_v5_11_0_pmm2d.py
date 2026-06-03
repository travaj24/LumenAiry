"""Hybrid 2-D PMM (`pmm_efficiency_2d`) -- the 2-D analogue of rcwa_efficiency_2d
for a separable rectangular pillar.

Coverage:
  * vacuum is exact and degree-independent (T00=1, R=0, energy=1);
  * a dielectric pillar conserves energy and matches the rcwa li oracle on the
    0-th order (both at matched Fourier truncation);
  * C4v symmetry of a square pillar at normal incidence (TE x-orders == TM
    y-orders; the 0-order is polarization-independent);
  * order symmetry T(+/-1,0)=T(0,+/-1) at normal incidence;
  * input guards (odd-node-count, n_orders-vs-degree, polarization).

The method is a HYBRID (projected-nodal layer + analytic plane-wave regions): it
has a Fourier-truncation `n_orders` floor like the FMM -- it is NOT no-floor like
the 1-D PMM (see the module docstring).
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.rcwa import rcwa_efficiency_2d

PX = PY = 0.8e-6
DEPTH = 0.3e-6
WL = 0.633e-6
FILL = 0.5
X0, X1 = (1 - FILL) / 2 * PX, (1 + FILL) / 2 * PX
Y0, Y1 = (1 - FILL) / 2 * PY, (1 + FILL) / 2 * PY


def _odict(o, R, T):
    return {(int(a), int(b)): (float(R[i]), float(T[i]))
            for i, (a, b) in enumerate(o)}


# --------------------------------------------------------------------------- #
# vacuum: exact and degree-independent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("degree", [7, 9, 11])
def test_vacuum_is_exact(pol, degree):
    o, R, T, dof = la.pmm_efficiency_2d(
        PX, PY, 1.0, 1.0, (0.2 * PX, 0.6 * PX), (0.2 * PY, 0.6 * PY),
        1.0, 1.0, DEPTH, WL, degree=degree, polarization=pol,
        theta=0.0, n_orders=6)
    d = _odict(o, R, T)
    assert abs(d[(0, 0)][1] - 1.0) < 1e-9      # T00 == 1
    assert d[(0, 0)][0] < 1e-12                 # R00 == 0
    assert float(R.sum()) < 1e-12               # no reflection at all
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# dielectric pillar: energy + match the rcwa oracle at matched truncation
# --------------------------------------------------------------------------- #
def _rcwa_oracle(pol, n):
    S = 256
    xs = (np.arange(S) + 0.5) / S * PX
    ys = (np.arange(S) + 0.5) / S * PY
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cell = np.where((X >= X0) & (X <= X1) & (Y >= Y0) & (Y <= Y1),
                    2.25, 1.0).astype(complex)
    orc = rcwa_efficiency_2d(PX, PY, cell, 1.0, 1.0, DEPTH, WL,
                             theta=1e-4, phi=0.0, polarization=pol,
                             n_orders_x=n, n_orders_y=n, formulation="li")
    return _odict(orc[0], orc[1], orc[2])


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pillar_matches_rcwa_oracle(pol):
    # both at n_orders = 9 (matched Fourier truncation); the geometry-conforming
    # nodal layer agrees with the staircased-Fourier FMM to a few 1e-3.
    od = _rcwa_oracle(pol, 9)
    o, R, T, _ = la.pmm_efficiency_2d(
        PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1), 1.0, 1.0, DEPTH, WL,
        degree=11, polarization=pol, theta=0.0, n_orders=9)
    d = _odict(o, R, T)
    assert abs(d[(0, 0)][1] - od[(0, 0)][1]) < 3e-3
    assert abs(d[(0, 0)][0] - od[(0, 0)][0]) < 3e-3
    assert abs(float(R.sum() + T.sum()) - 1.0) < 5e-3


@pytest.mark.parametrize("degree", [9, 11, 13])
def test_pillar_conserves_energy(degree):
    o, R, T, _ = la.pmm_efficiency_2d(
        PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1), 1.0, 1.0, DEPTH, WL,
        degree=degree, polarization="te", theta=0.0, n_orders=9)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 5e-3


# --------------------------------------------------------------------------- #
# symmetry invariants (oracle-free)
# --------------------------------------------------------------------------- #
def test_square_pillar_c4v_symmetry():
    # a SQUARE pillar at normal incidence has C4v symmetry: rotating the linear
    # polarization 90deg (te<->tm) swaps the x- and y-diffraction orders, and the
    # specular 0-order is polarization-independent.
    args = dict(degree=11, theta=0.0, n_orders=11)
    oe, Re, Te, _ = la.pmm_efficiency_2d(
        PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1), 1.0, 1.0, DEPTH, WL,
        polarization="te", **args)
    om, Rm, Tm, _ = la.pmm_efficiency_2d(
        PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1), 1.0, 1.0, DEPTH, WL,
        polarization="tm", **args)
    de, dm = _odict(oe, Re, Te), _odict(om, Rm, Tm)
    # specular order polarization-independent
    assert abs(de[(0, 0)][1] - dm[(0, 0)][1]) < 3e-3
    assert abs(de[(0, 0)][0] - dm[(0, 0)][0]) < 3e-3
    # TE's (1,0) == TM's (0,1) (x<->y swap under the 90deg pol rotation)
    assert abs(de[(1, 0)][1] - dm[(0, 1)][1]) < 3e-3
    assert abs(de[(0, 1)][1] - dm[(1, 0)][1]) < 3e-3


def test_normal_incidence_order_symmetry():
    # at normal incidence the +order and -order efficiencies are equal
    o, R, T, _ = la.pmm_efficiency_2d(
        PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1), 1.0, 1.0, DEPTH, WL,
        degree=11, polarization="te", theta=0.0, n_orders=11)
    d = _odict(o, R, T)
    assert abs(d[(1, 0)][1] - d[(-1, 0)][1]) < 1e-6
    assert abs(d[(0, 1)][1] - d[(0, -1)][1]) < 1e-6
    assert abs(d[(1, 1)][1] - d[(-1, -1)][1]) < 1e-6


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_rejects_even_node_count():
    with pytest.raises(ValueError, match="ODD"):
        la.pmm_efficiency_2d(PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1),
                             1.0, 1.0, DEPTH, WL, degree=8, n_orders=6)


def test_rejects_too_many_orders():
    with pytest.raises(ValueError, match="too large"):
        la.pmm_efficiency_2d(PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1),
                             1.0, 1.0, DEPTH, WL, degree=7, n_orders=20)


def test_rejects_bad_polarization():
    with pytest.raises(ValueError, match="polarization"):
        la.pmm_efficiency_2d(PX, PY, 2.25, 1.0, (X0, X1), (Y0, Y1),
                             1.0, 1.0, DEPTH, WL, degree=7, n_orders=6,
                             polarization="circular")


def test_pmm_efficiency_2d_exported():
    assert hasattr(la, "pmm_efficiency_2d")
    assert "pmm_efficiency_2d" in la.__all__
