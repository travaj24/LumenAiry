"""v5.14.0 -- 2-D anisotropic hybrid PMM (``pmm_jones_2d``).

The PMM mirror of ``rcwa_jones_2d``: full in-plane (3,3) tensor cells on the
exact-wall hybrid pipeline, via the SHARED dimension-agnostic tensor eigenmode
solve (``rcwa._core._layer_eigenmodes_tensor``) fed with PMM projected nodal
operators.  Pins:

* a SCALAR tensor cell reduces to ``pmm_efficiency_2d_cell('laurent')`` to
  numerical precision (same Q/P blocks -> same eig);
* an anisotropic (rotated-director LC) cell agrees with the independent
  ``rcwa_jones_2d`` and conserves energy for a lossless tensor;
* a UNIFORM anisotropic cell matches the validated 1-D ``pmm_jones_1d`` at
  duty=1 (the Berreman-grade oracle);
* a lossy tensor absorbs (no silent gain);
* out-of-plane entries raise (Phase-6 scope), bad shapes raise.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_efficiency_2d_cell,
    pmm_jones_1d,
    pmm_jones_2d,
)
from lumenairy.elements.rcwa import rcwa_jones_2d

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6
_S = 12


def _scalar_tensor_cell(eps_pillar=6.0 + 0j, eps_host=1.0 + 0j):
    sc = np.full((_S, _S), eps_host)
    sc[3:9, 4:10] = eps_pillar
    tc = np.zeros((_S, _S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = sc
    return sc, tc

def _lc_cell(rot=0.7, loss=0.0):
    """Host 2.25; pillar = in-plane uniaxial director rotated by ``rot``."""
    tc = np.zeros((_S, _S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = 1.5 ** 2 + loss * 1j, 1.7 ** 2 + loss * 1j
    c, s = np.cos(rot), np.sin(rot)
    tc[3:9, 4:10, 0, 0] = ne2 * c * c + no2 * s * s
    tc[3:9, 4:10, 1, 1] = ne2 * s * s + no2 * c * c
    tc[3:9, 4:10, 0, 1] = (ne2 - no2) * c * s
    tc[3:9, 4:10, 1, 0] = (ne2 - no2) * c * s
    tc[3:9, 4:10, 2, 2] = no2
    return tc


def _o0(o):
    return (o[:, 0] == 0) & (o[:, 1] == 0)


# --------------------------------------------------------------------------- #
# (1) scalar-tensor reduction to the scalar laurent path
# --------------------------------------------------------------------------- #

def test_scalar_tensor_reduces_to_scalar_laurent():
    sc, tc = _scalar_tensor_cell()
    o_j, R_j, T_j, _J = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                     degree=7, n_orders=4)
    for row, pol in ((1, "te"), (0, "tm")):     # row1 = incident Ey = TE
        o_s, R_s, T_s = pmm_efficiency_2d_cell(
            _P, _P, sc, 1.5, 1.0, _DEP, _WL, degree=7, n_orders=4,
            polarization=pol, formulation="laurent")
        assert np.array_equal(o_j, o_s)
        assert np.max(np.abs(R_j[row] - R_s)) < 1e-11
        assert np.max(np.abs(T_j[row] - T_s)) < 1e-11


# --------------------------------------------------------------------------- #
# (2) anisotropic cell vs rcwa_jones_2d + energy
# --------------------------------------------------------------------------- #

def test_lc_cell_vs_rcwa_jones_2d_and_energy():
    tc = _lc_cell()
    o1, R1, T1, J1 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=9, n_orders=4)
    # lossless tensor -> energy conserved per incident polarization
    for row in (0, 1):
        assert abs(float(R1[row].sum() + T1[row].sum()) - 1.0) < 2e-2
    # rcwa oracle on an exact 2x upsampling of the same geometry
    up = np.zeros((2 * _S, 2 * _S, 3, 3), complex)
    for a in range(3):
        for b in range(3):
            up[:, :, a, b] = np.kron(tc[:, :, a, b], np.ones((2, 2)))
    o2, R2, T2, J2 = rcwa_jones_2d(_P, _P, up, 1.5, 1.0, _DEP, _WL,
                                   n_orders_x=4, n_orders_y=4)
    m1, m2 = _o0(o1), _o0(o2)
    for row in (0, 1):
        assert abs(float(T1[row][m1][0]) - float(T2[row][m2][0])) < 2e-2
    # cross-polarization Jones present and consistent across suites
    assert np.max(np.abs(J1 - J2)) < 5e-2
    assert abs(J1[1, 0]) > 1e-3                  # genuine cross-pol coupling


# --------------------------------------------------------------------------- #
# (3) uniform anisotropic cell vs the validated 1-D solver (duty=1)
# --------------------------------------------------------------------------- #

def test_uniform_tensor_cell_matches_pmm_jones_1d():
    eps = np.array([[2.9, 0.25, 0.0],
                    [0.25, 2.4, 0.0],
                    [0.0, 0.0, 2.1]], dtype=complex)
    tc = np.zeros((4, 4, 3, 3), complex)
    tc[:, :] = eps
    o2, R2, T2, J2 = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                                  degree=7, n_orders=2)
    # 1-D oracle: a uniform "grating" (ridge == groove) at the same depth
    o1, R1, T1, J1 = pmm_jones_1d(_P, eps, eps, 1.5, 1.0, _DEP, 0.5, _WL,
                                  degree=10, stabilize=False)
    m2, m1 = _o0(o2), o1 == 0
    for row in (0, 1):
        assert abs(float(R2[row][m2][0]) - float(R1[row][m1][0])) < 1e-8
        assert abs(float(T2[row][m2][0]) - float(T1[row][m1][0])) < 1e-8
    assert np.max(np.abs(J2 - J1)) < 1e-8


# --------------------------------------------------------------------------- #
# (4) lossy tensor absorbs; guards
# --------------------------------------------------------------------------- #

def test_lossy_tensor_absorbs_not_gains():
    tc = _lc_cell(loss=0.5)
    o, R, T, _J = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL,
                               degree=9, n_orders=4)
    for row in (0, 1):
        tot = float(R[row].sum() + T[row].sum())
        assert tot < 1.0 + 2e-2
        assert tot < 0.98


def test_guards():
    _sc, tc = _scalar_tensor_cell()
    with pytest.raises(ValueError, match="Sx, Sy, 3, 3"):
        pmm_jones_2d(_P, _P, np.ones((4, 4), complex), 1.5, 1.0, _DEP, _WL)
    with pytest.raises(ValueError, match="formulation"):
        pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, formulation="bogus")
    with pytest.raises(ValueError, match="e_zz"):
        bad = tc.copy()
        bad[3:9, 4:10, 2, 2] = 0.0               # zero e_zz region
        pmm_jones_2d(_P, _P, bad, 1.5, 1.0, _DEP, _WL, degree=7, n_orders=4)
