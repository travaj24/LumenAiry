"""v5.14.0 -- multilayer 2-D hybrid PMM (``PMM2DStack``).

The 2-D PMM counterpart of ``RCWAStack``: uniform films + patterned scalar
cells + in-plane tensor cells cascaded through the Redheffer S-matrix in the
shared Rayleigh basis (each patterned layer keeps its OWN exact-wall SEM grid;
no union-grid constraint).  Pins:

* a single-layer stack reproduces the direct single-layer solvers;
* splitting a uniform film into two half-thickness layers is an identity;
* a two-patterned-layer stack agrees with ``RCWAStack`` on the same geometry;
* the tapered pillar's vertical limit reproduces the straight pillar, and a
  true taper conserves energy;
* the wavelength sweep matches per-wavelength solves; builder validation.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStack,
    pmm_efficiency_2d_cell,
    pmm_jones_2d,
)
from lumenairy.elements.rcwa import RCWAStack

_P = 0.6e-6
_WL = 0.55e-6


def _pillar_cell(S=6, lo=1, hi=4, eps=6.0 + 0j, host=1.0 + 0j):
    cell = np.full((S, S), host)
    cell[lo:hi, lo:hi] = eps
    return cell


def _o0(o):
    return (o[:, 0] == 0) & (o[:, 1] == 0)


# --------------------------------------------------------------------------- #
# (1) single-layer stack == direct solvers
# --------------------------------------------------------------------------- #

def test_single_scalar_layer_matches_direct():
    cell = _pillar_cell()
    st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    st.add_layer(0.25e-6, eps_cell=cell)
    o_s, R_s, T_s, _j = st.set_source(_WL).solve()
    # direct: TE drives Ey (row 1), TM drives Ex (row 0)
    for row, pol in ((1, "te"), (0, "tm")):
        o_d, R_d, T_d = pmm_efficiency_2d_cell(
            _P, _P, cell, 1.5, 1.0, 0.25e-6, _WL, degree=9, n_orders=4,
            polarization=pol)
        assert np.array_equal(o_s, o_d)
        assert np.max(np.abs(R_s[row] - R_d)) < 1e-11
        assert np.max(np.abs(T_s[row] - T_d)) < 1e-11


def test_single_tensor_layer_matches_pmm_jones_2d():
    S = 8
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = 2.25
    no2, ne2 = 1.5 ** 2, 1.7 ** 2
    c, s = np.cos(0.7), np.sin(0.7)
    tc[2:6, 3:7, 0, 0] = ne2 * c * c + no2 * s * s
    tc[2:6, 3:7, 1, 1] = ne2 * s * s + no2 * c * c
    tc[2:6, 3:7, 0, 1] = (ne2 - no2) * c * s
    tc[2:6, 3:7, 1, 0] = (ne2 - no2) * c * s
    tc[2:6, 3:7, 2, 2] = no2
    st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4, formulation="laurent")
    st.add_layer(0.25e-6, eps_tensor_cell=tc)
    o_s, R_s, T_s, J_s = st.set_source(_WL).solve()
    o_d, R_d, T_d, J_d = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, 0.25e-6, _WL,
                                      degree=9, n_orders=4,
                                      formulation="laurent")
    assert np.max(np.abs(R_s - R_d)) < 1e-11
    assert np.max(np.abs(T_s - T_d)) < 1e-11
    assert np.max(np.abs(J_s - J_d)) < 1e-11


# --------------------------------------------------------------------------- #
# (2) uniform-film split identity + RCWAStack cross-check
# --------------------------------------------------------------------------- #

def test_uniform_film_split_is_identity():
    cell = _pillar_cell()
    def build(split):
        st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                        n_orders=4)
        st.add_layer(0.2e-6, eps_cell=cell)
        if split:
            st.add_layer(0.1e-6, eps=2.25).add_layer(0.1e-6, eps=2.25)
        else:
            st.add_layer(0.2e-6, eps=2.25)
        return st.set_source(_WL).solve()
    o1, R1, T1, J1 = build(False)
    o2, R2, T2, J2 = build(True)
    assert np.max(np.abs(R1 - R2)) < 1e-10
    assert np.max(np.abs(T1 - T2)) < 1e-10
    assert np.max(np.abs(J1 - J2)) < 1e-10


def test_two_patterned_layers_vs_rcwastack():
    cell_a = _pillar_cell(eps=6.0)             # pillar layer
    cell_b = np.full((6, 6), 4.0 + 0j)
    cell_b[:, 2:5] = 1.0                       # stripe layer (different walls)
    st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    st.add_layer(0.15e-6, eps_cell=cell_a)
    st.add_layer(0.10e-6, eps_cell=cell_b)
    o1, R1, T1, J1 = st.set_source(_WL).solve()
    for row in (0, 1):
        assert abs(float(R1[row].sum() + T1[row].sum()) - 1.0) < 2e-2
    rc = RCWAStack(_P, period_y=_P, n_substrate=1.5, n_superstrate=1.0,
                   n_orders=4, n_orders_y=4)
    rc.add_layer(0.15e-6, eps_cell=np.kron(cell_a, np.ones((5, 5))))
    rc.add_layer(0.10e-6, eps_cell=np.kron(cell_b, np.ones((5, 5))))
    res = rc.set_source(_WL, theta=0.0).solve()
    o2, R2, T2 = res.efficiencies()
    m1, m2 = _o0(o1), _o0(o2)
    for row in (0, 1):
        assert abs(float(T1[row][m1][0]) - float(T2[row][m2][0])) < 2e-2


# --------------------------------------------------------------------------- #
# (3) tapered pillar
# --------------------------------------------------------------------------- #

def test_tapered_vertical_limit_matches_straight_pillar():
    bounds = (0.15 * _P, 0.65 * _P)
    st_t = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                      n_orders=4)
    st_t.add_tapered_pillar(0.24e-6, eps_pillar=6.0, eps_host=1.0,
                            x_bounds_bottom=bounds, y_bounds_bottom=bounds,
                            n_slices=4)
    o_t, R_t, T_t, J_t = st_t.set_source(_WL).solve()
    st_s = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                      n_orders=4)
    st_s.add_tapered_pillar(0.24e-6, eps_pillar=6.0, eps_host=1.0,
                            x_bounds_bottom=bounds, y_bounds_bottom=bounds,
                            n_slices=1)
    o_s, R_s, T_s, J_s = st_s.set_source(_WL).solve()
    assert np.max(np.abs(R_t - R_s)) < 1e-10   # identical slices collapse
    assert np.max(np.abs(T_t - T_s)) < 1e-10


def test_true_taper_conserves_energy():
    st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    st.add_tapered_pillar(0.24e-6, eps_pillar=6.0, eps_host=1.0,
                          x_bounds_bottom=(0.1 * _P, 0.7 * _P),
                          y_bounds_bottom=(0.1 * _P, 0.7 * _P),
                          x_bounds_top=(0.25 * _P, 0.55 * _P),
                          y_bounds_top=(0.25 * _P, 0.55 * _P),
                          n_slices=5)
    o, R, T, _j = st.set_source(_WL).solve()
    for row in (0, 1):
        assert abs(float(R[row].sum() + T[row].sum()) - 1.0) < 3e-2


# --------------------------------------------------------------------------- #
# (4) sweep + validation
# --------------------------------------------------------------------------- #

def test_sweep_matches_per_wavelength():
    cell = _pillar_cell()
    st = PMM2DStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=9,
                    n_orders=4)
    st.add_layer(0.2e-6, eps_cell=cell).add_layer(0.1e-6, eps=2.25)
    wls = (0.5e-6, 0.6e-6)
    o_sw, R_sw, T_sw = st.solve_vs_wavelength(wls)
    for i, w in enumerate(wls):
        o1, R1, T1, _ = st.set_source(float(w)).solve()
        assert np.array_equal(o_sw, o1)
        assert np.allclose(R_sw[i], R1, rtol=0, atol=0)
        assert np.allclose(T_sw[i], T1, rtol=0, atol=0)


def test_builder_validation():
    st = PMM2DStack(_P)
    with pytest.raises(ValueError, match="exactly ONE"):
        st.add_layer(0.1e-6)
    with pytest.raises(ValueError, match="exactly ONE"):
        st.add_layer(0.1e-6, eps=2.0, eps_cell=np.ones((4, 4)))
    with pytest.raises(ValueError, match="thickness"):
        st.add_layer(-1e-6, eps=2.0)
    with pytest.raises(ValueError, match="set_source"):
        PMM2DStack(_P).add_layer(0.1e-6, eps=2.0).solve()
    with pytest.raises(ValueError, match="at least one layer"):
        PMM2DStack(_P).set_source(_WL).solve()
    with pytest.raises(ValueError, match="formulation"):
        PMM2DStack(_P, formulation="bogus")
    with pytest.raises(NotImplementedError, match="out-of-plane"):
        bad = np.zeros((4, 4, 3, 3), complex)
        for i in range(3):
            bad[:, :, i, i] = 2.0
        bad[1:3, 1:3, 0, 2] = 0.3
        PMM2DStack(_P).add_layer(0.1e-6, eps_tensor_cell=bad)
