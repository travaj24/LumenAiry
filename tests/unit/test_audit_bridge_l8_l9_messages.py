"""Audit L-8 and L-9 (PLAN_DYNAMETA_BRIDGE_FINDINGS_2026_11_02, section 9):
two diagnostic-message fixes.  Neither changes any numerics.

L-9.  The hybrid cost guard (``twod._validate_cell_cost``) refuses a 128-px
rastered disk (r = 0.3 P) at degree 11: 45 x 45 strips, a 495 x 495 nodal
grid, 245025 DOF > 150000.  The refusal now names the route that fits a
curved inclusion -- ``PMM2DStackPure(...).add_layer(..., shapes=[Circle(...)])``
-- and the ``max_nodal_dof`` this layout would need.  Raise type and
behaviour are unchanged.

L-8.  ``twod_staggered._stag_map_nodes`` warned "moment error 5.00e-01; the
operators carry a quadrature error of that order" when a bar's horizontal
walls sit exactly on a circle's tangent points.  The wall line then squeezes
the cell between it and the arc to zero height at the tangent point; ``det
J`` vanishes quadratically along the line ``u = u_tangent`` through the
cell, so ``1 / det J`` and ``g11 / det J`` are NOT integrable and their
moments double with every doubling of the node count (measured: max |moment|
of g11 / det J 253, 508, 1016, 2034, 4068 at n = 18 ... 288), which makes
the relative change read exactly 0.5 at every count.  R / T converge in the
node count anyway (algebraically: R_y 1.88950e-3, 1.88970e-3, 1.88984e-3,
1.88994e-3 at n = 36, 72, 144, 288 -- ~1e-7, two decades under the n_modes
5 -> 7 discretisation error of 1e-5), so the number is a growth artefact,
not the operators' error.  The warning now names the pinched cells, reports
the moment error of the rest of the map (5.65e-14 here), and gives the
remedy.  A genuine unresolved map keeps the old wording.

The R / T pins below were computed on unmodified 5.50.0 (4c449fc1) in a
separate process; on that machine the post-change values are bit-identical.
The pins are compared at 5e-9 absolute, not bitwise, because a cross-platform
BLAS moves curved solves by up to ~1e-9 (test_pmm2d_staggered_curved_b's
round-off floor); 5e-9 still catches a changed node count (144 -> 288 moves
R_x by 1.8e-8 and R_y by 9.6e-8).  The same-process stub comparison is
bitwise.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle, PMM2DStackHybrid, PMM2DStackPure, Rect)
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.shapes2d import compile_shapes  # noqa: E402

_P = 600e-9
_LAM = 1.31e-6
_OLD = "the operators carry a quadrature error of that order"


# --------------------------------------------------------------------------- #
# L-9
# --------------------------------------------------------------------------- #

def test_l9_hybrid_refusal_points_to_pure_shapes_and_needed_dof():
    S = 128
    x = (np.arange(S) + 0.5) / S * _P
    X, Y = np.meshgrid(x, x, indexing="ij")
    cell = np.where(np.hypot(X - _P / 2, Y - _P / 2) <= 0.3 * _P, 4.0,
                    1.0).astype(complex)
    st = PMM2DStackHybrid(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                          degree=11, n_orders=3)
    with pytest.raises(ValueError) as ei:
        st.add_layer(250e-9, eps_cell=cell)
    msg = str(ei.value)
    assert "495 x 495" in msg
    assert "max_nodal_dof=150000" in msg
    assert "max_nodal_dof >= 245025" in msg
    assert "PMM2DStackPure" in msg and "shapes=" in msg
    assert "Circle(" in msg
    # the old routes are still offered
    assert "rcwa_efficiency_2d" in msg and "pmm_efficiency_2d_staggered" in msg


# --------------------------------------------------------------------------- #
# L-8
# --------------------------------------------------------------------------- #

def _cusp_shapes(dh=0.0):
    return [Circle(0.5 * _P, 0.5 * _P, 0.3 * _P, eps=4.0),
            Rect(0.82 * _P + 30e-9, 0.5 * _P, 60e-9, 0.6 * _P + dh, eps=4.0)]


def _map_nodes(shapes, M):
    _ec, xw, yw, cm = compile_shapes(_P, _P, shapes, 1.0)
    bx = TS.Basis1D(_P, np.asarray(xw), M)
    by = TS.Basis1D(_P, np.asarray(yw), M)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        n = TS._stag_map_nodes(bx, by, cm, M)
    return n, [str(x.message) for x in w]


def _solve(shapes):
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                        n_modes=5, n_orders=3)
    st.add_layer(250e-9, background_eps=1.0, shapes=shapes)
    st.set_source(_LAM)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _o, R, T = st.solve(jones=False)
    return np.asarray(R), np.asarray(T), [str(x.message) for x in w]


def test_l8_cusp_warns_with_pinch_wording():
    n, msgs = _map_nodes(_cusp_shapes(), 5)
    assert n == 144                          # unchanged: still the cap
    assert len(msgs) == 1, msgs
    m = msgs[0]
    assert "pinches a cell" in m
    assert "cell (1, 1)" in m and "cell (1, 3)" in m
    assert "tangent point" in m
    assert "1e-3 of the period" in m and "accept the result" in m
    assert "NOT the error of the operators" in m
    assert "(moment error 5.00e-01)" not in m   # the old phrasing
    assert "raw moment error 5.00e-01" in m      # reported as raw growth
    assert _OLD not in m
    # the rest of the map is resolved, and the message says so
    rest = float(m.split("rest of the map reads a moment error of ")[1]
                 .split(" ")[0])
    assert rest <= TS._STAG_MAP_QUAD_TOL


def test_l8_remedy_clears_the_warning():
    # walls 1e-3 P outside the tangent: no cusp, no warning, same node count
    n, msgs = _map_nodes(_cusp_shapes(dh=2e-3 * _P), 5)
    assert n == 144 and msgs == []


def test_l8_genuine_unresolved_map_keeps_old_wording():
    # a 0.3 nm gap between a circle and a bar: a sharp map, no pinched cell
    r = 0.3 * _P
    x_l = 0.5 * _P + r + 0.0005 * _P
    n, msgs = _map_nodes([Circle(0.5 * _P, 0.5 * _P, r, eps=4.0),
                          Rect(x_l + 0.025 * _P, 0.5 * _P, 0.05 * _P,
                               0.9 * _P, eps=4.0)], 6)
    assert n == 160
    assert len(msgs) == 1, msgs
    assert "not resolved" in msgs[0] and _OLD in msgs[0]
    assert "pinches" not in msgs[0]
    # the near-tangent case (walls 0.5 nm outside): a real, small error that
    # keeps the old wording too (R / T move ~1e-11 from 144 to 288 nodes)
    n, msgs = _map_nodes(_cusp_shapes(dh=1e-9), 5)
    assert n == 144 and len(msgs) == 1
    assert _OLD in msgs[0] and "pinches" not in msgs[0]


#: R / T (x-pol, y-pol) of the specular order, computed on 5.50.0 (4c449fc1)
_PIN = {
    "cusp": ((0.003854974182735034, 0.001889841533819107),
             (0.9961455224786062, 0.9981103159054555)),
    "ordinary": ((0.00048677935438394147, 0.0004867793543847394),
                 (0.9995135091490596, 0.9995135091490628)),
}


def test_l8_rt_unchanged_by_the_diagnostic():
    R, T, msgs = _solve(_cusp_shapes())
    assert any("pinches a cell" in m for m in msgs)
    assert not any(_OLD in m for m in msgs)
    # The pins were taken on Windows.  On the CI Linux runners the cusp case
    # reads T_y 6.0e-9 off (run 41646862421): its map quadrature is NOT
    # integrable at the pinched cell (moment error 0.5 at every node count),
    # which amplifies BLAS round-off far beyond the ordinary case's -- so the
    # cross-platform pin carries 5e-8 here.  The claim of this test, that the
    # diagnostic leaves R / T untouched, is the BITWISE in-process check below.
    np.testing.assert_allclose(R[:, 24], _PIN["cusp"][0], rtol=0, atol=5e-8)
    np.testing.assert_allclose(T[:, 24], _PIN["cusp"][1], rtol=0, atol=5e-8)
    # bitwise in-process: the same solve with the node count forced (no
    # diagnostic at all) gives the same bits
    orig = TS._stag_map_nodes
    try:
        TS._stag_map_nodes = lambda *a, **k: 144
        R2, T2, _ = _solve(_cusp_shapes())
    finally:
        TS._stag_map_nodes = orig
    assert np.array_equal(R, R2) and np.array_equal(T, T2)


def test_l8_ordinary_case_unchanged_and_silent():
    R, T, msgs = _solve([Circle(0.5 * _P, 0.5 * _P, 0.3 * _P, eps=4.0)])
    assert not any("Granet2DTransverseE" in m for m in msgs)
    np.testing.assert_allclose(R[:, 24], _PIN["ordinary"][0], rtol=0,
                               atol=5e-9)
    np.testing.assert_allclose(T[:, 24], _PIN["ordinary"][1], rtol=0,
                               atol=5e-9)
