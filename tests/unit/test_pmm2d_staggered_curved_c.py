"""CURVED-CELL MAP, Phase C, for the PURE staggered 2-D PMM: the SHAPE
PRIMITIVES (``Rect``, ``FilletRect``, ``Circle``, ``Ellipse``,
``SinusoidalWall``), ``compile_shapes``, the stack-level merge of every shape
layer into ONE map (``PMM2DStackPure.add_layer(shapes=...)``), the
single-layer convenience (``pmm_jones_2d_staggered(shapes=...)``), the exact
modal decomposition of the incident wave under a map, and the viewers drawing
curved cells -- gates C1-C16 of
``docs/audits/BUILD_PMM2D_CURVED_C_2026_10_02.md`` (plan
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.3).

Words.  A SHAPE PRIMITIVE takes physical geometry and lays out the solver's
``(u, v)`` wall grid and coordinate map so its outline is an exact grid line
(every corner, 45-degree point and tangency point a grid vertex).  The STACK
merges every layer's shapes into one wall grid and one map.  A FAIL-BEFORE
arm is a deliberately broken variant that must fail the bar.

Fixture (the planning probes' P3 / P4 fixture): lambda = 1, square period
1.2, depth 0.5, air over n = 1.45, eps 4 (n = 2) features, normal incidence
unless stated; the circle r = 0.36; the fillet pillar side 0.6.  Row 0 of
R / T is the input E along x, row 1 E along y.

EVERY BAR is derived from a measurement made by this build on 2026-10-02
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_c/``), stated next to
the assertion with its gap on both sides.  Floors: after this phase's
incident fix the mapped solve's R / T move <= 2.3e-14 under a 1e-15
perturbation (C9), so the round-off floor sits far below every bar; the
discretisation numbers (rung changes, closures) are deterministic
quantities, and no bar sits within a decade of one of them.  Sizes: 3 x 3
grids at M <= 5 and 5 x 5 (or 7 x 7) at M = 4.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    compile_shapes,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import shapes2d as SH  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_P = 1.2
_WL = 1.0
_DEPTH = 0.5
_NSUP, _NSUB = 1.0, 1.45
_R = 0.36
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "..", "..", "validation", "probe_pmm2d_curved")


def _stack(layers, M, theta=0.0, phi=0.0, period=_P, n_orders=3,
           retain=False, cmap=None):
    """``layers``: [(t, spec)], spec = (shapes, background_eps) | eps scalar |
    ('cell', eps_cell)."""
    st = PMM2DStackPure(period, period, n_superstrate=_NSUP,
                        n_substrate=_NSUB, n_modes=M, n_orders=n_orders,
                        cmap=cmap)
    for t, spec in layers:
        if isinstance(spec, tuple) and spec[0] == "cell":
            st.add_layer(t, eps_cell=spec[1])
        elif isinstance(spec, tuple):
            st.add_layer(t, shapes=spec[0], background_eps=spec[1])
        else:
            st.add_layer(t, eps=spec)
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # fail-before arms trip closure
        o, R, T, J = st.solve(retain_internal=retain)
    return st, np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)


def _idx(o, orders=_ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def _vec(o, R, T):
    i = _idx(o)
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def _circle3_explicit():
    cm, _w = CM._circle_map_3x3(_P, _R)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    return cm, eps


def _airy_normal(n2=2.0):
    k0 = 2 * np.pi / _WL
    n = (_NSUP, n2, _NSUB)
    r01 = (n[0] - n[1]) / (n[0] + n[1])
    r12 = (n[1] - n[2]) / (n[1] + n[2])
    ph = np.exp(2j * n[1] * k0 * _DEPTH)
    return abs((r01 + r12 * ph) / (1 + r01 * r12 * ph)) ** 2


def _film_err(cmap, M):
    n = cmap.shape[0]
    _st, o, R, T, _J = _stack([(_DEPTH, ("cell", np.full((n, n), 4.0 + 0j)))],
                              M, cmap=cmap)
    Rx = _airy_normal()
    i0 = _idx(o, [(0, 0)])[0]
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rx
    T[:, i0] -= 1.0 - Rx
    return float(max(np.abs(R).max(), np.abs(T).max()))


# =========================================================================== #
# C1 -- no shape, no map: today's bytes; the shapes route IS an explicit map
# =========================================================================== #
def test_c1_no_shapes_never_reaches_the_phase_c_code(monkeypatch):
    """Without ``shapes=`` (and without ``cmap=``) the shipped code runs: every
    function Phase C added is booby-trapped and the unmapped entries must not
    touch one.  The byte identity itself is a build-doc measurement against
    ``git archive 91d00288`` (``c1_compare.json``: every SHA-256 of Phase B's
    fixture set -- operators of every dispatch branch, modes, far projectors,
    R / T / Jones, absorption -- identical); this is its build-free
    restatement."""
    def boom(*a, **k):
        raise AssertionError("Phase C code reached on the no-shape path")
    monkeypatch.setattr(SH, "_merge", boom)
    monkeypatch.setattr(SP, "_stag_incident_coeffs_mapped", boom)
    monkeypatch.setattr(TS, "_stag_incident_load_mapped", boom)
    monkeypatch.setattr(TS, "_stag_incident_coeffs_mapped", boom)
    monkeypatch.setattr(SP.PMM2DStackPure, "_recompile_shapes", boom)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0 + 0.2j
    st = PMM2DStackPure(_P, _P, n_substrate=_NSUB, n_modes=4, n_orders=2)
    st.add_layer(0.2, eps_cell=eps)
    st.add_layer(0.1, eps=2.25)
    st.set_source(_WL, theta=0.3, phi=0.2)
    st.solve(retain_internal=True)
    st.layer_absorption()
    pmm_jones_2d_staggered(_P, _P, eps, _NSUB, 1.0, 0.3, _WL, n_modes=4,
                           n_orders=2)
    pmm_efficiency_2d_staggered(_P, _P, eps, _NSUB, 1.0, 0.3, _WL, degree=4,
                                n_orders=2)


def test_c1_shapes_route_is_the_explicit_map_route_byte_for_byte():
    """A composite layer (a circular HOLE in a slab painted over an
    off-centre ellipse in the next layer) through ``add_layer(shapes=...)``
    and through ``compile_shapes`` + ``cmap=`` + ``eps_cell=`` must be the
    SAME bytes: the shape layer is the explicit-map machinery, nothing else.
    Measured 2026-10-02: equal to the bit.  Fail-before: a map whose one
    curve parameter differs by 1e-12 relative has a DIFFERENT fingerprint
    (the cache key that would otherwise serve a stale eig)."""
    slab = Rect(0.6, 0.6, 1.2, 1.2, 4.0)
    hole = Circle(0.6, 0.6, 0.3, 1.0)
    M = 4
    st, o, R, T, J = _stack([(0.3, ([slab, hole], 1.0))], M)
    eps, xw, yw, cm = compile_shapes(_P, _P, [slab, hole], 1.0)
    assert np.array_equal(st._layers[0]["eps_cell"], eps)
    assert st.cmap.fingerprint == cm.fingerprint
    _st2, o2, R2, T2, J2 = _stack([(0.3, ("cell", eps))], M, cmap=cm)
    for a, b in ((o, o2), (R, R2), (T, T2), (J, J2)):
        assert np.array_equal(a, b)
    _e, _x, _y, cm_b = compile_shapes(
        _P, _P, [slab, Circle(0.6, 0.6, 0.3 * (1 + 1e-12), 1.0)], 1.0)
    assert cm_b.fingerprint != cm.fingerprint


def test_c1_rectangles_ride_the_unmapped_solver():
    """Rectangles only give the IDENTITY map, and the stack then runs the
    SHIPPED unmapped solver on the rectangles' walls (no quadrature path):
    against the identity map through the mapped path the answer agrees to
    round-off at NORMAL incidence, where the incident plane wave is an exact
    discrete mode on either path.  Measured 2026-10-02: 7.4e-15 (``c1_identity_oblique_M4.json``); bar
    1e-11 (the A2 bar).  (At oblique incidence the two routes differ by
    their incident treatment -- the unmapped least-squares overlap against
    the mapped exact decomposition -- at the discretisation level, falling
    with M: build doc section 4.3.)  The route is asserted directly: no map
    on the stack, walls recorded."""
    shp = [Rect(0.55, 0.62, 0.5, 0.4, 4.0)]
    st, o, R, T, J = _stack([(_DEPTH, (shp, 1.0))], 4)
    assert st.cmap is None and st._shape_walls is not None
    np.testing.assert_allclose(st._shape_walls[0], [0.0, 0.3, 0.8, 1.2],
                               atol=1e-15)
    eps, _x, _y, cm = compile_shapes(_P, _P, shp, 1.0)
    _st2, o2, R2, T2, J2 = _stack([(_DEPTH, ("cell", eps))], 4, cmap=cm)
    d = max(np.abs(R - R2).max(), np.abs(T - T2).max(), np.abs(J - J2).max())
    assert d <= 1e-11, d


# =========================================================================== #
# C2 / C3 -- the primitives ARE Phase B's gate maps
# =========================================================================== #
@functools.lru_cache(maxsize=None)
def _fem_oracle():
    """The planner's SAVED NGSolve oracle (``fem/summary.json``), provenance
    asserted as in Phase B's B3: three meshes, R + T - 1 below 1e-6; its own
    error bar 8.3e-6.  E along y (row 1)."""
    with open(os.path.join(_PROBE, "fem", "summary.json")) as f:
        d = json.load(f)
    assert d["best_from"] == ["h1.0_e20 p4", "h0.8_e30 p4", "h1.0 p6"]
    assert abs(d["RplusT"]["value"] - 1.0) < 1e-6
    ordf = {"0,0": [(0, 0)], "1,0": [(1, 0), (-1, 0)],
            "0,1": [(0, 1), (0, -1)],
            "1,1": [(1, 1), (-1, 1), (1, -1), (-1, -1)]}
    return {(side, mn): rec["value"] for side in ("R", "T")
            for key, rec in d[side].items() for mn in ordf[key]}


def test_c2_circle_primitive_is_the_phase_b_map_and_lands_on_the_fem():
    """``Circle(0.6, 0.6, 0.36)`` builds EXACTLY Phase B's 3 x 3 gate map
    (same walls, same arcs, same vertex images -> same fingerprint, same
    ``eps_cell``) and the 5 x 5 one with ``core=0.5``; the shapes route then
    reproduces the explicit-map solve to the BIT, and lands on the saved
    3-D FEM oracle as Phase B's B3 did.  Measured 2026-10-02
    (``c_ladders_summary.json``): distance to the FEM 8.2e-4 at M = 7 (Phase
    B: 8.18e-4 -- the two differ only by this phase's incident fix, 2e-7 at
    that rung, gate C9); bar 2.5e-3 (Phase B's, 0.5 decade above; the
    4-step staircase reads 7.1e-2 there, Phase B fail-before)."""
    cm3, eps3 = _circle3_explicit()
    e, xw, yw, cm = compile_shapes(_P, _P, [Circle(0.6, 0.6, _R, 4.0)], 1.0)
    assert cm.fingerprint == cm3.fingerprint
    assert np.array_equal(e, eps3)
    cm5, _w = CM._circle_map_5x5(_P, _R)
    _e5, _x5, _y5, cm5b = compile_shapes(
        _P, _P, [Circle(0.6, 0.6, _R, 4.0, core=0.5)], 1.0)
    assert cm5b.fingerprint == cm5.fingerprint
    a = pmm_jones_2d_staggered(_P, _P, None, _NSUB, _NSUP, _DEPTH, _WL,
                               shapes=[Circle(0.6, 0.6, _R, 4.0)],
                               background_eps=1.0, n_modes=5, n_orders=3)
    b = pmm_jones_2d_staggered(_P, _P, eps3, _NSUB, _NSUP, _DEPTH, _WL,
                               cmap=cm3, n_modes=5, n_orders=3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    o, R, T, _J = pmm_jones_2d_staggered(
        _P, _P, None, _NSUB, _NSUP, _DEPTH, _WL,
        shapes=[Circle(0.6, 0.6, _R, 4.0)], background_eps=1.0, n_modes=7,
        n_orders=3)
    o = np.asarray(o)
    ref = _fem_oracle()
    d = max(abs((R if side == "R" else T)[1, _idx(o, [mn])[0]] - v)
            for (side, mn), v in ref.items())
    assert d <= 2.5e-3, d


def test_c3_fillet_primitive_is_the_phase_b_map_and_moves_the_device():
    """``FilletRect(0.6, 0.6, 0.6, 0.6, r)`` builds EXACTLY Phase B's 5 x 5
    fillet map for r / side = 0.05, 0.1, 0.2 (fingerprint and ``eps_cell``),
    and ``r = 0`` is the sharp :class:`Rect` (no map).  The fillet moves the
    zeroth orders far above convergence -- geometry fidelity, Phase B's B6
    decision through the shapes route.  Measured 2026-10-02
    (``c_unit_c3fid.json``): r / side 0.2 at M = 4 (5 x 5) against the sharp
    square at M = 5 (3 x 3, comparable resolution): dR00 = -1.57e-3,
    dT00 = +8.8e-3 (Phase B at M = 5 / 6: -1.80e-3 / +7.65e-3); the
    fillet's own rung change 4 -> 5 reads 1.9e-4 (R00) / 1.3e-3 (T00), the
    square's 5 -> 6 4.2e-5 / 2.4e-3.  Bars (Phase B's): dR00 <= -5e-4 (0.5
    decade inside the shift, 0.4 above the rung change) and dT00 >= +2e-3.
    The same-area square moves R00 the WRONG way (+2.8e-4 at M = 5, B7)."""
    side = 0.6
    for ratio in (0.05, 0.1, 0.2):
        cmb, _w = CM._fillet_map_5x5(_P, side / 2, ratio * side)
        e, _x, _y, cm = compile_shapes(
            _P, _P, [FilletRect(0.6, 0.6, side, side, ratio * side, 4.0)],
            1.0)
        assert cm.fingerprint == cmb.fingerprint, ratio
        ref = np.ones((5, 5), complex)
        ref[1:4, 1:4] = 4.0
        assert np.array_equal(e, ref)
    st0, *_ = _stack([(_DEPTH, ([FilletRect(0.6, 0.6, side, side, 0.0, 4.0)],
                                1.0))], 4)
    assert st0.cmap is None                      # r = 0: the sharp Rect
    _s, o, R, T, _J = _stack([(_DEPTH, ([FilletRect(0.6, 0.6, side, side,
                                                    0.2 * side, 4.0)],
                                        1.0))], 4)
    _s, o0, R0, T0, _J0 = _stack([(_DEPTH, ([Rect(0.6, 0.6, side, side, 4.0)],
                                            1.0))], 5)
    i, i0 = _idx(o, [(0, 0)])[0], _idx(o0, [(0, 0)])[0]
    dR, dT = R[1, i] - R0[1, i0], T[1, i] - T0[1, i0]
    assert dR <= -5e-4, dR
    assert dT >= 2e-3, dT


# =========================================================================== #
# C4 -- the F5 / D6 traps are closed by construction
# =========================================================================== #
def _hand_circle_map(walls):
    """The 3 x 3 circle laid BY HAND on the given interior walls, the
    disk-cell corners moved onto the circle's 45-degree points (the arcs
    unchanged): the SAME exact disk on different walls."""
    c = (_P / 2, _P / 2)
    h = _R / np.sqrt(2.0)
    w = np.array([0.0, walls[0], walls[1], _P])
    V = np.empty((4, 4, 2))
    V[..., 0] = w[:, None]
    V[..., 1] = w[None, :]
    for i, x in ((1, c[0] - h), (2, c[0] + h)):
        for j, y in ((1, c[1] - h), (2, c[1] + h)):
            V[i, j] = (x, y)
    d = np.pi / 180
    curved = {("h", 1, 1): CM.Arc(c, _R, 225 * d, 315 * d),
              ("h", 1, 2): CM.Arc(c, _R, 135 * d, 45 * d),
              ("v", 1, 1): CM.Arc(c, _R, 225 * d, 135 * d),
              ("v", 2, 1): CM.Arc(c, _R, -45 * d, 45 * d)}
    return CM.TransfiniteMap(w, w, V, curved)


def test_c4_primitive_places_the_boundary_where_it_is_and_converges():
    """A circle whose boundary is NOT at a preimage of the shipped uniform
    lattice (walls 0.4, 0.8): the primitive places its walls at the
    45-degree points itself.  Two-sided against hand-built layouts:

    * RATE -- the uniform film under the primitive's map converges
      spectrally: measured 2026-10-02 (``c4_walls_film_M*.json``) 2.0e-6,
      1.1e-8, 8.4e-12, 3.2e-13 at M = 4 .. 7; bars M = 5 <= 1e-7 and the
      M = 4 -> 5 drop >= 1.5 decades (measured 2.3).
    * SAME DEVICE -- a hand-built map of the SAME disk on the lattice walls
      (corners moved onto the circle) converges to the primitive's answer:
      the pillar R / T differ by 2.3e-5 / 4.1e-6 / 1.7e-6 at M = 5 / 6 / 7
      (``c4_walls_pillar_M*.json``), against a rung change of 5.4e-3 at
      M = 5 -> 6: bar 5e-4 at M = 5 (1.3 decades above, 1 below the rung).
    * FAIL-BEFORE -- the boundary SNAPPED to the lattice (the circle through
      the lattice vertices, r = 0.2 sqrt 2 = 0.283, i.e. the boundary moved
      to the nearest grid preimage) is a different device: measured 0.13
      from the primitive at M = 5 (``c_misc_c4.json``); asserted >= 5e-3
      (1.4 decades below the reading, 1 above the same-device bar).
    (Measured too: where the walls go does NOT change the rate on these
    maps -- the lattice map's film reads 2.0e-6 / 1.3e-8 / 5.2e-12 -- the
    planning finding F5 is about ONE cell carrying a steep map, measured in
    the build doc section 4.2.)"""
    e, _x, _y, cm = compile_shapes(_P, _P, [Circle(0.6, 0.6, _R, 4.0)], 1.0)
    e4, e5 = _film_err(cm, 4), _film_err(cm, 5)
    assert e5 <= 1e-7, e5
    assert e4 / e5 >= 10 ** 1.5, (e4, e5)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = 4.0
    M = 5
    _s, o, R, T, _J = _stack([(_DEPTH, ([Circle(0.6, 0.6, _R, 4.0)], 1.0))],
                             M)
    _s, o2, R2, T2, _J2 = _stack([(_DEPTH, ("cell", eps))], M,
                                 cmap=_hand_circle_map((0.4, 0.8)))
    d_same = float(np.max(np.abs(_vec(o, R, T) - _vec(o2, R2, T2))))
    assert d_same <= 5e-4, d_same
    _s, o3, R3, T3, _J3 = _stack(
        [(_DEPTH, ([Circle(0.6, 0.6, 0.2 * np.sqrt(2.0), 4.0)], 1.0))], M)
    d_snap = float(np.max(np.abs(_vec(o, R, T) - _vec(o3, R3, T3))))
    assert d_snap >= 5e-3, d_snap


def test_c4_a_primitive_never_takes_u_walls_for_physical_ones():
    """Phase A verifier D6: ``SeparableStretch(u_walls=physical walls)``
    silently builds a DIFFERENT device -- its material walls sit at the
    images ``f(u_wall)``.  A primitive takes PHYSICAL geometry only and
    places the walls itself: the image of every outline vertex is the
    physical point asked for, to round-off.  Fail-before (the trap itself,
    measured 2026-10-02, ``c_misc_d6.json``): a 0.08 sine stretch fed the
    physical walls 0.20, 0.65 puts them at 0.269 and 0.629 -- 6.9e-2 off,
    ten decades above the primitive's 1e-12 bar (its own reading: round-off,
    <= 2.2e-16)."""
    rect = Rect(0.425, 0.6, 0.45, 1.2, 4.0)          # walls 0.20, 0.65
    for shp in (rect, Circle(0.58, 0.63, 0.33, 4.0),
                FilletRect(0.6, 0.58, 0.7, 0.5, 0.08, 4.0),
                SinusoidalWall("x", 0.3, 0.12, eps=4.0, width=0.5)):
        _e, _x, _y, cm = compile_shapes(_P, _P, [shp], 1.0)
        lay = shp._layout(_P, _P)
        for (u, v), xy in lay.vertices.items():
            sx = min(int(np.searchsorted(cm.u_bounds, u, "right")) - 1,
                     cm.shape[0] - 1)
            sy = min(int(np.searchsorted(cm.v_bounds, v, "right")) - 1,
                     cm.shape[1] - 1)
            g = cm.geom_points(sx, sy, np.array([u]), np.array([v]))
            got = np.array([g[0][0], g[1][0]])
            assert float(np.max(np.abs(got - xy))) <= 1e-12, (shp, u, v)
            assert abs(float(shp.signed_distance(*got))) <= 1e-12, shp
    trap = CM.SeparableStretch(np.array([0.0, 0.20, 0.65, _P]), 3,
                               fx=CM.SineStretch(0.08), period_y=_P)
    xb, _yb = trap.physical_walls()
    off = float(np.max(np.abs(xb[1:3] - [0.20, 0.65])))
    assert off >= 1e-2, off


# =========================================================================== #
# C5 -- refusals name the two shapes (and layers); the stack rolls back
# =========================================================================== #
def test_c5_refusals_name_both_shapes_and_layers():
    st = PMM2DStackPure(_P, _P, n_modes=4)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
    fp = st.cmap.fingerprint
    # crossing outlines in two layers
    with pytest.raises(ValueError, match=r"layer 1: Circle.*layer 2: Rect.*"
                                         r"CROSS"):
        st.add_layer(0.3, shapes=[Rect(0.6, 0.6, 0.5, 0.5, 2.0)],
                     background_eps=1.0)
    assert st.cmap.fingerprint == fp and len(st._layers) == 1   # rolled back
    # a rounded corner over a sharp one: one cell, two outlines
    st2 = PMM2DStackPure(_P, _P, n_modes=4)
    st2.add_layer(0.3, shapes=[FilletRect(0.6, 0.6, 0.6, 0.6, 0.06, 4.0)],
                  background_eps=1.0)
    with pytest.raises(ValueError, match=r"FilletRect.*Rect|Rect.*FilletRect"):
        st2.add_layer(0.3, shapes=[Rect(0.6, 0.6, 0.6, 0.6, 2.0)],
                      background_eps=1.0)
    # a straight edge laid on a fillet's flat side (which lives on the grid
    # line through the 45-degree points): a zero-height cell -- the merged
    # map folds, and the message names both
    with pytest.raises(ValueError, match=r"FOLDS.*(FilletRect.*Rect|Rect.*"
                                         r"FilletRect)"):
        compile_shapes(_P, _P, [FilletRect(0.6, 0.6, 0.6, 0.6, 0.06, 4.0),
                                Rect(0.6, 0.25, 0.2, 0.1, 2.0)], 1.0)
    # sliver: two edges 5e-4 of the period apart
    with pytest.raises(ValueError, match=r"SLIVER.*Rect.*Rect"):
        compile_shapes(_P, _P, [Rect(0.3, 0.6, 0.2, 0.2, 2.0),
                                Rect(0.5006, 0.6, 0.2, 0.2, 3.0)], 1.0)
    compile_shapes(_P, _P, [Rect(0.3, 0.6, 0.2, 0.2, 2.0),           # shared
                            Rect(0.5, 0.6, 0.2, 0.2, 3.0)], 1.0)     # wall
    # the fillet radius below the sliver contract: named, with the remedy
    with pytest.raises(ValueError, match=r"radius=0.*1\.4142e-3|1\.4142e-3.*"
                                         r"radius=0"):
        compile_shapes(_P, _P, [FilletRect(0.6, 0.6, 0.6, 0.6, 1.5e-3, 4.0)],
                       1.0)
    compile_shapes(_P, _P, [FilletRect(0.6, 0.6, 0.6, 0.6, 1.8e-3, 4.0)], 1.0)
    # scope refusals name their phase (Phase D, 2026-10-03, routes a
    # BLOCK-FORM tensor on a circle; Phase E1, 2026-10-03, an OUT-OF-PLANE
    # one too -- accepted now, gates in test_pmm2d_staggered_curved_e1.py)
    oop = np.diag([4.0, 3.0, 3.5]).astype(complex)
    oop[0, 2] = oop[2, 0] = 0.4
    compile_and_add = PMM2DStackPure(_P, _P, n_modes=4)
    compile_and_add.add_layer(
        0.3, shapes=[Circle(0.6, 0.6, 0.3, oop)], background_eps=1.0)
    with pytest.raises(NotImplementedError, match="Phase E"):
        PMM2DStackPure(_P, _P, layer_grids="per-layer").add_layer(
            0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
    with pytest.raises(ValueError, match="eps_cell"):
        st.add_layer(0.2, eps_cell=np.ones((3, 3)))
    cm3, eps3 = _circle3_explicit()
    with pytest.raises(ValueError, match="explicit cmap"):
        PMM2DStackPure(_P, _P, cmap=cm3).add_layer(
            0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
    with pytest.raises(ValueError, match="eps_cell"):
        pmm_jones_2d_staggered(_P, _P, eps3, _NSUB, 1.0, 0.3, _WL,
                               shapes=[Circle(0.6, 0.6, 0.3, 4.0)],
                               background_eps=1.0)
    with pytest.raises(ValueError, match="background_eps"):
        st.add_layer(0.2, shapes=[Circle(0.6, 0.6, 0.3, 4.0)])
    with pytest.raises(ValueError, match="unit cell"):
        compile_shapes(_P, _P, [Circle(0.2, 0.6, 0.3, 4.0)], 1.0)


# =========================================================================== #
# C6 -- two layers, two different shapes, one merged map
# =========================================================================== #
def test_c6_two_layer_stack_on_the_merged_map():
    """A circle (layer 1) nested inside a filleted square's footprint (layer
    2) on ONE merged 7 x 7 map, at M = 3 (the unit-test size; every
    gate here is structural, so the coarsest basis serves).  Measured
    2026-10-02 (``c_unit_c6.json``, ``c6b_absorb_M3.json``):

    * lossless closure (layer 2 vacuum-painted): 3.2e-3 (3.2e-3 with layer 2
      eps 2.25); bar 3e-2 (1 decade up; the no-cofactor projector reads
      0.1-0.4 on such cells, Phase A/B);
    * absorption, layer 2 lossy (eps 2.25 + 0.4i): the LOSSLESS circle layer
      absorbs 7.7e-15 (bar 1e-10, 4.1 decades up) -- with ``-R`` as the flux
      Gram (Phase A's A7 defect) it reads 1.7e-2 (8.2 decades above the
      bar); the lossy layer's absorption matches 1 - sum R - sum T to 2.2e-3
      at this M (discretisation level, recorded only);
    * the VACUUM identity on the SAME map: layer 2 painted vacuum-on-vacuum
      equals a uniform vacuum layer 2 riding the merged map explicitly (the
      shared geometric eig instead of a region eig): 2.2e-15, bar 1e-9;
      fail-before: painting it with eps 1.1, 8.2e-3."""
    circ = Circle(0.6, 0.6, 0.2, 4.0)

    def frame(e):
        return FilletRect(0.6, 0.6, 0.9, 0.9, 0.09, e)
    M = 3
    st, o, R, T, _J = _stack([(0.3, ([circ], 1.0)),
                              (0.2, ([frame(2.25 + 0.4j)], 1.0))], M,
                             retain=True)
    assert st.cmap.shape == (7, 7)
    A = np.asarray(st.layer_absorption())
    bal = 1 - R.sum(1) - T.sum(1)
    assert float(np.min(bal)) > 1e-2                 # the loss is real
    assert float(np.min(A[1])) > 1e-2                # ... and in layer 2
    a1 = float(np.max(np.abs(A[0])))                 # lossless layer 1
    assert a1 <= 1e-10, a1
    G = st._internal["G"]
    st._internal["G"] = -TS.Granet2DTransverseE(
        _P, _P, st.cmap.u_walls, st.cmap.v_walls, M,
        np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
    a1_fb = float(np.max(np.abs(np.asarray(st.layer_absorption())[0])))
    st._internal["G"] = G
    assert a1_fb >= 1e-3, a1_fb
    st2, o2, R2, T2, _J2 = _stack([(0.3, ([circ], 1.0)),
                                   (0.2, ([frame(1.0)], 1.0))], M)
    clo = float(np.max(np.abs(R2.sum(1) + T2.sum(1) - 1)))
    assert clo <= 3e-2, clo
    _s, o3, R3, T3, _J3 = _stack(
        [(0.3, ("cell", st2._layers[0]["eps_cell"])), (0.2, 1.0)], M,
        cmap=st2.cmap)
    d_vac = float(np.max(np.abs(_vec(o2, R2, T2) - _vec(o3, R3, T3))))
    assert d_vac <= 1e-9, d_vac
    _s, o4, R4, T4, _J4 = _stack([(0.3, ([circ], 1.0)),
                                  (0.2, ([frame(1.1)], 1.0))], M)
    d_fb = float(np.max(np.abs(_vec(o2, R2, T2) - _vec(o4, R4, T4))))
    assert d_fb >= 1e-4, d_fb


# =========================================================================== #
# C7 / C8 -- the convenience entry; the single-polarization entry refuses
# =========================================================================== #
def test_c7_convenience_entry_is_the_one_layer_stack_byte_for_byte():
    shp = [Circle(0.55, 0.62, 0.3, 4.0), Rect(0.6, 0.15, 1.2, 0.05, 2.0)]
    a = pmm_jones_2d_staggered(_P, _P, None, _NSUB, _NSUP, 0.4, _WL,
                               shapes=shp, background_eps=1.0, n_modes=3,
                               n_orders=3, theta=0.2, phi=0.4)
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=3, n_orders=3)
    st.add_layer(0.4, shapes=shp, background_eps=1.0)
    st.set_source(_WL, theta=0.2, phi=0.4)
    b = st.solve()
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_c8_efficiency_entry_refuses_shapes_and_points_at_jones():
    with pytest.raises(NotImplementedError,
                       match="pmm_jones_2d_staggered.*shapes="):
        pmm_efficiency_2d_staggered(_P, _P, np.ones((3, 3)), _NSUB, 1.0, 0.3,
                                    _WL, shapes=[Circle(0.6, 0.6, 0.3, 4.0)],
                                    background_eps=1.0)


# =========================================================================== #
# C9 -- the incident field under a map: exact, window-free, normalised
# =========================================================================== #
def test_c9_incident_decomposition_is_window_free_and_off_the_floor(
        monkeypatch):
    """Phase B finding F-B4 / Phase A verifier D4.  The shipped least-squares
    overlap ``cinc = lstsq(Hsup, delta_00)`` is UNDERdetermined under a map:
    its minimum-norm draw carries a round-off floor and makes R / T depend on
    ``n_orders``.  The exact (renormalised L2) decomposition removes both.
    Measured 2026-10-02 on the 3 x 3 circle at M = 5
    (``c9_incident_*_M5.json``, ``c9b_pillar_M5.json``): ``n_orders`` 2 vs 5
    -- 8.3e-16 after, 1.3e-5 before (the fail-before is the shipped overlap,
    reached through the real code by making the decomposition helper return
    None); a 1e-15 random perturbation of the half-space weights -- 5e-15
    after, 1.1e-7 before.  Bars 1e-12 (2.6 / 2.3 decades above the readings;
    5.1 / 4.9 below the fail-befores)."""
    cm, eps = _circle3_explicit()

    def run(n_orders):
        return _vec(*_stack([(_DEPTH, ("cell", eps))], 5, n_orders=n_orders,
                            cmap=cm)[1:4])

    def pert_run():
        orig = TS._stag_map_eff
        rng = np.random.default_rng(1)

        def pert(e, sg, g11, g12, g22):
            out = orig(e, sg, g11, g12, g22)
            a = np.asarray(e)
            if a.size > 1 and float(np.ptp(a.real)) == 0.0:   # half-spaces
                out = {k: v * (1 + 1e-15 * rng.standard_normal(np.shape(v)))
                       for k, v in out.items()}
            return out
        with monkeypatch.context() as mp:
            mp.setattr(TS, "_stag_map_eff", pert)
            return run(3)
    v2, v5, v3 = run(2), run(5), run(3)
    d_win = float(np.max(np.abs(v2 - v5)))
    d_pert = float(np.max(np.abs(pert_run() - v3)))
    assert d_win <= 1e-12, d_win
    assert d_pert <= 1e-12, d_pert
    monkeypatch.setattr(SP, "_stag_incident_coeffs_mapped",
                        lambda *a, **k: None)          # the shipped overlap
    d_win_fb = float(np.max(np.abs(run(2) - run(5))))
    assert d_win_fb >= 1e-9, d_win_fb


def test_c9_incident_renormalisation_keeps_the_film_exact(monkeypatch):
    """Under a map the L2 projection of the plane wave is not EXACTLY a unit
    order-0 field (its far field is delta_00 to the projection's own error),
    and the efficiencies are normalised to a unit incident amplitude -- so the
    decomposition is renormalised (2 x 2, order 0 only) to make that exact.
    Measured 2026-10-02 (``c9b_film_M*.json``), uniform film under the 3 x 3
    circle map vs the Airy slab: M = 6 renormalised 8.4e-12, bare L2
    8.8e-8 (fail-before: the helper called without ``H0``), the shipped
    overlap 4.3e-11.  Bar 1e-9 (2.1 decades above; 1.9 below the bare L2)."""
    cm, _eps = _circle3_explicit()
    e = _film_err(cm, 6)
    assert e <= 1e-9, e
    orig = TS._stag_incident_coeffs_mapped
    monkeypatch.setattr(SP, "_stag_incident_coeffs_mapped",
                        lambda *a, **k: orig(*a))
    e_bare = _film_err(cm, 6)
    assert e_bare >= 1e-8, e_bare


# =========================================================================== #
# C10 -- oblique and conical incidence through the shapes route
# =========================================================================== #
def test_c10_oblique_and_conical_shapes_route_is_the_explicit_map():
    """At (25 deg, 0) and (25 deg, 40 deg) the shapes route reproduces the
    explicit Phase B map to the BIT (so Phase B's B10 reciprocity gates carry
    over unchanged); the build doc re-measures the reciprocity ladder on the
    shapes route (``c_ladders_summary.json``)."""
    cm3, eps3 = _circle3_explicit()
    for th, ph in ((25.0, 0.0), (25.0, 40.0)):
        a = _stack([(_DEPTH, ([Circle(0.6, 0.6, _R, 4.0)], 1.0))], 5,
                   theta=np.deg2rad(th), phi=np.deg2rad(ph))
        b = _stack([(_DEPTH, ("cell", eps3))], 5, theta=np.deg2rad(th),
                   phi=np.deg2rad(ph), cmap=cm3)
        for x, y in zip(a[1:], b[1:]):
            assert np.array_equal(x, y)
        clo = float(np.max(np.abs(a[2].sum(1) + a[3].sum(1) - 1)))
        assert clo <= 5e-2, clo                      # (B10: 1.9e-3 at M = 6)


# =========================================================================== #
# C11 -- the viewers draw curves, not the wall grid
# =========================================================================== #
def test_c11_viewer_draws_the_analytic_outline():
    """``plot_geometry`` on a mapped stack draws every cell's PHYSICAL image
    and the material boundaries as curves through the map; every vertex of
    the drawn outline of a circle must sit on the circle.  Measured
    2026-10-02 (``c_misc_c11.json``): max | |p - c| - r | = 3.3e-16 over the
    256 drawn vertices; bar 1e-12 (3.5 decades up).  Fail-before: the (u, v)
    cell outline (what drawing the wall grid would show) is 0.105 (0.29 r)
    off, bar 0.05 r.  ``plot_section`` places the circle's boundary at
    0.6 -+ 0.36 on the cut y = 0.6 to 5.6e-17 (bisection on the shape's own
    outline); bar 1e-10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = PMM2DStackPure(_P, _P, n_modes=4)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, _R, 4.0)], background_eps=1.0)
    st.add_layer(0.2, eps=2.25)
    axes = st.plot_geometry()
    pts = np.concatenate([np.column_stack(ln.get_data())
                          for ln in axes[0].lines])
    assert pts.shape[0] >= 128
    dev = float(np.max(np.abs(np.hypot(pts[:, 0] - 0.6, pts[:, 1] - 0.6)
                              - _R)))
    assert dev <= 1e-12, dev
    assert len(axes[1].lines) == 0                   # uniform: no outline
    c = st.cmap
    s = np.linspace(0.0, 1.0, 64)
    U = c.u_bounds[1] + s * (c.u_bounds[2] - c.u_bounds[1])
    raw = np.column_stack([U, np.full_like(U, c.v_bounds[1])])
    dev_fb = float(np.max(np.abs(np.hypot(raw[:, 0] - 0.6, raw[:, 1] - 0.6)
                                 - _R)))
    assert dev_fb >= 0.05 * _R, dev_fb
    plt.close(axes[0].figure)
    runs = st._mapped_section(st._layers[0], 0, 0.6)
    edges = [r[1] for r in runs[:-1]]
    np.testing.assert_allclose(edges, [0.6 - _R, 0.6 + _R], atol=1e-10)
    ax = st.plot_section()
    plt.close(ax.figure)


# =========================================================================== #
# C12 -- every primitive's geometry is exact
# =========================================================================== #
def test_c12_mapped_area_perimeter_and_det_j_of_every_primitive():
    """The mapped region of each primitive (the image of the cells it
    paints) has the analytic area and the analytic outline length, and
    det J > 0 at every interior Gauss node.  Measured 2026-10-02
    (``c_unit_c12.json``): area and perimeter <= 3.8e-15 relative for Rect,
    Circle (3 x 3 and 5 x 5), FilletRect (w != h), Ellipse (axis-aligned and
    rotated 20 deg), SinusoidalWall (ridge and half-plane, 2 waves); min det J
    2.8e-3 (Gauss nodes approach the singular vertices).  Bars 1e-12 (area,
    perimeter) and > 0.  Fail-before: a circle 1e-9 larger reads 2e-9."""
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(48)
    shapes = [Rect(0.55, 0.62, 0.5, 0.4, 4.0),
              Circle(0.58, 0.63, 0.33, 4.0),
              Circle(0.6, 0.6, 0.36, 4.0, core=0.5),
              FilletRect(0.6, 0.58, 0.7, 0.5, 0.08, 4.0),
              Ellipse(0.6, 0.6, 0.40, 0.28, 4.0),
              Ellipse(0.6, 0.6, 0.40, 0.28, 4.0, angle=np.deg2rad(20.0)),
              SinusoidalWall("x", 0.3, 0.12, eps=4.0, width=0.5),
              SinusoidalWall("y", 0.5, 0.08, 2, 0.3, eps=4.0)]
    for sh in shapes:
        eps, _x, _y, cm = compile_shapes(_P, _P, [sh], 1.0)
        A, mind, L = 0.0, np.inf, 0.0
        nx, ny = cm.shape
        for sx in range(nx):
            Ju = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
            U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + Ju * xg
            for sy in range(ny):
                Jv = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
                V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + Jv * xg
                _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
                det = xu * yv - xv * yu
                mind = min(mind, float(det.min()))
                if eps[sx, sy] != 1.0:
                    A += float(np.sum(np.outer(wg, wg) * det)) * Ju * Jv
                for di, dj in ((1, 0), (0, 1)):
                    if eps[sx, sy] == eps[(sx + di) % nx, (sy + dj) % ny]:
                        continue
                    if di:
                        Vv = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) \
                            + Jv * xg
                        g = cm.geom_points(sx, sy, np.full_like(
                            Vv, cm.u_bounds[sx + 1]), Vv)
                        L += float(wg @ np.hypot(g[3], g[5])) * Jv
                    else:
                        Uu = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) \
                            + Ju * xg
                        g = cm.geom_points(sx, sy, Uu, np.full_like(
                            Uu, cm.v_bounds[sy + 1]))
                        L += float(wg @ np.hypot(g[2], g[4])) * Ju
        per = sh.perimeter()
        if isinstance(sh, SinusoidalWall) and sh.width is None:
            per += _P                     # the half-plane also meets the seam
        assert abs(A / sh.area() - 1.0) <= 1e-12, (sh, A, sh.area())
        assert abs(L / per - 1.0) <= 1e-12, (sh, L, per)
        assert mind > 0.0, (sh, mind)
    big = Circle(0.6, 0.6, _R * (1 + 1e-9), 4.0)
    assert abs(big.area() / Circle(0.6, 0.6, _R, 4.0).area() - 1) >= 1e-9


# =========================================================================== #
# C13 / C14 -- physical identities through the merge
# =========================================================================== #
def test_c13_two_by_two_circle_array_is_the_halved_period():
    """Four circles in the doubled period 2 p are the single circle in p:
    every ODD order of the doubled period must vanish.  The merge squares
    up nothing here (5 x 5) and every transition cell between two circles
    blends TWO arcs.  Measured 2026-10-02 (``c_unit_c13.json``,
    ``c_unit_c13_M4M5.json``): odd orders 5.3e-3 / 6.5e-6 / 3.4e-7 at
    M = 3 / 4 / 5, against 4.2e-2 / 2.5e-2 / 2.2e-2 when one circle is left
    out (fail-before: a mis-merged layout).  At M = 4: bar 1e-3 (2.2 decades
    above the reading, 1.4 below the fail-before).  The EVEN orders match the
    single circle at the discretisation level (5.6e-3 at M = 5, build
    doc)."""
    r = _R
    four = [Circle(cx, cy, r, 4.0) for cx in (0.6, 1.8) for cy in (0.6, 1.8)]

    def odd(shapes):
        st, o, R, T, _J = _stack([(_DEPTH, (shapes, 1.0))], 4, period=2 * _P)
        assert st.cmap.shape == (5, 5)
        m = (o[:, 0] % 2 != 0) | (o[:, 1] % 2 != 0)
        return float(max(R[:, m].max(), T[:, m].max()))
    assert odd(four) <= 1e-3
    assert odd(four[:3]) >= 1e-2


def test_c14_circle_is_four_fold_symmetric_and_an_ellipse_is_not():
    """``te(m, n) = tm(n, m)`` for the circle through the shapes route.
    Measured 2026-10-02 (``c_unit_c14.json``): 2.8e-14 at M = 5 (Phase B's
    lstsq-era floor read 8.9e-10 at M = 6; the C9 fix took it to round-off);
    an ellipse with semi-axes r, 1.02 r reads 5.8e-3 (fail-before).  Bar
    1e-10 (3.6 decades above; 7.8 below)."""
    def sym(shp):
        _s, o, R, T, _J = _stack([(_DEPTH, ([shp], 1.0))], 5)
        out = 0.0
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                i, j = _idx(o, [(m, n)])[0], _idx(o, [(n, m)])[0]
                out = max(out, abs(R[1, i] - R[0, j]), abs(T[1, i] - T[0, j]))
        return out
    assert sym(Circle(0.6, 0.6, _R, 4.0)) <= 1e-10
    assert sym(Ellipse(0.6, 0.6, _R, 1.02 * _R, 4.0)) >= 1e-4


# =========================================================================== #
# C15 / C16 -- no stale map; tensors routed or refused
# =========================================================================== #
def test_c15_a_compiled_shape_cannot_go_stale():
    """The stack compiles its shapes into the merged map at ``add_layer``
    (a cache keyed by the map fingerprint downstream).  A shape is therefore
    IMMUTABLE -- changing one after it was compiled would leave the stack's
    map silently stale -- and a shape with any parameter changed is a new
    map (a cache MISS), down to a 1e-12 relative radius change."""
    c = Circle(0.6, 0.6, 0.3, 4.0)
    st = PMM2DStackPure(_P, _P, n_modes=4)
    st.add_layer(0.3, shapes=[c], background_eps=1.0)
    with pytest.raises(AttributeError, match="immutable"):
        c.r = 0.31
    fps = {compile_shapes(_P, _P, [s], 1.0)[3].fingerprint for s in
           (c, Circle(0.6, 0.6, 0.3 * (1 + 1e-12), 4.0),
            Circle(0.6 + 1e-13, 0.6, 0.3, 4.0),
            Circle(0.6, 0.6, 0.3, 4.0, core=0.5))}
    assert len(fps) == 4
    assert compile_shapes(_P, _P, [Circle(0.6, 0.6, 0.3, 4.0)],
                          1.0)[3].fingerprint == st.cmap.fingerprint


def test_c16_tensors_ride_the_identity_map_and_the_curved_map():
    """Phase C routes tensors where no map is needed; Phase D makes them
    correct under a map.  A TENSOR rectangle rides the unmapped solver and
    equals the shipped integer-grid tensor solve (walls 0.4, 0.8 as an array
    vs the integer lattice: ULP-close, gate N2 of the mortar build) --
    measured 2026-10-02 3.6e-15 (``c_unit_c16.json``), bar 1e-12.  The same
    tensor on a circle needs the Jacobian itself (``c_misc_c16.json``: the
    best scalar surrogate ``s sqrt(g) g^-1`` of eps_t = [[4, 0.3], [0.3, 3]]
    is 0.20 off the true ``sqrt(g) J^-1 eps J^-T`` at the circle map's
    nodes); Phase C refused it naming Phase D, and since Phase D
    (2026-10-03) it SOLVES -- its gates are in
    ``tests/unit/test_pmm2d_staggered_curved_d.py``.  Here: the circle and a
    uniform tensor layer under a curved shape map solve and close (lossless,
    M = 4: closure recorded <= 1e-2, the M = 4 discretisation level of the
    circle, Phase B's ladder); an OUT-OF-PLANE tensor on a circle (and a
    uniform one under the curved map) is accepted since Phase E1 (gates in
    ``tests/unit/test_pmm2d_staggered_curved_e1.py``) -- here it only has to
    build."""
    eps_t = np.array([[4.0, 0.3, 0], [0.3, 3.0, 0], [0, 0, 3.5]], complex)
    st, o, R, T, J = _stack([(_DEPTH, ([Rect(0.6, 0.6, 0.4, 0.4, eps_t)],
                                       1.0))], 5)
    cell = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
    cell[1, 1] = eps_t
    o2, R2, T2, J2 = pmm_jones_2d_staggered(_P, _P, cell, _NSUB, _NSUP,
                                            _DEPTH, _WL, n_modes=5,
                                            n_orders=3)
    d = max(np.abs(R - R2).max(), np.abs(T - T2).max(),
            np.abs(J - np.asarray(J2)).max())
    assert d <= 1e-12, d
    _st, _o, Rc, Tc, _J = _stack([(_DEPTH, ([Circle(0.6, 0.6, 0.3, eps_t)],
                                            1.0))], 4)
    assert np.abs(Rc.sum(1) + Tc.sum(1) - 1.0).max() <= 1e-2
    st = PMM2DStackPure(_P, _P, n_modes=4, n_orders=3)
    st.add_layer(0.2, eps=eps_t)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
    st.set_source(_WL)
    _o, Ru, Tu = st.solve()[:3]
    assert np.abs(Ru.sum(1) + Tu.sum(1) - 1.0).max() <= 1e-2
    oop = eps_t.copy()
    oop[1, 2] = oop[2, 1] = 0.25
    st = PMM2DStackPure(_P, _P, n_modes=4)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.3, oop)], background_eps=1.0)
    st.add_layer(0.2, eps=oop)
    assert st.cmap is not None


# =========================================================================== #
# C17 -- the rotated ellipse (moved corner vertices) obeys its mirror
# =========================================================================== #
def test_c17_rotated_ellipse_is_its_own_mirror_image():
    """The ROTATED ellipse is the one primitive whose disk-cell corners are
    MOVED onto the outline (no axis-aligned rectangle is inscribed in a
    rotated ellipse), so its transition cells carry a bilinear shear the
    other layouts do not.  Physical identity: the ellipse at +a is the
    y-mirror of the ellipse at -a, so at normal incidence R(m, n; +a) =
    R(m, -n; -a) for both inputs.  Measured 2026-10-03
    (``c_misc_ell.json``): 1.5e-14 (M = 4), 8.6e-13 (M = 5); bar 1e-10 (2
    decades above).  Fail-before: the same identity WITHOUT the mirror
    (R(m, n; +a) against R(m, -n; +a)) reads 2.6e-2 / 1.8e-2 -- the rotated
    device is genuinely asymmetric, so the identity has teeth."""
    def solve(al):
        _s, o, R, T, _J = _stack([(_DEPTH, ([Ellipse(0.6, 0.6, 0.40, 0.28,
                                                     4.0, angle=al)], 1.0))],
                                 4)
        return {(int(m), int(n)): (R[:, k], T[:, k])
                for k, (m, n) in enumerate(o)}
    a = np.deg2rad(20.0)
    p, q = solve(a), solve(-a)
    mir = max(float(np.max(np.abs(p[(m, n)][s] - q[(m, -n)][s])))
              for (m, n) in p for s in (0, 1))
    same = max(float(np.max(np.abs(p[(m, n)][s] - p[(m, -n)][s])))
               for (m, n) in p for s in (0, 1))
    assert mir <= 1e-10, mir
    assert same >= 1e-3, same
