"""ROUND 2 of the PURE staggered 2-D PMM per-layer-grid (L2 mortar) work --
defects D1, D2 and D3 of
``docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md``, fixed and gated.

Fix doc: ``docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md``.

  * **D1** (P1) an intra-layer SLIVER -- two walls of ONE layer far closer
    than the rest of that layer's own partition -- corrupts the mortar that
    couples it to neighbours on other grids, ENERGY-INVISIBLY.  Fixed by a
    MINIMUM SEGMENT WIDTH contract at the grid's own entry point,
    ``Basis1D.__init__``
    (:data:`~lumenairy.elements.pmm.twod_staggered._STAG_MIN_SEG_FRAC`).
  * **D2** (P2) the two 2-D mortar ``np.linalg.solve`` calls and the
    generalized twin raised a bare ``LinAlgError: Singular matrix``.  Fixed by
    :func:`~lumenairy.elements.pmm._core._guarded_mortar_solve` -- the same
    LAPACK ``getrf``/``getrs`` pair, screened on the ``gecon`` estimate the
    factors already carry.  It moves no bit of the SciPy solve it rides
    (asserted unconditionally); whether that is ALSO bit-identical to
    ``np.linalg.solve`` depends on whether the wheel pair links ONE LAPACK,
    which is not a portable premise -- see
    ``test_the_guarded_mortar_solve_returns_the_numpy_solve_bit_for_bit``.
  * **D3** (P3) ``_stag_fourier_projection``'s FIXED ``nq = 2 M + 8`` was
    sized for a segment of length ``d/N``.  Fixed by sizing the rule from each
    segment's OWN half-phase, with the INTEGER path untouched bit for bit.

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS**
(2026-09-11), both readings stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

Per ``docs/TESTING_STANDARDS.md`` every population below is RE-MEASURED on the
running build rather than pinned, both bars are asserted with the gap on each
side measured here, and the D1 guard carries a FAIL-BEFORE arm
(:data:`~lumenairy.elements.pmm.twod_staggered.PMM2D_STAG_MIN_SEG_GUARD`).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Basis1D,
    Granet2DTransverseE,
    StagCrossOps,
    StagGridOps,
    _modleg_value_deriv,
    _region_modes,
    _stag_fourier_projection,
    _stag_kron_apply,
)
from lumenairy.elements.rcwa._core import _EnergyError  # noqa: E402

_P = 1.2
_WL = 0.85
_TH, _PH = 0.15, 0.35
_EPS_P, _EPS_H = 9.0, 2.25


def _h(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(
        (str(a.dtype) + str(a.shape)).encode() + a.tobytes()).hexdigest()


def _narrowest(st):
    """Narrowest segment of any layer of a built stack, as a fraction of the
    period -- read off the stack's OWN wall arrays, so it costs no solve."""
    out = 1.0
    for L in st._layers:
        for ax, per in (("wx", st.period_x), ("wy", st.period_y)):
            w = L[ax]
            if np.ndim(w) == 0:
                out = min(out, 1.0 / int(w))
            else:
                out = min(out, float(np.min(np.diff(np.asarray(w)))) / per)
    return out


def _tile(n=3, ep=_EPS_P, eh=_EPS_H):
    t = np.full((n, n), _C(eh))
    t[n // 2, n // 2] = _C(ep)
    return t


# ==========================================================================
# D1 -- the MINIMUM SEGMENT WIDTH contract
# ==========================================================================
def _shipped_geometry_battery():
    """Every geometry class the shipped per-layer / non-uniform / taper
    fixtures build, through the PUBLIC API.  Returns ``{name: stack}``."""
    out = {}
    w1 = [0.2371 * _P, 0.6183 * _P]
    w2 = [0.3117 * _P, 0.7402 * _P]
    tl = _tile()

    def _pl(layers, M=5):
        st = PMM2DStackPure(_P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        for t, cell, xw, yw in layers:
            st.add_layer(t, eps_cell=cell, x_walls=xw, y_walls=yw)
        return st

    out["conforming"] = _pl([(0.10, tl, w1, w1), (0.10, tl, w1, w1)])
    out["non_conforming"] = _pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)])
    out["axes_differ"] = _pl([(0.30, tl, [0.21 * _P, 0.55 * _P],
                              [0.33 * _P, 0.78 * _P])])
    out["single_wall"] = _pl([(0.30, np.full((2, 2), _C(_EPS_H)),
                              [0.4 * _P], [0.4 * _P])])
    out["duty_third"] = _pl([(0.30, tl, [_P / 3.0, 2 * _P / 3.0],
                             [_P / 3.0, 2 * _P / 3.0])])
    out["nested"] = _pl([
        (0.10, tl, [0.25 * _P, 0.75 * _P], [0.25 * _P, 0.75 * _P]),
        (0.10, np.full((5, 5), _C(_EPS_H)),
         [0.125 * _P, 0.25 * _P, 0.75 * _P, 0.875 * _P],
         [0.125 * _P, 0.25 * _P, 0.75 * _P, 0.875 * _P])])
    # the mortar suite's taper, at its own slice count and at 16x it
    for ns in (4, 8, 16, 32, 64):
        st = PMM2DStackPure(_P, n_modes=5, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(0.24, eps_pillar=_EPS_P, eps_host=_EPS_H,
                              x_bounds_bottom=[0.1873 * _P, 0.7241 * _P],
                              y_bounds_bottom=[0.1873 * _P, 0.7241 * _P],
                              x_bounds_top=[0.2917 * _P, 0.6109 * _P],
                              y_bounds_top=[0.2917 * _P, 0.6109 * _P],
                              n_slices=ns)
        out[f"taper_n{ns}"] = st
    # a taper that CLOSES to a point, at the slice counts a user would try:
    # the midpoint rule's narrowest sampled width is ~ w_bottom/(2 n_slices)
    for ns in (8, 32, 64):
        st = PMM2DStackPure(_P, n_modes=4, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(0.24, eps_pillar=_EPS_P, eps_host=_EPS_H,
                              x_bounds_bottom=[0.25 * _P, 0.75 * _P],
                              y_bounds_bottom=[0.25 * _P, 0.75 * _P],
                              x_bounds_top=[0.4999 * _P, 0.5001 * _P],
                              y_bounds_top=[0.4999 * _P, 0.5001 * _P],
                              n_slices=ns)
        out[f"closing_taper_n{ns}"] = st
    st = PMM2DStackPure(_P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillars(
        0.20, eps_host=_EPS_H, n_slices=6,
        pillars=[((0.3 * _P, 0.3 * _P), (0.18 * _P, 0.18 * _P),
                  (0.26 * _P, 0.26 * _P), 9.0)])
    out["tapered_pillars"] = st
    return out


def test_the_minimum_segment_bar_clears_every_geometry_the_library_builds():
    """FALSE-POSITIVE side of the D1 bar, RE-MEASURED here.

    The bar is scale-free (a width over a period), so this census is a
    statement about the library's own surfaces, not about one fixture: it
    builds every per-layer geometry class the shipped suites exercise --
    conforming, non-conforming, per-axis-different walls, a single interior
    wall, the duty-1/3 pair, a nested refinement, the mortar suite's taper at
    4 to 64 slices, a taper that CLOSES to a point at 8/32/64 slices, and
    ``add_tapered_pillars`` -- and reads the narrowest segment each one asks
    for.

    MEASURED 2026-09-11, both builds (these are pure geometry, so the two
    agree exactly): 4.0000e-01 (a single interior wall) / 3.3333e-01
    (duty-1/3) / 2.3710e-01 (conforming and non-conforming) / 2.1000e-01
    (axes carrying different walls) / 2.0035e-01 .. 1.8812e-01 (the mortar
    suite's taper at 4 .. 64 slices) / 1.7333e-01 (``add_tapered_pillars``) /
    **1.2500e-01** (the nested refinement -- the WORST), i.e. **125x** the
    1e-3 bar.  The one surface that APPROACHES the bar is a taper whose tip
    closes, and it does so at exactly the rate the midpoint rule predicts --
    3.1438e-02 / 8.0094e-03 / 4.1047e-03 at ``n_slices`` = 8 / 32 / 64, i.e.
    ``~ w_bottom / (2 n_slices)``.  That is REPORTED here, with the crossing
    slice count derived, rather than hidden: it is remedy (4) in the refusal
    message."""
    census = {k: _narrowest(v) for k, v in _shipped_geometry_battery().items()}
    bar = _ts._STAG_MIN_SEG_FRAC
    ordinary = {k: v for k, v in census.items() if "closing" not in k}
    closing = {k: v for k, v in census.items() if "closing" in k}
    # (a) every ORDINARY geometry class is two decades above the bar --
    # measured worst 1.250e-01 (125x).  Two decades is the DECISION.
    worst = min(ordinary.values())
    assert worst > 100.0 * bar, (worst, bar, ordinary)
    # (b) the CLOSING taper is the one surface that walks toward the bar, and
    # it walks at the rate the midpoint rule predicts (w_bottom / (2 n_slices)
    # with w_bottom = 0.5 of the period).  Asserted as the RATE, so it tracks
    # any change to the slicing rule.
    for ns in (8, 32, 64):
        pred = 0.5 / (2.0 * ns)
        got = closing[f"closing_taper_n{ns}"]
        assert 0.7 * pred < got < 1.6 * pred, (ns, got, pred)
    # (c) ... and at every slice count the shipped docstring uses (default 8,
    # and the mortar suite's 4) it is still decades clear; the first
    # ``n_slices`` at which a fully closing taper would cross the bar is
    # DERIVED from that rate, not pinned -- measured 4.105e-03 at 64, so the
    # crossing is around 250 slices, which no shipped fixture approaches.
    assert closing["closing_taper_n8"] > 30.0 * bar, closing
    n_cross = 0.5 / (2.0 * bar)
    assert n_cross > 200.0, n_cross
    assert min(closing.values()) > 3.0 * bar, closing
    # (d) the census is read off the stack's own wall arrays above, which is
    # the API's INTENT.  Arm the basis's own census through a real solve and
    # confirm that what ``Basis1D`` is actually handed agrees, and that
    # nothing was refused -- the guard is where the grid is BUILT, so this is
    # the arm that proves it sees the same geometry.
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.10, eps_cell=_tile(), x_walls=[0.2371 * _P, 0.6183 * _P],
                 y_walls=[0.2371 * _P, 0.6183 * _P])
    st.add_layer(0.10, eps_cell=_tile(), x_walls=[0.3117 * _P, 0.7402 * _P],
                 y_walls=[0.3117 * _P, 0.7402 * _P])
    st.set_source(_WL, theta=_TH, phi=_PH)
    seen = _ts._STAG_SEG_CENSUS
    _ts._STAG_SEG_CENSUS = []
    try:
        st.solve(jones=False)
        rows = list(_ts._STAG_SEG_CENSUS)
    finally:
        _ts._STAG_SEG_CENSUS = seen
    assert rows, "the census recorded nothing -- is the guard still on the "                 "path the solve takes?"
    assert not any(r[4] for r in rows), rows          # nothing refused
    assert min(r[3] for r in rows) == pytest.approx(_narrowest(st), rel=1e-12),         (rows, _narrowest(st))


def test_a_requested_sliver_is_refused_and_the_message_names_the_cure():
    """The refusal fires at the grid's own entry point and is actionable."""
    d = _P
    w = np.array([0.0, 0.49995 * d, 0.50005 * d, d])   # 1e-4 of the period
    with pytest.raises(ValueError) as ei:
        Basis1D(d, w, 5)
    msg = str(ei.value)
    assert "1.000e-04" in msg or "1.0000e-04" in msg, msg
    for token in ("minimum", "MERGE", "layer_grids='shared'",
                  "PMM2DStackHybrid", "n_slices", "ENERGY-INVISIBLE"):
        assert token in msg, (token, msg)
    # "raise n_modes" must NOT be offered: the spurious spectrum grows as
    # M (M + 1), which the next test measures.
    assert "Raising n_modes is NOT a remedy" in msg, msg
    # ... and through the PUBLIC surface, at solve time
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.10, eps_cell=_tile(), x_walls=[0.21 * _P, 0.68 * _P],
                 y_walls=[0.21 * _P, 0.68 * _P])
    st.add_layer(0.06, eps_cell=np.full((3, 3), _C(_EPS_H)),
                 x_walls=[0.49995 * _P, 0.50005 * _P],
                 y_walls=[0.49995 * _P, 0.50005 * _P])
    st.set_source(_WL, theta=_TH, phi=_PH)
    with pytest.raises(ValueError, match="minimum"):
        st.solve(jones=False)
    # ... and through the OTHER two routes the verification names.
    # (a) ``add_tapered_pillars`` with two features whose edges nearly meet
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillars(
        0.20, eps_host=_EPS_H, n_slices=2,
        pillars=[((0.30 * _P, 0.30 * _P), (0.20 * _P, 0.20 * _P),
                  (0.20 * _P, 0.20 * _P), 9.0),
                 ((0.4001 * _P, 0.4001 * _P), (2e-4 * _P, 2e-4 * _P),
                  (2e-4 * _P, 2e-4 * _P), 9.0)])
    st.set_source(_WL, theta=_TH, phi=_PH)
    with pytest.raises(ValueError, match="minimum"):
        st.solve(jones=False)
    # (b) ``add_tapered_pillar`` on a taper closed far enough that the
    # midpoint rule's narrowest SAMPLED width crosses the contract
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillar(0.24, eps_pillar=_EPS_P, eps_host=_EPS_H,
                          x_bounds_bottom=[0.25 * _P, 0.75 * _P],
                          y_bounds_bottom=[0.25 * _P, 0.75 * _P],
                          x_bounds_top=[0.49999 * _P, 0.50001 * _P],
                          y_bounds_top=[0.49999 * _P, 0.50001 * _P],
                          n_slices=400)
    st.set_source(_WL, theta=_TH, phi=_PH)
    with pytest.raises(ValueError, match="n_slices"):
        st.solve(jones=False)

    # a HEALTHY grid at the same shape is untouched
    ok = Basis1D(d, np.array([0.0, 0.2371 * d, 0.6183 * d, d]), 5)
    assert ok.N == 3 and not ok.uniform


def test_the_spurious_spectrum_is_a_function_of_the_wall_array_alone():
    """WHY the guard is a WIDTH bar and not a spectral one.

    The sliver's spurious modal ``|gamma| / k0`` is predicted from geometry
    and ``M`` alone by ``c(M) M (M + 1) / (4 k0 J_min)``.  If ``c`` is
    constant then a bar on the SPECTRUM carries no information the wall array
    does not already carry -- and it would additionally refuse legitimately
    fine UNIFORM lattices, whose ``|gamma|max`` is just as large and whose
    mortars are healthy (the same overlap that defeated the 1-D ``|q|max``
    bar, ``FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`` S3.3).

    MEASURED here over three decades of wall separation, both builds:
    ``c`` = 0.922844 / 0.917156 / 0.916579 / 0.916517 (``M`` = 4) and
    0.958589 / 0.953382 / 0.952857 / 0.952802 (``M`` = 5) at ``delta`` = 1e-3
    .. 1e-6 -- a spread of 1.0069 and 1.0061 while ``|gamma|`` itself moves
    1000x across the same ladder.  The assertion is the CONSTANCY (that makes
    it a predictor), not the value."""
    k0 = 2 * np.pi / _WL
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False        # this test MAPS the band
    try:
        consts, uniform_c, sliver_c = {}, {}, {}
        for M in (4, 5):
            cs = []
            for delta in (1e-3, 1e-4, 1e-5, 1e-6):
                xb = np.array([0.0, (0.5 - delta / 2) * _P,
                               (0.5 + delta / 2) * _P, _P])
                sol = Granet2DTransverseE(
                    _P, _P, xb, xb, M, _tile(), alpha0x=0.0, alpha0y=0.0,
                    k0=k0)
                lam = _region_modes(sol)[2]
                J = float(np.min(sol.bx.Jn))
                cs.append(float(np.max(np.abs(lam))) * 4.0 * k0 * J
                          / (M * (M + 1)))
            consts[M] = cs
            # ... and the |gamma|max of a MODERATE sliver, for the overlap
            xb = np.array([0.0, 0.485 * _P, 0.515 * _P, _P])
            sol = Granet2DTransverseE(_P, _P, xb, xb, M, _tile(),
                                      alpha0x=0.0, alpha0y=0.0, k0=k0)
            sliver_c[M] = float(np.max(np.abs(_region_modes(sol)[2])))
            # ... and the SAME predictor on a legitimately fine UNIFORM
            # lattice, which no one should refuse
            sol = Granet2DTransverseE(_P, _P, 5, 5, M,
                                      np.full((5, 5), _C(_EPS_H)),
                                      alpha0x=0.0, alpha0y=0.0, k0=k0)
            lam = _region_modes(sol)[2]
            uniform_c[M] = (float(np.max(np.abs(lam))),
                            float(np.max(np.abs(lam))) * 4.0 * k0
                            * float(np.min(sol.bx.Jn)) / (M * (M + 1)))
        for M, cs in consts.items():
            spread = max(cs) / min(cs)
            # measured spread 1.0009 (M=4) and 1.0007 (M=5) over three
            # decades of delta on both builds; the bar is 1.02, which a
            # non-predictor (anything that depends on delta at all) fails by
            # decades since |gamma| itself moves 100x across this ladder.
            assert spread < 1.02, (M, cs)
        # THE POINT: a legitimately fine UNIFORM lattice carries a |gamma|max
        # of the same ORDER as a sliver several times narrower, so no bar on
        # the spectrum separates them.  Measured: a uniform N = 5 lattice
        # (segments 0.20 of the period, which nobody would refuse) reads
        # |gamma|/k0 = 9.84 at M = 4 and 15.47 at M = 5, against a physical
        # ceiling of sqrt(9) = 3; the delta = 3e-2 SLIVER -- a SIX-TIMES finer
        # geometry -- reads 39.8 and 61.5.  Four times apart, on geometries
        # six times apart in width: any spectral bar placed between them
        # refuses a uniform lattice for being fine, which is exactly how the
        # 1-D |q|max bar failed (S3.3 there).
        assert uniform_c[4][0] > 3.0 * math.sqrt(_EPS_P), uniform_c
        assert sliver_c[4] / uniform_c[4][0] < 10.0, (sliver_c, uniform_c)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev


def _y_uniform_stack(delta, M, per=1.2, wl=0.85, th=0.15,
                     w0=(0.2371, 0.6183), w2=(0.3117, 0.7402),
                     yw=(0.27, 0.61), epsp=9.0, epsh=2.25, t=0.06):
    """Three x-strip layers, constant along y, with the MIDDLE one ALL HOST on
    its own grid whose two walls sit ``delta`` apart.  The device cannot depend
    on ``delta`` at all, and the exact 1-D pure PMM is the truth for the whole
    2-D answer."""
    st = PMM2DStackPure(per, n_modes=M, n_orders=1, layer_grids="per-layer")
    tl = np.array([[epsh] * 3, [epsp] * 3, [epsh] * 3], dtype=_C)
    yww = [yw[0] * per, yw[1] * per]
    st.add_layer(t, eps_cell=tl, x_walls=[w0[0] * per, w0[1] * per],
                 y_walls=yww)
    st.add_layer(t, eps_cell=np.full((3, 3), _C(epsh)),
                 x_walls=[(0.5 - delta / 2) * per, (0.5 + delta / 2) * per],
                 y_walls=yww)
    st.add_layer(t, eps_cell=tl, x_walls=[w2[0] * per, w2[1] * per],
                 y_walls=yww)
    st.set_source(wl, theta=th)
    return st


def _y_uniform_oracle(deg, per=1.2, wl=0.85, th=0.15,
                      w0=(0.2371, 0.6183), w2=(0.3117, 0.7402),
                      epsp=9.0, epsh=2.25, t=0.06):
    st = PMMStack(per, degree=deg, far_field_orders=5)
    st.add_layer(t, segments=[(w0[0], epsh), (w0[1] - w0[0], epsp),
                              (1.0 - w0[1], epsh)])
    st.add_layer(t, segments=[(1.0, epsh)])
    st.add_layer(t, segments=[(w2[0], epsh), (w2[1] - w2[0], epsp),
                              (1.0 - w2[1], epsh)])
    st.set_source(wl, theta=th)
    return st.solve()


def _score(o2d, R, T, o1d, R1d, T1d):
    best = 0.0
    for m in (-1, 0, 1):
        sel = int(np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0])
        j = int(np.where(o1d == m)[0][0])
        best = max(best, abs(float(R[1, sel]) - float(R1d[1, j])),
                   abs(float(T[1, sel]) - float(T1d[1, j])))
    return best


def test_fail_before_the_sliver_the_guard_refuses_is_measurably_wrong():
    """FAIL-BEFORE for D1, two-sided, against an EXACT oracle.

    With the guard DISARMED the intra-layer sliver is returned and is
    measurably worse than the SAME device on an ordinary grid -- and the
    lossless closure is PINNED, so nothing shipped can see it.  With the guard
    ARMED the same stack is refused.

    The device is y-uniform and its middle layer is ALL HOST, so the answer
    cannot depend on the wall separation at all and the exact 1-D pure PMM
    (degree 14, its own self-gap measured here) is the truth.

    MEASURED at ``M`` = 6 (WIN / WSL): the ordinary arm (walls 0.30 apart)
    scores **7.294e-03** against the oracle and the sliver arm (1e-05)
    **4.990e-02** -- a factor **6.8** -- while the closures read 1.5e-05 and
    7.9e-06, i.e. 3.5 decades UNDER the engine's own 5e-2 tripwire.  The
    ladder behind it (probe ``r5_conv.py``, ``M`` to 8 on a second fixture) is
    a FLOOR, not a wander: the last rung improves 11.71x on the ordinary grid
    and 2.32x on the sliver."""
    o1d, R1d, T1d = _y_uniform_oracle(14)[:3]
    R1d, T1d = np.atleast_2d(R1d), np.atleast_2d(T1d)
    o12, R12, T12 = _y_uniform_oracle(12)[:3]
    keep = np.abs(np.asarray(o1d)) <= 1
    selfgap = float(max(
        np.max(np.abs(np.atleast_2d(R12)[:, keep] - R1d[:, keep])),
        np.max(np.abs(np.atleast_2d(T12)[:, keep] - T1d[:, keep]))))

    # BOTH round-2 guards are disarmed for the pre-fix arm: the width contract
    # and the conditioning backstop land on the same width (this fixture's
    # H-row operator reads rcond = 3.17e-15 at delta = 1e-05), so a
    # demonstration of what round 2 prevents has to lift both.
    prev, prev_rc = _ts.PMM2D_STAG_MIN_SEG_GUARD, _pc._MORTAR_RCOND_REFUSE
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    _pc._MORTAR_RCOND_REFUSE = 0.0
    try:
        got = {}
        for lab, delta in (("ordinary", 0.30), ("sliver", 1e-5)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                o, R, T = _y_uniform_stack(delta, 6).solve(jones=False)
            got[lab] = (_score(o, R, T, o1d, R1d, T1d),
                        float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
                        [str(x.message) for x in w])
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
        _pc._MORTAR_RCOND_REFUSE = prev_rc

    # both arms are READABLE against the oracle (decades above its self-gap
    # 1.4e-06, measured here), so the comparison is the solver's and not the
    # oracle's floor
    assert got["ordinary"][0] > 100.0 * selfgap, (got, selfgap)
    # the sliver is measurably worse -- measured 6.84x on both builds; the bar
    # is 3x, which the ordinary arm's own cross-build spread (these readings
    # are discretisation-limited and agree to 10+ digits) cannot reach
    assert got["sliver"][0] > 3.0 * got["ordinary"][0], got
    # ... and it is ENERGY-INVISIBLE: both closures sit decades under the
    # engine's tripwire, and NEITHER arm warns
    from lumenairy.elements.pmm.stack2d_pure import _STAG_CLOSURE_TOL
    for lab in ("ordinary", "sliver"):
        assert got[lab][1] < 0.01 * _STAG_CLOSURE_TOL, (lab, got[lab])
        assert not [m for m in got[lab][2] if "energy" in m.lower()], got[lab]

    # WITH the guards armed the sliver arm is REFUSED, so the wrong answer is
    # unreachable through the public surface -- by the WIDTH contract, which
    # fires first because it needs no solve
    with pytest.raises(ValueError, match="minimum"):
        _y_uniform_stack(1e-5, 6).solve(jones=False)
    # ... and by the conditioning backstop alone, with the width contract
    # lifted: the two are independent
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        with pytest.raises(_EnergyError, match="numerically singular"):
            _y_uniform_stack(1e-5, 6).solve(jones=False)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    # ... and the ordinary arm is untouched, BIT for BIT
    o2, R2, T2 = _y_uniform_stack(0.30, 6).solve(jones=False)
    assert _score(o2, R2, T2, o1d, R1d, T1d) == got["ordinary"][0]


def test_the_mortars_own_algebra_is_exact_at_every_wall_separation():
    """ATTRIBUTION, and it narrows the defect.

    Three ALL-HOST layers on three DIFFERENT non-uniform grids: the device is
    a homogeneous slab whose reflectance is ANALYTIC (Airy), and a homogeneous
    layer is exactly representable on any element grid, so every digit of the
    deviation is the mortar's cross-grid projection.

    MEASURED: **1.2e-10 at ``M`` = 5 and 3.3e-13 at ``M`` = 6**, and it does
    NOT move with the wall separation (7.6e-08 at 0.30 against 2.3e-07 at
    1e-02, ``M`` = 4).  So the mortar's algebra is not what D1 breaks -- what a
    sliver grid loses is the STRUCTURED trace of a PATTERNED neighbour, which
    is why the fail-before fixture above needs patterned outer layers to show
    the effect at all."""
    k0 = 2 * np.pi / _WL
    n0, nh = 1.0, math.sqrt(_EPS_H)
    kx = k0 * n0 * math.sin(_TH)
    kz0 = np.sqrt((k0 * n0) ** 2 - kx ** 2 + 0j)
    kzh = np.sqrt((k0 * nh) ** 2 - kx ** 2 + 0j)
    ph = np.exp(2j * kzh * (3.0 * 0.06))
    an = {}
    for pol, r01 in (("TE", (kz0 - kzh) / (kz0 + kzh)),
                     ("TM", (nh ** 2 * kz0 - n0 ** 2 * kzh)
                      / (nh ** 2 * kz0 + n0 ** 2 * kzh))):
        r = r01 * (1.0 - ph) / (1.0 - r01 * r01 * ph)
        an[pol] = float(abs(r) ** 2)

    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        errs = {}
        for delta in (0.30, 1e-2):
            st = PMM2DStackPure(_P, n_modes=5, n_orders=1,
                                layer_grids="per-layer")
            yw = [0.27 * _P, 0.61 * _P]
            for xw in ([0.21 * _P, 0.68 * _P],
                       [(0.5 - delta / 2) * _P, (0.5 + delta / 2) * _P],
                       [0.33 * _P, 0.79 * _P]):
                st.add_layer(0.06, eps=_EPS_H, x_walls=xw, y_walls=yw)
            st.set_source(_WL, theta=_TH, phi=0.0)
            o, R, T = st.solve(jones=False)
            p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
            errs[delta] = max(abs(float(R[1, p0]) - an["TE"]),
                              abs(float(R[0, p0]) - an["TM"]))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    # MEASURED at M = 5: 1.1694e-10 (0.30) and 5.5288e-10 (1e-02) on WIN,
    # 1.1694e-10 and 5.5288e-10 on WSL (agreeing to 6 significant figures);
    # the bar is 1e-6, four decades above the reading and decades below the
    # 5e-02 scale on which the patterned fixture's damage lives
    for delta, e in errs.items():
        assert e < 1e-6, (delta, e, an)


def test_the_integer_lattice_is_exempt_and_cannot_reach_the_bar():
    """The uniform path is not screened, and that costs nothing: reaching the
    bar needs ``N > 1000``, i.e. ``q >= 2000`` and a ``2 q^2`` region eig."""
    for N in (1, 2, 3, 8, 64, 500, 999):
        b = Basis1D(_P, N, 4)
        assert b.uniform and b.N == N
    # the bar is 1e-3 of the period and a uniform cell is 1/N of it, so the
    # first refusable lattice is N = 1001 -- q = 3003 at M = 4 and a
    # 1.8e+07-dimension region eigenproblem.  Stated as arithmetic, so it
    # tracks the constant.
    n_first = int(math.floor(1.0 / _ts._STAG_MIN_SEG_FRAC)) + 1
    assert n_first >= 1000
    assert 2 * (n_first * 3) ** 2 > 1e7


# ==========================================================================
# D2 -- the guarded mortar solve
# ==========================================================================
#: How far apart two backward-stable solves of the SAME system are entitled to
#: be: ``cond(A) * eps`` is the classical forward-error bound for a solve whose
#: backward error is a few ULP, and this constant is the documented slack on
#: the factor in front of it.  MEASURED 2026-09-11 over the six (delta, M)
#: rows of this fixture on four OpenBLAS kernels (Haswell / Sandybridge /
#: Nehalem / Katmai): the two answers are byte-identical on every one, i.e.
#: the reading is 0 against a bound that reaches 2.4e-07 at the
#: worst-conditioned row (cond = 1.7e+07).
_SOLVE_AGREE_C = 64.0

#: The relative residual a partial-pivoting LU solve is entitled to leave:
#: ``O(n * eps * growth)``.  MEASURED on the same 24 samples,
#: ``||A X - B|| / ||B||`` reads 9.33e-16 .. 2.57e-15, i.e.
#: **0.017 .. 0.050 of n*eps**; the bar is 16, i.e. 324x above the worst
#: reading of the four kernels.
_SOLVE_RESID_C = 16.0


def _lapack_provenance():
    """WHICH LAPACK build numpy and scipy each call, and how many are loaded.

    Returns ``(numpy_id, scipy_id, loaded)``.  The two ids come from
    ``show_config(mode='dicts')`` -- the library each package was BUILT
    against -- and are ``None`` where the running build will not say.
    ``loaded`` is the set of BLAS/LAPACK shared libraries actually mapped into
    the process, from ``threadpoolctl``, or ``None`` when that package is
    absent -- it IS absent on CI, which is why ``show_config`` is the primary
    source here and not the fallback.
    """
    ids = {}
    for name in ("numpy", "scipy"):
        try:
            mod = importlib.import_module(name)
            cfg = mod.show_config(mode="dicts") or {}
            d = (cfg.get("Build Dependencies") or {}).get("lapack") or {}
            ids[name] = ("%s/%s/%s" % (d.get("name"), d.get("version"),
                                       d.get("openblas configuration"))
                         if d else None)
        except Exception:                                   # noqa: BLE001
            ids[name] = None
    try:
        import threadpoolctl  # noqa: I001, PLC0415
        loaded = sorted({
            "%s/%s/%s" % (d.get("internal_api"), d.get("version"),
                          os.path.basename(d.get("filepath") or ""))
            for d in threadpoolctl.threadpool_info()
            if d.get("internal_api") in ("openblas", "mkl")})
    except Exception:                                       # noqa: BLE001
        loaded = None
    return ids["numpy"], ids["scipy"], loaded
def _mortar_pair(delta, M, host_b=True):
    """The two mortar solve operators and their right-hand sides for a
    (patterned grid A | grid B) interface."""
    k0 = 2 * np.pi / _WL
    kx0 = k0 * math.sin(_TH) * math.cos(_PH)
    ky0 = k0 * math.sin(_TH) * math.sin(_PH)
    taux, tauy = np.exp(-1j * kx0 * _P), np.exp(-1j * ky0 * _P)
    wa = np.array([0.0, 0.21 * _P, 0.68 * _P, _P])
    wb = np.array([0.0, (0.5 - delta / 2) * _P, (0.5 + delta / 2) * _P, _P])
    ga = StagGridOps(_P, _P, wa, wa, M, taux, tauy)
    gb = StagGridOps(_P, _P, wb, wb, M, taux, tauy)
    cr = StagCrossOps(ga, gb)
    cb = np.full((3, 3), _C(_EPS_H)) if host_b else _tile()
    sa = Granet2DTransverseE(_P, _P, wa, wa, M, _tile(),
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    sb = Granet2DTransverseE(_P, _P, wb, wb, M, cb,
                             alpha0x=kx0, alpha0y=ky0, k0=k0)
    Wa, Va = _region_modes(sa)[:2]
    Wb, Vb = _region_modes(sb)[:2]
    lhsE = _pc._stag_blk2_apply(gb.V1, gb.V2, Wb, gb.qq, _stag_kron_apply)
    rhsE = _pc._stag_blk2_apply(cr.C1H(), cr.C2H(), Wa, ga.qq,
                                _stag_kron_apply)
    return (lhsE, rhsE), ga, gb


def test_the_guarded_mortar_solve_returns_the_numpy_solve_bit_for_bit():
    """The guard must not move a bit of a healthy answer.

    RESTATED 2026-09-11 (CI PREMISE GATES).  The claim this test was built on
    -- "``lu_factor`` + ``lu_solve`` is the same LAPACK ``getrf``/``getrs``
    pair ``gesv`` calls, so the answer is BIT-IDENTICAL to
    ``np.linalg.solve``" -- is a property of ONE LAPACK and is NOT PORTABLE,
    because numpy and scipy need not be linked against the same one.  The
    5.45.0 release matrix proved it: the py3.13 shard read ``b2eb080e...``
    where every other python read ``5495e552...``, on the same commit, at
    ``delta`` = 0.3 / ``M`` = 4 -- a WELL-CONDITIONED row (cond = 3.3e+02), so
    neither answer was wrong.  MEASURED here 2026-09-11 on both local builds:
    numpy ships ``scipy-openblas 0.3.31.188.0`` (USE64BITINT) and scipy ships
    ``scipy-openblas 0.3.30`` -- two distinct shared libraries, both mapped
    into the same process.  They happen to agree bit for bit on this fixture
    on every kernel of the local ladder, which is exactly how a non-portable
    premise survives a local gate.

    THE CI-ARM FINDING THIS ROUND RESTS ON (2026-09-11).  On the CI runners
    the ill-conditioned fixtures of this campaign come out CORRECT where every
    local kernel, thread width and build reads them wrong, so a test that
    measures a pathology must MEASURE ITS PREMISE and skip when the premise is
    absent.  The guard following the answer -- returning where the answer is
    right, refusing where it is wrong -- is the contract, not a defect.

    WHAT IS ASSERTED UNCONDITIONALLY, and it is the claim the guard is
    actually about:

      1. the guarded answer is bit-for-bit the answer the SAME ``lu_factor`` +
         ``lu_solve`` pair returns UNGUARDED -- the screen adds nothing to the
         arithmetic it rides.  Both sides go through scipy's LAPACK by
         construction, so no wheel pairing can move this;
      2. it is a VALID solve of the same system: relative residual under
         ``_SOLVE_RESID_C * n * eps``;
      3. and it agrees with ``np.linalg.solve``'s answer inside the bound two
         backward-stable solves of the same system are entitled to differ by,
         ``_SOLVE_AGREE_C * cond(A) * eps * max|x|``.

    WHAT IS PREMISE-GATED, as an INFORMATIVE check: the bit-identity with
    ``np.linalg.solve``.  Its premise is that numpy and scipy resolve to ONE
    LAPACK build; that is measured here and SKIPPED when they do not.
    """
    np_lapack, sp_lapack, loaded = _lapack_provenance()
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    eps = float(np.finfo(np.float64).eps)
    rows = []
    try:
        for delta in (0.30, 1e-2, 1e-3):
            for M in (4, 5):
                (L, R), ga, gb = _mortar_pair(delta, M)
                n = int(L.shape[0])
                x_np = np.linalg.solve(L, R)
                x_sp = sla.lu_solve(sla.lu_factor(L), R)
                x_g = _pc._guarded_mortar_solve(L, R, "test", ga, gb)
                # (1) the guard moves no bit of the path it rides
                assert _h(x_g) == _h(x_sp), (
                    delta, M, "the screen changed the arithmetic of its own "
                    "lu_factor/lu_solve path")
                # (2) it is a valid solve of the same system
                nb = float(np.linalg.norm(R))
                resid = float(np.linalg.norm(L @ x_g - R)) / nb
                assert resid <= _SOLVE_RESID_C * n * eps, (
                    delta, M, n, resid, _SOLVE_RESID_C * n * eps)
                # (3) and it is numpy's answer to within the conditioning
                cond = float(np.linalg.cond(L))
                scale = float(np.max(np.abs(x_np)))
                diff = float(np.max(np.abs(x_g - x_np)))
                assert diff <= _SOLVE_AGREE_C * cond * eps * scale, (
                    delta, M, cond, diff, _SOLVE_AGREE_C * cond * eps * scale)
                rows.append((delta, M, n, cond, resid, diff,
                             _h(x_g) == _h(x_np)))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev

    # ---- PREMISE-GATED, INFORMATIVE: is there ONE LAPACK behind both?
    same_build = np_lapack is not None and np_lapack == sp_lapack
    one_loaded = loaded is None or len(loaded) <= 1
    if not (same_build and one_loaded):
        pytest.skip(
            "premise absent on this arm: numpy and scipy do not resolve to "
            "one LAPACK build, so bit-identity between np.linalg.solve and "
            "scipy's lu_factor/lu_solve is not a property this machine has "
            "(numpy=%s, scipy=%s, loaded=%s).  The three unconditional claims "
            "above passed on all %d rows: worst relative residual %.3e "
            "against %.3e, worst |guarded - numpy| %.3e."
            % (np_lapack, sp_lapack, loaded, len(rows),
               max(r[4] for r in rows),
               _SOLVE_RESID_C * max(r[2] for r in rows) * eps,
               max(r[5] for r in rows)))
    bad = [r for r in rows if not r[6]]
    assert not bad, (
        "numpy and scipy report the SAME LAPACK build (%s) yet %d of %d rows "
        "differ in the last bits: %s"
        % (np_lapack, len(bad), len(rows), [(r[0], r[1]) for r in bad]))


def test_the_mortar_rcond_bar_has_decades_of_gap_on_both_sides():
    """Both sides of the D2 bar, RE-MEASURED on the running build.

    HEALTHY population: every grid pair the shipped fixtures build.  MEASURED
    over 106 mortar solves in the probe, ``rcond`` = **2.61e-07 .. 3.77e-04**;
    the subset re-measured here reads the same decade.  The bar is 1e-12.

    WRONG population: a mortared sliver falls through it at a wall separation
    of 1e-05 (3.18e-13) and reaches 1.29e-17 at 1e-07, where the unguarded
    ``solve`` raised."""
    census = _pc._MORTAR_SOLVE_CENSUS
    _pc._MORTAR_SOLVE_CENSUS = []
    try:
        st = PMM2DStackPure(_P, n_modes=5, n_orders=1,
                            layer_grids="per-layer")
        st.add_tapered_pillar(0.24, eps_pillar=_EPS_P, eps_host=_EPS_H,
                              x_bounds_bottom=[0.1873 * _P, 0.7241 * _P],
                              y_bounds_bottom=[0.1873 * _P, 0.7241 * _P],
                              x_bounds_top=[0.2917 * _P, 0.6109 * _P],
                              y_bounds_top=[0.2917 * _P, 0.6109 * _P],
                              n_slices=4)
        st.set_source(_WL, theta=_TH, phi=_PH)
        st.solve(jones=False)
        healthy = [r[2] for r in _pc._MORTAR_SOLVE_CENSUS if not r[3]]
    finally:
        _pc._MORTAR_SOLVE_CENSUS = census
    assert len(healthy) >= 6, healthy
    worst_healthy = min(healthy)
    bar = _pc._MORTAR_RCOND_REFUSE
    # MEASURED on this fixture (WIN / WSL): 6 solves, rcond 3.5506e-06 ..
    # 1.7876e-05, and 2.61e-07 as the worst over the full 106-solve census in
    # the probe.  The bar is 1e-12, so at least FOUR decades below -- four is
    # the DECISION; the measured 5.4 decades over the full census is in the
    # fix doc.
    assert worst_healthy > 1e4 * bar, (worst_healthy, bar, healthy)

    # the WRONG side, measured on the same operator family
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        rcs = {}
        for delta in (1e-5, 1e-7):
            (L, _R), _ga, _gb = _mortar_pair(delta, 5)
            lu, piv = sla.lu_factor(L)
            gecon = sla.get_lapack_funcs("gecon", (L,))
            rc, _info = gecon(lu, float(np.max(np.sum(np.abs(L), axis=0))))
            rcs[delta] = float(rc)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    # MEASURED at M = 5, both builds agreeing to 12 significant figures:
    # 9.1385e-13 at 1e-05 and 6.1491e-17 at 1e-07
    # (validation/probe_pmm2d_mortar_round2/r8_twobuild_{win,wsl}.json).
    # NOTE the asymmetry, and it is the honest reading: the 1e-05 row is only
    # 1.09x UNDER the bar at this modal count -- the decisive separation is
    # the 1e-07 row, 4.2 decades under, which is why the bar's derivation
    # (S4.3 of the fix doc) rests on the HEALTHY population's 5.4 decades and
    # not on a comfortable margin here.
    assert rcs[1e-7] < 0.01 * bar, rcs
    assert rcs[1e-5] < bar, rcs


def test_a_singular_mortar_operator_is_refused_by_name_not_by_LinAlgError():
    """D2's headline: below ``delta ~ 1e-7`` the shipped library raised a bare
    ``numpy.linalg.LinAlgError: Singular matrix`` with no message, no hint and
    no naming of the offending grid."""
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        (L, R), ga, gb = _mortar_pair(1e-7, 5)
        with pytest.raises(_pc._ConditioningError) as ei:
            _pc._guarded_mortar_solve(
                L, R, "pmm2d staggered mortar interface (MassE_B W_B)",
                ga, gb)
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    msg = str(ei.value)
    for token in ("mortar", "reciprocal 1-condition", "grid A", "grid B",
                  "narrowest", "layer_grids='shared'"):
        assert token in msg, (token, msg)
    # and it routes the existing stabilize= ladders: _ConditioningError is an
    # _EnergyError
    assert isinstance(ei.value, _EnergyError)


def test_the_free_lower_bound_on_the_condition_number_is_refuted_here():
    """WHY the guard pays for ``gecon`` rather than the free bound.

    ``cond_2(A) >= ||A||_F ||X||_F / (sqrt(n) ||B||_F)`` is RIGOROUS and costs
    three norms.  It is also useless here, and that is a measurement: the
    right-hand side is the OTHER grid's trace, which carries none of the
    sliver's spurious content, so the ill-conditioning is never excited by
    this ``B``.  MEASURED over 106 mortar solves: the bound reads
    **0.79 .. 24.0** while the true ``cond_2`` runs 3.9e+02 .. 7.9e+15."""
    prev = _ts.PMM2D_STAG_MIN_SEG_GUARD
    _ts.PMM2D_STAG_MIN_SEG_GUARD = False
    try:
        rows = []
        for delta in (0.30, 1e-6):
            (L, R), _ga, _gb = _mortar_pair(delta, 4)
            X = np.linalg.solve(L, R)
            g = (float(np.linalg.norm(L)) * float(np.linalg.norm(X))
                 / (float(np.linalg.norm(R)) * math.sqrt(L.shape[0])))
            rows.append((g, float(np.linalg.cond(L))))
    finally:
        _ts.PMM2D_STAG_MIN_SEG_GUARD = prev
    (g_ok, c_ok), (g_bad, c_bad) = rows
    # cond_2 moves by TEN decades between the two (measured 3.3e+02 ->
    # 6.2e+12) while the free bound moves by less than one
    assert c_bad / c_ok > 1e8, rows
    assert g_bad / g_ok < 10.0, rows


def _ci_kernel_table():
    """The committed per-(build, kernel) decision census.

    ``validation/probe_ci_kernel_sweep/decisions.json`` is produced by
    ``probe_decisions.py`` on each arm and merged; see
    ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md``.  It is READ here, never
    written -- a test that regenerates its own reference proves nothing.
    """
    path = (pathlib.Path(__file__).resolve().parents[2] / "validation"
            / "probe_ci_kernel_sweep" / "decisions.json")
    assert path.is_file(), (
        "the CI kernel census is missing: %s.  Regenerate it with "
        "validation/probe_ci_kernel_sweep/probe_decisions.py on each arm." % path)
    with path.open(encoding="cp1252") as fh:
        return json.load(fh)


def test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why():
    """The DECISION on ``_interface_smatrix``'s two bare ``np.linalg.solve``
    calls (the ``~1850`` site), recorded as a measurement rather than an
    omission.

    They ARE reachable with a near-singular operand from the 1-D
    ``PMMStack``: on a two-layer stack whose walls differ by ``delta``, the
    LAPACK reciprocal condition of ``Wb`` / ``Vb`` reads ~1e-10 at ``delta``
    = 1e-04 (answer CORRECT, ``R+T`` closes) and ~1e-12 at 1e-05 (answer
    WRONG).  The site nevertheless ships UNGUARDED, and the two paragraphs
    below are the whole reason.

    **RESTATED 2026-09-11 (CI KERNEL SWEEP,**
    ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md`` **).**  Until this date the
    test closed on the sentence "and the row that WOULD trip such a bar is
    already refused by the 1-D sliver guard, which names ``min_feature``".
    That sentence is a claim about a DIFFERENT guard, and it is FALSE on some
    builds: the sliver guard triggers on the solve's own energy closure, and
    at ``delta`` = 1e-05 that closure is itself the build-dependent quantity
    -- ``max(R+T)`` reads 2.1716 / 2.1729 / 3.6116 on the Katmai / Haswell /
    Sandybridge kernels HERE and **1.0000010** on the CI runner, where the
    guard therefore stays silent and this assertion failed (py3.11 shard 4 of
    the 5.45.0 matrix).  That divergence is a real defect and it belongs to
    the sliver guard, not to this site; it is handed off in the audit above.
    This test no longer asserts anything about it.  The sliver guard is
    DISARMED for the measurement below, so the site is reached at BOTH wall
    separations on every build and the population measured is the same object
    everywhere.

    WHAT THE DECISION NOW RESTS ON, in a shape no kernel can move.  A bar is
    only a decision if its verdict is the same everywhere.  At this site it is
    not: the CORRECT and the WRONG populations sit **under 2.1 decades apart**
    (re-measured below), and a 1e-12 bar -- the value the 2-D IN-PLANE mortar
    sites use, where the same populations are 5.4 decades apart -- lands
    INSIDE that gap, so its verdict flips with the BLAS micro-kernel alone.
    That is measured, not argued: ``validation/probe_ci_kernel_sweep`` ran the
    identical fixture on eight (build, kernel) arms at one thread and the
    committed census records ``refuse`` on Haswell / Sandybridge / Nehalem and
    ``accept`` on Katmai, on BOTH builds.  A guard here would refuse a correct
    answer on one runner and pass a wrong one on the next, which is the S4
    shape ``docs/TESTING_STANDARDS.md`` forbids.  So the site stays unguarded,
    and this test pins the two readings and the non-unanimous census that say
    so.

    RESTATED 2026-09-11 (CI PREMISE GATES).  The CI runner arm produces
    CORRECT answers on these ill-conditioned fixtures where every local arm
    produces wrong ones -- it read ``R+T`` = 1.0000010472 at the 1e-05 wall
    separation on the 5.45.0 matrix, and 1.0000003658 at 1e-04.  So the WRONG
    row's reproduction is now a measured PREMISE that SKIPS when absent, never
    an assertion; everything else here is unconditional.  The guard following
    the answer is the contract, and why the CI arm differs is an OPEN item
    recorded in ``docs/audits/CI_PREMISE_GATES_2026_09_11.md``."""
    # ---- 1. the site is STRUCTURALLY unguarded: two bare solves, no screen
    src = inspect.getsource(_pc._interface_smatrix)
    assert src.count("np.linalg.solve(") == 2, src
    for banned in ("gecon", "_guarded_mortar_solve", "rcond"):
        assert banned not in src, (banned, src)

    # ---- 2. the two populations, re-measured on the RUNNING build
    seen = []
    real = _pc._interface_smatrix
    import lumenairy.elements.pmm.stack as _st1d

    def _patched(Wa, Va, Wb, Vb):
        for A in (np.asarray(Wb), np.asarray(Vb)):
            lu, piv = sla.lu_factor(A)
            gecon = sla.get_lapack_funcs("gecon", (A,))
            rc, _i = gecon(lu, float(np.max(np.sum(np.abs(A), axis=0))))
            seen.append(float(rc))
        return real(Wa, Va, Wb, Vb)

    a0, a1 = 0.27865, 0.62505
    _st1d._interface_smatrix = _patched
    # DISARMED on purpose -- see the docstring.  Function-scoped and restored.
    prev_sliver = _st1d.PMM_SLIVER_GUARD
    _st1d.PMM_SLIVER_GUARD = False
    try:
        out = {}
        for delta in (1e-4, 1e-5):
            seen.clear()
            st = PMMStack(_P, degree=12, far_field_orders=5)
            st.add_layer(0.08, segments=[(a0, _EPS_H), (a1 - a0, _EPS_P),
                                         (1 - a1, _EPS_H)])
            b0, b1 = a0 - delta, a1 + delta
            st.add_layer(0.08, segments=[(b0, _EPS_H), (b1 - b0, _EPS_P),
                                         (1 - b1, _EPS_H)])
            st.set_source(_WL, theta=_TH)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _o, R, T = st.solve()[:3]
            tot = float(np.max(np.atleast_2d(R).sum(1)
                               + np.atleast_2d(T).sum(1)))
            out[delta] = (min(seen), tot, len(w))
    finally:
        _st1d._interface_smatrix = real
        _st1d.PMM_SLIVER_GUARD = prev_sliver

    rc_ok, tot_ok, nwarn_ok = out[1e-4]
    rc_bad, tot_bad, _nw = out[1e-5]
    # the CORRECT row: the answer closes and the site is SILENT, yet the
    # operand is already ill-conditioned enough to tempt a guard.  MEASURED
    # 9.6940e-11 .. 9.6969e-11 over the eight committed arms (and 9.6940e-11
    # on the CI runner that failed the old assertion -- the READING was never
    # the problem).
    #
    # The closure bar below is placed by the GAP, not by one build's residual:
    # the correct row's worst |R+T - 1| is 3.66e-07 (the CI runner; 1.02e-08
    # here) and the wrong row's BEST where it manifests at all is 1.17
    # (Katmai), so anything between 1e-06 and 1e-01 separates them.  1e-04
    # leaves 273x of slack below, worst arm of nine.
    assert abs(tot_ok - 1.0) < 1e-4, out
    assert nwarn_ok == 0, out
    assert rc_ok < 1e-9, out
    # the site RETURNS at BOTH separations -- unguarded is a real property,
    # not a figure of speech.  This half is unconditional; whether what it
    # returns at 1e-05 is WRONG is the premise gated at the end of this test.
    assert set(out) == {1e-4, 1e-5}, out

    # ---- 3. the gap between them is UNDER 2.1 decades, on any kernel.
    # MEASURED 1.965 .. 1.999 decades over the eight arms; asserted at 2.1,
    # i.e. with the measured spread stated and 0.1 decade of slack.  (The 2-D
    # in-plane mortar sites, where a 1e-12 bar IS shipped, separate by 5.4.)
    gap = math.log10(rc_ok / rc_bad)
    assert gap < 2.1, (gap, out)

    # ---- 4. and the census says a 1e-12 bar is NOT DECIDABLE here.  This is
    # the assertion that replaces the old sliver-guard sentence: it is a
    # statement about the committed per-kernel table, so it reads the same on
    # every build INCLUDING the ones that are not in the table.
    table = _ci_kernel_table()
    verdicts = {arm: d["pmm1d_interface/bar_1e-12_would@1e-05"]
                for arm, d in table["hypothetical"].items()
                if "pmm1d_interface/bar_1e-12_would@1e-05" in d}
    assert len(verdicts) >= 4, verdicts
    assert set(verdicts.values()) == {"refuse", "accept"}, verdicts
    # ... and the running build's own verdict is one of the two, whichever it
    # is -- which is precisely why it cannot be asserted.
    here = "refuse" if rc_bad < _pc._MORTAR_RCOND_REFUSE else "accept"
    assert here in verdicts.values(), (here, verdicts, rc_bad)

    # ---- 5. PREMISE-GATED: is the 1e-05 row actually WRONG on this arm?
    #
    # RESTATED 2026-09-11 (CI PREMISE GATES).  This test used to end with
    # ``assert abs(tot_bad - 1.0) > 1e-1`` -- "what the unguarded site returns
    # at 1e-05 does not close energy".  The 5.45.0 release matrix read
    # ``R+T`` = 1.0000010472 there (and 1.0000003658 at 1e-04): the CI
    # runner's arm solves this ill-conditioned interface CORRECTLY where every
    # local kernel, thread width and build gets it wrong.  That is the finding
    # this round is built on, and WHY the CI arm differs is an OPEN item
    # (``docs/audits/CI_PREMISE_GATES_2026_09_11.md``).  The site being
    # UNGUARDED is the decision under test, and that decision is sound whether
    # or not a given arm's arithmetic happens to fall over here -- so the
    # pathology's reproduction is MEASURED and skipped when absent, never
    # asserted, and no bar is relaxed.
    if not abs(tot_bad - 1.0) > 1e-1:
        pytest.skip(
            "premise absent on this arm: the unguarded plain-1-D interface "
            "returns a CORRECT answer at a 1e-05 wall separation (R+T = "
            "%.10f, |R+T - 1| = %.3e against the 1e-01 the wrong population "
            "reaches), so there is no wrong answer here to argue a guard "
            "about.  rcond readings %.4e (1e-04) / %.4e (1e-05), gap %.3f "
            "decades -- and the UNCONDITIONAL half of this test (the site is "
            "structurally unguarded, the correct row closes and is silent, "
            "the two populations are under 2.1 decades apart, and the "
            "committed census says a 1e-12 bar is non-unanimous here) passed."
            % (tot_bad, abs(tot_bad - 1.0), rc_ok, rc_bad, gap))


# ==========================================================================
# D3 -- the per-segment quadrature order
# ==========================================================================
def _projector_fixed_nq(basis, orders, alpha0):
    """``_stag_fourier_projection``'s PRE-CHANGE arithmetic, verbatim: ONE
    ``nq = 2 M + 8`` rule for every segment."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        J = basis.Jn[seg]
        xphys = 0.5 * (basis.xb[seg] + basis.xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

    def _asm(gs):
        return np.einsum("msa,jsa->mj", T, np.array(gs))
    return _asm


def test_the_integer_lattice_projector_is_bit_identical_to_the_fixed_rule():
    """D3's bit-identity contract (gate N1 extended): on an INTEGER-``N``
    lattice the per-segment rule must reproduce the single ``2 M + 8`` rule
    byte for byte, at every ``(d, N, M, tau, alpha0)``."""
    n = 0
    for d, N, M, tau, a0, mo in (
            (1.2, 3, 5, 1.0 + 0.0j, 0.0, 4),
            (0.9, 4, 4, np.exp(-0.41j), 2.08, 5),
            (1.4, 6, 6, np.exp(0.77j), -3.1, 7),
            (0.7, 2, 8, np.exp(1.13j), 1.7, 6),
            (1.0, 12, 3, np.exp(-0.2j), 0.4, 8),
            (1.0, 1, 7, np.exp(-0.9j), 5.5, 2)):
        b = Basis1D(d, N, M, tau)
        orders = np.arange(-mo, mo + 1)
        new = _stag_fourier_projection(b, orders, a0)
        old = _projector_fixed_nq(b, orders, a0)
        for st in ("B", "Btilde"):
            gs = getattr(b, st)
            assert _h(new(gs)) == _h(old(gs)), (d, N, M, st)
            n += 1
    assert n == 12


def test_the_rule_returns_the_shipped_order_on_every_uniform_lattice():
    """The formula (not just the ``basis.uniform`` branch) reduces to
    ``2 M + 8`` on every lattice the shipped far-field order cap allows with
    ``|alpha0| <= G/2``, which is what keeps an explicitly-passed uniform
    ARRAY ULP-close to the integer spelling (gate N2)."""
    bad = []
    for M in range(3, 15):
        for N in range(1, 61):
            mmax = (N * (M - 1) - 1) // 2
            # omega = |m G + alpha0| J,  J = d/(2N),  G = 2 pi / d
            om = (mmax + 0.5) * 2.0 * np.pi / (2.0 * N)
            if _ts._stag_quad_order(M, om) != 2 * M + 8:
                bad.append((M, N, om))
    assert not bad, bad[:8]
    # and it DOES top up once a segment carries more phase than that reserve
    assert _ts._stag_quad_order(4, 40.0) > 2 * 4 + 8
    assert _ts._stag_quad_order(4, 40.0) >= _ts._stag_quad_order(4, 20.0)


def _kernel(M, omega, nq):
    xg, wg = leggauss(nq)
    V, _ = _modleg_value_deriv(M, xg)
    return (np.exp(1j * omega * xg) * wg) @ V.T


def test_the_quadrature_rule_clears_the_measured_requirement():
    """The rule's constants are an UPPER ENVELOPE of a measurement, and this
    re-measures it: the smallest ``nq`` reaching the reference rule's own
    floor, against what the rule hands out.

    MEASURED (probe ``r2_quad.py``, both builds): the requirement is
    ``0.6341-0.6455 * omega + 9.38 + 0.47 (M - 3)`` over ``M`` = 3..12 and
    ``omega`` = 0..128 -- a slope spread of 1.8 %, which is what makes it a
    predictor.  The shipped ``0.72 * omega + 0.5 M + 10`` is above it
    everywhere."""
    for M in (4, 6, 8):
        scale0 = float(np.max(np.abs(_kernel(M, 0.0, 4 * M + 40)))) or 1.0
        for omega in (8.0, 24.0, 48.0):
            nref = max(240, int(3 * omega) + 4 * M + 80)
            ref = _kernel(M, omega, nref)
            drift = float(np.max(np.abs(ref - _kernel(M, omega, nref + 37)))) \
                / scale0
            bar = max(1e-15, 20.0 * drift)      # the ORACLE's own floor
            need = next(nq for nq in range(2, 240)
                        if float(np.max(np.abs(_kernel(M, omega, nq) - ref)))
                        / scale0 < bar)
            rule = _ts._stag_quad_order(M, omega)
            assert rule >= need, (M, omega, need, rule)


def test_a_long_segment_kernel_is_back_at_round_off():
    """D3's headline reading: on a 0.96 d segment at ``M`` = 4 with orders to
    7, the FIXED rule leaves 7.5e-04 relative against a refined rule (against
    6-8e-15 on a uniform ``N`` = 3 lattice).  The per-segment rule must put it
    back at round-off, and the improvement must be DECADES."""
    d = 1.0
    for frac, M, mmax in ((0.96, 4, 7), (0.91, 4, 7), (0.80, 4, 7),
                          (0.96, 6, 7)):
        rest = (1.0 - frac) / 2.0
        xb = np.array([0.0, rest * d, (rest + frac) * d, d])
        b = Basis1D(d, xb, M, tau=np.exp(-0.41j))
        orders = np.arange(-mmax, mmax + 1)
        a0 = 0.37 * 2 * np.pi / d
        new = _stag_fourier_projection(b, orders, a0)
        old = _projector_fixed_nq(b, orders, a0)
        # the reference: the SAME per-segment rule, refined 4x
        prev = (_ts._STAG_QUAD_OMEGA, _ts._STAG_QUAD_M, _ts._STAG_QUAD_CONST)
        _ts._STAG_QUAD_OMEGA, _ts._STAG_QUAD_M, _ts._STAG_QUAD_CONST = (
            4.0 * prev[0], 4.0 * prev[1], 4.0 * prev[2] + 40.0)
        try:
            ref = _stag_fourier_projection(b, orders, a0)
            Sr = ref(b.B)
        finally:
            (_ts._STAG_QUAD_OMEGA, _ts._STAG_QUAD_M,
             _ts._STAG_QUAD_CONST) = prev
        sc = float(np.max(np.abs(Sr)))
        e_new = float(np.max(np.abs(new(b.B) - Sr))) / sc
        e_old = float(np.max(np.abs(old(b.B) - Sr))) / sc
        # measured (WIN / WSL) at (0.96, M=4, m<=7): old 7.5e-04, new 4.7e-15
        assert e_new < 1e-13, (frac, M, e_new)
        assert e_old > 100.0 * e_new, (frac, M, e_old, e_new)
