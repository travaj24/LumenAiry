"""CURVED-CELL MAP, Phase E2, for the PURE staggered 2-D PMM: a DIFFERENT
coordinate map in every layer of ``PMM2DStackPure(layer_grids='per-layer')``,
two layers on different maps joined by the CURVED (non-separable) mortar of
:mod:`lumenairy.elements.pmm._curvemortar` -- gates E2-1 .. E2-8 of
``docs/audits/BUILD_PMM2D_CURVED_E2_2026_10_03.md`` (plan
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.5).

Words.  The CROSS-MASS is the overlap integral of one layer's basis against
the next layer's, pulled back through both maps; with two maps it does not
factor into 1-D pieces.  The FAST PATH is the Phase C stack-wide merged map,
taken when the shapes of every layer fit one map.  A FAIL-BEFORE arm is a
deliberately broken variant that must fail the bar.

Fixture (the planning probes'): lambda = 1, square period 1.2, air over
n = 1.45; layer 1 (depth 0.3) an eps-4 circle of r = 0.36; layer 2 (depth
0.25) a sinusoidal interface x = 0.6 + 0.12 sin(2 pi y / p) between air and
eps 2.25 -- it runs THROUGH the disk, so the two outlines CROSS in plan view
and the stack-wide merge refuses.  Row 0 of R / T is the input E along x,
row 1 E along y.

EVERY BAR is derived from a measurement made by this build on 2026-10-03
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_e2/``), stated next
to the assertion with its gap on both sides.  Sizes: 3 x 3 and 2 x 2 maps at
M <= 5.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    PMM2DStackPure,
    SinusoidalWall,
    compile_shapes,
)
from lumenairy.elements.pmm import _core  # noqa: E402
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import _curvemortar as CMM  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

_P = 1.2
_WL = 1.0
_NSUP, _NSUB = 1.0, 1.45
_R = 0.36
_D1, _D2 = 0.3, 0.25


def _circ(eps=4.0):
    return [Circle(0.6, 0.6, _R, eps)]


def _sinw(x0=0.6, A=0.12, eps=2.25):
    return [SinusoidalWall("x", x0, A, eps=eps)]


def _stack(layers, M, per_layer=True, n_orders=3):
    kw = dict(n_superstrate=_NSUP, n_substrate=_NSUB, n_modes=M,
              n_orders=n_orders)
    if per_layer:
        kw["layer_grids"] = "per-layer"
    st = PMM2DStackPure(_P, _P, **kw)
    for t, shp, bg in layers:
        if shp is None:
            st.add_layer(t, eps=bg)
        else:
            st.add_layer(t, shapes=shp, background_eps=bg)
    return st


def _solve(st, theta=0.0, phi=0.0, retain=False):
    st.set_source(_WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=retain)
    return np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)


def _closure(R, T):
    return float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))


def _diff(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


@functools.lru_cache(maxsize=None)
def _maps():
    _e, _x, _y, circ = compile_shapes(_P, _P, _circ(), 1.0)
    _e, _x, _y, sinx = compile_shapes(_P, _P, _sinw(), 1.0)
    return circ, sinx


@functools.lru_cache(maxsize=None)
def _overlap(M):
    st = _stack([(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)], M)
    return st, _solve(st)


# =========================================================================== #
# E2-1 -- no map: the shipped per-layer mortar never meets the new code
# =========================================================================== #
def test_e2_1_unmapped_per_layer_never_reaches_the_curved_mortar(monkeypatch):
    """E2-1 (bytes): the full SHA-256 set of the shipped staggered fixtures
    incl. 12 per-layer mortar hashes (non-conforming pairs at normal /
    oblique / conical incidence with absorption, the generalized mortar, a
    forced conforming mortar, a taper, the cross-mass factors) is 134 / 134
    identical to ``eae470d9`` (``e2_1_compare.json``).  Here the routing
    half: an UNMAPPED per-layer stack -- non-conforming, lossy, with a
    uniform layer, out-of-plane, absorption -- solves with the curved
    kernel booby-trapped; the same trap FIRES on a mapped per-layer stack
    (fail-before)."""
    def trap(*a, **k):
        raise AssertionError("the curved mortar was reached")
    monkeypatch.setattr(CMM, "curved_cross_mass", trap)
    half = np.array([[4.0, 1.0], [1.0, 1.0]], complex)
    third = np.ones((3, 3), complex)
    third[1, 1] = 2.25 + 0.1j
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=2, layer_grids="per-layer")
    st.add_layer(0.2, eps_cell=half, x_walls=[0.45], y_walls=[0.55])
    st.add_layer(0.1, eps=2.0)
    st.add_layer(0.2, eps_cell=third)
    st.set_source(_WL, theta=0.15, phi=0.3)
    st.solve(retain_internal=True)
    st.layer_absorption()
    oop = np.broadcast_to(np.eye(3, dtype=complex) * 2.0, (2, 2, 3, 3)).copy()
    oop[0, 0, 0, 2] = oop[0, 0, 2, 0] = 0.3
    st = PMM2DStackPure(_P, _P, n_modes=4, n_orders=2,
                        layer_grids="per-layer")
    st.add_layer(0.2, eps_cell=oop)
    st.add_layer(0.2, eps_cell=third)
    st.set_source(_WL, theta=0.15, phi=0.3)
    st.solve()
    g = TS.StagGridOps(_P, _P, 3, 3, 4, 1.0, 1.0)
    assert g.cmap is None and len(g.key()) == 2       # the shipped key
    st = _stack([(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)], 3)
    with pytest.raises(AssertionError, match="curved mortar was reached"):
        _solve(st)


# =========================================================================== #
# E2-2 -- the same map through the curved mortar is the shared solve
# =========================================================================== #
def test_e2_2_same_map_through_the_curved_mortar_is_the_shared_solve():
    """Two layers (a circle, a lossy film) on the SAME circle map: the
    per-layer stack (identical grids -> the square match) equals the shared
    stack BIT FOR BIT, and FORCED through the curved mortar it equals it to
    round-off.  Measured 2026-10-03 (``e2_2_same_map_M4.json``, M = 4): forced
    1.4e-15 / 1.3e-15 / 3.1e-15 at normal / oblique / conical; bar 1e-10
    (4.5 decades up -- the brief's exactness bar).  Fail-before: the V1/V2
    swap of the H-row masses off (the shipped G3 control) reads 0.80 .. 1.10;
    bar >= 0.1."""
    cm, _w = CM._circle_map_3x3(_P, _R)
    cell = np.ones((3, 3), complex)
    cell[1, 1] = 4.0
    th, ph = 0.3, 0.7

    def run(mode):
        if mode == "shared":
            st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP,
                                n_substrate=_NSUB, n_modes=4, n_orders=3,
                                cmap=cm)
            st.add_layer(0.3, eps_cell=cell)
            st.add_layer(0.2, eps=2.25 + 0.05j)
        else:
            st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP,
                                n_substrate=_NSUB, n_modes=4, n_orders=3,
                                layer_grids="per-layer")
            st.add_layer(0.3, eps_cell=cell, cmap=cm)
            st.add_layer(0.2, eps=2.25 + 0.05j, cmap=cm, n_modes=4)
        st.set_source(_WL, theta=th, phi=ph)
        if mode == "forced":
            return st._solve_per_layer(jones=True, retain_internal=False,
                                       force_mortar=True)
        return st.solve()
    s = run("shared")
    p = run("per-layer")
    assert all(np.array_equal(a, b) for a, b in zip(s, p))
    f = run("forced")
    d = max(np.abs(np.asarray(f[1]) - s[1]).max(),
            np.abs(np.asarray(f[2]) - s[2]).max(),
            np.abs(np.asarray(f[3]) - np.asarray(s[3])).max())
    assert d <= 1e-10, d
    _core.PMM2D_MORTAR_H_SWAP = False
    try:
        x = run("forced")
    finally:
        _core.PMM2D_MORTAR_H_SWAP = True
    assert np.abs(np.asarray(x[1]) - s[1]).max() >= 0.1


# =========================================================================== #
# E2-3 -- two different SEPARABLE stretches
# =========================================================================== #
def _stretch_pair():
    sa = CM.SeparableStretch.from_physical_walls(
        np.array([0, 0.3, 0.9, _P]), np.array([0, 0.3, 0.9, _P]),
        fx=CM.SineStretch(0.06 * _P))
    sb = CM.SeparableStretch.from_physical_walls(
        np.array([0, 0.5, 0.8, _P]), np.array([0, 0.2, 0.7, _P]),
        fy=CM.SineStretch(0.08 * _P), fx=CM.SineStretch(-0.03 * _P))
    return sa, sb


def test_e2_3_separable_cross_mass_is_the_1d_factorisation():
    """Under two separable stretches the 2-D cross-mass FACTORS:
    ``X11 = kron(Cy[Btilde], Cx'[B])`` with ``psi' = du_a / du_b`` on the
    x-integral of the u-component only (the covariant pullback), ``X12 = X21
    = 0``.  The 1-D integrals are evaluated here by brute force (200 Gauss
    nodes on every piece of the union of b's segments and the preimages of
    a's walls), sharing nothing with the 2-D kernel but the stencils.
    Measured (``e2_x_separable_M5.json``): 1.3e-14 / 5.4e-15, the
    off-diagonal blocks exactly 0; bar 1e-12 (1.9 decades up).  Fail-before:
    dropping ``psi'`` reads 0.42; bar >= 0.1."""
    from numpy.polynomial.legendre import leggauss
    sa, sb = _stretch_pair()
    tau = np.exp(-0.3j)
    ga = TS.StagGridOps(_P, _P, sa.u_walls, sa.v_walls, 4, tau, tau, cmap=sa)
    gb = TS.StagGridOps(_P, _P, sb.u_walls, sb.v_walls, 5, tau, tau, cmap=sb)

    def f_of(m, ax):
        return m.fx if ax == "x" else m.fy

    def psi(s, ax):
        fb, fa = f_of(sb, ax), f_of(sa, ax)
        x, dx = (s, np.ones_like(s)) if fb is None else fb(s, _P)
        if fa is None:
            return x, dx
        u = fa.inverse(x, _P)
        return u, dx / fa(u, _P)[1]

    def c1d(ba, bb, which, ax, deriv):
        Sa, Sb = np.asarray(getattr(ba, which)), np.asarray(getattr(bb, which))
        fb, fa = f_of(sb, ax), f_of(sa, ax)
        xa = ba.xb if fa is None else fa(ba.xb, _P)[0]
        pre = xa if fb is None else fb.inverse(xa, _P)
        cuts = np.unique(np.concatenate([bb.xb, pre, [0.0, _P]]))
        xg, wg = leggauss(200)
        C = np.zeros((Sb.shape[0], Sa.shape[0]), complex)
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            if s1 - s0 < 1e-14:
                continue
            s = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * xg
            ua, dp = psi(s, ax)
            eb = int(np.clip(np.searchsorted(bb.xb, 0.5 * (s0 + s1)) - 1, 0,
                             bb.N - 1))
            ea = int(np.clip(np.searchsorted(ba.xb, np.mean(ua)) - 1, 0,
                             ba.N - 1))
            ww = 0.5 * (s1 - s0) * wg * (dp if deriv else 1.0)
            C += (np.conj(Sb[:, eb, :]) @ (CMM._local_vals(bb, eb, s) * ww)
                  @ CMM._local_vals(ba, ea, ua).T @ Sa[:, ea, :].T)
        return C
    X = CMM.curved_cross_mass_adaptive(ga, gb)[0]
    qa, qb = ga.qq, gb.qq
    sc = np.abs(X).max()
    X11 = np.kron(c1d(ga.by, gb.by, "Btilde", "y", False),
                  c1d(ga.bx, gb.bx, "B", "x", True))
    X22 = np.kron(c1d(ga.by, gb.by, "B", "y", True),
                  c1d(ga.bx, gb.bx, "Btilde", "x", False))
    assert np.abs(X[:qb, :qa] - X11).max() / sc <= 1e-12
    assert np.abs(X[qb:, qa:] - X22).max() / sc <= 1e-12
    assert np.abs(X[:qb, qa:]).max() == 0.0 and np.abs(X[qb:, :qa]).max() == 0.0
    X11w = np.kron(c1d(ga.by, gb.by, "Btilde", "y", False),
                   c1d(ga.bx, gb.bx, "B", "x", False))
    assert np.abs(X[:qb, :qa] - X11w).max() / sc >= 0.1


def test_e2_3_two_stretches_converge_to_the_unmapped_device():
    """The two stretched layers (a pillar on x, y in 0.3 .. 0.9 stretched in
    x; eps 2.25 on x in 0.5 .. 0.8, y in 0.2 .. 0.7 stretched in y) move no
    physical wall, so the mapped per-layer stack must converge to the same
    device on its physical walls through the SHIPPED separable mortar.
    Measured (``e2_3_stretches_M*.json``): 2.1e-2 (M = 4), 2.0e-3 (M = 5);
    both arms against the 5 x 5 union-grid solve 3.6e-2 / 4.3e-3 (mapped) and
    3.6e-2 / 2.8e-3 (unmapped).  Bars: M = 5 <= 6e-3 (0.5 decades up), and
    the M = 4 -> 5 drop >= 0.5 decades (measured 1.0)."""
    sa, sb = _stretch_pair()
    sb = CM.SeparableStretch.from_physical_walls(
        np.array([0, 0.5, 0.8, _P]), np.array([0, 0.2, 0.7, _P]),
        fy=CM.SineStretch(0.08 * _P))
    c1 = np.ones((3, 3), complex)
    c1[1, 1] = 4.0
    c2 = np.ones((3, 3), complex)
    c2[1, 1] = 2.25

    def run(M, mapped):
        st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        if mapped:
            st.add_layer(0.3, eps_cell=c1, cmap=sa)
            st.add_layer(0.25, eps_cell=c2, cmap=sb)
        else:
            st.add_layer(0.3, eps_cell=c1, x_walls=[0.3, 0.9],
                         y_walls=[0.3, 0.9])
            st.add_layer(0.25, eps_cell=c2, x_walls=[0.5, 0.8],
                         y_walls=[0.2, 0.7])
        return _solve(st)
    d4 = _diff(run(4, True), run(4, False))
    d5 = _diff(run(5, True), run(5, False))
    assert d5 <= 6e-3, d5
    assert np.log10(d4 / d5) >= 0.5, (d4, d5)


# =========================================================================== #
# E2-4 -- a circle over a sinusoidal wall that CROSSES it
# =========================================================================== #
def test_e2_4_crossing_outlines_solve_through_the_curved_mortar():
    """The two outlines cross, the stack-wide merge refuses (named), and the
    per-layer maps carry the stack.  Lossless closure measured
    (``e2_4_closure_M*.json``): 6.5e-4, 1.8e-4, 2.4e-5, 7.6e-7, 9.7e-7 at
    M = 4 .. 8 (the stop condition, <= 1e-4 by M = 8, is met at M = 6).
    Bar at M = 5: <= 1e-3 (0.75 decades up).  Fail-before: the H-row
    V1/V2 swap off reads closure 0.10 / 0.12 (``e2_8_mutations_M5.json``);
    bar >= 1e-2."""
    st, (o, R, T, J) = _overlap(5)
    assert not st._perlayer_fast_ok()
    assert "CROSS" in (st._merge_refusal or "")
    assert _closure(R, T) <= 1e-3, _closure(R, T)
    _core.PMM2D_MORTAR_H_SWAP = False
    try:
        st2 = _stack([(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)], 5)
        o2, R2, T2, J2 = _solve(st2)
    finally:
        _core.PMM2D_MORTAR_H_SWAP = True
    assert _closure(R2, T2) >= 1e-2


def test_e2_4_absorption_across_differently_mapped_layers():
    """The disk lossy (eps 4 + 0.3i) over the lossless crossing wall: the
    per-layer flux form is each layer's PLAIN Gram (the metric-free z-flux),
    so the LOSSLESS layer absorbs nothing.  Measured
    (``e2_4_absorb_M4.json``): 1.5e-15 / 4.6e-16; bar 1e-10.  The budget
    sum(A) against 1 - R - T: 7.5e-3 / 5.4e-3 at M = 4 (the discretisation
    level; recorded).  Fail-before: -R (= C[chi_t]C under a map) as the flux
    form reads 5.0e-3 / 4.7e-3 in the lossless layer; bar >= 1e-3."""
    st = _stack([(_D1, _circ(4.0 + 0.3j), 1.0), (_D2, _sinw(), 1.0)], 4)
    o, R, T, J = _solve(st, retain=True)
    A = st.layer_absorption()
    assert np.abs(A[1]).max() <= 1e-10, A
    assert A[0].min() > 0.05
    # fail-before: -R as the flux form of every layer
    d = st._internal
    Rm = []
    for L, Mi in zip(st._layers, st._perlayer_modal_counts()):
        own = L["own"]
        s = TS.Granet2DTransverseE(_P, _P, own["cmap"].u_walls,
                                   own["cmap"].v_walls, Mi, own["cell"],
                                   k0=2 * np.pi, cmap=own["cmap"])
        Rm.append(-s.Rmat)
    amps = st._internal_amplitudes()

    def flux(i, z):
        Wf, Vf, lam_f, Wb, Vb, lam_b, t = d["modes"][i]
        cf, cb = amps[i]
        k0, qq = d["k0"], d["qq_of"][i]
        E = (Wf @ (np.exp(-lam_f * k0 * z * t)[:, None] * cf)
             + Wb @ (np.exp(lam_b * k0 * (1 - z) * t)[:, None] * cb))
        H = (Vf @ (np.exp(-lam_f * k0 * z * t)[:, None] * cf)
             + Vb @ (np.exp(lam_b * k0 * (1 - z) * t)[:, None] * cb))
        G = Rm[i]
        return np.real(np.sum(np.conj(H[qq:]) * (G[:qq, :qq] @ E[:qq]), 0)
                       - np.sum(np.conj(H[:qq]) * (G[qq:, qq:] @ E[qq:]), 0))
    one_minus_R = 1.0 - R.sum(1)
    finc = flux(0, 0.0) / one_minus_R
    a2 = (flux(1, 0.0) - flux(1, 1.0)) / finc
    assert np.abs(a2).max() >= 1e-3, a2


def test_e2_4_vacuum_layer_identity_and_the_merged_map_both_ways():
    """(i) A VACUUM-painted sinusoid layer on top of the circle (air above:
    a physical no-op) is homogeneous, so it RIDES the circle's grid and the
    stack is BIT-IDENTICAL to the same spacer on the shared circle map
    (``e2_4_vacuum_M*.json``: 0.0 at M = 4, 5, 6); kept on its OWN map (the
    no-ride instrument) it is a genuine curved mortar under the pillar's rim
    and reads 2.1e-3 / 3.4e-3 / 2.7e-3 from the circle alone (the shipped
    separable mortar's non-conforming spacer reads 4.1e-3 / 3.0e-3 / 1.2e-3,
    ``e2_v_spacer_shipped_M*.json``) -- recorded; bar here >= 1e-4 (it is not
    the identity).  (ii) A circle over a NON-overlapping wall (x0 = 0.12,
    A = 0.05) merges: the per-layer stack takes the fast path, bit-identical
    to the shared stack; forced through the curved mortar it agrees with
    the merged map at the non-conforming mortar's level -- 9.9e-2, 1.1e-2,
    9.7e-3, 4.5e-4 at M = 4 .. 7 (``e2_4_merged_M*.json``; the shipped
    separable analogue 2.8e-2 / 9.7e-3 / 5.6e-3 at M = 5 .. 7,
    ``e2_w_shipped_baseline_M*.json``).  Bar at M = 4: <= 0.2 (0.3 decades
    up); the ladder is the build doc's."""
    lay = [(_D2, _sinw(eps=1.0), 1.0), (_D1, _circ(), 1.0)]
    a = _solve(_stack(lay, 4))
    b = _solve(_stack([(_D2, None, 1.0), (_D1, _circ(), 1.0)], 4,
                      per_layer=False))
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    st = _stack(lay, 4)
    st._e2_no_ride = True
    c = _solve(st)
    assert _diff(c, b) >= 1e-4
    lay = [(_D1, _circ(), 1.0), (_D2, _sinw(0.12, 0.05), 1.0)]
    st = _stack(lay, 4)
    assert st._perlayer_fast_ok()
    m = _solve(st)
    s = _solve(_stack(lay, 4, per_layer=False))
    assert all(np.array_equal(x, y) for x, y in zip(m, s))
    st = _stack(lay, 4)
    st._e2_per_layer_maps = True
    p = _solve(st)
    assert _diff(p, m) <= 0.2


# =========================================================================== #
# E2-5 -- the cross-mass quadrature is spectral; the composite rule is not
# =========================================================================== #
def test_e2_5_cut_cell_quadrature_is_spectral():
    """The circle / crossing-sinusoid cross-mass at M = 4 (the sinusoid layer
    q-matched, M = 6) against its n = 69 value (``e2_5_quadrature_M4.json``):
    1.5e-2, 1.2e-5, 3.7e-8, 1.8e-10, 9.3e-13, 4.6e-16 at n = 4, 6, 8, 10, 12,
    16 -- spectral; the adaptive rule stops at n = 23 (1.9e-15).  Bar:
    n = 12 <= 1e-11 (1 decade up), the adaptive result <= 1e-13.
    Fail-before: ONE tensor Gauss rule per circle cell that ignores the
    sinusoid's walls inside it (the plan's composite-map route) is
    ALGEBRAIC -- 8.6e-4 at n = 64, 3.2e-4 at n = 128, 2.0e-4 at n = 256
    (``e2_d_design_M4.json``); bar n = 64 >= 1e-4."""
    circ, sinx = _maps()
    tau = np.exp(-0.4j)
    ga = TS.StagGridOps(_P, _P, circ.u_walls, circ.v_walls, 4, tau, tau,
                        cmap=circ)
    gb = TS.StagGridOps(_P, _P, sinx.u_walls, sinx.v_walls, 6, tau, tau,
                        cmap=sinx)
    ref = CMM.curved_cross_mass(ga, gb, 48)
    sc = np.abs(ref).max()
    X12 = CMM.curved_cross_mass(ga, gb, 12)
    assert np.abs(X12 - ref).max() / sc <= 1e-11
    Xa, n, chg = CMM.curved_cross_mass_adaptive(ga, gb)
    assert np.abs(Xa - ref).max() / sc <= 1e-13 and chg <= 1e-12
    Xc = CMM.curved_cross_mass(ga, gb, 64, cut=False)
    assert np.abs(Xc - ref).max() / sc >= 1e-4


# =========================================================================== #
# E2-6 -- oblique and conical incidence through the curved mortar
# =========================================================================== #
def test_e2_6_oblique_and_conical_closure_and_reciprocity():
    """The E2-4 device at (25 deg, 0) and (25 deg, 40 deg): lossless closure
    and reflection RECIPROCITY of order (-1, 0) -- singular values of the
    power-normalised Jones block against the reversed channel's.  Measured
    (``e2_6_angles_M*.json``): reciprocity 1.4e-4 / 7.5e-4 (M = 4), 1.1e-5 /
    1.1e-4, 7.7e-6 / 7.2e-6, 1.0e-6 / 7.3e-6 (M = 5 .. 7); closure <= 5.8e-3
    at M = 4.  Bars at M = 4: reciprocity <= 3e-3 (0.6 decades up), closure
    <= 2e-2.  Fail-before: pairing with the reverse ORDER (0, 0) reads
    0.039 / 0.067; bar >= 1e-2."""
    lay = [(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)]

    def block(st, o, mn):
        k = int(np.nonzero((o[:, 0] == mn[0]) & (o[:, 1] == mn[1]))[0][0])
        r = st._modal
        A = np.array([[r["rx"][c][k] for c in (0, 1)],
                      [r["ry"][c][k] for c in (0, 1)]])
        kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
        kxo, kyo, kzo = r["kx"][k], r["ky"][k], complex(r["kz_ref"][k]).real
        Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
        Wo = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                            / kzo ** 2)

        def msq(S, p):
            w, V = np.linalg.eigh(S)
            return (V * w ** p) @ V.conj().T
        return np.linalg.svd(msq(Wo, 0.5) @ A @ msq(Gin, -0.5),
                             compute_uv=False)
    for th_d, ph_d in ((25.0, 0.0), (25.0, 40.0)):
        th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
        st = _stack(lay, 4)
        o, R, T, J = _solve(st, th, ph)
        assert _closure(R, T) <= 2e-2
        sf = block(st, o, (-1, 0))
        kx = np.sin(th) * np.cos(ph) - _WL / _P
        ky = np.sin(th) * np.sin(ph)
        tr, pr = np.arcsin(np.hypot(kx, ky)), np.arctan2(-ky, -kx)
        st2 = _stack(lay, 4)
        o2, R2, T2, J2 = _solve(st2, tr, pr)
        assert _closure(R2, T2) <= 2e-2
        assert np.max(np.abs(sf - block(st2, o2, (-1, 0)))) <= 3e-3
        assert np.max(np.abs(sf - block(st2, o2, (0, 0)))) >= 1e-2


# =========================================================================== #
# E2-7 -- three layers, three maps
# =========================================================================== #
def test_e2_7_three_layers_on_three_maps():
    """A circle, a sinusoidal wall along x crossing it, a sinusoidal wall
    along y crossing both: two curved mortars in one cascade.  Measured
    (``e2_7_three_M*.json``): closure 1.5e-3, 2.1e-4, 7.4e-5, 2.9e-6 at
    M = 4 .. 7; with the middle layer lossy the two LOSSLESS layers absorb
    <= 2.5e-14 (M = 4) .. 2.7e-13 (M = 7).  Bars at M = 4: closure <= 1e-2,
    lossless absorption <= 1e-10."""
    def lay(e2):
        return [(0.25, _circ(), 1.0),
                (0.2, [SinusoidalWall("x", 0.6, 0.12, eps=e2)], 1.0),
                (0.2, [SinusoidalWall("y", 0.5, 0.1, eps=1.7)], 1.0)]
    st = _stack(lay(2.25), 4)
    assert not st._perlayer_fast_ok()
    o, R, T, J = _solve(st)
    assert _closure(R, T) <= 1e-2
    st = _stack(lay(2.25 + 0.2j), 4)
    o, R, T, J = _solve(st, retain=True)
    A = st.layer_absorption()
    assert max(np.abs(A[0]).max(), np.abs(A[2]).max()) <= 1e-10
    assert A[1].min() > 0.05


# =========================================================================== #
# E2-8 -- the mutation matrix (the arms not already two-sided above)
# =========================================================================== #
def test_e2_8_mutations_maps_ignored_and_the_forced_fast_path(monkeypatch):
    """(a) The cross-mass with both maps IGNORED (the shipped separable
    cross-mass on the two (u, v) grids) moves the E2-4 device by 6.7e-2 at
    M = 5 while its closure barely changes (2.0e-4 against 1.8e-4) -- a
    defect closure cannot see; it is caught by E2-3's factorisation (0.42)
    and by the both-ways comparison of E2-4 (3.9e-2 against the shipped
    1.1e-2 at M = 5, ``e2_8_mutations_M5.json``).  Bar here (M = 4): >= 1e-2.
    (b) The fast path FORCED onto the crossing pair (the merge's crossing
    test disabled) is refused by the merge's independent vertex-claim check
    ("DIFFERENT physical positions for the same grid vertex"), so the stack
    stays on the per-layer maps, unchanged.  (c) The Newton tolerance of the
    map inversion loosened 1e3x moves R / T by 2.6e-15 -- Newton converges
    quadratically past it; the load-bearing tolerance is the residual
    acceptance (recorded, ``e2_8_mutations_M5.json``)."""
    st, base = _overlap(4)
    monkeypatch.setattr(CMM, "StagCrossOpsMapped",
                        lambda ga, gb, tol=None: TS.StagCrossOps(ga, gb))
    bad = _solve(_stack([(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)], 4))
    assert _diff(bad, base) >= 1e-2
    monkeypatch.undo()
    from lumenairy.elements.pmm import shapes2d as S2
    monkeypatch.setattr(S2, "_crossing", lambda *a, **k: False)
    st2 = _stack([(_D1, _circ(), 1.0), (_D2, _sinw(), 1.0)], 4)
    assert not st2._perlayer_fast_ok()
    assert "DIFFERENT physical positions" in st2._merge_refusal
    assert _diff(_solve(st2), base) == 0.0


# =========================================================================== #
# API: the per-layer map keyword, refusals, the viewers
# =========================================================================== #
def test_e2_api_per_layer_maps_refusals_and_viewer():
    circ, _sinx = _maps()
    cell = np.ones((3, 3), complex)
    cell[1, 1] = 4.0
    with pytest.raises(NotImplementedError, match="add_layer"):
        PMM2DStackPure(_P, _P, layer_grids="per-layer", cmap=circ)
    with pytest.raises(ValueError, match="per-layer"):
        PMM2DStackPure(_P, _P).add_layer(0.2, eps_cell=cell, cmap=circ)
    st = PMM2DStackPure(_P, _P, n_modes=4, layer_grids="per-layer")
    with pytest.raises(ValueError, match="x_walls"):
        st.add_layer(0.2, eps_cell=cell, cmap=circ, x_walls=[0.3])
    with pytest.raises(ValueError, match="one entry per"):
        st.add_layer(0.2, eps_cell=np.ones((2, 2)), cmap=circ)
    with pytest.raises(NotImplementedError, match="Phase E"):
        st.add_layer(0.2, eps_cell=cell, cmap=circ, slant=(0.1, 0.0))
    oop = np.diag([2.0, 2.5, 2.0]).astype(complex)
    oop[0, 2] = oop[2, 0] = 0.3
    with pytest.raises(NotImplementedError, match="Phase E"):
        st.add_layer(0.2, eps=oop, cmap=circ)
    with pytest.raises(ValueError, match="cmap"):
        st.add_layer(0.2, shapes=_circ(), background_eps=1.0, cmap=circ)
    # a raw eps_cell layer joins a per-layer shape stack (no fast path)
    st = _stack([(_D1, _circ(), 1.0)], 4, n_orders=2)
    st.add_layer(0.2, eps_cell=np.array([[2.0, 1.0], [1.0, 1.0]], complex))
    assert not st._perlayer_fast_ok()
    o, R, T, J = _solve(st)
    assert np.all(np.isfinite(R))
    # the viewer draws each layer on its own map: the circle exactly
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st, _res = _overlap(3)
    axes = st.plot_geometry()
    pts = np.concatenate([np.column_stack(ln.get_data())
                          for ln in axes[0].lines])
    dev = float(np.max(np.abs(np.hypot(pts[:, 0] - 0.6, pts[:, 1] - 0.6)
                              - _R)))
    assert dev <= 1e-12, dev
    plt.close(axes[0].figure)


# =========================================================================== #
# The Phase E2 verifier's fold-in (VERIFY_PMM2D_CURVED_E2_2026_10_03.md)
# =========================================================================== #
def test_e2_v1_circle_graze_sliver_is_found_and_scales_like_its_area(
        monkeypatch):
    """V-E2-D1 on the CIRCLE (the verifier's ``test_ve2_3`` is the sinusoid
    case): an unmapped wall y = 0.24 + delta just inside the bottom of the
    r = 0.36 circle (M = 4) cuts a sliver whose share of the cross-mass is
    ~ its area ~ delta^1.5.  Measured 2026-10-03 (``e2_g_prepost.json``):
    the grazing refinement adds 1.17e-10 at delta = 1e-7 and 3.70e-9 at
    1e-6 -- ratio 31.6 = 10^1.5 -- i.e. the sliver the 65-sample run logic
    dropped.  (The verifier's brute force reads ~6.5e-7 against the kernel
    at every graze depth 1e-7 .. 1e-5; its own n = 20 vs 28 change there is
    4.2e-7, so that residual is the oracle's, ``e2_g_oracle_check.json``.)
    Bars: the delta = 1e-6 share >= 1e-9 (fail-before: the refinement off
    drops it), and the ratio within 10 % of 10^1.5."""
    circ = compile_shapes(_P, _P, [Circle(0.6, 0.6, _R, 4.0)], 1.0)[3]
    ga = TS.StagGridOps(_P, _P, circ.u_walls, circ.v_walls, 4, 1.0, 1.0,
                        cmap=circ)
    shares = []
    for d in (1e-7, 1e-6):
        gb = TS.StagGridOps(_P, _P, np.array([0.0, 0.45, _P]),
                            np.array([0.0, 0.24 + d, _P]), 4, 1.0, 1.0,
                            cmap=CM.IdentityMap(np.array([0.0, 0.45, _P]),
                                                np.array([0.0, 0.24 + d, _P])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xf = CMM.curved_cross_mass(ga, gb, 41)
            with monkeypatch.context() as m:
                m.setattr(CMM, "_grazing_refine",
                          lambda Pm, sx, sy, Om, e, tk: tk)
                Xo = CMM.curved_cross_mass(ga, gb, 41)
        shares.append(float(np.abs(Xf - Xo).max() / np.abs(Xf).max()))
    assert shares[1] >= 1e-9, shares
    assert abs(shares[1] / shares[0] / 10 ** 1.5 - 1.0) <= 0.1, shares
