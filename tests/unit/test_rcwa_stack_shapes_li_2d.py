"""Phase D (PLAN_DYNAMETA_BRIDGE_FINDINGS_2026_11_02, finding L-4):
``formulation='li'`` on 2-D ``RCWAStack`` ``shapes=`` layers of AXIS-ALIGNED
RECTANGLES.

Li's sequential rule for ``E_x`` is ``[[ [[1/eps]]_x^-1 ]]_y``: the inverse rule
along x at each y, then the direct rule along y (``E_y``: roles swapped).  For
rectangles the cell splits into y-bands between the rectangles' y-edges; in a
band the x-Toeplitz of ``1/eps`` is analytic and is INVERTED, and the y-Fourier
transform of the band-wise constant inverse is the exact sum over bands of
``A_band`` times the band's sinc coefficients
(``twod._li_rect_axis_operator``).  The result is the ``S -> inf`` limit of the
``eps_cell`` 'li' path (``_li_convolutions_2d``) on a node raster of the same
cell, placed exactly as that path places it.  Pinned here:

* one band: full-height rectangles on a 2-D stack reproduce C1's 1-D 'li';
* raster agreement: node-aligned ``eps_cell`` + 'li' converges TO analytic +
  'li' (Romberg in even powers of ``1/S``), per order, at conical incidence
  with live +-1 orders in x and y -- a square pillar, a rectangle pair that
  wraps the cell edge, an L built from two rectangles;
* rule separation: swapping which axis takes the inverse rule moves R by far
  more than the agreement tolerance;
* registration: internal operators are the conjugates of the PUBLIC ones (the
  shapes bridge rule), and a translated layer's order amplitudes pick up
  ``exp(-i (m Kx a + n Ky b))`` -- not the mirrored phase;
* the metal square pillar (the ``fff_nv`` M4 fixture) is energy-clean at every
  truncation and converges faster in N than analytic + Laurent;
* refusals (disks, ellipses, rotated rectangles -> Phase F; singular bands),
  the dispersive path, the even-parity fold, the eig-dedup key, JAX.

Measured (scratch gates, impl_d): raster Romberg residual 0.9e-12 / 1.2e-12 /
6.2e-12 (square / pair / L) at n_orders 5x5, 5.9e-11 / 1.6e-10 / 4.9e-11 at
9x9, against raw pixelation errors 2e-6 .. 1e-5 at S = 1024.
"""
import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa import RCWAStack
import lumenairy.elements.rcwa.stack as _stk
from lumenairy.elements.rcwa.twod import (
    _analytic_convolutions_2d,
    _analytic_li_operators_2d,
    _harmonic_orders_2d,
    _internal_analytic_li_operators,
    _internal_analytic_li_operators_2d,
)

LAM = 1.31e-6
P = 2.0e-6                    # +-1 orders propagate in x and y at 1.31 um
DEPTH = 150e-9
EPS_M = complex(-20.0, 2.0)
EPS_D = 6.0 + 0.8j
TH, PHI = 0.3, 0.7            # conical
N_SUB = 1.5

# (x0, w, y0, h, eps) in fractions of the period; every edge on a 1/16 node
SQUARE = [(3 / 16, 7 / 16, 5 / 16, 7 / 16, EPS_M)]
PAIR = [(13 / 16, 5 / 16, 2 / 16, 5 / 16, EPS_M),       # wraps x
        (4 / 16, 3 / 16, 9 / 16, 9 / 16, EPS_D)]         # wraps y
LSHAPE = [(2 / 16, 3 / 16, 2 / 16, 10 / 16, EPS_M),
          (5 / 16, 6 / 16, 2 / 16, 3 / 16, EPS_M)]       # touches the first


def _rect(x0, w, y0, h, eps, px=P, py=P):
    return {"shape": "rectangle", "eps": eps, "size": (w * px, h * py),
            "center": ((x0 + 0.5 * w) * px, (y0 + 0.5 * h) * py)}


def _node_cell(S, rects, bg=1.0 + 0j):
    """Node-lattice raster (x_j = j P / S), half-open, wrap-aware; every edge
    sits exactly on a node."""
    cell = np.full((S, S), bg, dtype=complex)
    for x0, w, y0, h, e in rects:
        a, n, b, m = (round(v * S) for v in (x0, w, y0, h))
        assert max(abs(a - x0 * S), abs(n - w * S), abs(b - y0 * S),
                   abs(m - h * S)) < 1e-9
        cell[np.ix_((a + np.arange(n)) % S, (b + np.arange(m)) % S)] = e
    return cell


def _stack2d(n, period=P, n_sub=N_SUB, ny=None):
    return lum.RCWAStack(period, period_y=period, n_superstrate=1.0,
                         n_substrate=n_sub, n_orders=n,
                         n_orders_y=n if ny is None else ny)


def _solve(st, theta=TH, phi=PHI, symmetry="auto"):
    res = st.set_source(LAM, theta=theta, phi=phi).solve(symmetry=symmetry)
    o, R, T = res.efficiencies()
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _shapes_layer(n, rects, formulation="li", shift=(0.0, 0.0), **kw):
    st = _stack2d(n)
    st.add_layer(DEPTH, shapes=[_rect((x0 + shift[0]) % 1.0, w,
                                      (y0 + shift[1]) % 1.0, h, e)
                                for x0, w, y0, h, e in rects],
                 eps_background=1.0 + 0j, formulation=formulation)
    return st


def _romberg_last(ys):
    """Eliminate S^-2, S^-4, ... from values at S, 2S, 4S, ..."""
    row, j = list(ys), 1
    while len(row) > 1:
        f = 4.0 ** j
        row = [(f * row[i + 1] - row[i]) / (f - 1) for i in range(len(row) - 1)]
        j += 1
    return row[0]


# ---------------------------------------------------------------------------
# one band
# ---------------------------------------------------------------------------

_LINES = [(0.0625, 0.21875, EPS_M), (0.46875, 0.125, EPS_D)]


def test_one_band_full_height_rectangles_reproduce_c1():
    """A 1-D grating written as full-height rectangles on a 2-D stack: one
    y-band, so Cxx = C1's [[1/eps]]^-1 (x) I_y and Cyy = [[eps]].  Conical
    incidence; the n = 0 row equals C1's 1-D stack, every n != 0 order is
    dark (measured 2.8e-15 / 4.6e-15)."""
    N, NY = 12, 2
    sh = [_rect(x0, w, 0.0, 1.0, e) for x0, w, e in _LINES]
    s1 = lum.RCWAStack(P, n_superstrate=1.0, n_substrate=N_SUB, n_orders=N)
    s1.add_layer(DEPTH, shapes=sh, eps_background=1.0 + 0j, formulation="li")
    o1, R1, T1 = _solve(s1, symmetry=False)
    s2 = _stack2d(N, ny=NY)
    s2.add_layer(DEPTH, shapes=sh, eps_background=1.0 + 0j, formulation="li")
    o2, R2, T2 = _solve(s2, symmetry=False)
    sel = o2[:, 1] == 0
    assert np.array_equal(o2[sel, 0], o1)
    assert np.max(np.abs(R2[:, sel] - R1)) <= 1e-12
    assert np.max(np.abs(T2[:, sel] - T1)) <= 1e-12
    assert np.max(np.abs(R2[:, ~sel])) <= 1e-20
    assert np.sum(R1[0] > 1e-4) >= 3                 # several live orders
    # operator level
    o1d, _ = _harmonic_orders_2d(N, 0)
    E1, C1 = _internal_analytic_li_operators(1.0, sh, o1d, N, 0, P, P)
    o2d, _ = _harmonic_orders_2d(N, NY)
    E2, Cxx, Cyy = _internal_analytic_li_operators_2d(1.0, sh, o2d, N, NY, P,
                                                      P)
    m, n = o2d[:, 0], o2d[:, 1]
    K = np.where(n[:, None] == n[None, :],
                 C1[(m + N)[:, None], (m + N)[None, :]], 0)
    assert np.max(np.abs(Cxx - K)) <= 1e-12
    assert np.max(np.abs(Cyy - E2)) <= 1e-12


def test_noy0_2d_stack_matches_1d_stack():
    """A 2-D stack with n_orders_y = 0 (y-invariant list) takes C1's operator
    and reproduces the 1-D stack."""
    sh = [_rect(x0, w, 0.0, 1.0, e) for x0, w, e in _LINES]
    s1 = lum.RCWAStack(P, n_superstrate=1.0, n_substrate=N_SUB, n_orders=10)
    s1.add_layer(DEPTH, shapes=sh, eps_background=1.0 + 0j, formulation="li")
    s0 = _stack2d(10, ny=0)
    s0.add_layer(DEPTH, shapes=sh, eps_background=1.0 + 0j, formulation="li")
    _, R1, T1 = _solve(s1)
    _, R0, T0 = _solve(s0)
    assert np.max(np.abs(R0 - R1)) <= 1e-13
    assert np.max(np.abs(T0 - T1)) <= 1e-13


# ---------------------------------------------------------------------------
# raster agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rects", [SQUARE, PAIR, LSHAPE],
                         ids=["square", "pair_wrapping", "L_shape"])
def test_node_raster_li_converges_to_analytic_li(rects):
    """Per order, both incident polarizations, R and T.  The raster is the
    analytic cell translated by half a pixel (a node-lattice sample), which no
    efficiency sees, so the residual is even in 1/S."""
    N = 4
    o, Ra, Ta = _solve(_shapes_layer(N, rects))
    assert np.sum(Ta[0] > 1e-3) >= 9              # live +-1 orders in x and y
    assert {(1, 0), (-1, 0), (0, 1), (0, -1)} <= {
        tuple(int(v) for v in o[i]) for i in np.flatnonzero(Ta[0] > 1e-5)}
    diffs = []
    for S in (128, 256, 512, 1024):
        st = _stack2d(N)
        st.add_layer(DEPTH, eps_cell=_node_cell(S, rects), formulation="li")
        _, Rr, Tr = _solve(st)
        diffs.append(np.concatenate([Rr - Ra, Tr - Ta], axis=1))
    raw = np.max(np.abs(diffs[-1]))
    ext = np.max(np.abs(_romberg_last(diffs)))
    assert raw > 1e-7                              # the pixelation error is real
    assert ext <= 1e-8, ext                        # ... and extrapolates away
    assert ext <= 1e-3 * raw


# ---------------------------------------------------------------------------
# rule separation and registration
# ---------------------------------------------------------------------------

def test_swapping_the_inverse_rule_axis_moves_r_far_beyond_tolerance(
        monkeypatch):
    """Swap Cxx <-> Cyy (the inverse rule along the WRONG axis for each
    component).  Measured: 4.9e-2 / 3.7e-2 / 2.9e-2 (square / pair / L) at
    n_orders 5x5, against the 1e-8 agreement bar of the raster gate."""
    _, R0, T0 = _solve(_shapes_layer(4, PAIR))
    orig = _stk._internal_analytic_li_operators_2d

    def swapped(*a):
        E, X, Y = orig(*a)
        return E, Y, X
    monkeypatch.setattr(_stk, "_internal_analytic_li_operators_2d", swapped)
    _, R1, T1 = _solve(_shapes_layer(4, PAIR))
    assert max(np.max(np.abs(R1 - R0)), np.max(np.abs(T1 - T0))) > 1e-3


def test_internal_li_2d_operators_are_conjugates_of_public_operators():
    M = 4
    orders, _ = _harmonic_orders_2d(M, M)
    sh = [_rect(*r) for r in PAIR]
    bg = 1.2 + 0.05j
    E, Cxx, Cyy = _internal_analytic_li_operators_2d(bg, sh, orders, M, M, P,
                                                     P)
    E_pub, _ = _analytic_convolutions_2d(bg, sh, orders, M, M, P, P)
    Cxx_pub, Cyy_pub = _analytic_li_operators_2d(bg, sh, orders, M, M, P, P)
    np.testing.assert_array_equal(E, np.conj(E_pub))
    np.testing.assert_array_equal(Cxx, np.conj(Cxx_pub))
    np.testing.assert_array_equal(Cyy, np.conj(Cyy_pub))
    # ... and NOT the old bridge (conj the eps, then build): that is exactly
    # the operator set of the point mirror eps(-x, -y)
    sh_c = [dict(s, eps=np.conj(s["eps"])) for s in sh]
    old_xx, old_yy = _analytic_li_operators_2d(np.conj(bg), sh_c, orders, M,
                                               M, P, P)
    sh_m = [dict(s, center=(P - s["center"][0], P - s["center"][1]))
            for s in sh]
    _, mir_xx, mir_yy = _internal_analytic_li_operators_2d(bg, sh_m, orders,
                                                           M, M, P, P)
    assert np.max(np.abs(old_xx - Cxx)) > 1e-2
    assert np.max(np.abs(old_xx - mir_xx)) <= 1e-12
    assert np.max(np.abs(old_yy - mir_yy)) <= 1e-12


def test_li_2d_shapes_order_phases_follow_a_translation():
    """Shift the layer by (a, b) = (P/4, 3P/8): t_mn -> t_mn
    exp(-i (m Kx a + n Ky b)) (exp(-i w t)); the point-mirrored bridge gave the
    conjugate phase.  Measured 1.4e-14 vs 0.30 for the mirrored phase."""
    a, b = 0.25, 0.375
    amps = []
    for shift in ((0.0, 0.0), (a, b)):
        st = _shapes_layer(4, PAIR, shift=shift)
        r = st.set_source(LAM, theta=TH, phi=PHI).solve()
        amps.append(r.per_order_amplitudes(port="transmission"))
    o = np.asarray(amps[0]["orders"])
    ph = np.exp(-2j * np.pi * (o[:, 0] * a + o[:, 1] * b))
    for c in ("Ex", "Ey"):
        np.testing.assert_allclose(amps[1][c], amps[0][c] * ph[None, :],
                                   atol=1e-10)
        assert np.max(np.abs(amps[1][c] - amps[0][c] * np.conj(ph)[None, :])) \
            > 1e-2


# ---------------------------------------------------------------------------
# metal pillar (fff_nv M4 fixture) and energy
# ---------------------------------------------------------------------------

_EPS_AG = (0.05 + 3.3j) ** 2          # -10.8875 + 0.33i
_M4 = dict(P=0.9e-6, depth=0.4e-6, wl=0.55e-6, theta=0.2, phi=0.3)


def _m4(n, form, eps=_EPS_AG):
    st = lum.RCWAStack(_M4["P"], period_y=_M4["P"], n_superstrate=1.0,
                       n_substrate=1.5, n_orders=n, n_orders_y=n)
    st.add_layer(_M4["depth"],
                 shapes=[{"shape": "rectangle", "eps": eps,
                          "size": (0.5 * _M4["P"], 0.5 * _M4["P"]),
                          "center": (0.5 * _M4["P"], 0.5 * _M4["P"])}],
                 eps_background=1.0 + 0j, formulation=form)
    o, R, T = st.set_source(_M4["wl"], theta=_M4["theta"],
                            phi=_M4["phi"]).solve().efficiencies()
    return np.asarray(R).sum(axis=1), np.asarray(T).sum(axis=1)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_m4_pillar_energy_clean_at_every_truncation(n):
    R, T = _m4(n, "li", complex(_EPS_AG.real, 0.0))       # lossless metal
    assert np.max(np.abs(R + T - 1.0)) <= 1e-12
    R, T = _m4(n, "li")                                    # lossy Ag: passive
    assert np.all(R >= 0) and np.all(T >= 0) and np.all(R + T < 1.0)


def test_m4_pillar_li_converges_faster_than_laurent():
    """Total R (rows TM-ish / TE-ish) on the lossy M4 pillar.  Full ladder
    (scratch gate 3), |R(N) - R_li(22)| for row 0:

    ======  ========  ========
    N       li        laurent
    ======  ========  ========
    4       2.8e-2    8.4e-2
    6       2.2e-2    4.2e-2
    8       1.2e-2    1.0e-2
    10      5.7e-3    3.2e-2
    12      6.0e-3    6.2e-2
    14      5.2e-3    7.3e-2
    16      3.2e-3    7.8e-2
    18      1.4e-3    7.4e-2
    ======  ========  ========

    li@22 = 0.32029 / 0.32647; laurent@22 = 0.26380 / 0.26116 -- Laurent is
    still 6.5e-2 away at 2025 orders and moves by 1e-2 .. 4e-2 between rungs
    (against its OWN N = 22 value it is 4.4e-3 .. 1.1e-1, non-monotone), while
    'li' settles smoothly.  Pinned here: the N = 8 -> 10 step (li 6.0e-3,
    laurent 4.0e-2)."""
    li = [_m4(n, "li")[0] for n in (8, 10)]
    la = [_m4(n, "laurent")[0] for n in (8, 10)]
    d_li = np.max(np.abs(li[1] - li[0]))
    d_la = np.max(np.abs(la[1] - la[0]))
    assert d_li < 1e-2 and d_la > 2.5e-2, (d_li, d_la)
    assert np.max(np.abs(li[1] - np.array([0.32028921, 0.32647317]))) < 1e-2


@pytest.mark.parametrize("rects", [SQUARE, PAIR, LSHAPE],
                         ids=["square", "pair_wrapping", "L_shape"])
def test_lossless_rectangles_conserve_energy(rects):
    lossless = [r[:4] + (complex(r[4].real, 0.0),) for r in rects]
    _, R, T = _solve(_shapes_layer(4, lossless))
    assert np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)) <= 1e-12


# ---------------------------------------------------------------------------
# refusals, dispersive, even parity, eig key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    {"shape": "disk", "eps": EPS_M, "radius": 0.3e-6},
    {"shape": "ellipse", "eps": EPS_M, "semi_axes": (0.3e-6, 0.2e-6)},
    dict(_rect(*SQUARE[0]), angle=0.3),
])
def test_non_rectangles_and_rotated_rectangles_are_refused(shape):
    st = _stack2d(3)
    with pytest.raises(ValueError, match="Phase F"):
        st.add_layer(DEPTH, shapes=[_rect(*PAIR[0]), shape],
                     eps_background=1.0 + 0j, formulation="li")
    # Laurent is unaffected
    _stack2d(3).add_layer(DEPTH, shapes=[shape], eps_background=1.0 + 0j)


def test_zero_background_refused_and_singular_band_raises():
    with pytest.raises(ValueError, match="non-zero eps_background"):
        _stack2d(3).add_layer(DEPTH, shapes=[_rect(*SQUARE[0])],
                              eps_background=0.0, formulation="li")
    # a lossless eps = -1 bar spanning half the period inside the band where
    # it sits: that band's average 1/eps is 0 -> ill-posed inverse rule
    st = _stack2d(4, n_sub=1.0)
    st.add_layer(DEPTH, shapes=[_rect(0.25, 0.5, 0.25, 0.25, -1.0 + 0j)],
                 eps_background=1.0 + 0j, formulation="li")
    with pytest.raises(np.linalg.LinAlgError, match="ill-posed"):
        _solve(st)


def test_background_valued_rectangles_equal_uniform_layer():
    st = _stack2d(3)
    st.add_layer(DEPTH, shapes=[_rect(*r[:4], 2.25 + 0j) for r in PAIR],
                 eps_background=2.25 + 0j, formulation="li")
    su = _stack2d(3)
    su.add_layer(DEPTH, eps=2.25 + 0j)
    _, R1, T1 = _solve(st)
    _, R2, T2 = _solve(su)
    assert np.max(np.abs(R1 - R2)) <= 1e-14
    assert np.max(np.abs(T1 - T2)) <= 1e-14


def test_dispersive_li_2d_shapes_match_concrete():
    epsf = lambda wl: EPS_M * (wl / LAM) ** 2       # noqa: E731
    st = _stack2d(3)
    sh = [dict(_rect(*PAIR[0]), eps=epsf), _rect(*PAIR[1])]
    st.add_layer(DEPTH, shapes=sh, eps_background=1.0 + 0j, formulation="li")
    st.set_source(LAM, theta=TH, phi=PHI)
    wls = np.array([1.25e-6, 1.31e-6])
    _, R, _, _ = st.solve_vs_wavelength(wls, max_workers=1)
    for i, wl in enumerate(wls):
        sc = _stack2d(3)
        sc.add_layer(DEPTH, shapes=[_rect(*PAIR[0][:4], epsf(wl)),
                                    _rect(*PAIR[1])],
                     eps_background=1.0 + 0j, formulation="li")
        _, Rc, _ = sc.set_source(wl, theta=TH, phi=PHI).solve().efficiencies()
        assert np.max(np.abs(np.asarray(R[i]) - np.asarray(Rc))) <= 1e-11
    # a dispersive DISK is refused up front, before any wavelength
    with pytest.raises(ValueError, match="Phase F"):
        _stack2d(3).add_layer(DEPTH, shapes=[{"shape": "disk", "eps": epsf,
                                              "radius": 0.3e-6}],
                              eps_background=1.0 + 0j, formulation="li")


def test_even_parity_fold_agrees_with_full_solve():
    """Centred square at normal incidence: the even-parity cascade reads the
    same (Cxx, Cyy) through _shapes_operators."""
    st = _stack2d(4)
    st.add_layer(DEPTH, shapes=[_rect(0.25, 0.5, 0.25, 0.5, EPS_M)],
                 eps_background=1.0 + 0j, formulation="li")
    _, Ra, Ta = _solve(st, theta=0.0, phi=0.0, symmetry="auto")
    _, Rf, Tf = _solve(st, theta=0.0, phi=0.0, symmetry=False)
    assert np.max(np.abs(Ra - Rf)) <= 1e-11
    assert np.max(np.abs(Ta - Tf)) <= 1e-11


def test_eig_key_separates_formulations_on_2d():
    def two(fa, fb):
        st = _stack2d(3)
        for f in (fa, fb):
            st.add_layer(DEPTH / 2, shapes=[_rect(*r) for r in PAIR],
                         eps_background=1.0 + 0j, formulation=f)
        return _solve(st)[1].sum()
    mixed, li, la = two("laurent", "li"), two("li", "li"), two("laurent",
                                                               "laurent")
    assert abs(mixed - li) > 1e-4 and abs(mixed - la) > 1e-4


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------

def test_jax_traced_spacer_around_li_2d_shapes_layer():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def build(e):
        st = _stack2d(2)
        st.add_layer(200e-9, eps=e)
        st.add_layer(DEPTH, shapes=[_rect(*r) for r in PAIR],
                     eps_background=1.0 + 0j, formulation="li")
        st.add_layer(100e-9, eps=e)
        return st.set_source(LAM, theta=TH, phi=PHI).solve().efficiencies()

    def loss_np(e):
        _, R, T = build(complex(e))
        R, T = np.asarray(R), np.asarray(T)
        return float(R[0].sum() + 0.3 * T[1].sum() + 2.0 * T[0, 7])

    def loss_jx(e):
        _, R, T = build(e + 0j)
        return jnp.real(R[0].sum() + 0.3 * T[1].sum() + 2.0 * T[0, 7])

    e0 = 2.1
    assert abs(float(loss_jx(jnp.asarray(e0))) - loss_np(e0)) <= 1e-12
    g = float(jax.grad(loss_jx)(jnp.asarray(e0)))
    h = 1e-5
    fd1 = (loss_np(e0 + h) - loss_np(e0 - h)) / (2 * h)
    fd2 = (loss_np(e0 + 2 * h) - loss_np(e0 - 2 * h)) / (4 * h)
    fd = (4 * fd1 - fd2) / 3
    assert abs(g - fd) <= 1e-6 * max(1.0, abs(fd)), (g, fd)
    assert abs(float(jax.jit(loss_jx)(jnp.asarray(e0)))
               - float(loss_jx(jnp.asarray(e0)))) <= 1e-12
    es = jnp.asarray([1.9, 2.4])
    v = np.asarray(jax.vmap(loss_jx)(es))
    assert np.max(np.abs(v - np.array([loss_np(1.9), loss_np(2.4)]))) <= 1e-12


def test_eig_key_still_dedupes_identical_2d_li_layers():
    st = _stack2d(3)
    for _ in range(2):
        st.add_layer(DEPTH / 2, shapes=[_rect(*r) for r in PAIR],
                     eps_background=1.0 + 0j, formulation="li")
    k0, k1 = (RCWAStack._layer_eig_key(L) for L in st.layers)
    assert k0 == k1 and k0[1] == "li"
