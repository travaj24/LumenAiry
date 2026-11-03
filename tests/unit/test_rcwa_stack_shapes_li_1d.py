"""Phase C1 (PLAN_DYNAMETA_BRIDGE_FINDINGS_2026_11_02, finding L-4):
``formulation='li'`` on 1-D ``RCWAStack`` ``shapes=`` layers.

The layer is built from the ANALYTIC form factors with the same placement the
``eps_cell`` 'li' path uses on a 1-D stack: ``Cxx = [[1/eps]]^{-1}`` (inverse
rule, wall-normal ``E_x``), ``Cyy = EZZ = [[eps]]`` (direct rule).  The shapes are
disjoint, so the ``1/eps`` Fourier series is the exact additive sum of the form
factors and the analytic layer is the ``S -> inf`` limit of a node-aligned
raster.  Pinned here:

* a shapes layer whose shapes all equal the background is the uniform layer;
* TE (incident ``E_y``, row 1, E along the lines) is unchanged vs Laurent --
  Li's rule reduces to Laurent for the wall-tangential field;
* the node-aligned raster + 'li' converges TO analytic + 'li' in even powers of
  ``1/S`` (Romberg-extrapolated residual << the raw pixelation error);
* a lossless dielectric line conserves energy;
* refusals: 2-D stacks, y-varying shape lists, a zero background;
* the eigenmode-dedup key and the even-parity fast path see the formulation.

Measured on the GATE B fixture (lambda 1.31 um, period 600 nm, eps -20+2i,
duty 0.5, 120 nm, air / n = 1.5): the Romberg residual is <= 4e-12 at
N = 32 / 64 / 128 (S = 8N..128N), and the analytic 'li' error against the PMM
limit 0.508768 is 5.1e-3 / 1.8e-3 / 8.5e-4 / 1.9e-4 at N = 32 / 64 / 128 / 512.
"""
import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.rcwa import RCWAStack

LAM = 1.31e-6
PER = 600e-9
EPS_M = complex(-20.0, 2.0)
DEPTH = 120e-9
N_SUB = 1.5
TM, TE = 0, 1                  # rows: incident E_x (across the lines), E_y


def _stack(n, n_sub=N_SUB):
    return lum.RCWAStack(PER, n_superstrate=1.0, n_substrate=n_sub,
                         n_orders=n)


def _shapes(lines):
    """Full-height rectangles from (x0_frac, w_frac, eps) triples."""
    return [{"shape": "rectangle", "eps": e, "size": (w * PER, PER),
             "center": ((x0 + 0.5 * w) * PER, 0.5 * PER)}
            for x0, w, e in lines]


def _node_cell(S, lines):
    """Node-lattice raster (x_j = j P / S), half-open, wrap-aware; every edge
    sits exactly on a node."""
    cell = np.ones(S, dtype=complex)
    for x0, w, e in lines:
        a, n = round(x0 * S), round(w * S)
        assert abs(a - x0 * S) < 1e-9 and abs(n - w * S) < 1e-9
        cell[(a + np.arange(n)) % S] = e
    return cell


def _solve(st, theta=0.0, phi=0.0, symmetry="auto"):
    res = st.set_source(LAM, theta=theta, phi=phi).solve(symmetry=symmetry)
    o, R, T = res.efficiencies()
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _shapes_solve(n, lines, formulation, **kw):
    st = _stack(n)
    st.add_layer(DEPTH, shapes=_shapes(lines), eps_background=1.0 + 0j,
                 formulation=formulation)
    return _solve(st, **kw)


GATE_B = [(0.25, 0.5, EPS_M)]


@pytest.mark.parametrize("eps", [2.25 + 0j, EPS_M])
@pytest.mark.parametrize("theta_deg", [0.0, 20.0])
def test_background_valued_shapes_equal_uniform_layer(eps, theta_deg):
    th = np.deg2rad(theta_deg)
    st = _stack(16)
    st.add_layer(DEPTH, shapes=_shapes([(0.25, 0.5, eps), (0.8, 0.1, eps)]),
                 eps_background=eps, formulation="li")
    o1, R1, T1 = _solve(st, theta=th)
    su = _stack(16)
    su.add_layer(DEPTH, eps=eps)
    o2, R2, T2 = _solve(su, theta=th)
    assert np.array_equal(o1, o2)
    assert np.max(np.abs(R1 - R2)) <= 1e-14
    assert np.max(np.abs(T1 - T2)) <= 1e-14


@pytest.mark.parametrize("theta_deg", [0.0, 20.0])
def test_te_row_unchanged_vs_laurent(theta_deg):
    th = np.deg2rad(theta_deg)
    _, Rl, Tl = _shapes_solve(24, GATE_B, "laurent", theta=th,
                              symmetry=False)
    _, Ri, Ti = _shapes_solve(24, GATE_B, "li", theta=th, symmetry=False)
    assert np.max(np.abs(Ri[TE] - Rl[TE])) <= 1e-14
    assert np.max(np.abs(Ti[TE] - Tl[TE])) <= 1e-14
    # ... while TM (E across the lines) genuinely changes factorization
    assert abs(Ri[TM].sum() - Rl[TM].sum()) > 1e-3


def test_fff_alias_and_even_parity_path_agree():
    _, Rf, Tf = _shapes_solve(16, GATE_B, "fff", symmetry=False)
    _, Ri, Ti = _shapes_solve(16, GATE_B, "li", symmetry=False)
    assert np.array_equal(Rf, Ri) and np.array_equal(Tf, Ti)
    _, Ra, Ta = _shapes_solve(16, GATE_B, "li", symmetry="auto")
    assert np.max(np.abs(Ra - Ri)) <= 1e-11
    assert np.max(np.abs(Ta - Ti)) <= 1e-11


def _romberg_last(ys):
    """Eliminate S^-2, S^-4, ... from values at S, 2S, 4S, ..."""
    row, j = list(ys), 1
    while len(row) > 1:
        f = 4.0 ** j
        row = [(f * row[i + 1] - row[i]) / (f - 1) for i in range(len(row) - 1)]
        j += 1
    return row[0]


@pytest.mark.parametrize("theta_deg", [0.0, 20.0])
def test_node_raster_li_converges_to_analytic_li(theta_deg):
    N = 16
    th = np.deg2rad(theta_deg)
    _, Ra, _ = _shapes_solve(N, GATE_B, "li", theta=th)
    diffs = []
    for m in (16, 32, 64, 128):
        st = _stack(N)
        st.add_layer(DEPTH, eps_cell=_node_cell(m * N, GATE_B),
                     formulation="li")
        _, Rr, _ = _solve(st, theta=th)
        diffs.append(Rr - Ra)
    for row in (TM, TE):
        raw = np.max(np.abs(diffs[-1][row]))
        ext = np.max(np.abs(_romberg_last([d[row] for d in diffs])))
        assert raw > 1e-9                 # the pixelation error is real ...
        assert ext <= 1e-10               # ... and extrapolates away
        assert ext <= 1e-4 * raw


@pytest.mark.parametrize("n_sub", [1.0, 1.5])
@pytest.mark.parametrize("theta_deg", [0.0, 20.0])
def test_lossless_dielectric_line_conserves_energy(n_sub, theta_deg):
    st = _stack(32, n_sub=n_sub)
    st.add_layer(300e-9, shapes=_shapes([(0.3, 0.37, 4.0 + 0j)]),
                 eps_background=1.0 + 0j, formulation="li")
    _, R, T = _solve(st, theta=np.deg2rad(theta_deg))
    closure = R.sum(axis=1) + T.sum(axis=1) - 1.0
    assert np.max(np.abs(closure)) <= 1e-12


def test_dispersive_li_shapes_match_concrete():
    epsf = lambda wl: EPS_M * (wl / LAM) ** 2       # noqa: E731
    st = _stack(12)
    sh = dict(_shapes(GATE_B)[0], eps=epsf)
    st.add_layer(DEPTH, shapes=[sh], eps_background=1.0 + 0j,
                 formulation="li")
    st.set_source(LAM)
    wls = np.array([1.25e-6, 1.31e-6])
    _, R, _, _ = st.solve_vs_wavelength(wls, max_workers=1)
    for i, wl in enumerate(wls):
        sc = _stack(12)
        sc.add_layer(DEPTH, shapes=_shapes([(0.25, 0.5, epsf(wl))]),
                     eps_background=1.0 + 0j, formulation="li")
        _, Rc, _ = sc.set_source(wl).solve().efficiencies()
        assert abs(R[i, TM].sum() - np.sum(Rc[TM])) <= 1e-11


def test_eig_key_separates_formulations():
    def layer(f):
        st = _stack(8)
        st.add_layer(DEPTH, shapes=_shapes(GATE_B), eps_background=1.0 + 0j,
                     formulation=f)
        return st.layers[0]
    assert RCWAStack._layer_eig_key(layer("li")) != \
        RCWAStack._layer_eig_key(layer("laurent"))
    # and a mixed stack does not silently reuse one layer's modes
    def two(fa, fb):
        st = _stack(16)
        for f in (fa, fb):
            st.add_layer(DEPTH / 2, shapes=_shapes(GATE_B),
                         eps_background=1.0 + 0j, formulation=f)
        return _solve(st, symmetry=False)[1][TM].sum()
    mixed, li, la = two("laurent", "li"), two("li", "li"), two("laurent",
                                                               "laurent")
    assert abs(mixed - li) > 1e-4 and abs(mixed - la) > 1e-4


def test_li_shapes_refused_on_2d_stack():
    for noy in (3, 0):
        st = lum.RCWAStack(PER, period_y=PER, n_orders=3, n_orders_y=noy)
        with pytest.raises(ValueError, match="1-D stacks only"):
            st.add_layer(DEPTH, shapes=_shapes(GATE_B), eps_background=1.0,
                         formulation="li")
    # Laurent shapes on a 2-D stack are unaffected
    st = lum.RCWAStack(PER, period_y=PER, n_orders=3, n_orders_y=3)
    st.add_layer(DEPTH, shapes=_shapes(GATE_B), eps_background=1.0)


def test_li_shapes_refuses_y_varying_list_and_zero_background():
    with pytest.raises(ValueError, match="y-INVARIANT"):
        _stack(5).add_layer(DEPTH, shapes=[{"shape": "disk", "eps": EPS_M,
                                            "radius": 100e-9}],
                            eps_background=1.0, formulation="li")
    with pytest.raises(ValueError, match="non-zero eps_background") as ei:
        _stack(5).add_layer(DEPTH, shapes=_shapes(GATE_B),
                            eps_background=0.0, formulation="li")
    assert "(got 0j)" in str(ei.value)            # plain, not np.complex128(...)
    with pytest.raises(ValueError, match="uniform and tensor"):
        _stack(5).add_layer(DEPTH, eps=2.0, formulation="li")


# Conical pin (independent oracle).  Centred (mirror-symmetric) dielectric line,
# eps 4, duty 0.37, 300 nm, air / n = 1.5, lambda 1.31 um, theta 20, phi 60.
# Reference: pmm_jones_1d_conical, degree 56 (deg 44 / 72 move it by 1.2e-8 /
# 6.1e-9 in row 0 and 8.8e-9 / 4.3e-9 in row 1).  At n_orders = 64, 'li' is
# off by -9.5e-8 / +6.1e-8 (rows 0 / 1) while Laurent is off by -3.6e-5 /
# +8.2e-6 -- so a 5e-7 bar passes 'li' with 5x margin and fails Laurent by 16x.
_CONICAL_PMM_REF = (0.0010759256291830242, 0.06429739949222246)


@pytest.mark.parametrize("formulation, passes", [("li", True),
                                                 ("laurent", False)])
def test_conical_dielectric_line_matches_pmm(formulation, passes):
    st = _stack(64)
    st.add_layer(300e-9, shapes=[{"shape": "rectangle", "eps": 4.0 + 0j,
                                  "size": (0.37 * PER, PER),
                                  "center": (0.5 * PER, 0.5 * PER)}],
                 eps_background=1.0 + 0j, formulation=formulation)
    _, R, _ = _solve(st, theta=np.deg2rad(20.0), phi=np.deg2rad(60.0))
    err = np.abs(R.sum(axis=1) - np.asarray(_CONICAL_PMM_REF))
    if passes:
        assert np.all(err <= 5e-7), err
    else:
        assert np.all(err > 5e-6), err        # the bar really separates them


@pytest.mark.parametrize("n", [8, 32])
def test_singular_inverse_rule_refuses(n):
    """A lossless eps = -1 line in eps = 1 at duty 0.5 has cell-average
    1/eps = 0: cond([[1/eps]]) is 2.5e15 (N 8) .. 4.3e13 (N 512).  Before the
    guard the analytic inverse silently returned R_TM 0.986 / 1.0009 with an
    energy closure of -1.35e-2."""
    st = _stack(n, n_sub=1.0)
    st.add_layer(DEPTH, shapes=_shapes([(0.25, 0.5, -1.0 + 0j)]),
                 eps_background=1.0 + 0j, formulation="li")
    with pytest.raises(np.linalg.LinAlgError, match="ill-posed"):
        _solve(st)


@pytest.mark.parametrize("eps", [complex(-20, 0), complex(-50, 5),
                                 complex(-1000, 0)])
def test_ordinary_metals_pass_the_conditioning_guard(eps):
    # measured cond([[1/eps]]) <= 3.3e4 for all of these, N 8..512
    _, R, T = _shapes_solve(32, [(0.25, 0.5, eps)], "li")
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
