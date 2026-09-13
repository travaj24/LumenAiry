"""VERIFY-A4 -- independent re-verification of WP-A4 (commit 32ba3ba2).

Every oracle here is written IN THIS FILE from a textbook identity, on a
fixture WP-A4 did not use.  Where a WP-A4 test already pins a property on its
own fixture this file adds a second, differently-shaped one; where the WP's
oracle imported the library helper it was verifying, the identity is restated
from first principles.

Measurements are dated 2026-09-12 on commit 32ba3ba2 (+ this verification
pass), Windows / python 3.14.6 / numpy 2.4.4, OPENBLAS_NUM_THREADS=1.
"""
from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest

from lumenairy.elements.lenses import _multi_indices_total_degree
from lumenairy.elements.lenses_maslov import (
    _GRAM_COND_MAX,
    _GRAM_COND_SINGULAR,
    _input_direction_cosines,
    _integrate_local_quadrature,
    _integrate_stationary_phase,
    _solve_fit,
    _v2_oscillation_bound,
    _van_vleck_density,
    apply_real_lens_maslov_vector,
)
from lumenairy.propagators.asymptotic_maslov import (
    lg00_sampling_waist_from_M,
    sym2x2_max_eigenvalue,
    van_vleck_weight,
)

# ===========================================================================
# S2 -- the closed-form quadratic Fresnel integral, on charts and knobs the
#       WP did not use (cross term, anamorphic half-widths, cx != cy != 1,
#       odd sample counts, window_sigma off the WP's ladder).
# ===========================================================================

_PO = 2
_MI = _multi_indices_total_degree(4, _PO)
_K1 = np.array([k[0] for k in _MI], np.int64)
_K2 = np.array([k[1] for k in _MI], np.int64)
_K3 = np.array([k[2] for k in _MI], np.int64)
_K4 = np.array([k[3] for k in _MI], np.int64)
_IDX = {k: j for j, k in enumerate(_MI)}


def _quadratic_chart(a, b, d, cx, cy):
    """Chebyshev coefficients of ``Phi = 0.5 (a u3^2 + 2b u3 u4 + d u4^2)``
    [waves] with ``s1x = cx u3``, ``s1y = cy u4`` [m]."""
    co = np.zeros(len(_MI))
    co[_IDX[(0, 0, 0, 0)]] += 0.25 * a + 0.25 * d   # u^2 = (T0 + T2)/2
    co[_IDX[(0, 0, 2, 0)]] += 0.25 * a
    co[_IDX[(0, 0, 0, 2)]] += 0.25 * d
    co[_IDX[(0, 0, 1, 1)]] += b                     # u3 u4 = T1 T1
    sx = np.zeros(len(_MI)); sx[_IDX[(0, 0, 1, 0)]] = cx
    sy = np.zeros(len(_MI)); sy[_IDX[(0, 0, 0, 1)]] = cy
    return co, sx, sy


def _fresnel_exact(a, b, d, cx, cy, hx, hy):
    """Textbook value of the integral the Maslov integrators evaluate.

        I = INT E_in sqrt|det ds1/dv2| exp(2 pi i Phi) d^2 v2

    With ``v2 = (hx u3, hy u4)`` and ``E_in == 1``,
    ``det(ds1/dv2) = cx cy / (hx hy)`` is constant, so

        I = sqrt(cx cy hx hy) INT exp(i pi u^T H u) d^2 u
          = sqrt(cx cy hx hy) exp(i pi sigma / 4) / sqrt|det H|

    (stationary-phase / Fresnel, ``H = [[a, b], [b, d]]``, ``sigma`` = the
    signature = #positive - #negative eigenvalues).  Derived here; nothing in
    it comes from the library.
    """
    H = np.array([[a, b], [b, d]], float)
    ev = np.linalg.eigvalsh(H)
    sigma = int(np.sum(ev > 0) - np.sum(ev < 0))
    return (math.sqrt(cx * cy * hx * hy)
            * np.exp(1j * np.pi * sigma / 4.0)
            / math.sqrt(abs(np.linalg.det(H))))


def _ones(s1x, s1y):
    return np.ones_like(s1x, dtype=np.complex128)


def _noprog(*a, **k):
    pass


def _run_pair(a, b, d, cx, cy, hx, hy, n, ws):
    co, sx, sy = _quadratic_chart(a, b, d, cx, cy)
    u0 = np.zeros((1, 1))
    inbox = np.array([True])
    lq = _integrate_local_quadrature(
        co, sx, sy, _K1, _K2, _K3, _K4, _PO, 1, u0, u0, inbox, hx, hy,
        _ones, 30, 1e-12, n, ws, _noprog, False)[0, 0]
    sp = _integrate_stationary_phase(
        co, sx, sy, _MI, _K1, _K2, _K3, _K4, _PO, 1, u0, u0, inbox, hx, hy,
        _ones, 30, 1e-12, _noprog, False)[0, 0]
    return _fresnel_exact(a, b, d, cx, cy, hx, hy), lq, sp


# (a, b, d, cx, cy, hx, hy) -- every one has a feature the WP's p6/p7 charts
# lack: a cross term, cx != cy (so the S4 sqrt does not cancel), anamorphic
# half-widths, an indefinite Hessian, or all four at once.
_S2_CHARTS = [
    pytest.param(37.0, 0.0, 5.0, 2.5, 0.4, 0.031, 0.019, id='anamorphic'),
    pytest.param(12.0, 9.0, 30.0, 1.7, 1.7, 0.02, 0.02, id='cross-term'),
    pytest.param(60.0, -28.0, 19.0, 0.31, 2.9, 0.045, 0.007, id='tilted'),
    pytest.param(-45.0, 11.0, 7.0, 1.0, 1.0, 0.02, 0.02, id='saddle'),
    pytest.param(-22.0, 0.0, -63.0, 0.8, 1.9, 0.011, 0.033, id='negative'),
    pytest.param(300.0, 120.0, 90.0, 1.3, 0.7, 0.05, 0.02, id='big-tilted'),
]


@pytest.mark.parametrize('a, b, d, cx, cy, hx, hy', _S2_CHARTS)
@pytest.mark.parametrize('n, ws', [(8, 3.0), (7, 2.5), (33, 2.5)],
                         ids=['n8ws3', 'n7ws2.5-odd', 'n33ws2.5-odd'])
def test_s2_local_quadrature_matches_the_closed_form_fresnel_integral(
        a, b, d, cx, cy, hx, hy, n, ws):
    """S2: the fixed ``local_quadrature`` reproduces the closed-form value.

    Bar.  The scheme divides its own Gaussian taper out of the QUADRATIC
    MODEL on the SAME finite lattice, so for a quadratic chart with constant
    amplitude it is exact up to float64 cancellation; the error floor is the
    ~``n^2``-term summation of O(1) values, i.e. ~1e-15.  Gate at 1e-11:
    4 decades above that floor and 13 decades below the PRE-FIX errors the
    audit measured on the same class of chart (relerr 1.19 at the shipped
    defaults, 9.12 with the axis swap, 2.5-3.1 with a cross term).  Measured
    here, worst over the 18 (chart, knob) pairs: 8.33e-15.

    The (n, ws) pairs deliberately include ODD sample counts and window
    widths off the WP's {3, 4, 6, 10} x {8, 16, 32, 64, 128} ladder.  They
    are all knobs whose tapered lattice stays INSIDE the fitted chart box on
    every chart here; the boundary itself is pinned separately in
    ``test_s2_window_wider_than_the_chart_box_degrades_but_is_bounded``.
    """
    exact, lq, _sp = _run_pair(a, b, d, cx, cy, hx, hy, n, ws)
    rel = abs(lq - exact) / abs(exact)
    assert rel < 1e-11, (
        f'local_quadrature = {lq!r} vs the closed-form Fresnel value '
        f'{exact!r}: relative error {rel:.3e} (measured <= 8.33e-15 on this '
        f'set; pre-fix the same class of chart scored 1.19 .. 9.12)')


@pytest.mark.parametrize('a, b, d, cx, cy, hx, hy', _S2_CHARTS)
def test_s2_stationary_phase_is_exact_on_a_quadratic_chart(
        a, b, d, cx, cy, hx, hy):
    """The asymptotic evaluator ``'auto'`` now falls back to is EXACT on a
    quadratic chart -- which is what makes the S2 routing change safe.

    Bar 1e-13: the leading-order saddle IS the exact answer for a quadratic
    exponent, so the only error is float64 round-off in ``1/sqrt(det H)``
    (~1e-16); measured worst 2.13e-16 over these six charts.
    """
    exact, _lq, sp = _run_pair(a, b, d, cx, cy, hx, hy, 8, 3.0)
    rel = abs(sp - exact) / abs(exact)
    assert rel < 1e-13, (
        f'stationary_phase = {sp!r} vs {exact!r}, relative {rel:.3e}')


def test_s2_swap_symmetry_holds_with_a_cross_term_and_anamorphic_widths():
    """S2(a): the pre-fix window was scaled by the Hessian EIGENVALUES but
    laid out on the COORDINATE axes, so swapping the two axes changed the
    answer (the audit measured a factor 7.7 in relative error).  The exact
    value is invariant under the simultaneous swap, so the integrator must
    be too.

    Bar 1e-12 relative -- 3 decades above the measured 1.22e-14 worst and
    well below any axis-swap signature.
    """
    for (a, b, d) in [(40.0, 0.0, 4.0), (12.0, 9.0, 30.0),
                      (300.0, 120.0, 90.0)]:
        e1, l1, _ = _run_pair(a, b, d, 2.5, 0.4, 0.031, 0.019, 8, 3.0)
        e2, l2, _ = _run_pair(d, b, a, 0.4, 2.5, 0.019, 0.031, 8, 3.0)
        assert abs(e1 - e2) / abs(e1) < 1e-15, 'premise: exact values agree'
        assert abs(l1 - l2) / abs(l1) < 1e-12, (
            f'(a,b,d)=({a},{b},{d}): swap asymmetry '
            f'{abs(l1 - l2) / abs(l1):.3e}')


def test_s2_window_wider_than_the_chart_box_degrades_but_is_bounded():
    """VERIFY finding.  The S2 docstring claims the scheme is exact "at ANY
    ``local_n_samples`` / ``local_window_sigma``".  It is exact only while
    the tapered lattice FITS INSIDE the fitted chart box: out-of-box samples
    are dropped (correctly -- the Chebyshev recurrences are not accurate
    there) while the taper correction is still computed on the full lattice.

    MEASURED on the 'anamorphic' chart above, where the larger principal
    width is ``sigma2_norm = 0.2524``:

        ws = 3.0 -> 3 sigma reaches |u| = 0.757 (inside)  : 6.60e-15
        ws = 3.7 -> 0.934 (inside)                        : 2.75e-15
        ws = 5.0 -> 1.262 (OUTSIDE)                       : 8.09e-02
        ws = 7.5 -> 1.893 (OUTSIDE)                       : 8.79e-02

    and up to 8.17e-01 on a chart whose small Hessian eigenvalue makes
    ``sigma2_norm = 1.785``.  This test pins the boundary so the claim is
    not read as unconditional, and pins that the out-of-box regime stays
    BOUNDED (the pre-fix ``np.clip`` over-count reached 2.0e+03).
    """
    inside = _run_pair(37.0, 0.0, 5.0, 2.5, 0.4, 0.031, 0.019, 8, 3.0)
    outside = _run_pair(37.0, 0.0, 5.0, 2.5, 0.4, 0.031, 0.019, 13, 7.5)
    r_in = abs(inside[1] - inside[0]) / abs(inside[0])
    r_out = abs(outside[1] - outside[0]) / abs(outside[0])
    assert r_in < 1e-11, f'in-box: {r_in:.3e}'
    assert 1e-3 < r_out < 1.0, (
        f'out-of-box relerr {r_out:.3e}: expected the honest truncation cost '
        f'(measured 8.79e-02), not exactness and not the pre-fix clip '
        f'over-count (7.5e+01 .. 2.0e+03)')


def test_s2_v2_oscillation_bound_is_the_total_variation_not_the_excursion():
    """S9: ``_v2_oscillation_bound`` weights each Chebyshev coefficient by
    ``max(k3, k4)``.  The identity behind it -- ``TV(T_k) = 2k`` on [-1, 1]
    -- is MEASURED here on a 400 001-point grid rather than asserted, and
    the bound is checked to actually bound a directly-sampled half-total-
    variation of a random order-8 chart.

    Measured: TV(T_k) = 2.000000 / 4.000000 / 6.000000 / 10.000000 /
    16.000000 for k = 1 / 2 / 3 / 5 / 8 (exact to 6 dp at this sampling).
    Random order-8 chart: bound 281.90 vs a measured half-TV of 8.07 along
    one v2 line, and 99.07 for the pre-fix excursion sum -- so the new
    estimator is 2.85x the old one on that chart, in line with the audit's
    measured 2.47x under-count.
    """
    u = np.linspace(-1.0, 1.0, 400_001)

    def T(k, x):
        return np.cos(k * np.arccos(np.clip(x, -1.0, 1.0)))

    for k in (1, 2, 3, 5, 8):
        tv = float(np.sum(np.abs(np.diff(T(k, u)))))
        assert abs(tv - 2.0 * k) < 1e-4, (
            f'premise TV(T_{k}) = 2k: measured {tv:.6f}')

    mi = _multi_indices_total_degree(4, 8)
    rng = np.random.default_rng(7)
    coef = rng.normal(scale=0.3, size=len(mi))
    bound = _v2_oscillation_bound(mi, coef)
    line = np.zeros_like(u)
    for c, k in zip(coef, mi):
        line += c * T(k[0], 0.3) * T(k[1], -0.2) * T(k[2], u) * T(k[3], 0.0)
    half_tv = 0.5 * float(np.sum(np.abs(np.diff(line))))
    old = float(np.sum(np.abs(coef) * np.array(
        [1.0 if (k[2] > 0 or k[3] > 0) else 0.0 for k in mi])))
    assert bound >= half_tv, (
        f'the estimator must BOUND the sampled half-total-variation: '
        f'bound {bound:.4f} < measured {half_tv:.4f}')
    assert bound > 2.0 * old, (
        f'and must exceed the pre-fix excursion sum on an order-8 chart '
        f'(measured 281.90 vs 99.07 = 2.85x); got {bound:.4f} vs {old:.4f}')
    # a chart with no v2 dependence must score exactly zero
    flat = np.zeros(len(mi))
    flat[[j for j, k in enumerate(mi) if k[2] == 0 and k[3] == 0]] = 1.0
    assert _v2_oscillation_bound(mi, flat) == 0.0


# ===========================================================================
# S4 / Y2 -- the Van Vleck density and weight, from the textbook identity.
# ===========================================================================

def test_s4_van_vleck_density_is_the_square_root_of_the_physical_jacobian():
    """S4: ``_van_vleck_density(|det ds1/du_v2|, hx, hy)`` must be
    ``|det ds1/dv2|^(1/2)``.

    Written out here: ``u3 = (v2x - c)/hx`` gives
    ``det(ds1/dv2) = det(ds1/du) / (hx hy)``, and Van Vleck-Morette in
    d = 2 takes its SQUARE ROOT.  Bar 8 ULP; measured 0 ULP over the ladder.
    Pre-fix the integrand carried the FIRST power, which on the free-space
    chart made the field ``i lambda z`` times the truth.
    """
    for det_u in (1e-9, 1.0, 3.7, 1e6):
        for hx, hy in ((0.02, 0.02), (0.031, 0.019), (0.5, 1e-3)):
            got = float(_van_vleck_density(det_u, hx, hy))
            want = math.sqrt(det_u / (hx * hy))
            assert abs(got - want) <= 8 * np.finfo(float).eps * want, (
                f'_van_vleck_density({det_u}, {hx}, {hy}) = {got!r} != '
                f'sqrt({det_u}/({hx}*{hy})) = {want!r}')
    # ... and it is NOT the pre-fix first power, on a fixture where the two
    # differ by 3 decades: det_u = 1e6, hx = hy = 0.02 -> 5.0e4 vs 2.5e9.
    assert abs(float(_van_vleck_density(1e6, 0.02, 0.02)) / 5.0e4 - 1) < 1e-12


def test_y2_van_vleck_weight_is_the_textbook_fresnel_amplitude():
    """Y2: ``van_vleck_weight(|det J|, lam) == -1j sqrt(|det J|) / lam``.

    Derivation (independent of the library): the d = 2 semiclassical kernel
    is ``K = (k / (2 pi i)) |det d2S/ds1 ds2|^(1/2) exp(i k S)``; changing
    the integration variable ``s1 -> v2`` at fixed ``s2`` contributes
    ``d^2 s1 = |det J| d^2 v2`` against
    ``|det d2S/ds1 ds2|^(1/2) = |det J|^(-1/2)``, leaving
    ``(1/(i lambda)) |det J|^(1/2)``.

    This is the SAME identity the two W6 brute-force oracles now use
    (``_vv_weight_textbook`` there); the point of duplicating it is that
    both files derive it rather than importing it.  Bar 8 ULP; measured
    0 ULP.
    """
    for det_J in (1e-12, 3.980321e-06, 1.0, 7.076126e-06, 1e9):
        for lam in (1.31e-6, 633e-9, 10.6e-6):
            a = complex(van_vleck_weight(det_J, lam))
            b = -1j * math.sqrt(det_J) / lam
            assert abs(a - b) <= 8 * np.finfo(float).eps * abs(b), (
                f'van_vleck_weight({det_J}, {lam}) = {a!r} != {b!r}')
    # the |L|^2 move the two LG-merit re-pins are derived from:
    # |w|^2 = |det J| / lambda^2, so |L|^2 moves by 1/(lambda^2 |det J|).
    det_J, lam = 3.980321e-06, 1.30e-6
    factor = abs(complex(van_vleck_weight(det_J, lam))) ** 2 / det_J ** 2
    assert abs(factor * (lam ** 2 * det_J) - 1.0) < 1e-12, (
        f'|w/detJ|^2 = {factor:.6e} should be 1/(lam^2 detJ) = '
        f'{1.0 / (lam ** 2 * det_J):.6e}')


# ===========================================================================
# Y3 -- the closed-form 2x2 eigenvalue and the shared w_o helper.
# ===========================================================================

def test_y3_sym2x2_max_eigenvalue_matches_numpy_linalg_including_degeneracy():
    """Y3: the closed form replacing ``eigvalsh`` must BE ``eigvalsh``.

    207 matrices: seven hand-picked corner cases (exact degeneracy, exact
    zero, 1e-14 scale, 1e12 scale, indefinite, a 1e-15 gap) plus 200 random
    normal ones.  Bar 1e-12 relative to ``max(|lam|, |a|, |b|, |d|)``, which
    is the scale ``eigvalsh`` itself is accurate to; measured worst 3.67e-16.
    """
    rng = np.random.default_rng(20260912)
    cases = [(1.0, 0.0, 1.0), (1e-14, 0.0, 1e-14), (3.0, 4.0, -3.0),
             (1e12, 1e-6, 1e12), (-5.0, 0.0, -7.0), (0.0, 0.0, 0.0),
             (1.0, 1e-15, 1.0 + 1e-15)]
    cases += [tuple(rng.normal(scale=10.0, size=3)) for _ in range(200)]
    worst = 0.0
    for (a, b, d) in cases:
        got = float(sym2x2_max_eigenvalue(np.float64(a), np.float64(b),
                                          np.float64(d), np))
        ref = float(np.linalg.eigvalsh(np.array([[a, b], [b, d]]))[-1])
        scale = max(abs(ref), abs(a), abs(b), abs(d), 1e-300)
        worst = max(worst, abs(got - ref) / scale)
    assert worst < 1e-12, f'worst relative departure {worst:.3e}'


def test_y3_lg00_sampling_waist_clamps_on_both_sides_and_has_a_fallback():
    """Y5: the cross-backend ``w_o`` contract is now enforced by ONE helper.
    Pin all three of its documented branches, since the JAX twin's NaN-free
    gradient depends on them.  Values are exact by construction, so the bars
    are exact equalities (no tolerance to derive).
    """
    assert float(lg00_sampling_waist_from_M(
        np.diag([1e18, 1e18]) + 0j, np)) == 1e-9        # lower clamp
    assert float(lg00_sampling_waist_from_M(
        np.diag([1e-4, 1e-4]) + 0j, np)) == 1.0         # upper clamp
    assert float(lg00_sampling_waist_from_M(
        np.diag([-1.0, -2.0]) + 0j, np)) == 1e-6        # lam_max <= 0
    # In-range and near-degenerate: the 1e-11 off-diagonal genuinely moves
    # lambda_max to 2 + 1e-11, so the reference is eigvalsh's own value, not
    # 1/sqrt(2).  Bar 1e-15 relative; measured 0.0.
    M = np.array([[2.0, 1e-11], [1e-11, 2.0]])
    got = float(lg00_sampling_waist_from_M(M + 0j, np))
    want = 1.0 / math.sqrt(float(np.linalg.eigvalsh(M)[-1]))
    assert abs(got - want) <= 1e-15 * want


# ===========================================================================
# NEW (WP-A4's own finding) -- the _solve_fit conditioning gate.
# ===========================================================================

def test_solve_fit_gate_threshold_is_derived_not_fitted_to_the_fixture():
    """The two constants must be derivable, not tuned.

    ``_GRAM_COND_SINGULAR`` is exactly ``1/eps`` -- the point at which the
    Gram's smallest eigenvalue is indistinguishable from round-off in its
    largest.  ``_GRAM_COND_MAX = 1e12`` leaves ``eps * 1e12 = 2.2e-04``
    relative accuracy in the squared system, i.e. ~4 float64 digits, and
    sits 3 decades above the audit's own WELL-conditioned fixture
    (cond(G) = 1.24e9 at poly_order 4) and 6 decades below the failing one
    (6.18e18) -- so it is not placed to make one fixture pass.
    """
    assert _GRAM_COND_SINGULAR == 1.0 / np.finfo(np.float64).eps
    assert _GRAM_COND_MAX == 1e12
    assert np.finfo(np.float64).eps * _GRAM_COND_MAX < 1e-3, (
        'the gate must leave at least 3 significant digits in the squared '
        'system')
    assert 1.24e9 < _GRAM_COND_MAX < 6.18e18, (
        'the gate must separate the audit\'s well-conditioned fixture from '
        'its rank-deficient one, with margin on both sides')


def test_solve_fit_gate_returns_the_minimum_norm_solution_on_a_deficient_gram():
    """The gate's PROPERTY, on a basis whose rank deficiency is built by
    hand (column 8 = column 1 + column 2) rather than discovered on a lens.

    Pre-fix, ``cho_factor`` on a numerically positive-semidefinite but
    rank-deficient Gram succeeds and returns an ARBITRARY member of the
    solution set -- the audit measured two runs of the same optic, whose
    design matrices agreed to 3.1e-15, coming back 0.869 WAVES apart.

    Measured here: cond(A) = 1.06e+16, cond(G) = 2.13e+302, rank 7 of 8.
    The gate returns exactly ``np.linalg.lstsq``'s minimum-norm solution
    (0.000e+00 difference -- the same call), and a 1e-14 perturbation of A
    moves the coefficients by 7.99e-15, i.e. it is a continuous function of
    the data.  Bars: 1e-30 on the lstsq identity (it is the same solve) and
    1e-9 on the perturbation stability, 5 decades below the pre-fix 0.869.
    """
    rng = np.random.default_rng(3)
    A = rng.normal(size=(60, 8))
    A[:, -1] = A[:, 0] + A[:, 1]
    RHS = A @ rng.normal(size=(8, 1))
    G = A.T @ A
    ev = np.linalg.eigvalsh(G)
    assert ev[-1] / max(ev[0], 1e-300) > _GRAM_COND_SINGULAR, (
        'premise: this Gram must be past the warn threshold')

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        c1 = _solve_fit(A, RHS)
        c2 = _solve_fit(A + 1e-14 * rng.normal(size=A.shape), RHS)
    assert any('RANK-DEFICIENT' in str(w.message) for w in rec), (
        f'a rank-deficient Gram must warn; got {[str(w.message) for w in rec]}')
    mn, *_ = np.linalg.lstsq(A, RHS, rcond=None)
    assert np.max(np.abs(c1 - mn)) < 1e-30
    assert np.max(np.abs(c1 - c2)) < 1e-9, (
        f'a 1e-14 perturbation of A moved the coefficients by '
        f'{np.max(np.abs(c1 - c2)):.3e} (pre-fix: 0.869 waves)')


def test_solve_fit_well_conditioned_path_is_bit_identical_to_the_cholesky():
    """The gate must not change a well-conditioned fit AT ALL -- that is the
    backward-compatibility half of the claim, and the only bar that can
    express it is bit-identity.  cond(G) = 2.39 here.
    """
    scipy_linalg = pytest.importorskip('scipy.linalg')
    rng = np.random.default_rng(11)
    A = rng.normal(size=(200, 12))
    RHS = rng.normal(size=(200, 3))
    G = A.T @ A
    ev = np.linalg.eigvalsh(G)
    assert ev[-1] / ev[0] < _GRAM_COND_MAX, 'premise: well conditioned'
    ref = scipy_linalg.cho_solve(
        scipy_linalg.cho_factor(G, check_finite=False), A.T @ RHS,
        check_finite=False)
    assert np.array_equal(_solve_fit(A, RHS), ref)


# ===========================================================================
# S10 (third sub-item) -- the vector wrapper's joint normalisation.
# ===========================================================================

def _vector_fixture():
    wl, N, dxg = 1.55e-6, 96, 16e-6
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (0.55e-3) ** 2)
    E_vec = np.stack([(0.8 * amp).astype(np.complex128),
                      (0.6 * amp).astype(np.complex128)], axis=0)
    pres = {'surfaces': [
        {'radius': 8e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -8e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [2.5e-3], 'aperture_diameter': 2.4e-3}
    return E_vec, dict(prescription=pres, wavelength=wl, dx=dxg,
                       integration_method='quadrature', n_v2=48, poly_order=4)


def test_s10_vector_normalisation_is_one_joint_scale_for_the_pair():
    """S10 third sub-item: ``normalize_output`` must not touch the
    polarization ratio.

    Pre-fix the mode was forwarded to the two scalar legs, which normalised
    INDEPENDENTLY -- ``'power'`` then forced ``P_x/P_y`` back to the
    post-Fresnel INPUT ratio, deleting the diattenuation the wrapper exists
    to compute.

    MEASURED on an f/3.3 N-BK7 biconvex, 45-degree-ish linear input
    (P_x/P_y = 1.777777777778 exactly): the propagated ratio is
    1.777777663332023 under ALL THREE modes -- bit-identical, 0 ULP -- and
    it departs from the input ratio by -6.4376e-08, which is the s/p
    diattenuation and is 5.154e+08 ULP of the ratio, so the pre-fix
    behaviour (ratio == input ratio exactly) is distinguishable by 8
    decades.  ``'power'`` restores the POST-FRESNEL pair's total power to
    1.000000000000000 (= 0.9219035 of the raw input pair's, i.e. the
    two-surface Fresnel transmission survives in the absolute scale);
    ``'none'`` is 28.46x that, so it really is un-normalised.
    """
    E_vec, kw = _vector_fixture()
    ratios, powers = {}, {}
    for mode in ('none', 'power', 'peak'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens_maslov_vector(
                E_vec.copy(), normalize_output=mode, **kw)
        px = float(np.sum(np.abs(out[0]) ** 2))
        py = float(np.sum(np.abs(out[1]) ** 2))
        ratios[mode], powers[mode] = px / py, px + py
    r0 = ratios['none']
    for mode in ('power', 'peak'):
        # One joint scale ``s`` makes the ratio ``(a s)^2 / (b s)^2``, which
        # is exact up to the rounding of the two products: bar 4 ULP.
        # Measured: 0 ULP for 'power', 1 ULP for 'peak'.
        assert abs(ratios[mode] - r0) <= 4 * np.spacing(r0), (
            f"normalize_output={mode!r} changed P_x/P_y from {r0!r} to "
            f"{ratios[mode]!r} by {abs(ratios[mode] - r0) / np.spacing(r0):.1f}"
            f" ULP -- it must be ONE joint scale")
    r_in = (float(np.sum(np.abs(E_vec[0]) ** 2))
            / float(np.sum(np.abs(E_vec[1]) ** 2)))
    dev = abs(r0 / r_in - 1.0)
    assert dev > 1e-9, (
        f'the output ratio ({r0!r}) must still carry the s/p diattenuation '
        f'(measured -6.4376e-08 relative to the input ratio {r_in!r}); got '
        f'{dev:.3e}.  Equality with the input ratio is the pre-fix defect.')
    # The reference is the POST-FRESNEL pair -- the field the scalar legs were
    # handed -- so the surface transmission stays in the absolute scale.
    p_in = _post_fresnel_power(E_vec, kw)
    p_raw = float(np.sum(np.abs(E_vec[0]) ** 2 + np.abs(E_vec[1]) ** 2))
    assert abs(p_in / p_raw - 0.9219035) < 1e-5, (
        f'premise: the two-surface Fresnel transmission on this fixture is '
        f'0.9219035; measured {p_in / p_raw!r}')
    assert abs(powers['power'] / p_in - 1.0) < 1e-12, (
        f"normalize_output='power' must restore the PAIR's post-Fresnel "
        f'total power; got {powers["power"] / p_in!r}')
    assert abs(powers['none'] / p_in - 1.0) > 1.0, (
        f"premise: 'none' must NOT be normalised (measured 28.46x the "
        f'post-Fresnel power on this chart); got '
        f'{powers["none"] / p_in:.4f}')


def _post_fresnel_power(E_vec, kw):
    """``sum(|ExM|^2 + |EyM|^2)`` -- the pair the two scalar legs are handed,
    i.e. the input after the per-pixel Fresnel Jones.  Rebuilt here from the
    same two public pieces the wrapper uses, so the pin does not read the
    number back out of the function under test."""
    from lumenairy.elements.lenses_maslov import _input_direction_cosines
    from lumenairy.propagators.gbd import _fresnel_jones_matrix_per_beamlet
    N = E_vec.shape[-1]
    dxg = kw['dx']
    ix = np.arange(N)
    Ix, Iy = np.meshgrid(ix, ix, indexing='xy')
    xb = (Ix.ravel() - N / 2.0) * dxg
    yb = (Iy.ravel() - N / 2.0) * dxg
    ux, uy = _input_direction_cosines(E_vec, dxg, dxg, kw['wavelength'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        P, _alive = _fresnel_jones_matrix_per_beamlet(
            xb, yb, ux.ravel(), uy.ravel(), kw['prescription'],
            kw['wavelength'])
    ex = np.asarray(E_vec[0]).ravel()
    ey = np.asarray(E_vec[1]).ravel()
    mx = P[:, 0, 0] * ex + P[:, 0, 1] * ey
    my = P[:, 1, 0] * ex + P[:, 1, 1] * ey
    return float(np.sum(np.abs(mx) ** 2 + np.abs(my) ** 2))


def test_s10_vector_rejects_an_unknown_normalize_output():
    E_vec, kw = _vector_fixture()
    with pytest.raises(ValueError, match=r'apply_real_lens_maslov_vector: '
                                         r'normalize_output must be'):
        apply_real_lens_maslov_vector(E_vec, normalize_output='rms', **kw)


def test_s10_vector_launch_directions_are_the_field_s_own_local_wavevector():
    """S10 first sub-item (partial): the polarization base rays are launched
    along the input field's local wavevector, not axially.

    Three properties, each with an exact or derived bar:

    * a REAL, non-negative input gives EXACTLY (0, 0) -- so a collimated
      call is bit-identical to the pre-v5.46 ``ux = uy = 0``;
    * a pure tilt ``exp(i k0 t x)`` is recovered as ``ux = t``;
    * a converging wavefront ``exp(-i pi r^2 / (lam f))`` gives
      ``ux = -x/f`` up to the forward-difference bias ``dx/(2f)``.

    MEASURED (lambda = 1.55 um, dx = 16 um, f = 20 mm): max|ux| = 0.0 flat;
    mean ux = 0.030000 for t = 0.03 (grid Nyquist 0.0484 -- a larger tilt
    wraps, which is the estimator's documented limit and was measured doing
    exactly that at t = 0.05); max departure from ``-x/f`` = 4.000e-04,
    which is ``dx/(2f) = 16e-6/(2*20e-3) = 4.0e-04``.
    """
    wl, N, dxg = 1.55e-6, 96, 16e-6
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / (0.55e-3) ** 2)
    flat = np.stack([(0.8 * amp).astype(np.complex128),
                     (0.6 * amp).astype(np.complex128)], axis=0)
    ux, uy = _input_direction_cosines(flat, dxg, dxg, wl)
    assert np.max(np.abs(ux)) == 0.0 and np.max(np.abs(uy)) == 0.0

    t = 0.03          # below the grid Nyquist lambda/(2 dx) = 0.0484
    tilted = flat * np.exp(1j * 2 * np.pi / wl * t * X)[None, :, :]
    ux2, uy2 = _input_direction_cosines(tilted, dxg, dxg, wl)
    assert abs(float(np.mean(ux2)) - t) < 1e-9
    assert abs(float(np.mean(uy2))) < 1e-12

    f = 20e-3
    conv = flat * np.exp(-1j * np.pi / wl * (X ** 2 + Y ** 2) / f)[None, :, :]
    uxc, _ = _input_direction_cosines(conv, dxg, dxg, wl)
    r = np.hypot(X, Y)
    sel = (r > 0.1e-3) & (r < 0.4e-3)
    bias = dxg / (2.0 * f)
    err = float(np.max(np.abs(uxc[sel] - (-X[sel] / f))))
    assert err <= 1.05 * bias, (
        f'max |ux - (-x/f)| = {err:.3e}; the forward-difference bias alone '
        f'is dx/(2f) = {bias:.3e} (measured 3.000e-04)')


# ===========================================================================
# Ruling 1 -- the LG merit is a dimensionless coupling.
# ===========================================================================

def _lg_fixture(R1=30e-3, ap=12e-3, t=3.0e-3):
    return {'surfaces': [
        {'radius': R1, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [t], 'aperture_diameter': ap}


def test_ruling1_aberration_free_reference_is_the_same_object_when_unaberrated():
    """``aberration_free_reference_fit`` must return the SAME fit when there
    is nothing to remove -- that is what makes the normalised coupling
    exactly 1.0 with no round-off, which is the only 0-ULP bar available
    for "an aberration-free reference reads 1.0".
    """
    from lumenairy.propagators.asymptotic import (
        aberration_free_reference_fit,
        fit_canonical_polynomials,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = fit_canonical_polynomials(_lg_fixture(), wavelength=1.30e-6,
                                        pupil_box_half=0.18,
                                        source_box_half=20e-6, poly_order=6)
    ref = aberration_free_reference_fit(fit)
    assert ref is not fit, 'premise: this chart carries cubic+ pupil phase'
    hi = [j for j, k in enumerate(fit.multi_indices) if k[2] + k[3] >= 3]
    lo = [j for j, k in enumerate(fit.multi_indices) if k[2] + k[3] < 3]
    assert np.all(np.asarray(ref.coef_phi)[hi] == 0.0)
    assert np.array_equal(np.asarray(ref.coef_phi)[lo],
                          np.asarray(fit.coef_phi)[lo])
    assert np.array_equal(ref.coef_s1x, fit.coef_s1x), (
        'the geometry -- hence |det J| -- must be untouched')
    assert np.array_equal(ref.coef_s1y, fit.coef_s1y)
    assert aberration_free_reference_fit(ref) is ref, (
        'idempotent: an already-aberration-free fit comes back unchanged, '
        'so |L/L_ref|^2 is exactly 1.0')


def test_ruling1_sigma_branch_coupling_falls_monotonically_with_aberration():
    """Ruling 1's acceptance property, on the sigma-grid branch.

    Ladder: scale the fit's cubic-and-higher PUPIL phase by alpha and read
    ``|L_00|^2 / |L_00,ref|^2``.  MEASURED (f/2.5 N-BK7 plano-convex,
    pupil_box_half = 0.18, w_s = 20 um, w_p = 0.05, output_modes
    [(0,0), (1,0)], sigma_grid_n = 64):

        alpha  0        0.5      1.0      2.0      4.0
        S      1.000000 0.901312 0.811431 0.659170 0.447292

    alpha = 0 is EXACT (``aberration_free_reference_fit`` returns the same
    object, so it is the same computation twice), so that bar is 0 ULP; the
    monotonicity bar is a strict inequality whose smallest measured step is
    0.090 -- 15 decades above float64 noise on an O(1) value.

    The same ladder on the CLOSED-FORM branch RISES (1.000000 / 1.006273 /
    1.012518 / 1.024914 / 1.049254 / 1.095510 at alpha = 0 / 0.25 / 0.5 /
    1 / 2 / 4, on an 8-mode sigma request), which is why the merit's default
    is 'sigma'.
    """
    from lumenairy.propagators.asymptotic import (
        aberration_free_reference_fit,
        aberration_tensor,
        fit_canonical_polynomials,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = fit_canonical_polynomials(_lg_fixture(), wavelength=1.30e-6,
                                        pupil_box_half=0.18,
                                        source_box_half=20e-6, poly_order=6)
        s2 = (fit.s2x_centre, fit.s2y_centre)
        hi = np.array([1.0 if (k[2] + k[3]) >= 3 else 0.0
                       for k in fit.multi_indices])
        base = np.asarray(fit.coef_phi).copy()
        modes = [(0, 0), (1, 0)]
        w_o = None
        S = []
        for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
            f_a = dataclasses.replace(
                fit, coef_phi=base * (1.0 - hi) + alpha * base * hi)
            Ta = aberration_tensor(
                f_a, s2_image=s2, source_modes=[(0, 0)],
                pupil_modes=[(0, 0)], output_modes=modes, w_s=20e-6,
                w_p=0.05, w_o=w_o, sigma_grid_n=64)
            w_o = Ta.w_o
            Tr = aberration_tensor(
                aberration_free_reference_fit(f_a), s2_image=s2,
                source_modes=[(0, 0)], pupil_modes=[(0, 0)],
                output_modes=modes, w_s=20e-6, w_p=0.05, w_o=w_o,
                sigma_grid_n=64)
            S.append(abs(complex(Ta.L[0, 0])) ** 2
                     / abs(complex(Tr.L[0, 0])) ** 2)
    assert S[0] == 1.0, (
        f'alpha = 0 must be EXACTLY 1.0 (the same computation twice); '
        f'got {S[0]!r}')
    assert all(S[i] > S[i + 1] for i in range(len(S) - 1)), (
        f'the coupling must fall monotonically with aberration; got {S}')
    assert S[-1] < 0.6, (
        f'and must have real dynamic range (measured 0.447292 at alpha = 4); '
        f'got {S[-1]:.6f}')


# ===========================================================================
# S6 -- the saddle warning must fire on a non-flat input and ONLY then.
# ===========================================================================

def _collimated_gaussian(N, dxg, w, wl=1.31e-6):
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128), X, Y


def test_s6_wavefront_na_is_zero_for_a_collimated_beam_of_any_width():
    """VERIFY finding (fixed here).  The S6 gate must test the spread of the
    input's LOCAL WAVEVECTOR, not the second moment of ``|FFT(E_in)|^2``.

    A collimated beam of finite width has a real angular spectrum -- a
    Gaussian of waist ``w`` spreads by ``lambda / (pi w)`` -- while its
    geometric launch direction is ``v1 = 0`` everywhere, which is the case
    the OPD-only saddle gets RIGHT.  WP-A4 gated on the spectral moment, so
    the warning fired on every collimated beam narrower than ~1 mm.

    MEASURED (lambda = 1.31 um, dx = 10 um), 3-sigma NA of a flat-phase
    Gaussian:

        waist w        0.25 mm   0.80 mm   2.0 mm    4.0 mm
        FFT 2nd moment 3.54e-03  1.10e-03  4.2e-04   2.1e-04   <- 2 of 4
                                                                  above 1e-3
        wavefront NA   0.0       0.0       0.0       0.0

    The wavefront bar is EXACT zero (a real non-negative field has
    ``arg(E[i+1] conj(E[i])) == 0`` identically), so no tolerance is needed.
    """
    from lumenairy.elements.lenses_maslov import _wavefront_na
    wl = 1.31e-6
    for N, dxg, w in ((96, 10e-6, 0.25e-3), (256, 10e-6, 0.8e-3),
                      (512, 10e-6, 2.0e-3)):
        E, _X, _Y = _collimated_gaussian(N, dxg, w, wl)
        assert _wavefront_na(E, dxg, dxg, wl) == 0.0, (
            f'a flat-phase Gaussian of waist {w * 1e3:.2f} mm must have zero '
            f'wavefront NA')


@pytest.mark.parametrize('label, kind, value, expect', [
    ('tilt 0.002 rad', 'tilt', 0.002, 6.000e-03),
    ('tilt 0.01 rad', 'tilt', 0.01, 3.000e-02),
    ('diverging f = -20 mm', 'lens', -20e-3, 8.421e-02),
    ('converging f = +50 mm', 'lens', 50e-3, 3.368e-02),
])
def test_s6_wavefront_na_recovers_a_real_divergence(label, kind, value,
                                                    expect):
    """... and on an input that really is non-flat it recovers the physical
    number, so the gate has not simply been switched off.

    Derivation: for a pure tilt ``t`` every pixel has ``|u| = t``, so the
    3-sigma statistic is ``3 t`` (6.000e-03 / 3.000e-02 measured for
    t = 0.002 / 0.01).  For a lens phase ``exp(-i pi r^2 / (lam f))`` it is
    ``3 sqrt(<r^2>) / |f|`` with ``<r^2>`` the intensity-weighted mean over
    the Gaussian; measured 8.421e-02 / 3.368e-02 at f = -20 / +50 mm, which
    the spectral estimator reproduces to 8.42e-02 / 3.37e-02.  Bar 1 %: the
    estimator's own forward-difference bias is ``dx / (2 f)`` = 2.5e-04 /
    1.0e-04 of the value on the two lens cases.
    """
    from lumenairy.elements.lenses_maslov import _wavefront_na
    wl, N, dxg = 1.31e-6, 256, 10e-6
    E, X, Y = _collimated_gaussian(N, dxg, 0.8e-3, wl)
    if kind == 'tilt':
        E = E * np.exp(1j * 2 * np.pi / wl * value * X)
    else:
        E = E * np.exp(-1j * np.pi / wl * (X ** 2 + Y ** 2) / value)
    got = _wavefront_na(E, dxg, dxg, wl)
    assert abs(got / expect - 1.0) < 0.01, (
        f'{label}: wavefront NA {got:.6e}, expected {expect:.6e}')


def test_s6_saddle_warning_fires_only_on_a_non_flat_input():
    """The end-to-end gate, through ``apply_real_lens_maslov``.

    ``'stationary_phase'`` on a tilted input must warn; on the SAME chart
    with a collimated (flat-phase) input of the same width it must not;
    ``'quadrature'`` -- which integrates the true integrand -- must never;
    and ``collimated_input=True`` must silence it.

    MEASURED: the collimated case's FFT second moment is 3.54e-03, i.e.
    3.5x the 1e-3 threshold, so the pre-VERIFY gate warned here (measured
    "NA = 0.0035" on exactly this call).
    """
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    wl, N, dxg = 1.31e-6, 96, 10e-6
    flat, X, _Y = _collimated_gaussian(N, dxg, 0.25e-3, wl)
    tilted = flat * np.exp(1j * 2 * np.pi / wl * 0.01 * X)
    pres = {'surfaces': [
        {'radius': 12e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -12e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [2.0e-3], 'aperture_diameter': 0.8e-3}

    def saddle_warnings(E, **extra):
        kw = dict(prescription=pres, wavelength=wl, dx=dxg, poly_order=4)
        kw.update(extra)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            apply_real_lens_maslov(E.copy(), **kw)
        return [str(w.message) for w in rec
                if 'saddle of the OPD alone' in str(w.message)]

    # The saddle carries the input's fitted local wavevector (audit S6), so a
    # 0.01 rad tilt is the case that is COMPUTED CORRECTLY and must be SILENT
    # (measured k1 fit residual 3.61e-13 on this fixture, against the 0.5
    # bar).  The warning marks the FALLBACK: an input whose local wavevector
    # the chart cannot represent.  The same tilt carrying 0.6 rad rms phase
    # noise measures 7.50e-01 and is refused, so both arms of the gate are
    # pinned here.
    assert not saddle_warnings(tilted,
                               integration_method='stationary_phase'), (
        'a 0.01 rad tilted input is now expanded about its OWN launch ray '
        'and must NOT warn')
    speckled = tilted * np.exp(
        1j * 0.6 * np.random.default_rng(4).standard_normal(tilted.shape))
    assert saddle_warnings(speckled,
                           integration_method='stationary_phase'), (
        'a speckled input, whose local wavevector an order-4 chart cannot '
        'fit, must still trip the S6 warning -- the fallback keeps the '
        'OPD-only saddle')
    assert not saddle_warnings(flat, integration_method='stationary_phase'), (
        'a COLLIMATED input must not -- its 3.54e-03 angular spectrum is '
        'diffraction, not divergence')
    assert not saddle_warnings(tilted, integration_method='quadrature',
                               n_v2=32), (
        "'quadrature' integrates the true integrand and is unaffected")
    assert not saddle_warnings(tilted, integration_method='stationary_phase',
                               collimated_input=True), (
        'collimated_input=True must silence it')


# ===========================================================================
# S3 -- the exit-vertex leg with an IMMERSED rear surface (n_exit != 1).
# ===========================================================================

def test_s3_exit_vertex_prices_the_sag_leg_at_the_exit_medium_index():
    """S3: ``at_exit_vertex()`` resolves ``n_exit`` from the prescription.
    Every fixture WP-A4 used ends in air, where ``n_exit = 1`` and the
    mistake is invisible.

    Fixture: N-BK7 biconvex whose REAR surface is immersed in N-SF11
    (``n = 1.747969`` at 1.31 um).  The vertex OPD must be
    ``opd_surface + n_exit * t`` with ``t = -z/N``; the ``n_exit = 1``
    mistake is off by ``(n_exit - 1) |sag|`` = 2.413e-06 m = 1.84 waves.
    Bars: exact 0.0 on the correct identity (it is the same arithmetic) and
    > 1e-7 m on the wrong one (3 decades above float64 noise on a ~2 mm OPL).
    """
    from lumenairy import raytrace as rt
    from lumenairy.glass import get_glass_index
    from lumenairy.raytrace import RayBundle

    wl = 1.31e-6
    pres = {'surfaces': [
        {'radius': 30e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -30e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'N-SF11'}],
        'thicknesses': [2.0e-3], 'aperture_diameter': 1.0e-3}
    n = 9
    h = np.linspace(-0.45e-3, 0.45e-3, n)
    z0 = np.zeros(n)
    rays = RayBundle(x=h, y=z0.copy(), z=z0.copy(), L=z0.copy(),
                     M=z0.copy(), N=np.ones(n), opd=z0.copy(),
                     wavelength=wl, alive=np.ones(n, bool))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tr = rt.trace(rays, rt.surfaces_from_prescription(pres), wl)
    ev = tr.at_exit_vertex()
    im = tr.image_rays
    t = -im.z / im.N
    n_exit = float(get_glass_index('N-SF11', wl))
    assert abs(n_exit - 1.747969) < 1e-5, f'premise: n = {n_exit!r}'
    assert np.max(np.abs(im.z)) > 1e-6, 'premise: the rear surface has sag'
    assert np.max(np.abs(ev.opd - (im.opd + n_exit * t))) == 0.0
    wrong = float(np.max(np.abs(ev.opd - (im.opd + 1.0 * t))))
    assert wrong > 1e-7, (
        f'the n_exit = 1 mistake would be {wrong:.3e} m (measured '
        f'2.413e-06 m = 1.84 waves at 1.31 um) -- if this is ~0 the exit '
        f'index is not being resolved from the prescription')
    assert np.max(np.abs(ev.z)) == 0.0, 'the rays must land on z = 0'


# ===========================================================================
# S5 -- the deprecated compensator API.
# ===========================================================================

def test_s5_compensator_api_is_a_warned_no_op_that_still_validates():
    """S5: the three compensating functions must be no-ops, must warn
    through ``_deprecation``, and must keep validating their field argument
    so an existing pipeline fails the same way it used to.  All three
    properties are exact (identity / warning count / raised type), so there
    is no numeric bar to derive.
    """
    from lumenairy.propagators.gbd import (
        asm_field_to_gbd,
        gbd_asm_gouy_phase,
        gbd_field_to_asm,
    )
    E = np.ones((8, 8), dtype=np.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        phase = gbd_asm_gouy_phase(1e-3, 1e-6, 5e-6)
        a = gbd_field_to_asm(E, z=1e-3, wavelength=1e-6, dx=5e-6)
        b = asm_field_to_gbd(E, z=1e-3, wavelength=1e-6, dx=5e-6)
    msgs = [str(w.message).lower() for w in rec]
    assert phase == 0.0
    assert np.array_equal(a, E) and np.array_equal(b, E)
    assert sum('deprecat' in m for m in msgs) == 3, (
        f'all three must warn as deprecated; got {msgs}')
    with pytest.raises(ValueError):
        gbd_field_to_asm(np.ones(4), z=1e-3, wavelength=1e-6, dx=5e-6)


def test_ruling1_merit_warns_when_the_reference_optic_collapses():
    """VERIFY finding about my own fix: the aberration-free reference is a
    REFERENCE SPHERE only while the pupil phase it removes is a
    perturbation.

    On the validation suite's 51.5 mm N-BK7 singlet at
    ``object_distance = 200 mm`` with ``pupil_box_half = 0.02``, the fit
    carries **4.206e+05 waves** of cubic-and-higher pupil phase.  Zeroing
    that does not remove aberration -- it builds a different optic, whose
    focus moves away from ``s2_image``: measured ``|L_ref(0,0)|^2 =
    6.16e-10`` against ``|L(0,0)|^2 = 0.799``, i.e. a coupling of
    **1.30e+09**.  The merit must SAY so rather than report that as a
    Strehl ratio.

    The bars are structural (a warning fires / does not), so there is no
    numeric tolerance to derive; the ``> 10`` trigger is 10x above the
    physical ceiling of 1 and 8 decades below the measured 1.3e+09, and the
    well-posed control below sits at 1.0029.
    """
    import lumenairy as la
    from lumenairy.optimize.core import LGAberrationMerit

    class _Ctx:
        prescription = None
        wavelength = 0.0
        N = 64
        dx = 20e-6

    pres = la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                           aperture=12.0e-3)
    pres['object_distance'] = 200e-3
    ctx = _Ctx()
    ctx.prescription = pres
    ctx.wavelength = 1.31e-6
    merit = LGAberrationMerit(
        targets={(2, 0): 1.0}, field_points=[(0.0, 0.0)],
        w_s=20e-6, w_p=0.02,
        fit_kwargs=dict(source_box_half=20e-6, pupil_box_half=0.02,
                        n_field=6, n_pupil=6, poly_order=4))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        val = float(merit.evaluate(ctx))
    msgs = [str(w.message) for w in rec
            if 'NOT Strehl-normalised' in str(w.message)]
    assert msgs, (
        'a chart whose reference optic collapses must warn; got '
        f'{[str(w.message)[:80] for w in rec]}')
    assert 'waves of cubic-and-higher pupil phase' in msgs[0]
    assert np.isfinite(val), 'and it must still return a usable number'

    # ... and a well-posed chart must NOT warn (measured coupling 1.0029).
    ctx2 = _Ctx()
    ctx2.prescription = _lg_fixture(R1=30e-3, ap=12e-3)
    ctx2.wavelength = 1.30e-6
    ctx2.dx = 10e-6
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter('always')
        v2 = float(LGAberrationMerit(
            targets={(0, 0): 1.0}, field_points=[(0.0, 0.0)],
            w_s=20e-6, w_p=0.05).evaluate(ctx2))
    assert not any('NOT Strehl-normalised' in str(w.message) for w in rec2), (
        'a well-posed chart must not trip the collapse guard')
    assert -0.01 < v2 < 0.01, (
        f'and its deficit must be near zero at best focus (measured '
        f'-2.87e-03); got {v2!r}')


# ===========================================================================
# VERIFY-A4 follow-up -- O-1, O-2, O-6.
# ===========================================================================

def test_o1_local_quadrature_warns_when_its_lattice_leaves_the_chart_box():
    """O-1: the truncation that costs the exactness must be AUDIBLE.

    ``_integrate_local_quadrature`` divides its Gaussian taper back out of a
    quadratic model computed on the FULL lattice, so dropping out-of-box
    samples breaks the cancellation.  Measured on the anamorphic chart
    (``sigma2_norm = 0.2524``), against the closed-form Fresnel value:

        window_sigma  reach in u   relerr      samples dropped
        3.0           0.757        6.60e-15    0 / 64
        2.5 (n = 33)  0.631        3.41e-15    0 / 1089
        5.0 (n = 11)  1.262        8.09e-02    44 / 121   (36.4 %)
        7.5 (n = 13)  1.893        8.79e-02    78 / 169   (46.2 %)

    So the warning must fire on the last two and not on the first two, and
    the message must carry the fraction.  Both bars are structural (a
    warning fires / does not); the 1 % trigger is derived in
    ``_warn_local_window_truncation``.
    """
    from lumenairy.elements.lenses_maslov import (
        _LOCAL_WINDOW_DROP_WARN_FRAC,
    )
    assert _LOCAL_WINDOW_DROP_WARN_FRAC == 0.01

    def _drop_warnings(n, ws):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            ex, lq, _sp = _run_pair(37.0, 0.0, 5.0, 2.5, 0.4, 0.031, 0.019,
                                    n, ws)
        msgs = [str(w.message) for w in rec
                if 'window samples' in str(w.message)]
        return abs(lq - ex) / abs(ex), msgs

    for n, ws in ((8, 3.0), (33, 2.5)):
        rel, msgs = _drop_warnings(n, ws)
        assert rel < 1e-11, f'premise: n={n} ws={ws} must be in-box'
        assert not msgs, (
            f'n={n} ws={ws} keeps the lattice inside the box (relerr '
            f'{rel:.3e}) and must not warn; got {msgs}')

    for n, ws, want_frac in ((11, 5.0, 36.4), (13, 7.5, 46.2)):
        rel, msgs = _drop_warnings(n, ws)
        assert rel > 1e-3, f'premise: n={n} ws={ws} must be out-of-box'
        assert msgs, (
            f'n={n} ws={ws} drops {want_frac:.1f} % of its lattice and is '
            f'{rel:.3e} from the closed form -- it must warn')
        assert 'local_quadrature' in msgs[0] and '%' in msgs[0]
        assert f'{want_frac:.1f} %' in msgs[0], (
            f'the message must carry the measured dropped fraction '
            f'({want_frac:.1f} %); got {msgs[0][:200]!r}')


def test_o2_levin_van_vleck_density_is_the_shared_helper_bit_for_bit():
    """O-2: ``_integrate_levin`` now calls ``_van_vleck_density`` instead of
    writing ``sqrt(|det J_norm|)`` out by hand, at both of its integrand
    closures.

    The composition has to be BIT-IDENTICAL or the change is not free:
    ``_van_vleck_density(d, 1, 1)`` is ``d ** 0.5``, NumPy's ``** 0.5`` is
    ``sqrt`` bit-for-bit, and the box Jacobian ``sqrt(hx * hy)`` is applied
    exactly as before.  Bar: ``np.array_equal`` over 1e5 samples x 3
    anamorphic half-width pairs -- an exact bar, no tolerance.

    (The OTHER association, ``_van_vleck_density(d, hx, hy) * hx * hy``, is
    the same number to 1-2 ULP but not bit-identical -- measured 2.7e-16 /
    3.3e-16 / 3.0e-16 max relative -- which is why the unit-half-width form
    is the one in the code.)
    """
    rng = np.random.default_rng(1)
    d = np.abs(rng.normal(size=100_000)) * 1e3
    assert np.array_equal(d ** 0.5, np.sqrt(d)), (
        'premise: NumPy ** 0.5 must be sqrt bit-for-bit')
    for hx, hy in ((0.02, 0.02), (0.031, 0.019), (0.5, 1e-3)):
        hand = np.sqrt(hx * hy) * np.sqrt(d)
        shared = float(np.sqrt(hx * hy)) * _van_vleck_density(d, 1.0, 1.0)
        assert np.array_equal(hand, shared), (
            f'hx={hx} hy={hy}: the shared helper must reproduce the '
            f'hand-written Levin density bit-for-bit; max rel '
            f'{np.max(np.abs(hand - shared) / hand):.3e}')


def test_o6_exit_vertex_with_a_mirror_last_surface():
    """O-6: the exit-vertex transfer on a system whose LAST surface is a
    MIRROR.  Every WP-A4 and audit fixture ends in a refracting surface.

    The convention ``raytrace/exit_vertex.py`` documents: ``n_exit`` is a
    PHYSICAL (positive) index even for a mirror, and for a mirror it is
    ``glass_before`` -- the reflected ray travels back through the medium it
    arrived in.  The Welford ``n' = -n`` bookkeeping lives in the trace, not
    in the OPL, and the sign comes back through ``N < 0`` in ``t = -z/N``.

    Fixture: an N-BK7 front surface and an internally-reflecting R = -25 mm
    back surface, so the reflected ray runs back through the GLASS.
    MEASURED: ``N = -0.998741`` after reflection, sag range 15.4 um,
    ``n(N-BK7) = 1.503583``.  The vertex OPD equals
    ``opd_surface + n_BK7 * t`` EXACTLY (0.0); the "it must be air" mistake
    would be off by 7.783e-06 m = 5.9 waves at 1.31 um.
    """
    from lumenairy import raytrace as rt
    from lumenairy.glass import get_glass_index
    from lumenairy.raytrace import _make_bundle

    wl = 1.31e-6
    pres = {'surfaces': [
        {'radius': 40e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': -25e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'N-BK7', 'is_mirror': True}],
        'thicknesses': [3.0e-3], 'aperture_diameter': 2.0e-3}
    n = 9
    h = np.linspace(-0.9e-3, 0.9e-3, n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = _make_bundle(x=h, y=np.zeros(n), L=np.zeros(n), M=np.zeros(n),
                         wavelength=wl)
        tr = rt.trace(b, rt.surfaces_from_prescription(pres), wl)
    ev = tr.at_exit_vertex()
    im = tr.image_rays
    assert float(np.mean(im.N)) < -0.9, (
        'premise: the last surface must actually reflect (N < 0); got '
        f'{float(np.mean(im.N)):+.6f}')
    assert np.max(np.abs(im.z)) > 1e-6, 'premise: the mirror has sag'
    t = -im.z / im.N
    n_glass = float(get_glass_index('N-BK7', wl))
    n_air = float(get_glass_index('air', wl))
    assert np.max(np.abs(ev.opd - (im.opd + n_glass * t))) == 0.0, (
        'a mirror exit must be priced at glass_before, not glass_after')
    wrong = float(np.max(np.abs(ev.opd - (im.opd + n_air * t))))
    assert wrong > 1e-7, (
        f'the "exit medium is air" mistake would be {wrong:.3e} m '
        f'(measured 7.783e-06 m = 5.9 waves); if this is ~0 the reflected '
        f'exit index is not being resolved')
    assert np.max(np.abs(ev.z)) == 0.0


_O6_FIELDS = [
    # (source_centre, traced chief ray, |a3|, |a4|) -- all MEASURED.
    pytest.param((100e-6, 0.0), (9.665156e-05, 0.0), 2.74e+03, 2.6e-10,
                 id='x-offset'),
    pytest.param((0.0, 150e-6), (0.0, 1.449788e-04), 8.5e-10, 4.00e+03,
                 id='y-offset'),
    pytest.param((-70e-6, 70e-6), (-6.765608e-05, 6.765608e-05),
                 1.89e+03, 1.89e+03, id='diagonal'),
]


@pytest.mark.parametrize('src, chief, a3_want, a4_want', _O6_FIELDS)
def test_o6_y1_psf_lands_on_the_chief_ray_at_three_field_angles(
        src, chief, a3_want, a4_want):
    """O-6: Y1's property at a SECOND (and third) field angle.

    WP-A4 pinned it at ``source_centre = (100 um, 0)``, where the
    rank-deficient design puts the whole ramp in ``a3`` and leaves ``a4`` at
    2.6e-10 -- so the ``a4 u4`` half of the Y1 fix is never exercised.  A
    y-offset source puts 4.00e+03 waves in ``a4`` and 8.5e-10 in ``a3``, and
    a diagonal one splits 1.89e+03 into each.

    Oracle: an independent ray trace of the chief ray from the SAME
    prescription, launched from the object plane at z = -0.1 m.

    MEASURED (grid pitch 92.3 um):

        source            traced chief              PSF peak            miss
        (100, 0) um       ( 9.665156e-05, 0)        ( 9.785275e-05, 0)  1.20 um
        (0, 150) um       (0,  1.449788e-04)        (0,  1.467808e-04)  1.80 um
        (-70, 70) um      (-6.765608e-05, +same)    (-6.851335e-05, ..) 1.21 um

    Bar 20 um: 11x above the worst measured miss, one fifth of a grid pitch,
    and 35x below the ~700 um the pre-Y1 default flag produced on the
    x-offset fixture.
    """
    from lumenairy import raytrace as rt
    from lumenairy.raytrace import _make_bundle
    import lumenairy as la
    from lumenairy.propagators.asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )

    wl = 1.31e-6

    def _pres():
        p = la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                            aperture=10e-3)
        p['object_distance'] = 0.1
        return p

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = _make_bundle(x=np.array([src[0]]), y=np.array([src[1]]),
                         L=np.array([0.0]), M=np.array([0.0]), wavelength=wl)
        b.z = np.full(1, -0.1)
        r = rt.trace(b, rt.surfaces_from_prescription(_pres()), wl,
                     output_filter='last')
        cx, cy = float(r.image_rays.x[0]), float(r.image_rays.y[0])
        assert math.hypot(cx - chief[0], cy - chief[1]) < 1e-6, (
            f'oracle drift: traced chief ({cx:.6e}, {cy:.6e}) vs the '
            f'recorded {chief}')

        fit = fit_canonical_polynomials(
            _pres(), wl, source_box_half=20e-6, pupil_box_half=0.02,
            n_field=8, n_pupil=8, poly_order=6,
            source_centre=src, extract_linear_phase=True)
        a3 = abs(float(fit.linear_coeffs_phi[3]))
        a4 = abs(float(fit.linear_coeffs_phi[4]))
        assert abs(a3 / a3_want - 1.0) < 0.05 or a3 < 1e-6, (
            f'premise |a3| = {a3:.4e}, recorded {a3_want:.3e}')
        assert abs(a4 / a4_want - 1.0) < 0.05 or a4 < 1e-6, (
            f'premise |a4| = {a4:.4e}, recorded {a4_want:.3e}')
        assert max(a3, a4) > 1e2, (
            'premise: this fit must carry a v2-linear ramp at all')

        half = fit.s2x_halfrange * 0.9
        ax = np.linspace(-half, half, 41) + fit.s2x_centre
        ay = np.linspace(-half, half, 41) + fit.s2y_centre
        X, Y = np.meshgrid(ax, ay, indexing='xy')
        E = np.asarray(propagate_modal_asymptotic(
            fit, source_point=src, w_s=20e-6, w_p=0.02,
            v2_centre=(fit.v2x_centre, fit.v2y_centre),
            s2_grid_x=X, s2_grid_y=Y))
    k = np.unravel_index(np.argmax(np.abs(E)), E.shape)
    miss = math.hypot(float(X[k]) - cx, float(Y[k]) - cy)
    assert miss < 20e-6, (
        f'src={src}: PSF at ({float(X[k]):.6e}, {float(Y[k]):.6e}) vs the '
        f'traced chief ray at ({cx:.6e}, {cy:.6e}) -- miss '
        f'{miss * 1e6:.2f} um (measured 1.20 / 1.80 / 1.21 um on the three '
        f'field angles; pre-Y1 the x-offset case was ~700 um off)')
