"""INDEPENDENT VERIFICATION of Phase A of the curved-cell map for the pure
staggered 2-D PMM -- decision tests for the gaps the verifier's mutation
matrix found in ``tests/unit/test_pmm2d_staggered_curved_a.py``
(``docs/audits/VERIFY_PMM2D_CURVED_A_2026_10_02.md`` section 9).

Words.  A coordinate map ``(x, y) = Phi(u, v)`` distorts the solver's
rectangular ``(u, v)`` wall grid in the physical plane; ``J = d(x, y)/d(u, v)``
is its Jacobian, ``g = J^T J`` the metric and ``sqrt(g) = det J``.  The build
gated only the identity and SEPARABLE stretches, on which the metric has no
shear (``g12 = 0``): the mixed effective-tensor entries
(``eps'_12 = -eps g12 / sqrt g``, ``chi_12 = g12 / sqrt g``) and the two
off-diagonal blocks of the cofactor far-field projector are identically zero
there, so no Phase-A test could see their sign, their placement or their
omission.  A SHEARED map (``g12 != 0``) with straight walls is used here:
``x = u + cx sin(k u) F(v)``, ``y = v + cy sin(k v) G(u)`` on walls at
``0, p/2, p`` (``sin(k u)`` vanishes on every wall, so the walls stay where
they are while every cell interior is sheared).

Fixture: lambda 1.3, square period 1.0, air over n = 1.52, depth 0.4, eps
6.25 (the verifier's, not the build's).  EVERY BAR below carries the
measurement it was derived from (2026-10-02; Windows CPython 3.14.6 / numpy
2.4.4 / scipy 1.17.1 and WSL CPython 3.12.3 / numpy 2.4.6 / scipy 1.17.1, BLAS
pinned) and the engineered mutant it must catch
(``validation/probe_pmm2d_curved/verify_a/v9_readings_*.json``).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm._curvemap import (  # noqa: E402
    CellMap,
    IdentityMap,
    SeparableStretch,
    SineStretch,
)

_P = 1.0
_WL = 1.3
_K0 = 2 * np.pi / _WL
_NSUP, _NSUB = 1.0, 1.52
_DEPTH = 0.4
_EPS = 6.25 + 0j
_XW = np.array([0.0, 0.20, 0.65, _P])
_YW = np.array([0.0, 0.25, 0.70, _P])
_K = 2 * np.pi / _P


class _Shear(CellMap):
    """The sheared map (module docstring); amplitudes in units of the
    period.  det J lies in [0.46, 1.64] at (0.06, 0.05)."""

    def __init__(self, cx=0.06, cy=0.05):
        self.cx, self.cy = cx * _P, cy * _P
        self._init_walls(np.array([0.0, 0.5, 1.0]) * _P,
                         np.array([0.0, 0.5, 1.0]) * _P, _P, _P)
        self.validate()

    def geom(self, sx, sy, U, V):
        U = np.asarray(U, float)[:, None]
        V = np.asarray(V, float)[None, :]
        F = np.sin(_K * V) + 0.4 * np.cos(2 * _K * V)
        Fp = _K * np.cos(_K * V) - 0.8 * _K * np.sin(2 * _K * V)
        G = np.cos(_K * U) + 0.3 * np.sin(2 * _K * U)
        Gp = -_K * np.sin(_K * U) + 0.6 * _K * np.cos(2 * _K * U)
        out = (U + self.cx * np.sin(_K * U) * F,
               V + self.cy * np.sin(_K * V) * G,
               1 + self.cx * _K * np.cos(_K * U) * F,
               self.cx * np.sin(_K * U) * Fp,
               self.cy * np.sin(_K * V) * Gp,
               1 + self.cy * _K * np.cos(_K * V) * G)
        shp = (U.size, V.size)
        return tuple(np.broadcast_to(t, shp).astype(float) for t in out)

    def _key(self):
        return ("_Shear", self.cx, self.cy)


class _Harmonic:
    """Two-harmonic stretch with a phase (no mirror symmetry), duck-typing
    SineStretch: f(u) = u + a1 sin(k u) + a2 [sin(2 k u + phi) - sin phi],
    a1 / a2 in units of the period."""

    def __init__(self, a1=0.10, a2=0.04, phi=0.9):
        self.a1, self.a2, self.phi = a1, a2, phi

    def __call__(self, u, period):
        k = 2 * np.pi / period
        u = np.asarray(u, float)
        a1, a2 = self.a1 * period, self.a2 * period
        return (u + a1 * np.sin(k * u)
                + a2 * (np.sin(2 * k * u + self.phi) - np.sin(self.phi)),
                1 + a1 * k * np.cos(k * u)
                + 2 * a2 * k * np.cos(2 * k * u + self.phi))

    def check(self, period, axis):
        assert np.min(self(np.linspace(0, period, 20001), period)[1]) > 0

    def inverse(self, x, period):
        x = np.asarray(x, float)
        u = x.copy()
        for _ in range(200):
            f, fp = self(u, period)
            du = (f - x) / fp
            u = u - du
            if np.max(np.abs(du), initial=0.0) <= 1e-16 * period:
                break
        return u

    def key(self):
        return ("_Harmonic", self.a1, self.a2, self.phi)


def _airy_normal(eps=_EPS):
    n2 = np.sqrt(eps)
    r12 = (_NSUP - n2) / (_NSUP + n2)
    r23 = (n2 - _NSUB) / (n2 + _NSUB)
    t12 = 2 * _NSUP / (_NSUP + n2)
    t23 = 2 * n2 / (n2 + _NSUB)
    ph = np.exp(1j * n2 * _K0 * _DEPTH)
    den = 1 + r12 * r23 * ph ** 2
    return (abs((r12 + r23 * ph ** 2) / den) ** 2,
            abs(t12 * t23 * ph / den) ** 2 * _NSUB / _NSUP)


# =========================================================================== #
# GAP 1 -- the shear terms: eps'_12 / chi_12 sign and placement, and the
# off-diagonal cofactor blocks P12 / P21 (mutants m01b, m04, m04b, m06)
# =========================================================================== #
def test_verify_a_sheared_film_is_exact():
    """A uniform film under the SHEARED map must reproduce the exact Airy
    slab at normal incidence: every order other than (0, 0) carries nothing
    and (0, 0) carries the slab's R / T.  M = 7 on the 2 x 2 (u, v) grid.

    Measured 2026-10-02 (``v9_readings_tip_{win,wsl}.json``): 2.64e-08
    (Windows) / 2.66e-08 (WSL) -- the error is the field's representation on
    the sheared grid and falls spectrally (1.2e-3, 6.8e-6, 7.6e-7, 2.6e-8,
    4.2e-9 at M = 4..8, ``v3_film_shear_win.json``).  Bar 1e-6: 1.6 decades
    above the reading.  The engineered mutants read (M = 7,
    ``v9_readings_m*_win.json``): off-diagonal projector blocks dropped
    5.1e-04 (2.7 decades above the bar), the cofactor's off-diagonal entries
    swapped 0.114, the cofactor replaced by J^T 0.103, the sign of
    eps'_12 flipped 0.543, the sign of chi_12 flipped 0.364 -- none of which
    any test of the build can see (every map it gates has g12 = 0)."""
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=7, n_orders=3, cmap=_Shear())
    st.add_layer(_DEPTH, eps_cell=np.full((2, 2), _EPS))
    st.set_source(_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _J = st.solve()
    Rex, Tex = _airy_normal()
    i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rex
    T[:, i0] -= Tex
    err = float(max(np.abs(R).max(), np.abs(T).max()))
    assert err <= 1e-6, err


# =========================================================================== #
# GAP 2 -- the assembled operators against an INDEPENDENT oracle (mutants
# m03 = the adaptive node count pinned to 2M + 8 at the CALL SITE, m04c =
# chi33 inverted; and any slip in the 18-block formulas)
# =========================================================================== #
def _w1d(basis, lset, op, rset, w, eps_seg, nq):
    """INT conj(L_i)^(a) w(u) R_j^(b) du by an nq-node Gauss rule per
    segment, written from the basis definition (op: 'm' none, 'd' on R,
    'dL' on L)."""
    xg, wg = leggauss(nq)
    V, Vp = TS._modleg_value_deriv(basis.M, xg)
    Ls = np.asarray(getattr(basis, lset))
    Rs = np.asarray(getattr(basis, rset))
    out = np.zeros((Ls.shape[0], Rs.shape[0]), complex)
    for s in range(basis.N):
        u = 0.5 * (basis.xb[s] + basis.xb[s + 1]) + basis.Jn[s] * xg
        fa, gc, sc = ((V, V, basis.Jn[s]) if op == "m" else
                      (V, Vp, 1.0) if op == "d" else (Vp, V, 1.0))
        out += ((np.conj(Ls[:, s, :]) @ fa) * (w(u) * eps_seg[s] * wg * sc)
                ) @ (Rs[:, s, :] @ gc).T
    return out


def _oracle_R_L(bx, by, f, eps_x, nq=1000):
    """R and L of a y-uniform cell under x = f(u), y = v, from 1-D factors:
    sqrt g = f', chi11 = f', chi22 = chi33 = 1 / f', e11 = eps / f',
    e22 = e33 = eps f' (the module's own block formulas, re-assembled)."""
    per = bx.d
    fp = (lambda u: f(u, per)[1])                       # noqa: E731
    ifp = (lambda u: 1.0 / f(u, per)[1])                # noqa: E731
    one = np.ones_like
    o_x, o_y = np.ones(bx.N), np.ones(by.N)
    X = (lambda l, op, r, w, e: _w1d(bx, l, op, r, w, e, nq))  # noqa: E731
    Y = (lambda l, op, r: _w1d(by, l, op, r, one, o_y, nq))    # noqa: E731
    K = np.kron
    B, T = "B", "Btilde"
    k0 = _K0
    R11 = -K(Y(T, "m", T), X(B, "m", B, ifp, o_x))
    R22 = -K(Y(B, "m", B), X(T, "m", T, fp, o_x))
    Mbb_x, Mbb_y = X(B, "m", B, one, o_x), Y(B, "m", B)
    Curl = np.concatenate([K(Y(B, "d", T) / k0, Mbb_x),
                           -K(Mbb_y, X(B, "d", T, one, o_x) / k0)], axis=1)
    Gwi = np.linalg.inv(K(Mbb_y, Mbb_x))
    Stt = -Curl.conj().T @ (Gwi @ K(Mbb_y, X(B, "m", B, ifp, o_x)) @ Gwi
                            ) @ Curl
    Ktz = np.concatenate([-K(Y(T, "m", T), X(B, "d", T, ifp, o_x)) / k0,
                          -K(Y(B, "d", T), X(T, "m", T, fp, o_x)) / k0],
                         axis=0)
    Kzt = np.concatenate([-K(Y(T, "m", T), X(T, "dL", B, ifp, eps_x)) / k0,
                          -K(Y(T, "dL", B), X(T, "m", T, fp, eps_x)) / k0],
                         axis=1)
    Schur = Ktz @ np.linalg.solve(K(Y(T, "m", T), X(T, "m", T, fp, eps_x)),
                                  Kzt)
    qq = R11.shape[0]
    Rm = np.zeros((2 * qq, 2 * qq), complex)
    Rm[:qq, :qq], Rm[qq:, qq:] = R11, R22
    Lm = np.zeros_like(Rm)
    Lm[:qq, :qq] = K(Y(T, "m", T), X(B, "m", B, ifp, eps_x))
    Lm[qq:, qq:] = K(Y(B, "m", B), X(T, "m", T, fp, eps_x))
    return Rm, Lm + Stt - Schur


def test_verify_a_mapped_operators_match_an_independent_oracle():
    """For a SEPARABLE map and a y-uniform cell every one of the 18 weighted
    blocks factors into 1-D weighted masses / derivatives; this test builds
    them from the basis definition with a 1000-node rule per segment and
    re-assembles R and L, then compares the SOLVER'S operators (built through
    its own call to the adaptive node count) on the asymmetric two-harmonic
    stretch (min f' = 0.154).  M = 4.

    Measured 2026-10-02 (``v9_readings_tip_*.json``): max relative error of
    R / L 4.26e-13 (both builds); the oracle's own self-gap (2000 vs 3000
    nodes) 6.1e-14 (``v2_sep_oracle_win.json``).  Bar 1e-10: 2.4 decades
    above.  Mutants (``v9_readings_m*_win.json``): the node count pinned to
    2M + 8 at the solver's call site 3.26e-03 (7.5 decades above the bar;
    the build's own adaptive test calls the node function DIRECTLY and passes
    on this mutant, as do the other twelve), chi33 = sqrt g instead of
    1 / sqrt g 0.77."""
    f = _Harmonic()
    cm = SeparableStretch.from_physical_walls(_XW, _YW, fx=f)
    eps = np.ones((3, 3), complex)
    eps[1, :] = _EPS
    M = 4
    s = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, M, eps,
                               k0=_K0, cmap=cm)
    Rr, Lr = _oracle_R_L(s.bx, s.by, f, eps[:, 0])
    for got, ref in ((s.Rmat, Rr), (s.Lmat, Lr)):
        rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        assert rel <= 1e-10, rel


# =========================================================================== #
# GAP 3 -- the map protocol's refusals (mutants m05, m07, m08)
# =========================================================================== #
def test_verify_a_fingerprint_sees_the_curve_parameters():
    """The fingerprint is documented as 'equal exactly when two maps describe
    the same geometry'.  The build's check compares two stretches whose WALLS
    differ too, so it passes with the curve parameters dropped from the hash
    (mutant m05).  Same walls, different amplitude must differ; equal content
    must agree; a different map class on the same walls must differ."""
    a = SeparableStretch(3, 3, fx=SineStretch(0.05), period_x=_P, period_y=_P)
    b = SeparableStretch(3, 3, fx=SineStretch(0.06), period_x=_P, period_y=_P)
    c = SeparableStretch(3, 3, fx=SineStretch(0.05), period_x=_P, period_y=_P)
    assert np.array_equal(a.u_bounds, b.u_bounds)
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == c.fingerprint
    assert (IdentityMap(3, 3, _P, _P).fingerprint
            != SeparableStretch(3, 3, period_x=_P, period_y=_P).fingerprint)


class _Fold(CellMap):
    """x = u + c sin(k u): folds the plane (f' < 0 somewhere) when c k > 1.
    Deliberately does NOT run validate(), so the SOLVER'S own per-node
    det J check is the only line of defence."""

    def __init__(self, c):
        self.c = c * _P
        self._init_walls(2, 2, _P, _P)

    def geom(self, sx, sy, U, V):
        U = np.asarray(U, float)[:, None]
        V = np.asarray(V, float)[None, :]
        shp = (U.size, V.size)
        X = U + self.c * np.sin(_K * U)
        xu = 1 + self.c * _K * np.cos(_K * U)
        return tuple(np.broadcast_to(t, shp).astype(float) for t in
                     (X, V, xu, 0 * xu, 0 * xu, 1 + 0 * xu))

    def _key(self):
        return ("_Fold", self.c)


def test_verify_a_solver_refuses_a_folding_map():
    """The solver re-checks det J > 0 at every quadrature node (documented;
    mutant m07 removes it and all 13 build tests still pass).  A map that
    folds (c k = 1.26) must raise; the same map below the fold (c k = 0.63)
    must build."""
    eps = np.full((2, 2), 2.0 + 0j)
    TS.Granet2DTransverseE(_P, _P, 2, 2, 4, eps, k0=_K0, cmap=_Fold(0.1))
    with warnings.catch_warnings():
        # the adaptive node count runs BEFORE the det J check and warns that
        # the folding weights do not converge (verify doc, defect D3)
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="det J <= 0"):
            TS.Granet2DTransverseE(_P, _P, 2, 2, 4, eps, k0=_K0,
                                   cmap=_Fold(0.2))


class _Translate(CellMap):
    """x = u + c: lattice-periodic with a periodic Jacobian (J = I), but the
    side u = 0 lands on x = c -- the map frame and the lab no longer share an
    origin (every Jones phase would move)."""

    def __init__(self, c):
        self.c = c * _P
        self._init_walls(2, 2, _P, _P)
        self.validate()

    def geom(self, sx, sy, U, V):
        U = np.asarray(U, float)[:, None]
        V = np.asarray(V, float)[None, :]
        shp = (U.size, V.size)
        one = np.ones(shp)
        return (np.broadcast_to(U + self.c, shp).astype(float),
                np.broadcast_to(V, shp).astype(float), one, 0 * one,
                0 * one, one)

    def _key(self):
        return ("_Translate", self.c)


def test_verify_a_validation_refuses_a_translated_map():
    """``CellMap.validate`` requires the cell boundary to map onto itself.
    The build's Skew map fails PERIODICITY as well, so removing the boundary
    check (mutant m08) leaves every build test green; a pure translation is
    periodic and must still be refused.  Two-sided: no translation passes."""
    _Translate(0.0)
    with pytest.raises(ValueError, match="onto the line"):
        _Translate(0.1)


# =========================================================================== #
# GAP 4 -- the far projector's per-cell PHASE rule (mutant m09: the mapped
# projector run at a fixed 2M + 16 nodes)
# =========================================================================== #
def test_verify_a_mapped_projector_keeps_the_phase_rule_on_a_long_cell():
    """The mapped far projector sizes its rule by the shipped per-segment
    phase rule on the cell's PHYSICAL extent.  The build's A2 check uses
    cells of <= 0.6 p and orders to +-3, where the fixed floor 2M + 16 is
    already enough, so dropping the phase rule passes every build test.  On
    a cell spanning 0.96 p with orders to +-10 at an oblique Bloch phase the
    identity-map projector must still equal the shipped separable one.

    Measured 2026-10-02 (``_va_m09b.py``, M = 4): 1.50e-15 relative on both
    builds; the fixed-2M+16 mutant 2.96e-05.  Bar 1e-11: 4 decades above the
    reading, 6.5 below the mutant."""
    w = np.array([0.0, 0.04, _P])
    M = 4
    bx = TS.Basis1D(_P, w, M, np.exp(-1j * 0.9 * _K0 * _P))
    by = TS.Basis1D(_P, w, M, np.exp(-1j * 0.3 * _K0 * _P))
    ox = np.arange(-10, 11)
    Pa = TS._far_projector_2d(bx, by, ox, ox, 0.9 * _K0, 0.3 * _K0)
    Pb = TS._far_projector_2d(bx, by, ox, ox, 0.9 * _K0, 0.3 * _K0,
                              cmap=IdentityMap(w, w))
    assert Pb[2] is None and Pb[3] is None
    for i in (0, 1):
        rel = float(np.abs(Pb[i] - Pa[i]).max() / np.abs(Pa[i]).max())
        assert rel <= 1e-11, (i, rel)
