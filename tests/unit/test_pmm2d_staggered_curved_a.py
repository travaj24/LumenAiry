"""CURVED-CELL MAP, Phase A, for the PURE staggered 2-D PMM: the
variable-coefficient (2-D Gauss quadrature) assembly, the map protocol, the
three traps of the plan, and the cofactor far field -- gated on a separable
per-axis stretch.  Gates A1-A7 of
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.1, as built
and measured in ``docs/audits/BUILD_PMM2D_CURVED_A_2026_10_02.md``.

Words.  A coordinate map ``(x, y) = Phi(u, v)`` bends (here: stretches) the
solver's rectangular ``(u, v)`` wall grid in the physical plane; the solver
then works on the covariant field components with the effective tensors
``eps'_t = eps sqrt(g) g^-1``, ``eps'_33 = eps sqrt(g)``, ``chi_t = g /
sqrt(g)``, ``chi33 = 1 / sqrt(g)`` (``J = d(x, y)/d(u, v)``, ``g = J^T J``,
``sqrt(g) = det J``).  A FAIL-BEFORE arm is a deliberately broken variant
that must fail the bar, proving the test can see the defect it guards.

Fixture (the planning probes' P2 fixture): lambda = 1, square period 1.2,
depth 0.5, air over n = 1.45, eps 4 features, normal incidence.  ``stripe`` =
y-uniform ridge x in [0.25, 0.85] (exact oracle ``pmm_efficiency_1d``,
degree 40, self-gap 4.1e-07 measured 2026-10-02); ``film`` = uniform eps 4
(exact oracle: the Airy slab).  The stretch is ``x = u + a sin(2 pi u / p)``
with the (u, v) walls at the PREIMAGES of the physical walls (the physical
device is unchanged, so the mapped solve must converge to the unmapped
answer).

EVERY BAR is derived from a measurement made by this build on 2026-10-02
(Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, OMP/OPENBLAS/MKL = 1;
probe JSON under ``validation/probe_pmm2d_curved/build_a/``), stated in the
assertion's comment with its gap on both sides.  Every compared quantity is a
deterministic DISCRETISATION number (or a round-off residual whose bar sits
decades above round-off), never a per-build reading.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import hashlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm._curvemap import (  # noqa: E402
    CellMap,
    IdentityMap,
    SeparableStretch,
    SineStretch,
)

_P = 1.2
_WL = 1.0
_DEPTH = 0.5
_NSUP, _NSUB = 1.0, 1.45
_K0 = 2 * np.pi / _WL
_XW = np.array([0.0, 0.25, 0.85, _P])
_YW = np.array([0.0, 0.4, 0.8, _P])
_YWP = np.array([0.0, 0.30, 0.90, _P])
_ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
         (-1, -1)]


def _cell(kind, eps=4.0 + 0j):
    c = np.ones((3, 3), complex)
    if kind == "stripe":
        c[1, :] = eps
    elif kind == "pillar":
        c[1, 1] = eps
    else:
        c[:] = eps
    return c


def _stretch(a_frac, yw=_YW):
    return SeparableStretch.from_physical_walls(_XW, yw,
                                                fx=SineStretch(a_frac * _P))


def _solve(cmap, cell, M, retain=False, n_orders=3):
    st = PMM2DStackPure(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    st.add_layer(_DEPTH, eps_cell=cell)
    st.set_source(_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # fail-before arms trip closure
        o, R, T, J = st.solve(retain_internal=retain)
    return o, R, T, J, st


def _vec(o, R, T, rows=(0, 1)):
    idx = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
           for m, n in _ORD9]
    return np.concatenate([R[list(rows)][:, idx].ravel(),
                           T[list(rows)][:, idx].ravel()])


def _h(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _NoCofactorView:
    """FAIL-BEFORE view of a map for the far projector only: the kernel
    positions kept, the cofactor det J * J^-T replaced by the identity (the
    covariant coefficients read as if Cartesian; planning probe P2b
    ``trap_F_no_cofactor``)."""

    def __init__(self, cmap):
        self._m = cmap

    def geom(self, sx, sy, U, V):
        X, Y, xu, xv, yu, yv = self._m.geom(sx, sy, U, V)
        return (X, Y, np.ones_like(xu), np.zeros_like(xv), np.zeros_like(yu),
                np.ones_like(yv))


def _patch_no_cofactor(monkeypatch):
    orig = SP._far_projector_2d

    def patched(bx, by, ox, oy, a0x=0.0, a0y=0.0, cmap=None):
        if cmap is not None:
            cmap = _NoCofactorView(cmap)
        return orig(bx, by, ox, oy, a0x, a0y, cmap=cmap)
    monkeypatch.setattr(SP, "_far_projector_2d", patched)


@functools.lru_cache(maxsize=None)
def _stripe_m8():
    """The A4 / A6 correct arm: stripe, a = 0.05 p, M = 8 (one solve, shared
    by both gates; ~20 s)."""
    return _solve(_stretch(0.05), _cell("stripe"), 8)


@functools.lru_cache(maxsize=None)
def _oracle_1d():
    from lumenairy.elements.pmm.oned import pmm_efficiency_1d
    ref = {}
    for row, pol in ((0, "tm"), (1, "te")):
        o, R, T = pmm_efficiency_1d(_P, 2.0, 1.0, _NSUB, _NSUP, _DEPTH,
                                    0.6 / _P, _WL, polarization=pol,
                                    degree=40, far_field_orders=7)
        d = {int(m): (float(R[i]), float(T[i]))
             for i, m in enumerate(np.asarray(o))}
        Rv = [d.get(m, (0, 0))[0] if n == 0 else 0.0 for m, n in _ORD9]
        Tv = [d.get(m, (0, 0))[1] if n == 0 else 0.0 for m, n in _ORD9]
        ref[row] = np.array(Rv + Tv)
    return ref


# =========================================================================== #
# A1 -- no map = today's bytes
# =========================================================================== #
def test_a1_no_map_never_reaches_the_mapped_code(monkeypatch):
    """``cmap=None`` is a DISPATCH to the shipped Kronecker code, not the
    identity map: every mapped-only function is booby-trapped and the full
    family of unmapped calls (scalar, tensor, magnetic, a multilayer stack
    with absorption, the per-layer mortar, the single-polarization entry) must
    not touch one.  The byte identity against the parent commit itself is a
    build-doc measurement (``a1_compare.json``: 109 / 109 SHA-256 identical
    on operators, modes, R / T / Jones and absorption) because the parent tree
    is not importable here; this is its build-free restatement -- the code
    that could change the bytes is provably not executed."""
    def boom(*a, **k):
        raise AssertionError("mapped code reached on the no-map path")
    for name in ("_stag_quad_weighted", "_stag_map_weights",
                 "_stag_map_nodes", "_far_projector_mapped",
                 "_stag_quad_axis_factor"):
        monkeypatch.setattr(TS, name, boom)
    pil = np.array([[4.0, 1.0], [1.0, 1.0]], complex)
    TS.pmm_efficiency_2d_staggered(_P, _P, pil, 1.45, 1.0, 0.3, _WL,
                                   degree=4, n_orders=2)
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    lc = np.empty((2, 2, 3, 3), complex)
    lc[:] = 2.0 * np.eye(3)
    lc[0, 0] = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
    TS.pmm_jones_2d_staggered(_P, _P, lc, 1.45, 1.0, 0.3, _WL, degree=4,
                              n_orders=2, theta=0.2, phi=0.3)
    TS.pmm_jones_2d_staggered(_P, _P, pil, 1.45, 1.0, 0.3, _WL, degree=4,
                              n_orders=2, mu_cell=np.full((2, 2), 1.3 + 0j))
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=2)
    st.add_layer(0.2, eps=2.1)
    st.add_layer(0.3, eps_cell=np.array([[4.0 + 0.3j, 1.0], [1.0, 1.0]]))
    st.set_source(_WL)
    st.solve(retain_internal=True)
    st.layer_absorption()
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=4, n_orders=2, layer_grids="per-layer")
    st.add_layer(0.3, eps_cell=pil, x_walls=[0.5], y_walls=[0.45])
    st.set_source(_WL)
    st.solve()


def test_a1_fail_before_the_hash_sees_the_identity_map():
    """The hash used for A1 must be able to SEE a round-off change: the
    identity map through the quadrature path (the same operators to 1e-14,
    gate A2) is NOT byte-identical to the no-map path.  Measured 2026-10-02
    (``a1_compare.json``): 28 of 36 operator hashes differ, max |dLmat|
    9.7e-15.  The decision asserted is 'at least one of the five assembled
    operators differs in its bytes' -- with thousands of entries reassociated
    by the quadrature, a bit-identical outcome is not a legal reading on any
    build."""
    eps = _cell("pillar")
    s0 = TS.Granet2DTransverseE(_P, _P, 3, 3, 5, eps, k0=_K0)
    s1 = TS.Granet2DTransverseE(_P, _P, 3, 3, 5, eps, k0=_K0,
                                cmap=IdentityMap(3, 3, _P, _P))
    differs = [_h(getattr(s0, n)) != _h(getattr(s1, n))
               for n in ("Rmat", "Lmat", "Stt", "Schur")]
    differs.append(_h(*s0.Et_blocks) != _h(*s1.Et_blocks))
    assert any(differs)


# =========================================================================== #
# A2 -- the identity map through the quadrature path
# =========================================================================== #
_B, _T = "B", "Btilde"
_BLOCKS = [  # the 18 weighted blocks of _assemble: (x-spec, y-spec)
    ((_B, "m", _B), (_T, "m", _T)), ((_B, "m", _T), (_T, "m", _B)),
    ((_T, "m", _B), (_B, "m", _T)), ((_T, "m", _T), (_B, "m", _B)),
    ((_B, "m", _B), (_T, "m", _T)), ((_T, "m", _T), (_B, "m", _B)),
    ((_B, "m", _T), (_T, "m", _B)), ((_T, "m", _B), (_B, "m", _T)),
    ((_B, "m", _B), (_B, "m", _B)), ((_T, "m", _T), (_T, "m", _T)),
    ((_B, "d", _T), (_T, "m", _T)), ((_B, "m", _T), (_T, "d", _T)),
    ((_T, "m", _T), (_B, "d", _T)), ((_T, "d", _T), (_B, "m", _T)),
    ((_T, "dL", _B), (_T, "m", _T)), ((_T, "m", _B), (_T, "dL", _T)),
    ((_T, "m", _T), (_T, "dL", _B)), ((_T, "dL", _T), (_T, "m", _B)),
]


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_a2_identity_map_reproduces_the_kron_operators():
    """Bar 1e-11 relative (plan: 2.3 decades above the planning P1
    measurement).  Re-measured 2026-10-02 (``a2_identity.json``) on a 3 x 3
    pillar on NON-UNIFORM walls: every one of the 18 block kernels <= 1.6e-14,
    the assembled operators (R, L, S_tt, Schur, [eps_t], plain Gram vs -R)
    <= 5.5e-14, the far projector <= 4.5e-15 -- 2.3 decades under the bar --
    while a 1e-6 p stretch moves L by 7.8e-6 .. 1.0e-5 (asserted below,
    5.9 decades above it).  M = 5 here (the JSON also carries M = 7)."""
    M = 5
    eps = _cell("pillar")
    s0 = TS.Granet2DTransverseE(_P, _P, _XW, _YWP, M, eps, k0=_K0)
    cm = IdentityMap(_XW, _YWP)
    s1 = TS.Granet2DTransverseE(_P, _P, _XW, _YWP, M, eps, k0=_K0, cmap=cm)
    # the 18 block KERNELS, quadrature vs kron, one per flavour / set pairing
    rule = TS._stag_map_quad_rule(M)
    Wn = np.broadcast_to(eps[:, :, None, None], eps.shape + (rule[0].size,) * 2)
    bx, by = s0.bx, s0.by
    for xs, ys in _BLOCKS:
        Q = TS._stag_quad_weighted(bx, by, xs, ys, Wn, rule)
        if xs[1] == "m" and ys[1] == "m":
            K = s0._eps_weighted(
                (bx, bx.m_ref, getattr(bx, xs[0]), getattr(bx, xs[2])),
                (by, by.m_ref, getattr(by, ys[0]), getattr(by, ys[2])), eps)
        else:
            K = s0._eps_dir(bx, *xs, by, *ys, wmap=eps)
        assert _rel(Q, K) <= 1e-11, (xs, ys, _rel(Q, K))
    # the ASSEMBLED operators
    for name in ("Rmat", "Lmat", "Stt", "Schur"):
        assert _rel(getattr(s1, name), getattr(s0, name)) <= 1e-11, name
    for k in range(2):
        assert _rel(s1.Et_blocks[k], s0.Et_blocks[k]) <= 1e-11
    # identity: the shear blocks are exactly absent (zero weight, no quadrature)
    assert not np.any(s1.Et_offdiag[0]) and not np.any(s1.Et_offdiag[1])
    qq = s1.q * s1.q
    G = np.zeros_like(s1.Rmat)
    G[:qq, :qq], G[qq:, qq:] = s1.Ggram_blocks
    assert _rel(G, -s0.Rmat) <= 1e-11
    # the far projector (oblique, so the Bloch kernel is exercised)
    ox = np.arange(-3, 4)
    b2x = TS.Basis1D(_P, _XW, M, np.exp(-1j * 0.9 * _P))
    b2y = TS.Basis1D(_P, _YWP, M, np.exp(-1j * 0.4 * _P))
    Pa = TS._far_projector_2d(b2x, b2y, ox, ox, 0.9, 0.4)
    Pb = TS._far_projector_2d(b2x, b2y, ox, ox, 0.9, 0.4, cmap=cm)
    assert Pb[2] is None and Pb[3] is None
    assert _rel(Pb[0], Pa[0]) <= 1e-11 and _rel(Pb[1], Pa[1]) <= 1e-11
    # UPPER GAP: the bar can see a 1e-6 p map (measured 7.8e-6 relative on L)
    s2 = TS.Granet2DTransverseE(
        _P, _P, _XW, _YWP, M, eps, k0=_K0,
        cmap=SeparableStretch(_XW, _YWP, fx=SineStretch(1e-6 * _P)))
    assert _rel(s2.Lmat, s1.Lmat) > 1e-9


# =========================================================================== #
# A3 -- a uniform film under a stretch is exact
# =========================================================================== #
def _film_err(M, monkeypatch=None):
    from_ = SeparableStretch(3, 3, fx=SineStretch(0.15 * _P), period_x=_P,
                             period_y=_P)
    o, R, T, _J, _st = _solve(from_, _cell("film"), M)
    k0 = _K0
    n2 = 2.0
    r12 = (_NSUP - n2) / (_NSUP + n2)
    r23 = (n2 - _NSUB) / (n2 + _NSUB)
    t12 = 2 * _NSUP / (_NSUP + n2)
    t23 = 2 * n2 / (n2 + _NSUB)
    ph = np.exp(1j * n2 * k0 * _DEPTH)
    Rex = abs((r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)) ** 2
    Tex = abs(t12 * t23 * ph / (1 + r12 * r23 * ph ** 2)) ** 2 * _NSUB / _NSUP
    i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    R = R.copy()
    T = T.copy()
    R[:, i0] -= Rex
    T[:, i0] -= Tex
    return [float(max(np.abs(R[r]).max(), np.abs(T[r]).max())) for r in (0, 1)]


def test_a3_film_under_stretch_is_exact_and_spectral(monkeypatch):
    """a = 0.15 p (the 33:1 stretch), uniform 3 x 3 (u, v) lattice.
    Measured 2026-10-02 (``a3_film.json``), TM (incident E_x) / TE:
    M = 4: 4.0e-07 / 3.9e-15, M = 5: 5.3e-09 / 2.9e-15, M = 6: 1.29e-12 /
    1.8e-14 -- the planning numbers to every printed digit.  Bars (plan):
    1e-10 at M = 6 (1.9 decades above 1.29e-12) and a TM decay M = 4 -> 6 of
    at least 4 decades (measured 5.5).  FAIL-BEFORE: the far field without
    the cofactor reads 0.96 at M = 6 -- 9.0 decades above the bar."""
    e4 = _film_err(4)
    e6 = _film_err(6)
    assert max(e6) <= 1e-10, e6
    assert np.log10(e4[0] / max(e6[0], 1e-300)) >= 4.0, (e4, e6)
    _patch_no_cofactor(monkeypatch)
    bad = _film_err(6)
    assert max(bad) > 1e-2, bad


# =========================================================================== #
# A4 -- stretch self-consistency on the TE stripe
# =========================================================================== #
def test_a4_te_stripe_under_stretch_matches_the_1d_oracle():
    """Stripe, a = 0.05 p, M = 8, TE (incident E_y) against
    ``pmm_efficiency_1d`` at degree 40.  Bar 5e-3 (plan).  Measured
    2026-10-02 (``a4_stripe_a0.05.json``): 4.48e-04 -- 1.05 decades under the
    bar -- the planning value to three digits; the no-cofactor fail-before
    (next test) reads 0.216, 1.6 decades above.  The ladder (build doc)
    reaches 9.7e-07 at M = 10, inside the plan's STOP condition (1e-5)."""
    o, R, T, _J, _st = _stripe_m8()
    err = float(np.max(np.abs(_vec(o, R, T, rows=(1,)) - _oracle_1d()[1])))
    assert err <= 5e-3, err


def test_a4_fail_before_no_cofactor(monkeypatch):
    """The same TE stripe with the far projector handed the no-cofactor view
    of the map must FAIL the 5e-3 bar.  Measured 2026-10-02: 0.2157 at M = 8
    (``a4_nocof_a0.05_M8.json``, planning 0.215); run here at M = 6, where it
    reads the same O(0.2) (the defect is a far-field mis-projection, not a
    discretisation error: 0.2175 measured, a4_nocof_a0.05_M6.json) while the correct arm reads 1.65e-03."""
    _patch_no_cofactor(monkeypatch)
    o, R, T, _J, _st = _solve(_stretch(0.05), _cell("stripe"), 6)
    err = float(np.max(np.abs(_vec(o, R, T, rows=(1,)) - _oracle_1d()[1])))
    assert err > 5e-3, err


# =========================================================================== #
# A5 -- the geometric split survives the map
# =========================================================================== #
def _split_rel(s, eps):
    qq = s.q * s.q
    E = np.zeros_like(s.Rmat)
    E[:qq, :qq], E[qq:, qq:] = s.Et_blocks
    if s.Et_offdiag is not None:
        E[:qq, qq:], E[qq:, :qq] = s.Et_offdiag
    return float(np.max(np.abs(E / eps + s.Rmat)) / np.max(np.abs(s.Rmat)))


def test_a5_geometric_split_minus_R_equals_eps_t_over_eps():
    """-R == [eps'_t] / eps for a uniform isotropic region under the map
    (plan 2.4: det chi_t = 1, so -R = chi_t^-1 = sqrt(g) g^-1).  Bar 1e-12
    (plan).  Measured 2026-10-02 (``a5_split.json``): <= 7.2e-16 over
    a = 0, 0.05, 0.15 p, eps = 1, 2.1025, 4 + 0.5i, M = 4, 6 -- 3.1 decades
    under the bar.  FAIL-BEFORE: a uniform MAGNETIC region (no map, the
    shipped magnetic route), where the split is genuinely broken: 1.0
    (scalar mu = 2) and 0.5 (mu = diag(2, 1, 1)) -- 11.7 decades above."""
    cm = _stretch(0.15)
    for eps in (1.0 + 0j, 4.0 + 0.5j):
        s = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 5,
                                   np.full((3, 3), eps), k0=_K0, cmap=cm)
        assert _split_rel(s, eps) <= 1e-12
    for mu in (np.full((2, 2), 2.0 + 0j),
               np.broadcast_to(np.diag([2.0, 1.0, 1.0]).astype(complex),
                               (2, 2, 3, 3)).copy()):
        s = TS.Granet2DTransverseE(_P, _P, 2, 2, 4,
                                   np.full((2, 2), 2.1025 + 0j), k0=_K0,
                                   mu_cell=mu)
        assert _split_rel(s, 2.1025) > 1e-2


# =========================================================================== #
# A6 -- the H-partner Gram, two-sided
# =========================================================================== #
def test_a6_hpartner_plain_gram_closes():
    """Correct arm: every mapped region (patterned layer AND half-spaces)
    recovers H through the PLAIN block Gram.  Bar: lossless closure <= 1e-5
    at M = 8 (plan).  Measured 2026-10-02 (``a6_hgram_M8.json``): 2.28e-06
    on the stripe (0.64 decades under), 1.03e-06 on the pillar."""
    _o, R, T, _J, _st = _stripe_m8()
    clo = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))
    assert clo <= 1e-5, clo


def test_a6_fail_before_mixed_hpartner(monkeypatch):
    """MIXED arm, engineered through the real stack path: the half-spaces'
    geometric cache recovers H through -R (the shipped unmapped
    ``_homog_geom_cache``) while the patterned layer keeps the plain Gram.
    Bar: closure > 1e-2 (plan).  Measured 2026-10-02: 0.157 at M = 8 and
    0.158 at M = 6 (``a6_hgram_M8.json`` / ``a6_hgram_M6.json`` -- the
    defect is an interface mismatch, M-independent), 1.2 decades above; run
    at M = 6."""
    orig = SP._homog_geom_cache

    def mixed(solver):
        W0, g2, GW0, SttW0, _Ginv, qq = orig(solver)
        return W0, g2, GW0, SttW0, np.linalg.inv(-solver.Rmat), qq
    monkeypatch.setattr(SP, "_homog_geom_cache", mixed)
    _o, R, T, _J, _st = _solve(_stretch(0.05), _cell("stripe"), 6)
    clo = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))
    assert clo > 1e-2, clo


# =========================================================================== #
# A7 -- absorption under a map
# =========================================================================== #
def test_a7_absorption_closure_under_stretch():
    """A LOSSY stripe (eps 4 + 0.5i) under the a = 0.05 p stretch, M = 7:
    sum_i layer_absorption_i == 1 - sum R - sum T.  NOT measured by the
    planning probes; measured 2026-10-02 (``a7_absorption.json``) on the
    stripe at M = 4..8: correct arm 3.1e-03, 1.7e-03, 2.0e-04, 1.99e-05,
    1.9e-06 (the unmapped solver on the same walls: 7.2e-08 at M = 7); the
    fail-before -R AS THE FLUX GRAM 0.046, 0.070, 0.070, 0.070, 0.070.  Bar
    1e-3 at M = 7: 1.7 decades above the correct arm, 1.8 decades below the
    fail-before (which is engineered here by replacing the retained Gram and
    re-running the real ``layer_absorption``)."""
    cm = _stretch(0.05)
    _o, R, T, _J, st = _solve(cm, _cell("stripe", 4.0 + 0.5j), 7,
                              retain=True)
    target = 1 - R.sum(axis=1) - T.sum(axis=1)
    A = st.layer_absorption().sum(axis=0)
    assert float(np.max(np.abs(A - target))) <= 1e-3
    s = TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 7,
                               np.full((3, 3), _NSUP ** 2 + 0j), k0=_K0,
                               cmap=cm)
    st._internal["G"] = (-s.Rmat).copy()
    Abad = st.layer_absorption().sum(axis=0)
    assert float(np.max(np.abs(Abad - target))) > 1e-3


# =========================================================================== #
# scope, refusals and the map protocol
# =========================================================================== #
def test_scope_refusals_name_their_phase():
    cm = _stretch(0.05)
    with pytest.raises(NotImplementedError, match="Phase E"):
        PMM2DStackPure(_P, _P, layer_grids="per-layer", cmap=cm)
    with pytest.raises(NotImplementedError, match="pmm_jones_2d_staggered"):
        TS.pmm_efficiency_2d_staggered(_P, _P, _cell("stripe"), 1.45, 1.0,
                                       0.5, _WL, cmap=cm)
    st = PMM2DStackPure(_P, _P, n_modes=4, cmap=cm)
    # Phase D (2026-10-03) lifted the BLOCK-FORM tensor and mu refusals these
    # lines used to pin (block-form tensors and mu are routed under a map,
    # gates in tests/unit/test_pmm2d_staggered_curved_d.py); the
    # OUT-OF-PLANE tensor and the slant stay refused, naming Phase E
    oop = np.diag([2.0, 2.5, 2.0]).astype(complex)
    oop[0, 2] = oop[2, 0] = 0.3
    with pytest.raises(NotImplementedError, match="Phase E"):
        st.add_layer(0.2, eps=oop)
    with pytest.raises(NotImplementedError, match="Phase E"):
        st.add_layer(0.2, eps=2.0, mu=oop)
    with pytest.raises(NotImplementedError, match="Phase E"):
        st.add_layer(0.2, eps_cell=_cell("stripe"), slant=(0.1, 0.0))
    with pytest.raises(NotImplementedError, match="Phase E"):
        TS.Granet2DTransverseE(_P, _P, cm.u_walls, cm.v_walls, 4,
                               np.broadcast_to(oop, (3, 3, 3, 3)), cmap=cm)
    with pytest.raises(ValueError, match="union-grid|common"):
        st.add_layer(0.2, eps_cell=np.ones((2, 2)))
    with pytest.raises(ValueError, match="wall grid"):
        TS.Granet2DTransverseE(_P, _P, 3, 3, 4, _cell("stripe"), cmap=cm)
    st.add_layer(0.2, eps_cell=_cell("stripe"))
    # Phase C (2026-10-02) lifted the viewer refusal this line used to pin:
    # a mapped stack now draws the PHYSICAL images of its cells (gate C11 of
    # tests/unit/test_pmm2d_staggered_curved_c.py checks the drawn outline
    # against the analytic curve), so the call must succeed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes = st.plot_geometry()
    plt.close(axes[0].figure)


def test_map_protocol_validation():
    """The constructors refuse a folding stretch, the solver refuses a map
    with det J <= 0 at any quadrature node, and the base validation refuses a
    map that is not lattice-periodic -- each two-sided against a legal map."""
    SeparableStretch(3, 3, fx=SineStretch(0.15 * _P), period_x=_P,
                     period_y=_P)                      # |a| < p/(2 pi): legal
    with pytest.raises(ValueError, match="folds the plane"):
        # 2 pi * 0.2 = 1.26 > 1: the slope changes sign
        SeparableStretch(3, 3, fx=SineStretch(0.2 * _P), period_x=_P,
                         period_y=_P)

    class Skew(CellMap):
        """x = u + c v: det J = 1 but NOT lattice-periodic in v."""

        def __init__(self, c):
            self.c = c
            self._init_walls(3, 3, _P, _P)

        def geom(self, sx, sy, U, V):
            X = U[:, None] + self.c * V[None, :]
            Y = np.broadcast_to(V[None, :], X.shape).astype(float)
            one = np.ones_like(X)
            return X, Y, one, self.c * one, 0.0 * one, one

        def _key(self):
            return (self.c,)
    Skew(0.0).validate()
    with pytest.raises(ValueError, match="lattice-periodic|onto the line"):
        Skew(0.1).validate()
    # fingerprints: content-equal maps agree, different ones do not
    a = _stretch(0.05)
    assert a.fingerprint == _stretch(0.05).fingerprint
    assert a.fingerprint != _stretch(0.06).fingerprint


def test_quadrature_is_adaptive_under_a_strong_stretch():
    """BUILD FINDING (``a2q_quadrature_fixed_base.json``): the planning rule
    nq = 2M + 8 leaves R / T 5.2e-04 (M = 5), 7.8e-05 (M = 7), 5.3e-06
    (M = 9) from an 8x finer rule under the a = 0.15 p stretch (its weights
    carry 1 / f' with f' down to 0.058), while the adaptive rule reads 0.0,
    0.0, 3.1e-11 (``a2q_quadrature_adaptive.json``).  The DECISION pinned
    here: the identity map (polynomial weights) keeps 2M + 8 exactly, and the
    a = 0.15 p stretch is driven to at least 4x more nodes (measured 144 at
    M = 5, i.e. 8x)."""
    M = 5
    bx = TS.Basis1D(_P, 3, M)
    assert TS._stag_map_nodes(bx, bx, IdentityMap(3, 3, _P, _P), M) == 2 * M + 8
    cm = _stretch(0.15)
    bx = TS.Basis1D(_P, cm.u_walls, M)
    by = TS.Basis1D(_P, cm.v_walls, M)
    assert TS._stag_map_nodes(bx, by, cm, M) >= 4 * (2 * M + 8)
