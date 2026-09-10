"""NON-UNIFORM SEGMENT BOUNDARIES in the staggered modified-Legendre basis
(Granet 2023 Eq. 31) -- gates N1-N7 of
``docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md``.

Formulation and the GO decision: the F5 section of
``docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md``.

Eq. 31 maps EACH segment individually, so nothing in the paper requires the
segments to be equal; the uniform lattice was an implementation choice, and
the whole change is ONE change at FOUR sites -- the scalar jacobian ``J``
becomes the per-segment vector ``J_n`` in ``Basis1D._global_matrix``,
``_global_pair_segmat``, ``Granet2DTransverseE._eps_dir``'s inline ``segmat``
and ``_stag_fourier_projection``.  Lifting it is what makes an arbitrary TAPER
representable at all: a wall that moves 1.8 nm per z-slice on a 700 nm period
needs ``N ~ 390`` uniform segments (``q >= 1170``, eig ``>= 2.7e+06``) and 3
non-uniform ones (``q = 3(M-1)``, eig 1152 at ``M = 9``).

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS**
(2026-09-11), and both readings are stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

Shapes used throughout, per ``docs/TESTING_STANDARDS.md``: bit-identity claims
are SAME-BUILD two-arm by sha256 of the raw bytes; every accuracy claim is
scored against an INDEPENDENT oracle with a bar DERIVED from that oracle's own
self-gap or from a triangle inequality, never fitted; and every site whose
correctness a uniform-lattice test cannot see carries a FAIL-BEFORE arm.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _C,
    Basis1D,
    Granet2DTransverseE,
    _global_pair_segmat,
    _stag_fourier_projection,
)

_PER = 0.9
_WL = 0.6
_THETA = 0.20
_EPS_P, _EPS_H = 6.0, 2.25


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# N1 -- the INTEGER path is BIT-IDENTICAL to the scalar-J library
# --------------------------------------------------------------------------
def _scalar_J(b):
    """The ONE scalar jacobian the pre-2026-09-11 basis carried."""
    return 0.5 * (b.d / b.N)


def _global_matrix_scalarJ(b, ref, setL, setR, eps_seg=None):
    """``Basis1D._global_matrix``'s PRE-CHANGE arithmetic, verbatim."""
    N = b.N
    J = _scalar_J(b)
    if ref is b.m_ref:
        scale = J
    elif ref is b.s_ref:
        scale = 1.0 / J
    else:
        scale = 1.0
    L_ten, R_ten = np.array(setL), np.array(setR)
    w_seg = np.ones(N, dtype=_C) * scale
    if eps_seg is not None:
        w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    return np.einsum("isa,s,jsa->ij", np.conj(L_ten), w_seg, RR)


def _pair_segmat_scalarJ(b, ref, setL, setR):
    J = _scalar_J(b)
    scale = J if ref is b.m_ref else (1.0 / J if ref is b.s_ref else 1.0)
    L_ten, R_ten = np.array(setL), np.array(setR)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    return scale * np.einsum("isa,jsa->sij", np.conj(L_ten), RR)


def _fourier_scalarJ(b, orders, alpha0=0.0):
    from numpy.polynomial.legendre import leggauss
    d, N, M = b.d, b.N, b.M
    G = 2.0 * np.pi / d
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = _ts._modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    J = _scalar_J(b)
    for seg in range(N):
        xphys = 0.5 * (b.xb[seg] + b.xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

    def _asm(gs):
        return np.einsum("msa,jsa->mj", T_local, np.array(gs))
    return _asm


def _eps_dir_scalarJ(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
    """``Granet2DTransverseE._eps_dir``'s PRE-CHANGE inline ``segmat``."""
    def segmat(basis, lset, op, rset):
        Lt = np.array(getattr(basis, lset))
        Rt = np.array(getattr(basis, rset))
        if op == "m":
            RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
            scale = _scalar_J(basis)
        elif op == "dL":
            RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
            scale = 1.0
        else:
            RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
            scale = 1.0
        return scale * np.einsum("isa,jsa->sij", np.conj(Lt), RR)
    Gx = segmat(bx, lx, opx, rx)
    Gy = segmat(by, ly, opy, ry)
    eps = self.eps_cell if wmap is None else wmap
    out = np.zeros((Gy.shape[1] * Gx.shape[1], Gy.shape[2] * Gx.shape[2]),
                   dtype=_C)
    for sx in range(bx.N):
        Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
        out += np.kron(Wy, Gx[sx])
    return out


def _cells(N):
    """The three cell KINDS the assembly dispatches on, on an N x N lattice."""
    sc = np.full((N, N), _EPS_H + 0j)
    sc[0, 0] = _EPS_P
    ten = np.zeros((N, N, 3, 3), dtype=_C)
    ten[:] = np.diag([_EPS_H, _EPS_H, _EPS_H])
    ten[0, 0] = np.array([[6.0, 0.9j, 0.0], [-0.9j, 6.0, 0.0], [0.0, 0.0, 5.0]])
    oop = np.zeros((N, N, 3, 3), dtype=_C)
    oop[:] = np.diag([_EPS_H, _EPS_H, _EPS_H])
    c, s = np.cos(0.6), np.sin(0.6)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    oop[0, 0] = rot @ np.diag([1.5 ** 2, 1.5 ** 2, 1.9 ** 2]) @ rot.T
    return {"scalar": sc, "tensor": ten, "oop": oop}


def test_n1_integer_walls_reproduce_the_scalar_jacobian_bit_for_bit():
    """N1.  ``walls = N`` (int) must reproduce the pre-change basis BYTE for
    BYTE -- on every matrix family, and on the ASSEMBLED pencils of all three
    cell kinds.

    This is the gate that makes ``x_walls=None`` a no-op promise.  It is a
    SAME-BUILD two-arm sha256 comparison against a scalar-``J``
    reimplementation of each site, so it is a claim about the arithmetic, not
    about a tolerance.  MEASURED: worst ``|d| = 0.0e+00`` and 0 hash
    mismatches over 6 grids x 8 families on BOTH builds; the 2-D pencils
    ``Rmat/Lmat/Stt/Schur/Agen/Bgen`` over 3 grids x 5 cell kinds (scalar,
    in-plane tensor, out-of-plane, magnetic, slanted) likewise identical.
    """
    worst = 0.0
    n = 0
    for d, N, M in ((1.2, 2, 5), (0.9, 3, 6), (0.9, 4, 4), (1.2, 6, 4),
                    (0.7, 5, 7), (1.55, 3, 8)):
        b = Basis1D(d, N, M, np.exp(-1j * 0.31 * d))
        assert b.uniform and b.J == 0.5 * d / N
        eps_seg = np.linspace(1.4, 5.7, N)
        for ref, sL, sR, es in ((b.m_ref, b.Btilde, b.Btilde, None),
                                (b.m_ref, b.B, b.B, None),
                                (b.m_ref, b.Btilde, b.Btilde, eps_seg),
                                (b.s_ref, b.Btilde, b.Btilde, None),
                                (b.c_ref, b.B, b.Btilde, None)):
            new = b._global_matrix(ref, sL, sR, es)
            old = _global_matrix_scalarJ(b, ref, sL, sR, es)
            assert _h(new) == _h(old)
            worst = max(worst, float(np.max(np.abs(new - old))))
            n += 1
        for ref in (b.m_ref, b.s_ref):
            new = _global_pair_segmat(b, ref, b.Btilde, b.Btilde)
            old = _pair_segmat_scalarJ(b, ref, b.Btilde, b.Btilde)
            assert _h(new) == _h(old)
            worst = max(worst, float(np.max(np.abs(new - old))))
            n += 1
        ords = np.arange(-3, 4)
        new = _stag_fourier_projection(b, ords, 0.31)(b.B)
        old = _fourier_scalarJ(b, ords, 0.31)(b.B)
        assert _h(new) == _h(old)
        worst = max(worst, float(np.max(np.abs(new - old))))
        n += 1
    assert n == 48
    assert worst == 0.0, worst

    # the ASSEMBLED pencils, all three dispatch kinds
    for N, M in ((2, 5), (3, 4)):
        for kind, cell in _cells(N).items():
            sol = Granet2DTransverseE(1.2, 1.2, N, N, M, cell, alpha0x=0.2,
                                      alpha0y=0.13, k0=2 * np.pi)
            orig = Granet2DTransverseE._eps_dir
            try:
                Granet2DTransverseE._eps_dir = _eps_dir_scalarJ
                ref = Granet2DTransverseE(1.2, 1.2, N, N, M, cell,
                                          alpha0x=0.2, alpha0y=0.13,
                                          k0=2 * np.pi)
            finally:
                Granet2DTransverseE._eps_dir = orig
            got = 0
            for attr in ("Rmat", "Lmat", "Stt", "Schur", "Agen", "Bgen"):
                a, c = getattr(sol, attr, None), getattr(ref, attr, None)
                if a is None:
                    continue
                assert _h(a) == _h(c), (N, M, kind, attr)
                got += 1
            assert got >= 2, (kind, got)


def test_n2_explicit_uniform_walls_are_ulp_close_not_bit_identical():
    """N2.  An EXPLICIT ``np.linspace`` wall array is NOT required to be
    bit-identical to the integer path, and the difference is arithmetic, not
    physics: ``linspace`` computes ``start + i*step`` and pins its last
    element, so ``linspace[i+1] - linspace[i]`` is not always the same double
    as ``d/N``.

    MEASURED: exactly 0.0 at ``d = 1.2`` for ``N = 2, 3, 4, 6``; 1.96e-16
    relative at ``d = 0.9, N = 4`` (both builds).  That is why the integer path
    is kept DISTINCT -- routing the int through the array path would make N1's
    unconditional claim depend on the period."""
    worst = 0.0
    for d, N, M in ((1.2, 2, 5), (1.2, 4, 5), (0.9, 4, 5), (0.9, 3, 6)):
        tau = np.exp(-1j * 0.31 * d)
        bi = Basis1D(d, N, M, tau)
        ba = Basis1D(d, np.linspace(0.0, d, N + 1), M, tau)
        assert bi.uniform and not ba.uniform
        assert bi.J is not None and ba.J is None      # loud, not silent
        Mi = bi.mass(bi.Btilde, bi.Btilde)
        Ma = ba.mass(ba.Btilde, ba.Btilde)
        worst = max(worst, float(np.max(np.abs(Mi - Ma)))
                    / float(np.max(np.abs(Mi))))
    # a magnitude bar with 3 decades of gap on both sides: ULP-level above,
    # and the smallest real signal this basis carries (N3's 9.6e-04 gap at
    # M = 5) far above.  MEASURED worst 1.96e-16 (WIN) / 1.96e-16 (WSL).
    assert worst < 1e-13, worst


def test_non_uniform_basis_refuses_the_scalar_jacobian_loudly():
    """``Basis1D.J`` is ``None`` on a non-uniform basis ON PURPOSE: an
    un-migrated reader must raise a ``TypeError`` immediately rather than
    silently apply one segment's scaling to all of them."""
    b = Basis1D(_PER, [0.0, 0.31 * _PER, _PER], 5)
    assert b.J is None and b.h is None
    assert b.N == 2
    with pytest.raises(TypeError):
        _ = 2.0 * b.J
    np.testing.assert_allclose(b.Jn, 0.5 * np.diff(b.xb))


@pytest.mark.parametrize("walls, msg", [
    ([0.0, 0.5 * _PER], "0 .. d"),                 # does not reach the period
    ([0.0, 0.7 * _PER, 0.3 * _PER, _PER], "increasing"),
    ([0.0, 0.3 * _PER, 0.3 * _PER, _PER], "increasing"),
])
def test_basis_wall_validation_raises(walls, msg):
    with pytest.raises(ValueError, match=msg):
        Basis1D(_PER, walls, 5)


# --------------------------------------------------------------------------
# N7 -- FAIL-BEFORE for the four J -> J_n sites
# --------------------------------------------------------------------------
def _nu_stack(M):
    """One layer on an ARBITRARY 3-segment non-uniform grid (walls at 0.2371
    and 0.6183 of the period), which the uniform lattice cannot represent at
    any affordable ``N``."""
    tile = np.array([[_EPS_H] * 3, [_EPS_P] * 3, [_EPS_H] * 3], dtype=_C)
    st = PMM2DStackPure(_PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=tile,
                 x_walls=[0.2371 * _PER, 0.6183 * _PER],
                 y_walls=[0.2371 * _PER, 0.6183 * _PER])
    st.set_source(_WL, theta=_THETA)
    return st


def _u_stack(M):
    """The same DEVICE class on the uniform lattice -- the arm every site's
    scalar-``J`` revert must leave BIT-IDENTICAL."""
    cell = np.full((3, 3), _EPS_H + 0j)
    cell[1, :] = _EPS_P
    st = PMM2DStackPure(_PER, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=cell)
    st.set_source(_WL, theta=_THETA)
    return st


def _rt(st):
    o, R, T = st.solve(jones=False)
    return np.concatenate([np.asarray(R).ravel(), np.asarray(T).ravel()])


def _patch_site(site):
    """Revert ONE of the four ``J -> J_n`` sites to the scalar jacobian."""
    if site == "global_matrix":
        orig = Basis1D._global_matrix
        Basis1D._global_matrix = _global_matrix_scalarJ
        return lambda: setattr(Basis1D, "_global_matrix", orig)
    if site == "global_pair_segmat":
        orig = _ts._global_pair_segmat
        _ts._global_pair_segmat = _pair_segmat_scalarJ
        return lambda: setattr(_ts, "_global_pair_segmat", orig)
    if site == "eps_dir":
        orig = Granet2DTransverseE._eps_dir
        Granet2DTransverseE._eps_dir = _eps_dir_scalarJ
        return lambda: setattr(Granet2DTransverseE, "_eps_dir", orig)
    orig = _ts._stag_fourier_projection
    _ts._stag_fourier_projection = _fourier_scalarJ
    return lambda: setattr(_ts, "_stag_fourier_projection", orig)


@pytest.mark.parametrize("site", ["global_matrix", "global_pair_segmat",
                                  "eps_dir", "fourier_projection"])
def test_n7_fail_before_each_scalar_jacobian_site(site):
    """N7.  Reverting ANY ONE of the four sites to the scalar ``J`` must

      (a) leave every UNIFORM solve BIT-IDENTICAL -- which is the whole reason
          a uniform-only gate cannot see any of them, exactly as a
          conforming-only gate cannot see the mortar's V1/V2 swap; and
      (b) BREAK a non-uniform solve by a margin decades above round-off.

    Both arms run in the SAME process on the SAME build, so (a) is a true
    byte claim and (b) is a ratio.  MEASURED margins are stated below."""
    _ts._STAG_GEO_CACHE.clear()
    good_u = _rt(_u_stack(5))
    good_n = _rt(_nu_stack(5))
    restore = _patch_site(site)
    try:
        _ts._STAG_GEO_CACHE.clear()
        with warnings.catch_warnings():
            # a deliberately BROKEN arm blows the lossless-closure tripwire --
            # which is the tripwire doing its job, not a defect to report here
            warnings.simplefilter("ignore", UserWarning)
            bad_u = _rt(_u_stack(5))
            bad_n = _rt(_nu_stack(5))
    finally:
        restore()
        _ts._STAG_GEO_CACHE.clear()
    # (a) the uniform arm does not move a single bit
    assert _h(bad_u) == _h(good_u), (site, float(np.max(np.abs(bad_u - good_u))))
    # (b) the non-uniform arm moves by decades.  The bar is a DECISION bar:
    # "the answer changed at the percent level", six decades above the
    # round-off floor the uniform arm just demonstrated (0.0) and two decades
    # below the smallest measured break.  MEASURED break per site (WIN / WSL)
    # is recorded in the build doc's N7 table.
    moved = float(np.max(np.abs(bad_n - good_n)))
    assert moved > 1e-3, (site, moved)


# --------------------------------------------------------------------------
# N3 / N4 -- against INDEPENDENT oracles
# --------------------------------------------------------------------------
def _oracle_1d(widths, eps, deg, thickness=0.30):
    """The EXACT 1-D pure PMM on the same device.  ``widths`` are absolute
    metres; ``PMMStack.add_layer`` takes FRACTIONS of the period."""
    st = PMMStack(_PER, degree=deg, far_field_orders=5)
    st.add_layer(thickness, segments=[(w / _PER, e)
                                      for w, e in zip(widths, eps)])
    st.set_source(_WL, theta=_THETA)
    return st.solve()


def _oracle_pair(widths, eps):
    """The exact 1-D oracle at degree 14, plus its OWN degree-12 self-gap --
    the number every bar below is derived from."""
    o14, R14, T14 = _oracle_1d(widths, eps, 14)[:3]
    o12, R12, T12 = _oracle_1d(widths, eps, 12)[:3]
    R14, T14 = np.atleast_2d(R14), np.atleast_2d(T14)
    keep = np.abs(np.asarray(o14)) <= 1
    gap = float(max(np.max(np.abs(np.atleast_2d(R12)[:, keep] - R14[:, keep])),
                    np.max(np.abs(np.atleast_2d(T12)[:, keep] - T14[:, keep]))))
    return np.asarray(o14), R14, T14, gap


def _score_vs_1d(o2d, R, T, o1d, R1d, T1d):
    """max |R - R_1D|, |T - T_1D| over the retained ``n = 0`` orders, TE row
    (incident ``E_y``) -- the stripe stack is y-uniform, so the exact 1-D pure
    PMM is the oracle for the WHOLE 2-D answer."""
    best = 0.0
    for m in (-1, 0, 1):
        sel = np.where((o2d[:, 0] == m) & (o2d[:, 1] == 0))[0][0]
        j = int(np.where(o1d == m)[0][0])
        best = max(best, abs(float(R[1, sel]) - float(R1d[1, j])),
                   abs(float(T[1, sel]) - float(T1d[1, j])))
    return best


def test_n3_two_exact_wall_representations_agree_inside_the_triangle_bar():
    """N3.  A duty-1/3 stripe has TWO exact-wall representations -- a
    NON-uniform 2-segment grid with its wall at ``P/3`` and the uniform
    3-segment lattice -- and they must converge to the same answer.

    The bar is the TRIANGLE INEQUALITY against the exact 1-D oracle, derived
    per rung and not fitted: two representations of ONE device may disagree by
    at most the SUM of their own distances to truth.  MEASURED 5/5 rungs
    inside on both builds; the NU arm reaches its accuracy on 2/3 of the
    uniform arm's degrees of freedom (``q = 2(M-1)`` vs ``3(M-1)``)."""
    o1d, R1d, T1d, self_gap = _oracle_pair(
        (_PER / 3.0, 2.0 * _PER / 3.0), (_EPS_P, _EPS_H))
    # the oracle is readable throughout: its own degree-12-vs-14 self-gap
    # (MEASURED 1.55e-05 WIN) sits well below the smallest gap scored below
    assert self_gap < 1e-4, self_gap

    cnu = np.array([[_EPS_P, _EPS_P], [_EPS_H, _EPS_H]], dtype=_C)
    cu = np.full((3, 3), _EPS_H + 0j)
    cu[0, :] = _EPS_P
    for M in (4, 5, 6, 7):
        stn = PMM2DStackPure(_PER, n_modes=M, n_orders=1,
                             layer_grids="per-layer")
        stn.add_layer(0.30, eps_cell=cnu, x_walls=[_PER / 3.0],
                      y_walls=[_PER / 3.0])
        stn.set_source(_WL, theta=_THETA)
        on, Rn, Tn = stn.solve(jones=False)
        stu = PMM2DStackPure(_PER, n_modes=M, n_orders=1)
        stu.add_layer(0.30, eps_cell=cu)
        stu.set_source(_WL, theta=_THETA)
        ou, Ru, Tu = stu.solve(jones=False)
        en = _score_vs_1d(on, Rn, Tn, o1d, R1d, T1d)
        eu = _score_vs_1d(ou, Ru, Tu, o1d, R1d, T1d)
        gap = 0.0
        for m in (-1, 0, 1):
            i1 = np.where((on[:, 0] == m) & (on[:, 1] == 0))[0][0]
            i2 = np.where((ou[:, 0] == m) & (ou[:, 1] == 0))[0][0]
            gap = max(gap, abs(float(Rn[1, i1]) - float(Ru[1, i2])),
                      abs(float(Tn[1, i1]) - float(Tu[1, i2])))
        assert gap <= en + eu + 4.0 * self_gap, (M, gap, en, eu)


def test_n4_arbitrary_walls_converge_to_the_exact_1d_answer():
    """N4.  A cell whose walls sit at 0.2371 and 0.6183 of the period -- walls
    the shipped uniform lattice cannot represent at any affordable ``q`` --
    converges to the EXACT 1-D answer on its own 3-segment grid.

    Made y-uniform so the exact 1-D ``PMMStack`` at degree 14 IS the truth for
    the whole 2-D stack.  The claim is a DECISION -- three rungs, each at
    least 2x tighter than the one below, ending decades above the oracle's own
    self-gap -- not a pinned reading.  MEASURED (WIN / WSL) in the build doc's
    N4 table."""
    w0, w1 = 0.2371 * _PER, 0.6183 * _PER
    o1d, R1d, T1d, self_gap = _oracle_pair(
        (w0, w1 - w0, _PER - w1), (_EPS_H, _EPS_P, _EPS_H))
    tile = np.array([[_EPS_H] * 3, [_EPS_P] * 3, [_EPS_H] * 3], dtype=_C)
    errs = []
    for M in (4, 5, 7):
        st = PMM2DStackPure(_PER, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=tile, x_walls=[w0, w1], y_walls=[w0, w1])
        st.set_source(_WL, theta=_THETA)
        o, R, T = st.solve(jones=False)
        errs.append(_score_vs_1d(o, R, T, o1d, R1d, T1d))
        # two-sided lossless closure, tightening with M
        assert abs(float(R.sum(1)[1] + T.sum(1)[1] - 1.0)) < 5e-3, (M, errs)
    # a DECISION: each rung at least 2x tighter than the last, and the last is
    # two decades inside the coarsest -- the shape a converging discretization
    # has and a wrong one does not.  MEASURED errs at q = 9 / 12 / 18:
    # 3.3205e-02 / 3.8906e-03 / 4.6695e-04 on WIN and 3.3205e-02 / 3.8906e-03 /
    # 4.6695e-04 on WSL, i.e. the two builds agree to 10+ digits because these
    # readings are DISCRETISATION-limited.  The ladder continues to 5.3543e-06
    # at q = 24 with closure 1.98e-10 (build doc, S4.3).
    assert errs[1] < 0.5 * errs[0] and errs[2] < 0.5 * errs[1], errs
    # ... and the last rung is still READABLE against the oracle (above its
    # own self-gap 8.5373e-06), so the measurement has not run into the
    # oracle's floor and the convergence above is the solver's, not noise.
    assert errs[-1] < 1e-3 and errs[-1] > 2.0 * self_gap, (errs, self_gap)


def test_n6_conforming_identity_holds_on_non_uniform_grids():
    """N6.  Forcing the mortar between IDENTICAL NON-UNIFORM grids must
    reproduce the identical-grid bypass.

    This is the conforming identity of the mortar gates, re-established on the
    partition class the taper actually uses, through THREE forced interfaces.
    It is the control that separates "the multi-layer machinery on arbitrary
    walls" from "the mortar": the two agree at round-off, so any residual a
    taper shows is the device, not the interface.  MEASURED 2.1e-16 / 3.6e-16
    (probe, 5 forced interfaces); the bar below is a magnitude bar with
    decades to the smallest real signal."""
    w0, w1 = 0.2371 * _PER, 0.6183 * _PER
    tile = np.array([[_EPS_H] * 3, [_EPS_P] * 3, [_EPS_H] * 3], dtype=_C)

    def _build(M):
        st = PMM2DStackPure(_PER, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        for _ in range(3):
            st.add_layer(0.10, eps_cell=tile, x_walls=[w0, w1],
                         y_walls=[w0, w1])
        st.set_source(_WL, theta=_THETA)
        return st

    for M in (5, 6):
        o1, R1, T1, J1 = _build(M).solve()
        o2, R2, T2, J2 = _build(M)._solve_per_layer(
            jones=True, retain_internal=False, force_mortar=True)
        d = max(float(np.max(np.abs(R1 - R2))), float(np.max(np.abs(T1 - T2))),
                float(np.max(np.abs(J1 - J2))))
        assert d < 1e-11, (M, d)


def test_axes_may_carry_different_wall_positions():
    """The staggered tensor basis needs equal SEGMENT COUNTS per axis
    (``bx.dim == by.dim``); the wall POSITIONS may differ freely, which is
    what a rectangular tapered feature needs."""
    tile = np.array([[_EPS_H] * 3, [_EPS_P] * 3, [_EPS_H] * 3], dtype=_C)
    st = PMM2DStackPure(_PER, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.30, eps_cell=tile, x_walls=[0.21 * _PER, 0.55 * _PER],
                 y_walls=[0.33 * _PER, 0.78 * _PER])
    st.set_source(_WL, theta=_THETA)
    o, R, T = st.solve(jones=False)
    assert abs(float(R.sum(1)[1] + T.sum(1)[1] - 1.0)) < 5e-3
    # ...but UNEQUAL counts are refused, loudly
    st2 = PMM2DStackPure(_PER, n_modes=5, n_orders=1, layer_grids="per-layer")
    with pytest.raises(ValueError, match="segments"):
        st2.add_layer(0.30, eps_cell=tile, x_walls=[0.21 * _PER],
                      y_walls=[0.33 * _PER, 0.78 * _PER])


def test_wall_spec_accepts_both_spellings_and_refuses_mismatches():
    """Walls are accepted as the hybrid's INTERIOR list or as a FULL boundary
    array, and a tile whose strip count disagrees with them is refused."""
    tile = np.array([[_EPS_H] * 3, [_EPS_P] * 3, [_EPS_H] * 3], dtype=_C)
    w = [0.2371 * _PER, 0.6183 * _PER]
    full = [0.0] + w + [_PER]

    def _solve(xw, yw):
        st = PMM2DStackPure(_PER, n_modes=5, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=tile, x_walls=xw, y_walls=yw)
        st.set_source(_WL, theta=_THETA)
        return st.solve(jones=False)[1]

    np.testing.assert_array_equal(_solve(w, w), _solve(full, full))
    st = PMM2DStackPure(_PER, n_modes=5, n_orders=1, layer_grids="per-layer")
    with pytest.raises(ValueError, match="strips"):
        st.add_layer(0.30, eps_cell=np.full((2, 2), _EPS_H + 0j),
                     x_walls=w, y_walls=w)


def test_parity_reduction_is_refused_on_an_unsymmetric_wall_set():
    """``x -> d - x`` sends segment ``n`` to ``N-1-n``, which is a signed
    permutation of the LOCAL functions only when the two segments have equal
    length.  On an unsymmetric non-uniform wall set the parity gauge must
    report NO structure, so the out-of-plane solve takes the dense
    ``4 q^2`` eig -- correct by construction -- instead of a wrong block
    reduction."""
    d, M = _PER, 5
    sym = Basis1D(d, [0.0, 0.3 * d, 0.7 * d, d], M)
    asym = Basis1D(d, [0.0, 0.2371 * d, 0.6183 * d, d], M)
    assert _ts._stag_parity_1d(sym) is not None
    assert _ts._stag_parity_1d(asym) is None
    assert _ts._stag_parity_1d(Basis1D(d, 3, M)) is not None
