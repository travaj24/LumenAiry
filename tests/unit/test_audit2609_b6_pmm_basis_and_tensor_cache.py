"""WP-B6 (audit 2026-09-11, Wave 4) -- the two PMM items WP-A12/A13 deferred
with designs: the ultraspherical (Gegenbauer) basis for the TM wall corner, and
the k0-free projected TENSOR operator cache for ``PMM2DStackHybrid``.

Findings pinned here
--------------------
* **G3 (P2), WP-A12 sec. 6 item 3** -- the audit's alternative (c): swap the
  Legendre/GLL nodal basis for an ultraspherical Gauss-Lobatto one and
  "recover exponential convergence for the TM wall-corner singularity".
  MEASURED AND NOT SHIPPED.  Two facts decide it, and both are re-measured
  here on the running build rather than quoted:

  1. with the element integrals evaluated EXACTLY the whole one-parameter
     family is a change of nodal basis of the SAME C0 piecewise-P_N space, so
     it cannot move the answer at all (:func:`test_b6_an_exactly_integrated_
     ultraspherical_basis_cannot_move_the_answer`);
  2. what the family actually varies in the shipped assembly is the NODAL
     QUADRATURE, and only the GLL rule makes the lumped mass satisfy
     summation-by-parts -- the discrete integration-by-parts identity the
     energy algebra rests on (:func:`test_b6_only_the_gll_rule_satisfies_
     summation_by_parts`, :func:`test_b6_a_non_gll_nodal_rule_loses_the_exact_
     energy_identity`).

  So nothing in ``lumenairy/elements/pmm`` gained a ``basis=`` knob, and these
  three tests are the record of why: they state what a future attempt must
  re-measure.  The full rate ladder is in ``fixes/WP-B6_REPORT.md`` sec. 2.1.
* **G10(d) (P2), WP-A13 sec. 6.1** -- ``PMM2DStackHybrid._build_layer_modes``
  passed ``lops=None`` for a tensor layer, so ``_geom_cache`` stored
  ``(ax, ay, None)`` and ``_tensor_layer_modes`` rebuilt the per-axis
  projections (``_axis_projection`` + ``pinv``) and the ``_proj`` sandwiches at
  EVERY wavelength and EVERY angle, where the scalar branch cached them.  The
  source-free half of the assembly is now
  :func:`~lumenairy.elements.pmm.twod_jones._tensor_projected_ops` and the
  stack caches it; the operators are asserted BIT-IDENTICAL with and without
  the cache, and the saving is asserted as a deterministic BUILD COUNT, never
  as wall clock.

  Two of those build-count contracts are asserted TWICE, at different
  instruments, because the obvious instrument cannot see the regression it
  exists to forbid.  ``_count_builds`` spies on ``stack2d``'s binding of
  ``_tensor_projected_ops``, which counts the assembly that FILLS the cache
  but not the one ``_tensor_layer_modes`` performs for itself through
  ``twod_jones``'s own global when it is handed no ``ops``; and
  ``::test_b6_the_cached_entry_serves_both_truncations`` builds a fresh stack
  per truncation, so it never shares one entry between the two.
  ``::test_b6_the_stack_hands_its_cached_build_to_the_layer_modes_builder``
  and ``::test_b6_one_entry_serves_a_truncation_FLIP_on_the_same_stack`` close
  those two, counting BOTH module bindings and flipping ``truncation`` on ONE
  stack; each names the source mutation it was measured to catch.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander

from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_1d
from lumenairy.elements.pmm import _core as PC
from lumenairy.elements.pmm import stack2d as S2
from lumenairy.elements.pmm import twod_jones as TJ
from lumenairy.elements.pmm.twod import _build_axis
from lumenairy.elements.pmm.twod_jones import (
    _tensor_layer_modes,
    _tensor_projected_ops,
)

_EPS = float(np.finfo(float).eps)


# =========================================================================== #
# G3 / WP-A12 sec. 6 item 3 -- the ultraspherical basis, measured and declined
# =========================================================================== #
#
# The family, built here rather than in the library because the library does
# not ship it.  The ultraspherical Gauss-Lobatto nodes for parameter ``lam``
# are +/-1 plus the roots of ``d/dx C_N^(lam)``, i.e. the roots of
# ``C_{N-1}^(lam+1) ~ P_{N-1}^(a,a)`` with ``a = lam + 1/2``; ``lam = 1/2``
# is the Legendre/GLL rule the library ships.  Roots by Golub-Welsch on the
# monic symmetric-Jacobi recurrence ``beta_n = n(n+2a)/((2n+2a+1)(2n+2a-1))``.

def _ultraspherical_nodes(degree, lam):
    if degree == 1:
        return np.array([-1.0, 1.0])
    k = np.arange(1, degree - 1)
    a = lam + 0.5
    beta = k * (k + 2.0 * a) / ((2.0 * k + 2.0 * a + 1.0)
                                * (2.0 * k + 2.0 * a - 1.0))
    J = np.diag(np.sqrt(beta), 1) + np.diag(np.sqrt(beta), -1)
    return np.concatenate([[-1.0], np.sort(np.linalg.eigvalsh(J)), [1.0]])


def _interpolatory_weights(nodes):
    """``w_i = INT_{-1}^{1} l_i(x) dx`` -- the unweighted rule at ``nodes``,
    from the Legendre Vandermonde (``V^T w = (2, 0, ..., 0)``).  At the GLL
    nodes this IS the GLL weight set (GLL is interpolatory)."""
    n = len(nodes) - 1
    m = np.zeros(n + 1)
    m[0] = 2.0
    return np.linalg.solve(legvander(nodes, n).T, m)


def _lagrange_values_and_derivatives(nodes, xq):
    n = len(nodes)
    Lv = np.ones((len(xq), n))
    for j in range(n):
        for k in range(n):
            if k != j:
                Lv[:, j] *= (xq - nodes[k]) / (nodes[j] - nodes[k])
    Dv = np.zeros((len(xq), n))
    for j in range(n):
        for m in range(n):
            if m == j:
                continue
            term = np.ones(len(xq)) / (nodes[j] - nodes[m])
            for k in range(n):
                if k not in (j, m):
                    term *= (xq - nodes[k]) / (nodes[j] - nodes[k])
            Dv[:, j] += term
    return Lv, Dv


def _exact_reference_element(nodes):
    """``(M, K, C)`` on ``[-1, 1]``: ``INT l_i l_j``, ``INT l_i' l_j'``,
    ``INT l_i l_j'`` -- EXACT (Gauss-Legendre, ``n + 4`` points, so degree
    ``2n + 7 >= 2N``)."""
    n = len(nodes)
    xq, wq = leggauss(n + 4)
    Lv, Dv = _lagrange_values_and_derivatives(nodes, xq)
    return (Lv.T * wq) @ Lv, (Dv.T * wq) @ Dv, (Lv.T * wq) @ Dv


def _patched_nodes_weights(lam):
    cache = {}

    def f(degree: int):
        if degree not in cache:
            nd = _ultraspherical_nodes(int(degree), lam)
            w = _interpolatory_weights(nd)
            nd.setflags(write=False)
            w.setflags(write=False)
            cache[degree] = (nd, w)
        return cache[degree]
    return f


def _exactly_integrated_build_sem():
    """``_core._build_sem`` with the element integrals evaluated EXACTLY --
    i.e. the CONSISTENT (non-diagonal) mass the shipped lumped assembly
    approximates.  Everything else, including the periodic ``l2g`` wrap, is the
    shipped body."""
    cache = {}

    def build(period, d_wall, eps_ridge, eps_groove, degree,
              n_ridge_el, n_groove_el, grade):
        ref_nodes, _ref_w = PC._gll_nodes_weights(degree)
        if int(degree) not in cache:
            cache[int(degree)] = _exact_reference_element(np.asarray(ref_nodes))
        Mref, Kref, Cref = cache[int(degree)]
        rb = PC._graded_boundaries(0.0, d_wall, n_ridge_el, grade)
        gb = PC._graded_boundaries(d_wall, period, n_groove_el, grade)
        elem_bnds = (list(zip(rb[:-1], rb[1:], [eps_ridge] * n_ridge_el))
                     + list(zip(gb[:-1], gb[1:], [eps_groove] * n_groove_el)))
        n_el = len(elem_bnds)
        l2g = np.zeros((n_el, degree + 1), dtype=int)
        gid = 0
        for e in range(n_el):
            for a in range(degree + 1):
                if a == 0 and e > 0:
                    l2g[e, a] = l2g[e - 1, degree]
                else:
                    l2g[e, a] = gid
                    gid += 1
        last = l2g[n_el - 1, degree]
        l2g[l2g == last] = 0
        n_glob = last
        z = [np.zeros((n_glob, n_glob), dtype=PC._C) for _ in range(7)]
        S0, Peps, Pinv, L, Linv, C, Cinv = z
        for e in range(n_el):
            xl, xr, eps = elem_bnds[e]
            J = 0.5 * (xr - xl)
            inv = 1.0 / eps
            ix = np.ix_(l2g[e], l2g[e])
            S0[ix] += Mref * J
            Peps[ix] += eps * (Mref * J)
            Pinv[ix] += inv * (Mref * J)
            L[ix] += Kref / J
            Linv[ix] += inv * (Kref / J)
            C[ix] += Cref
            Cinv[ix] += inv * Cref
        return dict(S0=S0, Peps=Peps, Pinv=Pinv, L=L, Linv=Linv, C=C,
                    Cinv=Cinv, n_glob=n_glob, l2g=l2g, elem_bnds=elem_bnds,
                    degree=degree, ref_nodes=ref_nodes)
    return build


# the auditor's lossless high-contrast TM cell (repro/PMM-1D/p8b_conv_rate.py),
# oblique so that more than one order propagates and energy closure is a real
# constraint rather than a tautology on the single specular order
_CELL = dict(period=0.6e-6, n_ridge=3.48 + 0j, depth=0.1e-6, duty=0.5,
             wl=0.633e-6, angle=np.deg2rad(10.0))


def _solve_1d(degree, pol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_efficiency_1d(
            _CELL["period"], _CELL["n_ridge"], 1.0, 1.0, 1.0, _CELL["depth"],
            _CELL["duty"], _CELL["wl"], angle=_CELL["angle"],
            polarization=pol, degree=degree, elements_per_region=1, grade=True,
            far_field_orders=15, stabilize=False)


@pytest.mark.parametrize("degree", [8, 16, 24, 32])
def test_b6_only_the_gll_rule_satisfies_summation_by_parts(degree):
    """The lumped SEM mass is legitimate ONLY because the GLL nodal rule is
    exact to degree ``2N-1``, which makes ``(diag(w), D)`` satisfy
    summation-by-parts EXACTLY::

        M D + (M D)^T = B,     B = diag(-1, 0, ..., 0, +1)

    -- the discrete integration by parts every energy / reciprocity identity in
    this module rests on.  No other node set in the ultraspherical family has
    it, and losing it is not a rounding effect: the residual jumps from the
    arithmetic floor to O(1).

    Bars, both derived and both re-measured here.  BELOW: the residual is an
    ``O(n)`` accumulation of products of order one, so its roundoff floor is
    ``n * eps``; the bar carries 1000x of slack over that (measured on this
    build 2026-09-13: 1.3e-14 at degree 8 rising to 1.9e-13 at degree 32,
    against a bar of 1.6e-12 .. 7.3e-12).  ABOVE: every other ``lam`` is
    asserted at more than 1e6 x the bar -- measured 3.7e-01 .. 1.5e+00, i.e.
    about eleven decades clear, so this is not a tolerance question.
    """
    n = degree + 1
    bar = 1.0e3 * n * _EPS
    B = np.zeros((n, n))
    B[0, 0], B[-1, -1] = -1.0, 1.0

    nodes, w = PC._gll_nodes_weights(degree)
    Cm = np.diag(np.asarray(w)) @ PC._lagrange_derivative_matrix(nodes)
    gll_resid = float(np.max(np.abs(Cm + Cm.T - B)))
    assert gll_resid < bar, (
        f"the SHIPPED GLL rule lost summation-by-parts: residual "
        f"{gll_resid:.3e} against the {bar:.3e} arithmetic bar")

    # the same construction on the ultraspherical nodes, and the same
    # construction with EXACT integration (which restores it for every lam)
    for lam in (0.0, 0.25, 0.75, 1.0, 1.5):
        nd = _ultraspherical_nodes(degree, lam)
        Cl = np.diag(_interpolatory_weights(nd)) @ PC._lagrange_derivative_matrix(nd)
        lumped = float(np.max(np.abs(Cl + Cl.T - B)))
        assert lumped > 1.0e6 * bar, (
            f"lam={lam}: the lumped ultraspherical rule was expected to BREAK "
            f"summation-by-parts; residual {lumped:.3e} sits under "
            f"{1.0e6 * bar:.3e}")
        _M, _K, Cex = _exact_reference_element(nd)
        exact = float(np.max(np.abs(Cex + Cex.T - B)))
        assert exact < bar, (
            f"lam={lam}: EXACT integration must restore the identity; "
            f"residual {exact:.3e} against {bar:.3e}")

    # and the GLL node set is where the shipped rule actually sits
    assert np.max(np.abs(np.asarray(nodes)
                         - _ultraspherical_nodes(degree, 0.5))) < 1.0e-13


def test_b6_an_exactly_integrated_ultraspherical_basis_cannot_move_the_answer(
        monkeypatch):
    """The decisive negative for WP-A12 sec. 6 item 3.

    Galerkin on a fixed space with EXACT integration is invariant under a
    change of nodal basis (the operators transform by a congruence, the pencil
    by a similarity), so the whole ``lam`` family returns ONE answer.  The
    ``lam`` knob can therefore only ever act through the QUADRATURE, which is
    the variational crime the lumped mass already commits -- it is not a
    property of the basis, and it cannot change the corner's approximation
    rate.

    Build-free by construction: the test measures BOTH arms on the running
    build and compares them to each other.  ABOVE the exact-integration
    deviation sits the LUMPED family's own deviation at the same ``lam`` and
    degree -- the smallest real signal the option was supposed to deliver --
    and the bar is 1e-4 of it.  Measured 2026-09-13: exact 3.1e-13 (TM) /
    5.9e-14 (TE) against lumped 2.5e-05 (TM) / 8.8e-06 (TE), i.e. a factor of
    ~1e8, and the bar sits four decades under the lumped signal.
    """
    degree = 24
    got = {}
    for tag, lam, exact in (("exact-0.5", 0.5, True), ("exact-1.0", 1.0, True),
                            ("lumped-0.5", 0.5, False),
                            ("lumped-1.0", 1.0, False)):
        with monkeypatch.context() as mp:
            mp.setattr(PC, "_gll_nodes_weights", _patched_nodes_weights(lam))
            if exact:
                mp.setattr(PC, "_build_sem", _exactly_integrated_build_sem())
            out = {}
            for pol in ("te", "tm"):
                _o, R, T = _solve_1d(degree, pol)
                out[pol] = np.concatenate([np.real(R), np.real(T)])
            got[tag] = out
        PC._clear_pmm_caches()

    for pol in ("te", "tm"):
        exact_dev = float(np.max(np.abs(got["exact-1.0"][pol]
                                        - got["exact-0.5"][pol])))
        lumped_dev = float(np.max(np.abs(got["lumped-1.0"][pol]
                                         - got["lumped-0.5"][pol])))
        assert lumped_dev > 1.0e-7, (
            f"{pol}: the LUMPED arms must actually differ for this comparison "
            f"to mean anything; measured {lumped_dev:.3e}")
        assert exact_dev < 1.0e-4 * lumped_dev, (
            f"{pol}: an exactly-integrated basis swap moved the answer by "
            f"{exact_dev:.3e}, which is not 1e-4 of the lumped family's own "
            f"{lumped_dev:.3e} -- the change-of-basis invariance this WP's "
            f"negative rests on no longer holds")


def test_b6_a_non_gll_nodal_rule_loses_the_exact_energy_identity(monkeypatch):
    """The measured PRICE of the family, and the second reason nothing shipped.

    A lossless grating at OBLIQUE incidence closes ``sum(R) + sum(T) = 1`` to
    the arithmetic floor on the shipped basis, because the GLL lumping keeps
    summation-by-parts exactly.  Swap the nodal rule and the closure defect
    becomes a physical-looking number -- large enough to matter, small enough
    that no tripwire in the library fires (``_check_energy`` screens at 5 %)
    and that ``_energy_clean_pick``'s "evidently lossless" classifier
    (``< 1e-6``) can flip.

    Both closures are measured here; the claim is the RATIO, so no build-
    dependent absolute floor is pinned.  Measured 2026-09-13 at degree 20:
    GLL 5.8e-14 / 2.5e-14 (te/tm) against lam=1.0 6.5e-07 / 3.1e-07, a ratio
    of ~1e7 against the 1e3 bar.
    """
    for pol in ("te", "tm"):
        _o, R, T = _solve_1d(20, pol)
        gll = abs(float(np.sum(R) + np.sum(T)) - 1.0)
        assert gll < 1.0e-9, (
            f"{pol}: the shipped basis no longer conserves energy on a "
            f"lossless cell (defect {gll:.3e}) -- the premise of this "
            f"comparison is gone")
        with monkeypatch.context() as mp:
            mp.setattr(PC, "_gll_nodes_weights", _patched_nodes_weights(1.0))
            _o, R2, T2 = _solve_1d(20, pol)
        PC._clear_pmm_caches()
        ultra = abs(float(np.sum(R2) + np.sum(T2)) - 1.0)
        assert ultra > 1.0e3 * max(gll, 1.0e-13), (
            f"{pol}: the ultraspherical rule was expected to LOSE the exact "
            f"energy identity; closure {ultra:.3e} against the shipped "
            f"{gll:.3e}")


# =========================================================================== #
# G10(d) / WP-A13 sec. 6.1 -- the k0-free projected TENSOR operator cache
# =========================================================================== #

_PX = _PY = 0.42e-6
_WL2 = 0.62e-6
_DEG2, _NO2 = 7, 3


def _rot3(exx, eyy, exy, ezz):
    t = np.zeros((3, 3), dtype=complex)
    t[0, 0], t[1, 1], t[2, 2] = exx, eyy, ezz
    t[0, 1] = t[1, 0] = exy
    return t


def _oop3(exx, eyy, ezz, exz, eyz):
    """A RECIPROCAL out-of-plane tensor (``e_xz = e_zx``, ``e_yz = e_zy``), so
    the cell is passive and the shared energy tripwire stays quiet -- an
    asymmetric one legitimately gives ``R + T != 1`` (module docstring) and
    would bury this file's assertions in warnings."""
    t = _rot3(exx, eyy, 0.0, ezz)
    t[0, 2] = t[2, 0] = exz
    t[1, 2] = t[2, 1] = eyz
    return t


_A = _rot3(6.2 + 0.11j, 3.4, 0.9 - 0.05j, 5.1)
_B = _rot3(2.05, 2.6 + 0.04j, -0.4, 2.3)
_OA = _oop3(6.2 + 0.11j, 3.4, 5.1, 0.7, 0.21)
_OB = _oop3(2.05, 2.6 + 0.04j, 2.3, -0.4, 0.17)

_UNIFORM = np.array([[_A]])
_SEPX = np.array([[_A], [_B]])
_SEPY = np.array([[_A, _B]])
_CROSS = np.array([[_A, _B], [_B, _A]])
_SEPX_O = np.array([[_OA], [_OB]])
_CROSS_O = np.array([[_OA, _OB], [_OB, _OA]])

_BRANCHES = {
    "uniform": ([], [], _UNIFORM, "laurent", "uniform"),
    "separable_x": ([0.21e-6], [], _SEPX, "laurent", "x"),
    "separable_x_li": ([0.21e-6], [], _SEPX, "li", "x"),
    "separable_x_fffnv": ([0.21e-6], [], _SEPX, "fff_nv", "x"),
    "separable_y": ([], [0.19e-6], _SEPY, "laurent", "y"),
    "crossed": ([0.21e-6], [0.19e-6], _CROSS, "laurent", "xy"),
    "crossed_li": ([0.21e-6], [0.19e-6], _CROSS, "li", "xy"),
    "out_of_plane_sep": ([0.21e-6], [], _SEPX_O, "laurent", "x"),
    "out_of_plane_crossed": ([0.21e-6], [0.19e-6], _CROSS_O, "li", "xy"),
}
_OPNAMES = ("GxF", "GyF", "Cxx", "Cxy", "Cyx", "Cyy", "EZZ")


def _source(theta_deg, phi_deg, n_orders=_NO2, wl=_WL2, circular=False):
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x, order_y = np.tile(ox, len(oy)), np.repeat(oy, len(ox))
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    kx0 = float(np.sin(th) * np.cos(ph))
    ky0 = float(np.sin(th) * np.sin(ph))
    keep = None
    if circular:
        r2 = min(n_orders / _PX, n_orders / _PY) ** 2
        keep = ((order_x / _PX) ** 2 + (order_y / _PY) ** 2) <= r2 * (1 + 1e-9)
    return dict(ox=ox, oy=oy, k0=2.0 * np.pi / wl, kx0=kx0, ky0=ky0,
                kxv=kx0 + order_x * (wl / _PX),
                kyv=ky0 + order_y * (wl / _PY), keep=keep)


def _capture_operators(monkeypatch, x_walls, y_walls, tile, formulation, src,
                       ops=None, slant=None, block_eig=False):
    """The seven projected operators (plus the four out-of-plane blocks) AT THE
    HAND-OFF to ``_layer_eigenmodes_tensor`` -- the point the cached and the
    uncached path must agree on."""
    grabbed = {}
    orig = TJ._layer_eigenmodes_tensor

    def spy(GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ, slant=None, sym_gauge=None,
            **oop):
        for name, v in zip(_OPNAMES, (GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ)):
            grabbed[name] = np.array(v, copy=True)
        for k, v in oop.items():
            if v is not None:
                grabbed[k] = np.array(v, copy=True)
        if sym_gauge is not None:
            grabbed["gauge"] = np.array(sym_gauge, copy=True)
        return orig(GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ, slant=slant,
                    sym_gauge=sym_gauge, **oop)

    monkeypatch.setattr(TJ, "_layer_eigenmodes_tensor", spy)
    ax = _build_axis(_PX, x_walls, _DEG2, 1, False)
    ay = _build_axis(_PY, y_walls, _DEG2, 1, False)
    _tensor_layer_modes(ax, ay, x_walls, y_walls, np.conj(tile), src["k0"],
                        src["kx0"], src["ky0"], src["ox"], src["oy"],
                        src["kxv"], src["kyv"], formulation, slant=slant,
                        block_eig=block_eig, keep=src["keep"])
    monkeypatch.undo()
    assert grabbed, "the spy never saw the hand-off"
    return grabbed


@pytest.mark.parametrize("branch", sorted(_BRANCHES))
@pytest.mark.parametrize("circular", [False, True])
def test_b6_the_cached_tensor_ops_reproduce_the_uncached_build_bit_for_bit(
        monkeypatch, branch, circular):
    """G10(d)'s gate: the assembly was MOVED into ``_tensor_projected_ops``, so
    every one of the seven operators (and the four out-of-plane blocks, and the
    block-eig gauge) must be ``np.array_equal`` between the path that rebuilds
    and the path that is handed a cached build.  Not ``allclose``: the whole
    point of splitting the k0-free half out is that the arithmetic is the same
    arithmetic in the same order.
    """
    x_walls, y_walls, tile, formulation, _kind = _BRANCHES[branch]
    src = _source(17.0, 33.0, circular=circular)
    fresh = _capture_operators(monkeypatch, x_walls, y_walls, tile,
                               formulation, src)
    ax = _build_axis(_PX, x_walls, _DEG2, 1, False)
    ay = _build_axis(_PY, y_walls, _DEG2, 1, False)
    ops = _tensor_projected_ops(ax, ay, x_walls, y_walls, np.conj(tile),
                                src["ox"], src["oy"], formulation)

    grabbed = {}
    orig = TJ._layer_eigenmodes_tensor

    def spy(GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ, slant=None, sym_gauge=None,
            **oop):
        for name, v in zip(_OPNAMES, (GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ)):
            grabbed[name] = np.array(v, copy=True)
        for k, v in oop.items():
            if v is not None:
                grabbed[k] = np.array(v, copy=True)
        return orig(GxF, GyF, Cxx, Cxy, Cyx, Cyy, EZZ, slant=slant,
                    sym_gauge=sym_gauge, **oop)

    monkeypatch.setattr(TJ, "_layer_eigenmodes_tensor", spy)
    _tensor_layer_modes(ax, ay, x_walls, y_walls, np.conj(tile), src["k0"],
                        src["kx0"], src["ky0"], src["ox"], src["oy"],
                        src["kxv"], src["kyv"], formulation, keep=src["keep"],
                        ops=ops)
    monkeypatch.undo()

    assert set(grabbed) == set(fresh), (
        f"{branch}: the cached path handed over a different operator SET "
        f"({sorted(set(grabbed) ^ set(fresh))})")
    for name in sorted(fresh):
        assert np.array_equal(grabbed[name], fresh[name]), (
            f"{branch} circular={circular}: {name} is not bit-identical "
            f"(max|d| = {np.max(np.abs(grabbed[name] - fresh[name])):.3e})")


def test_b6_a_supplied_ops_is_actually_consumed(monkeypatch):
    """Non-vacuity guard for the test above: a ``_tensor_layer_modes`` that
    ignored ``ops`` and rebuilt every time would pass a bit-identity check
    trivially.  Hand it a DELIBERATELY perturbed build and require the
    perturbation to come out the other side."""
    x_walls, y_walls, tile, formulation, _k = _BRANCHES["crossed"]
    src = _source(17.0, 33.0)
    ax = _build_axis(_PX, x_walls, _DEG2, 1, False)
    ay = _build_axis(_PY, y_walls, _DEG2, 1, False)
    ops = _tensor_projected_ops(ax, ay, x_walls, y_walls, np.conj(tile),
                                src["ox"], src["oy"], formulation)
    bent = dict(ops)
    bent["Cxx"] = ops["Cxx"] * 2.0
    bent["Gx0F"] = ops["Gx0F"] * 3.0

    grabbed = {}
    orig = TJ._layer_eigenmodes_tensor

    def spy(GxF, GyF, Cxx, *a, **kw):
        grabbed["GxF"] = np.array(GxF, copy=True)
        grabbed["Cxx"] = np.array(Cxx, copy=True)
        return orig(GxF, GyF, Cxx, *a, **kw)

    monkeypatch.setattr(TJ, "_layer_eigenmodes_tensor", spy)
    _tensor_layer_modes(ax, ay, x_walls, y_walls, np.conj(tile), src["k0"],
                        src["kx0"], src["ky0"], src["ox"], src["oy"],
                        src["kxv"], src["kyv"], formulation, keep=src["keep"],
                        ops=bent)
    monkeypatch.undo()
    assert np.array_equal(grabbed["Cxx"], ops["Cxx"] * 2.0), (
        "the supplied ops dict was ignored -- Cxx came back un-perturbed")
    expect = bent["Gx0F"] / src["k0"] + src["kx0"] * bent["IpxF"]
    assert np.array_equal(grabbed["GxF"], expect), (
        "GxF was not rebuilt from the supplied k0-free part")


@pytest.mark.parametrize("branch", sorted(_BRANCHES))
def test_b6_the_k0_free_split_is_the_algebra_it_claims(branch):
    """``kind`` is the contract that lets a cached build be re-sourced, so it
    is asserted directly rather than inferred from the answer: a wall-less axis
    carries NO k0-free part and is exactly ``diag(k)``; a patterned one is
    exactly ``Gx0F/k0 + kx0*IpxF``.  Getting ``kind`` wrong on a separable cell
    would put a ``diag(k)`` on the patterned axis -- a silently wrong layer
    whose energy still closes."""
    x_walls, y_walls, tile, formulation, kind = _BRANCHES[branch]
    src = _source(17.0, 33.0)
    ax = _build_axis(_PX, x_walls, _DEG2, 1, False)
    ay = _build_axis(_PY, y_walls, _DEG2, 1, False)
    ops = _tensor_projected_ops(ax, ay, x_walls, y_walls, np.conj(tile),
                                src["ox"], src["oy"], formulation)
    assert ops["kind"] == kind

    for axis, g0, ip, k0v, kt0 in (("x", "Gx0F", "IpxF", "kxv", "kx0"),
                                   ("y", "Gy0F", "IpyF", "kyv", "ky0")):
        patterned = axis in kind
        assert (ops[g0] is not None) is patterned, (
            f"{branch}: {g0} presence disagrees with kind={kind!r}")
        assert (ops[ip] is not None) is patterned
        if not patterned:
            continue
        built = ops[g0] / src["k0"] + src[kt0] * ops[ip]
        assert built.shape == (len(src["ox"]) * len(src["oy"]),) * 2

    nf = len(src["ox"]) * len(src["oy"])
    for name in ("Cxx", "Cxy", "Cyx", "Cyy", "EZZ"):
        assert ops[name].shape == (nf, nf)
    offp = ops["oop"]["EZX"] is not None
    assert offp == bool(np.max(np.abs(np.conj(tile)[..., [0, 1, 2, 2],
                                                   [2, 2, 0, 1]])) > 1e-12)
    assert all((v is not None) == offp for v in ops["oop"].values())


def _tensor_stack(cell, *, formulation="li", cache_max_bytes=None,
                  truncation="rectangular"):
    st = PMM2DStackHybrid(_PX, _PY, n_substrate=1.5, n_superstrate=1.0,
                          degree=_DEG2, n_orders=_NO2,
                          formulation=formulation, truncation=truncation,
                          cache_max_bytes=cache_max_bytes)
    st.add_layer(0.18e-6, eps_tensor_cell=cell)
    st.add_layer(0.09e-6, eps=2.25)
    return st


def _count_builds(monkeypatch, run):
    """``(n_tensor_assemblies, n_tensor_layer_modes_calls)`` for ``run()``."""
    n = [0, 0]
    orig_ops, orig_tlm = S2._tensor_projected_ops, S2._tensor_layer_modes

    def spy_ops(*a, **kw):
        n[0] += 1
        return orig_ops(*a, **kw)

    def spy_tlm(*a, **kw):
        n[1] += 1
        return orig_tlm(*a, **kw)

    monkeypatch.setattr(S2, "_tensor_projected_ops", spy_ops)
    monkeypatch.setattr(S2, "_tensor_layer_modes", spy_tlm)
    out = run()
    monkeypatch.undo()
    return tuple(n), out


def _count_every_assembly(monkeypatch, run):
    """Source-free tensor assemblies for ``run()``, counted at BOTH module
    bindings of ``_tensor_projected_ops``.

    ``stack2d`` and ``twod_jones`` hold SEPARATE names for that function, and
    the rebuild ``_tensor_layer_modes`` performs when it is handed no ``ops``
    goes through ``twod_jones``'s own global.  A spy installed only on
    ``stack2d`` cannot see it, so it cannot tell "the stack reused its cached
    build" from "the stack stored a build it never used and every sweep point
    rebuilt anyway" -- the two halves of G10(d).  Counting both bindings is
    what makes the saving falsifiable.
    """
    n = [0]
    for mod in (S2, TJ):
        orig = mod._tensor_projected_ops

        def spy(*a, _o=orig, **kw):
            n[0] += 1
            return _o(*a, **kw)

        monkeypatch.setattr(mod, "_tensor_projected_ops", spy)
    out = run()
    monkeypatch.undo()
    return n[0], out


_WLS = np.array([0.55e-6, 0.60e-6, 0.65e-6, 0.70e-6, 0.75e-6])
_THETAS = (0.0, 0.11, 0.23, 0.41, 0.55)


def _wl_sweep(st):
    out = []
    for wl in _WLS:
        st.set_source(float(wl), theta=0.21, phi=0.37)
        out.append(tuple(np.asarray(v) for v in st.solve()))
    return out


def _angle_sweep(st):
    out = []
    for th in _THETAS:
        st.set_source(_WL2, theta=float(th), phi=0.37)
        out.append(tuple(np.asarray(v) for v in st.solve()))
    return out


@pytest.mark.parametrize("cell,label", [(_SEPX, "separable"),
                                        (_CROSS, "crossed"),
                                        (_SEPX_O, "out-of-plane")])
@pytest.mark.parametrize("sweep", [_wl_sweep, _angle_sweep])
def test_b6_a_tensor_sweep_assembles_the_projected_operators_once(
        monkeypatch, cell, label, sweep):
    """The saving, stated as a deterministic BUILD COUNT rather than a clock
    (this box runs ~20 sibling agents; WP-A12 records three failed timing
    probes).  ``_tensor_layer_modes`` is still entered once per sweep point --
    the eig genuinely depends on the source -- while the source-free assembly
    now happens ONCE.

    Fail-before, in-process and two-sided: with the geometry cache priced out
    (``cache_max_bytes=1``) the count returns to one build per point, and the
    answers of the two arms are asserted BIT-IDENTICAL, which is the
    refuse-never-degrade contract the priced caches carry everywhere else."""
    npts = len(_WLS)
    (n_asm, n_tlm), warm = _count_builds(
        monkeypatch, lambda: sweep(_tensor_stack(cell)))
    assert n_tlm == npts, f"{label}: expected {npts} layer-mode builds"
    assert n_asm == 1, (
        f"{label}: the source-free tensor assembly ran {n_asm} times over a "
        f"{npts}-point sweep; the cache is not being hit")

    (n_asm_cold, n_tlm_cold), cold = _count_builds(
        monkeypatch,
        lambda: sweep(_tensor_stack(cell, cache_max_bytes=1)))
    assert (n_asm_cold, n_tlm_cold) == (npts, npts), (
        f"{label}: pricing the cache out must restore one assembly per point, "
        f"got {n_asm_cold}")

    for i, (a, b) in enumerate(zip(warm, cold)):
        for j, (x, y) in enumerate(zip(a, b)):
            assert np.array_equal(x, y), (
                f"{label}: sweep point {i}, return {j} is not bit-identical "
                f"with and without the cache "
                f"(max|d| = {np.max(np.abs(x - y)):.3e})")


@pytest.mark.parametrize("cell,label", [(_SEPX, "separable"),
                                        (_CROSS, "crossed"),
                                        (_SEPX_O, "out-of-plane")])
@pytest.mark.parametrize("sweep", [_wl_sweep, _angle_sweep])
def test_b6_the_stack_hands_its_cached_build_to_the_layer_modes_builder(
        monkeypatch, cell, label, sweep):
    """G10(d)'s saving, counted where the rebuild would actually happen.

    ``::test_b6_a_tensor_sweep_assembles_the_projected_operators_once`` counts
    ``stack2d``'s binding of ``_tensor_projected_ops``, which sees the
    assembly that FILLS the cache but not the one
    ``_tensor_layer_modes`` performs for itself when the stack hands it no
    ``ops``.  Dropping ``ops=tops`` at ``stack2d.py:1127`` therefore restores
    the pre-WP-B6 behaviour in full -- one stored-and-unused build plus one
    rebuild per sweep point -- while leaving that count at 1.  MEASURED
    2026-09-13 with exactly that one-word mutation applied to an isolated copy
    of the tree: all 44 tests stayed GREEN.  This one counts both bindings and
    reads 1 + 5 = 6 against the required 1.

    Fail-before (the same mutation, same run): 6 assemblies over a 5-point
    sweep.  Two-sided: the priced-out arm must return to one per point, and
    every return of every point must be bit-identical between the arms.
    """
    npts = len(_WLS)
    n_warm, warm = _count_every_assembly(
        monkeypatch, lambda: sweep(_tensor_stack(cell)))
    assert n_warm == 1, (
        f"{label}: the source-free tensor assembly ran {n_warm} times over a "
        f"{npts}-point sweep counting BOTH module bindings -- the stack is "
        f"caching a build that _tensor_layer_modes is not being given")

    n_cold, cold = _count_every_assembly(
        monkeypatch, lambda: sweep(_tensor_stack(cell, cache_max_bytes=1)))
    assert n_cold == npts, (
        f"{label}: pricing the cache out must restore one assembly per point, "
        f"got {n_cold}")

    for i, (a, b) in enumerate(zip(warm, cold)):
        for j, (x, y) in enumerate(zip(a, b)):
            assert np.array_equal(x, y), (
                f"{label}: sweep point {i}, return {j} is not bit-identical "
                f"with and without the cache")


def test_b6_one_entry_serves_a_truncation_FLIP_on_the_same_stack(monkeypatch):
    """The "one entry serves both truncations" property, asserted on ONE
    cache.

    ``truncation`` is deliberately absent from ``_geom_key`` because the
    cached build is the FULL order box and ``keep`` is applied at use.
    ``::test_b6_the_cached_entry_serves_both_truncations`` parametrises over
    the truncation but builds a FRESH stack for each value, so it never shares
    an entry between the two and stays green if ``truncation`` is added to the
    key.  MEASURED 2026-09-13 with
    ``self.truncation`` appended to ``_geom_key`` on an isolated copy of the
    tree: all 44 tests stayed GREEN.  Here one stack solves rectangular, has
    ``truncation`` flipped, and solves again; the assembly must still have run
    exactly ONCE.

    Fail-before (that mutation): 2 assemblies.  Non-vacuity: the two arms are
    required to have DIFFERENT order-set shapes, so "one entry served both" is
    a real claim rather than two identical solves.
    """
    st = _tensor_stack(_CROSS)

    def both():
        out = []
        for trunc in ("rectangular", "circular"):
            st.truncation = trunc
            st.set_source(_WL2, theta=0.21, phi=0.37)
            out.append(tuple(np.asarray(v) for v in st.solve()))
        return out

    n_asm, got = _count_every_assembly(monkeypatch, both)
    assert got[0][0].shape != got[1][0].shape, (
        f"the two truncations must retain DIFFERENT order sets for this "
        f"contract to mean anything; both gave {got[0][0].shape}")
    assert n_asm == 1, (
        f"a truncation flip on ONE stack rebuilt the projected tensor "
        f"operators {n_asm} times; the cached build is the full order box and "
        f"must serve both order-set shapes")

    for i, trunc in enumerate(("rectangular", "circular")):
        fresh = _tensor_stack(_CROSS, truncation=trunc)
        fresh.set_source(_WL2, theta=0.21, phi=0.37)
        ref = fresh.solve()
        for j, (x, y) in enumerate(zip(got[i], ref)):
            assert np.array_equal(np.asarray(x), np.asarray(y)), (
                f"{trunc}: return {j} from the flipped stack is not "
                f"bit-identical to a freshly built one")


@pytest.mark.parametrize("truncation", ["rectangular", "circular"])
def test_b6_the_cached_entry_serves_both_truncations(monkeypatch, truncation):
    """``keep`` is applied at USE, inside ``_tensor_layer_modes``, so the cached
    build is the FULL box and one entry serves both order-set shapes -- the
    same property ``_restrict_lops`` gives the scalar branch."""
    (n_asm, _n), warm = _count_builds(
        monkeypatch,
        lambda: _wl_sweep(_tensor_stack(_CROSS, truncation=truncation)))
    assert n_asm == 1
    (_a, _b), cold = _count_builds(
        monkeypatch,
        lambda: _wl_sweep(_tensor_stack(_CROSS, truncation=truncation,
                                        cache_max_bytes=1)))
    for a, b in zip(warm, cold):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_b6_the_geom_key_splits_on_the_formulation():
    """The cached entry now carries the projected TENSOR operators, whose
    ``EZZ`` is ``inv([[1/e_zz]])`` under ``'li'`` and the direct ``[[e_zz]]``
    otherwise -- so ``formulation`` became a solver parameter of the cached
    VALUE and had to join the key (the W7 A11 shape: plain public attributes
    with no property guard, mutated after a solve).

    Fail-before: the two rules are first shown to give genuinely different
    answers, so 'the mutated object agrees with a fresh one' cannot pass
    vacuously."""
    a = _tensor_stack(_CROSS, formulation="li")
    b = _tensor_stack(_CROSS, formulation="laurent")
    for st in (a, b):
        st.set_source(_WL2, theta=0.21, phi=0.37)
    ra, rb = a.solve(), b.solve()
    spread = max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                 for x, y in zip(ra[1:], rb[1:]))
    assert spread > 1.0e-6, (
        f"the two Ez rules must disagree for this contract to be testable; "
        f"measured {spread:.3e}")

    L = a._layers[0]
    ka = a._geom_key(L)
    a.formulation = "laurent"
    assert a._geom_key(L) != ka, "_geom_key does not carry the formulation"

    mutated = a.solve()
    for i, (x, y) in enumerate(zip(mutated, rb)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            f"return {i}: mutating formulation after a solve served the STALE "
            f"tensor operators (max|d| = "
            f"{np.max(np.abs(np.asarray(x) - np.asarray(y))):.3e})")


def test_b6_the_cached_tensor_operators_are_handed_out_read_only():
    """W7 A13's poisoning guard, extended to the new cache slot: the stack
    hands out the STORED arrays by identity, so a caller that writes into one
    would silently move every later solve on the same key."""
    st = _tensor_stack(_CROSS)
    st.set_source(_WL2, theta=0.21, phi=0.37)
    st.solve()

    def writeable(obj, out):
        if isinstance(obj, np.ndarray):
            if obj.flags.writeable:
                out.append(obj.shape)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                writeable(v, out)
        elif isinstance(obj, dict):
            for v in obj.values():
                writeable(v, out)

    bad = []
    for v in st._geom_cache.values():
        writeable(v, bad)
    assert bad == [], bad

    tops = [v[3] for v in st._geom_cache.values() if v[3] is not None]
    assert len(tops) == 1, "the tensor layer's operators were not cached"
    assert tops[0]["kind"] == "xy"
