"""WP-A13 (audit 2026-09-11) -- PMM2DStackHybrid: per-slot Li routing, cache-key
completeness, frozen periods, the eig-cache refusal signal, circular truncation.

Findings pinned here
--------------------
* **G5 (P1)** -- ``PMM2DStackHybrid`` never routed the per-slot Li operators
  ``EpnxF``/``EpnyF``, so under its DEFAULT ``formulation='li'`` a y-patterned
  layer got the y-axis inverse rule on the **Ex** slot and Laurent on Ey (BOTH
  slots anti-Li).  One physical grating drawn along x and along y then gave two
  different answers, and a one-layer stack disagreed with
  ``pmm_efficiency_2d_cell``, which is documented as the same physics.
* **G12 (P3)** -- ``symmetry`` was absent from ``_mode_key`` (a tensor layer's
  modal set is built by the parity-sign block reduction when it is on and by
  the dense 4Nf zgeev when it is off), and ``period_x``/``period_y`` were plain
  attributes although ``add_layer`` freezes the walls in METRES.
* **G10** -- ``cache_stats()['eig']['refused']`` was the only signal that a
  sweep had silently lost all modal reuse; circular truncation existed only on
  ``pmm_efficiency_2d[_cell]``.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStackHybrid,
    pmm_efficiency_2d_cell,
)

_WL = 1.0e-6
_P = 0.47e-6
_DEP = 0.3e-6
_NSUP, _NSUB = 1.0, 1.5
_S = 24


def _stripes():
    """One 1-D Si grating (eps 12.25 / 1.0, duty 1/2, walls at 1/4 and 3/4),
    drawn along x and the same array transposed -- a pure 90 deg rotation of
    ONE physical problem (the audit's p3b / q1b_stackli fixture)."""
    cx = np.full((_S, _S), 1.0 + 0j)
    cx[6:18, :] = 12.25
    return cx, cx.T.copy()


def _stack(cell, *, degree, n_orders, formulation, symmetry, **kw):
    st = PMM2DStackHybrid(_P, _P, n_superstrate=_NSUP, n_substrate=_NSUB,
                          degree=degree, n_orders=n_orders,
                          formulation=formulation, symmetry=symmetry, **kw)
    st.add_layer(_DEP, eps_cell=cell)
    st.set_source(_WL, theta=0.0, phi=0.0)
    return st.solve()


# --------------------------------------------------------------------------- #
# G5 -- the P1: per-slot Li routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("degree,n_orders", [(7, 5), (11, 5), (11, 9)])
@pytest.mark.parametrize("symmetry", ["auto", False])
def test_g5_li_is_90deg_rotation_invariant_on_the_stack(degree, n_orders,
                                                        symmetry):
    """One grating drawn along x vs along y must give the SAME answer.

    Oracle: the 90 deg rotation is an exact symmetry of Maxwell's equations, so
    ``T(E_x, x-patterned)`` and ``T(E_y, y-patterned)`` are the same number --
    nothing in the code produces the comparison.

    BAR 1e-11 on R/T and 1e-10 on the Jones.  Derivation: the two solves run
    different (transposed) matrices through LAPACK, so the floor is their
    independent round-off.  MEASURED post-fix over this grid -- worst
    ``max|dT| = 1.14e-13``, ``max|dR| = 1.13e-13``, ``|dJ| = 1.51e-12`` against
    ``max|T| ~ 0.75`` and ``|J| ~ 0.99``; the bars sit ~2 decades above that.
    MEASURED pre-fix on the SAME grid (the routing defect): ``max|dT| =
    4.98e-03`` at degree 7 / n_orders 5 and ``T00`` differing by ``1.23e-03``
    (5.3 % of T00) at degree 11 / n_orders 9, with ``|dJ| = 2.32e-02`` -- 8 to
    11 decades above the bars, so this test fails loudly on the pre-fix code.
    ``formulation='laurent'`` was symmetric to 1.07e-14 throughout, which is
    what isolates the cause to the per-slot routing rather than the grid.
    """
    cx, cy = _stripes()
    ox_, Rx, Tx, Jx = _stack(cx, degree=degree, n_orders=n_orders,
                             formulation="li", symmetry=symmetry)
    oy_, Ry, Ty, Jy = _stack(cy, degree=degree, n_orders=n_orders,
                             formulation="li", symmetry=symmetry)
    idx = {tuple(int(v) for v in row): i for i, row in enumerate(oy_)}
    perm = [idx[(int(n), int(m))] for m, n in ox_]
    assert np.max(np.abs(Tx[0] - Ty[1][perm])) < 1e-11
    assert np.max(np.abs(Rx[0] - Ry[1][perm])) < 1e-11
    assert abs(Jx[0, 0] - Jy[1, 1]) < 1e-10
    # the answer is non-trivial (guards against a degenerate all-zero pass)
    assert np.max(Tx) > 0.5


@pytest.mark.parametrize("n_orders", [5, 9])
def test_g5_one_layer_stack_equals_the_single_cell_entry(n_orders):
    """``PMM2DStackHybrid`` with ONE layer and ``pmm_efficiency_2d_cell`` are
    documented as the same physics; under ``formulation='li'`` they disagreed
    for the y-patterned orientation.

    BAR 1e-10.  Oracle: the two entry points build the SAME projected operators
    and the same one-layer cascade, so the only difference is LAPACK round-off
    on separately assembled matrices.  MEASURED post-fix: 3.8e-14 (n_orders 5)
    and 8.2e-13 (n_orders 9).  MEASURED pre-fix on the y-patterned orientation:
    ``T00`` 0.0322424723 (stack) against 0.0334736588 (cell entry), i.e.
    1.23e-03 -- 7 decades above the bar.
    """
    cx, cy = _stripes()
    for cell, col, pol in ((cx, 0, "tm"), (cy, 1, "te")):
        ref = pmm_efficiency_2d_cell(_P, _P, cell, _NSUB, _NSUP, _DEP, _WL,
                                     polarization=pol, degree=11,
                                     n_orders=n_orders, formulation="li")
        o_, R_, T_, _J = _stack(cell, degree=11, n_orders=n_orders,
                                formulation="li", symmetry="auto")
        idx = {tuple(int(v) for v in row): i for i, row in enumerate(o_)}
        perm = [idx[tuple(int(v) for v in row)] for row in ref[0]]
        assert np.max(np.abs(np.asarray(ref[2]) - T_[col][perm])) < 1e-10


def test_g5_the_routed_pair_is_what_the_stack_now_passes():
    """Structural gate on the fix itself: ``_layer_modes_projected`` must be
    reached with BOTH per-slot operators, and they must be the ones
    ``_scalar_projected_ops`` routed -- not a single ``EpnF``.

    Without this, a future refactor could restore 90 deg invariance by
    accident (e.g. by falling back to ``laurent``) and the numeric tests above
    would still pass.
    """
    import lumenairy.elements.pmm.stack2d as s2
    from lumenairy.elements.pmm.twod import (
        _build_axis,
        _layer_modes_projected,
        _scalar_projected_ops,
    )
    seen = {}
    real = _layer_modes_projected

    def spy(GxF, GyF, EpsF, EinvF, EpnF, **kw):
        seen.update(kw)
        return real(GxF, GyF, EpsF, EinvF, EpnF, **kw)

    cx, cy = _stripes()
    s2._layer_modes_projected = spy
    try:
        _stack(cy, degree=7, n_orders=3, formulation="li", symmetry=False)
    finally:
        s2._layer_modes_projected = real
    assert seen.get("EpnxF") is not None and seen.get("EpnyF") is not None
    # on a y-patterned cell the INVERSE rule belongs on Ey and Laurent on Ex
    ax = _build_axis(_P, [], 7, [1], False)
    ay = _build_axis(_P, [_P / 4, 3 * _P / 4], 7, [1, 1, 1], False)
    tile = np.array([[1.0 + 0j, 12.25 + 0j, 1.0 + 0j]])
    o3 = np.arange(-3, 4)
    lops = _scalar_projected_ops(ax, ay, tile, o3, o3, _P, _P)
    assert np.array_equal(lops["EpnxF"], lops["EpsF"])      # Ex keeps Laurent
    assert not np.array_equal(lops["EpnyF"], lops["EpsF"])  # Ey is inverse-rule


# --------------------------------------------------------------------------- #
# G12 -- cache-key completeness and the frozen periods
# --------------------------------------------------------------------------- #

def _oop_tensor_cell(S=12):
    t = np.zeros((S, S, 3, 3), dtype=complex)
    t[...] = 2.25 * np.eye(3)
    blk = np.array([[4.0, 0.0, 0.6], [0.0, 3.4, 0.0], [0.6, 0.0, 3.9]],
                   dtype=complex)
    t[3:9, 3:9] = blk
    return t


def test_g12_symmetry_is_in_the_modal_cache_key():
    """Mutating ``symmetry`` between solves must NOT serve the stale modal set.

    ``_build_layer_modes`` passes ``block_eig=self.symmetry`` to
    ``_tensor_layer_modes``, so an out-of-plane tensor layer's modal set is a
    DIFFERENT computation at the two settings (one 2Nf parity-sign eig vs the
    dense 4Nf zgeev).  Oracle: a FRESH object at the new setting.

    BAR 1e-13.  The parity-sign reduction is exact, so a correct reuse-vs-fresh
    difference is machine noise; the defect is not a magnitude but an IDENTITY
    -- pre-fix the mutated object returned an answer BIT-IDENTICAL to the
    pre-mutation one (``|J_b - J_a| = 0.0`` exactly) while a fresh
    ``symmetry=False`` object differed by 7.16e-16, so the assertion that
    catches it is ``J_b != J_a`` combined with ``J_b == J_fresh``.
    """
    P, Py, dep = 0.9e-6, 0.9e-6, 0.3e-6
    t = _oop_tensor_cell()

    def mk(sym):
        st = PMM2DStackHybrid(P, Py, n_superstrate=1.0, n_substrate=1.45,
                              degree=7, n_orders=3, symmetry=sym)
        st.add_layer(dep, eps_tensor_cell=t)
        st.set_source(_WL, theta=0.0, phi=0.0)
        return st

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh_false = mk(False).solve()[3]
        st = mk(True)
        j_true = st.solve()[3]
        st.symmetry = False
        j_reused = st.solve()[3]
    # the mutation is HONOURED (pre-fix this was bit-identical to j_true)
    assert not np.array_equal(np.asarray(j_reused), np.asarray(j_true))
    # ... and lands on the fresh object's answer
    assert np.max(np.abs(np.asarray(j_reused) - np.asarray(fresh_false))) < 1e-13


def test_g12_periods_are_frozen_after_the_first_layer():
    """``add_layer`` converts the walls to METRES, so a later period change
    would re-solve the OLD walls at a NEW period -- a silently different duty
    cycle, measured ``|J_reused - J_fresh(same duty)| = 4.390e-01`` against the
    parameter's true sensitivity 6.492e-01.  It must refuse instead."""
    st = PMM2DStackHybrid(0.9e-6, 0.9e-6, degree=7, n_orders=3)
    st.period_x = 1.0e-6                      # before any layer: allowed
    assert st.period_x == pytest.approx(1.0e-6)
    cell = np.full((12, 12), 1.0 + 0j)
    cell[3:9, 3:9] = 12.25
    st.add_layer(0.3e-6, eps_cell=cell)
    for name in ("period_x", "period_y"):
        with pytest.raises(ValueError, match="FROZEN once a layer"):
            setattr(st, name, 1.05e-6)
    assert st.period_x == pytest.approx(1.0e-6)   # unchanged by the refusal


# --------------------------------------------------------------------------- #
# G10 -- the eig-cache refusal signal, and circular truncation
# --------------------------------------------------------------------------- #

def test_g10_eig_cache_refusal_is_surfaced():
    """At budget the modal cache REFUSES rather than degrades: the answer is
    returned but not retained, so every later solve re-eigs.  Nothing said so;
    ``cache_stats()['eig']['refused']`` was the only signal."""
    P = 0.9e-6
    cell = np.full((8, 8), 1.0 + 0j)
    cell[2:6, 2:6] = 12.25
    # OBLIQUE: at normal incidence on a centro-symmetric cell the even-parity
    # fold bypasses the per-layer modal cache entirely, so there is nothing to
    # refuse.
    st = PMM2DStackHybrid(P, P, n_superstrate=1.0, n_substrate=1.45, degree=7,
                          n_orders=3, cache_max_bytes=1)
    st.add_layer(0.3e-6, eps_cell=cell)
    st.set_source(_WL, theta=0.2, phi=0.3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = st.solve()
    assert st.cache_stats()["eig"]["refused"] > 0
    msgs = [str(x.message) for x in w
            if "modal (eig) cache REFUSED" in str(x.message)]
    assert len(msgs) == 1, msgs
    assert "ANSWER IS UNCHANGED" in msgs[0]
    # ... once per instance, not once per solve (a sweep must not be flooded)
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        b = st.solve()
    assert not [x for x in w2 if "modal (eig) cache REFUSED" in str(x.message)]
    # refuse-never-degrade: the ANSWER is unaffected by the budget
    st_ok = PMM2DStackHybrid(P, P, n_superstrate=1.0, n_substrate=1.45,
                             degree=7, n_orders=3)
    st_ok.add_layer(0.3e-6, eps_cell=cell)
    st_ok.set_source(_WL, theta=0.2, phi=0.3)
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter("always")
        c = st_ok.solve()
    assert st_ok.cache_stats()["eig"]["refused"] == 0
    assert not [x for x in w3 if "modal (eig) cache REFUSED" in str(x.message)]
    assert np.array_equal(np.asarray(a[2]), np.asarray(c[2]))
    assert np.array_equal(np.asarray(b[2]), np.asarray(c[2]))


def test_g10_circular_truncation_reaches_the_stack():
    """Lalanne-1997 circular truncation existed only on
    ``pmm_efficiency_2d[_cell]``.  On the stack it must retain the same order
    set the single-cell entry retains, and agree with it.

    BAR 1e-10 on the per-order efficiencies: the stack and the single-cell
    entry build the same operators and the same one-layer cascade, so the
    floor is LAPACK round-off (measured 3.9e-14 at normal incidence).  The
    retained-order count is an exact integer identity.
    """
    P, dep = 0.9e-6, 0.3e-6
    cell = np.full((8, 8), 1.0 + 0j)
    cell[2:6, 2:6] = 12.25
    got = {}
    for tr in ("rectangular", "circular"):
        st = PMM2DStackHybrid(P, P, n_superstrate=1.0, n_substrate=1.45,
                              degree=9, n_orders=5, formulation="laurent",
                              truncation=tr)
        st.add_layer(dep, eps_cell=cell)
        st.set_source(_WL, theta=0.0, phi=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got[tr] = st.solve()
        ref = pmm_efficiency_2d_cell(P, P, cell, 1.45, 1.0, dep, _WL,
                                     polarization="tm", degree=9, n_orders=5,
                                     formulation="laurent", truncation=tr)
        assert len(got[tr][0]) == len(ref[0])
        idx = {tuple(int(v) for v in row): i for i, row in enumerate(got[tr][0])}
        perm = [idx[tuple(int(v) for v in row)] for row in ref[0]]
        assert np.max(np.abs(np.asarray(ref[2]) - got[tr][2][0][perm])) < 1e-10
    # the circle strictly drops the box corners: (2n+1)^2 = 121 -> 81 here
    assert len(got["circular"][0]) == 81
    assert len(got["rectangular"][0]) == 121


def test_g10_rectangular_truncation_is_untouched():
    """The circular option must be a pure ADDITION: at the default the stack
    is bit-for-bit what it always was.

    Stated as an IN-PROCESS INVARIANCE rather than as a literal ``T00``
    (VERIFY-A13, TESTING_STANDARDS durability rule: nothing may pin a prior
    version's number, and an eigendecomposition-derived efficiency is exactly
    the quantity a BLAS build is entitled to move in its last bits).  Three
    claims, none of them a tolerance:

    1. the order set is the full rectangular box, ``(2n+1)^2``;
    2. the default stack is ``np.array_equal`` to the same stack built with
       ``truncation='rectangular'`` SPELLED OUT -- so the option's presence
       changes no bit of the path it defaults to;
    3. the F8 code really is reachable on this fixture, i.e. ``'circular'``
       retains strictly fewer orders -- otherwise (1) and (2) would hold
       vacuously on a build where the option did nothing at all.
    """
    cx, _cy = _stripes()
    o_d, R_d, T_d, J_d = _stack(cx, degree=11, n_orders=5, formulation="li",
                                symmetry="auto")
    o_r, R_r, T_r, J_r = _stack(cx, degree=11, n_orders=5, formulation="li",
                                symmetry="auto", truncation="rectangular")
    assert len(o_d) == (2 * 5 + 1) ** 2
    assert np.array_equal(np.asarray(o_d), np.asarray(o_r))
    assert np.array_equal(np.asarray(R_d), np.asarray(R_r))
    assert np.array_equal(np.asarray(T_d), np.asarray(T_r))
    assert np.array_equal(np.asarray(J_d), np.asarray(J_r))
    o_c, _Rc, _Tc, _Jc = _stack(cx, degree=11, n_orders=5, formulation="li",
                                symmetry="auto", truncation="circular")
    assert len(o_c) < len(o_d)
    # non-degenerate answer, so the identity above is not an all-zero pass
    assert np.max(T_d) > 0.5
