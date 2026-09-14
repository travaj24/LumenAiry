"""WP-B5 regression gates -- the three RCWA items WP-A14 deferred with designs
(``fixes/WP-A14_REPORT.md`` section 6, D1 / D2 / D3).

Every numeric bar below carries its ORACLE, the oracle's own error floor, the
MEASURED value on this build and the decades of gap on both sides
(``docs/TESTING_STANDARDS.md`` S1-S5).  Nothing here asserts a wall clock: the
performance claims are gated as OPERATION COUNTS (how many guarded inverses a
solve records) and the report carries the medians.

Two oracles are written out in this file rather than imported, so no gate can
be satisfied by the code it tests:

* :func:`_star_oracle` -- the Redheffer composition from its DEFINING coupled
  system: the internal up/down amplitudes of the gap solved as one dense
  ``2n`` system, with no star algebra, no push-through identity and no
  block inverse.
* the x<->y MIRROR itself -- a symmetry of Maxwell's equations, not of any
  implementation: mirror the cell (pixel grid transposed AND the tensor
  components permuted ``(x, y, z) -> (y, x, z)``) and the Jones matrix must
  come back conjugated by ``P = [[0, 1], [1, 0]]``.

The third oracle is the library's own Berreman 4x4 (``elements/berreman.py``),
used only where its independence is exact: it carries NO Fourier factorization
at all, which is the entire object of D3.

Items covered
-------------
D1  the two genuinely Toeplitz inverses of the 1-D solve (``oned.py:135`` and
    ``oned.py:709``).  MEASURED and REFUSED -- ``scipy.linalg.solve_toeplitz``
    costs 12-20x the explicit inverse at the sizes these sites run and lands
    two decades further from the equation it solves.  The gates here pin the
    two facts the decision rests on, so a future build that changes either
    re-opens it.
D2  the two-interface closed form: the last star of a single-layer cascade is
    applied to the source instead of assembled, which is one guarded inverse
    instead of two and one ``2N`` product instead of twelve.
D3  the OFF-PLANE (full 3x3) ``fff_nv`` operator is symmetrized over the two
    factorization orders, as the in-plane 2x2 one already was (H3).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import inspect
import warnings

import numpy as np
import pytest
import scipy.linalg as sla

import lumenairy.elements.rcwa._core as _rc
import lumenairy.elements.rcwa.twod as _twod
from lumenairy.elements.berreman import berreman_jones_1d
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_2d,
    uniaxial_tensor,
)
from lumenairy.elements.rcwa._core import (
    _guarded_inverse,
    _redheffer_star,
    _redheffer_star_rt,
    _toeplitz_1d,
)
from lumenairy.elements.rcwa.oned import _binary_step_coeffs

_C = np.complex128

#: Ag and Au at 633 nm (Johnson-Christy), the metals of the convergence ladder
#: WP-A14 section 6 named for D1 and D2.
_AG = 0.135 + 3.99j
_AU = 0.1834 + 3.4332j

#: The 1-D metallic fixture every D1 / D2 gate below runs on.
_LADDER = dict(period=1.0e-6, n_groove=1.0, n_substrate=1.5,
               n_superstrate=1.0, depth=0.25e-6, duty_cycle=0.5,
               wavelength=0.633e-6, formulation="li")


# ===========================================================================
# Oracle 1: the Redheffer composition from its defining coupled system
# ===========================================================================

def _star_oracle(SA, SB, cinc):
    """``(S11 c, S21 c)`` of the composition of ``SA`` (region 1 | 2) and
    ``SB`` (region 2 | 3), from the DEFINITION rather than the star formula.

    With ``u`` the down-going and ``v`` the up-going amplitude in the gap and
    no drive from below::

        u = A21 c + A22 v          v = B11 u
        out_up = A11 c + A12 v     out_down = B21 u

    which is the dense ``2n`` system ``[[I, -A22], [-B11, I]] [u; v] =
    [A21 c; 0]``.  Solving it whole shares no algebra with
    :func:`_redheffer_star` (no block inverse, no push-through identity) and
    no code with :func:`_redheffer_star_rt`.
    """
    A11, A12, A21, A22 = SA
    B11, _B12, B21, _B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=_C)
    b = A21 @ cinc                        # (n,) or (n, k)
    rhs = np.concatenate([b, np.zeros_like(b)], axis=0)
    uv = np.linalg.solve(np.block([[I, -A22], [-B11, I]]), rhs)
    u, v = uv[:n], uv[n:]
    return A11 @ cinc + A12 @ v, B21 @ u


def _random_smatrices(n, seed, scale=0.3):
    rng = np.random.default_rng(seed)

    def mk():
        return scale * (rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    return tuple(mk() for _ in range(4)), tuple(mk() for _ in range(4))


def _capture_chain(pol="tm", M=50, metal=_AG, **kw):
    """The (SA, SB) pair the 1-D single-layer solve hands to its last star."""
    import lumenairy.elements.rcwa.oned as _oned
    cap = []
    orig = _oned._redheffer_star_rt

    def spy(SA, SB, cinc):
        cap.append((SA, SB, cinc))
        return orig(SA, SB, cinc)

    _oned._redheffer_star_rt = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_efficiency_1d(n_ridge=metal, polarization=pol, n_orders=M,
                               **{**_LADDER, **kw})
    finally:
        _oned._redheffer_star_rt = orig
    assert cap, "the 1-D single-layer solve no longer routes through the "\
                "two-interface closed form"
    return cap[-1]


# ===========================================================================
# D2 -- the two-interface closed form
# ===========================================================================

#: D2's agreement bar against the independent star oracle.
#:
#: ORACLE FLOOR.  The oracle is one dense ``2n`` LU solve, so its own backward
#: error is ``O(eps * cond)``; the ASSEMBLED star -- the shipped
#: :func:`_redheffer_star`, which this file also runs against the oracle --
#: reads 1.2e-15 / 4.4e-15 / 3.9e-14 relative on the three random pairs and
#: 1.6e-16 .. 5.3e-16 on the captured metallic chains, and that reading IS the
#: floor.  MEASURED for the closed form on the same operands: 1.6e-15 /
#: 4.6e-15 / 1.6e-14 and 1.1e-16 .. 2.6e-16 -- the same floor, ratio
#: 0.24 .. 1.29.  BAR 1e-9, five decades above the worst of either.  The gap
#: above the bar is not asserted from a guess: the second claim in each test
#: is a RATIO between the two formulations, so the gate fails if the closed
#: form ever drifts three decades off the shipped star whatever the absolute
#: numbers do.
_D2_ORACLE_BAR = 1e-9


@pytest.mark.parametrize("n,seed", [(12, 1), (24, 5), (48, 9)])
def test_d2_closed_form_matches_the_independent_star_oracle(n, seed):
    """D2.  ORACLE: the defining coupled system, solved whole (``_star_oracle``
    -- no star algebra, no push-through identity).

    Both the assembled star and the closed form must reproduce it, and the
    closed form must not be the worse of the two by more than the oracle's own
    floor: the claim is that folding the star onto the source is a
    re-association, not an approximation.
    """
    SA, SB = _random_smatrices(n, seed)
    rng = np.random.default_rng(seed + 100)
    c = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    r_or, t_or = _star_oracle(SA, SB, c)
    scale = max(float(np.max(np.abs(r_or))), float(np.max(np.abs(t_or))))
    C = _redheffer_star(SA, SB)
    r_as, t_as = C[0] @ c, C[2] @ c
    r_cf, t_cf = _redheffer_star_rt(SA, SB, c)
    d_as = float(max(np.max(np.abs(r_as - r_or)),
                     np.max(np.abs(t_as - t_or)))) / scale
    d_cf = float(max(np.max(np.abs(r_cf - r_or)),
                     np.max(np.abs(t_cf - t_or)))) / scale
    assert d_as < _D2_ORACLE_BAR, (n, seed, d_as)
    assert d_cf < _D2_ORACLE_BAR, (n, seed, d_cf)
    # neither formulation may be more than 3 decades worse than the other --
    # measured ratio 1.29 / 1.05 / 0.40 over these three sizes.
    assert d_cf < 1e3 * max(d_as, 1e-17), (d_cf, d_as)


@pytest.mark.parametrize("pol,M", [("te", 11), ("tm", 11), ("tm", 50),
                                   ("tm", 100)])
def test_d2_closed_form_matches_the_oracle_on_the_metallic_chain(pol, M):
    """D2 on the operands it actually runs on: the ``(SA, SB)`` pair a 1-D Ag
    solve hands to its last star, whose ``A22`` carries the evanescent
    propagator's 300-decade dynamic range.

    Same oracle, same bar.  MEASURED: assembled 1.6e-16 .. 5.3e-16, closed
    form 1.1e-16 .. 2.6e-16 (relative), i.e. the two are indistinguishable at
    the oracle's floor, more than six decades inside the bar.
    """
    SA, SB, cinc = _capture_chain(pol=pol, M=M)
    r_or, t_or = _star_oracle(SA, SB, cinc)
    scale = max(float(np.max(np.abs(r_or))), float(np.max(np.abs(t_or))))
    C = _redheffer_star(SA, SB)
    r_as, t_as = C[0] @ cinc, C[2] @ cinc
    r_cf, t_cf = _redheffer_star_rt(SA, SB, cinc)
    d_as = float(max(np.max(np.abs(r_as - r_or)),
                     np.max(np.abs(t_as - t_or)))) / scale
    d_cf = float(max(np.max(np.abs(r_cf - r_or)),
                     np.max(np.abs(t_cf - t_or)))) / scale
    assert d_as < _D2_ORACLE_BAR and d_cf < _D2_ORACLE_BAR, (d_as, d_cf)
    assert d_cf < 1e3 * max(d_as, 1e-17), (d_cf, d_as)


def test_d2_a_single_layer_solve_records_one_star_inverse_not_two():
    """D2, as an OPERATION COUNT (never a wall clock).

    The census the M1 guard carries is the instrument: with ``_INV_CENSUS``
    armed, a 1-D single-layer solve must record exactly three guarded
    inverses -- the two interface mode-matches and ONE star denominator,
    ``I - B11 A22``, under the same site string :func:`_redheffer_star` uses.

    FAIL-BEFORE, demonstrated in this test rather than quoted: assembling the
    same star records TWO star denominators.
    """
    prev = _rc._INV_CENSUS
    try:
        _rc._INV_CENSUS = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_efficiency_1d(n_ridge=_AG, polarization="tm", n_orders=21,
                               **_LADDER)
        sites = [row[0] for row in _rc._INV_CENSUS]
    finally:
        _rc._INV_CENSUS = prev
    assert sites.count("rcwa Redheffer star (I - B11 A22)") == 1, sites
    assert sites.count("rcwa Redheffer star (I - A22 B11)") == 0, sites
    assert sites.count("rcwa interface mode-match (a+b)") == 2, sites
    assert len(sites) == 3, sites
    # fail-before: the assembled star on the same operands pays both
    SA, SB, _c = _capture_chain(pol="tm", M=21)
    try:
        _rc._INV_CENSUS = []
        _redheffer_star(SA, SB)
        star_sites = [row[0] for row in _rc._INV_CENSUS]
    finally:
        _rc._INV_CENSUS = prev
    assert sorted(star_sites) == ["rcwa Redheffer star (I - A22 B11)",
                                  "rcwa Redheffer star (I - B11 A22)"], \
        star_sites


@pytest.mark.parametrize("M", [11, 50, 100])
def test_d2_the_retained_denominator_is_the_tighter_reading(M):
    """D2.  Dropping ``I - A22 B11`` from the census may not hide a
    conditioning failure.

    The two denominators are SIMILAR (``(I - A22 B11) A22 = A22 (I - B11
    A22)``), so they share a spectrum; what the census actually reads is the
    EQUILIBRATED ``rcond``, which similarity does not preserve.  The claim
    gated here is the one that matters: the RETAINED matrix is never the
    looser reading by more than the instrument's own resolution.  MEASURED on
    this ladder, retained / dropped: 0.340 / 0.523, 0.167 / 0.670, 0.108 /
    0.671 -- the retained one is 1.5x to 6.2x TIGHTER, and it is the one that
    keeps falling as the truncation grows.
    """
    SA, SB, _c = _capture_chain(pol="tm", M=M)
    A22, B11 = SA[3], SB[0]
    n = A22.shape[0]
    I = np.eye(n, dtype=_C)
    X, Y = I - B11 @ A22, I - A22 @ B11
    rx = _rc._rcond_1_equilibrated(X, np.linalg.inv(X))
    ry = _rc._rcond_1_equilibrated(Y, np.linalg.inv(Y))
    # 1.05 admits the instrument's own last-bit spread; measured ratios are
    # 1.54 / 4.01 / 6.21 in this direction and never the other way.
    assert rx <= 1.05 * ry, (M, rx, ry)


def test_d2_zero_block_shortcuts_take_no_inverse_at_all():
    """D2.  The two zero-block shapes :func:`_redheffer_star` shortcuts must
    shortcut here too -- a chain that pays no star inverse there may not start
    paying one here, or the census of every propagation-shaped star would
    change.  The census staying EMPTY is the claim; it is an operation count,
    not a clock.

    The values are gated as well, at two different strengths, because only one
    of them is an identity.  With ``B11`` the exact zero block, ``A12 @ B11``
    is exactly zero and ``A11 + 0`` is exactly ``A11``, so ``S11 c`` is the
    SAME floating-point expression both ways: tolerance-at-0.0 (the standing
    rule -- never ``array_equal``).  ``S21 c`` is ``B21 @ (A21 @ c)`` here
    against ``(B21 @ A21) @ c`` there -- a re-association, not an identity --
    and is gated relatively: MEASURED 2.8e-16 of ``max|t|`` (1.8e-15 absolute
    on a ``|t|`` of 6.6), against a 1e-12 bar four decades above it.
    """
    n = 16
    SA, SB = _random_smatrices(n, 3)
    rng = np.random.default_rng(77)
    c = rng.standard_normal((n,)) + 1j * rng.standard_normal((n,))
    Z = np.zeros((n, n), dtype=_C)
    prev = _rc._INV_CENSUS
    for tag, (A, B) in (("B11=0", (SA, (Z, SB[1], SB[2], SB[3]))),
                        ("A22=0", ((SA[0], SA[1], SA[2], Z), SB))):
        C = _redheffer_star(A, B)
        r_as, t_as = C[0] @ c, C[2] @ c
        try:
            _rc._INV_CENSUS = []
            r_cf, t_cf = _redheffer_star_rt(A, B, c)
            rows = list(_rc._INV_CENSUS)
        finally:
            _rc._INV_CENSUS = prev
        assert rows == [], (tag, rows)
        if tag == "B11=0":
            assert float(np.max(np.abs(r_cf - r_as))) == 0.0, tag
        for got, want in ((r_cf, r_as), (t_cf, t_as)):
            scale = max(float(np.max(np.abs(want))), 1e-300)
            assert float(np.max(np.abs(got - want))) < 1e-12 * scale, tag


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_d2_energy_closes_through_the_closed_form(pol):
    """D2.  A LOSSLESS dielectric grating must still close.

    ORACLE: the conservation theorem, whose only floor is the solve's own
    arithmetic.  BAR 1e-11 on ``|sum R + sum T - 1|``; MEASURED 4.4e-16 ..
    2.2e-13 over ``n_orders`` 11 / 51 / 101 x both polarizations -- 1.7
    decades inside, and the quantity a broken composition moves by O(1).
    """
    kw = dict(_LADDER)
    kw.update(n_ridge=2.04, formulation="laurent", n_substrate=1.0)
    for M in (11, 51, 101):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T = rcwa_efficiency_1d(polarization=pol, n_orders=M, **kw)
        assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 1e-11, (pol, M)


def test_d2_the_multi_layer_even_fold_still_assembles_its_star():
    """D2's SCOPE, pinned as a decision.

    ``_symmetric_cascade_rt`` is imported by ``elements/pmm/stack2d.py`` and
    ``pmm/twod_jones.py``, which are another work package's files: closing its
    last star on the sources would move the PMM engines' last bits from inside
    the RCWA package.  It therefore keeps the assembled star deliberately, and
    this test fails if that decision is silently reversed.  Its single-layer
    twin ``_symmetric_solve_rt`` has no such caller and does use the closed
    form.
    """
    cascade = inspect.getsource(_rc._symmetric_cascade_rt)
    assert "_redheffer_star_rt(" not in cascade
    assert "_redheffer_star(" in cascade
    single = inspect.getsource(_rc._symmetric_solve_rt)
    assert "_redheffer_star_rt(" in single


# ===========================================================================
# D3 -- the off-plane (3x3) fff_nv operator
# ===========================================================================

_D3 = dict(period=0.5e-6, depth=0.3e-6, wl=0.633e-6, S=96)
_PERM = [1, 0, 2]
_P2 = np.array([[0.0, 1.0], [1.0, 0.0]])
#: A uniaxial pillar in AIR (n_o = 2.0, n_e = 2.6): the contrast the
#: factorization-order artefact scales with.  ``phi = 45 deg`` puts the
#: director in the x = y plane, so the cell is its own x<->y mirror.
_DIAG = uniaxial_tensor(2.0, 2.6, np.deg2rad(40.0), phi=np.deg2rad(45.0))
_SKEW = uniaxial_tensor(2.0, 2.6, np.deg2rad(40.0), phi=np.deg2rad(20.0))


def _mirror(cell):
    """The x<->y mirror image of a tensor cell: pixel grid transposed AND the
    component labels permuted ``(x, y, z) -> (y, x, z)``."""
    return np.transpose(cell, (1, 0, 2, 3))[:, :, _PERM, :][:, :, :, _PERM]


def _cell(kind, tilt, back=1.0 + 0j):
    S = _D3["S"]
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    c = np.zeros((S, S, 3, 3), dtype=complex)
    c[:] = np.eye(3) * back
    if kind == "square":
        m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 5)
    elif kind == "rect":
        m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 8)
    elif kind == "stripe":
        m = np.abs(ii - S // 2) < S // 5
    elif kind == "uniform":
        m = np.ones((S, S), dtype=bool)
    else:
        m = ((ii - (S - 1) / 2.0) ** 2
             + (jj - (S - 1) / 2.0) ** 2) <= (0.25 * S) ** 2
    c[m] = tilt
    return c


def _oop_jones(cell, M, symmetrize=True, theta=0.0, phi=0.0):
    """``rcwa_jones_2d`` on the OFF-PLANE ``fff_nv`` branch, with the private
    ``symmetrize`` switch reachable so the fail-before is measured here."""
    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rcwa_jones_2d(_D3["period"], _D3["period"], cell, 1.5, 1.0,
                                 _D3["depth"], _D3["wl"], theta=theta,
                                 phi=phi, n_orders_x=M, n_orders_y=M,
                                 formulation="fff_nv")[3]
    if symmetrize:
        return run()
    orig = _twod._li_convolutions_2d_tensor_full
    _twod._li_convolutions_2d_tensor_full = (
        lambda *a, **k: orig(*a, **{**k, "symmetrize": False}))
    try:
        return run()
    finally:
        _twod._li_convolutions_2d_tensor_full = orig


#: D3's symmetry bar.  ORACLE FLOOR: this path's own arithmetic, measured on
#: the three arms of this file where the factorization ORDER cannot enter at
#: all -- the off-diagonal probe (5.4e-16 .. 1.6e-14), the y-uniform stripe
#: where the two orders coincide analytically (7.8e-15 .. 2.6e-14), and the
#: Berreman arm on a laterally uniform cell (1.1e-15 .. 7.1e-15).  MEASURED
#: after: 2.3e-15 .. 1.6e-14 across both cells, M = 3..6 and both probes --
#: the same floor.  BAR 1e-12: 1.8 decades above the worst of any of those,
#: and eight below the 2.0e-04 .. 5.0e-03 the single-order operator puts there
#: (asserted as the fail-before, so the gap is measured on both sides).
_D3_SYM_BAR = 1e-12


@pytest.mark.parametrize("kind,pre_fix_floor", [("square", 2e-04),
                                                ("disk", 1e-03)])
def test_d3_offplane_fff_nv_keeps_the_cells_own_mirror(kind, pre_fix_floor):
    """D3.  ORACLE: the x<->y MIRROR of Maxwell's equations.

    The pattern is its own transpose and the director lies in the x = y plane,
    so the whole problem is invariant under the mirror and ``J = P J P`` --
    ``Jxx == Jyy`` and ``Jxy == Jyx`` EXACTLY at normal incidence.  There is no
    physics in the difference, only the factorization order.

    FAIL-BEFORE, measured here through the private ``symmetrize=False``
    switch: 9.10e-04 (square) and 4.71e-03 (disk) at M = 3, falling as ~1/M to
    3.32e-04 and 2.35e-03 at M = 6 -- on a Jones matrix whose own scale is
    0.11 .. 0.23, i.e. 0.4 % .. 4 % of spurious form birefringence on a cell
    that has NONE.
    """
    cell = _cell(kind, _DIAG)
    assert np.array_equal(cell, _mirror(cell)), \
        "fixture is not its own x<->y mirror"
    for M in (3, 4, 5, 6):
        J = _oop_jones(cell, M)
        assert abs(J[0, 0] - J[1, 1]) < _D3_SYM_BAR, (kind, M, J)
        assert abs(J[0, 1] - J[1, 0]) < _D3_SYM_BAR, (kind, M, J)
    J_old = _oop_jones(cell, 3, symmetrize=False)
    assert abs(J_old[0, 0] - J_old[1, 1]) > pre_fix_floor, (
        f"{kind}: the single-order 3x3 operator no longer breaks the mirror "
        f"-- fixture stale")


@pytest.mark.parametrize("kind,pre_fix_floor", [("rect", 1e-04),
                                                ("disk", 1e-03)])
def test_d3_offplane_fff_nv_is_mirror_covariant(kind, pre_fix_floor):
    """D3, the general form of the same oracle, on a cell with NO symmetry of
    its own: mirror the cell and the Jones matrix must come back as
    ``P J P``.  This is the property that holds for every cell, so it cannot
    be satisfied by a fixture that happens to be symmetric.

    BAR ``_D3_SYM_BAR``.  MEASURED after: 5.1e-15 .. 1.6e-14 (Jones scale
    0.11 .. 0.18).  FAIL-BEFORE, measured here: 2.03e-04 .. 5.01e-03.
    """
    cell = _cell(kind, _SKEW)
    assert not np.array_equal(cell, _mirror(cell)), \
        "fixture is accidentally mirror-symmetric -- it proves nothing"
    for M in (3, 4, 5):
        J, Jm = _oop_jones(cell, M), _oop_jones(_mirror(cell), M)
        assert float(np.max(np.abs(Jm - _P2 @ J @ _P2))) < _D3_SYM_BAR, \
            (kind, M)
    Jb = _oop_jones(cell, 3, symmetrize=False)
    Jbm = _oop_jones(_mirror(cell), 3, symmetrize=False)
    assert float(np.max(np.abs(Jbm - _P2 @ Jb @ _P2))) > pre_fix_floor, (
        f"{kind}: the single-order 3x3 operator is no longer x<->y "
        f"asymmetric -- fixture stale")


@pytest.mark.parametrize("theta_deg,phi_deg", [(0.0, 0.0), (14.0, 29.0),
                                               (25.0, 45.0)])
def test_d3_uniform_rotated_director_matches_the_berreman_oracle(theta_deg,
                                                                 phi_deg):
    """D3.  ORACLE: a conical Berreman 4x4 solve (``elements/berreman.py``) on
    a UNIFORM rotated-director uniaxial cell.

    Berreman takes the tensor directly and carries NO Fourier factorization,
    which is exactly what D3 changes -- so it is an independent reading of the
    answer the symmetrized operator has to keep giving.  On a laterally
    uniform cell the two factorization orders coincide analytically, so this
    gates that the symmetrisation is INERT where it must be, at the same time
    as it gates the off-plane path's absolute correctness.

    BAR 1e-11 on ``max|dJ|`` against a Jones of scale 0.24 .. 0.27, i.e. a
    relative 4e-11.  MEASURED: 1.1e-15 .. 7.1e-15 at M = 1 and 2, both with
    and without the symmetrisation -- four decades inside.
    """
    cell = _cell("uniform", _SKEW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _R, _T, Jr, _Jt = berreman_jones_1d(
            [(_SKEW, _D3["depth"])], 1.5, 1.0, _D3["wl"],
            angle=np.deg2rad(theta_deg), phi=np.deg2rad(phi_deg))
    for M in (1, 2):
        for sym in (False, True):
            J = _oop_jones(cell, M, symmetrize=sym,
                           theta=np.deg2rad(theta_deg),
                           phi=np.deg2rad(phi_deg))
            assert float(np.max(np.abs(J - Jr))) < 1e-11, (M, sym, J, Jr)


def test_d3_separable_stripe_is_unchanged_by_the_symmetrisation():
    """D3.  For a y-UNIFORM (separable) cell the two factorization orders
    coincide analytically, so the mean must reproduce the single order.

    BAR 1e-10 on ``max|J_sym - J_L2L1|``.  MEASURED: 7.8e-15 / 1.3e-14 /
    2.6e-14 at M = 3 / 5 / 7 -- the extra arithmetic's rounding, four decades
    inside the bar and TWELVE decades below the 0.327 of GENUINE form
    birefringence the stripe has (asserted too, so the gate cannot pass on a
    cell with no signal).
    """
    stripe = _cell("stripe", _SKEW)
    for M in (3, 5, 7):
        Ja, Jb = _oop_jones(stripe, M), _oop_jones(stripe, M, symmetrize=False)
        assert float(np.max(np.abs(Ja - Jb))) < 1e-10, M
        assert abs(Ja[0, 0] - Ja[1, 1]) > 0.3, (M, Ja)


def test_d3_reduces_exactly_to_the_in_plane_operator_it_generalizes():
    """D3.  On a cell with NO off-plane components the symmetrized 3x3
    operator must be the symmetrized 2x2 one (H3) with an inert z row and
    column -- the sense in which this IS the generalization of the shipped
    in-plane fix, and the proof that the in-plane path's operator is untouched
    by it.

    Tolerance-at-0.0 on the max absolute difference (the standing rule).
    MEASURED: 0.0 on all four in-plane blocks at M = 3 and 5, and the four
    off-plane blocks are exactly zero.
    """
    S = 64
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    m = ((ii - (S - 1) / 2.0) ** 2
         + (jj - (S - 1) / 2.0) ** 2) <= (0.25 * S) ** 2
    exx = np.where(m, 6.25 + 0j, 1.0 + 0j)
    eyy = np.where(m, 4.00 + 0j, 1.0 + 0j)
    exy = np.where(m, 0.50 + 0j, 0.0 + 0j)
    ezz = np.where(m, 5.00 + 0j, 1.0 + 0j)
    cell = np.zeros((S, S, 3, 3), dtype=complex)
    cell[:, :, 0, 0], cell[:, :, 1, 1] = exx, eyy
    cell[:, :, 0, 1], cell[:, :, 1, 0] = exy, exy
    cell[:, :, 2, 2] = ezz
    for M in (3, 5):
        orders, _N = _twod._harmonic_orders_2d(M, M)
        eh = _twod._li_convolutions_2d_tensor_full(cell, orders, M, M, np)
        two = _twod._li_convolutions_2d_tensor(exx, exy, exy, eyy, orders,
                                               M, M, np)
        for (a, b), C in (((0, 0), two[0]), ((0, 1), two[1]),
                          ((1, 0), two[2]), ((1, 1), two[3])):
            assert float(np.max(np.abs(eh[(a, b)] - C))) == 0.0, (M, a, b)
        for a, b in ((0, 2), (1, 2), (2, 0), (2, 1)):
            assert float(np.max(np.abs(eh[(a, b)]))) == 0.0, (M, a, b)


def test_d3_energy_closes_on_the_off_plane_path():
    """D3.  The symmetrized off-plane operator must still conserve energy on a
    LOSSLESS rotated-director cell.

    ORACLE: the conservation theorem.  BAR 1e-9 on ``|R + T - 1|`` per
    incident polarization (the off-plane 4N cascade's own closure is looser
    than the 2N one's).  MEASURED: 1.4e-14 at M = 2 and 1.8e-14 at M = 3, over
    both polarizations -- five decades inside.
    """
    cell = _cell("square", _SKEW)
    for M in (2, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = rcwa_jones_2d(
                _D3["period"], _D3["period"], cell, 1.5, 1.0, _D3["depth"],
                _D3["wl"], theta=np.deg2rad(14.0), phi=np.deg2rad(29.0),
                n_orders_x=M, n_orders_y=M, formulation="fff_nv")
        tot = np.asarray(R).sum(axis=1) + np.asarray(T).sum(axis=1)
        assert float(np.max(np.abs(tot - 1.0))) < 1e-9, (M, tot)


# ===========================================================================
# D1 -- the two Toeplitz inverses: the two facts the REFUSAL rests on
# ===========================================================================

def _inverse_rule_toeplitz(n_metal, duty, M):
    """The library's own ``[[1/eps]]`` for a binary metal grating in air, in
    the INTERNAL (loss-bridge-conjugated) convention, plus the first column /
    first row ``scipy.linalg.solve_toeplitz`` wants."""
    N = 2 * M + 1
    eps_r = np.conj(_C(n_metal)) ** 2
    c = _binary_step_coeffs(_C(1.0), 1.0 / eps_r, duty, N, np)
    T = _toeplitz_1d(c, M)
    centre = (c.shape[0] - 1) // 2
    return T, c[centre:centre + N], c[centre::-1][:N]


def _equilibrated_backward_residual(A, Y, X):
    """Row-equilibrated ``||A Y - X|| / ||X||`` -- the same instrument
    :func:`_guarded_inverse` scores an inverse on, applied to the product each
    candidate actually has to produce."""
    s = np.max(np.abs(A), axis=1)
    s = np.where(s == 0, 1.0, s)
    num = float(np.max(np.abs((A @ Y - X) / s[:, None])))
    den = max(float(np.max(np.abs(X / s[:, None]))), 1e-300)
    return num / den


@pytest.mark.parametrize("metal,M", [(_AG, 50), (_AG, 200), (_AU, 50),
                                     (_AU, 200)])
def test_d1_levinson_lands_outside_what_the_package_closure_admits(metal, M):
    """D1, half one of the refusal.  TWO-SIDED, on the library's own matrix.

    The design asked for ``scipy.linalg.solve_toeplitz`` (Levinson) wherever
    the inverse is immediately multiplied.  Measured against the LU inverse on
    the product these sites need (``T^-1 X`` with ``X = diag(kx)``):

    * Levinson's own equilibrated backward residual is 1.2e-13 .. 4.7e-13,
      against 2.6e-15 .. 6.6e-15 for both the explicit inverse and an LU
      ``solve`` -- a factor 19 .. 67;
    * its ANSWER differs from the shipped one by 6.3e-13 .. 1.8e-11 relative,
      i.e. ABOVE the 1.4e-13 closure this package holds, while the LU
      ``solve`` differs by 2.9e-15 .. 1.1e-14, a decade BELOW it.

    So the tolerance Levinson needs is not one the H2 / M1 census admits, and
    the gate here is the DECISION, with a gap on both sides of the package's
    own closure: the LU route sits 12.5x under it, the Levinson route 4.5x
    over it, and the two are separated by a factor 56.  (The cost is the other
    half and is not assertable without a clock: 12x .. 20x the explicit
    inverse at these sizes, reported in ``fixes/WP-B5_REPORT.md``.)
    """
    T, col, row = _inverse_rule_toeplitz(metal, 0.5, M)
    N = 2 * M + 1
    X = np.diag(np.linspace(-1.7, 1.7, N).astype(_C))
    Y_inv = np.linalg.inv(T) @ X
    Y_lu = np.linalg.solve(T, X)
    Y_lev = sla.solve_toeplitz((col, row), X)
    scale = float(np.max(np.abs(Y_inv)))
    r_inv = _equilibrated_backward_residual(T, Y_inv, X)
    r_lev = _equilibrated_backward_residual(T, Y_lev, X)
    d_lu = float(np.max(np.abs(Y_lu - Y_inv))) / scale
    d_lev = float(np.max(np.abs(Y_lev - Y_inv))) / scale
    assert r_lev > 5.0 * r_inv, (metal, M, r_lev, r_inv)
    assert d_lev > 1.4e-13, (metal, M, d_lev)
    assert d_lu < 1.4e-13, (metal, M, d_lu)


@pytest.mark.parametrize("metal", [_AG, _AU, 1.374 + 7.620j, 3.48 + 2.79j])
def test_d1_the_inverse_rule_toeplitz_is_not_the_ill_conditioned_matrix(metal):
    """D1, half two of the refusal -- and a correction of record.

    WP-A14 deferred D1 partly because ``[[1/eps]]`` for a metallic grating
    "is exactly the matrix the M1 conditioning census found reaching ``cond
    ~1e13``".  It is not: that reading belongs to the interface mode-match
    ``a + b``, which :func:`_interface_smatrix`'s own docstring records.
    MEASURED here over the metallic ladder (four metals x duty 0.1 / 0.5 / 0.9
    x ``n_orders`` 50 / 200) the worst ``cond([[1/eps]])`` is 2.51e+02 and the
    worst ``cond([[eps]])`` 2.51e+02.

    BAR 1e+06 -- 3.6 decades above the worst measurement and 7 below the 1e+13
    the deferral attributed to it.  The consequence is that D1's refusal rests
    on COST and on Levinson's own backward error, not on this matrix being
    near-singular; a future build that makes it so re-opens the question, which
    is what this gate is for.
    """
    for duty in (0.1, 0.5, 0.9):
        for M in (50, 200):
            T, _c, _r = _inverse_rule_toeplitz(metal, duty, M)
            assert float(np.linalg.cond(T)) < 1e6, (metal, duty, M)


def test_d1_the_two_sites_still_form_the_explicit_inverse():
    """D1's DECISION, pinned where it lives.

    Both sites keep the explicit inverse; the gates above are what makes that
    a measured choice rather than an omission.  If either site is ever changed
    to a factorization route, this test fails and its author has to re-run the
    two measurements above and re-state the report's D1 section.
    """
    import lumenairy.elements.rcwa.oned as _oned
    conv = inspect.getsource(_oned._binary_grating_convolutions)
    assert "xp.linalg.inv(_toeplitz_1d(inv_c, n_orders))" in conv
    body = inspect.getsource(_oned.rcwa_efficiency_1d)
    assert "EPS_inv1 = xp.linalg.inv(EPS)" in body


def test_d1_guarded_inverse_is_untouched_by_this_work_package():
    """D1 / D2.  The guard itself must still be the pre-M1 arithmetic on the
    default path: ``_guarded_inverse`` returns ``xp.linalg.inv`` bit for bit
    whenever the census is disarmed.  Tolerance-at-0.0 on the max absolute
    difference.
    """
    rng = np.random.default_rng(13)
    prev = _rc._INV_CENSUS
    try:
        _rc._INV_CENSUS = None
        for n in (7, 33, 64):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            got = _guarded_inverse(A, "probe")
            assert float(np.max(np.abs(got - np.linalg.inv(A)))) == 0.0, n
    finally:
        _rc._INV_CENSUS = prev


# ===========================================================================
# The 2-D entry points the closed form also serves
# ===========================================================================

@pytest.mark.parametrize("formulation", ["laurent", "li", "fff_nv"])
def test_d2_two_d_entry_points_still_close(formulation):
    """D2.  ``rcwa_efficiency_2d`` and ``rcwa_jones_2d`` take the same closed
    form; a lossless cell must still close on both.

    BAR 1e-10 on ``|R + T - 1|``.  MEASURED at M = 3 and 4: 2.2e-16 .. 3.2e-15
    (``rcwa_efficiency_2d``, ``'laurent'`` / ``'li'``) and 6.7e-16 .. 6.9e-15
    (``rcwa_jones_2d``, all three formulations) -- five decades inside.

    ``rcwa_efficiency_2d(formulation='fff_nv')`` is EXCLUDED from the
    efficiency arm, and not by oversight: on that entry ``'fff_nv'`` is the
    NORMAL-VECTOR method, whose closure on a cornered cell is its own
    documented artefact (5.7e-04 / 4.8e-04 here, unmoved by this work package
    -- the ladder diff in ``fixes/WP-B5_REPORT.md`` puts the whole 2-D
    efficiency family at 1.6e-15 before vs after).  On ``rcwa_jones_2d`` the
    same spelling means the Li-2003 successive factorization, which closes at
    1.6e-15, so that arm keeps all three.
    """
    S = 48
    ii, jj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    m = (np.abs(ii - S // 2) < S // 5) & (np.abs(jj - S // 2) < S // 5)
    cell = np.where(m, 6.25 + 0j, 2.25 + 0j)
    for M in (3, 4):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eff = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6,
                                     0.633e-6, theta=np.deg2rad(9.0),
                                     n_orders_x=M, n_orders_y=M,
                                     formulation=formulation)
            _o, R, T, _J = rcwa_jones_2d(
                0.5e-6, 0.5e-6, cell[:, :, None, None] * np.eye(3), 1.5, 1.0,
                0.3e-6, 0.633e-6, theta=np.deg2rad(9.0), n_orders_x=M,
                n_orders_y=M, formulation=formulation)
        if formulation != "fff_nv":
            tot = float(np.sum(eff[1]) + np.sum(eff[2]))
            assert abs(tot - 1.0) < 1e-10, (formulation, M, tot)
        totj = np.asarray(R).sum(axis=1) + np.asarray(T).sum(axis=1)
        assert float(np.max(np.abs(totj - 1.0))) < 1e-10, (formulation, M)


# ===========================================================================
# VERIFY-WP-B5 additions: the boundary of D2's re-association, and the
# ordering of D3's Schur fold
# ===========================================================================

#: The leaky guided-mode fixture that drives ``I - B11 A22`` near-singular
#: FROM THE PUBLIC API.  A weakly modulated high-index slab in air whose +-1
#: diffraction order is EVANESCENT in both half-spaces (``wavelength /
#: period`` = 1.407 > 1, so total internal reflection at both faces) and
#: PROPAGATING inside the layer: an eigenvalue of ``B11 A22`` then has modulus
#: ``1 - O(dn^2)`` with a phase the thickness tunes, so the cavity denominator
#: can be driven arbitrarily close to singular with ``|X| <= 1`` throughout --
#: no growing propagator anywhere.
_GMR = dict(period=0.45e-6, n_groove=2.0, n_substrate=1.0, n_superstrate=1.0,
            duty_cycle=0.5, wavelength=0.633e-6, formulation="li",
            n_orders=15)


def _gmr_chain(dn, depth, pol):
    """``(SA, SB, cinc)`` the 1-D solve hands its last star on the leaky
    guided-mode fixture, or ``None`` when the solve refuses."""
    import lumenairy.elements.rcwa.oned as _oned
    cap = []
    orig = _oned._redheffer_star_rt

    def spy(SA, SB, cinc):
        cap.append((SA, SB, cinc))
        return orig(SA, SB, cinc)

    _oned._redheffer_star_rt = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_efficiency_1d(n_ridge=2.0 + dn, depth=depth,
                               polarization=pol, **_GMR)
    except _rc._EnergyError:
        return None
    finally:
        _oned._redheffer_star_rt = orig
    return cap[-1] if cap else None


def _gmr_gap(dn, depth, pol):
    """``min |1 - eig(B11 A22)|`` -- exactly zero at the cavity resonance."""
    chain = _gmr_chain(dn, depth, pol)
    if chain is None:
        return 1.0
    SA, SB, _c = chain
    ev = np.linalg.eigvals(np.asarray(SB[0]) @ np.asarray(SA[3]))
    return float(np.min(np.abs(1.0 - ev)))


def _gmr_resonance(dn, pol):
    """The thickness that puts the fixture ON resonance, located at RUNTIME on
    the running build (a coarse scan, then golden section on the eigenvalue
    gap) -- the state is engineered, never hoped for."""
    depths = np.linspace(0.90e-6, 1.50e-6, 121)
    vals = [_gmr_gap(dn, d, pol) for d in depths]
    i = int(np.argmin(vals))
    a, b = depths[max(i - 1, 0)], depths[min(i + 1, len(depths) - 1)]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _gmr_gap(dn, c, pol), _gmr_gap(dn, d, pol)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _gmr_gap(dn, c, pol)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _gmr_gap(dn, d, pol)
        if b - a < 1e-19:
            break
    return 0.5 * (a + b)


def test_d2_a_near_singular_star_denominator_is_reachable_and_neither_form_is_better():
    """D2's residual risk, stated as the two-sided property it actually has.

    The re-association is neutral only while ``I - B11 A22`` is well
    conditioned.  A GROWING layer propagator is one way to break that and the
    branch cut closes it; a HIGH-Q CAVITY RESONANCE is the other, it needs no
    growing propagator, and it IS reachable from :func:`rcwa_efficiency_1d`
    alone -- so the 1.7e-15 / 3.1e-15 envelope measured on the
    well-conditioned entry-point matrix is a statement about that population,
    not a bound on the entry point.

    ENGINEERED, not hoped for: the resonance is located at runtime by a scan
    plus golden section on the eigenvalue gap of ``B11 A22``
    (:func:`_gmr_resonance`), on the running build's own arithmetic.  The
    premise is ASSERTED, not skipped -- if no rung of the family reaches
    ``cond`` 1e10 the claim below needs re-deriving and this test says so.

    ORACLE: :func:`_star_oracle`, the defining coupled system solved whole.

    BARS, both derived from the conditioning the fixture reaches:

    * ``cond >= 1e10`` at the worst rung (MEASURED 1.6e12 / 2.4e12 / 6.4e12 /
      1.8e13 on four of the six rungs, the other two landing at 6.8e03 and
      4.2e06 -- two to three decades of margin on the four that qualify) with
      ``max|A22| <= 1`` (MEASURED exactly 1.000000 on every rung -- there is
      no growing propagator here, which is what makes this the REACHABLE
      route into the regime);
    * the star output is then undetermined by MORE than the package's
      well-conditioned envelope: ``max(|closed - oracle|, |star - oracle|) >
      1e-9`` (MEASURED 6.5e-06 / 1.9e-05 / 3.3e-04 / 9.7e-04 on those four
      rungs -- 3.8 to 6.0 decades above the bar, and 9.3 to 11.5 above the
      3.1e-15 the well-conditioned entry-point matrix moves.  OFF resonance
      the same six rungs read 6.6e-17 .. 2.9e-16, which is the two-sided
      contrast);
    * and it IS the conditioning, not a defect of either association: BOTH
      sit inside ``50 * cond * eps`` (MEASURED worst 9.7e-04 against an
      envelope of 1.9e-01, two and a half decades inside), and the two differ
      from each other (3.0e-07 .. 4.2e-04) by less than either differs from
      the oracle.
    """
    eps = float(np.finfo(np.float64).eps)
    rows = []
    for dn in (3e-2, 1e-2, 1e-3):
        for pol in ("te", "tm"):
            depth = _gmr_resonance(dn, pol)
            chain = _gmr_chain(dn, depth, pol)
            assert chain is not None, (dn, pol, depth)
            SA, SB, cinc = chain
            A22, B11 = np.asarray(SA[3]), np.asarray(SB[0])
            n = A22.shape[0]
            Md = np.eye(n, dtype=_C) - B11 @ A22
            cond = float(np.linalg.cond(Md))
            closed = _redheffer_star_rt(SA, SB, cinc)
            S = _redheffer_star(SA, SB)
            star = (S[0] @ cinc, S[2] @ cinc)
            ref = _star_oracle(SA, SB, cinc)
            sc = max(float(np.max(np.abs(ref[0]))),
                     float(np.max(np.abs(ref[1]))))

            def rel(x, y, sc=sc):
                return max(float(np.max(np.abs(x[0] - y[0]))),
                           float(np.max(np.abs(x[1] - y[1])))) / sc
            rows.append((dn, pol, cond, float(np.max(np.abs(A22))),
                         rel(closed, ref), rel(star, ref)))

    assert all(r[3] <= 1.0 + 1e-12 for r in rows), \
        f"the fixture grew a propagator -- |X| <= 1 no longer holds: {rows}"
    hard = [r for r in rows if r[2] >= 1e10]
    assert hard, (
        "no rung of the leaky guided-mode family reached cond(I - B11 A22) = "
        f"1e10, so the premise of D2's residual-risk statement is gone and "
        f"the bars below need re-deriving: {rows}")
    worst = max(hard, key=lambda r: max(r[4], r[5]))
    assert max(worst[4], worst[5]) > 1e-9, (
        "the near-singular star denominator no longer moves the answer past "
        f"the well-conditioned envelope -- re-measure D2's residual risk: "
        f"{worst}")
    for dn, pol, cond, _a22, dc, ds in hard:
        envelope = 50.0 * cond * eps
        assert dc < envelope, ("closed form", dn, pol, cond, dc, envelope)
        assert ds < envelope, ("assembled star", dn, pol, cond, ds, envelope)


def test_d3_the_symmetrisation_is_the_raw_mean_so_the_l3_fold_runs_after_it():
    """D3's ordering, pinned where it lives.

    :func:`_li_convolutions_2d_tensor_full` must return the mean of the two
    factorization orders on the RAW ``ehat`` blocks, because its one caller
    applies the ``l3-`` ``E_z`` fold (Li 2003 Eq. 27) to what it gets back and
    ALSO feeds the raw cross-blocks to the generalized generator's own
    ``inv(EZZ)``.  Taking the mean after a per-order fold would hand the
    caller two quantities from different operators.

    Tolerance-at-0.0 on the nine blocks against the mean recomputed here from
    :func:`_li_tensor_full_l2l1` and its transposed run (the standing rule for
    an identity that is exact by construction).

    THE ORDERING IS NOT COSMETIC, and that is asserted rather than assumed:
    ``Schur(mean)`` and ``mean(Schur)`` of the very same two orders differ by
    2.37e-05 absolute / 1.69e-05 relative on this cell at M = 3 (2.25e-05 /
    1.60e-05 at M = 4) -- BAR 1e-8 relative, three decades under the
    measurement.  Without this test the swap is silent: every other D3 gate
    stays green under it (the x<->y mirror is a symmetry of BOTH orderings),
    which is exactly why it is here.
    """
    cell = _cell("disk", _SKEW)

    def schur(d):
        zi = np.linalg.inv(d[(2, 2)])
        return {(r, s): d[(r, s)] - d[(r, 2)] @ zi @ d[(2, s)]
                for r in range(2) for s in range(2)}

    for M in (3, 4):
        orders, _N = _twod._harmonic_orders_2d(M, M)
        A = _twod._li_tensor_full_l2l1(cell, orders, M, np)
        epsT = np.transpose(cell, (1, 0, 2, 3))[:, :, _PERM, :][:, :, :, _PERM]
        Braw = _twod._li_tensor_full_l2l1(epsT, orders[:, ::-1], M, np)
        B = {(r, s): Braw[(_PERM[r], _PERM[s])]
             for r in range(3) for s in range(3)}
        mean = {k: 0.5 * (A[k] + B[k]) for k in A}

        shipped = _twod._li_convolutions_2d_tensor_full(cell, orders, M, M, np)
        for k in mean:
            assert float(np.max(np.abs(shipped[k] - mean[k]))) == 0.0, (M, k)

        mt = schur(mean)
        tm = {k: 0.5 * (schur(A)[k] + schur(B)[k]) for k in schur(A)}
        scale = max(float(np.max(np.abs(mt[k]))) for k in mt)
        gap = max(float(np.max(np.abs(mt[k] - tm[k]))) for k in mt) / scale
        assert gap > 1e-8, (
            f"M={M}: the two fold orderings agree to {gap:.3e}, so this "
            f"fixture no longer distinguishes them -- pick one that does")
