"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory M -- M7, M8, M10 + the
uniform-cell ``fff_nv`` crash found while fixing M2-M6/M9.

NEW (from the 4602b7f report)
    A spatially UNIFORM ``eps_cell`` with ``formulation='fff_nv'`` died with a
    bare ``AssertionError: _nv_field_2d: normal-vector field exceeds unit norm
    (max |N| = nan)`` plus two ``RuntimeWarning: invalid value encountered in
    divide``.  Mechanism (measured): the indicator ``|eps - eps[0,0]|`` is
    identically 0 on a uniform cell, so ``mag.max() == 0``, the regulariser
    ``eps_reg = 1e-3 * (0 + 1e-300)`` UNDERFLOWS when squared, and the
    normalisation is 0/0 -> NaN.  The M4 non-separability guard already exempted
    a uniform cell ("its normal-vector field is undefined ... the N-field must
    not be built here"), but the solver then built it anyway two lines later.
    The ``'li'`` path had the precedent since v5.14.1 (twod.py: uniform cell ->
    ``formulation = 'laurent'``, verified to return bit-identical bytes to an
    explicit ``'laurent'`` call), so ``'fff_nv'`` now routes the same way rather
    than raising: with no walls every rule coincides.

M7 (MEDIUM, conventions)
    Order-count kwargs are spelled three ways across the family
    (``n_orders`` / ``n_orders_x``+``n_orders_y`` / ``degree``+
    ``far_field_orders``), ``formulation`` defaults differ between siblings, and
    two engines were missing controls their siblings had.  LEAST-BREAKING fix:
    nothing renamed, the missing kwargs ADDED (``formulation`` / ``stabilize`` /
    ``symmetry`` on ``rcwa_efficiency_2d_shapes``, ``formulation`` on
    ``rcwa_jones_1d``) with defaults chosen so every existing call is
    bit-identical, and the spellings/defaults reconciled in ONE cross-family
    table (``rcwa/_core.py`` module docstring).  The propagating-order mask was
    genuinely inconsistent -- 9 PMM far-field sites + rcwa used
    ``Re(kz) > 0`` (what both docstrings document) while 5 one-dimensional PMM
    sites used ``Re(kz) > 1e-12`` -- and is now aligned on ``> 0``.

M8 (MEDIUM)
    ``n_orders_y >= 1`` was over-broad.  ``N_y = 0`` is exact on a y-INVARIANT
    cell (the ``n != 0`` harmonics are decoupled): measured agreement with
    ``rcwa_efficiency_1d`` of ``max|dR| <= 5.2e-15`` / ``max|dT| <= 3.9e-14``
    over TE/laurent, TM/li and TE/li at ``n_orders_x`` 5 and 12, closure
    ~1e-14 -- while the forced minimum tripled the retained-harmonic count and
    so cost 27x the ``O(N^3)`` eigensolve (raw eig measured 3.28 ms at N=25 vs
    89.8 ms at N=75; end-to-end 20 ms vs 12.1 s on this box).  A y-VARYING cell
    at ``n_orders_y=0`` silently solved the y-AVERAGED structure instead
    (measured ``R00 = 0.00988026829053`` == the explicitly y-averaged cell, vs
    a converged ``0.0201934658126``, with closure ``-2.2e-16`` -- energy
    conservation provably cannot catch it), so that case now raises.

M10 (LOW, hygiene)
    Behaviour-free except for the explicit ``_`` discards; the pins here guard
    the two docstring claims that ARE checkable -- that ``_sem_modes(robust=)``
    really is inert, and that ``_analytic_convolutions_2d`` still returns its
    (unused-by-any-formulation) second matrix.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import _core as _pcore
from lumenairy.elements.rcwa import _core as _rcore
from lumenairy.elements.rcwa import (
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
    rcwa_jones_1d,
)
from lumenairy.elements.rcwa import twod as _twod

_C = np.complex128
P = 0.9e-6
WL = 0.633e-6
DISK = [{"shape": "disk", "eps": 6.25, "radius": 0.25e-6}]
RECT = [{"shape": "rectangle", "eps": 6.25 + 0.5j, "size": (0.4e-6, 0.3e-6)}]


def _uniform(S=21, eps=6.25 + 0j):
    return np.full((S, S), eps, dtype=_C)


def _square(S=25, half=0.25, eps_in=6.25 + 0j, eps_out=1.0 + 0j):
    u = (np.arange(S) + .5) / S
    xx, yy = np.meshgrid(u, u, indexing="ij")
    return np.where((np.abs(xx - .5) < half) & (np.abs(yy - .5) < half),
                    eps_in, eps_out).astype(_C)


# y-INVARIANT (x-only) cell sampled exactly like the 1-D solver's internal grid,
# so the two engines see the SAME Fourier coefficients.
DUTY, N_R, N_G, DEPTH, THETA = 0.4, 2.5, 1.0, 0.22e-6, 0.17


def _ystripe(Sx=4096, Sy=1):
    x = (np.arange(Sx) + 0.5) / Sx
    prof = np.where(x < DUTY, N_R ** 2, N_G ** 2).astype(_C)
    return np.repeat(prof[:, None], Sy, axis=1)


def _lc_tensor(no, ne, ang):
    d = np.array([np.cos(ang), np.sin(ang), 0.0])
    return (no ** 2) * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)


# ===========================================================================
# NEW -- uniform cell + fff_nv routes to laurent (was AssertionError: nan)
# ===========================================================================

@pytest.mark.parametrize("eps", [6.25 + 0j, (0.05 + 3.3j) ** 2])
def test_new_uniform_cell_fff_nv_matches_laurent_bit_for_bit(eps):
    """The crash case, plus its lossy twin (which would also have taken the
    METALLIC branch of the M4 gate had it reached one)."""
    cell = _uniform(21, eps)
    args = (0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.2e-6, WL)
    kw = dict(theta=0.2, phi=0.3, polarization="tm", n_orders_x=2, n_orders_y=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # the 0/0 divide
        o_n, R_n, T_n = rcwa_efficiency_2d(*args, formulation="fff_nv", **kw)
    o_l, R_l, T_l = rcwa_efficiency_2d(*args, formulation="laurent", **kw)
    assert np.asarray(R_n).tobytes() == np.asarray(R_l).tobytes()
    assert np.asarray(T_n).tobytes() == np.asarray(T_l).tobytes()
    assert np.array_equal(o_n, o_l)


def test_new_uniform_cell_fff_nv_prepared_path():
    """``prepare_rcwa_2d`` carried the same crash and gets the same routing."""
    kw = dict(n_orders_x=2, n_orders_y=2)
    pn = _twod.prepare_rcwa_2d(0.5e-6, 0.5e-6, _uniform(), 1.5, 1.0, 0.2e-6,
                               formulation="fff_nv", **kw)
    pl = _twod.prepare_rcwa_2d(0.5e-6, 0.5e-6, _uniform(), 1.5, 1.0, 0.2e-6,
                               formulation="laurent", **kw)
    assert pn.formulation == "laurent"
    assert pn.fff is None                    # no normal-vector tensor built
    assert np.array_equal(pn.EPS, pl.EPS)


def test_new_uniform_predicate_and_patterned_cells_untouched():
    """The routing is uniform-ONLY: a patterned cell still builds the N-field
    (and still meets the M4 gate)."""
    assert _twod._uniform_cell(_uniform())
    assert _twod._uniform_cell(np.full((8, 8), 3.0 + 1e-13, dtype=_C))
    assert not _twod._uniform_cell(_square())
    # lossless dielectric square: inside fff_nv's validated scope -> solves
    o, R, T = rcwa_efficiency_2d(
        0.5e-6, 0.5e-6, _square(), 1.0, 1.0, 0.18e-6, WL, theta=1e-8, phi=0.0,
        polarization="tm", n_orders_x=2, n_orders_y=2, formulation="fff_nv")
    assert float(R.sum() + T.sum()) == pytest.approx(1.0, abs=5e-2)
    # metal square: still gated (M4), i.e. the routing did not bypass the guard
    with pytest.raises(ValueError, match="NON-SEPARABLE"):
        rcwa_efficiency_2d(0.5e-6, 0.5e-6, _square(25, .25, (0.05 + 3.3j) ** 2),
                           1.5, 1.0, 0.2e-6, WL, theta=0.2, phi=0.3,
                           polarization="tm", n_orders_x=2, n_orders_y=2,
                           formulation="fff_nv")


def test_new_nv_field_still_raises_when_built_on_a_uniform_cell():
    """The assertion itself is NOT weakened -- it is simply unreachable from the
    solvers now.  (Documents the mechanism: 0/0 -> nan.)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(AssertionError, match="max \\|N\\| = nan"):
            _twod._nv_field_2d(_uniform(), 1.0, 1.0)


# ===========================================================================
# M8 -- n_orders_y = 0
# ===========================================================================

def test_m8_validator_allows_zero_y_orders_only():
    _rcore._validate_geometry("probe", n_orders=3, n_orders_y=0)      # no raise
    with pytest.raises(ValueError, match=r"n_orders must be an integer >= 1"):
        _rcore._validate_geometry("probe", n_orders=0, n_orders_y=3)
    with pytest.raises(ValueError, match="only the Y count may be 0"):
        _rcore._validate_geometry("probe", n_orders=0, n_orders_y=3)
    with pytest.raises(ValueError, match=r"n_orders_y must be an integer >= 0"):
        _rcore._validate_geometry("probe", n_orders=3, n_orders_y=-1)
    with pytest.raises(ValueError, match=r"n_orders_y must be an integer >= 0"):
        _rcore._validate_geometry("probe", n_orders=3, n_orders_y=0.5)
    # a 1-D call keeps the plain message (no y count to talk about)
    with pytest.raises(ValueError, match=r"integer >= 1, got 0\."):
        _rcore._validate_geometry("probe", n_orders=0)


@pytest.mark.parametrize("pol,form", [("te", "laurent"), ("tm", "li"),
                                      ("te", "li")])
@pytest.mark.parametrize("nox", [5, 12])
def test_m8_y_invariant_zero_y_orders_matches_the_1d_solver(pol, form, nox):
    """The claim that makes ``N_y = 0`` legitimate: on a y-invariant cell it IS
    the rigorous 1-D answer."""
    o2, R2, T2 = rcwa_efficiency_2d(
        P, P, _ystripe(), 1.5, 1.0, DEPTH, WL, theta=THETA, phi=0.0,
        polarization=pol, n_orders_x=nox, n_orders_y=0, formulation=form,
        symmetry=False)
    o1, R1, T1 = rcwa_efficiency_1d(
        P, N_R, N_G, 1.5, 1.0, DEPTH, DUTY, WL, angle=THETA, polarization=pol,
        n_orders=nox, formulation=form)
    assert o2.shape == (2 * nox + 1, 2)
    assert np.array_equal(o2[:, 1], np.zeros(2 * nox + 1, dtype=o2.dtype))
    idx = [int(np.where(o2[:, 0] == m)[0][0]) for m in np.asarray(o1)]
    assert np.max(np.abs(np.asarray(R2)[idx] - np.asarray(R1))) < 1e-13
    assert np.max(np.abs(np.asarray(T2)[idx] - np.asarray(T1))) < 1e-12
    assert abs(float(R2.sum() + T2.sum()) - 1.0) < 1e-12


def test_m8_zero_y_orders_cuts_the_harmonic_count_by_three():
    """WHY the forced minimum was a cost bug: N_y = 0 -> 1/3 the harmonics ->
    1/27 the O(N^3) eigensolve.  (Harmonic counts, not wall-clock: a timing
    assertion is not reproducible on a shared box.)"""
    n0 = len(_twod._harmonic_orders_2d(12, 0)[0])
    n1 = len(_twod._harmonic_orders_2d(12, 1)[0])
    assert (n0, n1) == (25, 75)
    assert (n1 / n0) ** 3 == pytest.approx(27.0)


def test_m8_y_varying_cell_is_rejected_not_silently_averaged():
    """The audit's own configuration (P = 0.5 um, 64x64 square pillar,
    theta = 0.17): the trap is that the y-averaged answer is energy-clean and
    ~2x off, so only an input guard can catch it."""
    per = 0.5e-6
    u = (np.arange(64) + 0.5) / 64
    xx, yy = np.meshgrid(u, u, indexing="ij")
    sq = np.where((np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25),
                  6.25 + 0j, 1.0 + 0j).astype(_C)
    with pytest.raises(ValueError, match="needs a y-INVARIANT cell"):
        rcwa_efficiency_2d(per, per, sq, 1.5, 1.0, DEPTH, WL, theta=THETA,
                           n_orders_x=4, n_orders_y=0)
    # ... and the mechanism it protects against: with the y-average taken by
    # hand, N_y = 0 solves THAT structure (frozen value), which is ~2x off the
    # y-resolved answer while conserving energy to 2e-16.
    avg = np.repeat(sq.mean(axis=1)[:, None], 1, axis=1)
    _o, Ra, Ta = rcwa_efficiency_2d(per, per, avg, 1.5, 1.0, DEPTH, WL,
                                    theta=THETA, phi=0.0, n_orders_x=4,
                                    n_orders_y=0, symmetry=False)
    _o3, R3, _T3 = rcwa_efficiency_2d(per, per, sq, 1.5, 1.0, DEPTH, WL,
                                      theta=THETA, phi=0.0, n_orders_x=4,
                                      n_orders_y=2, symmetry=False)
    p_a = int(np.where(_o[:, 0] == 0)[0][0])
    p_3 = int(np.where((_o3[:, 0] == 0) & (_o3[:, 1] == 0))[0][0])
    assert float(Ra[p_a]) == pytest.approx(0.0098802682905296448, rel=1e-9)
    assert abs(float(Ra.sum() + Ta.sum()) - 1.0) < 1e-14      # energy-clean
    assert float(R3[p_3]) > 1.8 * float(Ra[p_a])              # the silent error


def test_m8_y_invariance_guard_does_not_hit_the_1d_stack_sentinel():
    """``RCWAStack`` uses ``noy = 0`` INTERNALLY to mean "1-D stack"
    (``self.is_1d``), and a 1-D stack has always accepted a 2-D-shaped cell
    (its y-average is the mono-periodic structure).  The M8 guard is therefore
    opt-in (``strict_y``) -- it must NOT fire there, but MUST fire for a stack
    explicitly built 2-D with ``n_orders_y=0``.  (Regression: the first cut of
    the guard broke test_audit_s1_2_rcwa_lossless_tripwire.)"""
    from lumenairy.elements.rcwa import RCWAStack
    u = (np.arange(24) + .5) / 24
    xx, yy = np.meshgrid(u, u, indexing="ij")
    cell2d = np.where((np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25),
                      6.25 + 0j, 2.25 + 0j)
    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=5)
    assert st.is_1d and st.n_orders_y == 0
    st.add_layer(0.1e-6, eps_cell=cell2d)                  # accepted, as before
    st2 = RCWAStack(period=1.0e-6, period_y=1.0e-6, n_substrate=1.5,
                    n_orders=5, n_orders_y=0)
    assert not st2.is_1d
    with pytest.raises(ValueError, match="needs a y-INVARIANT cell"):
        st2.add_layer(0.1e-6, eps_cell=cell2d)
    st2.add_layer(0.1e-6, eps_cell=_ystripe(1024, 4))      # y-invariant: fine


def test_m8_zero_y_orders_rejects_circular_truncation():
    with pytest.raises(ValueError, match="ZERO radius"):
        rcwa_efficiency_2d(P, P, _ystripe(), 1.5, 1.0, DEPTH, WL,
                           n_orders_x=4, n_orders_y=0, truncation="circular")


def test_m8_prepared_and_uniform_paths_accept_zero_y_orders():
    prep = _twod.prepare_rcwa_2d(P, P, _ystripe(), 1.5, 1.0, DEPTH,
                                 n_orders_x=5, n_orders_y=0)
    assert prep.N == 11
    o, R, T = rcwa_efficiency_2d(P, P, _uniform(), 1.5, 1.0, DEPTH, WL,
                                 n_orders_x=4, n_orders_y=0)
    assert len(o) == 9 and abs(float(R.sum() + T.sum()) - 1.0) < 1e-12


# ===========================================================================
# M7 (a) -- the missing kwargs, and the default paths they must not move
# ===========================================================================

def test_m7_shapes_gained_formulation_stabilize_symmetry():
    names = inspect.signature(
        inspect.unwrap(rcwa_efficiency_2d_shapes)).parameters
    for kw in ("formulation", "stabilize", "symmetry"):
        assert kw in names, kw
    # defaults chosen for bit-compatibility, NOT copied blindly from the
    # pixel solver (symmetry='auto' there, False here -- see the docstring)
    assert names["formulation"].default == "laurent"
    assert names["stabilize"].default is False
    assert names["symmetry"].default is False


@pytest.mark.parametrize("shapes,kw,r00,t00", [
    (DISK, dict(theta=0.0, phi=0.0, polarization="te", n_orders_x=3,
                n_orders_y=3), 0.061324200068491735, 0.064048643113290699),
    (RECT, dict(theta=0.2, phi=0.3, polarization="tm", n_orders_x=3,
                n_orders_y=3), 0.030268231609004673, 0.42883543183866879),
])
def test_m7_shapes_default_path_is_bit_identical(shapes, kw, r00, t00):
    """Adding the kwargs must not move the default answer.

    Two levels, because only one of them is portable:

    * the frozen values were measured on the PRE-fix tree (7ea2eb9) and are held
      to ``rel=1e-11`` -- NOT to the bit.  The analytic-shape solve runs a dense
      LAPACK eig whose summation order depends on the BLAS thread count
      (measured on this box: ``R00 = 0.061324200068491735`` at default threads
      vs ``0.06132420006795196`` at ``OMP_NUM_THREADS=1``, 8.8e-12 relative), so
      an ``==`` here would pin the threading environment, not the code;
    * the bit-for-bit claim is checked WITHIN the process, where the threading
      is common: passing the new kwargs at their defaults must reproduce the
      implicit-default bytes exactly.
    """
    o, R, T = rcwa_efficiency_2d_shapes(P, P, 1.0, shapes, 1.5, 1.0, 0.3e-6,
                                        WL, **kw)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert float(R[p0]) == pytest.approx(r00, rel=1e-11)
    assert float(T[p0]) == pytest.approx(t00, rel=1e-11)
    # explicit defaults == implicit defaults, bit for bit
    o2, R2, T2 = rcwa_efficiency_2d_shapes(
        P, P, 1.0, shapes, 1.5, 1.0, 0.3e-6, WL, formulation="laurent",
        stabilize=False, symmetry=False, **kw)
    assert np.asarray(R2).tobytes() == np.asarray(R).tobytes()
    assert np.asarray(T2).tobytes() == np.asarray(T).tobytes()


def test_m7_shapes_rejects_the_rules_it_cannot_implement():
    kw = dict(theta=0.0, phi=0.0, n_orders_x=3, n_orders_y=3)
    for form in ("li", "fff", "auto", "fff_nv"):
        with pytest.raises(NotImplementedError, match="analytic-shape path"):
            rcwa_efficiency_2d_shapes(P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL,
                                      formulation=form, **kw)
    with pytest.raises(ValueError, match="formulation must be"):
        rcwa_efficiency_2d_shapes(P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL,
                                  formulation="bogus", **kw)


def test_m7_shapes_symmetry_and_stabilize_actually_work():
    kw = dict(theta=0.0, phi=0.0, polarization="te", n_orders_x=3, n_orders_y=3)
    o, R, T = rcwa_efficiency_2d_shapes(P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL,
                                        **kw)
    o_s, R_s, T_s = rcwa_efficiency_2d_shapes(
        P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL, symmetry="auto", **kw)
    # the fold engages on a centred shape: same physics, not the same bits
    assert np.asarray(R_s).tobytes() != np.asarray(R).tobytes()
    assert np.max(np.abs(np.asarray(R_s) - np.asarray(R))) < 1e-11
    assert abs(float(R_s.sum() + T_s.sum()) - 1.0) < 1e-10
    # oblique -> precondition fails -> transparent fall back, bit for bit
    ob = dict(theta=0.2, phi=0.3, polarization="tm", n_orders_x=3, n_orders_y=3)
    a = rcwa_efficiency_2d_shapes(P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL, **ob)
    b = rcwa_efficiency_2d_shapes(P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL,
                                  symmetry="auto", **ob)
    assert np.asarray(b[1]).tobytes() == np.asarray(a[1]).tobytes()
    # the energy-clean case needs no ladder rung, so stabilize is a no-op here
    o_l, R_l, T_l = rcwa_efficiency_2d_shapes(
        P, P, 1.0, DISK, 1.5, 1.0, 0.3e-6, WL, stabilize=True, **kw)
    assert len(o_l) == len(o)
    assert np.asarray(R_l).tobytes() == np.asarray(R).tobytes()


def test_m7_jones_1d_gained_formulation():
    names = inspect.signature(inspect.unwrap(rcwa_jones_1d)).parameters
    assert "formulation" in names
    assert names["formulation"].default == "li"     # NOT 'auto': see docstring


def test_m7_jones_1d_default_path_is_bit_identical():
    """Frozen on the PRE-fix tree (7ea2eb9); ``rel=1e-11`` rather than ``==``
    for the same BLAS-threading reason as the shapes twin above.  The
    bit-for-bit claim is the alias loop at the end (same process, same
    threading)."""
    o, R, T, jr = rcwa_jones_1d(P, _lc_tensor(1.5, 1.7, 0.9), np.eye(3), 1.5,
                                1.0, 0.3e-6, 0.45, WL, angle=0.25, n_orders=6)
    assert complex(jr[0, 0]) == pytest.approx(
        complex(-0.18216190259927903, 0.011312428104752179), rel=1e-11)
    assert complex(jr[1, 0]) == pytest.approx(
        complex(-0.025780469223113889, 0.003854822163207039), rel=1e-11)
    assert float(R[0, 6]) == pytest.approx(0.033948831185424952, rel=1e-11)
    assert float(T[0, 6]) == pytest.approx(0.56038943435996447, rel=1e-11)
    for alias in ("li", "auto", "fff", "LI"):
        got = rcwa_jones_1d(P, _lc_tensor(1.5, 1.7, 0.9), np.eye(3), 1.5, 1.0,
                            0.3e-6, 0.45, WL, angle=0.25, n_orders=6,
                            formulation=alias)
        assert np.asarray(got[3]).tobytes() == np.asarray(jr).tobytes()


def test_m7_jones_1d_laurent_is_a_real_second_rule():
    """'laurent' must be the DIRECT rule everywhere: EXACTLY 'li' on a uniform
    tensor (there ``[[1/exx]]^{-1} == [[exx]]``, so the two are bit-identical,
    not merely close) and genuinely different on a patterned one.

    Convergence to the common limit was measured separately rather than pinned
    -- it needs a large truncation, and the approach is not monotone:
    ``|dJ00|`` = 2.76e-4 (n_orders 6), 1.44e-2 (12), 8.11e-4 (24), 3.79e-5
    (48) on this cell.
    """
    t = _lc_tensor(1.5, 1.7, 0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _rcore._EnergyWarning)
        u_li = rcwa_jones_1d(P, t, t, 1.5, 1.0, 0.3e-6, 0.45, WL, n_orders=6,
                             formulation="li")
        u_la = rcwa_jones_1d(P, t, t, 1.5, 1.0, 0.3e-6, 0.45, WL, n_orders=6,
                             formulation="laurent")
        assert np.asarray(u_la[3]).tobytes() == np.asarray(u_li[3]).tobytes()
        a = rcwa_jones_1d(P, t, 2.25 * np.eye(3), 1.5, 1.0, 0.3e-6, 0.45, WL,
                          n_orders=6, formulation="li")
        b = rcwa_jones_1d(P, t, 2.25 * np.eye(3), 1.5, 1.0, 0.3e-6, 0.45, WL,
                          n_orders=6, formulation="laurent")
    assert abs(complex(a[3][0, 0]) - complex(b[3][0, 0])) > 1e-5
    assert np.asarray(a[3]).tobytes() != np.asarray(b[3]).tobytes()


def test_m7_jones_1d_rejects_laurent_off_plane_and_unknown_rules():
    offp = np.array([[2.25, 0, 0.3], [0, 2.25, 0], [0.3, 0, 2.6]])
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        rcwa_jones_1d(P, offp, 2.25 * np.eye(3), 1.5, 1.0, 0.3e-6, 0.45, WL,
                      n_orders=5, formulation="laurent")
    with pytest.raises(ValueError, match="formulation must be"):
        rcwa_jones_1d(P, np.eye(3), np.eye(3), 1.5, 1.0, 0.3e-6, 0.45, WL,
                      n_orders=5, formulation="zzz")
    with pytest.raises(ValueError, match="formulation must be"):
        _rcore._tensor_convolutions({k: np.ones(8) for k in
                                     ("xx", "xy", "yx", "yy", "zz")}, 2, "zzz")


# ===========================================================================
# M7 (b) -- the cross-family table is the ONE place the spellings live
# ===========================================================================

def test_m7_cross_family_table_documents_every_spelling():
    doc = _rcore.__doc__
    assert "CROSS-FAMILY KWARG / DEFAULT MAP" in doc
    for token in ("n_orders_x", "far_field_orders", "degree", "RCWAStack",
                  "rcwa_efficiency_2d_shapes", "pmm_jones_2d", "formulation",
                  "stabilize", "symmetry", "Re(kz) > 0"):
        assert token in doc, token
    assert "rcwa/_core.py" in _pcore.__doc__      # the PMM-side pointer


# ===========================================================================
# M7 (c) -- ONE propagating-order mask across the family
# ===========================================================================

def test_m7_pmm_1d_mask_keeps_an_order_just_above_cutoff():
    """Direct unit probe of the aligned threshold: an order at
    ``Re(kz) = 5e-13`` is BELOW the old 1e-12 floor but genuinely propagating,
    so it must carry its (tiny) flux, as it always did in rcwa and in the 9
    other PMM far-field sites."""
    N = 3
    orders = np.array([-1, 0, 1])
    I2 = np.eye(2 * N, dtype=_C)
    kx = np.zeros(N)
    kz = np.array([1.0, 5e-13, 1.0], dtype=_C)
    R_eff, T_eff, jones = _pcore._assemble_jones_farfield(
        I2, I2, I2, I2, orders, kx, kz, kz, 1.0, 0.0, N)
    # NB abs=0: pytest.approx's DEFAULT 1e-12 absolute tolerance would swallow
    # the whole effect (the pre-fix value is an exact 0.0).
    assert float(R_eff[0, 1]) > 0.0 and float(T_eff[0, 1]) > 0.0
    assert float(R_eff[0, 1]) == pytest.approx(5e-13, rel=1e-9, abs=0.0)
    assert float(T_eff[0, 1]) == pytest.approx(5e-13, rel=1e-9, abs=0.0)
    # an exactly-at-cutoff / evanescent order is still zeroed
    kz0 = np.array([1.0, 0.0, 1.0], dtype=_C)
    R0, T0, _j = _pcore._assemble_jones_farfield(
        I2, I2, I2, I2, orders, kx, kz0, kz0, 1.0, 0.0, N)
    assert float(R0[0, 1]) == 0.0 and float(T0[0, 1]) == 0.0


def test_m7_no_1e_12_propagating_floor_survives_in_pmm():
    for fn in (_pcore._assemble_jones_farfield, _pcore._scalar_farfield_RT):
        src = inspect.getsource(fn)
        mask = [ln for ln in src.splitlines()
                if "where(" in ln and "real(kz" in ln]
        assert mask, fn.__name__
        assert all("> 0.0" in ln for ln in mask), (fn.__name__, mask)
        assert not any("1e-12" in ln for ln in mask), (fn.__name__, mask)


# ===========================================================================
# M10 -- the checkable half of the hygiene pass
# ===========================================================================

def test_m10_sem_modes_robust_flag_really_is_inert():
    """The docstring now says ``robust`` is accepted and IGNORED; prove it (the
    old text claimed it selected the legacy forward branch)."""
    mats = _pcore._build_sem(P, DUTY * P, 6.25 + 0j, 1.0 + 0j, 8, 1, 1, False)
    k0 = 2.0 * np.pi / WL
    for pol in ("te", "tm"):
        for kx0 in (0.0, 0.3 * k0):
            a = _pcore._sem_modes(mats, k0, pol, kx0, robust=False)
            b = _pcore._sem_modes(mats, k0, pol, kx0, robust=True)
            for x, y in zip(a, b):
                if x is None:
                    assert y is None
                else:
                    assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_m10_analytic_convolutions_still_returns_the_unused_inverse():
    orders, _n = _twod._harmonic_orders_2d(2, 2)
    EPS, EPS_inv = _twod._analytic_convolutions_2d(1.0, DISK, orders, 2, 2,
                                                   P, P)
    assert EPS.shape == EPS_inv.shape == (25, 25)
    # it is [[1/eps]] (a Laurent transform), NOT the inverse-rule operator
    assert not np.allclose(EPS_inv, np.linalg.inv(EPS))
    assert "NOT the inverse-rule operator" in \
        _twod._analytic_convolutions_2d.__doc__


def test_m10_nv_field_wedge_arm_is_documented_as_study_only():
    """The arm no public entry point reaches: kept as the closed-form
    cross-check, so it must keep WORKING and keep saying so."""
    Nx, Ny = _twod._nv_field_2d(_square(32), 1.0, 1.0, method="xy_wedge")
    assert np.allclose(np.hypot(Nx, Ny), 1.0)
    # whitespace-normalised: 3.13+ de-indents docstrings at compile time
    doc = " ".join(_twod._nv_field_2d.__doc__.split())
    assert "NO public entry point selects it" in doc
    with pytest.raises(ValueError, match="method must be"):
        _twod._nv_field_2d(_square(32), 1.0, 1.0, method="nope")
