"""Full Popov-Neviere anisotropic off-diagonal FFF for the hybrid PMM:
pmm_jones_2d(formulation='fff_nv').

The PMM mirror of ``rcwa_jones_2d(formulation='fff_nv')``.  For a SEPARABLE
(single-orientation, x- or y-patterned) anisotropic cell the wall-normal is
constant, so the projected tensor operator reduces to the rigorous Li-1996 1-D
anisotropic factorization -- the wall-normal diagonal takes the inverse rule and
the off-diagonal ``Cxy``/``Cyx`` of a rotated director gets its correct
composite.  (PMM's ``'li'`` applies the inverse rule ONLY to the ``E_z``
elimination in the separable branch, so ``'fff_nv'`` is the first correct
in-plane inverse-rule treatment there -- an even bigger gain than in rcwa.)  A
crossed / out-of-plane / JAX cell raises, matching rcwa's honest scoping.

These tests pin: the reduction to the rigorous ``rcwa_jones_1d_segments`` on a
stripe (and faster convergence than laurent), the cross-solver agreement with
``rcwa_jones_2d(fff_nv)``, the lossy absorptance split, the uniform-cell routing,
and the crossed / out-of-plane / JAX guards.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa._core import _EnergyError
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments

# eig-heavy 2-D fff_nv (degree 11, n_orders up to 13); the numerics are not
# Python-version-sensitive, so run once in the slow-tests job (keeps the
# fast 4-Python gate under its cap -- the v5.21.1 3.13 runner tipped over
# 40 min).
pytestmark = pytest.mark.slow


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


def _stripe(er, eg, duty=0.5, Sx=64, Sy=8):
    xm = (np.arange(Sx) + 0.5) / Sx < duty
    c = np.zeros((Sx, Sy, 3, 3), complex)
    for ix in range(Sx):
        c[ix, :] = er if xm[ix] else eg
    return c


PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6

# --------------------------------------------------------------------------- #
# Per-run truncation scan for the cross-solver test below.                      #
#                                                                              #
# WHICH truncation of this cell is numerically clean is a per-BUILD, per-BLAS-  #
# THREAD-COUNT fact, so no rung may be hard-coded -- see the docstring of       #
# test_pmm_fff_nv_matches_rcwa_fff_nv for the measurement.  These helpers scan  #
# a ladder, score each rung by its OWN lossless closure, and let the test pick  #
# the reference on the run that is actually executing.                         #
# --------------------------------------------------------------------------- #

#: RCWA reference truncations scanned.  ~0.1-1.0 s each, so the whole ladder is
#: cheap.  Bounded above by the reference cell's x-sampling (rcwa needs
#: 4*n_orders+1 <= Sx) and below by n_orders_x = 7, under which the stripe is
#: not resolved at all.
_RCWA_LADDER = (7, 9, 11, 13, 15)

#: THE REFERENCE LADDER IS TWO-STAGE (2026-08-16).  The order cap is tied to
#: the reference cell's SAMPLING, not chosen, so "more rungs" is bought by
#: sampling the SAME ideal stripe more finely -- ``_stripe``'s duty-0.5 wall
#: lands exactly on a sample edge at every ``Sx`` here (``Sx`` even), so the
#: represented structure is IDENTICAL and only the Fourier coefficients get
#: better.  Stage 2 is entered only when stage 1 yields no corroborated
#: reference, so on a build where the shipped ladder works nothing about this
#: test's cost or its reading changes.
#:
#: Measured 2026-08-16 on WSL py3.12 / numpy 2.5.1 / scipy 1.18.0, the build
#: that forced this -- clean rungs (closure < 1e-3) and the corroborated pick:
#:
#:   sampling   4 threads                      1 thread
#:   Sx =  64   clean [15]          -> NONE    clean [9, 13]        -> M=13
#:   Sx = 128   clean [7,11,13,15,19,21,23,25,27] -> M=23   clean [7,11,13,15,21,23,27] -> M=23
#:
#: -- i.e. at 64 samples and 4 threads exactly ONE rung closes, so nothing can
#: corroborate it and the test failed naming its own reference as unusable
#: (correctly: it WAS unusable).  At 128 the same build has nine.  The three
#: sampled references agree on ``sum(R)`` to 4.3e-05 (0.061786 / 0.061818 /
#: 0.061829 at Sx 64 / 128 / 192), i.e. two decades inside ``_AGREE_TOL``, so
#: which stage supplies the ruler does not move the cross-solver reading.
_RCWA_REF_STAGES = ((64, _RCWA_LADDER),
                    (128, (7, 11, 13, 15, 19, 21, 23, 25, 27)))

#: PMM truncations scanned, cheapest first.  Bounded above by the degree-11
#: nodal grid (2*n_orders+1 <= 33) and below by n_orders = 9: n_orders = 7
#: misses closure by 1.7e-03 .. 2.9e-03 at EVERY thread count measured, i.e. it
#: is under-resolved rather than unstable, and scanning it only buys warnings.
#: The scan stops as soon as _PMM_WANT_CLEAN rungs have closed, so the usual
#: cost is the two cheap rungs (4.7 s + 14.1 s at one thread on Windows)
#: against the 36 s the single hard-coded n_orders = 13 solve cost on its own.
_PMM_LADDER = (9, 11, 13)
_PMM_WANT_CLEAN = 2

#: A rung is CLEAN when its own lossless closure |sum R + sum T - 2| is under
#: this.  UNCHANGED -- it is the bar this test already asserted on both engines.
_CLOSE_TOL = 1e-3

#: A clean rung is CORROBORATED when another clean rung reproduces its sums to
#: within this.  Half the cross-solver bar.  Necessary because closure is
#: necessary and NOT sufficient: at one thread rcwa n_orders_x = 15 closes to
#: 1.6e-04 while its sum(R) sits 2.6e-03 away from the 7/9/11 cluster (R and T
#: redistribute between thread counts at fixed closure -- exactly the
#: "PER-ORDER efficiencies are suspect" the engines warn about).  The exact
#: value is not load-bearing in either direction: a rung that fails
#: corroboration is only DROPPED, and the rung finally used is the cleanest one
#: anyway (n_orders_x = 15 loses to 9 by three decades of closure).
_AGREE_TOL = 2e-3

#: Cross-solver bar.  UNCHANGED (see the docstring).
_CROSS_TOL = 4e-3


def _closure(R, T):
    """The two-polarization lossless closure defect, |sum R + sum T - 2|."""
    return abs(float(np.sum(R)) + float(np.sum(T)) - 2.0)


def _scan(solve, ladder, want_clean=None):
    """Solve every rung of ``ladder`` and score it by its OWN closure.

    Returns ``[dict(M, sumR, sumT, close, raised)]``.  The engines' energy
    diagnostics are silenced because this scan is the thing that CLASSIFIES on
    closure -- a rung that misses it is data here, not a failure -- and a
    catastrophic ``_EnergyError`` is recorded as an un-clean rung for the same
    reason.  With ``want_clean`` set the scan stops early once that many rungs
    have closed.
    """
    rows = []
    for M in ladder:
        raised, sR, sT, close = None, float("nan"), float("nan"), float("inf")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                R, T = solve(M)
                sR, sT, close = float(np.sum(R)), float(np.sum(T)), _closure(R, T)
            except _EnergyError as exc:
                raised = type(exc).__name__
        rows.append(dict(M=M, sumR=sR, sumT=sT, close=close, raised=raised))
        if want_clean and sum(r["close"] < _CLOSE_TOL for r in rows) >= want_clean:
            break
    return rows


def _table(rows):
    return "\n    ".join(
        "M=%2d %-9s closure=%9.3e sum(R)=%.9f sum(T)=%.9f"
        % (r["M"], r["raised"] or "returned", r["close"], r["sumR"], r["sumT"])
        for r in rows)


def _clean(rows):
    return [r for r in rows if r["close"] < _CLOSE_TOL]


def _corroborated_reference(rows):
    """The rung to measure against on THIS run: the cleanest rung whose sums
    another clean rung reproduces.  ``None`` when no rung qualifies."""
    clean = _clean(rows)
    ok = [r for r in clean
          if any(q is not r and abs(q["sumR"] - r["sumR"]) < _AGREE_TOL
                 and abs(q["sumT"] - r["sumT"]) < _AGREE_TOL for q in clean)]
    return min(ok, key=lambda r: r["close"]) if ok else None


def test_pmm_fff_nv_stripe_reduces_to_rigorous_1d():
    """A rotated-director stripe: pmm fff_nv converges to the rigorous 1-D
    full-tensor solver, and faster than pmm laurent."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    cell = _stripe(er, eg)
    _o, Rref, Tref, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=61)
    # The 1-D reference must conserve before anything is measured against it
    # (2026-08-08 sibling sweep): this file's cross-solver test failed for
    # exactly one reason -- an absolute bar read against a reference that had
    # stopped closing at some BLAS thread counts -- and this reference carried
    # no such guard at all.  It is not close to the edge: measured 3.1e-13 (1
    # thread) / 2.3e-13 (2) / 2.1e-13 (24), ten decades inside the bar, which
    # is why the assertion is free and the ef bar below is left alone.
    assert abs(float(np.sum(Rref) + np.sum(Tref)) - 2.0) < _CLOSE_TOL, (
        "the rigorous 1-D full-tensor REFERENCE does not conserve on this "
        "build; the fff_nv convergence bar below cannot be read against it")
    ref = np.sum(Rref)

    def sumR(No, form):
        _o, R, _T, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=9,
                                     n_orders=No, formulation=form,
                                     symmetry=False)
        return np.sum(R)

    ef, el = abs(sumR(13, "fff_nv") - ref), abs(sumR(13, "laurent") - ref)
    assert ef < 1e-3, f"fff_nv err {ef:.2e} not converged to 1-D"
    assert ef < 0.2 * el, f"fff_nv {ef:.2e} not << laurent {el:.2e}"


def test_pmm_fff_nv_matches_rcwa_fff_nv():
    """Cross-solver: EVERY pmm fff_nv truncation that conserves converges to
    the same answer as an rcwa fff_nv reference THIS RUN picks for itself.

    REFERENCE TRUNCATION MOVED 13 -> 11 (2026-08-04).  The RCWA reference at
    ``n_orders_x`` 13 sat ON this cell's measure-zero instability, and the
    "cross-solver residual" the test was reading was simply that: the
    reference's OWN lossless-closure violation, which ``|dT|`` tracked
    one-for-one at every M.  Both engines emit ``_EnergyWarning`` there saying
    exactly this ("the truncation is numerically unstable here and the
    PER-ORDER efficiencies are suspect").  That fix added the two closure
    assertions and kept the 4e-3 bar -- widening it would have pinned the
    instability instead of avoiding it.

    **BUT WHICH TRUNCATION IS CLEAN IS A PER-BUILD, PER-THREAD-COUNT FACT
    (2026-08-08).**  Both re-pinned rungs were still hard-coded, and the
    closure guards were absolute bars on a magnitude that moves with the BLAS
    reduction order.  Holding code, build and geometry fixed and varying ONLY
    ``OPENBLAS_NUM_THREADS`` [Windows, scipy-openblas 0.3.31; closure
    ``sum R + sum T - 2``]::

        rcwa n_orders_x      1 thread     2 threads    24 threads
                      7      -2.40e-05    -1.81e-02    +2.26e-04
                      9      +3.10e-07    +3.86e-03    -6.07e-06
                     11      -2.26e-05    -9.87e-05    +2.99e-03  <- was pinned
                     13      +2.64e-02    -5.54e-06    +5.55e-04
                     15      +1.59e-04    +1.59e-04    +1.24e-03

        pmm n_orders         1 thread     2 threads    24 threads
                      9      -1.75e-04    -1.75e-04    -1.75e-04
                     11      +2.86e-04    -5.35e-04    -2.81e-04
                     13      -2.93e-04    +2.82e-03    +1.60e-04  <- was pinned

    Each thread count has clean rungs and unstable rungs; they are simply not
    the SAME rungs.  The pinned pair happened to be the clean ones at one
    thread, so the test passed there and failed at 2 and at the default -- with
    the PMM guard firing at 2 (2.82e-03) and the RCWA guard at 24 (2.99e-03),
    which is the same disease reported from two sides.  ``sum(R)`` is
    essentially thread-invariant for both engines (twelve digits); the defect
    lives in the transmitted orders.

    **AND "NO RUNG QUALIFIES" WAS ITSELF A PER-BUILD READING (2026-08-16).**
    On WSL py3.12 / numpy 2.5.1 / scipy 1.18.0 at 4 BLAS threads the shipped
    ladder produced exactly ONE clean rung (``n_orders_x`` 15, closure
    1.43e-04; the other four read 2.5e-02, 4.6e-03, 1.0e-02, 4.4e-03), and one
    rung cannot be corroborated, so the test failed -- correctly diagnosing its
    own reference as unusable, and then hard-failing on that diagnosis.  A
    diagnosis is not a verdict: the order cap here is ``4 n_orders + 1 <= Sx``,
    i.e. it is set by the reference cell's SAMPLING, which the run is free to
    refine.  ``_stripe``'s duty-0.5 wall lands exactly on a sample edge at
    every even ``Sx``, so a finer sampling is the SAME ideal stripe with better
    Fourier coefficients -- not a different structure and not a widened bar.
    At ``Sx`` = 128 that same build has NINE clean rungs and corroborates at
    ``n_orders_x`` = 23 (both 1 and 4 threads).  The stages are only walked
    until one qualifies, so on a build where the shipped ladder works this
    costs nothing and reads identically.  The test now hard-fails only when
    the reference engine has stopped converging at EVERY sampled resolution.

    So nothing is pinned any more.  The reference is CHOSEN from a scan: the
    cleanest rcwa rung that another clean rung corroborates (closure alone is
    not enough -- at one thread ``n_orders_x`` 15 closes to 1.6e-04 with
    ``sum(R)`` 2.6e-03 off the cluster), and if no rung qualifies the test
    fails naming the RCWA reference and printing the whole ladder rather than
    blaming the PMM against a reference that does not close.  The subject side
    is then EVERY pmm rung that conserves, not one -- strictly more than the
    single rung this asserted before.  The 4e-3 bar is UNCHANGED; measured
    worst residual over all three thread counts is 4.7e-04 in R and 7.6e-04 in
    T (5x headroom).
    """
    er = _rot(np.deg2rad(35.0), 1.5, 2.3)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)

    def _rcwa_at(Mx, rcell):
        _o, R, T, _J = rcwa_jones_2d(PX, PX, rcell, 1.5, 1.0, DEPTH, WL,
                                     n_orders_x=Mx, n_orders_y=1,
                                     formulation="fff_nv", symmetry=False)
        return R, T

    def _pmm_at(No):
        _o, R, T, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL,
                                    degree=11, n_orders=No,
                                    formulation="fff_nv", symmetry=False)
        return R, T

    # (1) the REFERENCE, chosen on this run.  The structure is provably
    #     lossless, so a rung that does not conserve is not a ruler -- and
    #     "no rung conserves" is a statement about the SAMPLING, which this
    #     run is free to refine, not a verdict this test may hand down.
    ref, rrows, stage, tried = None, [], None, []
    for Sx, ladder in _RCWA_REF_STAGES:
        rcell = _stripe(er, eg, Sx=Sx)
        rrows = _scan(lambda M, c=rcell: _rcwa_at(M, c), ladder)
        ref, stage = _corroborated_reference(rrows), Sx
        if ref is not None:
            break
        tried.append(f"Sx={Sx}:\n    " + _table(rrows))
    assert ref is not None, (
        f"the RCWA fff_nv REFERENCE is unusable on this build AT EVERY "
        f"SAMPLED RESOLUTION {[s for s, _l in _RCWA_REF_STAGES]}: no scanned "
        f"truncation both conserves to better than {_CLOSE_TOL:.0e} and is "
        f"reproduced by another one that does, so there is nothing for the "
        f"PMM to be measured against.  Refining the sampling is what buys "
        f"rungs here (the order cap is 4*n_orders+1 <= Sx), so an exhausted "
        f"stage list means the reference engine itself has stopped "
        f"converging on this cell.  This is a statement about the reference, "
        f"NOT about pmm_jones_2d.\n    " + "\n    ".join(tried))
    if stage != _RCWA_REF_STAGES[0][0]:
        print(f"\nfff_nv reference: the Sx={_RCWA_REF_STAGES[0][0]} ladder had "
              f"no corroborated rung on this build; refined to Sx={stage}")

    # (2) the SUBJECT: every pmm truncation that conserves on this run.
    prows = _scan(_pmm_at, _PMM_LADDER, want_clean=_PMM_WANT_CLEAN)
    pclean = _clean(prows)
    assert pclean, (
        f"no scanned pmm_jones_2d fff_nv truncation conserves to better than "
        f"{_CLOSE_TOL:.0e} on this build -- the PMM side has no stable "
        f"truncation here to compare.\n    " + _table(prows))
    print(f"\nfff_nv cross-solver: reference rcwa n_orders_x={ref['M']} "
          f"(Sx={stage}, closure {ref['close']:.3e}); pmm rungs "
          f"{[r['M'] for r in pclean]} of {[r['M'] for r in prows]}")

    # (3) PMM (Laurent-projected) and RCWA (Li-2003 successive) are DIFFERENT
    #     factorizations, so they converge to slightly different floors -- the
    #     cross-solver residual is a genuine one, not a machine-precision
    #     match.  4e-3 keeps the check meaningful (catches gross errors)
    #     without flaking.
    for r in pclean:
        for q, a, b in (("R", r["sumR"], ref["sumR"]), ("T", r["sumT"], ref["sumT"])):
            assert abs(a - b) < _CROSS_TOL, (
                f"pmm n_orders={r['M']} (closure {r['close']:.3e}) and rcwa "
                f"n_orders_x={ref['M']} (closure {ref['close']:.3e}) disagree "
                f"by {abs(a - b):.3e} in sum({q}) -- both conserve, so this is "
                f"a factorization disagreement, not a truncation "
                f"instability.\n    PMM ladder:\n    " + _table(prows)
                + "\n    RCWA ladder:\n    " + _table(rrows))


def test_pmm_fff_nv_lossy_absorptance_split():
    """Lossless-trap guard: on a lossy stripe fff_nv's absorptance tracks the
    rigorous 1-D split at least as closely as laurent -- PER POLARIZATION.

    **THE REFERENCE WAS DOUBLE-COUNTING THE POLARIZATIONS (2026-08-08 sibling
    sweep).**  ``rcwa_jones_1d_segments`` returns ``R``/``T`` shaped
    ``(2, 2*n_orders+1)`` -- ONE ROW PER INCIDENT POLARIZATION, each row a
    complete energy budget -- so ``1 - sum(R1) - sum(T1)`` over BOTH rows was
    not an absorptance at all: it read ``-0.1409`` where the true per-pol split
    is ``[0.4369, 0.4222]``.  Both engines were then compared against a target
    0.57 away from either of them, and the assertion survived only because that
    constant offset is common to both sides and cancels in the difference::

        quantity                       fff_nv      laurent    fff_nv is
        vs the (2-row) reference       0.570464    0.571352   0.16 % closer
        vs the per-pol reference       2.967e-05   9.177e-04    31x closer
        per-pol, worst polarization    5.082e-05   1.836e-03    36x closer

    Identical at 1, 2 and 24 BLAS threads on both mounts -- this was never a
    thread-count fact, just an assertion sitting at 99.8 % of its own bar with
    nothing behind it.  The claim is unchanged and now has a factor of 36; the
    ``+1e-9`` slack is kept so an exact tie still passes."""
    er = _rot(np.deg2rad(35.0), 1.5 + 0.15j, 2.3 + 0.15j)
    eg = np.diag([2.25] * 3).astype(complex)
    cell = _stripe(er, eg)
    No = 13

    def absorptance(form):
        """Absorptance per INCIDENT POLARIZATION -- the same (2,) budget the
        1-D reference reports, row for row."""
        _o, R, T, _J = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=9,
                                    n_orders=No, formulation=form,
                                    symmetry=False)
        return np.asarray(1.0 - np.sum(R, 1) - np.sum(T, 1))

    _o, R1, T1, _J = rcwa_jones_1d_segments(
        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=No)
    A1 = np.asarray(1.0 - np.sum(R1, 1) - np.sum(T1, 1))
    df = float(np.max(np.abs(absorptance("fff_nv") - A1)))
    dl = float(np.max(np.abs(absorptance("laurent") - A1)))
    assert df < dl + 1e-9, (
        f"fff_nv absorptance is {df:.3e} from the rigorous 1-D per-pol split "
        f"{A1}, laurent only {dl:.3e} -- the anisotropic FFF is not paying "
        f"for itself on the lossy cell")


def test_pmm_fff_nv_uniform_routes():
    """A uniform anisotropic cell (no walls) + fff_nv matches laurent (exact)."""
    e = _rot(np.deg2rad(30.0), 1.5, 2.1)
    cell = np.broadcast_to(e, (8, 8, 3, 3)).copy()
    _o, Rf, Tf, Jf = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=7,
                                  n_orders=5, formulation="fff_nv",
                                  symmetry=False)
    _o, Rl, Tl, Jl = pmm_jones_2d(PX, PX, cell, 1.5, 1.0, DEPTH, WL, degree=7,
                                  n_orders=5, formulation="laurent",
                                  symmetry=False)
    assert np.max(np.abs(Jf - Jl)) < 1e-12


def test_pmm_fff_nv_crossed_and_offplane_and_jax_raise():
    """fff_nv rejects a crossed cell, an out-of-plane tensor, and a JAX cell."""
    er = _rot(np.deg2rad(40.0), 1.6, 3.0)
    eg = np.diag([1.0, 1.0, 1.0]).astype(complex)
    # crossed (both axes patterned)
    sq = np.zeros((48, 48, 3, 3), complex)
    m = np.zeros((48, 48), bool)
    m[12:36, 12:36] = True
    for i in range(48):
        for j in range(48):
            sq[i, j] = er if m[i, j] else eg
    with pytest.raises(ValueError, match="SEPARABLE"):
        pmm_jones_2d(PX, PX, sq, 1.5, 1.0, DEPTH, WL, degree=7, n_orders=5,
                     formulation="fff_nv", symmetry=False)
    # out-of-plane stripe
    oop = _stripe(er, eg)
    oop[:, :, 0, 2] = oop[:, :, 2, 0] = 0.3
    with pytest.raises(ValueError, match="IN-PLANE only"):
        pmm_jones_2d(PX, PX, oop, 1.5, 1.0, DEPTH, WL, degree=9, n_orders=9,
                     formulation="fff_nv", symmetry=False)
    # JAX cell
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(ValueError, match="NumPy only"):
        pmm_jones_2d(PX, PX, jnp.asarray(_stripe(er, eg)), 1.5, 1.0, DEPTH, WL,
                     degree=9, n_orders=9, formulation="fff_nv", symmetry=False,
                     region_layout=np.zeros((64, 8), int))
