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

#: Isotropic groove permittivity of the CROSS-SOLVER stripe.
#:
#: DELIBERATELY NOT 2.25 (2026-09-10,
#: ``docs/audits/VERIFY_WOOD_LIST_AND_FFFNV_2026_09_10.md`` follow-up A; the
#: same move ``tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py`` made for its
#: own stripe, and for the same reason).  2.25 is simultaneously the director's
#: ordinary permittivity (``no^2``, ``no = 1.5``) and ``n_substrate^2``, so a
#: 2.25 groove makes a block of LAYER modes EXACTLY degenerate with the
#: substrate region's at every truncation; the interface inverse then reads the
#: rounding floor, which is a per-BUILD, per-BLAS-THREAD-COUNT quantity.  Every
#: piece of scanning apparatus this file used to carry existed to survive that.
#:
#: MEASURED 2026-09-10 -- RCWA fff_nv lossless closure over ``_RCWA_LADDER`` at
#: ``Sx`` = 64, on Windows py3.14.6 / numpy 2.4.4 at 1 and at 4 BLAS threads and
#: on WSL py3.12.3 / numpy 2.4.6 at 1::
#:
#:     groove   worst closure over the 5 rungs      rungs under 1e-3
#:     2.25     2.64e-02 / 1.72e-02 / 8.22e-03      4 / 4 / 2  of 5
#:     2.10     5.37e-14 / 4.66e-14 / 5.37e-14      5 / 5 / 5  of 5
#:
#: -- and at 2.10 every ``sum(R)``, every ``sum(T)`` and every PMM rung's
#: closure agree to all nine printed digits across the three configurations,
#: where at 2.25 the SAME rung reads ``sum(R)`` = 0.064400 / 0.061786 / 0.062006
#: on them.  With the coincidence gone the reference needs no search: it is
#: taken at a fixed converged truncation and its theorem is ASSERTED.
_STRIPE_EPS_GROOVE = 2.10

#: RCWA reference ladder, ~0.02 s per rung (``n_orders_y`` = 1).  Bounded above
#: by the cell's x-sampling (rcwa needs ``4*n_orders + 1 <= Sx``) and below by
#: ``n_orders_x`` = 7, under which the stripe is not resolved at all.  Every
#: rung's own lossless closure is now ASSERTED rather than scored: on a
#: non-degenerate cell the Li-1996 theorem holds at all of them, so a rung that
#: misses it means the fixture has re-acquired a coincidence and nothing here
#: may be measured against it.
_RCWA_LADDER = (7, 9, 11, 13, 15)

#: The rung the PMM is measured against.  The HIGHEST the sampling allows
#: (``4*15 + 1 = 61 <= 64``), i.e. the best-converged one: ``sum(R)`` over the
#: ladder runs 0.066481993 (7) / 0.066439317 (9) / 0.066416821 (11) /
#: 0.066403334 (13) / 0.066394475 (15), so the rungs span 8.8e-05 and the
#: reference's own residual truncation error is a few 1e-05 -- an order below
#: the cross-solver residual it is used to measure.  FIXED rather than picked
#: on the run: once every rung closes to 1e-14, "the cleanest rung" is a choice
#: between machine-noise readings and would itself be a per-build fact.
#:
#: HISTORY (kept; the apparatus this replaces was right about its symptom).
#: 2026-08-08 replaced a hard-coded pair of rungs with a per-run closure scan,
#: because WHICH rung was clean moved with ``OPENBLAS_NUM_THREADS``; 2026-08-16
#: added a SECOND sampling stage (``Sx`` = 128) because on WSL py3.12 / numpy
#: 2.5.1 at 4 threads the ``Sx`` = 64 ladder produced exactly one clean rung and
#: one rung cannot be corroborated.  Both readings were real.  Neither cause
#: was the truncation or the sampling: it was the groove sitting on
#: ``no^2 = n_substrate^2`` (see ``_STRIPE_EPS_GROOVE``).  With that detuned,
#: all five rungs close to 1e-14 on all three configurations measured, so the
#: two-stage ladder, the corroboration filter and its tolerance are gone.
_RCWA_REF_ORDERS = 15

#: A rigorous RCWA rung's own Li-1996 lossless closure.  It is EXACT for a
#: lossless stack, so this bar separates "the theorem holds" from "it does
#: not"; it is not a convergence measurement.  Worst reading over the five
#: rungs and the three configurations: 5.373e-14 -- 4.3 decades under this bar,
#: which is itself ~5 decades under the 8.2e-03..2.6e-02 the index-coincident
#: groove produced on the same ladder.
_RCWA_SOUND_CLOSURE = 1e-9

#: PMM truncations scanned, cheapest first.  Bounded above by the degree-11
#: nodal grid (2*n_orders+1 <= 33) and below by n_orders = 9: n_orders = 7
#: misses closure by 1.7e-03 .. 2.9e-03 at EVERY thread count measured, i.e. it
#: is under-resolved rather than unstable, and scanning it only buys warnings.
#: The scan stops as soon as _PMM_WANT_CLEAN rungs have closed, so the usual
#: cost is the two cheap rungs (4.7 s + 14.1 s at one thread on Windows)
#: against the 36 s the single hard-coded n_orders = 13 solve cost on its own.
_PMM_LADDER = (9, 11, 13)
_PMM_WANT_CLEAN = 2

#: A PMM rung is CLEAN when its own lossless closure |sum R + sum T - 2| is
#: under this.  VALUE UNCHANGED; the sizing is now measured (2026-09-10).  The
#: hybrid PMM does NOT reach machine closure here and is not expected to -- its
#: Fourier-projection floor at degree 11 is the residual -- so unlike the RCWA
#: side this stays a truncation bar.  On the detuned fixture the three rungs
#: read 1.524e-04 (9) / 9.499e-05 (11) / 1.897e-04 (13), IDENTICAL to nine
#: digits on Windows at 1 and at 4 BLAS threads and on WSL, so the 5.3x gap to
#: this bar is headroom over a value with no build scatter at all rather than
#: margin inside a spread.  (On the coincident groove the same rungs read
#: 1.746e-04 / 2.856e-04 / 2.935e-04 on Windows-1 but 2.704e-03 / 6.130e-04 /
#: 2.210e-04 on WSL -- rung 9 crossing the bar, which is what the scan existed
#: to absorb.)  A rung over the bar is DROPPED, not a failure; only an empty
#: clean set fails.
_CLOSE_TOL = 1e-3

#: Cross-solver bar.  VALUE UNCHANGED; sized from measurement (2026-09-10).
#: PMM (Laurent-projected) and RCWA (Li-2003 successive) are DIFFERENT
#: factorizations converging to slightly different floors, so the residual is a
#: real quantity, not a machine-precision match.  On the detuned fixture,
#: against ``_RCWA_REF_ORDERS``, the scanned PMM rungs read |d sum R| =
#: 3.006e-04 (9) / 5.777e-04 (11) and |d sum T| = 4.530e-04 / 6.727e-04 --
#: identical to nine digits on all three configurations.  4e-3 therefore sits
#: 5.9x above a build-free measurement and ~15x below the 1e-2-and-up a gross
#: factorization error would give; there is no wider placement that keeps both
#: gaps, and no build spread for the lower one to hide in.
_CROSS_TOL = 4e-3


def _closure(R, T):
    """The two-polarization lossless closure defect, |sum R + sum T - 2|."""
    return abs(float(np.sum(R)) + float(np.sum(T)) - 2.0)


def _sums(R, T):
    """One solve's row for :func:`_table` / :func:`_clean`, minus the ``M``."""
    return dict(sumR=float(np.sum(R)), sumT=float(np.sum(T)),
                close=_closure(R, T), raised=None)


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
    the same answer as the rcwa fff_nv reference, whose own energy theorem is
    asserted at every rung of its ladder.

    **THE FIXTURE WAS THE PROBLEM ALL ALONG (2026-09-10,
    ``docs/audits/VERIFY_WOOD_LIST_AND_FFFNV_2026_09_10.md`` follow-up A).**
    The groove is ``_STRIPE_EPS_GROOVE`` (2.10) now, not the director's own
    ordinary permittivity: 2.25 was simultaneously ``no^2`` and
    ``n_substrate^2``, an EXACT layer<->region mode coincidence whose closure
    defect is a reading of the rounding floor and therefore moves with the
    LAPACK build and the BLAS thread count.  Everything the three history
    entries below describe -- the re-pinning, the per-run closure scan, the
    second sampling stage -- was that one fact seen from three angles.
    Detuned, the whole test is build-free: all five rcwa rungs close to
    <= 5.4e-14 and every ``sum(R)`` / ``sum(T)`` and PMM closure agrees to
    nine digits on Windows py3.14/np2.4.4 at 1 AND at 4 BLAS threads and on
    WSL py3.12/np2.4.6.  So the reference is FIXED again (at
    ``_RCWA_REF_ORDERS``) and its theorem is ASSERTED at every rung instead of
    scored -- which is also the tripwire that fires if the coincidence is ever
    reintroduced.  The two-stage ``_RCWA_REF_STAGES`` ladder, the
    ``_corroborated_reference`` filter and its ``_AGREE_TOL`` are gone; the
    PMM-side scan stays, because that side's ~1e-4 closure is a real Fourier
    floor rather than an instability.  The history below is kept as written --
    every measurement in it stands, only its diagnosis was one level too
    shallow.

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
    eg = np.diag([_STRIPE_EPS_GROOVE] * 3).astype(complex)
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

    # (1) the REFERENCE, at a FIXED converged truncation, with the rigorous
    #     engine's OWN energy theorem asserted at EVERY rung of the ladder.
    #     The structure is provably lossless and Li-1996 closure is exact for
    #     it, so a rung that misses it is not a truncation statement at all --
    #     it means the cell has re-acquired a layer<->region mode coincidence,
    #     which is the one state nothing here may be measured against.
    rrows = [dict(M=M, **_sums(*_rcwa_at(M, _stripe(er, eg, Sx=64))))
             for M in _RCWA_LADDER]
    worst = max(r["close"] for r in rrows)
    assert worst < _RCWA_SOUND_CLOSURE, (
        f"the RCWA fff_nv reference violates its OWN exact Li-1996 lossless "
        f"closure by {worst:.3e} somewhere in {list(_RCWA_LADDER)}, so its "
        f"per-order answers are suspect and there is nothing for the PMM to "
        f"be measured against.  That is a property of the FIXTURE, not of "
        f"either engine: check whether the cell has re-acquired a "
        f"layer<->region mode coincidence (see _STRIPE_EPS_GROOVE).\n    "
        + _table(rrows))
    ref = next(r for r in rrows if r["M"] == _RCWA_REF_ORDERS)

    # (2) the SUBJECT: every pmm truncation that conserves on this run.  This
    #     side really is a truncation scan -- the hybrid PMM's Fourier floor
    #     leaves ~1e-4 of closure at every rung (see _CLOSE_TOL), so "which
    #     rungs are usable" is a genuine question here, unlike (1).
    prows = _scan(_pmm_at, _PMM_LADDER, want_clean=_PMM_WANT_CLEAN)
    pclean = _clean(prows)
    assert pclean, (
        f"no scanned pmm_jones_2d fff_nv truncation conserves to better than "
        f"{_CLOSE_TOL:.0e} on this build -- the PMM side has no stable "
        f"truncation here to compare.\n    " + _table(prows))
    print(f"\nfff_nv cross-solver: reference rcwa n_orders_x={ref['M']} "
          f"(closure {ref['close']:.3e}, ladder worst {worst:.3e}); pmm rungs "
          f"{[r['M'] for r in pclean]} of {[r['M'] for r in prows]}")

    # (3) PMM (Laurent-projected) and RCWA (Li-2003 successive) are DIFFERENT
    #     factorizations, so they converge to slightly different floors -- the
    #     cross-solver residual is a genuine one, not a machine-precision
    #     match.  See _CROSS_TOL for the two-sided sizing of the 4e-3.
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
