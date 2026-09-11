"""O-11 ROUND 4 -- the sliver guard decides on ANSWERS, not on an energy
reading (2026-09-11).

WHAT THE RELEASE CI MATRIX SHOWED.  Rounds 1-3 built the guard around
``max R+T``: round 1 refused above ``_STACK_SUPERUNITY_BAR``, round 2 lowered
the gate to ``_SLIVER_TRIGGER_BAR`` and made the ATTRIBUTION the same reading
taken again on a snapped grid, round 3 made that second reading relative.  The
5.45.0 release matrix failed twenty distinct tests of this family, on
different pythons for different tests, because that reading is amplified
rounding through a ``1/w^2``-conditioned interface.  Measured on this box, the
SAME fixture row (degree 14, ``delta`` = 1e-4 of a 1.2 um period):

    kernel        max R+T      err / delta      continuity
    Haswell       2.17298      4789             WRONG
    Prescott      2.17297      4789             WRONG
    Nehalem       3.61242      9301             WRONG
    Sandybridge   1.00000      1.155            RIGHT

-- and CI's own kernel read it RIGHT at ``R+T`` = 1.000115, which is why
``test_fail_before_the_pre_fix_path_only_warns_it_does_not_refuse`` captured
no warning at all there and three sibling tests failed with DID NOT RAISE.

WHAT ROUND 4 CHANGES.  Two things, and neither is a bar:

1.  THE TRIGGER IS THE GEOMETRY.  The guard arbitrates whenever the geometric
    screen finds a MANUFACTURED cross-layer sliver -- a union cell no single
    layer asked for, at least ``_SLIVER_OWN_SCALE_RATIO`` times finer than the
    finest wall spacing any layer did ask for.  That is a deterministic fact
    about the wall coordinates and the ``min_feature``, identical on every
    kernel.  It no longer matters what the solve reads.
2.  THE ATTRIBUTION IS A COMPARISON OF ANSWERS.  Three extra solves: the
    prescribed ``min_feature = 2 w_wide P`` grid (``A_M``), the same walls
    CLOSED onto one coordinate (``A_L``), and that closed wall DISPLACED by
    one widest manufactured cell (``A_L'``).  ``d0 = |A_sliver - A_M|`` is the
    shipped move; ``d12 = |A_L - A_L'|`` is THIS DEVICE's own answer change
    for a wall displacement of exactly the sliver's size.  The sliver is the
    attributed cause when ``d0`` passes both the geometric floor
    (``_SLIVER_MOVE_FACTOR`` widest cells) and ``_SLIVER_WALL_RATIO`` times
    ``d12``.

WHAT THIS FILE ASSERTS.  Two kinds of thing.

* RUNTIME, on the building build: the two bars, two-sided, with each
  population measured here; and the arbiter's contract.
* CROSS-ARM, from the committed decision tables in ``validation/
  probe_fix_sliver_round4/p4_decisions_<build>_<coretype>_t<threads>.json``:
  that no arm returns a wrong answer in silence, that no arm refuses a right
  one, and that WHERE THE ANSWER IS THE SAME ON TWO ARMS THE DECISION IS TOO.

AN ARM IS TWO AXES, NOT ONE.  The kernel is the axis the CI matrix made
visible.  It is not, on its own, the axis that reaches CI: the 5.45.0 matrix
runs on GitHub-hosted ``ubuntu-latest`` = AMD EPYC 7763 (Zen 3), which has no
AVX-512 and therefore dispatches to the SAME Haswell-class kernel this box
defaults to -- the aliasing measured here for ``OPENBLAS_CORETYPE=ZEN``.  What
CI does differently is leave BLAS UNPINNED on a four-core runner while every
file in this family pins one thread, and the thread count is a hazard this
module has already been bitten by: ``test_m1_conditioning_guard.py`` records a
closure moving seven decades on the SAME cell between one and two OpenBLAS
threads.  So the decision tables are measured on BOTH ladders -- kernels
{Haswell, Katmai, Nehalem, Sandybridge} x threads {1, 2, 4, unpinned} x two
builds -- and the pin is asserted along each of them separately.

That last clause is the honest form of "the decision is build-independent",
and the round-4 audit states why the unconditional form is not achievable: in
the conditioning-collapse band the ANSWER itself is a property of the kernel
(the table above), so a guard that made the same decision there would have to
be ignoring the answer -- which is precisely the defect being repaired.  What
round 4 can and does guarantee is that the guard ADDS no kernel dependence of
its own, and that its residual disagreements are exactly the rows on which the
library's own answers disagree.

Evidence: ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND4_2026_09_11.md`` and
``validation/probe_fix_sliver_round4/``.
"""
import json
import os

# The sliver fixture is a near-degenerate eigenproblem: the classification the
# tests read must not move with the BLAS reduction order, so pin one thread
# before numpy is imported (the pattern of the six sibling sliver files).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir,
                                      "validation",
                                      "probe_fix_sliver_round4"))

# ---- the O-11 fixture, verbatim from the three sibling fix files ----------
_P = 1.2e-6
_WL = 0.85e-6
_THETA = 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505
_NO_SNAP = _P * 1e-12


def _stack(delta, degree=14, *, nl=2, min_feature=None, n_sup=1.0, n_sub=1.0,
           eps=_EP, theta=_THETA, ffo=None):
    kw = {} if ffo is None else dict(far_field_orders=ffo)
    st = PMMStack(_P, n_superstrate=n_sup, n_substrate=n_sub, degree=degree,
                  min_feature=(_NO_SNAP if min_feature is None
                               else min_feature), **kw)
    for k in range(nl):
        d = delta * k / max(nl - 1, 1)
        a, b = _A0 - d, _B0 + d
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, eps), (1.0 - b, _EH)])
    st.set_source(_WL, theta=theta)
    return st


def _raw(st):
    """The unguarded solve -- the pre-fix code path, bit for bit."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    i = np.argsort(np.asarray(o).ravel())
    R = np.real(np.asarray(R))[:, i]
    T = np.real(np.asarray(T))[:, i]
    return (np.asarray(o).ravel()[i], R, T,
            float(np.max(R.sum(axis=-1) + T.sum(axis=-1))))


def _move(a, b, *, pol=None):
    """Max ``|dR|``, ``|dT|`` over the orders two solves share.  ``pol=None``
    is what the arbiter compares; ``pol=1`` is the campaign's convention."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(a[1][sl][:, ia] - b[1][sl][:, ib]).max(),
                     np.abs(a[2][sl][:, ia] - b[2][sl][:, ib]).max()))


def _kind(e, d):
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


# ==========================================================================
# THE ARMS: the committed decision tables
# ==========================================================================
def _tables():
    """Every committed per-arm decision table, keyed by
    ``(build, kernel, requested, threads_requested)``.

    The ARM is a pair of axes, not one.  The kernel is the axis the CI matrix
    made visible; the THREAD COUNT is the axis that actually separates this
    box from CI, because CI's runners are EPYC 7763 (Zen 3, no AVX-512) and
    therefore dispatch to the same Haswell-class kernel as this box's default
    -- what CI does differently is leave BLAS unpinned on four cores while
    every file in this family pins one thread."""
    out = {}
    for fn in sorted(os.listdir(_PROBE)):
        if not (fn.startswith("p4_decisions_") and fn.endswith(".json")):
            continue
        with open(os.path.join(_PROBE, fn), encoding="utf-8") as fh:
            d = json.load(fh)
        arm = d["arm"]
        out[(arm["build"], arm["kernel"], arm["requested"],
             str(arm["threads_requested"]))] = d
    return out


def _rows(table):
    return {(r["case"], "%.17g" % r["delta"]): r for r in table["rows"]}


def test_the_decision_tables_cover_the_arms_this_round_was_measured_on():
    """The cross-arm pin is only worth what its arms are worth, so the arms
    are asserted rather than assumed: two BUILDS (different python, different
    numpy, different LAPACK) and at least four DISTINCT OpenBLAS kernels.

    ``requested`` is what ``OPENBLAS_CORETYPE`` asked for and ``kernel`` is
    what OpenBLAS actually dispatched to.  They differ, and round 4's first
    finding is that they differ for ``ZEN``: the bundled
    ``scipy-openblas 0.3.31`` maps ``ZEN`` onto its Haswell kernel on this
    CPU, bit for bit, so an arm requested as ZEN is not a distinct arm.  The
    audit records that; this test counts the DISTINCT kernels, which is the
    quantity that carries the claim.

    ``threads_requested`` is the SECOND ladder, and the one that reaches CI.
    CI's runners are EPYC 7763 -- Zen 3, no AVX-512 -- so they dispatch to
    the same Haswell-class kernel this box defaults to; what they do
    differently is leave BLAS UNPINNED on four cores, while every sliver test
    file and every probe here pins one thread.  Since this module's own
    sibling ``test_m1_conditioning_guard.py`` records a closure moving seven
    decades between one and two OpenBLAS threads, the thread count is a known
    hazard in this code and not a speculative axis, so the arms must cover it
    explicitly: at least {1, 2, 4, UNPINNED}."""
    t = _tables()
    assert len(t) >= 26, sorted(t)
    assert len({k[0] for k in t}) >= 2, sorted(t)
    assert len({k[1] for k in t}) >= 4, sorted(t)
    threads = {k[3] for k in t}
    assert {"1", "2", "4", "UNPINNED"} <= threads, sorted(threads)
    # and the two ladders must actually be CROSSED, not run one at a time:
    # every distinct kernel carries the PINNED ladder on both builds, and the
    # UNPINNED rung is present on both builds.  The unpinned rung is not
    # required at every kernel, and the reason is a measurement, not a
    # shortcut: this box has 24 hardware threads, so "unpinned" here means 24
    # threads on eigenproblems a few hundred wide, which is >30x slower than
    # the same run pinned -- while CI's unpinned lane is a FOUR-core runner,
    # i.e. the ``t4`` arm, which IS measured in full at every kernel.
    for build in sorted({k[0] for k in t}):
        for kern in sorted({k[1] for k in t if k[0] == build}):
            got = {k[3] for k in t if k[0] == build and k[1] == kern}
            assert {"1", "2", "4"} <= got, (build, kern, sorted(got))
    # the UNPINNED rung: present on BOTH builds, and present at EVERY distinct
    # kernel on at least one of them.  The asymmetry is a property of the two
    # OpenBLAS builds, measured, not a gap: on Linux an unpinned OpenBLAS
    # sizes its pool from the CPUs the process can SEE, so narrowing the
    # affinity to four reproduces CI's four-core runner exactly (read back:
    # ``num_threads`` = 4) and the full table runs at ``t4`` speed.  The
    # Windows build ignores the affinity mask and takes all 24 hardware
    # threads whatever it can see, which on eigenproblems this size does not
    # finish -- so the Windows unpinned arm is a declared SUBSET, and it is
    # the extreme rather than the CI configuration.
    unpinned = {k for k in t if k[3] == "UNPINNED"}
    assert {k[0] for k in unpinned} == {k[0] for k in t}, sorted(unpinned)
    assert ({k[1] for k in unpinned} == {k[1] for k in t}), sorted(unpinned)
    # Every arm runs the SAME rows, with one declared exception.  An arm may
    # carry a REDUCED row set only if it says so in its own JSON (``subset``
    # names the case prefixes it kept), and then its rows must be a subset of
    # the full table rather than a different table.  The exception exists for
    # exactly one configuration: with all three thread variables REMOVED,
    # OpenBLAS takes all 24 hardware threads of this box and these small
    # spectral-element eigenproblems spend their time in thread launch --
    # measured >30x slower than the same run pinned, so the full table does
    # not finish.  CI's unpinned lane is a FOUR-core runner, which is the
    # ``t4`` arm, and that one IS measured in full.
    full = {k: v for k, v in t.items() if not v.get("subset")}
    assert len(full) >= 24, sorted(full)
    ref = None
    for key, table in sorted(full.items()):
        rows = _rows(table)
        assert len(rows) >= 100, (key, len(rows))
        if ref is None:
            ref = set(rows)
        else:
            assert set(rows) == ref, (key, len(set(rows) ^ ref))
    for key, table in sorted(t.items()):
        if not table.get("subset"):
            continue
        rows = set(_rows(table))
        assert 9 <= len(rows) < len(ref), (key, len(rows))
        assert rows <= ref, (key, sorted(rows - ref)[:5])


#: How far past the DEVICE'S OWN answer change per unit wall displacement a
#: returned row may sit and still be allowed to be silent.  It is the
#: campaign's RIGHT cutoff, ``err <= 10 delta``, with ``delta`` replaced by
#: what moving that wall actually does to this device -- round 3's own
#: slope-normalised continuity rule, one step looser than its factor of 3.
_SLOPE_RIGHT = 10.0


def test_no_arm_returns_a_wrong_answer_in_silence():
    """THE SOUNDNESS PROPERTY, on every arm, and the one place it has to be
    stated against the device's own slope rather than against the wall step.

    A row the ABSOLUTE continuity rule calls WRONG -- ``err > 100 delta`` on
    BOTH polarizations against the exact ``delta -> 0`` limit, on that arm's
    own numbers -- may be REFUSED, and it may be RETURNED under a warning that
    names the wall sensitivity.  It may be returned with nothing said ONLY if
    it is RIGHT by the SLOPE-NORMALISED rule, i.e. its error is within
    ``_SLOPE_RIGHT`` times what displacing that wall by one sliver width does
    to this device anyway.

    That exception is not a loophole, it is the absolute rule's own documented
    limit.  The round-2 verification records that the 10x/100x rule embeds
    ``dR/dx`` = O(1) -- on its steepest fixture, ``S_steep``, the rule has NO
    correct rows at all -- and round 3 introduced a slope-normalised form for
    exactly this reason.  Measured over 558 screened rows of fourteen devices
    on this box: 21 rows are returned silently, every one of them on the two
    deliberately steep counter-fixtures (a guided-mode resonance sitting on
    its own resonance and a 12-slice taper), their absolute ``err/delta`` runs
    106 .. 221 against a MEASURED ``dR/dx`` of 26 .. 146, and the worst
    slope-normalised score among them is **4.03** -- against this bar of 10,
    i.e. 2.48x -- while the population the guard ATTRIBUTES starts at 15.34 on
    the same quantity."""
    bad = []
    for key, table in sorted(_tables().items()):
        for k, r in sorted(_rows(table).items()):
            if r.get("returned_kind") != "wrong" or r["decision"] != "returned":
                continue
            d12, w = r.get("d12"), r.get("w_wide")
            slope = (d12 / w) if (d12 and w) else None
            score = (r["returned_eod_both"] / slope) if slope else float("inf")
            if score > _SLOPE_RIGHT:
                bad.append((key, k, r["returned_eod_both"], slope, score))
    assert not bad, bad[:10]


def test_no_arm_refuses_a_right_answer():
    """The other side of the same statement, and the one round 1 failed on
    110 of 648 realistic staircases."""
    bad = []
    for key, table in sorted(_tables().items()):
        for k, r in sorted(_rows(table).items()):
            if r["kind"] == "right" and r["decision"] == "refused":
                bad.append((key, k, r["eod_both"], r["worst"]))
    assert not bad, bad[:10]


#: When two arms are said to AGREE ABOUT THE ANSWER.  Same continuity class is
#: not enough -- "wrong by 180x the wall shift" and "wrong by 259,000x" are
#: both WRONG and are not the same answer.  The second condition is that their
#: distances from the exact ``delta -> 0`` limit agree within this factor,
#: which is the campaign's own RIGHT cutoff used as an agreement window.
_ANSWER_AGREE = 10.0


def test_the_decision_is_the_same_on_every_arm_wherever_the_answer_is():
    """THE CROSS-ARM PIN.

    For every row on which all arms agree about the ANSWER -- the same
    continuity class, and distances from the exact ``delta -> 0`` limit within
    ``_ANSWER_AGREE`` of one another -- the guard's decision must be the same
    on every one of them too.  Where the arms do not agree about the answer
    the decision is allowed to differ, and the row is COUNTED rather than
    asserted: those rows are the library's own conditioning collapse, not the
    guard's arbitration, and the count is in the failure message so a
    regression that widens it is visible.

    The pin is taken TWICE: once over the 24 arms that carry the whole
    table, and once over ALL arms -- the declared subset arm included -- on
    the rows they share.  Measured 2026-09-11: over the 24 full arms, 68 of
    123 rows are answer-stable at this window and the decision is identical
    on every one of them; over all 29 arms, 7 of the 15 shared rows are, and
    likewise.  The rows that are not answer-stable are the
    conditioning-collapse band, where the unguarded answers differ between
    arms by up to three decades in ``err/delta`` (measured: 179.6 against
    2.589e+05 on one row of the ``B_vis`` ladder at ``delta`` = 3e-06)."""
    tables = _tables()
    assert len(tables) >= 8, sorted(tables)
    full = {k: v for k, v in tables.items() if not v.get("subset")}
    for label, group, floor in (("full arms", full, 40),
                                ("all arms", tables, 5)):
        keys = sorted(set.intersection(*[set(_rows(t))
                                         for t in group.values()]))
        rows = {k: {a: _rows(t)[k] for a, t in group.items()} for k in keys}
        unstable, disagree, stable = [], [], 0
        for k in keys:
            kinds = {r["kind"] for r in rows[k].values()}
            eods = [r["eod_both"] for r in rows[k].values()]
            decisions = {r["decision"] for r in rows[k].values()}
            agree = (len(kinds) == 1
                     and max(eods) <= _ANSWER_AGREE * max(min(eods), 1e-300))
            if not agree:
                unstable.append((k, sorted(kinds), sorted(decisions)))
                continue
            stable += 1
            if len(decisions) > 1:
                disagree.append((k, sorted(kinds),
                                 {a: r["decision"]
                                  for a, r in rows[k].items()}))
        assert stable >= floor, (label, stable, len(keys))
        assert not disagree, (
            "%s: the guard decided differently on arms that agree about the "
            "answer: %r (%d answer-stable, %d answer-unstable rows in this "
            "set)" % (label, disagree[:6], stable, len(unstable)))


def test_the_decision_does_not_move_with_the_thread_count():
    """THE SECOND CROSS-ARM PIN, and the one that actually reaches CI.

    The kernel ladder is not enough on its own, and the reason is worth
    stating rather than assuming.  The 5.45.0 matrix runs on GitHub-hosted
    ``ubuntu-latest`` = AMD EPYC 7763 (Zen 3).  Zen 3 has no AVX-512, so
    ``scipy-openblas`` dispatches it to the SAME Haswell-class kernel this box
    defaults to -- exactly the aliasing the sibling test above records for
    ``OPENBLAS_CORETYPE=ZEN``.  CI is therefore NOT on a kernel this box
    cannot reach.  What CI has that rounds 1-3 never varied is the THREAD
    COUNT: its fast unit lane leaves BLAS unpinned on four cores, while every
    file in this family pins ``OMP/OPENBLAS/MKL_NUM_THREADS=1``.

    That axis is a known hazard in this very module, not a speculative one:
    ``tests/unit/test_m1_conditioning_guard.py`` records a closure residual
    moving from ``6.65e-06`` to ``2.14e+01`` on the SAME cell between one and
    two OpenBLAS threads.  A guard keyed on a numeric reading is exposed to
    the thread count in the same way it is exposed to the kernel.

    So this test holds the BUILD and the dispatched KERNEL fixed and varies
    only the thread count {1, 2, 4, unpinned}, and requires the decision to be
    identical wherever the arms agree about the answer -- the same rule the
    kernel pin uses, so that neither ladder is judged more leniently than the
    other.  Rows on which the answer itself moves with the thread count are
    COUNTED and reported, because a nonzero count is a fact about the
    library's conditioning that a later round has to be able to see.

    Measured 2026-09-11 over the committed arms, eight ladders (two builds x
    four dispatched kernels): 828 answer-stable rows, **0** decisions
    differing, 48 answer-unstable.  The answer itself moves with the thread
    count on 101 to 123 of the 123 rows of every ladder, and in the collapse
    band it moves a long way -- ``err/delta`` spreads to **3.4e+04** between
    one, two and four threads at a fixed build and kernel.  Nehalem is the
    one kernel whose classification does not move at all."""
    tables = _tables()
    groups = {}
    for key, d in tables.items():
        groups.setdefault((key[0], key[1]), []).append((key, d))
    ladders = [g for g in groups.values() if len(g) >= 3]
    assert len(ladders) >= 8, sorted(
        (k, len(v)) for k, v in groups.items())
    stable, unstable, disagree = 0, 0, []
    for group in ladders:
        keys = sorted(set.intersection(*[set(_rows(d)) for _, d in group]))
        for k in keys:
            rs = [(key, _rows(d)[k]) for key, d in group]
            kinds = {r["kind"] for _, r in rs}
            eods = [r["eod_both"] for _, r in rs]
            if not (len(kinds) == 1
                    and max(eods) <= _ANSWER_AGREE * max(min(eods), 1e-300)):
                unstable += 1
                continue
            stable += 1
            decisions = {r["decision"] for _, r in rs}
            if len(decisions) > 1:
                disagree.append((k, {key[2] + "/t" + key[3]: r["decision"]
                                     for key, r in rs}))
    assert stable >= 400, (stable, unstable)
    assert not disagree, (
        "the guard decided differently at two thread counts of the SAME "
        "build and the SAME dispatched kernel: %r (%d answer-stable, %d "
        "answer-unstable rows)" % (disagree[:6], stable, unstable))

    # ... and the positive half, which is what makes the pin worth having:
    # the ENERGY READING rounds 1-3 decided on moves by more than two decades
    # across this ladder alone -- same build, same kernel, only the thread
    # count changed -- on rows round 4 decides identically.  Measured
    # 2026-09-11: 1307x on ``ladder:C_nir`` at degree 20 (Sandybridge),
    # 452.6x on ``ladder:B_vis`` at degree 14 (Haswell), then 263.4x, 143.1x
    # and 140.2x -- and the same numbers on both builds to four figures.
    spread = []
    for group in ladders:
        keys = sorted(set.intersection(*[set(_rows(d)) for _, d in group]))
        for k in keys:
            rs = [_rows(d)[k] for _, d in group]
            if len({r["decision"] for r in rs}) != 1:
                continue
            v = [abs(r["worst"] - 1.0) for r in rs]
            if min(v) > 0.0:
                spread.append((max(v) / min(v), k))
    assert spread, "no row is common to a whole thread ladder"
    spread.sort(reverse=True)
    assert spread[0][0] > 100.0, (
        "the energy reading moves by less than two decades across the thread "
        "ladder on every identically-decided row (worst %.4g at %r); the "
        "thread arms are then not exercising the defect"
        % (spread[0][0], spread[0][1]))


def test_the_energy_reading_is_arm_dependent_where_the_decision_is_not():
    """The round's own premise, measured across the committed arms: there are
    rows whose ``max R+T`` moves by DECADES between kernels while the guard
    decides them identically.  Rounds 1-3 keyed both the trigger and the
    attribution on that reading, so on those rows they could not have."""
    tables = {k: v for k, v in _tables().items() if not v.get("subset")}
    keys = sorted(set.intersection(*[set(_rows(t)) for t in tables.values()]))
    rows = {k: {arm: _rows(t)[k] for arm, t in tables.items()} for k in keys}
    spread = []
    for k in keys:
        decisions = {r["decision"] for r in rows[k].values()}
        if len(decisions) != 1:
            continue
        v = [abs(r["worst"] - 1.0) for r in rows[k].values()]
        lo, hi = min(v), max(v)
        if lo > 0.0:
            spread.append((hi / lo, k, sorted(decisions)))
    assert spread, "no row is common to every arm"
    spread.sort(reverse=True)
    # measured 2026-09-11 over the committed arms: 3015 on one ``B_vis`` row,
    # 1490 on an O-11 one, and five rows above 600.  The bar is two decades.
    assert spread[0][0] > 100.0, (
        "the energy reading moves by less than two decades on every "
        "identically-decided row of this set (worst %.4g at %r); the arms are "
        "then not exercising the defect" % (spread[0][0], spread[0][1]))


# ==========================================================================
# THE TWO BARS, two-sided and re-derived on the running build
# ==========================================================================
def test_the_two_bars_are_two_sided_on_the_running_build():
    """``docs/TESTING_STANDARDS.md`` rule 5.  Both criteria, both populations,
    measured here rather than pinned:

    * ``_SLIVER_MOVE_FACTOR`` is a GEOMETRIC FLOOR -- the campaign's own
      ``err > 100 delta`` WRONG rule, applied to the answer move in units of
      the widest manufactured cell.  Every CORRECT row must sit under it.
    * ``_SLIVER_WALL_RATIO`` is the same 100x applied to a MEASURED
      denominator: the device's own answer change when the contested wall is
      displaced by one widest cell.  It is what separates a corrupted answer
      from a device that is genuinely sensitive to that wall -- the class the
      round-2 verification found at ``move/w_wide`` = 833.78 and the round-3
      verification at 906.56, both on CORRECT answers.

    The assertion is the DECISION plus the separation between the two
    populations this test measured, never a multiple of either constant."""
    right, wrong = [], []
    for deg in (12, 14, 20):
        ref = _raw(_stack(0.0, deg))
        for d in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
            st = _stack(d, deg)
            cur = _raw(st)
            got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
            if got is None or got[1] is None:
                continue
            kind = _kind(_move(cur, ref), d)
            ev = got[1]
            rec = (deg, d, ev["d0_over_w"], ev["d0_over_d12"], got[0])
            if kind == "right":
                right.append(rec)
            elif kind == "wrong":
                wrong.append(rec)
    assert len(right) >= 4, (right, wrong)
    attributed = [r for r in wrong if r[4] == "sliver"]
    # INVARIANT, and it is the one users feel: no CORRECT row reaches either
    # bar, so none of them can be attributed.  True on every arm, wrong rows
    # present or not.
    assert max(r[2] for r in right) < ps._SLIVER_MOVE_FACTOR, right
    # PREMISE-GATED.  PREMISE-GATED 2026-09-11 (CI PREMISE GATES): the CI runner arm solves these ill-conditioned fixtures CORRECTLY where every local arm solves them wrong, so a population of WRONG rows is a reading of the running arm's arithmetic and not a property of the library.  It is measured and skipped with the reading when absent, never asserted.  See docs/audits/CI_PREMISE_GATES_2026_09_11.md.
    if len(wrong) < 4 or len(attributed) < 4:
        pytest.skip(
            "premise absent on this arm: the ladder produces %d WRONG "
            "arbitrated rows (%d of them attributed to the sliver) against "
            "the 4 this separation claim needs, beside %d CORRECT ones.  The "
            "CORRECT side -- no right row reaches the move bar -- was "
            "asserted above.  wrong = %s"
            % (len(wrong), len(attributed), len(right), wrong))
    # ... every attributed row passes both ...
    assert min(r[2] for r in attributed) > ps._SLIVER_MOVE_FACTOR, attributed
    assert min(r[3] for r in attributed) > ps._SLIVER_WALL_RATIO, attributed
    # ... and the two populations are a decade apart on each quantity, which
    # is a property of the populations rather than of the constants.
    assert (min(r[2] for r in attributed)
            > 10.0 * max(r[2] for r in right)), (attributed, right)
    assert (min(r[3] for r in attributed)
            > 10.0 * max(r[3] for r in right)), (attributed, right)


def test_the_sensitivity_denominator_is_a_real_measurement_not_round_off():
    """``d12`` is only a criterion if it is a SENSITIVITY.  Two ways it could
    fail to be, both checked:

    * if the second sliver-free grid were the same device merely TRANSLATED,
      ``d12`` would be round-off and every move would beat it.  Closing the
      walls LEFT and closing them RIGHT is exactly that degenerate pair on any
      symmetric wall opening -- measured here, the two answers agree to
      1e-6 of what the shift produces;
    * if the displacement were not applied at all, ``d12`` would be zero.

    So the shift is what makes the denominator real, and it is applied to ONE
    wall group, which changes the ridge WIDTH rather than the position."""
    st = _stack(1e-4, 14)
    segs = [L[1] for L in st._layers]
    mf = float(st.min_feature) / _P
    hit = ps._cross_layer_sliver(segs, mf)
    assert hit is not None
    w = hit[3]

    def _solve_of(sg):
        clone = st._min_feature_clone(float(st.min_feature))
        clone._layers = [(L[0], sg[i], L[2])
                         for i, L in enumerate(st._layers)]
        clone._src = dict(st._src)
        return _raw(clone)

    left = _solve_of(ps._sliver_collapsed_segments(segs, mf, "left"))
    right = _solve_of(ps._sliver_collapsed_segments(segs, mf, "right"))
    shift = _solve_of(ps._sliver_collapsed_segments(segs, mf, "left", w))
    d_side = _move(left, right)
    d_shift = _move(left, shift)
    assert d_shift > 0.0, d_shift
    # measured 2026-09-11 on this fixture: 1.99e-09 against 4.52e-04, i.e.
    # 4.4e-06 -- the bar is 1e-04, which carries 23x, and the claim is the
    # DECADES between them rather than the value.
    assert d_side < 1e-4 * d_shift, (d_side, d_shift)
    # and the arbiter uses the second, not the first
    cur = _raw(st)
    _v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
    assert abs(ev["d12"] - d_shift) < 1e-12 * max(d_shift, 1e-30), (ev, d_shift)


def test_the_arbiter_is_reached_at_every_reading_including_exactly_unity():
    """The trigger removal, asserted at the two readings that matter: exactly
    unity (which rounds 1-3 returned from before doing anything) and a
    SUB-unity total (which no super-unity test can see at all -- open item
    R2-A, where the verification measured 11 sub-unity WRONG rows and 29 of
    878 wrong rows at or below the trigger)."""
    st = _stack(1e-4, 14)
    cur = _raw(st)
    for reading in (1.0, 0.5, 1.0 + ps._SLIVER_TRIGGER_BAR):
        got = ps._sliver_arbiter(st, reading, cur[1], cur[2], None)
        assert got is not None, reading
        assert got[0] in ("sliver", "wall", "truncation"), (reading, got[0])
        assert got[1]["violation"] == max(reading - 1.0, 0.0), (reading,
                                                               got[1])


def test_an_answer_that_agrees_with_the_sliver_free_grids_is_returned():
    """THE CI CASE, stated as a contract rather than as a row.

    Measured on the 5.45.0 runners (EPYC 7763, unpinned, py3.10-3.13): the
    two-layer O-11 fixture at degree 12 with ``delta`` = 1e-5 and
    ``far_field_orders`` = 5 solves to ``max R+T`` = 1.0000010 and the answer
    is (near) CORRECT.  Every arm of this box reads 2.17 or 3.61 on the same
    geometry and is WRONG by 4.8e+04 .. 9.3e+04 times the wall shift, and the
    ``rcond`` of the interface system agrees to 0.03 % across all of them --
    so the divergence is in the ANSWER, not in the conditioning, and no local
    arm reproduces CI's numerics.

    The guard therefore MUST NOT refuse on CI what it refuses here, and the
    property that makes that true is the one this test pins: the arbiter's
    FIRST question is whether the shipped answer agrees with the grids that
    carry no sliver, and when it does the verdict is ``truncation`` and the
    answer is RETURNED -- whatever the solve read.  Two halves:

    * every screened row whose ``d0`` is under the geometric floor is
      ``truncation`` and comes back from ``solve()`` without raising;
    * the verdict on every screened row is IDENTICAL when the arbiter is
      handed a reading of 0.5, of exactly 1.0, and the one actually measured.
      A reading of 1.0000010 is therefore not a special case: no reading is.
    """
    floor = ps._SLIVER_MOVE_FACTOR
    small, checked = 0, 0
    for deg in (12, 14):
        for d in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
            st = _stack(d, deg)
            if ps._sliver_screen(st, require_passive=False) is None:
                continue
            cur = _raw(st)
            verdicts = set()
            ev = None
            for reading in (0.5, 1.0, cur[3]):
                got = ps._sliver_arbiter(_stack(d, deg), reading,
                                         cur[1], cur[2], None)
                assert got is not None, (deg, d, reading)
                verdicts.add(got[0])
                ev = got[1]
            checked += 1
            assert len(verdicts) == 1, (deg, d, sorted(verdicts))
            verdict = verdicts.pop()
            assert ev is not None, (deg, d)
            agrees = ev["d0_over_w"] <= floor
            if agrees:
                small += 1
                assert verdict == "truncation", (deg, d, verdict, ev)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = _stack(d, deg).solve()
                assert out is not None and len(out) == 4, (deg, d)
            else:
                assert verdict in ("sliver", "wall"), (deg, d, verdict, ev)
    assert checked >= 8, checked
    # PREMISE-GATED.  PREMISE-GATED 2026-09-11 (CI PREMISE GATES): the CI runner arm solves these ill-conditioned fixtures CORRECTLY where every local arm solves them wrong, so a population of WRONG rows is a reading of the running arm's arithmetic and not a property of the library.  It is measured and skipped with the reading when absent, never asserted.  See docs/audits/CI_PREMISE_GATES_2026_09_11.md.  Here the premise points the other way: the
    # RETURN side needs a row whose move its OWN wall sensitivity accounts
    # for.
    if small < 1:
        pytest.skip(
            "premise absent on this arm: no screened row of this ladder "
            "agrees with its sliver-free grids (%d rows checked, every one "
            "arbitrated as sliver or wall), so the RETURN side of the "
            "contract is not exercised here.  Every row's verdict was still "
            "asserted to be reading-independent and on the right side of the "
            "arbiter's own bars." % checked)


def test_a_stack_with_no_manufactured_cell_is_never_arbitrated():
    """Two-sided on the trigger's replacement: the geometric screen is what
    decides whether the three extra solves are paid, so a stack that carries
    no manufactured cell must cost nothing at ANY reading."""
    calls = []
    real_p, real_c = ps._sliver_probe_solve, ps._sliver_collapse_solve
    ps._sliver_probe_solve = lambda *a, **k: (calls.append(1),
                                              real_p(*a, **k))[1]
    ps._sliver_collapse_solve = lambda *a, **k: (calls.append(1),
                                                 real_c(*a, **k))[1]
    try:
        for d in (0.0, 3e-3):
            st = _stack(d, 14)
            assert ps._sliver_screen(st, require_passive=False) is None, d
            cur = _raw(st)
            for reading in (1.0, 5.0):
                assert ps._sliver_arbiter(st, reading, cur[1], cur[2],
                                          None) is None, (d, reading)
        assert calls == [], calls
    finally:
        ps._sliver_probe_solve, ps._sliver_collapse_solve = real_p, real_c


def test_a_traced_stack_is_outside_the_guard_entirely():
    """ROUND 4, NEW -- a consequence of taking passivity out of the screen.

    Rounds 1-3 excluded a JAX-traced stack for free: the screen required
    ``_stack_provably_passive``, which cannot resolve a traced index and
    answered False.  Round 4 asks the geometric question WITHOUT passivity, so
    the exclusion has to be explicit -- otherwise the arbiter's three
    re-solves would be run under a trace, where they are neither cheap nor
    meaningful.

    Asserted by MONKEYPATCHING the stack's own ``_holds_traced`` rather than
    by importing jax, so the test runs in the numpy-only lane too."""
    st = _stack(1e-4, 14)
    assert ps._sliver_screen(st, require_passive=False) is not None
    st._holds_traced = lambda: True
    assert ps._sliver_screen(st, require_passive=False) is None
    assert ps._sliver_screen(st) is None
    cur = _raw(_stack(1e-4, 14))
    assert ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None) is None
    assert ps._within_layer_hazard(st, None) is None


def test_the_refusal_message_states_both_criteria_and_neither_is_the_energy():
    """The message is where a right-conclusion-wrong-reason hides.  A round-4
    refusal must state BOTH measured criteria with their bars, and must say
    plainly that the energy reading is corroboration and not the criterion --
    because that sentence is the difference between this round and the three
    before it."""
    msg = None
    for deg in (14, 12, 20):
        for d in (1e-4, 3e-5, 1e-5, 5e-6):
            try:
                _stack(d, deg).solve()
            except ValueError as exc:
                if "NEAR-COINCIDENT-WALL SLIVER" in str(exc):
                    msg = str(exc)
                    break
        if msg:
            break
    assert msg is not None, "no row of the ladder is refused on this build"
    for token in ("three extra solves", "widest manufactured cell",
                  "DISPLACED by one widest manufactured cell",
                  "%g" % ps._SLIVER_MOVE_FACTOR,
                  "%g" % ps._SLIVER_WALL_RATIO,
                  "not the device's own wall sensitivity"):
        assert token in msg, (token, msg[-1200:])
    if "CORROBORATION" in msg:
        assert "not a criterion" in msg, msg[-1200:]
        assert "property of the" in msg and "BLAS kernel" in msg, msg[-1200:]


@pytest.mark.parametrize("name", ["_SLIVER_WALL_RATIO"])
def test_the_new_constant_is_documented_where_it_is_defined(name):
    """Every bar in this family carries its derivation at its definition.  The
    round-4 constant is the campaign's own ``err > 100 delta`` rule applied to
    a MEASURED denominator, and the comment has to say so and has to name the
    two counter-fixtures that forced it."""
    import inspect
    src = inspect.getsource(ps)
    i = src.index(name + " = ")
    head = src[max(0, i - 4000):i]
    assert "833.78" in head and "906.56" in head, head[-1500:]
    assert "dR/dx" in head, head[-1500:]
    assert getattr(ps, name) == 100.0
