"""CROSS-KERNEL CONSISTENCY of the library's guard DECISIONS.

Fix doc: ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md``.

WHY THIS FILE EXISTS.  The 5.45.0 release matrix went red on CI while the
same commit was green on both local builds, and every failure was a
DECISION -- a guard that refuses on one machine and returns on another, or a
test whose bar was pinned on a build-dependent reading.  Nothing in the gate
could have caught that locally, because the gate only ever ran on one BLAS
micro-kernel.  This file is the missing instrument: it reads a committed
census of every guard decision on twenty-four (build, kernel, thread-width)
arms and asserts they AGREE, then re-takes the cheap half of that census on
whatever arm is running now and asserts it joins the consensus.

RE-RECORDED 2026-09-12 (WP-A23), and the reason is the point of this round.
The committed census recorded ``sliver/pmm1d@1e-05`` as ``wrong`` on every
measured arm; by 2026-09-12 every arm on this tree read it ``correct``, and
:func:`test_this_arm_agrees_with_the_committed_census` was red at its RULE
check.  The cause was MEASURED, not inferred, by flipping one constant in
process: audit finding G2 (WP-A12, ``56a76f22``) raised
``pmm/stack.py::_MIN_FEATURE_DEFAULT_FRAC`` from ``period*1e-5`` to
``period*1e-3``, and the census's 1-D fixture -- a CROSS-LAYER wall pair
1e-04 / 1e-05 of a period apart -- did not pin ``min_feature``.  At the new
default both pairs SNAP to coincidence: the two layers become geometrically
identical, the interface's reciprocal condition moves 9.7297e-13 ->
5.2104e-04 (nine decades) and ``max(R+T)`` 3.6124215325 -> 1.0000000000,
so the section stopped constructing the ill-conditioned interface every row
in it is about.  The fixture now PINS ``min_feature = period*1e-5``
(``probe_decisions._1D_MF_PINNED``) -- the same remedy WP-A12 applied to its
three inherited fixtures and VERIFY-A12 applied to the mortar round-2
rationale test this section exists to mirror -- and the shipped default is
censused beside it under the ``@mf-default`` tag, so the default change is a
row rather than an erasure.  Nothing was relaxed: the pin makes the fixture
ill-conditioned again, which is strictly harder.

THE OPEN ITEM OF 2026-09-11 IS CLOSED, by the same round and again by
measurement.  ``docs/audits/CI_PREMISE_GATES_2026_09_11.md`` 2 recorded, as
OPEN, why the CI runner answers these fixtures correctly where every local
arm got them wrong, having ruled out the kernel, the thread width, the numpy
version and the OS.  It could not rule in the kernel because the 2026-09-11
host could not EXECUTE the AVX-512 (``SkylakeX``) kernels -- SIGILL on the
first BLAS call.  This workstation can: on an Intel Xeon w3-2535 the arm
``WSL-SkylakeX-t4`` reads ``R+T`` = **1.0000010471871335** at the 1e-05 wall
separation and **1.0000003658397656** at 1e-04, which are the transcribed CI
runner's two readings BIT FOR BIT, with the sliver guard returning exactly as
it did there; two independent runs agree to the last bit.  So the CI arm is a
Linux + AVX-512 arm, the divergence IS a micro-kernel effect after all, and
the census now carries a MEASURED arm that reproduces it rather than only a
transcription.  Note the pairing that makes it a kernel statement and not an
OS one: ``WIN-SkylakeX-t1`` on the same silicon, the same kernel name and the
same numpy/scipy reads 3.6124215325 -- ``wrong``.  Build and kernel together,
not either alone.

WHAT AN ARM IS, AND WHY THERE ARE TWO AXES.  An arm is a
``(build, kernel, thread-width)`` triple.

The KERNEL axis: the bundled scipy-openblas is DYNAMIC_ARCH, so
``OPENBLAS_CORETYPE`` re-dispatches the whole BLAS/LAPACK kernel set at import
time -- a different reduction order and blocking for the same arithmetic,
which is exactly what moves a rounding-level guard reading.  Two measured
caveats are recorded in
``validation/probe_ci_kernel_sweep/probe_decisions.py``: ``ZEN`` is not a
distinct kernel in these wheels (it resolves to ``Haswell``, which is also
what auto-detection gives on a Zen CPU), and ``SKYLAKEX`` is unreachable on a
non-AVX-512 host (SIGILL).

CORRECTED 2026-09-12 (WP-A23).  Both caveats hold, but the sentence that used
to follow the first one -- "CI's runners are AMD EPYC 7763, Zen 3, no
AVX-512, so the default local arm IS the CI kernel" -- does NOT.  It was a
premise, never a measurement (the 5.45.0 logs print no CPU model; see
``CI_PREMISE_GATES_2026_09_11.md`` 2.2), and it is now refuted: the CI
runner's readings reproduce BIT FOR BIT on ``WSL-SkylakeX-t4`` here and on no
other arm.  The second caveat is a property of the 2026-09-11 HOST, not of
the wheels: on this workstation (Intel Xeon w3-2535, Sapphire Rapids)
``SKYLAKEX`` is the AUTO-DETECTED default, and the reachable ladder is five
kernels rather than four.

The THREAD axis, and it is not optional: CI's FAST lane deliberately leaves
BLAS UNPINNED (``unit-tests.yml``: "this job deliberately does NOT pin BLAS at
run time") while the SLOW and JAX lanes pin to one thread.  A reduction split
across four threads is a different summation order from the same reduction on
one, in the same way a different kernel is, so a census taken only at one
thread cannot speak about the lane where most of the 5.45.0 red appeared.  The
committed table therefore carries ``t1`` arms across the whole kernel ladder
and a ``t4`` arm on the CI kernel -- ``t4`` and not ``tauto``, because
"unpinned" means "as many threads as the machine has" and this workstation
has 24 where the runner has about four (see
:func:`test_the_census_spans_both_axes_and_more_than_one_build`).

THE CONTRACT, CORRECTED 2026-09-11 (CI PREMISE GATES) -- and the correction
is the point of this round.  The first version of this file asserted that
every arm takes the same OUTCOME on every row.  The 5.45.0 matrix then failed
it, and the failure was RIGHT to happen and WRONG to be a failure: on the CI
runner arm (ubuntu, AMD EPYC 7763, unpinned BLAS, pip wheels of numpy 2.4.6 /
scipy 1.17.1) the ill-conditioned 1-D interface fixture at a 1e-05 wall
separation comes out **CORRECT** (``R+T`` = 1.0000010472) where every local
arm -- four OpenBLAS kernels x one and four thread widths x two builds --
comes out **WRONG** (``R+T`` >= 1.17).  The guards then took different
outcomes there, exactly as designed: ``closes`` against ``open``, ``silent``
against ``warn``, ``return`` against ``refuse``.  A guard whose job is to
refuse wrong answers and return correct ones MUST follow the answer, so
demanding outcome equality across arms is demanding that it stop.

The census therefore records, beside every decision, the ANSWER CLASS that
decision is about -- ``correct`` / ``grey`` / ``wrong`` by the campaign's own
closure rule, measured WITH THE GUARD DISARMED -- and the claims are:

  * ``decisions`` + ``classes`` -- RULE conformance.  Per arm, the decision
                         must be one the RULE permits for that row's class
                         (``probe_decisions._RULES``).  Across arms, rows with
                         the SAME class must take the SAME decision.  A row
                         whose decision differs while its class does NOT, or a
                         kernel-independent row (no class at all) that
                         differs, is still a P1: same code, same input, same
                         answer, different verdict.  A row whose CLASS differs
                         between arms is REPORTED, not failed -- that is the
                         guard following the answer, which is the contract.
  * ``hypothetical``  -- what a bar the library does NOT ship would decide at
                         a site the library deliberately leaves UNGUARDED.
                         These must STAY non-unanimous.  They are the
                         standing evidence that the site is undecidable, and
                         if they ever became unanimous the omission would
                         need re-arguing rather than silently keeping.

WHY the CI arm's arithmetic differs was an OPEN item in
``docs/audits/CI_PREMISE_GATES_2026_09_11.md``; it is answered above and in
:func:`test_the_census_carries_an_arm_that_answers_these_fixtures_correctly`.

THE THREE PROVENANCE KINDS (2026-09-12), because a census that can only ever
hold one machine's arms goes stale the moment that machine changes:

  * ``live``       -- measured by the probe on the tree named in the arm's
                      own ``tree`` field.  Only these are required to answer
                      every row, and the coverage requirements that stop the
                      table going archival are asserted of these.
  * ``historical`` -- measured, on an OLDER tree; keyed ``<arm>@<date>``.
                      The 2026-09-11 arms are here.  They are kept, not
                      deleted: they are the only evidence of a host that
                      could not execute AVX-512, and they are the before-side
                      of the ``min_feature`` change.  They still have to obey
                      the RULE -- they were real runs of real code -- but
                      they cover only the rows that existed when they were
                      taken.
  * ``synthetic``  -- transcribed from logs, never run (the CI arm).

A ROW WHOSE CLASS DIFFERS BETWEEN ARMS IS STILL REPORTED, NOT FAILED, and
that is now load-bearing in a second way: two LIVE arms on the same silicon
disagree in class at 1e-05 (``WSL-SkylakeX-*`` correct, every other live arm
wrong).  The rule holds on both, which is exactly the claim that matters.

BUDGET, and how it is ENFORCED.  Reading and comparing the table is free; the
current-arm re-take is restricted to the four CHEAP sections (~3 s, measured).
That restriction is asserted as a ROW COUNT -- the re-take covers exactly the
census's 24 cheap rows and none of the 30 expensive ones -- and never as a
wall clock, per TESTING_STANDARDS S1.  A second's reading on a box shared with
a dozen agents says nothing about whether the right work was done, and a
seconds bar goes green on the very regression it is there to catch (a fast
machine can run both expensive sections inside any plausible limit).  The
elapsed time is PRINTED for triage.  The mortar and T22
sections are the expensive half and are covered by the committed table plus
their own fix/verify files, so this gate stays well inside its 30 s budget.

THE SYNTHETIC CI ARM.  ``arms/ci_RUNNER_t1.json`` is not a measurement: it is
transcribed from the 5.45.0 matrix logs, marked ``"synthetic": true``, and
carries a per-key ``provenance_detail`` saying whether each value was read out
of a failure message or inferred from the absence of a mismatch.  It is in the
table because it is the only evidence of the one machine whose arithmetic
solves these fixtures correctly, and it is marked so nobody mistakes a
transcription for a run.  It is exempt from the "every arm answered every row"
check, because a transcription covers only the rows the logs printed.

REGENERATING THE TABLE (after a deliberate, argued decision change, or when
the live half has gone stale because the library moved under it)::

    # one process per kernel: OPENBLAS_CORETYPE is read at BLAS LOAD time.
    # Leave it unset for the auto-detected kernel -- on an AVX-512 host that
    # is SkylakeX, on a Zen host Haswell.
    for k in "" HASWELL SANDYBRIDGE NEHALEM KATMAI; do
      OPENBLAS_CORETYPE=$k OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python \\
        validation/probe_ci_kernel_sweep/probe_decisions.py \\
        --out validation/probe_ci_kernel_sweep/arms/<build>_live_$k_t1.json
    done
    # ... and one MULTI-THREAD arm, because CI's fast lane runs unpinned
    python validation/probe_ci_kernel_sweep/make_ci_arm.py   # the CI arm
    python validation/probe_ci_kernel_sweep/merge_arms.py

Arms already in the table that the new run does NOT replace must be marked
``"historical": true`` with ``"recorded"`` and ``"tree"`` first (they are
then keyed ``<arm>@<date>``), so that an old reading can never be mistaken
for a current one.  Do not delete them: on 2026-09-12 the 2026-09-11 arms
were the only record of a host without AVX-512, and of the library before
its ``min_feature`` default moved.
"""
import os

# The current-arm re-take is pinned to ONE thread, deliberately, and the
# THREAD axis is carried by the committed ``t4`` arms instead.  Two reasons,
# both measured.  (1) An unpinned re-take is not reproducible as a gate: this
# probe is a few hundred SMALL solves, and on a 24-thread workstation the
# per-solve thread spawn and sync dominate so hard that the re-take blows the
# budget below -- the same effect AUDIT_CI_TEST_TIME_2026_08_03 S1 measured
# when it declined to pin the fast lane.  (2) It would buy nothing anyway:
# every decision is identical at t1 and t4 on both builds, which is the
# claim the table already records.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE_DIR = _ROOT / "validation" / "probe_ci_kernel_sweep"
_TABLE = _PROBE_DIR / "decisions.json"

#: the sections the current arm RE-TAKES.  Cheap by measurement (1.4 / 0.1 /
#: 1.8 / 0.0 s); the mortar and T22 sections cost 4.7 and 5.3 s and are left
#: to the committed table so this gate keeps a wide margin under 30 s.
_CHEAP = ("interface", "sliver", "band", "branch_cut")

#: the decision PREFIXES each cheap section owns, so the re-take can be
#: compared against exactly the rows it is responsible for.
_PREFIX = {
    "interface": "pmm1d_interface/",
    "sliver": "sliver/",
    "band": "band/",
    "branch_cut": "branch_cut/",
}

#: the decision PREFIXES the two EXPENSIVE sections own.  The re-take must
#: produce NONE of these -- that is what the BUDGET note in the module
#: docstring means, stated as a set membership instead of as seconds.
_EXPENSIVE_PREFIX = ("mortar/", "t22/")

#: the re-take's OPERATION COUNT, which is what the budget is really about.
#: RE-MEASURED 2026-09-12 (WP-A23) on branch audit-fixes-2026-09 (2622449f)
#: by running the four cheap probes directly: 31 decisions (interface 12,
#: sliver 3, band 4, branch_cut 12), 6 hypotheticals, 12 answer classes.
#:
#: It was 24 / 4 before this round.  The seven extra decisions and two extra
#: hypotheticals are the THIRD 1-D case (``@mf-default``: the same wall
#: separation solved at the SHIPPED ``min_feature`` default rather than the
#: pinned one, which is how the default change is censused instead of
#: erasing the rows around it) plus the three ``pmm1d_interface/notices@``
#: rows, which carry the answer-INDEPENDENT voices -- today the union grid's
#: snap notice -- so that ``warned@`` can keep meaning "the energy tripwire
#: spoke".  See ``probe_decisions._1D_CASES`` and ``_notice_label``.
#:
#: The same 31 is the union of the cheap rows over every LIVE census arm, so
#: the number is not transcribed from one run -- it is a property of the
#: committed table, and ``test_this_arm_agrees_with_the_committed_census``
#: derives it from the table rather than reading this constant.  Recorded
#: here so a reviewer can see the size of the thing without running it.  For
#: scale: every live arm carries 61 rows, the other 30 being the expensive
#: sections (mortar 15, t22 15) this gate deliberately does not re-take; the
#: historical arms carry the 54 that existed in 2026-09-11.
_RETAKE_DECISIONS = 31
_RETAKE_HYPOTHETICALS = 6

#: the DATE the live half of the census was recorded, and the tree it was
#: recorded on.  Asserted below against what the arms themselves carry, so a
#: table whose live arms were taken on a different tree from the one this
#: file documents is a visible failure rather than a quiet drift.
_LIVE_RECORDED = "2026-09-12"
_LIVE_TREE_PREFIX = "audit-fixes-2026-09"


def _measured(table):
    """The arms that were actually RUN, i.e. not the transcribed CI arm.

    Includes the HISTORICAL arms: they were run, on an older tree.  Use
    :func:`_live` where the claim is about the tree under test.
    """
    return sorted(a for a, m in table["arms"].items()
                  if not m.get("synthetic"))


def _live(table):
    """The arms measured on the CURRENT tree.

    These are the ones a coverage requirement has to be asserted of.  A
    requirement satisfied by an archived arm is satisfied forever, which is
    precisely how the 2026-09-11 table kept passing its span check while its
    rows described a library that no longer existed.
    """
    return sorted(a for a, m in table["arms"].items()
                  if not m.get("synthetic") and not m.get("historical"))


def _archived(table):
    """Arms that cover only the rows they covered when they were taken --
    the transcriptions and the historical runs alike."""
    return sorted(a for a, m in table["arms"].items()
                  if m.get("synthetic") or m.get("historical"))


def _class_of(table, arm, key):
    """The answer class this arm recorded for a row, or ``None`` when the row
    is not an answer-following guard row (the band / branch-cut / mortar /
    T22 sections decide on geometry or on a fixed spectrum)."""
    return (table.get("classes", {}).get(arm) or {}).get(key)


def _rule_violations(rule_for, decisions, classes):
    """Every row whose decision is not one its class permits.

    ``(key, klass, decision, permitted)`` per violation.  A row with a rule
    but no recorded class is a violation too: the census cannot say the guard
    followed the answer if it did not record the answer.
    """
    out = []
    for key, got in sorted(decisions.items()):
        table = rule_for(key)
        if table is None:
            continue
        klass = classes.get(key)
        allowed = table.get(klass)
        if allowed is None or got not in allowed:
            out.append((key, klass, got, sorted(table.get(klass, ()))))
    return out


def _load_table():
    if not _TABLE.is_file():
        pytest.fail(
            "the CI kernel census is missing: %s.  It is a COMMITTED file; "
            "regenerate it with validation/probe_ci_kernel_sweep/"
            "probe_decisions.py on each arm followed by merge_arms.py (see "
            "this module's docstring)." % _TABLE)
    with _TABLE.open(encoding="cp1252") as fh:
        return json.load(fh)


def _probe_module():
    """Load the probe BY PATH.

    ``validation/`` is not a package on ``sys.path``, and making it one to
    satisfy an import would put probe scripts into the library's import
    namespace.  Loading by path also guarantees this gate and the census are
    reading the SAME fixture definitions -- duplicating them here is the one
    thing a consistency check must not do.
    """
    path = _PROBE_DIR / "probe_decisions.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("_ci_kernel_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
# 1 -- the table itself has to be worth reading
# ======================================================================
def test_the_census_spans_both_axes_and_more_than_one_build():
    """A census taken on one machine in one configuration proves nothing.

    The table must span BOTH axes -- at least two distinct BLAS kernels and
    at least two distinct thread widths, one of which must be the UNPINNED
    arm, because that is the configuration CI's fast lane actually runs --
    and at least two builds.

    RESTATED 2026-09-11 (CI PREMISE GATES).  Those spans are now required of
    the MEASURED arms only.  The census also carries a TRANSCRIBED arm -- the
    CI runner, read out of the 5.45.0 matrix logs, marked ``"synthetic":
    true`` -- and a transcription is evidence but not a configuration anybody
    ran, so it must not be able to satisfy a coverage requirement on its own.
    Every arm must also carry a ``classes`` block, because the contract the
    next test asserts is rule conformance and a table with no classes cannot
    express it.

    RESTATED AGAIN 2026-09-12 (WP-A23), and this is the anti-staleness half.
    Spanning two builds proves nothing about the CODE if both arms were taken
    a year ago: the 2026-09-11 table passed every line above while its rows
    described a library whose ``min_feature`` default had since moved two
    decades.  So the spans stay required of the measured arms -- historical
    ones included, because they were run -- and a SECOND, freshness
    requirement is added of the LIVE arms alone: at least six of them, on at
    least two kernels and two thread widths, all recorded on the tree this
    file names.  Kernels and widths rather than builds, deliberately: a
    second BUILD may be genuinely unavailable on a given host (it is a whole
    second interpreter and wheel set), while a second KERNEL is one
    environment variable away on any DYNAMIC_ARCH build, so requiring it
    costs nothing honest and catches a census taken in one configuration.
    """
    t = _load_table()
    arms = t["arms"]
    assert len(arms) >= 6, sorted(arms)
    # the spans below are asserted of the MEASURED arms only: a transcribed
    # arm is evidence, but it is not a configuration anybody ran here, and it
    # must not be able to satisfy a coverage requirement on its own.
    run = _measured(t)
    assert len(run) >= 6, run
    kernels = {arms[a]["kernel"] for a in run}
    builds = {arms[a]["build"] for a in run}
    widths = {arms[a]["thread_arm"] for a in run}
    assert len(kernels) >= 2, kernels
    assert len(builds) >= 2, builds
    assert len(widths) >= 2, widths
    assert "t1" in widths, sorted(widths)

    # ---- FRESHNESS.  The live half has to be able to speak about the code
    # as it is now, and has to say which code that was.
    now = _live(t)
    assert len(now) >= 6, (
        "the census carries %d LIVE arm(s) (%s) -- arms measured on the "
        "current tree.  A table of archived arms cannot fail when the "
        "library changes under it, which is exactly what happened on "
        "2026-09-11 -> 2026-09-12: re-take the ladder with "
        "probe_decisions.py and merge_arms.py." % (len(now), now))
    live_kernels = {arms[a]["kernel"] for a in now}
    live_widths = {arms[a]["thread_arm"] for a in now}
    assert len(live_kernels) >= 2, sorted(live_kernels)
    assert len(live_widths) >= 2, sorted(live_widths)
    for a in now:
        assert arms[a]["recorded"] == _LIVE_RECORDED, (
            a, arms[a]["recorded"], _LIVE_RECORDED)
        assert arms[a]["tree"].startswith(_LIVE_TREE_PREFIX), (
            a, arms[a]["tree"], _LIVE_TREE_PREFIX)
    # ... and every HISTORICAL arm has to say when it was taken and what it
    # measured.  An undated old reading is folklore, not evidence.
    for a in _archived(t):
        if arms[a].get("historical"):
            assert arms[a]["recorded"], a
            assert arms[a]["tree"], a
            assert a.endswith("@" + arms[a]["recorded"]), (a, arms[a])
            assert not arms[a].get("synthetic"), a
    # A MULTI-THREAD arm is required, because CI's fast lane runs BLAS
    # unpinned and a table of one-thread arms cannot speak about it.
    #
    # ``t4`` is the faithful stand-in for that lane, NOT ``tauto``, and the
    # distinction is measured rather than assumed: "unpinned" means "OpenBLAS
    # picks from the core count", so on the 2-4 core runner it means about
    # four threads, while on the 16-core/24-thread workstation this census was
    # taken on it means TWENTY-FOUR -- a different arm wearing the same label.
    # (It is also pathologically slow there: this probe is a few hundred SMALL
    # solves, and spawning and syncing 24 BLAS threads per solve dominates,
    # which is the same effect AUDIT_CI_TEST_TIME_2026_08_03 S1 measured when
    # it declined to pin the fast lane.)  A ``tauto`` arm is welcome in the
    # table and is compared like any other; it is just not what satisfies
    # this requirement.
    assert widths - {"t1"}, (
        "the census has no MULTI-THREAD arm.  CI's fast lane runs BLAS "
        "unpinned on a 2-4 core runner, so a table of one-thread arms cannot "
        "speak about it: take one at OMP/OPENBLAS/MKL_NUM_THREADS=4.  Widths "
        "present: %s" % sorted(widths))
    # the arm KEY is what was MEASURED, not what was requested: several
    # requested coretypes resolve to the same kernel (``ZEN`` and an
    # unrecognised name both land on ``Haswell``), and two arms that ran
    # identical code must not look like independent evidence.
    #
    # A HISTORICAL arm carries ``@<date>`` after the triple, because the same
    # (build, kernel, width) re-censused on a later tree is not a duplicate
    # to be rejected -- it is the before/after pair this round is about --
    # and the LIVE arm keeps the plain triple so the re-take below still
    # finds its own row.
    for arm, m in arms.items():
        stem = arm.split("@", 1)[0]
        assert stem == "%s-%s-%s" % (m["build"], m["kernel"],
                                     m["thread_arm"]), (arm, m)
        assert ("@" in arm) == bool(m.get("historical")), (arm, m)
    assert len(set(arms)) == len(arms)
    # every MULTI-THREAD arm must record the width OpenBLAS actually chose --
    # see the note above on why the label alone does not identify the arm.
    for arm, m in arms.items():
        if m["thread_arm"] != "t1" and not m.get("synthetic"):
            assert m.get("blas_threads"), (arm, m)
    # every arm must carry a ``classes`` block, even an empty one: the
    # contract below is rule conformance, and a table with no classes cannot
    # express it.
    assert set(t.get("classes", {})) == set(arms), (
        sorted(set(arms) - set(t.get("classes", {}))),
        sorted(set(t.get("classes", {})) - set(arms)))
    # ... and at least one row must actually BE classed, on every measured
    # arm -- an all-empty classes block would pass the line above.
    for arm in run:
        assert t["classes"][arm], arm


# ======================================================================
# 2 -- THE claim: every arm obeys the same RULE, and arms that computed
#      the same ANSWER take the same decision
# ======================================================================
def test_every_arm_follows_the_rule_and_agrees_where_the_answer_agrees():
    """The headline, RESTATED 2026-09-11 (CI PREMISE GATES).

    Two claims, and the second is deliberately weaker than what this file
    asserted before:

      1. RULE CONFORMANCE, per arm.  Every answer-following row's decision is
         one the RULE permits for the class that row was measured at.  This is
         the claim that actually protects users: a correct answer is never
         refused and a wrong one is never returned in silence, on any arm.
      2. AGREEMENT WHERE THE ANSWER AGREES.  Two arms that computed the same
         CLASS must take the same decision.  Two arms that computed different
         classes are ALLOWED to decide differently -- that is the guard
         following the answer -- and the divergence is REPORTED, with both
         classes, rather than failed.

    A row with NO class (band, branch-cut, mortar, T22 -- decided on geometry
    or on a fixed spectrum, with no answer to follow) must still be identical
    on every arm.  A disagreement there is the original P1.
    """
    t = _load_table()
    dec = t["decisions"]
    rule_for = _probe_module().rule_for
    arms = sorted(dec)
    keys = sorted({k for d in dec.values() for k in d})
    assert keys, "the census carries no decisions at all"

    # ---- 1. the rule, on every arm
    bad = {}
    for a in arms:
        v = _rule_violations(rule_for, dec[a],
                             t.get("classes", {}).get(a, {}))
        if v:
            bad[a] = v
    assert not bad, (
        "%d arm(s) take a guard decision the RULE does not permit for the "
        "answer class they measured -- this is a P1: a correct answer was "
        "refused, or a wrong one returned in silence.  (row, class, "
        "decision, permitted):\n%s"
        % (len(bad), json.dumps(bad, indent=1, sort_keys=True)))

    # ---- 2. agreement, per class
    split, following = {}, {}
    for k in keys:
        by_class = {}
        for a in arms:
            if k not in dec[a]:
                continue
            by_class.setdefault(_class_of(t, a, k), {})[a] = dec[a][k]
        for klass, seen in by_class.items():
            if len(set(seen.values())) > 1:
                split["%s [class=%s]" % (k, klass)] = seen
        if len(by_class) > 1:
            following[k] = {a: (_class_of(t, a, k), dec[a][k])
                            for a in arms if k in dec[a]}
    assert not split, (
        "%d guard DECISION(s) differ between arms that computed the SAME "
        "answer class -- this is a P1, not a flake: the same code, on the "
        "same input, with the same answer, refuses for one user and answers "
        "for another:\n%s"
        % (len(split), json.dumps(split, indent=1, sort_keys=True)))

    # ---- REPORTED, not failed: the guard following the answer.
    if following:
        print("\nrows whose ANSWER CLASS differs between arms (the guard "
              "following the answer -- reported, not failed):\n%s"
              % json.dumps(following, indent=1, sort_keys=True))

    # ---- and every LIVE arm answered every row: a row missing from one arm
    # is a census hole, which would read as agreement above and must not.
    #
    # The ARCHIVED arms are exempt, and the exemption is bounded rather than
    # blanket.  The transcribed CI arm covers only the rows the 5.45.0 logs
    # printed; a historical arm covers only the rows that existed when it was
    # taken (the 2026-09-11 arms predate the ``@mf-default`` case and the
    # ``notices@`` rows, and inventing readings for them would be exactly the
    # transcription-as-measurement mistake this table is marked up to
    # prevent).  What IS required of an archived arm is that every row it
    # does carry is a row the live probe still produces: an orphan row is a
    # renamed fixture, and it would sit in the table comparing against
    # nothing at all.
    now = _live(t)
    live_keys = sorted({k for a in now for k in dec[a]})
    holes = {a: sorted(set(live_keys) - set(dec[a])) for a in now
             if set(live_keys) - set(dec[a])}
    assert not holes, holes
    orphans = {a: sorted(set(dec[a]) - set(live_keys)) for a in _archived(t)
               if set(dec[a]) - set(live_keys)}
    assert not orphans, (
        "archived arm(s) carry rows the live probe no longer produces: %s.  "
        "Either a fixture was renamed (re-key the archived arms in the same "
        "change) or a section was dropped (which is the coverage loss this "
        "file exists to make visible)." % json.dumps(orphans, indent=1,
                                                    sort_keys=True))


# ======================================================================
# 3 -- the counterexample must STAY a counterexample
# ======================================================================
def test_the_unguarded_sites_stay_undecidable_across_the_kernels():
    """The other direction, and it is not symmetry for its own sake.

    ``hypothetical`` holds the verdict a bar the library does NOT ship would
    return at a site it deliberately leaves unguarded -- today, the 1e-12
    rcond bar at the plain 1-D ``_interface_smatrix`` site (see
    ``tests/unit/test_fix_pmm2d_mortar_round2.py::
    test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why``).
    The reason that site ships unguarded is that the bar's verdict FLIPS with
    the BLAS kernel.  If this ever became unanimous, the omission would rest
    on a premise that no longer holds and would have to be re-argued -- so
    the non-unanimity is asserted, not assumed.

    STRENGTHENED 2026-09-12 (WP-A23).  Non-unanimity over the WHOLE table is
    too weak now that the table spans two trees: archived arms alone could
    supply both verdicts while every current arm agreed, and the argument
    for leaving the site unguarded would then rest entirely on history.  The
    same claim is therefore asserted a second time over the LIVE arms alone.
    MEASURED on this tree: ``refuse`` on eleven live arms and ``accept`` on
    ``WIN-Katmai-t1`` -- one kernel, on one build, moving ``rcond`` from
    9.7309e-13 to 1.0538e-12 across a 1e-12 bar.  That is the whole argument,
    reproduced live: a 1.08x spread in a reading, either side of a bar.

    (This test is also why the 1-D fixture had to keep its ``min_feature``
    pin.  At the shipped default the same fixture is snapped and reads
    ``rcond`` = 5.2104e-04 on every arm -- five orders above the bar, so
    ``accept`` unanimously.  A census that only measured the default would
    have reported this site as decidable and retired a correct omission.)
    """
    t = _load_table()
    hyp = t.get("hypothetical", {})
    arms = sorted(a for a in hyp if hyp[a])
    assert arms, "the census carries no hypothetical-bar verdicts"
    keys = sorted({k for a in arms for k in hyp[a]})
    undecidable = {
        k: {a: hyp[a].get(k, "<absent>") for a in arms}
        for k in keys
        if len({hyp[a].get(k) for a in arms}) > 1
    }
    assert "pmm1d_interface/bar_1e-12_would@1e-05" in undecidable, (
        "the 1e-12 bar at the plain 1-D interface site now reads the SAME on "
        "every arm in the census.  That is the premise the site's unguarded "
        "status rests on, so it has changed: re-argue the omission (or arm "
        "the guard) rather than deleting this assertion.  Verdicts: %s"
        % json.dumps({a: hyp[a].get("pmm1d_interface/bar_1e-12_would@1e-05")
                      for a in arms}, sort_keys=True))
    assert set(undecidable["pmm1d_interface/bar_1e-12_would@1e-05"].values()) \
        == {"refuse", "accept"}, undecidable

    # ---- and the same, over the arms measured on THIS tree, so the omission
    # is not being justified entirely by history.
    key = "pmm1d_interface/bar_1e-12_would@1e-05"
    now = {a: hyp[a][key] for a in _live(t) if key in hyp.get(a, {})}
    assert len(now) >= 6, now
    assert set(now.values()) == {"refuse", "accept"}, (
        "the 1e-12 bar reads the SAME on every arm measured on the current "
        "tree (%s).  Over the whole table it is still non-unanimous, but "
        "that half is now archived evidence, and an omission cannot rest on "
        "a premise no current configuration reproduces: re-argue the "
        "unguarded site (or arm the guard) rather than deleting this "
        "assertion." % json.dumps(now, indent=1, sort_keys=True))


# ======================================================================
# 4 -- the arm running RIGHT NOW has to follow the same RULE
# ======================================================================
def test_this_arm_agrees_with_the_committed_census():
    """Re-take the cheap half of the census here and compare.

    This is what makes the gate local rather than archival: a kernel nobody
    has censused yet (a future CI runner, a colleague's laptop) either joins
    the table or fails here with its own arm named.  Only the four cheap
    sections are re-taken -- see the module docstring's BUDGET note.

    RESTATED 2026-09-11 (CI PREMISE GATES).  What "joins" means changed, and
    the 5.45.0 matrix is why: this test failed there with arm
    ``WSL-unknown-t1`` disagreeing on three rows --
    ``pmm1d_interface/answer@1e-05`` closes against open,
    ``pmm1d_interface/warned@1e-05`` silent against warn, and
    ``sliver/pmm1d@1e-05`` return against refuse -- which are exactly the
    rows where the CI arm's answer is CORRECT and every local arm's is WRONG.
    The guard FOLLOWED THE ANSWER, which is the designed behaviour, so the
    comparison was being made on the wrong thing.  It now compares the RULE:

      * this arm's decisions must be what the rule permits for the classes
        this arm measured (unconditional -- this is the user-facing claim);
      * and they must match the census rows THAT WERE TAKEN AT THE SAME
        CLASS.  A census row taken at a different class says nothing about
        this arm and is reported, not compared.

    RE-RECORDED 2026-09-12 (WP-A23).  This test was RED at step 1 on
    2026-09-12: ``pmm1d_interface/warned@1e-04`` and ``@1e-05`` read ``warn``
    at class ``correct``, which the rule forbids.  Neither the rule nor the
    guard was at fault.  Two things had changed under the census and both are
    fixed in the probe rather than here:

      * the 1-D fixture inherited a ``min_feature`` default that WP-A12's G2
        had raised two decades, so the union grid snapped its colliding walls
        away and the fixture was no longer the ill-conditioned interface
        every row is about (the module docstring carries the nine-decade
        ``rcond`` move).  ``min_feature`` is now pinned in the fixture;
      * and the snap SAYS so, which is a deliberate user-facing notice added
        by the same change.  ``warned@`` was decided as "did ANY warning come
        out", so a geometry notice on a correct answer read as a guard crying
        wolf.  ``warned@`` now measures the ENERGY TRIPWIRE -- the voice its
        own comment always named -- and the notices have their own row.

    The rule table itself is untouched, which is the point: a ``correct``
    answer still may not be warned about.
    """
    t = _load_table()
    dec = t["decisions"]

    m = _probe_module()
    here, hyp, rea, cls = {}, {}, {}, {}
    t0 = time.perf_counter()
    m._interface_site_population(here, hyp, rea, cls)
    m._sliver_decisions(here, rea, cls)
    m._band_decisions(here, rea)
    m._branch_cut_decisions(here, rea)
    elapsed = time.perf_counter() - t0

    arm, _build, _kernel, _tag, _nthreads, _ksrc, _kdet = m._arm_id()
    prefixes = tuple(_PREFIX[s] for s in _CHEAP)

    # ---- 1. THE RULE, on the arm running right now.  Unconditional.
    viol = _rule_violations(m.rule_for, here, cls)
    assert not viol, (
        "arm %r takes %d guard decision(s) the RULE does not permit for the "
        "answer class measured HERE -- a correct answer refused, or a wrong "
        "one returned in silence.  (row, class, decision, permitted): %s"
        % (arm, len(viol), json.dumps(viol, indent=1, sort_keys=True)))

    # ---- 2. agreement with the census rows taken at the SAME class
    mismatch, other_class = {}, {}
    for k, v in sorted(here.items()):
        mine = cls.get(k)
        same = {a: dec[a][k] for a in dec
                if k in dec[a] and _class_of(t, a, k) == mine}
        diff = {a: (_class_of(t, a, k), dec[a][k]) for a in dec
                if k in dec[a] and _class_of(t, a, k) != mine}
        if diff:
            other_class[k] = {"here": (mine, v), "census": diff}
        vals = set(same.values())
        if vals and (len(vals) > 1 or v not in vals):
            mismatch[k] = {"here": v, "class": mine, "census_same_class": same}
    assert not mismatch, (
        "arm %r disagrees with the committed census on %d guard decision(s) "
        "taken at the SAME answer class: %s.  Same code, same input, same "
        "answer, different verdict -- either this arm has found a "
        "kernel-dependent decision the census does not cover (add it with "
        "probe_decisions.py + merge_arms.py and FIX the guard) or a guard "
        "changed and the census is stale."
        % (arm, len(mismatch), json.dumps(mismatch, indent=1,
                                          sort_keys=True)))
    if other_class:
        print("\narm %r computed a DIFFERENT answer class from some census "
              "arms on %d row(s) -- the guard following the answer, reported "
              "and not failed:\n%s"
              % (arm, len(other_class),
                 json.dumps(other_class, indent=1, sort_keys=True)))

    # ---- 3. every cheap row this arm produced must EXIST in the census: a
    # new decision that nobody has censused is a hole, and it would pass the
    # comparison above by being absent from every arm.
    uncensused = sorted(k for k in here
                        if k.startswith(prefixes)
                        and not any(k in d for d in dec.values()))
    assert not uncensused, (
        "these decisions are produced by the probe but are in NO arm of the "
        "census: %s.  Re-run the arms and merge." % uncensused)

    # ---- 4. THE BUDGET, as an OPERATION COUNT rather than a wall clock.
    #
    # What the module docstring's BUDGET note promises is that this re-take
    # runs the four CHEAP sections and leaves the two expensive ones
    # (mortar 4.7 s, T22 5.3 s) to the committed table.  ``elapsed < 20.0``
    # was a proxy for that and the exact shape TESTING_STANDARDS S1 rules
    # out: it goes red when the shared box is loaded -- this workstation runs
    # a dozen agents -- and it goes GREEN on the regression it exists to
    # catch, because a fast enough machine can probe both expensive sections
    # inside 20 s.  The property underneath is a COUNT, and a count is
    # deterministic on every build.
    #
    # ORACLE: the committed census itself.  The cheap rows are enumerated
    # from the table (union over all arms), so the expected number is derived,
    # never transcribed, and regenerating the table moves the bar with it.
    # The re-take must produce exactly that set: fewer means a section
    # stopped being probed and this gate went partly blind; more means a
    # fixture was added without a census row (which the ``uncensused`` check
    # above catches from the other side).
    #
    # RE-MEASURED 2026-09-12 (WP-A23): 31 decisions probed (interface 12,
    # sliver 3, band 4, branch_cut 12) against 31 cheap rows in the census,
    # 6 hypotheticals, 0 expensive rows -- and 30 expensive rows per live
    # census arm that this re-take does not touch.  Both sides exact, gap 0.
    # The union is taken over ALL arms and the archived ones carry a SUBSET
    # (54 rows against 61; the orphan check in
    # test_every_arm_follows_the_rule_and_agrees_where_the_answer_agrees is
    # what makes "subset" true rather than assumed), so the union equals the
    # live row set and the count is a property of the current probe.
    census_cheap = {k for d in dec.values() for k in d
                    if k.startswith(prefixes)}
    assert len(census_cheap) == _RETAKE_DECISIONS, (
        "the census's cheap half is %d row(s), not the %d this file was "
        "written over.  That is not a failure on its own -- fixtures are "
        "allowed to be added -- but the count above is the SIZE of what this "
        "gate re-takes, so update _RETAKE_DECISIONS (and the docstring's "
        "BUDGET note) in the same change that regenerates the table."
        % (len(census_cheap), _RETAKE_DECISIONS))
    probed = set(here)
    assert probed == census_cheap, (
        "the cheap re-take no longer covers exactly the cheap half of the "
        "census: probed %d row(s), census carries %d.  missing from this "
        "run: %s; probed but not censused: %s.  Either a cheap section "
        "stopped being re-taken here -- this gate is then archival for those "
        "rows, which is what it exists not to be -- or the fixtures moved."
        % (len(probed), len(census_cheap),
           sorted(census_cheap - probed), sorted(probed - census_cheap)))
    expensive = sorted(k for k in here if k.startswith(_EXPENSIVE_PREFIX))
    assert not expensive, (
        "the re-take produced %d row(s) from the EXPENSIVE sections: %s.  "
        "Those cost 4.7 s (mortar) and 5.3 s (T22) by the measurement in "
        "this module's docstring and are covered by the committed table plus "
        "their own fix/verify files; pulling them in here is how a cheap "
        "gate becomes one nobody runs.  Either revert that, or move the "
        "section names into _CHEAP and re-measure the docstring's BUDGET "
        "note." % (len(expensive), expensive))
    assert len(hyp) == _RETAKE_HYPOTHETICALS, (
        "the re-take probed %d hypothetical bar(s), not the %d the census "
        "was built over: %s.  A hypothetical is the standing evidence that "
        "an unguarded site is undecidable "
        "(test_the_unguarded_sites_are_still_undecidable), so losing one "
        "silently retires that argument."
        % (len(hyp), _RETAKE_HYPOTHETICALS, sorted(hyp)))
    # Wall clock is PRINTED, never asserted: it is the number a reviewer
    # wants when this file starts feeling slow, and it is worthless as a bar.
    # 1.7 s for the four cheap sections on the 2026-09-12 calibration run.
    print("\narm %r re-took %d decision(s) + %d hypothetical(s) in %.2f s "
          "(wall clock reported for triage, not asserted -- "
          "TESTING_STANDARDS S1)" % (arm, len(probed), len(hyp), elapsed))


# ======================================================================
# 5 -- the CI runner arm, which is the whole reason this round exists
# ======================================================================
def test_the_census_carries_an_arm_that_answers_these_fixtures_correctly():
    """The finding of 2026-09-11, pinned so it cannot be quietly dropped --
    and, since 2026-09-12, no longer resting on a transcription.

    The census must still carry the transcribed CI runner arm, MARKED as a
    transcription.  What changed is what it is evidence OF.

    RESTATED 2026-09-12 (WP-A23).  The old form asserted that EVERY measured
    arm reads the 1e-05 sliver row ``wrong`` and only the CI arm reads it
    ``correct``.  That is now false, and not because anything went stale:
    ``WSL-SkylakeX-t1`` and ``-t4`` -- Linux on AVX-512 silicon, a
    configuration the 2026-09-11 host could not execute at all (SIGILL) --
    read it CORRECT, and ``WSL-SkylakeX-t4`` reproduces the transcribed CI
    readings BIT FOR BIT: ``R+T`` = 1.0000010471871335 at 1e-05 and
    1.0000003658397656 at 1e-04, over two independent runs.  So the claim
    this test defends is stronger than it was:

      1. the row is CLASS-DIVERGENT across the census -- some arms correct,
         some wrong.  That divergence is the premise the gates in six test
         families cite, and if it ever became unanimous those gates would be
         skipping (or asserting) for a stale reason;
      2. and the correct side is now held by a MEASURED arm, not only by a
         transcription -- which is what makes the CI runner's arithmetic
         reproducible here at all.

    The pairing is what makes it a statement about the kernel: the same
    fixture, the same silicon, the same numpy and scipy, on
    ``WIN-SkylakeX-t1`` reads 3.6124215325 -- ``wrong``.
    """
    t = _load_table()
    synth = [a for a, mtd in t["arms"].items() if mtd.get("synthetic")]
    assert synth, (
        "the census no longer carries the transcribed CI runner arm.  It is "
        "the record of the machine the 5.45.0 matrix actually ran on, and "
        "the premise gates in six test families cite it: regenerate it with "
        "validation/probe_ci_kernel_sweep/make_ci_arm.py.")
    for a in synth:
        assert t["arms"][a].get("provenance"), a
        assert t["arms"][a].get("provenance_detail"), a
    key = "sliver/pmm1d@1e-05"
    run = _measured(t)
    by_class = {}
    for a in run + synth:
        if key in t["decisions"][a]:
            by_class.setdefault(_class_of(t, a, key), []).append(a)

    # ---- 1. the divergence itself, over the whole census
    assert set(by_class) == {"correct", "wrong"}, (
        "the 1e-05 sliver row is no longer CLASS-DIVERGENT across the census "
        "(%s).  Every premise gate in the six families "
        "(docs/audits/CI_PREMISE_GATES_2026_09_11.md) rests on this "
        "divergence: re-measure them before touching this assertion.  If the "
        "divergence really has gone, the gates must be folded into their "
        "unconditional siblings, not left skipping."
        % json.dumps({k: sorted(v) for k, v in by_class.items()},
                     indent=1, sort_keys=True))

    # ---- 2. and the CORRECT side is held by an arm somebody RAN.
    correct_measured = [a for a in by_class["correct"] if a in run]
    assert correct_measured, (
        "the only arm reading the 1e-05 sliver row CORRECT is the "
        "transcription (%s).  That was the state on 2026-09-11 and it is "
        "what made the CI failure unreproducible for a day: the reproducing "
        "configuration is Linux on an AVX-512 kernel (WSL-SkylakeX-*).  If "
        "this host cannot take that arm, say so in the report rather than "
        "weakening the claim." % sorted(by_class["correct"]))
    wrong_measured = [a for a in by_class["wrong"] if a in run]
    assert wrong_measured, sorted(by_class["wrong"])

    # ---- 3. the DECISIONS follow the answer, which is the contract: every
    # ``wrong`` arm refuses, every ``correct`` arm returns.  A single row
    # here breaking that pairing is the P1 the rule check exists for, seen
    # from the other side.
    assert {t["decisions"][a][key] for a in by_class["wrong"]} == {"refuse"}, \
        {a: t["decisions"][a][key] for a in by_class["wrong"]}
    assert {t["decisions"][a][key] for a in by_class["correct"]} == \
        {"return"}, {a: t["decisions"][a][key] for a in by_class["correct"]}

    # ---- 4. the bit-identity that CLOSED the 2026-09-11 open item.  Not a
    # tolerance: the two readings are the same float64, so ``==`` is the
    # honest comparison and any drift at all is a finding.  MEASURED twice,
    # independently, on WSL-SkylakeX-t4.
    ci = synth[0]
    twins = [a for a in correct_measured
             if t["readings"][a].get("pmm1d_interface/R+T@1e-05")
             == t["readings"][ci].get("pmm1d_interface/R+T@1e-05")
             and t["readings"][a].get("pmm1d_interface/R+T@1e-04")
             == t["readings"][ci].get("pmm1d_interface/R+T@1e-04")]
    assert twins, (
        "no measured arm reproduces the transcribed CI runner's readings bit "
        "for bit any more.  That reproduction (WSL-SkylakeX-t4: R+T = "
        "%r at 1e-05, %r at 1e-04) is what identifies the CI arm as Linux + "
        "AVX-512 and closes the 2026-09-11 OPEN item; losing it silently "
        "re-opens it.  Correct-class measured arms and their readings: %s"
        % (t["readings"][ci].get("pmm1d_interface/R+T@1e-05"),
           t["readings"][ci].get("pmm1d_interface/R+T@1e-04"),
           json.dumps({a: t["readings"][a].get("pmm1d_interface/R+T@1e-05")
                       for a in correct_measured}, indent=1, sort_keys=True)))


def test_the_cheap_sections_cover_every_area_the_sweep_found_a_flip_in():
    """A guard against the gate quietly narrowing.

    The re-take above is only as good as the sections it runs.  This pins
    which prefixes are re-taken locally, so dropping one (to save a second,
    say) is a visible edit rather than a silent loss of coverage.
    """
    t = _load_table()
    keys = {k for d in t["decisions"].values() for k in d}
    for section, prefix in _PREFIX.items():
        assert section in _CHEAP, section
        assert any(k.startswith(prefix) for k in keys), (section, prefix)
    m = _probe_module()
    for section in _CHEAP:
        assert hasattr(m, {"interface": "_interface_site_population",
                           "sliver": "_sliver_decisions",
                           "band": "_band_decisions",
                           "branch_cut": "_branch_cut_decisions"}[section])


def test_the_1d_section_still_pins_the_min_feature_that_makes_it_a_fixture():
    """The structural half of the 2026-09-12 finding, pinned in the probe.

    Every row of sections A and B is about an ill-conditioned interface built
    from a CROSS-LAYER wall pair 1e-04 / 1e-05 of a period apart.  ``min_
    feature`` is the knob that SNAPS such a pair away, so a fixture that
    inherits the library default stops being that fixture the moment the
    default crosses the separation -- which is what WP-A12's G2 did
    (``period*1e-5`` -> ``period*1e-3``), silently, turning every row in the
    section into a trivially-correct reading and leaving this file red at its
    rule check for a reason that had nothing to do with any guard.

    So the pin is a STRUCTURAL invariant of the census, and it is asserted
    here rather than left as a comment: the fixture must pin, the pinned
    value must be below both separations, and the SHIPPED default must still
    be censused beside it so the next default change is a row rather than an
    erasure.  See the module docstring's RE-RECORDED note for the measured
    nine-decade ``rcond`` move.
    """
    m = _probe_module()
    assert m._1D_MF_PINNED == 1.0e-5, m._1D_MF_PINNED
    pinned = [c for c in m._1D_CASES if c[1] is not None]
    default = [c for c in m._1D_CASES if c[1] is None]
    assert pinned and default, m._1D_CASES
    # the pin has to be BELOW every separation it is asked to preserve, or it
    # snaps the very thing it is there to keep.  (1e-05 of a period is the
    # tighter of the two separations, and the pin equals it: the union grid
    # merges STRICTLY closer than min_feature, which is why equality is the
    # boundary that preserves the pair rather than the one that removes it.)
    for delta, frac, _tag in pinned:
        assert frac <= delta, (delta, frac)
    # and the default case must name itself, so a reader of the table can see
    # which rows are "as shipped" without consulting this file.
    assert all("mf-default" in c[2] for c in default), m._1D_CASES
    # the probe must also still SPLIT the voices: ``warned@`` is the energy
    # tripwire, and the geometry notice has its own row.  A regression to
    # "any warning" reads as a guard crying wolf on a correct answer.
    assert m._VOICE_ENERGY and m._VOICE_SNAP
    assert m._notice_label(0, 0) == "none"
    assert m._notice_label(1, 0) == "snap"
    assert m._notice_label(1, 2) == "snap+other"
