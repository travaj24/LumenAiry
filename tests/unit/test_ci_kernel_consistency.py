"""CROSS-KERNEL CONSISTENCY of the library's guard DECISIONS.

Fix doc: ``docs/audits/CI_KERNEL_SWEEP_2026_09_11.md``.

WHY THIS FILE EXISTS.  The 5.45.0 release matrix went red on CI while the
same commit was green on both local builds, and every failure was a
DECISION -- a guard that refuses on one machine and returns on another, or a
test whose bar was pinned on a build-dependent reading.  Nothing in the gate
could have caught that locally, because the gate only ever ran on one BLAS
micro-kernel.  This file is the missing instrument: it reads a committed
census of every guard decision on twelve (build, kernel, thread-width) arms
and asserts they AGREE, then re-takes the cheap half of that census on whatever arm is running
now and asserts it joins the consensus.

WHAT AN ARM IS, AND WHY THERE ARE TWO AXES.  An arm is a
``(build, kernel, thread-width)`` triple.

The KERNEL axis: the bundled scipy-openblas is DYNAMIC_ARCH, so
``OPENBLAS_CORETYPE`` re-dispatches the whole BLAS/LAPACK kernel set at import
time -- a different reduction order and blocking for the same arithmetic,
which is exactly what moves a rounding-level guard reading.  Two measured
caveats are recorded in
``validation/probe_ci_kernel_sweep/probe_decisions.py``: ``ZEN`` is not a
distinct kernel in these wheels (it resolves to ``Haswell``, which is also
what auto-detection gives on a Zen CPU -- and CI's runners are AMD EPYC 7763,
Zen 3, no AVX-512 -- so the default local arm IS the CI kernel), and
``SKYLAKEX`` is unreachable on a non-AVX-512 host (SIGILL).

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

WHY the CI arm's arithmetic differs is NOT solved here and is an OPEN item:
``docs/audits/CI_PREMISE_GATES_2026_09_11.md``.

BUDGET.  Reading and comparing the table is free; the current-arm re-take is
restricted to the four CHEAP sections (~3 s, measured).  The mortar and T22
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

REGENERATING THE TABLE (after a deliberate, argued decision change)::

    for k in HASWELL SANDYBRIDGE NEHALEM KATMAI; do
      OPENBLAS_CORETYPE=$k OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        MKL_NUM_THREADS=1 python \\
        validation/probe_ci_kernel_sweep/probe_decisions.py \\
        --out validation/probe_ci_kernel_sweep/arms/<build>_$k.json
    done
    python validation/probe_ci_kernel_sweep/make_ci_arm.py   # the CI arm
    python validation/probe_ci_kernel_sweep/merge_arms.py
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


def _measured(table):
    """The arms that were actually RUN, i.e. not the transcribed CI arm."""
    return sorted(a for a, m in table["arms"].items()
                  if not m.get("synthetic"))


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
    for arm, m in arms.items():
        assert arm == "%s-%s-%s" % (m["build"], m["kernel"], m["thread_arm"]), \
            (arm, m)
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

    # ---- and every MEASURED arm answered every row: a row missing from one
    # arm is a census hole, which would read as agreement above and must not.
    # The transcribed CI arm is exempt: it covers only the rows the 5.45.0
    # logs printed, and says so in its own provenance.
    run = _measured(t)
    run_keys = sorted({k for a in run for k in dec[a]})
    holes = {a: sorted(set(run_keys) - set(dec[a])) for a in run
             if set(run_keys) - set(dec[a])}
    assert not holes, holes


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

    arm, _build, _kernel, _tag, _nthreads = m._arm_id()
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

    # every cheap row this arm produced must EXIST in the census: a new
    # decision that nobody has censused is a hole, and it would pass the
    # comparison above by being absent from every arm.
    uncensused = sorted(k for k in here
                        if k.startswith(prefixes)
                        and not any(k in d for d in dec.values()))
    assert not uncensused, (
        "these decisions are produced by the probe but are in NO arm of the "
        "census: %s.  Re-run the arms and merge." % uncensused)
    # the budget this file promises, measured on the running build
    assert elapsed < 20.0, elapsed


# ======================================================================
# 5 -- the CI runner arm, which is the whole reason this round exists
# ======================================================================
def test_the_census_carries_the_ci_arm_that_answers_these_fixtures_correctly():
    """The finding of 2026-09-11, pinned so it cannot be quietly dropped.

    The census must carry the transcribed CI runner arm, it must be MARKED as
    a transcription, and it must still be class-divergent from the measured
    arms at the 1e-05 wall separation: correct there, wrong here.  If a later
    change makes the local arms answer that row correctly too, this test
    fails -- and that failure is the gate working, because the premise gates
    across six test families
    (``docs/audits/CI_PREMISE_GATES_2026_09_11.md``) rest on this divergence
    and would then all be skipping for a stale reason.
    """
    t = _load_table()
    synth = [a for a, mtd in t["arms"].items() if mtd.get("synthetic")]
    assert synth, (
        "the census no longer carries the transcribed CI runner arm.  It is "
        "the only evidence of the machine whose arithmetic solves these "
        "ill-conditioned fixtures correctly, and the premise gates in six "
        "test families cite it: regenerate it with "
        "validation/probe_ci_kernel_sweep/make_ci_arm.py.")
    for a in synth:
        assert t["arms"][a].get("provenance"), a
        assert t["arms"][a].get("provenance_detail"), a
    key = "sliver/pmm1d@1e-05"
    run = _measured(t)
    here_classes = {_class_of(t, a, key) for a in run}
    there_classes = {_class_of(t, a, key) for a in synth
                     if key in t["decisions"][a]}
    assert here_classes == {"wrong"}, (
        "the measured arms no longer read the 1e-05 sliver row as WRONG "
        "(%s).  The premise gates that cite this divergence are then stale: "
        "re-measure them." % sorted(here_classes))
    assert there_classes == {"correct"}, (
        "the transcribed CI arm no longer reads the 1e-05 sliver row as "
        "CORRECT (%s)." % sorted(there_classes))
    # ... and the DECISIONS differ accordingly, which is the guard following
    # the answer rather than a P1.
    assert {t["decisions"][a][key] for a in run} == {"refuse"}, \
        {a: t["decisions"][a][key] for a in run}
    assert {t["decisions"][a][key] for a in synth} == {"return"}, \
        {a: t["decisions"][a][key] for a in synth}


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
