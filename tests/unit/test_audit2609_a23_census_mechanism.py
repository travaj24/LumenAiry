"""WHY the CI kernel census went stale between 2026-09-11 and 2026-09-12.

Fix doc: ``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
WP-A23_REPORT.md``.  Census: ``validation/probe_ci_kernel_sweep/``.

THE FINDING.  ``tests/unit/test_ci_kernel_consistency.py::
test_this_arm_agrees_with_the_committed_census`` was RED at its RULE check:
the committed census recorded ``sliver/pmm1d@1e-05`` as ``wrong`` on all ten
measured arms, while this tree read all eight 1-D rows ``correct``, and two
``pmm1d_interface/warned@`` rows took ``warn`` at class ``correct``, which
the census's rule table forbids.

THE MECHANISM, and this file exists so it is a measurement rather than an
attribution.  WP-A12's finding G2 (commit ``56a76f22``) raised
``lumenairy/elements/pmm/stack.py::_MIN_FEATURE_DEFAULT_FRAC`` from
``period*1e-5`` to ``period*1e-3``.  The census's 1-D fixture is a
CROSS-LAYER wall pair 1e-04 / 1e-05 of a period apart and did not pin
``min_feature``, so the new default -- two decades ABOVE both separations --
snapped both pairs to coincidence.  The two layers become geometrically
IDENTICAL and the near-singular interface every row in that section is about
ceases to exist.

Two claims, and the split is the point:

  * :func:`test_the_raised_default_snaps_the_censused_fixture_out_of_existence`
    is UNCONDITIONAL.  Whether a wall pair ``delta`` apart survives a
    ``min_feature`` threshold is decided by comparing two floats the caller
    supplies; no BLAS kernel participates, so the nine-decade ``rcond``
    move, the snap notice and the collapse of the two layers onto one
    geometry are properties of every arm.
  * :func:`test_and_that_is_what_moved_the_answer_class` is PREMISE-GATED,
    because the answer the ill-conditioned interface produces is exactly the
    quantity the whole census exists to say is build-dependent.  MEASURED
    2026-09-12: the pinned fixture reads ``max(R+T)`` = 3.6124215325 on
    ``WIN-SkylakeX-t1`` (wrong) and 1.0000010471871335 on ``WSL-SkylakeX-t4``
    (correct, and bit-identical to the transcribed CI runner).  On an arm of
    the second kind there is no wrong answer here to move, so the premise is
    measured and skipped with its reading -- never asserted.  Its
    unconditional sibling above is named in the skip message.

This is a build property, not a resource precondition, so
``docs/TESTING_STANDARDS.md`` S2 ("never ``pytest.skip`` on a resource
check") does not apply: the library-facing half is asserted on every arm and
only the pathology's reproduction is gated, on a measurement printed into the
log.  The shape and the reasoning are
``docs/audits/CI_PREMISE_GATES_2026_09_11.md`` section 3.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from lumenairy.elements.pmm import stack as _st1d  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "validation" / "probe_ci_kernel_sweep" / "probe_decisions.py"

#: the default the census was taken at, and the one the library ships now.
#: The first is a historical constant and is written here rather than read
#: from the library, because the whole finding is that the library's value
#: MOVED -- reading it would make this file agree with whatever it becomes.
_MF_OLD_FRAC = 1.0e-5


def _probe():
    """The census probe, loaded BY PATH -- the same way the consistency gate
    loads it, so this file and the census cannot drift into two definitions
    of the same fixture."""
    spec = importlib.util.spec_from_file_location("_a23_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve_at(mf_frac, delta):
    """``(rcond, max(R+T), voices)`` for the census fixture at one
    ``min_feature`` setting, with the sliver guard DISARMED.

    Disarmed for the same reason the census disarms it: armed, the guard can
    refuse and there is then no reading at all, and this file is about the
    ARITHMETIC the guard is deciding on, not about the decision.
    """
    m = _probe()
    real = _st1d._interface_smatrix
    prev = _st1d.PMM_SLIVER_GUARD
    seen = []

    def _patched(Wa, Va, Wb, Vb):
        for A in (np.asarray(Wb), np.asarray(Vb)):
            lu, _piv = sla.lu_factor(A)
            gecon = sla.get_lapack_funcs("gecon", (A,))
            rc, _i = gecon(lu, float(np.max(np.sum(np.abs(A), axis=0))))
            seen.append(float(rc))
        return real(Wa, Va, Wb, Vb)

    _st1d._interface_smatrix = _patched
    _st1d.PMM_SLIVER_GUARD = False
    try:
        st = m._pmm1d_two_layer(delta, mf_frac)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _o, R, T = st.solve()[:3]
        tot = float(np.max(np.atleast_2d(R).sum(1) + np.atleast_2d(T).sum(1)))
        return min(seen), tot, m._voices(w)
    finally:
        _st1d._interface_smatrix = real
        _st1d.PMM_SLIVER_GUARD = prev


# ======================================================================
# 1 -- UNCONDITIONAL: the raised default removes the fixture
# ======================================================================
def test_the_raised_default_snaps_the_censused_fixture_out_of_existence():
    """Pure geometry, therefore true on every arm.

    The census fixture's object of measurement is a near-singular interface
    produced by a cross-layer wall pair.  ``min_feature`` decides whether
    that pair survives, by comparing the separation against a threshold --
    arithmetic no BLAS kernel takes part in.  So this is a two-sided
    statement that holds everywhere:

      * at ``min_feature = period*1e-5`` (the value the census was taken at,
        and the value the probe now PINS) both separations survive and the
        interface is ill-conditioned: MEASURED ``rcond`` = 9.6940e-11 at
        ``delta`` = 1e-04 and 9.7297e-13 at 1e-05, and the solve is silent
        about the geometry;
      * at the SHIPPED default (``period*1e-3`` since WP-A12 G2) both pairs
        snap, ``rcond`` = 4.2942e-04 and 5.2104e-04, and the union grid says
        so with ``_pmm_union_grid: snapped 2 pair(s) ...``.

    BARS, and their derivation.  The two populations are separated by NINE
    decades (9.73e-13 against 5.21e-04), so the bars are placed a long way
    inside that gap and on opposite sides of it: the pinned readings must
    stay under 1e-09 (3.7 decades of slack above the worst pinned reading)
    and the default ones over 1e-06 (2.7 decades below the best snapped one).
    Nothing in this fixture has ever landed between them, and a bar placed in
    an empty gap of nine decades cannot be moved by a kernel -- which is the
    property that makes this half unconditional.
    """
    pinned, snapped = {}, {}
    for delta in (1e-4, 1e-5):
        pinned[delta] = _solve_at(_MF_OLD_FRAC, delta)
        snapped[delta] = _solve_at(None, delta)

    for delta in (1e-4, 1e-5):
        rc_p, _tot_p, (e_p, s_p, _env_p, o_p) = pinned[delta]
        rc_s, _tot_s, (_e_s, s_s, _env_s, o_s) = snapped[delta]
        # the pinned fixture IS an ill-conditioned interface ...
        assert rc_p < 1e-9, (delta, rc_p)
        # ... and the shipped default's is not, by nine decades
        assert rc_s > 1e-6, (delta, rc_s)
        assert math.log10(rc_s / rc_p) > 6.0, (delta, rc_p, rc_s)
        # the snap is SILENT when nothing is snapped and LOUD when something
        # is.  Both halves matter: the notice is the only thing that tells a
        # caller their requested geometry is not the one that was solved.
        assert s_p == 0, (delta, s_p, e_p, o_p)
        assert s_s >= 1, (delta, s_s)

    # ... and the mechanism, stated as the thing that actually happened: at
    # the shipped default the two layers are solved as ONE geometry, so the
    # two wall separations become indistinguishable.  Measured as the
    # readings themselves: 1e-04 and 1e-05 differ under the pin and agree
    # under the default.
    assert pinned[1e-4][0] != pinned[1e-5][0], pinned
    assert abs(math.log10(pinned[1e-4][0] / pinned[1e-5][0])) > 1.5, pinned
    assert abs(math.log10(snapped[1e-4][0] / snapped[1e-5][0])) < 0.2, snapped


def test_the_census_probe_pins_min_feature_and_still_censuses_the_default():
    """The repair, pinned structurally so it cannot be undone by accident.

    The probe must build its 1-D fixture at the OLD default and must ALSO
    carry a case at the shipped default -- the first so the section measures
    an ill-conditioned interface at all, the second so the next default
    change shows up as a row rather than as rows quietly going trivial.
    """
    m = _probe()
    assert m._1D_MF_PINNED == _MF_OLD_FRAC, m._1D_MF_PINNED
    tags = {c[2]: c for c in m._1D_CASES}
    assert "1e-04" in tags and "1e-05" in tags, sorted(tags)
    assert any(c[1] is None for c in m._1D_CASES), m._1D_CASES
    assert any("mf-default" in tg for tg in tags), sorted(tags)
    # the library default really is the one that motivated the pin, so a
    # future change back to 1e-5 is visible here rather than silent.
    assert _st1d._MIN_FEATURE_DEFAULT_FRAC == 1.0e-3, (
        "the library's min_feature default is now %g, not the 1e-3 this "
        "file's measurements were taken against.  Re-measure the nine-decade "
        "rcond gap above and the census's @mf-default rows before changing "
        "this line." % _st1d._MIN_FEATURE_DEFAULT_FRAC)


def test_the_interface_voices_are_split_so_a_notice_is_not_a_false_alarm():
    """The second half of the repair, and why it is not a relaxed bar.

    The census's ``pmm1d_interface/warned@`` row carries an answer-following
    RULE: a ``correct`` answer must be SILENT.  It was decided as "did ANY
    warning come out", which was the same thing while the energy tripwire was
    the only voice at this site.  G2 added a second, deliberate one -- the
    snap notice -- that speaks about the GEOMETRY on a correct answer, and
    the row then read ``warn`` at class ``correct``: a rule violation
    reporting a guard crying wolf, when no guard had spoken.

    The classifier now separates them, and this pins BOTH directions on
    readings taken here: the tripwire fires on the wrong answer and not on
    the correct one, and the notice fires exactly where the snap happens.
    The rule itself is unchanged -- a correct answer still may not be warned
    about.
    """
    m = _probe()
    # the snapped (correct) solve: notice, no tripwire
    _rc, tot_s, (e_s, s_s, _env, _o) = _solve_at(None, 1e-5)
    assert abs(tot_s - 1.0) < 1e-5, tot_s
    assert e_s == 0, ("the energy tripwire fired on an answer that closes to "
                      "%.3e" % abs(tot_s - 1.0))
    assert s_s >= 1
    assert m._notice_label(s_s, 0) == "snap"

    # ... and an environment advisory is not a voice about this fixture.
    # MEASURED 2026-09-12: the WSL build reachable from this workstation is a
    # venv without psutil and raises "psutil not installed - assuming 4 GB
    # available memory" three times per solve, which would otherwise make two
    # builds disagree on a row that has no answer class -- i.e. read as a P1.
    class _Rec:
        category = RuntimeWarning

        def __init__(self, msg):
            self.message = msg

    assert m._is_env_advisory(_Rec(
        "psutil not installed - assuming 4 GB available memory.  Install "
        "psutil for accurate memory-aware batching."))
    assert not m._is_env_advisory(_Rec("energy not conserved (max R+T = 3.61)"))
    assert m._voices([_Rec("energy not conserved"),
                      _Rec("_pmm_union_grid: snapped 2 pair(s)"),
                      _Rec("psutil not installed - Install psutil for x"),
                      _Rec("something nobody has seen before")]) == \
        (1, 1, 1, 1)


# ======================================================================
# 2 -- PREMISE-GATED: and that is what moved the answer CLASS
# ======================================================================
def test_and_that_is_what_moved_the_answer_class():
    """The census's own reading, reproduced by flipping one setting.

    UNCONDITIONAL half (asserted before the gate): the shipped default reads
    this fixture CORRECT at both separations.  That is the direct consequence
    of the geometry test above -- a stack whose two layers are identical has
    no near-singular interface to get wrong -- and it holds on every arm.

    PREMISE-GATED half: that the PINNED fixture reads WRONG here, which is
    what makes the class actually MOVE.  That premise is a reading of this
    arm's arithmetic: MEASURED 2026-09-12, ``max(R+T)`` = 3.6124215325 on
    ``WIN-SkylakeX-t1`` against 1.0000010471871335 on ``WSL-SkylakeX-t4`` --
    the same library, the same fixture, the same silicon, one build and one
    micro-kernel apart, and the second is bit-for-bit the transcribed CI
    runner.  Ten of the twelve live census arms read it wrong and two read it
    right, so this gate fires on about one arm in six and must never be an
    assertion.

    The closure bars are the campaign's own (``probe_decisions._CLASS_CORRECT``
    = 1e-5, ``_CLASS_WRONG`` = 1e-2), placed by the GAP between the two
    populations -- correct rows reach 3.66e-07 at worst and wrong rows 1.17
    at best, five empty decades apart -- not by any one build's residual.
    """
    m = _probe()
    _rc_s4, tot_s4, _v = _solve_at(None, 1e-5)
    _rc_s3, tot_s3, _v3 = _solve_at(None, 1e-4)
    # ---- UNCONDITIONAL: at the shipped default both rows are CORRECT.
    assert m._answer_class(tot_s4) == "correct", tot_s4
    assert m._answer_class(tot_s3) == "correct", tot_s3

    _rc_p, tot_p, _vp = _solve_at(_MF_OLD_FRAC, 1e-5)
    klass = m._answer_class(tot_p)
    if klass != "wrong":
        pytest.skip(
            "premise absent on this arm: the PINNED census fixture "
            "(min_feature = period*1e-5) already answers the 1e-05 wall "
            "separation %s here -- max(R+T) = %.16g, |R+T - 1| = %.3e "
            "against the 1e-02 the wrong population reaches -- so there is "
            "no wrong answer for the raised default to have moved.  This is "
            "the CI-runner class of arm (WSL-SkylakeX-* reproduce the "
            "5.45.0 runner's 1.0000010471871335 bit for bit).  The "
            "UNCONDITIONAL claims passed: the shipped default reads %.16g "
            "(correct) at 1e-05 and %.16g at 1e-04, and "
            "test_the_raised_default_snaps_the_censused_fixture_out_of_"
            "existence pinned the nine-decade rcond move that causes it."
            % (klass, tot_p, abs(tot_p - 1.0), tot_s4, tot_s3))

    # ---- the class MOVED, and only the one setting moved it.
    assert m._answer_class(tot_s4) == "correct" != klass, (tot_p, tot_s4)
    # and the pinned reading is the census's own wrong population, not some
    # third thing: |R+T - 1| >= 1, i.e. two decades past the wrong bar.
    assert abs(tot_p - 1.0) > 1.0, tot_p


def test_the_committed_census_agrees_with_what_this_file_measures():
    """The table and this file must not become two stories.

    The census's own ``@mf-default`` rows record exactly the comparison made
    above.  Reading them here closes the loop: if someone re-records the
    table without the pin, or drops the default case, this fails rather than
    leaving this file arguing about a fixture the census no longer takes.
    """
    import json
    path = _ROOT / "validation" / "probe_ci_kernel_sweep" / "decisions.json"
    with path.open(encoding="cp1252") as fh:
        t = json.load(fh)
    live = [a for a, mtd in t["arms"].items()
            if not mtd.get("synthetic") and not mtd.get("historical")]
    assert live, sorted(t["arms"])
    for a in live:
        d, c = t["decisions"][a], t["classes"][a]
        # the default-tagged rows exist, are CORRECT on every arm, and carry
        # the snap notice; that is the "after" side of the mechanism.
        assert c["sliver/pmm1d@1e-05@mf-default"] == "correct", (a, c)
        assert d["pmm1d_interface/notices@1e-05@mf-default"] == "snap", (a, d)
        # ... and the pinned row is silent about geometry, which is what
        # makes it the same fixture the 2026-09-11 arms measured.
        assert d["pmm1d_interface/notices@1e-05"] == "none", (a, d)
    # every HISTORICAL arm read the pinned row WRONG -- that is the census
    # this round re-recorded, and it is kept rather than deleted.
    hist = [a for a, mtd in t["arms"].items() if mtd.get("historical")]
    assert hist, sorted(t["arms"])
    assert {t["classes"][a]["sliver/pmm1d@1e-05"] for a in hist} == {"wrong"}, \
        {a: t["classes"][a]["sliver/pmm1d@1e-05"] for a in hist}


def test_the_threadpoolctl_branch_records_the_same_per_library_table_as_the_fallback(monkeypatch):
    """``_arm_id`` has two instruments: ``threadpoolctl`` when it is installed,
    the ``ctypes`` read-back otherwise.  The fallback fills ``blas_libraries``
    (one row per loaded BLAS build: corename and thread width); the
    threadpoolctl branch used to leave it ``{}``, and nothing noticed because
    no host in the census had threadpoolctl until the dependency was
    installed at campaign close -- the first six arms recorded through it
    came back with an empty table.  The branch is exercised here with a
    synthetic ``threadpool_info`` so the two instruments cannot drift apart
    on what an arm records.
    """
    import threadpoolctl

    m = _probe()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [
        {"internal_api": "openblas", "filepath": "C:/x/libscipy_openblas-abc.dll",
         "architecture": "SkylakeX", "num_threads": 1},
        {"internal_api": "openmp", "filepath": "C:/x/libgomp.dll", "num_threads": 20},
        {"internal_api": "mkl", "filepath": "C:/x/mkl_rt.2.dll", "num_threads": 4},
    ])
    arm, build, arch, tag, nthreads, source, detail = m._arm_id()
    assert source == "threadpoolctl", source
    assert arch == "SkylakeX" and nthreads == 1, (arch, nthreads)
    assert arm.endswith("-SkylakeX-" + tag), arm
    # one row per BLAS build, keyed by the library basename, OpenMP runtimes ignored
    assert detail == {
        "libscipy_openblas-abc.dll": {"corename": "SkylakeX", "num_threads": 1},
        "mkl_rt.2.dll": {"corename": "unknown", "num_threads": 4},
    }, detail

