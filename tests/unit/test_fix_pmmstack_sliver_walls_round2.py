"""O-11 ROUND 2 -- the ``PMMStack`` sliver guard's ARBITER (2026-09-11).

Round 1 shipped a screen-and-refuse CONJUNCTION: a manufactured cross-layer
cell at least 100x finer than the geometry, AND super-unity above 1e-2 on a
provably passive stack.  Its verification
(``docs/audits/VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md``) refuted the
margins in BOTH directions -- 110 of 648 realistic staircases were refused
although their answer was within 0.35-8.8x the physical wall shift, and 8 of
660 wrong solves returned unwarned because the WRONG population reaches DOWN
to ``R+T-1`` = +7.14e-03, BELOW the bar.

Both are one defect: super-unity DETECTS but does not ATTRIBUTE.  What
attributes, measured, is ONE extra solve on the grid the prescribed
``min_feature`` would produce -- if the super-unity vanishes AND the answer
moves far past the geometric perturbation that snap describes, the sliver
caused it; if it survives, degree / n_slices did.

Everything asserted here is MEASURED ON THE RUNNING BUILD, and where a bar
could be pinned this file asserts the DECISION (refused / returned) instead,
which is the durable statement: the verification showed round 1's margin
assertions passing on a 13-row ladder and failing by 42x on a 120-row grid of
the same family.

RESTATED 2026-09-11 (ROUND 4).  Eight tests in this file failed the 5.45.0
release CI matrix, on different pythons for different tests, and all of them
on the same shape of claim: a NAMED fixture row reads super-unity, or is a
"round-1 miss", or raises.  The energy reading is amplified rounding through a
``1/w^2``-conditioned interface -- the same row reads 1.000115 on one BLAS
kernel and 3.61242 on another -- so a test that names a row and asserts a
reading is pinning a kernel.  Round 4 removed that reading from the guard's
decision entirely, and this file no longer asserts one.

Two further restatements run through the file:

* every classification is taken on BOTH incident polarizations, which is the
  statistic ``_sliver_answer_move`` uses;
* a WRONG row must be REFUSED or RETURNED UNDER A WARNING -- never returned in
  silence.  Round 4 added a third verdict, ``'wall'``, for a move the device
  own measured sensitivity to the contested wall explains, and those rows are
  returned with a warning that names it.

Evidence: ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md``,
``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND4_2026_09_11.md`` and
``validation/probe_pmmstack_sliver_round2/`` +
``validation/probe_fix_sliver_round4/``.
"""
import os

# The sliver fixture is a near-degenerate eigenproblem: the classification the
# tests read must not move with the BLAS reduction order, so pin one thread
# before numpy is imported (the pattern of test_v5_13_0_pmm_tapered).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

# ---- the O-11 fixture, verbatim from the round-1 file --------------------
_P = 1.2e-6
_WL = 0.85e-6
_THETA = 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505
_NO_SNAP = _P * 1e-12


def _stack(delta, degree=14, *, min_feature=None, n_sup=1.0, n_sub=1.0,
           eps=_EP, theta=_THETA, nl=2, ffo=None):
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
    """Unguarded solve -- the pre-fix code path, bit for bit."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    i = np.argsort(np.asarray(o).ravel())
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return (np.asarray(o).ravel()[i], np.asarray(R)[:, i],
            np.asarray(T)[:, i], float(np.max(tot)))


def _guarded(st):
    """``(refused, message_or_None, result_or_None, warnings)``."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = st.solve()
        return (False, None, out, [str(w.message) for w in rec])
    except ValueError as exc:
        return (True, str(exc), None, [])
    finally:
        ps.PMM_SLIVER_GUARD = was


def _err(a, b, pol=None):
    """Per-order distance over the orders two solves SHARE.

    ``pol=1`` is the campaign own classification convention; ``pol=None``
    (ROUND 4, and the default now) scores BOTH incident polarizations, which
    is what :func:`~lumenairy.elements.pmm.stack._sliver_answer_move`
    compares.  A pol-1-only score calls a row correct that the guard is
    looking at through pol 0 -- the round-2 verification measured 0.0106x the
    physical shift on pol 1 and 316x on pol 0 for one row of this file own
    out-of-plane director."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(np.real(a[1][sl][:, ia])
                            - np.real(b[1][sl][:, ib])).max(),
                     np.abs(np.real(a[2][sl][:, ia])
                            - np.real(b[2][sl][:, ib])).max()))


def _kind(e, d):
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


_REFUSED_ROW = []


def _a_refused_row():
    """``(degree, delta)`` of a row the guard REFUSES on the running build.

    ROUND 4.  Three path tests below assert that the arbiter is handed the
    right wavelength by asserting that the call RAISES.  Rounds 1-3 named
    ``(14, 1e-4)`` for that, and the 5.45.0 release CI matrix returned that
    row on four shards -- three tests failed with DID NOT RAISE, on the
    PLUMBING claim, because of the physics of a different row.  The row is
    searched for here instead: the plumbing claim does not care which one it
    is."""
    if _REFUSED_ROW:
        return _REFUSED_ROW[0]
    screened = 0
    for deg in (14, 12, 20, 16):
        for d in (1e-4, 3e-5, 1e-5, 5e-6, 3e-6, 1e-6):
            st = _stack(d, deg)
            if ps._sliver_screen(st, require_passive=False) is not None:
                screened += 1
            try:
                st.solve()
            except ValueError as exc:
                if "NEAR-COINCIDENT-WALL SLIVER" in str(exc):
                    _REFUSED_ROW.append((deg, d))
                    return _REFUSED_ROW[0]
    # Nothing was refused.  Say which of the two possible reasons it is
    # before giving up: the GEOMETRY is a deterministic function of the wall
    # coordinates and must be there on every build, and if it is, then what
    # is absent is a WRONG ANSWER rather than the guard.
    assert screened == 24, (
        "the geometric screen fired on only %d of the 24 rows of this "
        "ladder; that is a guard defect, not an arithmetic one" % screened)
    pytest.skip(
        "no row of the O-11 ladder is refused on this build, and the "
        "geometric screen fires on all 24 of them: what is missing is a "
        "WRONG ANSWER, not the guard. Round 4 decides on the answer, and "
        "the CI runners solve this same fixture to max R+T = 1.0000010 "
        "where this box reads 2.17 / 3.61. The three path tests below "
        "assert PLUMBING (which wavelength the arbiter re-solves at) and "
        "can only do so through a refusal. See S4.2 and R4-G.")


def _outcome(st):
    """``'refused'`` / ``'warned'`` / ``'silent'`` -- the only three things the
    guard can do to a solve, and the vocabulary round 4 two-sided tests use.

    A WRONG answer may be refused, and it may be returned with a warning that
    says the number depends on where the contested wall is put; what it may
    never be is returned in silence."""
    refused, _msg, _out, warns = _guarded(st)
    if refused:
        return "refused"
    return "warned" if warns else "silent"


# The eight rows the verification found round 1 RETURNING although the
# continuity rule does not call them correct.  Regenerated from the same
# deterministic geomspace the census used, so the deltas are the exact
# float64s -- a five-significant-figure copy is a DIFFERENT geometry that
# reads a completely different answer (verification S4.3).
_QUIET_A = [float(x) for x in np.geomspace(3e-5, 1e-6, 60)]
_QUIET_B = [float(x) for x in np.geomspace(3e-3, 1e-6, 120)]
_ROUND1_MISSES = [
    (20, _QUIET_B[111]), (14, _QUIET_A[51]), (12, _QUIET_A[45]),
    (8, _QUIET_A[56]), (10, _QUIET_A[48]), (14, _QUIET_A[32]),
    (10, _QUIET_B[96]), (8, _QUIET_A[44]),
]


# ==========================================================================
# FAIL-BEFORE: round 1's decision, executed on round 1's code path
# ==========================================================================
def test_fail_before_round1_refuses_a_solve_that_tracks_the_physical_shift(
        monkeypatch):
    """The ROUND-1 path is still live and is still reachable: when the extra
    solves cannot be run the arbiter answers ``'unknown'`` and the guard falls
    back to exactly what round 1 did.  Forcing that branch executes the
    pre-round-2 decision, and it REFUSES a solve whose answer is within the
    physical wall shift -- the defect round 2 fixed.

    RESTATED 2026-09-11 (round 4).  This used to assert that the fixture READS
    ``max R+T`` above ``_STACK_SUPERUNITY_BAR``, which is what round 1
    triggered on and which the CI matrix showed to be a kernel fact.  What is
    a property of the library is that round 1 decision function refuses on
    that reading whatever the answer is, so the reading is HANDED to it here
    instead of being asserted of the fixture.

    The fixture is the verification own false-positive row: lossy substrate,
    1.2 rad, degree 6, a HARMLESS 1e-3 sliver."""
    def _st(d):
        return _stack(d, 6, n_sup=2.5, n_sub=1.5 + 0.05j, eps=12.0,
                      theta=1.2, ffo=31)

    d = 1e-3
    ref, cur = _raw(_st(0.0)), _raw(_st(d))
    e = _err(cur, ref)
    assert _kind(e, d) == "right", (e, e / d)          # the sliver is harmless
    assert ps._stack_provably_passive(_st(d)) is True
    assert ps._sliver_screen(_st(d)) is not None       # conjunct (a) fires

    # ROUND 1, as a decision function: given a reading above its bar it refuses
    # this stack on the geometry and the theorem alone -- it never looks at the
    # answer, which is the defect.
    assert ps._sliver_refusal(
        _st(d), 1.0 + 2.0 * ps._STACK_SUPERUNITY_BAR) is not None

    # ... and the same through the live 'unknown' branch, which is round 1
    # behaviour bit for bit, whenever the reading reaches that bar.
    monkeypatch.setattr(ps, "_sliver_probe_solve", lambda *a, **k: None)
    if cur[3] > 1.0 + ps._STACK_SUPERUNITY_BAR:
        refused, msg, _out, _w = _guarded(_st(d))
        assert refused, "round 1 path must refuse this correct solve"
        assert "NEAR-COINCIDENT-WALL SLIVER" in msg
        assert "could NOT be run on this path" in msg, msg[-600:]
    monkeypatch.undo()

    # ROUNDS 2-4: the arbiter runs, declines to attribute, and the solve
    # returns -- and round 4 returns it whatever the reading is.
    refused, _msg, out, _warns = _guarded(_st(d))
    assert not refused, "the arbiter must return it"
    assert out is not None and len(out) == 4
    v, ev = ps._sliver_arbiter(_st(d), cur[3], cur[1], cur[2], None)
    assert v == "truncation", (v, ev)


# ==========================================================================
# THE FALSE-POSITIVE CENSUS: a 60-configuration arm of the 648-config box
# ==========================================================================
def test_no_correct_solve_in_the_realistic_staircase_box_is_refused():
    """The verification's box, sub-sampled to 60 configurations: lossy
    substrates, superstrate 2.5 / 3.5, theta 1.2-1.45, degree 6-10, 2 and 4
    slices, wall steps 0.36-3.6 nm on a 1.2 um period.  Round 1 refused 110 of
    648 of these (17.0 %) whose answers track the exact ``delta -> 0`` limit.

    TWO-SIDED and decision-only: every configuration whose answer is CORRECT by
    continuity must RETURN, and the arm is only evidence if round 1 would have
    refused a good number of them -- which is asserted on the same rows."""
    import itertools
    rows = []
    for nsub, nsup, th, deg, nl, d in itertools.product(
            (1.5 + 0.05j, 3.0 + 2.0j), (2.5, 3.5), (1.2, 1.45), (6, 8),
            (2, 4), (1e-3, 3e-4)):
        def _st(delta):
            return _stack(delta, deg, n_sup=nsup, n_sub=nsub, eps=12.0,
                          theta=th, nl=nl, ffo=31)
        ref, cur = _raw(_st(0.0)), _raw(_st(d))
        e = _err(cur, ref)
        if _kind(e, d) != "right":
            continue                       # not provably harmless -> not this arm
        round1 = (ps._stack_provably_passive(_st(d))
                  and ps._sliver_screen(_st(d)) is not None
                  and cur[3] > 1.0 + ps._STACK_SUPERUNITY_BAR)
        refused, msg, _out, _w = _guarded(_st(d))
        rows.append((round1, refused, e / d, cur[3]))
        assert not refused, (
            f"REFUSED a correct solve: nsub={nsub} nsup={nsup} theta={th} "
            f"degree={deg} nl={nl} delta={d:g}, err = {e / d:.2f} x the "
            f"physical shift, R+T = {cur[3]:.6g}\n{(msg or '')[:400]}")
    assert len(rows) >= 40, len(rows)
    would_have = sum(1 for r in rows if r[0])
    assert would_have >= 8, (
        f"only {would_have} of {len(rows)} correct rows reach round 1's "
        f"refusal, so this arm is not exercising the defect")


# ==========================================================================
# THE FALSE-NEGATIVE CENSUS: the eight rows round 1 returned
# ==========================================================================
def test_no_wrong_row_of_the_round1_miss_set_is_returned_in_silence():
    """RENAMED and RESTATED 2026-09-11 (ROUND 4).  This was
    ``test_the_round1_misses_are_refused_or_are_below_the_trigger`` and it
    asserted, of eight NAMED rows, that each one (a) is not correct and (b)
    reads ``max R+T`` at or below ``_STACK_SUPERUNITY_BAR``.  Both are kernel
    facts.  The 5.45.0 release CI matrix read ``(20, 1.71299e-06)`` at
    ``R+T`` = 3.49325 on python 3.10, 2.11822 on 3.11 and 2.21342 on 3.12 and
    3.13 -- so the test failed on premise (b) on four shards, each with a
    different number.

    Round 4 has no trigger to be below.  What the eight rows are for now is
    the SAFETY property, which does not name a reading: whatever the
    continuity rule says about a row ON THIS KERNEL, a row it calls WRONG is
    refused or is returned under a warning, and a row it calls RIGHT is
    returned untouched.  The rows are still regenerated from the census own
    ``geomspace`` so they are the exact float64s (a five-significant-figure
    copy is a different geometry -- verification S4.3)."""
    refs = {}
    seen = []
    for deg, d in _ROUND1_MISSES:
        ref = refs.setdefault(deg, _raw(_stack(0.0, deg)))
        cur = _raw(_stack(d, deg))
        kind = _kind(_err(cur, ref), d)
        out = _outcome(_stack(d, deg))
        seen.append((deg, d, kind, out, cur[3]))
        if kind == "wrong":
            assert out != "silent", (
                "degree %d, delta %g: WRONG and returned in SILENCE "
                "(R+T = %.6g)" % (deg, d, cur[3]))
        elif kind == "right":
            assert out != "refused", (deg, d, cur[3])
    kinds = [r[2] for r in seen]
    assert kinds.count("wrong") + kinds.count("grey") >= 4, seen


def test_the_arbiter_runs_on_a_stack_that_reads_no_super_unity_at_all():
    """ROUND 4, NEW -- the defect the release CI matrix exposed, as a test.

    Rounds 1-3 returned from :func:`~lumenairy.elements.pmm.stack.
    _warn_stack_energy` before doing anything at all whenever the solve read
    at or below ``1 + _SLIVER_TRIGGER_BAR``.  A sliver-corrupted answer that
    happens to read 1.000115 on the running kernel was therefore returned in
    silence, and the SAME row read 2.17 on another kernel and was refused.

    Round 4 decides on the geometry, so it must arbitrate a screened stack
    whose reading is at unity.  Asserted by handing the arbiter a reading of
    exactly 1.0: the verdict must be a real one, and the two criteria must
    still be computed."""
    st = _stack(1e-4, 14)
    cur = _raw(st)
    got = ps._sliver_arbiter(st, 1.0, cur[1], cur[2], None)
    assert got is not None, "the screen did not fire on the O-11 sliver"
    verdict, ev = got
    assert verdict in ("sliver", "wall", "truncation"), verdict
    assert ev is not None and ev["d12"] >= 0.0 and ev["move"] >= 0.0, ev
    # ... and the round-3 criterion, evaluated on the same evidence, is
    # VACUOUS at this reading: with no violation to remove, its closure arm
    # admits anything the snapped grid does.
    violation = 0.0
    closure = max(ps._SLIVER_ATTRIB_CLOSURE,
                  violation * ps._SLIVER_CLOSURE_FRACTION)
    assert closure == ps._SLIVER_ATTRIB_CLOSURE
    # and round 2/3 would never have got here at all
    assert 1.0 <= 1.0 + ps._SLIVER_TRIGGER_BAR


def test_the_hazard_band_is_still_refused_and_the_outside_is_untouched():
    """The round-1 two-sided claim, re-asserted at the round-2 trigger: inside
    the band every wrong row is refused, outside it every correct row is
    returned BIT FOR BIT."""
    for deg in (12, 14):
        ref = _raw(_stack(0.0, deg))
        for d in (3e-3, 1e-3, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5):
            cur = _raw(_stack(d, deg))
            kind = _kind(_err(cur, ref), d)
            refused, msg, out, _w = _guarded(_stack(d, deg))
            if kind == "wrong":
                assert refused, (deg, d, cur[3])
                assert "NEAR-COINCIDENT-WALL SLIVER" in msg
            elif kind == "right":
                assert not refused, (deg, d, cur[3], (msg or "")[:200])
                assert np.array_equal(np.asarray(out[1])[:, np.argsort(
                    np.asarray(out[0]).ravel())], cur[1]), (deg, d)


# ==========================================================================
# THE ARBITER'S OWN CONTRACT, re-derived on the running build
# ==========================================================================
def test_the_arbiter_separates_the_two_causes_on_this_build():
    """The two populations the arbiter's two criteria must separate, measured
    here.

    RESTATED 2026-09-11 (ROUND 4).  It used to assert, of five NAMED rows,
    that the verdict is ``'sliver'`` and that the SNAPPED SUPER-UNITY sits a
    decade below ``_SLIVER_ATTRIB_CLOSURE``.  The verdict at a named row is a
    kernel fact -- the CI matrix read ``(14, 1e-4)`` as ``truncation`` with
    ``snapped_super_unity`` = 7.1e-15 and ``move/w_wide`` = 0.312 on four
    shards, because on those kernels the sliver did not move that answer at
    all -- and the closure is no longer a criterion.

    What is asserted now is the round-4 pairing, on rows CLASSIFIED here:
    a row the continuity rule calls WRONG whose move passes the geometric
    floor is attributed, a row it calls RIGHT never is, and the two
    populations are separated on the quantity that carries the decision --
    ``move / d12``, the sliver's move in units of the device own MEASURED
    answer change for a wall displacement of one sliver width."""
    wrongs, rights = [], []
    for deg in (12, 14, 20):
        ref = _raw(_stack(0.0, deg))
        for d in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
            st = _stack(d, deg)
            cur = _raw(st)
            kind = _kind(_err(cur, ref), d)
            got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
            if got is None or got[1] is None:
                continue
            v, ev = got
            row = (deg, d, v, ev["d0_over_w"], ev["d0_over_d12"])
            if kind == "wrong":
                wrongs.append(row)
            elif kind == "right":
                rights.append(row)
                assert v != "sliver", row
    assert len(wrongs) >= 4 and len(rights) >= 4, (wrongs, rights)
    attributed = [r for r in wrongs if r[2] == "sliver"]
    assert len(attributed) >= 4, (wrongs, rights)
    # each population on its own side of the two bars ...
    assert max(r[3] for r in rights) < ps._SLIVER_MOVE_FACTOR, rights
    assert min(r[3] for r in attributed) > ps._SLIVER_MOVE_FACTOR, attributed
    assert min(r[4] for r in attributed) > ps._SLIVER_WALL_RATIO, attributed
    # ... and a decade of separation BETWEEN the two, which is a property of
    # the populations rather than of the constants (round-2 defect D-3).
    assert (min(r[3] for r in attributed)
            > 10.0 * max(r[3] for r in rights)), (attributed, rights)
    assert (min(r[4] for r in attributed)
            > 10.0 * max(r[4] for r in rights)), (attributed, rights)


def test_the_move_floor_sits_above_the_correct_populations_own_envelope():
    """RENAMED and RESTATED 2026-09-11 (ROUND 4) from
    ``test_the_trigger_sits_above_the_correct_populations_envelope``.

    There is no trigger any more: the arbiter runs on the geometry.  What has
    to clear the correct population instead is the arbiter's GEOMETRIC FLOOR,
    ``_SLIVER_MOVE_FACTOR`` widest manufactured cells, and that is measured
    here across three degrees and both sides of the onset.

    The old test measured ``|R+T-1|`` and demanded it stay under
    ``_SLIVER_TRIGGER_BAR``; that quantity is the amplified rounding the CI
    matrix moved by four decades between kernels."""
    envelope = 0.0
    n = 0
    for deg in (10, 14, 16, 20):
        ref = _raw(_stack(0.0, deg))
        for d in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
            st = _stack(d, deg)
            cur = _raw(st)
            if _kind(_err(cur, ref), d) != "right":
                continue
            got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
            if got is None or got[1] is None:
                continue
            envelope = max(envelope, got[1]["d0_over_w"])
            n += 1
    # the count is a POPULATION size and is itself arm-dependent -- which rows
    # of a ladder are correct is exactly what moves with the kernel (measured
    # on this box: 10 on Haswell, 9 on Katmai).  Six is the floor at which the
    # envelope is still a population statement rather than one row's.
    assert n >= 6, n
    assert envelope < ps._SLIVER_MOVE_FACTOR, (envelope,
                                               ps._SLIVER_MOVE_FACTOR)


def test_the_arbiter_costs_three_solves_and_only_on_a_screened_stack():
    """RENAMED and RESTATED 2026-09-11 (ROUND 4) from
    ``test_the_arbiter_costs_one_solve_and_only_on_a_triggered_stack``.

    Round 4 pays MORE and pays it on MORE stacks, deliberately: three extra
    solves -- the prescribed ``min_feature`` grid, the walls CLOSED onto one
    coordinate, and that closed wall DISPLACED by one widest manufactured cell
    -- on every stack the geometric screen fires on, whatever the solve reads.
    That is the whole cost of making the decision independent of the reading,
    and it is asserted rather than described.

    Two-sided: on a stack with NO manufactured cell nothing is paid at all."""
    snaps, closes = [], []
    real_p, real_c = ps._sliver_probe_solve, ps._sliver_collapse_solve

    def _cp(*a, **k):
        snaps.append(a[1])
        return real_p(*a, **k)

    def _cc(*a, **k):
        closes.append(a[1])
        return real_c(*a, **k)

    ps._sliver_probe_solve, ps._sliver_collapse_solve = _cp, _cc
    try:
        _guarded(_stack(1e-4, 14))            # a manufactured sliver
        assert len(snaps) == 1 and len(closes) == 2, (snaps, closes)
        snaps.clear()
        closes.clear()
        # 3e-3 of a period is an own-scale ratio of 92.7, BELOW
        # _SLIVER_OWN_SCALE_RATIO, so nothing is manufactured; 0.0 is the
        # coincident-wall limit; and the DEFAULT min_feature snaps a 3e-6
        # collision away before the cascade ever sees it.
        for d in (3e-3, 0.0):
            _guarded(_stack(d, 14))
        _guarded(_stack(3e-6, 14, min_feature=_P * 1e-5))
        assert snaps == [] and closes == [], (snaps, closes)
        # ... and a CORRECT screened stack IS arbitrated now, which round 2
        # only did above its trigger.  That is the cost, stated.
        _guarded(_stack(1e-3, 14))
        assert len(snaps) == 1 and len(closes) == 2, (snaps, closes)
    finally:
        ps._sliver_probe_solve, ps._sliver_collapse_solve = real_p, real_c


def test_the_closed_grid_carries_no_sliver_and_is_the_same_device():
    """ROUND 4, NEW -- the contract of the second and third solves.

    ``_sliver_collapsed_segments`` closes every flagged wall group onto one
    coordinate, so the grid it produces must (a) carry no manufactured cell at
    all -- otherwise the arbiter would be measuring one pathology against
    another -- and (b) still be the same device: every layer widths still sum
    to a period, and the displacement is bounded by the flagged group own
    width.

    The SHIFT arm is what makes ``d12`` a sensitivity rather than round-off:
    closing LEFT and closing RIGHT differ by a pure TRANSLATION on any
    symmetric wall opening, and a translation leaves every efficiency
    unchanged.  Measured here."""
    st = _stack(1e-4, 14)
    segs = [L[1] for L in st._layers]
    mf = float(st.min_feature) / _P
    assert ps._cross_layer_sliver(segs, mf) is not None
    left = ps._sliver_collapsed_segments(segs, mf, "left")
    right = ps._sliver_collapsed_segments(segs, mf, "right")
    shifted = ps._sliver_collapsed_segments(segs, mf, "left", 1e-4)
    for got in (left, right, shifted):
        assert got is not None
        assert ps._cross_layer_sliver(got, mf) is None      # (a)
        for layer in got:
            assert abs(sum(w for w, _e in layer) - 1.0) < 1e-12   # (b)
    # The two closures carry the SAME ridge width and differ only by where
    # that ridge sits -- a TRANSLATION -- while the shift changes the width.
    # Structurally: the interior (non-boundary) widths match between the two
    # closures and do not match the shifted one.
    assert [round(w, 12) for w, _e in left[0][1:-1]] == \
           [round(w, 12) for w, _e in right[0][1:-1]], (left[0], right[0])
    assert [round(w, 12) for w, _e in left[0][1:-1]] != \
           [round(w, 12) for w, _e in shifted[0][1:-1]], (left[0], shifted[0])
    assert all(w > 0.0 for layer in shifted for w, _e in layer)

    # ... and that is why the SHIFT and not the side is what makes ``d12`` a
    # sensitivity.  Measured on this build: the two closures answer the same
    # to round-off, while displacing the closed wall by one widest cell moves
    # the answer by four decades more.
    def _solve_segs(sg):
        clone = st._min_feature_clone(float(st.min_feature))
        clone._layers = [(L[0], sg[i], L[2])
                         for i, L in enumerate(st._layers)]
        clone._src = dict(st._src)
        return _raw(clone)

    a_l, a_r, a_s = _solve_segs(left), _solve_segs(right), _solve_segs(shifted)
    d_side = _err(a_l, a_r)
    d_shift = _err(a_l, a_s)
    assert d_shift > 0.0, d_shift
    assert d_side < 1e-3 * d_shift, (d_side, d_shift)


def test_the_probe_solve_can_never_arbitrate_itself():
    """The clone the arbiter solves carries ``_sliver_probe``, so a sliver it
    still has (it cannot, but the invariant is what matters) can never recurse
    into another probe."""
    st = _stack(1e-4, 14)
    clone = st._min_feature_clone(_NO_SNAP)
    clone._src = dict(st._src)
    clone._sliver_probe = True
    assert ps._sliver_screen(clone) is None
    assert ps._sliver_arbiter(clone, 2.17, None, None, None) is None
    assert ps._within_layer_hazard(clone, None) is None
    # ... and without the mark the same clone DOES screen
    clone2 = st._min_feature_clone(_NO_SNAP)
    clone2._src = dict(st._src)
    assert ps._sliver_screen(clone2) is not None


def test_the_move_is_taken_on_the_orders_the_two_solves_share():
    """The snapped grid can resolve a different number of orders (measured: on
    359 of 637 arbitrated rows).  Both order sets are ``arange(-half, half+1)``
    so the shared set is the CENTRED overlap -- and a wider array that agrees
    on the overlap must read a move of exactly 0."""
    A = np.arange(10.0).reshape(2, 5)
    B = np.zeros((2, 9))
    B[:, 2:7] = A
    assert ps._sliver_answer_move(A, A, B, B) == 0.0
    B[:, 0] = 1e3                                  # outside the overlap
    assert ps._sliver_answer_move(A, A, B, B) == 0.0
    B[:, 4] += 0.25                                # inside it
    assert ps._sliver_answer_move(A, A, B, B) == pytest.approx(0.25)
    assert ps._sliver_answer_move(A, A, np.zeros((3, 5)), np.zeros((3, 5))) \
        is None


# ==========================================================================
# PASSIVITY: the anisotropic / liquid-crystal class (verification V-5)
# ==========================================================================
def _uniaxial(theta, axis="xy", no=1.50, ne=1.72, kappa=0.0):
    d = np.diag([(ne + 1j * kappa) ** 2, (no + 1j * kappa) ** 2,
                 (no + 1j * kappa) ** 2]).astype(complex)
    c, s = np.cos(theta), np.sin(theta)
    R = (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) if axis == "xy"
         else np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]))
    return R @ d @ R.T


def test_a_rotated_uniaxial_director_is_provably_passive():
    """Round 1's ``_tensor_is_passive`` accepted DIAGONAL tensors only, so the
    guard was structurally silent on the entire birefringent class.  A rotated
    uniaxial director is Hermitian in exact arithmetic and only NEARLY so in
    floats, which is what the round-off deadband is for -- and a GAIN payload
    must still fail, by decades."""
    for axis in ("xy", "xz"):
        for th in (0.0, np.pi / 6.0, np.pi / 4.0, 1.1):
            M = _uniaxial(th, axis)
            assert ps._segment_passive(M) is True, (axis, th)
            assert ps._segment_passive(_uniaxial(th, axis, kappa=0.2)) \
                is True, ("lossy", axis, th)
            assert ps._segment_passive(_uniaxial(th, axis, kappa=-1e-3))\
                is False, ("gain", axis, th)
    # gyrotropic: Hermitian, lossless, passive
    G = np.eye(3, dtype=complex) * 4.0
    G[0, 1], G[1, 0] = 0.35j, -0.35j
    assert ps._segment_passive(G) is True
    # a NON-Hermitian off-diagonal has no exact passivity argument
    N = np.eye(3, dtype=complex) * 4.0
    N[0, 1] = 0.2
    assert ps._segment_passive(N) is False
    # the deadband is round-off sized, not a licence: it must be far below
    # the smallest gain that could matter
    assert ps._PASSIVE_ANTIHERM_DEADBAND < 1e-13


def test_the_guard_now_reaches_a_liquid_crystal_sliver():
    """The measured consequence of round 2's passivity widening: the same O-11
    sliver on a 45-degree in-plane LC director read ``R+T`` = 2.18 under round
    1 and only WARNED.  Two-sided on three tensor classes -- in-plane
    director, OUT-OF-PLANE director and gyrotropic -- each against the
    sliver-FREE control of the same stack.

    RESTATED 2026-09-11 (ROUND 4).  This used to assert, of a NAMED
    ``delta`` = 3e-5 row on each class, that it reads above
    ``_SLIVER_TRIGGER_BAR`` and is REFUSED.  Both are kernel facts: the CI
    matrix read ``lc_in_plane`` at ``R+T`` = 1.00000013 on python 3.10 (so the
    trigger premise failed) and returned ``lc_out_of_plane`` on 3.11, 3.12 and
    3.13 (so the refusal failed).  What is asserted now is the DECISION
    against the continuity classification measured on the running kernel, plus
    the two deterministic facts this test really exists for: the classes are
    PROVABLY PASSIVE (which round 1's diagonal-only test could not see), and
    the guard reaches them at all.

    RESTATED AGAIN 2026-09-11 (CI PREMISE GATES).  Round 4's replacement still
    ended with ``assert any(r[1] == "wrong" for r in reached)`` -- at least
    one tensor class must be WRONG on the running arm -- and the 5.45.0 matrix
    failed exactly there on py3.10 shard 2, reading
    ``[('lc_in_plane', 'right', 'silent'), ('lc_out_of_plane', 'right',
    'silent'), ('gyrotropic', 'right', 'silent')]``: on the CI runner arm all
    three anisotropic slivers come out CORRECT, so the guard -- correctly --
    returns all three in silence.  That is the guard following the answer,
    which is the contract.  The reproduction of a wrong answer is therefore a
    measured PREMISE that skips with its readings, and the pol-0 / pol-1
    disagreement (a magnitude on a named row) sits behind the same gate.  Why
    the CI arm differs is an OPEN item:
    ``docs/audits/CI_PREMISE_GATES_2026_09_11.md``."""
    classes = {}
    classes["lc_in_plane"] = _uniaxial(np.pi / 4.0, "xy")
    classes["lc_out_of_plane"] = _uniaxial(np.pi / 6.0, "xz")
    G = np.eye(3, dtype=complex) * 4.0
    G[0, 1], G[1, 0] = 0.35j, -0.35j
    classes["gyrotropic"] = G
    reached = []
    for name, M in classes.items():
        st = _stack(3e-5, 14, eps=M)
        assert ps._stack_provably_passive(st) is True, name
        assert ps._sliver_screen(st) is not None, name      # it is REACHED
        ref = _raw(_stack(0.0, 14, eps=M))
        cur = _raw(_stack(3e-5, 14, eps=M))
        kind = _kind(_err(cur, ref), 3e-5)
        out = _outcome(_stack(3e-5, 14, eps=M))
        reached.append((name, kind, out))
        if kind == "wrong":
            assert out != "silent", (name, cur[3])
        elif kind == "right":
            assert out != "refused", (name, cur[3])
        # the control: the same tensor, no manufactured cell
        refused2, msg2, out2, _w2 = _guarded(_stack(3e-3, 14, eps=M))
        assert not refused2, (name, (msg2 or "")[:300])
        assert out2 is not None
    assert len(reached) == 3, reached

    # ... and the NON-Hermitian payload, which no exact argument makes
    # passive, is never REFUSED.  ROUND 4 does warn on it when the arbiter
    # attributes -- the refusal's scope is what this round declines to widen,
    # not the guard's voice (verification defect R3-C).  UNCONDITIONAL.
    N = np.eye(3, dtype=complex) * 4.0
    N[0, 1] = 0.2
    assert ps._stack_provably_passive(_stack(3e-5, 14, eps=N)) is False
    refused3, msg3, out3, _w3 = _guarded(_stack(3e-5, 14, eps=N))
    assert not refused3, (msg3 or "")[:300]
    assert out3 is not None

    # ---- PREMISE-GATED: does a sliver on ANY of the three tensor classes
    #      actually corrupt the answer on this arm?  See the RESTATED AGAIN
    #      paragraph above -- on the CI runner none of them does.
    if not any(r[1] == "wrong" for r in reached):
        pytest.skip(
            "premise absent on this arm: the 3e-05 sliver leaves every one "
            "of the three anisotropic classes CORRECT by the campaign's own "
            "continuity rule, so there is no wrong answer here for the "
            "widened passivity test to have reached: (class, continuity "
            "class, guard outcome) = %s.  The unconditional half -- all "
            "three are provably passive, the screen reaches all three, the "
            "pairing (wrong is never silent, right is never refused) holds, "
            "the sliver-free control is never refused, and the "
            "non-Hermitian payload is never refused -- passed above."
            % reached)

    # The OUT-OF-PLANE director is decided on evidence a polarization-1-only
    # statistic cannot see, which is why the move is taken on BOTH: scored
    # against the exact delta -> 0 limit, pol 1 is 0.01x the physical shift
    # (i.e. "correct") while pol 0 is 316x.  Re-derived here.
    M = classes["lc_out_of_plane"]
    ref, cur = _raw(_stack(0.0, 14, eps=M)), _raw(_stack(3e-5, 14, eps=M))
    per_pol = [_err(cur, ref, pol=p) for p in (0, 1)]
    # RESTATED 2026-09-11 (round 4).  This asserted that pol 1 is RIGHT and
    # pol 0 is WRONG by the absolute rule.  The second half is a magnitude on
    # a named row and moves with the kernel: measured on this box, pol 0 reads
    # 316x the physical shift on Haswell and 73.9x on Nehalem -- GREY there,
    # not WRONG.  What does NOT move, and is the whole reason the move is
    # taken on both polarizations, is that the two DISAGREE by decades.
    assert per_pol[1] <= 10.0 * 3e-5, per_pol      # pol 1 looks CORRECT ...
    assert per_pol[0] > 10.0 * per_pol[1], per_pol   # ... pol 0 does not


def test_a_not_provably_passive_sliver_is_warned_and_never_refused():
    """ROUND 4, NEW (verification defect R3-C).

    ``_stack_provably_passive`` answers False for anything it cannot resolve
    -- a keyed or dispersive payload, a lossy off-diagonal tensor -- and rounds
    1-3 were therefore SILENT on such a stack however plainly the sliver had
    moved its answer: the round-3 verification measured a keyed ``prepare()``
    stack returned at ``R+T`` = 2.7598 with zero probes and one generic
    warning.

    Round 4's arbitration is a comparison of ANSWERS and needs no theorem, so
    it runs there too -- and where it attributes, the same attribution is
    delivered as a WARNING.  The refusal's scope is unchanged."""
    N = np.eye(3, dtype=complex) * 4.0
    N[0, 1] = 0.2                       # non-Hermitian: no passivity argument
    seen = []
    for d in (1e-4, 3e-5, 1e-5):
        st = _stack(d, 14, eps=N)
        assert ps._stack_provably_passive(st) is False, d
        assert ps._sliver_screen(st, require_passive=False) is not None, d
        assert ps._sliver_screen(st) is None, d       # the refusal cannot fire
        refused, msg, out, warns = _guarded(_stack(d, 14, eps=N))
        assert not refused, (d, (msg or "")[:200])
        assert out is not None
        if any("WARNING (the answer is RETURNED, see below)" in w
               for w in warns):
            seen.append(d)
    assert seen, (
        "no row of the non-Hermitian ladder was attributed on this build; "
        "R3-C is then untested here")


# ==========================================================================
# THE WITHIN-LAYER ARM (verification V-6): warned, never refused
# ==========================================================================
def _liner_stack(d, degree):
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=degree,
                  min_feature=_NO_SNAP)
    segs = [(0.30, _EH), (d, _EP), (0.70 - d, _EH)]
    st.add_layer(_DZ, segments=segs)
    st.add_layer(_DZ, segments=segs)
    st.set_source(_WL, theta=_THETA)
    return st


def test_a_within_layer_sliver_is_warned_with_the_mechanism_never_refused():
    """A sliver-thin feature ONE layer owns is the geometry the caller asked
    for, so no ``min_feature`` can remove it and it is never refused.  But at
    1e-7 of a period the SAME 1/w^2 mechanism is already catastrophic (the
    verification measured err = 1.05 with ``R+T`` from 0.571 to 4.19), so the
    solve must say so and name the routes that keep the feature off the shared
    grid."""
    seen = []
    for deg in (8, 12, 16):
        st = _liner_stack(1e-7, deg)
        assert ps._cross_layer_sliver([L[1] for L in st._layers],
                                      float(st.min_feature) / _P) is None
        cur = _raw(_liner_stack(1e-7, deg))
        refused, msg, out, warns = _guarded(_liner_stack(1e-7, deg))
        assert not refused, (deg, (msg or "")[:200])
        assert out is not None
        if cur[3] > 1.0 + ps._SLIVER_TRIGGER_BAR:
            hit = [w for w in warns if "WITHIN-LAYER feature" in w]
            assert hit, (deg, cur[3], warns)
            assert "layer_grids='per-layer'" in hit[0]
            assert "mortar" in hit[0]
            seen.append(deg)
    assert seen, "no degree of the 1e-7 liner reached the trigger"


def test_the_within_layer_warning_is_silent_on_ordinary_geometry():
    """Two-sided: the arm must not speak on a liner a caller would really
    build, nor on an ordinary staircase whose super-unity is truncation."""
    for d in (1e-2, 1e-3, 1e-4, 1e-5):
        for deg in (8, 14):
            _r, _m, _o, warns = _guarded(_liner_stack(d, deg))
            assert not [w for w in warns if "WITHIN-LAYER feature" in w], \
                (d, deg, warns)
    st = _stack(1e-3, 6, n_sup=2.5, n_sub=1.5 + 0.05j, eps=12.0, theta=1.2,
                ffo=31)
    _r, _m, _o, warns = _guarded(st)
    assert not [w for w in warns if "WITHIN-LAYER feature" in w], warns


# ==========================================================================
# THE PATHS: the sweep must arbitrate at ITS OWN wavelength
# ==========================================================================
def test_the_sweep_arbitrates_at_its_own_wavelength_not_a_stale_set_source():
    """``solve_vs_wavelength`` never writes ``stack._src``, so the record left
    by an earlier ``set_source`` is a DIFFERENT physics.  The round-2 wiring
    passes the sweep's own wavelength explicitly; this asserts the arbiter's
    re-solve is handed that one."""
    seen = []
    real = ps._sliver_probe_solve

    def _spy(stack, mf, src):
        seen.append(float(src["wl"]))
        return real(stack, mf, src)

    deg, d = _a_refused_row()
    st = _stack(d, deg)
    st.set_source(0.4e-6, theta=_THETA)         # a STALE, different source
    ps._sliver_probe_solve = _spy
    try:
        with pytest.raises(ValueError, match="NEAR-COINCIDENT-WALL SLIVER"):
            st.solve_vs_wavelength([_WL], angle=_THETA, max_workers=1)
    finally:
        ps._sliver_probe_solve = real
    assert seen == [_WL], seen


def test_the_sweep_arbitrates_the_same_way_at_any_worker_count():
    """The verification could not check the guard raising from inside a
    thread-pool sweep.  ``_store`` runs on the CALLING thread in both the
    serial and the threaded branch, so the arbiter's one extra solve is a
    main-thread solve at any worker count -- and a HEALTHY sweep must still be
    byte-identical across worker counts, which is that method's own contract."""
    seen = []
    real = ps._sliver_probe_solve

    def _spy(stack, mf, src):
        seen.append(float(src["wl"]))
        return real(stack, mf, src)

    deg, d = _a_refused_row()
    ps._sliver_probe_solve = _spy
    try:
        for mw in (1, 2, 4):
            seen.clear()
            with pytest.raises(ValueError, match="NEAR-COINCIDENT-WALL SLIVER"):
                _stack(d, deg).solve_vs_wavelength(
                    [_WL, 0.9e-6], angle=_THETA, max_workers=mw)
            assert seen == [_WL], (mw, seen)
        seen.clear()
        outs = []
        for mw in (1, 2, 4):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, R, _T = _stack(3e-3, 14).solve_vs_wavelength(
                    [_WL, 0.9e-6], angle=_THETA, max_workers=mw)
            outs.append(np.asarray(R))
        # 3e-3 of a period is an own-scale ratio of 92.7, below
        # _SLIVER_OWN_SCALE_RATIO: the geometric screen never fires, so round
        # 4 pays nothing on this sweep either.
        assert seen == [], seen                 # healthy: never probed
        assert all(np.array_equal(outs[0], x) for x in outs[1:])
    finally:
        ps._sliver_probe_solve = real


def test_the_prepared_path_arbitrates_at_the_wavelength_it_was_given():
    """``prepare()`` never requires ``set_source`` at all, so on that path the
    stack's own record is unset -- the wavelength must come from the call."""
    seen = []
    real = ps._sliver_probe_solve

    def _spy(stack, mf, src):
        seen.append(float(src["wl"]))
        return real(stack, mf, src)

    deg, d = _a_refused_row()
    st = _stack(d, deg)
    st._src = None
    prepared = st.prepare()
    ps._sliver_probe_solve = _spy
    try:
        with pytest.raises(ValueError, match="NEAR-COINCIDENT-WALL SLIVER"):
            prepared.solve(wavelength=_WL, angle=_THETA)
    finally:
        ps._sliver_probe_solve = real
    assert seen == [_WL], seen


def test_the_conical_path_carries_the_arbiter_too():
    """The verification mapped the conical cascade's onset and found the same
    floor there; the arbiter must reach it, and must return the CORRECT rows
    of that path untouched."""
    def _conical(d):
        st = _stack(d, 14)
        st.set_source(_WL, theta=_THETA, phi=0.62)
        return st

    ref = _raw(_conical(0.0))
    seen = {}
    for d in (3e-3, 1e-3, 1e-4, 3e-5, 1e-5):
        cur = _raw(_conical(d))
        kind = _kind(_err(cur, ref), d)
        out = _outcome(_conical(d))
        seen[d] = (kind, out, cur[3])
        if kind == "wrong":
            assert out != "silent", (d, cur[3])
        elif kind == "right":
            assert out != "refused", (d, cur[3])
    assert sum(1 for v in seen.values() if v[1] == "refused") >= 1, seen
    assert sum(1 for v in seen.values() if v[1] != "refused") >= 2, seen
    # RESTATED 2026-09-11 (round 4).  This used to end by asserting that the
    # 3e-5 row is WRONG, is REFUSED, and reads super-unity below round 1's
    # bar -- "where round 2's LOWER trigger pays".  There is no trigger now,
    # and that row is the class round 4 separates rather than refuses: on the
    # Haswell kernel it is wrong by 114.9x the physical shift while the device
    # own measured wall sensitivity accounts for 38.6x of that, so it is
    # RETURNED under the wall-sensitivity warning; on the Katmai kernel the
    # same row is only GREY (1.000052 super-unity) and is returned in silence,
    # correctly.  So even THAT is a kernel fact and is not asserted of the
    # named row -- the pairing above already covers it, and what is asserted
    # here is only that the conical path reaches the guard at all.
    assert ps._sliver_screen(_conical(3e-5)) is not None, seen


def test_the_per_layer_window_path_is_arbitrated_on_its_OWN_grid():
    """Remedy (3) of the refusal is ``layer_grids='per-layer'`` above
    ``2 * window_halfwidth + 1`` layers, where the windows are NOT the union --
    so the geometric screen (which reads the union) and the grid the cascade
    actually ran on are different objects there.  The arbiter does not care:
    its evidence is a re-solve on the SAME path, so the attribution is measured
    on the answer the caller got.  Two-sided on a 5-layer staircase."""
    def _st(delta):
        st = PMMStack(_P, degree=14, min_feature=_NO_SNAP,
                      layer_grids="per-layer")
        for k in range(5):
            d = delta * (k % 2)
            a, b = _A0 - d, _B0 + d
            st.add_layer(_DZ, segments=[(a, _EH), (b - a, _EP), (1.0 - b, _EH)])
        st.set_source(_WL, theta=_THETA)
        return st

    ref = _raw(_st(0.0))
    seen = []
    for d in (3e-3, 1e-4, 3e-5):
        cur = _raw(_st(d))
        c = np.intersect1d(cur[0], ref[0])
        ia, ib = np.searchsorted(cur[0], c), np.searchsorted(ref[0], c)
        e = float(max(np.abs(cur[1][:, ia] - ref[1][:, ib]).max(),
                      np.abs(cur[2][:, ia] - ref[2][:, ib]).max()))
        out = _outcome(_st(d))
        seen.append((d, e / d, out, cur[3]))
        if e > 100.0 * d:
            assert out != "silent", (d, e / d, cur[3])
        elif e <= 10.0 * d:
            assert out != "refused", (d, e / d, cur[3])
    # RESTATED 2026-09-11 (round 4): the CI matrix read all three rows of this
    # 5-layer per-layer stack at err/delta = 4.15 / 4.39 / 4.47 and returned
    # every one, so ``any refused and any returned`` failed there.  What is
    # asserted is the pairing, plus that the screen reaches this path at all.
    #
    # RESTATED AGAIN 2026-09-11 (CI PREMISE GATES).  Round 4's replacement
    # ``assert any(v[2] != "silent" for v in seen)`` failed on py3.10 shard 1
    # of the same matrix: the CI arm read the three rows at
    # (0.003, 4.146, 'silent', 1.0000000002), (1e-4, 4.391, 'silent',
    # 1.0000009842), (3e-5, 4.469, 'silent', 1.0000052617) -- every row
    # CORRECT by the continuity rule (err/delta well inside 10) and every row
    # therefore, correctly, returned in silence.  Whether the guard has
    # anything to SAY on this path is a property of the running arm's
    # arithmetic; that the screen REACHES the path is not, and neither is the
    # pairing asserted in the loop above.
    #
    # INVARIANT: the geometric screen reaches the per-layer window path.
    assert ps._sliver_screen(_st(3e-5)) is not None
    # PREMISE-GATED: does the guard have anything to say here on this arm?
    if not any(v[2] != "silent" for v in seen):
        pytest.skip(
            "premise absent on this arm: every row of the 5-layer per-layer "
            "staircase is CORRECT by the continuity rule, so the guard "
            "returns all of them in silence and there is no attribution to "
            "score.  (delta, err/delta, outcome, R+T) = %s.  The pairing and "
            "the screen's reach were asserted above."
            % [(v[0], "%.4g" % v[1], v[2], "%.10g" % v[3]) for v in seen])


# ==========================================================================
# THE SWITCH still restores the pre-fix path, bit for bit
# ==========================================================================
def test_the_fail_before_switch_still_disarms_everything_round_2_added():
    """``PMM_SLIVER_GUARD = False`` is the contract: no refusal, no arbiter
    solve, no within-layer warning -- and the returned numbers are the ones
    the pre-fix library returned."""
    calls = []
    real = ps._sliver_probe_solve
    real_c = ps._sliver_collapse_solve
    ps._sliver_probe_solve = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    ps._sliver_collapse_solve = \
        lambda *a, **k: (calls.append(1), real_c(*a, **k))[1]
    try:
        was = ps.PMM_SLIVER_GUARD
        ps.PMM_SLIVER_GUARD = False
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                out = _stack(1e-4, 14).solve()
                out2 = _liner_stack(1e-7, 16).solve()
        finally:
            ps.PMM_SLIVER_GUARD = was
    finally:
        ps._sliver_probe_solve = real
        ps._sliver_collapse_solve = real_c
    assert calls == [], calls
    assert out is not None and out2 is not None
    msgs = [str(w.message) for w in rec]
    assert not [m for m in msgs if "WITHIN-LAYER feature" in m], msgs
    assert not [m for m in msgs if "is NOT what moved this answer" in m], msgs
    ref = _raw(_stack(1e-4, 14))
    i = np.argsort(np.asarray(out[0]).ravel())
    assert np.array_equal(np.asarray(out[1])[:, i], ref[1])
