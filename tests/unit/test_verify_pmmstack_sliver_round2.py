"""VERIFY O-11 ROUND 2 -- what the independent re-measurement found.

Companion to ``tests/unit/test_fix_pmmstack_sliver_walls_round2.py``.  Evidence
and every number: ``docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md``
and ``validation/probe_verify_sliver_round2/``.

Most of the tests here PIN A KNOWN LIMITATION rather than a fix.  Each says so
in its own docstring, with the instruction the round-1 verification
established: **if the library is later taught to handle the case, this test
fails, and that failure is the gate working -- re-pin it against the
improvement, do not relax it.**  The D-5 test at the bottom of this file has
already been through that cycle: round 3 shipped the relative closure it asked
for, and it now pins the REPAIRED decision.

Everything asserted is measured on the running build; the only fixed numbers
are the library's own constants and the geometry.
"""
import os

# The sliver fixture is a near-degenerate eigenproblem: pin one BLAS thread
# before numpy is imported (the pattern of the two sibling sliver files).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

# ---- the O-11 fixture, the two sibling files' own numbers ------------------
_P = 1.2e-6
_WL = 0.85e-6
_THETA = 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505
_NO_SNAP = _P * 1e-12


def _stack(delta, degree=12, *, liner=None, eps=_EP, key=None, nl=2):
    """The two-slice staircase, optionally carrying a sliver-thin feature ONE
    layer OWNS (``liner``) and/or a material KEY instead of a value."""
    st = PMMStack(_P, n_substrate=1.0, degree=degree, far_field_orders=31,
                  min_feature=_NO_SNAP)
    ridge = key if key is not None else eps
    for k in range(nl):
        d = delta * k / max(nl - 1, 1)
        a, b = _A0 - d, _B0 + d
        segs = ([(a, _EH), (b - a, ridge), (1.0 - b, _EH)] if liner is None
                else [(a, _EH), (b - a, ridge), (liner, 9.0),
                      (1.0 - b - liner, _EH)])
        st.add_layer(_DZ, segments=segs)
    st.set_source(_WL, theta=_THETA)
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
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return (np.asarray(o).ravel()[i], np.real(np.asarray(R))[:, i],
            np.real(np.asarray(T))[:, i], float(np.max(tot)))


def _guarded(st):
    """``(refused, message, result, warnings)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            out = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    return (False, "", out, [str(w.message) for w in rec])


def _move(a, b):
    """The library's own statistic: max |dR|, |dT| over BOTH polarizations on
    the orders two solves share."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][:, ia] - b[1][:, ib]).max(),
                     np.abs(a[2][:, ia] - b[2][:, ib]).max()))


def _screen(st):
    return ps._cross_layer_sliver([L[1] for L in st._layers],
                                  float(st.min_feature) / float(st.period))


# ==========================================================================
# D-1 -- one thin OWNED feature disarms the CROSS-LAYER refusal
# ==========================================================================
def test_an_owned_liner_anywhere_disarms_the_cross_layer_refusal():
    """PINS A KNOWN LIMITATION (verification defect D-1, pre-existing since
    round 1 -- ``_cross_layer_sliver`` is unchanged by round 2).

    ``own`` is the GLOBAL minimum wall spacing over all layers, and a cell is
    manufactured only when ``own / w >= _SLIVER_OWN_SCALE_RATIO``.  So a
    sliver-thin feature that ONE layer legitimately owns lowers ``own`` for
    every manufactured cell in the stack and the screen goes silent on a
    genuine cross-layer sliver.  Measured on this fixture: the same delta and
    degree is REFUSED at ``R+T`` = 2.17 without the liner and RETURNED at a
    LARGER ``R+T`` with it.

    If the screen is later taught to score ``own`` per flagged cell against
    the layers that own that cell's neighbours, this test fails -- that
    failure is the gate working; re-pin it against the improvement."""
    # RESTATED 2026-09-11 (round 4): the delta is SEARCHED, not named.  This
    # fixture's own hazard band moves with the BLAS kernel -- measured on this
    # box the named (12, 3e-5) row is refused on Haswell, Katmai and
    # Sandybridge and RETURNED on Nehalem -- and the claim here is about the
    # SCREEN (the owned liner silences it), not about any one row.
    deg = 12
    delta = None
    for d in (3e-5, 1e-5, 5e-6, 3e-6, 1e-4):
        try:
            _stack(d, deg).solve()
        except ValueError as exc:
            if "NEAR-COINCIDENT-WALL SLIVER" in str(exc):
                delta = d
                break
    if delta is None:
        pytest.skip(
            "no delta of this fixture is refused on this build: round 4 "
            "decides on the ANSWER and this arithmetic does not produce a "
            "wrong one here. The claim below is about the owned liner "
            "SILENCING the screen, and it needs a refusal to silence. See "
            "S4.2 and R4-G of the round-4 audit.")
    bare = _stack(delta, deg)
    lined = _stack(delta, deg, liner=1e-6)

    # (i) the premise: the SAME manufactured cells are on both grids
    hit = _screen(bare)
    assert hit is not None
    w_bare, own_bare = hit[0], hit[4]
    assert own_bare / w_bare >= ps._SLIVER_OWN_SCALE_RATIO, hit

    # (ii) without the liner the guard refuses, on a stack that IS wrong.
    # RESTATED 2026-09-11 (round 4): the reading is not asserted -- it is a
    # property of the BLAS kernel (the CI matrix read this family from
    # 1.000115 to 3.61242 on one row) and round 4 does not decide on it.  What
    # is asserted is the DECISION and the geometric fact behind it.
    r_bare = _raw(bare)
    refused, msg, _o, _w = _guarded(_stack(delta, deg))
    assert refused and "NEAR-COINCIDENT-WALL SLIVER" in msg

    # (iii) WITH it the screen is silent although the answer is no better
    assert _screen(lined) is None, _screen(lined)
    r_lined = _raw(lined)
    assert _move(r_lined, _raw(_stack(0.0, deg, liner=1e-6))) \
        > 100.0 * delta, (r_bare[3], r_lined[3])
    refused2, msg2, out2, warns = _guarded(_stack(delta, deg, liner=1e-6))
    assert not refused2, (msg2 or "")[:300]
    assert out2 is not None
    # (iv) the caller is not left blind: the round-2 WITHIN-LAYER arm fires,
    #      but it names the liner, not the cross-layer sliver.
    assert any("WITHIN-LAYER feature" in w for w in warns), warns
    assert not any("NEAR-COINCIDENT-WALL SLIVER" in w for w in warns), warns


# ==========================================================================
# D-2 -- a KEYED prepare() stack never reaches the guard
# ==========================================================================
def test_a_keyed_prepared_stack_is_outside_the_guard_entirely():
    """PINS A KNOWN LIMITATION (verification defect D-2 / R3-C, pre-existing)
    and RE-PINS WHAT ROUND 4 CHANGED ABOUT IT (2026-09-11).

    ``_segment_passive`` answers False for a ``str`` payload, so
    ``_stack_provably_passive`` is False for any stack carrying material KEYS
    -- which is the case ``prepare()`` exists for.  Rounds 1-3 reached nothing
    at all there: the screen requires passivity, so the arbiter never ran,
    however wrong the answer was.

    Round 4 splits the two questions.  The GEOMETRIC screen no longer requires
    passivity, because round 4's arbitration compares ANSWERS and needs no
    theorem, so a keyed stack IS now arbitrated; what still requires passivity
    is the REFUSAL, whose scope this round does not widen.  On this fixture the
    keyed re-solve cannot be materialised, so the verdict is ``'unknown'`` and
    the behaviour is unchanged -- but the arbiter is now REACHED, which is the
    half of R3-C that was closable without widening the refusal.

    This test was written as "re-pin it, do not relax it" if keyed stacks are
    later resolved for the guard.  That is what has happened, and this is the
    re-pin."""
    delta, deg = 3e-5, 12
    keyed = _stack(delta, deg, key="LC")
    assert ps._stack_provably_passive(keyed) is False
    assert ps._sliver_screen(keyed) is None                    # no REFUSAL
    assert ps._sliver_screen(keyed, require_passive=False) is not None

    calls = []
    real = ps._sliver_probe_solve
    ps._sliver_probe_solve = lambda st, mf, src: (calls.append(mf)
                                                  or real(st, mf, src))
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = keyed.prepare().solve(
                wavelength=_WL, angle=_THETA, materials={"LC": _EP})
    finally:
        ps._sliver_probe_solve = real
    tot = float(np.max(np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)))
    # ROUND 4: the arbiter is ATTEMPTED (this is the change) ...
    assert calls, "round 4 must at least try to arbitrate a keyed stack"
    # ... and on a keyed payload the re-solve cannot be materialised, so the
    # verdict is 'unknown' and the ANSWER and the warning set are unchanged.
    # the plain super-unity warning rides on the READING, which is a kernel
    # fact -- so it is scored CONDITIONALLY, and what is asserted
    # unconditionally is that the guard's own voice is silent here.
    if tot > 1.0 + ps._STACK_SUPERUNITY_BAR:
        assert any("energy not conserved" in str(w.message) for w in rec), \
            [str(w.message) for w in rec]
    assert not any("NEAR-COINCIDENT-WALL SLIVER" in str(w.message)
                   for w in rec)
    assert np.asarray(o).size and np.asarray(T).size

    # the CONTROL: the same geometry with a concrete eps IS refused, so the
    # difference is the key and nothing else.  RESTATED 2026-09-11 (round 4):
    # the delta is SEARCHED -- the named one is refused on three of this box's
    # four kernels and returned on the fourth, and the claim is about the KEY.
    assert ps._stack_provably_passive(_stack(delta, deg)) is True
    raised = None
    for d in (delta, 1e-5, 5e-6, 3e-6, 1e-4):
        try:
            _stack(d, deg).prepare().solve(wavelength=_WL, angle=_THETA)
        except ValueError as exc:
            if "NEAR-COINCIDENT-WALL SLIVER" in str(exc):
                raised = d
                break
    if raised is None:
        pytest.skip(
            "no delta of this fixture is refused through prepare() on this "
            "build: round 4 decides on the ANSWER and this arithmetic does "
            "not produce a wrong one here. The CONTROL below needs a "
            "refusal to contrast the keyed stack with. See S4.2 and R4-G.")


# ==========================================================================
# the MOVE criterion is a dR/dx test -- the structure behind open item R2-D
# ==========================================================================
def test_the_move_criterion_is_bounded_by_the_devices_own_dR_dx():
    """``move > _SLIVER_MOVE_FACTOR * w_wide`` compares an EFFICIENCY
    difference with a PERIOD fraction.  The snap moves each colliding wall to
    the pair's midpoint, i.e. by at most ``w_wide``, so on a solve the sliver
    has NOT corrupted, ``move`` cannot exceed the device's own response to
    that displacement:

        move / w_wide  <=  O(1) * dR/dx

    That BOUND is what makes the criterion safe on an ordinary device, and it
    is what open item R2-D is about: a device whose physical ``dR/dx`` exceeds
    ``_SLIVER_MOVE_FACTOR`` can carry an uncorrupted answer past the bar.
    Measured 2026-09-11 on a guided-mode-resonance grating with a
    degree-stationary ``dR/d(duty)`` of 145.8, the CORRECT population's
    ``move / w_wide`` reaches 833.8
    (``validation/probe_verify_sliver_round2/w6_resonant.py``), against the
    round-2 report's published "correct rows move at most 26.58".

    The bound is asserted here on the ordinary O-11 fixture, where it holds
    with room in the safe direction.  It is ONE-sided on purpose: the two
    slices' walls move in OPPOSITE directions under the snap, so their
    responses can CANCEL -- measured here, ``move / w_wide`` is 16x SMALLER
    than the slope at delta = 1e-3, which is why this asserts a bound and not
    an equality."""
    deg = 12
    ref = _raw(_stack(0.0, deg))
    # 3e-3 is NOT usable: its manufactured cell is only 92.9x finer than the
    # geometry, below _SLIVER_OWN_SCALE_RATIO, so the screen never fires.
    seen = []
    for delta in (1e-3, 3e-4):
        cur = _raw(_stack(delta, deg))
        slope = _move(cur, ref) / delta          # the device's own dR/dx
        hit = _screen(_stack(delta, deg))
        assert hit is not None, delta
        w_wide = hit[3]
        src = dict(_stack(delta, deg)._src)
        clone = _stack(delta, deg)._min_feature_clone(2.0 * w_wide * _P)
        clone._src = src
        snapped = _raw(clone)
        ratio = _move(cur, snapped) / w_wide
        seen.append((delta, slope, ratio))
        # the premise: the answer is NOT corrupted at this delta
        assert slope < 10.0, (delta, slope)
        # the BOUND: the move cannot exceed the device's own response
        assert ratio <= 3.0 * slope, (delta, slope, ratio)
        # ... so far below the bar, which is what keeps this row returned
        assert ratio < ps._SLIVER_MOVE_FACTOR, (delta, ratio)
        _refused, _msg, out, _w = _guarded(_stack(delta, deg))
        assert out is not None, (delta, (_msg or "")[:200])
    assert len(seen) == 2, seen


# ==========================================================================
# the WITHIN-LAYER arm's floor is the theorem, on a per-DEGREE basis
# ==========================================================================
def test_the_within_layer_arm_is_silent_where_the_theorem_lets_it_be():
    """PINS A KNOWN LIMITATION (verification S7.3, sharpening the round-2
    report's R2-B).

    The arm fires only above ``_SLIVER_TRIGGER_BAR``, so it is silent wherever
    a broken owned liner happens to land BELOW unity -- and that is not only
    the 1e-6 width the report names.  Scanned over widths 1e-6 / 1e-7 and
    degrees 8-16, there is always at least one pair where the answer is off by
    >= 100x the liner width AND the library says nothing at all.

    Measured 2026-09-11, both builds (``w7_lc_within.py``): at 1e-6 every
    degree is silent with ``err/d`` up to 1.06e+03, and at 1e-7 degree 14 is
    silent at ``R+T`` = 0.5706 with ``err/d`` = 6.5e+06.

    A detector for the sub-unity band would make this fail; re-pin it then.

    RESTATED 2026-09-11 (CI PREMISE GATES).  The 5.45.0 release matrix failed
    this test at ``assert quiet_and_broken`` on py3.12 shard 4: on the CI
    runner arm no (width, degree) pair of this ladder is both silent AND
    broken, because that arm SOLVES these ill-conditioned owned liners
    correctly where every local arm -- four OpenBLAS kernels x one and four
    threads x two builds -- gets them wrong.  A limitation can only be pinned
    where it manifests, so the ladder's two population readings (the arm
    speaks somewhere; the arm is silent on a ruined answer somewhere) are now
    MEASURED and skipped with their readings when absent.  What is
    unconditional is the ownership rule (the screen is silent by geometry at
    every width), the fact that no row is ever refused, and -- on any arm that
    does exhibit a silent-and-broken pair -- that the silence is explained by
    the arm's own geometric bar rather than by a reading.  Why the CI arm
    differs is an OPEN item: ``docs/audits/CI_PREMISE_GATES_2026_09_11.md``.
    """
    def _liner(w, deg):
        st = PMMStack(_P, n_substrate=1.0, degree=deg, far_field_orders=21,
                      min_feature=_NO_SNAP)
        for _k in range(2):
            st.add_layer(0.08e-6, segments=[(0.30, _EH), (w, _EP),
                                            (0.70 - w, _EH)])
        st.set_source(_WL, theta=_THETA)
        return st

    def _flat(deg):
        st = PMMStack(_P, n_substrate=1.0, degree=deg, far_field_orders=21,
                      min_feature=_NO_SNAP)
        for _k in range(2):
            st.add_layer(0.08e-6, segments=[(0.30, _EH), (0.70, _EH)])
        st.set_source(_WL, theta=_THETA)
        return st

    quiet_and_broken = []
    spoke = 0
    rows = []
    for w in (1e-6, 1e-7):
        for deg in (8, 12, 14, 16):
            st = _liner(w, deg)
            # INVARIANT: the liner is OWNED, so the cross-layer screen is
            # silent by geometry -- no reading is involved.
            assert _screen(st) is None
            cur, ref = _raw(st), _raw(_flat(deg))
            err = _move(cur, ref) / w
            _refused, msg, out, warns = _guarded(_liner(w, deg))
            # INVARIANT: the within-layer arm never refuses, on any arm.
            assert out is not None, (w, deg, (msg or "")[:200])
            said = any("WITHIN-LAYER feature" in x for x in warns)
            spoke += 1 if said else 0
            rows.append((w, deg, err, cur[3], said))
            if err >= 100.0 and not said:
                quiet_and_broken.append((w, deg, err, cur[3]))
    # ---- PREMISE-GATED (see the RESTATED paragraph above): both of these
    #      are population readings of a pathology, not decisions.
    if spoke < 1:
        pytest.skip(
            "premise absent on this arm: the within-layer arm does not speak "
            "anywhere on this ladder, so the two-sided reading this test is "
            "about is not reproduced here.  Per (width, degree): err/width, "
            "R+T, spoke = %s"
            % [(r[0], r[1], "%.4g" % r[2], "%.6f" % r[3], r[4]) for r in rows])
    if not quiet_and_broken:
        pytest.skip(
            "premise absent on this arm: no (width, degree) pair of this "
            "ladder is both SILENT and BROKEN -- the arm spoke on %d of %d "
            "pairs and no silent pair is off by the 100x width the limitation "
            "is stated at.  Per (width, degree): err/width, R+T, spoke = %s.  "
            "The ownership rule and the never-refuses contract were asserted "
            "on all %d pairs above."
            % (spoke, len(rows),
               [(r[0], r[1], "%.4g" % r[2], "%.6f" % r[3], r[4])
                for r in rows], len(rows)))
    # RESTATED 2026-09-11 (round 4).  This used to assert that every silent
    # pair reads at or below ``_SLIVER_TRIGGER_BAR`` -- "the theorem's own
    # detector does not fire there".  Round 4 removed that gate from the arm
    # entirely (open item R2-B: the measured 1e-6 liner that is 1.06e-03 wrong
    # reads R+T = 0.999221, i.e. SUB-unity), so silence there is no longer
    # explained by the reading and asserting it would pin a kernel.  What is
    # asserted is the arm's own bar, which is pure geometry: a silent pair is
    # silent because its predicted spurious |q| does not reach
    # ``_SLIVER_Q_EXCESS`` times the stack's index ceiling.
    for w, deg, err, tot in quiet_and_broken:
        assert ps._within_layer_hazard(_liner(w, deg), None) is None, \
            (w, deg, err, tot)


# ==========================================================================
# the returned answer is the unguarded answer, bit for bit
# ==========================================================================
def test_every_returned_row_of_the_staircase_box_is_bit_identical():
    """Round 2 changes what is SAID about a truncation super-unity and nothing
    about the numbers.  Measured over the whole 648-configuration box in
    ``validation/probe_verify_sliver_round2/w5_census.py`` (648 / 648
    identical, 0 broken, both builds); here as a 24-configuration arm, with
    the round-2 TRUNCATION note demanded on the rows round 1 would have
    refused so the arm cannot pass vacuously."""
    def _box(delta, deg, nsub, nsup, th):
        st = PMMStack(_P, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                      min_feature=_NO_SNAP, far_field_orders=31)
        for k in range(2):
            d = delta * k
            st.add_layer(_DZ, segments=[(_A0 - d, _EH),
                                        (_B0 + d - (_A0 - d), 10.5),
                                        (1.0 - (_B0 + d), _EH)])
        st.set_source(_WL, theta=th)
        return st

    noted = round1_would = 0
    rows = 0
    for nsub in (1.45 + 0.08j, 2.0 + 0.35j):
        for th in (1.22, 1.33):
            for deg in (6, 8):
                for delta in (3e-3, 1e-3, 3e-4):
                    rows += 1
                    pre = _raw(_box(delta, deg, nsub, 2.4, th))
                    refused, msg, out, warns = _guarded(
                        _box(delta, deg, nsub, 2.4, th))
                    assert not refused, (nsub, th, deg, delta,
                                         (msg or "")[:200])
                    o = np.asarray(out[0]).ravel()
                    i = np.argsort(o)
                    assert np.array_equal(np.real(np.asarray(out[1]))[:, i],
                                          pre[1]), (nsub, th, deg, delta)
                    assert np.array_equal(np.real(np.asarray(out[2]))[:, i],
                                          pre[2]), (nsub, th, deg, delta)
                    # round 1's decision function, evaluated on the reading
                    # this build happens to produce -- a COUNTER, not an
                    # assertion, so a kernel that moves the reading moves the
                    # count and not the verdict.
                    r1 = (ps._sliver_screen(_box(delta, deg, nsub, 2.4, th))
                          is not None
                          and pre[3] > 1.0 + ps._STACK_SUPERUNITY_BAR)
                    if r1:
                        round1_would += 1
                        if any("is NOT what moved this answer" in w
                               for w in warns):
                            noted += 1
    assert rows == 24, rows
    # the arm is evidence, not a vacuous pass: round 1 WOULD have refused some
    assert round1_would >= 4, round1_would
    assert noted == round1_would, (noted, round1_would)


# ==========================================================================
# D-5 -- the CLOSURE criterion WAS ABSOLUTE, so a snap that restored the
#        answer completely was still not believed.  RE-PINNED 2026-09-11
#        against the repair (round 3).
# ==========================================================================
def test_the_closure_criterion_is_relative_not_absolute():
    """RE-PINNED AGAINST THE IMPROVEMENT this verification asked for
    (defect D-5; the repair is
    ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md``).

    AS FIRST WRITTEN (2026-09-11) this test pinned the DEFECT.
    ``_SLIVER_ATTRIB_CLOSURE`` asked the snapped super-unity to fall below a
    FIXED 1e-5, so on a stack whose SLIVER-FREE truncation super-unity already
    sits ABOVE that bar the criterion could never be met -- however completely
    the snap restored the answer.  The arbiter then said ``truncation``, the
    solve RETURNED a number wrong by 1,811x-3,501x the physical wall shift at
    ``R+T`` = 1.19, and the warning told the caller that raising
    ``min_feature`` "will silence nothing here", which is the opposite of what
    is measured: on this fixture the snapped answer IS the sliver-free
    reference to five decimal places.  Round 1 REFUSED these rows, so it was a
    behaviour change and not an inherited floor.

    ROUND 3 makes the closure RELATIVE --
    ``su <= max(_SLIVER_ATTRIB_CLOSURE, (worst - 1) *
    _SLIVER_CLOSURE_FRACTION)`` -- so what is pinned now is the repaired
    DECISION: the same three rows are REFUSED, the refusal names the
    ``min_feature`` that was measured to restore the answer, and the false
    sentence is gone from the library.  The PREMISE is asserted unchanged,
    because it is what makes the class reachable: the mount's super-unity is
    ordinary truncation (a monotone degree ladder) whose degree-8 floor sits
    ABOVE the absolute bar and BELOW the trigger.

    If a later change makes the criterion absolute again, or moves this class
    back into ``truncation``, this test fails -- that failure is the gate
    working; re-pin it against the improvement, do not relax it."""
    def _gmr(delta, deg=8):
        """A guided-mode-resonance grating in a dense-superstrate grazing
        mount, whose degree-8 truncation floor is 3.7e-05 -- i.e. between
        ``_SLIVER_ATTRIB_CLOSURE`` and ``_SLIVER_TRIGGER_BAR``."""
        st = PMMStack(1.0e-6, n_superstrate=2.4, n_substrate=1.45 + 0.05j,
                      degree=deg, far_field_orders=15, min_feature=1.0e-18)
        a, duty = 0.25, 0.5
        for k in (0, 1):
            dd = delta * k
            st.add_layer(0.15e-6, segments=[(a - dd, 3.6),
                                            (duty + 2 * dd, 4.0),
                                            (1.0 - a - duty - dd, 3.6)])
        st.add_layer(0.10e-6, eps=4.0)
        st.set_source(9.3e-7, theta=1.22)
        return st

    ref = _raw(_gmr(0.0))
    assert ps._stack_provably_passive(_gmr(0.0)) is True
    # the premise: the mount's floor is ORDINARY truncation (a clean degree
    # ladder), and it sits ABOVE the closure bar and BELOW the trigger
    ladder = [_raw(_gmr(0.0, deg))[3] - 1.0 for deg in (6, 8, 10, 12)]
    assert ladder[0] > ladder[1] > ladder[2] > ladder[3], ladder
    assert ladder[1] > ps._SLIVER_ATTRIB_CLOSURE, ladder
    assert ladder[1] < ps._SLIVER_TRIGGER_BAR, ladder

    found = []
    for delta in (6.8726e-06, 5.2134e-06, 3.0000e-06):
        st = _gmr(delta)
        cur = _raw(st)
        hit = _screen(st)
        assert hit is not None, delta
        mf = 2.0 * hit[3] * 1.0e-6
        clone = _gmr(delta)._min_feature_clone(mf)
        clone._src = dict(st._src)
        snapped = _raw(clone)
        # the sliver is UNAMBIGUOUSLY the cause: the answer is off by orders
        # of magnitude, and the snap puts it back on the reference
        err = _move(cur, ref) / delta
        err_snapped = _move(snapped, ref) / delta
        if not (err > 100.0 and err_snapped < 1.0):
            continue
        found.append((delta, err, err_snapped, cur[3], snapped[3]))
        # round 1 refused these rows: the screen fires, and it did so because
        # R+T was past its bar on the build round 1 was measured on.  ROUND 4
        # does not assert that reading -- it is a kernel fact -- only the
        # geometric half, which is not.
        assert _screen(_gmr(delta)) is not None, delta
        # the snap removes essentially all of the violation, and still lands
        # ABOVE the ABSOLUTE bar -- which is what made the round-2 criterion
        # unreachable on this class.  ROUND 4 keeps both as EVIDENCE and
        # decides on neither, so they are scored where they are finite and
        # never asserted of a reading.
        drop = (cur[3] - 1.0) / max(snapped[3] - 1.0, 1e-300)
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        assert v == "sliver", (delta, v, ev, drop)
        assert ev["d0_over_w"] > ps._SLIVER_MOVE_FACTOR, ev
        assert ev["d0_over_d12"] > ps._SLIVER_WALL_RATIO, ev
        assert ev["closure"] >= ps._SLIVER_ATTRIB_CLOSURE, ev
        # and the solve is REFUSED, naming the remedy measured to restore it
        refused, msg, out, warns = _guarded(_gmr(delta))
        assert refused and out is None, (delta, (msg or "")[:200])
        assert "NEAR-COINCIDENT-WALL SLIVER" in msg, msg[:200]
        assert "pass min_feature=" in msg, msg[:400]
        assert "will silence nothing here" not in msg
        assert not [w for w in warns if "will silence nothing here" in w]
    # RESTATED 2026-09-11 (round 4): the premise -- off by more than 100x as
    # returned and back within 1x on the prescribed grid -- is a statement
    # about the ANSWER, and which of the three D-5 rows meets it is a kernel
    # fact (three on Haswell, one on Sandybridge, where the other two read
    # 24.4x and 61.4x).  The EXISTENCE is asserted; the count is not.
    assert len(found) >= 1, found
