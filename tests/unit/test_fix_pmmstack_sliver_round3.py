"""O-11 ROUND 3 -- the sliver arbiter's closure is RELATIVE (2026-09-11).

Round 2 attributes a super-unity to the sliver when the re-solve on the
prescribed ``min_feature`` grid reads ``su <= _SLIVER_ATTRIB_CLOSURE`` (an
ABSOLUTE 1e-5) AND the answer moves past ``_SLIVER_MOVE_FACTOR`` widest
manufactured cells.  Its independent verification
(``docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md``, defect D-5)
showed the first half cannot be met at all on a stack whose SLIVER-FREE
truncation super-unity already sits above 1e-5: on a guided-mode grating in a
dense-superstrate grazing mount the snap removes a 621x-5,181x super-unity and
puts the answer back on the sliver-free reference to ``err/delta`` = 0.0019,
and round 2 still returned the wrong number -- off by 1,811x-3,501x the
physical wall shift at ``R+T`` = 1.19 -- under a warning saying the prescribed
remedy would "silence nothing".

Round 3 makes the closure RELATIVE: the snap must REMOVE most of the
violation,

    su_snapped <= max(_SLIVER_ATTRIB_CLOSURE,
                      (worst - 1) * _SLIVER_CLOSURE_FRACTION)

keeping the absolute value as the lower arm, so the criterion is a strict
WIDENING -- a ``sliver`` verdict can never become a ``truncation`` one.

Everything asserted here is MEASURED ON THE RUNNING BUILD.  Where a bar could
be pinned, the DECISION is asserted instead, and the one bar test states each
population's envelope and the margin it carries, per
``docs/TESTING_STANDARDS.md`` rule 5.

Evidence: ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md`` and
``validation/probe_fix_sliver_round3/``.
"""
import itertools
import os

# The sliver fixture is a near-degenerate eigenproblem: the classification the
# tests read must not move with the BLAS reduction order, so pin one thread
# before numpy is imported (the pattern of the three sibling sliver files).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

# ---- the D-5 fixture: a guided-mode grating in a dense-superstrate grazing
# ---- mount, whose sliver-FREE truncation floor at degree 8 sits BETWEEN
# ---- _SLIVER_ATTRIB_CLOSURE and _SLIVER_TRIGGER_BAR ------------------------
_GP = 1.0e-6
_GWL = 9.3e-7
_GTH = 1.22
_D5_DELTAS = (6.8726e-06, 5.2134e-06, 3.0000e-06)

# ---- the realistic staircase box, the CORRECT population --------------------
_BP = 1.2e-6
_BWL = 0.85e-6
_BA0, _BB0 = 0.27865, 0.62505
_BEH = 2.25
_NO_SNAP = 1e-12


def _gmr(delta, degree=8, *, wl=_GWL, nsup=2.4, nsub=complex(1.45, 0.05),
         theta=_GTH, duty=0.5):
    """``delta`` opens the second slice's walls by that fraction of a period,
    which is what manufactures the cross-layer sliver."""
    a = 0.5 - duty / 2.0
    st = PMMStack(_GP, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  far_field_orders=15, min_feature=_GP * _NO_SNAP)
    for k in (0, 1):
        dd = delta * k
        st.add_layer(0.15e-6, segments=[(a - dd, 3.6), (duty + 2 * dd, 4.0),
                                        (1.0 - a - duty - dd, 3.6)])
    st.add_layer(0.10e-6, eps=4.0)
    st.set_source(wl, theta=theta)
    return st


def _box(delta, degree, nsub, nsup, theta, nl, eps):
    st = PMMStack(_BP, n_superstrate=nsup, n_substrate=nsub, degree=degree,
                  min_feature=_BP * _NO_SNAP, far_field_orders=31)
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(0.32e-6 / 4,
                     segments=[(_BA0 - dd, _BEH),
                               (_BB0 + dd - (_BA0 - dd), eps),
                               (1.0 - (_BB0 + dd), _BEH)])
    st.set_source(_BWL, theta=theta)
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


def _guarded(st):
    """``(refused, message, result, warning_texts)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            out = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    return (False, "", out, [str(w.message) for w in rec])


def _move(a, b, *, pol=None):
    """Max ``|dR|``, ``|dT|`` over the orders two solves share.  ``pol=None``
    is what the library's arbiter compares; ``pol=1`` is the campaign's
    ``err`` convention."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(a[1][sl][:, ia] - b[1][sl][:, ib]).max(),
                     np.abs(a[2][sl][:, ia] - b[2][sl][:, ib]).max()))


def _screen(st):
    return ps._cross_layer_sliver([L[1] for L in st._layers],
                                  float(st.min_feature) / float(st.period))


def _snapped(st, build, *args, **kw):
    """The unguarded solve on the grid the refusal would prescribe."""
    hit = _screen(st)
    mf = 2.0 * hit[3] * float(st.period)
    clone = build(*args, **kw)._min_feature_clone(mf)
    clone._src = dict(st._src)
    return _raw(clone), hit


def _drop(worst, su_snap):
    return (max(worst - 1.0, 0.0) / su_snap) if su_snap > 0.0 else float("inf")


def _kind(e, d):
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


# ==========================================================================
# (a) THE D-5 REPRODUCER, as a DECISION on both sides
# ==========================================================================
def test_the_d5_rows_are_refused_and_the_refusal_names_min_feature():
    """The three rows defect D-5 names are now REFUSED, with the sliver
    message and the ``min_feature`` remedy.

    The premise is measured here, not assumed: the mount is provably passive,
    its sliver-FREE degree ladder is monotone (so its super-unity is ordinary
    truncation and not a second pathology), and the degree-8 floor sits ABOVE
    ``_SLIVER_ATTRIB_CLOSURE`` and BELOW ``_SLIVER_TRIGGER_BAR`` -- which is
    exactly the configuration on which the ABSOLUTE closure can never be met.

    The rows are then required to be unambiguous: off by more than 100x the
    physical wall shift as returned, and back within 1x of it on the
    prescribed grid.  A row that does not meet that premise is SKIPPED rather
    than failed (a build that moves one row out of the band must not turn this
    into a red test), and the test asserts an existence of at least two."""
    ref = _raw(_gmr(0.0))
    assert ps._stack_provably_passive(_gmr(0.0)) is True
    ladder = [_raw(_gmr(0.0, deg))[3] - 1.0 for deg in (6, 8, 10, 12)]
    assert ladder[0] > ladder[1] > ladder[2] > ladder[3], ladder
    assert ladder[1] > ps._SLIVER_ATTRIB_CLOSURE, ladder
    assert ladder[1] < ps._SLIVER_TRIGGER_BAR, ladder

    found = []
    for delta in _D5_DELTAS:
        st = _gmr(delta)
        cur = _raw(st)
        snapped, hit = _snapped(st, _gmr, delta)
        err = _move(cur, ref, pol=1) / delta
        err_snapped = _move(snapped, ref, pol=1) / delta
        if not (err > 100.0 and err_snapped < 1.0):
            continue
        found.append((delta, err, err_snapped))
        # the snap removes essentially all of the violation, but LANDS ON the
        # mount's own truncation floor -- above the ABSOLUTE bar
        su = max(snapped[3] - 1.0, 0.0)
        assert su > ps._SLIVER_ATTRIB_CLOSURE, (delta, su)
        assert _drop(cur[3], su) > 1.0 / ps._SLIVER_CLOSURE_FRACTION, \
            (delta, _drop(cur[3], su))
        # the arbiter attributes it, and the library refuses
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        assert v == "sliver", (delta, v, ev)
        assert ev["move"] / ev["w_wide"] > ps._SLIVER_MOVE_FACTOR, ev
        assert ev["closure"] > ps._SLIVER_ATTRIB_CLOSURE, ev
        refused, msg, out, warns = _guarded(_gmr(delta))
        assert refused and out is None, (delta, (msg or "")[:200])
        assert "NEAR-COINCIDENT-WALL SLIVER" in msg, msg[:200]
        # the remedy the refusal names is the one that was measured to work
        assert f"min_feature={2.0 * hit[3] * _GP:.4g}" in msg, msg[:400]
        assert "ATTRIBUTION, MEASURED ON THIS CALL" in msg
        # and the round-2 sentence that D-5 showed to be false is gone
        assert "will silence nothing here" not in msg
        assert not [w for w in warns if "will silence nothing here" in w]
    assert len(found) >= 2, found


def test_the_correct_rows_of_the_staircase_box_are_still_returned_bit_identical():
    """The other side of the same decision: the realistic staircase box round
    2 exists to protect.  Every configuration here is CORRECT by the
    campaign's continuity rule (asserted, not assumed), every one is RETURNED,
    and every returned answer is BIT-identical to the unguarded solve -- so
    the relative closure changes what is SAID about these solves and nothing
    about the numbers."""
    n_rows, n_arb = 0, 0
    for nsub, nsup, theta in itertools.product(
            (1.45 + 0.08j, 3.4 + 1.7j), (2.4, 3.2), (1.22, 1.44)):
        ref = _raw(_box(0.0, 6, nsub, nsup, theta, 2, 10.5))
        for d in (3e-3, 1e-3, 3e-4):
            st = _box(d, 6, nsub, nsup, theta, 2, 10.5)
            cur = _raw(st)
            e = _move(cur, ref, pol=1)
            assert _kind(e, d) == "right", (nsub, nsup, theta, d, e / d)
            refused, msg, out, _w = _guarded(
                _box(d, 6, nsub, nsup, theta, 2, 10.5))
            assert not refused, (nsub, nsup, theta, d, (msg or "")[:200])
            i = np.argsort(np.asarray(out[0]).ravel())
            assert np.array_equal(np.real(np.asarray(out[1]))[:, i], cur[1])
            assert np.array_equal(np.real(np.asarray(out[2]))[:, i], cur[2])
            n_rows += 1
            # the arbiter RAN on this row (screen + trigger), and returning it
            # is therefore its ``truncation`` verdict -- no second probe solve
            # is spent to read what the decision already says
            if _screen(st) is not None and cur[3] - 1.0 > ps._SLIVER_TRIGGER_BAR:
                n_arb += 1
    assert n_rows == 24, n_rows
    assert n_arb >= 8, n_arb


# ==========================================================================
# (b) THE BAR: the two DROP populations, each measured here
# ==========================================================================
def test_the_closure_fraction_separates_the_two_drop_populations():
    """``_SLIVER_CLOSURE_FRACTION`` is a bar on the super-unity DROP factor
    ``(worst - 1) / su_snapped``: the snap must remove ``1 / fraction`` of the
    violation.  Both populations are measured HERE, on this build.

    * BELOW it, the CORRECT population: rows whose answer already tracks the
      exact ``delta -> 0`` limit have little for the snap to remove.  Measured
      2026-09-11 on both builds, agreeing to 10 significant figures: the 291
      arbitrated correct rows of the 648-configuration census box reach
      **2.2893**, and over a wider 576-configuration box's 1,078 arbitrated
      correct rows the FINITE envelope is **36.611**, so the shipped 100x
      carries **2.73x** (``validation/probe_fix_sliver_round3/s1_census.py``
      and ``s4_lower_envelope.py``).  21 of those 1,078 have an INFINITE drop
      because their snapped solve leaves the super-unity regime -- which is
      the set the ROUND-2 ABSOLUTE bar admits too, so the relative closure
      adds no correct row to what the closure already let through, and the
      MOVE criterion holds all of them out (their ``move / w_wide`` reaches
      37.007 against the 100x bar).
    * ABOVE it, the D-5 population: rows whose snapped answer IS the
      sliver-free reference but whose residue is the mount's own truncation
      floor.  Measured over 88 such rows on five mounts and two degrees: drop
      **49.107 .. 5,304.6**, of which the shipped bar recovers 85
      (``s3_dropgap.py``); the three it does not are open item R3-A.

    This test re-measures both statistics on subsets of the same two
    families, and asserts each population on its own side of the bar plus the
    SEPARATION between the two populations it measured -- a property of the
    populations, not a multiple of the constant."""
    corr = []
    for nsub, nsup, theta in itertools.product(
            (1.45 + 0.08j, 3.4 + 1.7j), (2.4, 3.2), (1.22, 1.33, 1.44)):
        ref = _raw(_box(0.0, 6, nsub, nsup, theta, 2, 10.5))
        for d in (3e-3, 1e-3, 3e-4):
            st = _box(d, 6, nsub, nsup, theta, 2, 10.5)
            cur = _raw(st)
            if _screen(st) is None or cur[3] - 1.0 <= ps._SLIVER_TRIGGER_BAR:
                continue
            if _kind(_move(cur, ref, pol=1), d) != "right":
                continue
            snapped, _hit = _snapped(st, _box, d, 6, nsub, nsup, theta, 2,
                                     10.5)
            corr.append(_drop(cur[3], max(snapped[3] - 1.0, 0.0)))
    d5 = []
    ref5 = _raw(_gmr(0.0))
    for delta in _D5_DELTAS:
        st = _gmr(delta)
        cur = _raw(st)
        snapped, _hit = _snapped(st, _gmr, delta)
        if _move(cur, ref5, pol=1) / delta <= 100.0:
            continue
        d5.append(_drop(cur[3], max(snapped[3] - 1.0, 0.0)))
    assert len(corr) >= 12, corr
    assert len(d5) >= 2, d5
    bar = 1.0 / ps._SLIVER_CLOSURE_FRACTION
    # each population on its own side of the bar -- the DECISION
    assert max(corr) < bar, (max(corr), bar)
    assert min(d5) > bar, (min(d5), bar)
    # and the separation between the two populations THIS test measured
    assert min(d5) > 10.0 * max(corr), (min(d5), max(corr))


def test_the_relative_closure_can_only_widen_the_absolute_one():
    """The round-3 criterion is a strict WIDENING of round 2's, by
    construction and on measured rows: the closure the arbiter applies is the
    MAX of the round-2 absolute value and the relative one, so it is never
    smaller, and therefore no solve round 2 attributed can stop being
    attributed.  Checked on rows from both populations -- the D-5 rows, where
    the relative arm binds, and the census box's correct rows, where the
    verdict is unchanged."""
    seen = 0
    for build, args in ((_gmr, (_D5_DELTAS[0],)),
                        (_gmr, (_D5_DELTAS[1],)),
                        (_box, (3e-3, 6, 1.45 + 0.08j, 2.4, 1.22, 2, 10.5)),
                        (_box, (1e-3, 6, 3.4 + 1.7j, 3.2, 1.44, 4, 10.5))):
        st = build(*args)
        cur = _raw(st)
        got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        if got is None:
            continue
        _v, ev = got
        if ev is None:
            continue
        seen += 1
        assert ev["closure"] >= ps._SLIVER_ATTRIB_CLOSURE, ev
        round2 = (ev["snapped_super_unity"] <= ps._SLIVER_ATTRIB_CLOSURE
                  and ev["move"] > ps._SLIVER_MOVE_FACTOR * ev["w_wide"])
        round3 = (ev["snapped_super_unity"] <= ev["closure"]
                  and ev["move"] > ps._SLIVER_MOVE_FACTOR * ev["w_wide"])
        assert (not round2) or round3, (args, ev)
    assert seen >= 3, seen


# ==========================================================================
# (c) THE ROUND-2 FIXTURES ARE BIT-IDENTICAL UNDER THE NEW CRITERION
# ==========================================================================
_O_P = 1.2e-6
_O_WL = 0.85e-6
_O_A0, _O_B0 = 0.27865, 0.62505
_O_EH, _O_EP = 2.25, 9.0


def _o11(delta, degree=14, *, n_sup=1.0, n_sub=1.0, eps=_O_EP, theta=0.15,
         ffo=None):
    """The round-2 file's own O-11 fixture, verbatim."""
    kw = {} if ffo is None else dict(far_field_orders=ffo)
    st = PMMStack(_O_P, n_superstrate=n_sup, n_substrate=n_sub, degree=degree,
                  min_feature=_O_P * _NO_SNAP, **kw)
    for k in (0, 1):
        a, b = _O_A0 - delta * k, _O_B0 + delta * k
        st.add_layer(0.32e-6 / 4,
                     segments=[(a, _O_EH), (b - a, eps), (1.0 - b, _O_EH)])
    st.set_source(_O_WL, theta=theta)
    return st


def test_the_round2_fixtures_arbitrate_identically_under_the_relative_closure():
    """The verdicts the round-2 file pins, re-read under the new criterion.

    The five SLIVER rows must still be ``sliver`` (they are the rows whose
    snapped super-unity vanishes outright, so the widened closure cannot move
    them), the three TRUNCATION rows must still be ``truncation`` (they are
    held out by the MOVE criterion, which round 3 does not touch), and the
    returned answers must still be BIT-identical to the unguarded solve."""
    sliver, trunc = [], []
    for deg, d in ((14, 1e-4), (14, 3e-5), (12, 3e-5), (20, 3e-5), (16, 1e-5)):
        st = _o11(d, deg)
        cur = _raw(st)
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        assert v == "sliver", (deg, d, v, ev)
        sliver.append((ev["snapped_super_unity"], ev["move"] / ev["w_wide"],
                       ev["drop"]))
    for nsub, th, deg, d in ((1.5 + 0.05j, 1.2, 6, 1e-3),
                             (3.0 + 2.0j, 1.45, 6, 1e-3),
                             (1.5 + 0.05j, 1.3, 8, 3e-4)):
        st = _o11(d, deg, n_sup=2.5, n_sub=nsub, eps=12.0, theta=th, ffo=31)
        cur = _raw(st)
        got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        if got is None or cur[3] <= 1.0 + ps._SLIVER_TRIGGER_BAR:
            continue
        v, ev = got
        assert v == "truncation", (nsub, th, deg, d, v, ev)
        trunc.append((ev["snapped_super_unity"], ev["move"] / ev["w_wide"],
                      ev["drop"]))
        # and it is the MOVE criterion that holds it out, not the closure
        assert ev["move"] < ps._SLIVER_MOVE_FACTOR * ev["w_wide"], ev
        # RETURNED, and bit-identical to the unguarded answer
        refused, msg, out, _w = _guarded(
            _o11(d, deg, n_sup=2.5, n_sub=nsub, eps=12.0, theta=th, ffo=31))
        assert not refused, (nsub, th, deg, d, (msg or "")[:200])
        i = np.argsort(np.asarray(out[0]).ravel())
        assert np.array_equal(np.real(np.asarray(out[1]))[:, i], cur[1])
        assert np.array_equal(np.real(np.asarray(out[2]))[:, i], cur[2])
    assert sliver and trunc, (sliver, trunc)
    # the sliver rows close outright, so the widened bar cannot reach them
    assert max(s[0] for s in sliver) <= ps._SLIVER_ATTRIB_CLOSURE / 10.0, sliver
    assert min(s[2] for s in sliver) > 1.0 / ps._SLIVER_CLOSURE_FRACTION, sliver
    assert min(s[1] for s in sliver) > ps._SLIVER_MOVE_FACTOR, sliver
    assert max(t[1] for t in trunc) < ps._SLIVER_MOVE_FACTOR, trunc


def test_the_truncation_note_states_the_measured_drop_and_promises_nothing():
    """The warning round 3 changed.  Round 2 ended every ``truncation`` note
    with "Raising min_feature will silence nothing here", which D-5 measured
    to be false on a reachable class.  The note now quotes the measured DROP
    factor and the residue the prescribed grid leaves, and says which of the
    two criteria was not met -- and the sentence that was false is gone from
    the library entirely."""
    import inspect
    # comments stripped: the round-3 comment block QUOTES the sentence it
    # removed, and the claim here is about what the library SAYS
    src = "\n".join(ln for ln in inspect.getsource(ps).splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "will silence nothing here" not in src

    found = None
    said = []
    for nsub, nsup, theta, d in itertools.product(
            (1.45 + 0.08j, 3.4 + 1.7j), (2.4, 3.2), (1.22, 1.44),
            (3e-3, 1e-3, 3e-4)):
        st = _box(d, 6, nsub, nsup, theta, 2, 10.5)
        cur = _raw(st)
        # the note rides on the plain super-unity WARNING, so the row has to
        # be above that bar as well as arbitrated
        if _screen(st) is None or cur[3] - 1.0 <= ps._STACK_SUPERUNITY_BAR:
            continue
        got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        if got is None or got[0] != "truncation":
            continue
        _refused, _m, _o, warns = _guarded(_box(d, 6, nsub, nsup, theta, 2,
                                                10.5))
        said += warns
        note = [w for w in warns if "is NOT what moved this answer" in w]
        if not note:
            continue
        found = (got[1], note[0])
        break
    assert found is not None, "no truncation-noted row in the box subset"
    assert not [w for w in said if "will silence nothing here" in w], said
    ev, text = found
    # the note quotes the measured drop and the residue, to the library's own
    # formatting -- so the sentence cannot drift from what was measured
    assert f"{ev['drop']:.4g}x drop" in text, (text, ev)
    assert f"1+{ev['snapped_super_unity']:.3g}" in text, (text, ev)
    assert "reduce n_slices or raise degree" in text, text
    # and it names WHICH of the two criteria was not met, with that
    # criterion's own bar -- neither branch may promise anything about
    # min_feature that was not measured on this call
    if ev["snapped_super_unity"] <= ev["closure"]:
        assert f"{ps._SLIVER_MOVE_FACTOR:g}x an attribution asks" in text, text
        assert "removes the cell without moving the answer" in text, text
    else:
        assert (f"{1.0 / ps._SLIVER_CLOSURE_FRACTION:.3g}x an attribution "
                f"asks") in text, text
        assert f"leaves 1+{ev['snapped_super_unity']:.3g} standing" in text, \
            text
