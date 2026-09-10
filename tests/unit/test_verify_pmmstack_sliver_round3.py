"""O-11 ROUND 3, INDEPENDENT VERIFICATION -- the five gaps the round-3 test
file does not cover (2026-09-11).

``tests/unit/test_fix_pmmstack_sliver_round3.py`` pins the D-5 repair, the
bit-identity of the staircase box, the closure's two drop populations, the
widening property and the note's FORMAT.  Re-measuring the same claims on an
independent 2,304-row box and on nine D-5 mounts of four different device
mechanisms
(``docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND3_2026_09_11.md``,
``validation/probe_verify_sliver_round3/``) leaves five things unasserted
anywhere, and each is a DECISION rather than a bar:

1. **A CORRECT row past the MOVE bar exists, and must still be returned.**
   The fix reports the correct population's ``move / w_wide`` envelope as
   79.032 against the 100x bar and calls that 1.27x "the bar to watch".  On
   an independent box a CORRECT row reads **161.07**: the bar is not merely
   close, it is CROSSED.  Nothing refuses that row because the CLOSURE arm
   fails on it (drop 0.875 against the 100 an attribution demands), so the
   guard is right -- but the property that keeps it right is the CONJUNCTION,
   and no test asserts that a correct row past the move bar is returned.

2. **The relative closure's own floor (R3-A) is reachable and is not a
   knife-edge.**  On a mount whose sliver-free truncation floor sits just
   under the trigger, a row can be WRONG by 741x the physical wall shift, be
   fully restored by the prescribed snap, move 741x the widest manufactured
   cell -- and still be RETURNED, because its DROP is 5.21 against the 100 the
   closure demands.  The fix records this as open item R3-A with a measured
   floor of 49.107; the family reaches **3.669**.  The test pins the LIMIT as
   a decision so that a later round which closes it fails here and has to
   re-pin, and so that the limit cannot be forgotten.

3. **The ``truncation`` note's TRUTH, not its format.**  The note asserts the
   sliver "is NOT what moved this answer".  That is FALSE on any returned row
   whose answer is WRONG by continuity while the answer on the prescribed grid
   is RIGHT -- which is precisely the D-5 shape.  Measured over 308 returned
   noted rows of the independent box: 0 false.  The test re-measures the
   property on a subset.

4. **The D-5 band is not one mount.**  The fix's D-5 fixture is a single
   guided-mode grating.  Nine mounts across four mechanisms land in the band
   on the verifier's box; the test asserts THREE independent mechanisms reach
   it and are refused, so a later change that narrows the band shows up.

5. **A FALSE REFUSAL exists, and it is INHERITED from round 2.**  Both rounds
   state that no CORRECT row is ever attributed.  A directed sweep of the same
   box's worst mounts at DEGREE 4 finds three wall steps where the library
   refuses an answer the continuity rule calls RIGHT, because the snapped
   solve leaves the super-unity regime (so the closure is satisfied outright,
   under round 2's ABSOLUTE bar as much as round 3's relative one) while the
   mount's own slope carries the move past its bar.  The test pins the defect
   in the shape round 2's verification used: repairing it makes this test
   fail, which is the gate working.  Defect V-4 of the report.

Every number quoted in an assertion message is measured on the running build;
the fixed values are the library's own constants and the geometry.
"""
import itertools
import os

# The sliver fixtures are near-degenerate eigenproblems: the classification
# these tests read must not move with the BLAS reduction order, so pin one
# thread before numpy is imported (the pattern of the four sibling files).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

_NO_SNAP = 1e-12


# ---------------------------------------------------------------- helpers --
def _raw(st):
    """The unguarded solve -- the pre-guard code path, bit for bit."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    R = np.real(np.asarray(R))[:, i]
    T = np.real(np.asarray(T))[:, i]
    return (o[i], R, T, float(np.max(R.sum(axis=-1) + T.sum(axis=-1))))


def _guarded(st):
    """``(refused, message, payload_or_None, warning_texts)``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            out = st.solve()
        except ValueError as exc:
            return (True, str(exc), None, [str(w.message) for w in rec])
    return (False, "", out, [str(w.message) for w in rec])


def _move(a, b, *, pol=None):
    """Max ``|dR|``, ``|dT|`` over the orders two solves share.  ``pol=None``
    is what the arbiter compares; ``pol=1`` is the campaign's ``err``."""
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


def _drop(worst, su):
    return (max(worst - 1.0, 0.0) / su) if su > 0.0 else float("inf")


def _kind(e, d):
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


# ------------------------------------------------------------- fixtures ----
def _stair(delta, *, period, wl, theta, a0, b0, e_lo, e_hi, dz, nl,
           degree, nsub, nsup, ffo):
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub,
                  degree=degree, far_field_orders=ffo,
                  min_feature=period * _NO_SNAP)
    for k in range(nl):
        dd = delta * k / max(nl - 1, 1)
        st.add_layer(dz, segments=[(a0 - dd, e_lo),
                                   (b0 + dd - (a0 - dd), e_hi),
                                   (1.0 - (b0 + dd), e_lo)])
    st.set_source(wl, theta=theta)
    return st


#: The box mount whose CORRECT rows cross the MOVE bar (gap 1) and the same
#: box's ordinary rows, which carry the truncation note (gap 3).
_BOX = dict(period=1.35e-6, wl=1.064e-6, a0=0.3120, b0=0.6790, e_lo=2.56,
            dz=0.12e-6, ffo=31)


def _box(delta, *, nsub=complex(1.52, 0.03), nsup=2.05, theta=1.18,
         degree=6, e_hi=6.76, nl=3):
    return _stair(delta, nsub=nsub, nsup=nsup, theta=theta, degree=degree,
                  e_hi=e_hi, nl=nl, **_BOX)


#: A dense-superstrate grazing staircase whose degree-6 sliver-free floor is
#: 4.52e-04 -- inside the D-5 band and only 2.2x below the trigger, which is
#: what puts its fully-restoring rows BELOW the drop the closure demands.
def _graze(delta, degree=6):
    return _stair(delta, period=1.18e-6, wl=0.72e-6, theta=1.31, a0=0.2870,
                  b0=0.7150, e_lo=3.24, e_hi=4.41, dz=0.19e-6, nl=2,
                  degree=degree, nsup=2.28, nsub=complex(1.46, 0.06), ffo=15)


#: A near-WOOD mount: theta is solved so the -3 order sits just inside its
#: Rayleigh cutoff.  Its degree-8 floor is 9.92e-04, i.e. essentially AT the
#: trigger, which is the extreme of the R3-A band.
def _wood(delta, degree=8, order=-3, period=2.05e-6, n_sup=2.10,
          wl=1.064e-6, inside=2e-3):
    s = (-1.0 * n_sup - order * wl / period) / n_sup * (1.0 - inside)
    th = float(np.arcsin(float(np.clip(s, -0.999999, 0.999999))))
    return _stair(delta, period=period, wl=wl, theta=th, a0=0.2410,
                  b0=0.7040, e_lo=2.31, e_hi=9.61, dz=0.17e-6, nl=2,
                  degree=degree, nsup=n_sup, nsub=complex(2.60, 0.35),
                  ffo=25)


#: The FALSE-REFUSAL mount: a low-degree, many-order, lossy grazing staircase
#: whose sliver-free solve reads BELOW unity, so the snapped solve leaves the
#: super-unity regime and the closure is satisfied outright, while the mount's
#: own dR/dx carries the answer past the MOVE bar.
_FR_DELTAS = (1.6622244079925e-05, 1.2689610031679234e-05,
              7.395465531108587e-06)


def _fr(delta, degree=4):
    return _stair(delta, period=1.02e-6, wl=0.633e-6, theta=1.35,
                  a0=0.3120, b0=0.6790, e_lo=2.56, e_hi=12.25, dz=0.12e-6,
                  nl=3, degree=degree, nsup=3.10, nsub=complex(2.90, 1.10),
                  ffo=31)


#: A guided-mode grating and a Fabry-Perot cavity, the other two mechanisms
#: whose sliver-free floors land in the D-5 band.
def _gmr(delta, degree=6):
    a = 0.5 - 0.46 / 2.0
    st = PMMStack(0.82e-6, n_superstrate=2.15,
                  n_substrate=complex(1.52, 0.04), degree=degree,
                  far_field_orders=15, min_feature=0.82e-6 * _NO_SNAP)
    for k in (0, 1):
        dd = delta * k
        st.add_layer(0.12e-6, segments=[(a - dd, 4.41),
                                        (0.46 + 2.0 * dd, 5.29),
                                        (1.0 - a - 0.46 - dd, 4.41)])
    st.add_layer(0.13e-6, eps=5.29)
    st.set_source(0.78e-6, theta=1.28)
    return st


def _fp(delta, degree=8):
    st = PMMStack(1.45e-6, n_superstrate=1.78, n_substrate=complex(3.48, 0.6),
                  degree=degree, far_field_orders=21,
                  min_feature=1.45e-6 * _NO_SNAP)
    st.add_layer(0.11e-6, segments=[(0.2140, 2.10), (0.5220, 10.24),
                                    (0.2640, 2.10)])
    st.add_layer(1.02e-6, eps=2.10)
    st.add_layer(0.11e-6, segments=[(0.2140 - delta, 2.10),
                                    (0.5220 + 2.0 * delta, 10.24),
                                    (0.2640 - delta, 2.10)])
    st.set_source(1.31e-6, theta=0.94)
    return st


# ==========================================================================
# GAP 1 -- a CORRECT row PAST the move bar, and the conjunction that saves it
# ==========================================================================
def test_a_correct_row_past_the_move_bar_is_still_returned():
    """The move criterion alone does NOT separate the correct population.

    This row is CORRECT by the campaign's continuity rule (asserted here, not
    assumed) and its answer moves **161x** the widest manufactured cell --
    past the 100x bar that criterion applies.  It is returned anyway, because
    the CLOSURE arm reads a drop of 0.87 against the 100 an attribution asks
    for.  So the property that makes the guard right on this row is the
    CONJUNCTION, not either arm; a change that ever drops one arm would refuse
    a correct answer here.

    Measured 2026-09-11, Windows and WSL agreeing to 8 significant figures:
    ``move / w_wide`` = 161.0729 / 161.0729, drop = 0.8749904 / 0.8749904,
    ``err/delta`` = 3.087 (RIGHT), ``R+T`` = 1.05539.  The correct
    population's move envelope over the 2,304-row box this row comes from is
    that same 161.07, against the fix's published 79.03."""
    delta = 1.0e-4
    ref = _raw(_box(0.0))
    st = _box(delta)
    cur = _raw(st)
    err = _move(cur, ref, pol=1)
    assert _kind(err, delta) == "right", (err / delta, cur[3])

    snapped, hit = _snapped(st, _box, delta)
    su = max(snapped[3] - 1.0, 0.0)
    move_w = _move(cur, snapped) / hit[3]
    drop = _drop(cur[3], su)
    # the row is arbitrated at all
    assert cur[3] - 1.0 > ps._SLIVER_TRIGGER_BAR, cur[3]
    # ... the MOVE arm is MET on a correct row -- the bar is crossed
    assert move_w > ps._SLIVER_MOVE_FACTOR, (move_w, err / delta)
    # ... and the CLOSURE arm is what holds it out, by a wide margin
    assert drop < 1.0 / ps._SLIVER_CLOSURE_FRACTION, (drop, move_w)
    assert su > max(ps._SLIVER_ATTRIB_CLOSURE,
                    (cur[3] - 1.0) * ps._SLIVER_CLOSURE_FRACTION), su

    v, ev = ps._sliver_arbiter(_box(delta), cur[3], cur[1], cur[2], None)
    assert v == "truncation", (v, ev)
    assert ev["move"] / ev["w_wide"] > ps._SLIVER_MOVE_FACTOR, ev
    # the DECISION: returned, and bit-identical to the unguarded answer
    refused, msg, out, _w = _guarded(_box(delta))
    assert not refused, (msg or "")[:200]
    i = np.argsort(np.asarray(out[0]).ravel())
    assert np.array_equal(np.real(np.asarray(out[1]))[:, i], cur[1])
    assert np.array_equal(np.real(np.asarray(out[2]))[:, i], cur[2])


# ==========================================================================
# GAP 2 -- R3-A: the relative closure's own floor, pinned as a DECISION
# ==========================================================================
def test_the_relative_closure_leaves_a_restorable_wrong_answer_returned():
    """Open item R3-A, as a measured limit rather than a sentence.

    On a mount whose sliver-FREE truncation floor sits just under the trigger,
    the arbiter cannot demand a 100x drop: the violation is barely above the
    trigger and the snapped residue is the mount's own floor, so the ratio is
    single-digit however completely the snap restores the answer.  Both rows
    below are WRONG as returned, are RIGHT on the prescribed grid, and move
    far past the MOVE bar -- and both are RETURNED.

    Measured 2026-09-11 on both builds: the grazing mount's degree-6 floor is
    4.5171e-04 (2.2x below the trigger), its row reads drop **5.205** against
    the 100 demanded, ``err/delta`` **741.26** returned and **0.107** snapped,
    ``move / w_wide`` 741.36; the near-Wood mount's degree-8 floor is
    9.9188e-04 (1.008x below the trigger), its row reads drop **3.669**,
    ``err/delta`` **212.76** returned and **1.675** snapped, ``move / w_wide``
    212.58.  The fix reports this population's drop floor as 49.107, which is
    a property of its five mounts; the family reaches 3.669, i.e. INSIDE the
    correct population's own drop range.

    If a later round closes R3-A this test fails -- that failure is the gate
    working; re-pin it against the improvement, do not relax it."""
    seen = []
    for name, build, delta in (("graze", _graze, 2.452760662977706e-06),
                               ("wood", _wood, 2.3950266199874907e-05)):
        ref = _raw(build(0.0))
        floor = ref[3] - 1.0
        # the premise: the mount's own floor is inside the D-5 band and close
        # enough to the trigger that the drop cannot reach the demanded 100
        assert ps._SLIVER_ATTRIB_CLOSURE < floor < ps._SLIVER_TRIGGER_BAR, \
            (name, floor)
        st = build(delta)
        cur = _raw(st)
        if cur[3] - 1.0 <= ps._SLIVER_TRIGGER_BAR:
            continue                       # not arbitrated on this build
        snapped, hit = _snapped(st, build, delta)
        su = max(snapped[3] - 1.0, 0.0)
        err = _move(cur, ref, pol=1) / delta
        err_snap = _move(snapped, ref, pol=1) / delta
        drop = _drop(cur[3], su)
        move_w = _move(cur, snapped) / hit[3]
        if not (err > 100.0 and err_snap < 10.0):
            continue                       # premise not met -- skip, not fail
        seen.append((name, drop, err, move_w))
        # every piece of evidence an attribution needs EXCEPT the drop
        assert move_w > ps._SLIVER_MOVE_FACTOR, (name, move_w)
        assert _kind(_move(snapped, ref, pol=1), delta) == "right", \
            (name, err_snap)
        # ... and the drop is below what the closure demands, so the row is
        # RETURNED: that is the limit, measured
        assert drop < 1.0 / ps._SLIVER_CLOSURE_FRACTION, (name, drop)
        refused, msg, out, _w = _guarded(build(delta))
        assert not refused, (name, (msg or "")[:200])
        assert out is not None
    # Both rows meet the premise on both builds today (the guards carry 2.1x
    # and 6.0x); the assertion is an EXISTENCE so that a build which moves one
    # row out of the band reports through the other rather than turning this
    # into a red test.
    assert seen, "no row met the premise on this build"
    # ... and not marginally below the bar: a full DECADE inside it, which is
    # why loosening _SLIVER_CLOSURE_FRACTION by a step or two does not reach
    # these rows either.  Measured 2026-09-11: 5.205 and 3.669 against the
    # decade-inside value of 10, i.e. 1.9x and 2.7x, cross-build spread 1.2e-08.
    assert min(s[1] for s in seen) < 0.1 / ps._SLIVER_CLOSURE_FRACTION, seen


# ==========================================================================
# GAP 3 -- the truncation note's TRUTH on a returned row
# ==========================================================================
def test_the_truncation_note_is_never_false_on_a_returned_row():
    """The note says the sliver "is NOT what moved this answer".  On a
    returned row that is WRONG by continuity while the answer on the
    prescribed grid is RIGHT, that sentence is false -- and that is exactly
    the shape defect D-5 raised against round 2.

    Every returned row of this subset that carries the note is scored against
    the exact ``delta -> 0`` reference and against the prescribed-grid solve.
    Measured 2026-09-11 over the 308 returned noted rows of a 2,304-row box:
    **0** false, on both builds; 261 of the 308 are RIGHT as returned and 47
    are GREY.  Here the same property is re-measured on a subset."""
    noted = 0
    false = []
    for nsub, nl, delta in itertools.product(
            (complex(1.52, 0.03), complex(2.90, 1.10)), (2, 3),
            (3e-3, 1e-3, 3e-4, 1e-4)):
        ref = _raw(_box(0.0, nsub=nsub, nl=nl))
        st = _box(delta, nsub=nsub, nl=nl)
        cur = _raw(st)
        if _screen(st) is None or cur[3] - 1.0 <= ps._SLIVER_TRIGGER_BAR:
            continue
        refused, _m, _o, warns = _guarded(_box(delta, nsub=nsub, nl=nl))
        if refused:
            continue
        if not [w for w in warns if "is NOT what moved this answer" in w]:
            continue
        noted += 1
        snapped, _hit = _snapped(st, _box, delta, nsub=nsub, nl=nl)
        k_ret = _kind(_move(cur, ref, pol=1), delta)
        k_snap = _kind(_move(snapped, ref, pol=1), delta)
        if k_ret == "wrong" and k_snap == "right":
            false.append((nsub, nl, delta, k_ret, k_snap))
    assert noted >= 3, noted
    assert not false, false


# ==========================================================================
# GAP 5 -- a FALSE REFUSAL exists, and it is INHERITED from round 2
# ==========================================================================
def test_a_correct_answer_is_refused_when_the_snap_leaves_the_superunity_regime():
    """A DEFECT-PINNING test, in the shape round 2's verification used: it
    asserts the WRONG behaviour so that repairing it makes this test fail.
    If the guard stops refusing this row, that failure is the gate working --
    re-pin it against the improvement, do not delete it.

    Both rounds claim no CORRECT row is ever attributed: round 3's report says
    "0 at every fraction from 1e-1 to 1e-3, on both builds, because the MOVE
    criterion holds every one out", and round 2's verification reports
    0 / 648 false positives.  A directed scan of a mount at DEGREE 4 -- a
    low-degree, many-order, lossy grazing mount whose sliver-free solve reads
    BELOW unity -- finds three wall steps where the conjunction fires on an
    answer the campaign's own continuity rule calls RIGHT.

    Why both arms are met with no pathology present:

    * the snapped solve reads at or BELOW unity, so ``su_snapped`` is exactly
      0 and the closure is satisfied outright -- by the ROUND-2 ABSOLUTE bar
      as well as by the relative one, which is why this is inherited and not a
      round-3 regression;
    * the mount's own ``dR/dx`` is large, so a snap that shifts the walls by
      one manufactured cell moves the answer 115-256 cells -- past the MOVE
      bar -- without the answer being wrong.

    Measured 2026-09-11 on both builds, and on the round-3 branch point
    ``f2371e0`` as well: 3 / 3 refused by the library on all three arms;
    ``err/delta`` 0.979 / 1.065 / 1.051 (RIGHT); and the remedy the refusal
    names first does NOT help -- the answer on the prescribed grid is FARTHER
    from the degree-16 solve than the returned answer is, on all three rows.
    """
    conv = _raw(_fr(0.0, degree=16))
    ref = _raw(_fr(0.0))
    refused = 0
    for delta in _FR_DELTAS:
        st = _fr(delta)
        cur = _raw(st)
        err = _move(cur, ref, pol=1)
        if _kind(err, delta) != "right":
            continue                       # premise not met -- skip, not fail
        snapped, hit = _snapped(st, _fr, delta)
        su = max(snapped[3] - 1.0, 0.0)
        move_w = _move(cur, snapped) / hit[3]
        # the row is arbitrated, and BOTH arms are met
        assert cur[3] - 1.0 > ps._SLIVER_TRIGGER_BAR, (delta, cur[3])
        assert su <= ps._SLIVER_ATTRIB_CLOSURE, (delta, su)
        assert move_w > ps._SLIVER_MOVE_FACTOR, (delta, move_w)
        # ... and the closure is met by the ROUND-2 ABSOLUTE bar as well, so
        # the relative closure has nothing to do with this refusal
        assert su <= ps._SLIVER_ATTRIB_CLOSURE, (delta, su)
        assert su <= max(ps._SLIVER_ATTRIB_CLOSURE,
                         (cur[3] - 1.0) * ps._SLIVER_CLOSURE_FRACTION), su
        v, _ev = ps._sliver_arbiter(_fr(delta), cur[3], cur[1], cur[2], None)
        assert v == "sliver", (delta, v)
        # ... and the library refuses a CORRECT answer
        got, msg, out, _w = _guarded(_fr(delta))
        assert got and out is None, (delta, err / delta)
        assert "NEAR-COINCIDENT-WALL SLIVER" in msg, msg[:200]
        refused += 1
        # the remedy it names first does not improve the answer
        assert (_move(snapped, conv, pol=1)
                >= _move(cur, conv, pol=1)), (delta, "snap helped")
    assert refused >= 2, refused


# ==========================================================================
# GAP 4 -- the D-5 band is reached by three independent mechanisms
# ==========================================================================
def test_the_d5_band_is_reached_by_three_independent_mechanisms():
    """The class defect D-5 names is not one grating.

    A D-5 mount is one whose SLIVER-FREE truncation super-unity sits between
    ``_SLIVER_ATTRIB_CLOSURE`` and ``_SLIVER_TRIGGER_BAR`` -- the band on which
    round 2's absolute closure can never be met.  Three physically unrelated
    devices land in it: a guided-mode grating in a dense-superstrate grazing
    mount (degree-6 floor 7.63e-05), a Fabry-Perot cavity between two
    corrugated mirrors (degree-8 floor 8.27e-05), and a near-Wood mount whose
    -2 order sits just inside its Rayleigh cutoff (degree-8 floor 1.55e-04).
    Nine such mounts across four mechanisms were found on the verifier's box;
    three are asserted here, each refused at a wall step where the answer is
    wrong and the prescribed snap restores it."""
    mounts = (("gmr", _gmr, 1.0e-6),
              ("fp", _fp, 1.0e-6),
              ("wood2", (lambda d, degree=8: _wood(
                  d, degree=degree, order=-2, period=1.62e-6, n_sup=1.90)),
               3.0e-6))
    ok = []
    for name, build, delta in mounts:
        ref = _raw(build(0.0))
        floor = ref[3] - 1.0
        assert ps._SLIVER_ATTRIB_CLOSURE < floor < ps._SLIVER_TRIGGER_BAR, \
            (name, floor)
        st = build(delta)
        cur = _raw(st)
        snapped, hit = _snapped(st, build, delta)
        su = max(snapped[3] - 1.0, 0.0)
        err = _move(cur, ref, pol=1) / delta
        err_snap = _move(snapped, ref, pol=1) / delta
        if not (err > 100.0 and err_snap < 10.0):
            continue                       # premise not met -- skip, not fail
        # the ABSOLUTE bar cannot be met here: the residue IS the floor
        assert su > ps._SLIVER_ATTRIB_CLOSURE, (name, su)
        drop = _drop(cur[3], su)
        assert drop > 1.0 / ps._SLIVER_CLOSURE_FRACTION, (name, drop)
        assert _move(cur, snapped) / hit[3] > ps._SLIVER_MOVE_FACTOR, name
        refused, msg, _o, _w = _guarded(build(delta))
        assert refused, (name, delta, cur[3], drop)
        assert "NEAR-COINCIDENT-WALL SLIVER" in msg, msg[:200]
        assert "ATTRIBUTION, MEASURED ON THIS CALL" in msg, msg[:400]
        ok.append((name, floor, drop, err))
    assert len(ok) >= 3, ok
