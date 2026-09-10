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

Evidence: ``docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md`` and
``validation/probe_pmmstack_sliver_round2/``.
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


def _err(a, b):
    """Polarization-1 distance, the campaign's own classification statistic."""
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][1][ia] - b[1][1][ib]).max(),
                     np.abs(a[2][1][ia] - b[2][1][ib]).max()))


def _kind(e, d):
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


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
    """The ROUND-1 path is still live and is still reachable: when the one
    extra solve cannot be run the arbiter answers ``'unknown'`` and the guard
    falls back to exactly what round 1 did.  Forcing that branch executes the
    pre-round-2 decision, and it REFUSES a solve whose answer is within the
    physical wall shift -- the defect this round fixes.

    The fixture is the verification's own false-positive row: lossy substrate,
    1.2 rad, degree 6, a HARMLESS 1e-3 sliver."""
    def _st(d):
        return _stack(d, 6, n_sup=2.5, n_sub=1.5 + 0.05j, eps=12.0,
                      theta=1.2, ffo=31)

    d = 1e-3
    ref, cur = _raw(_st(0.0)), _raw(_st(d))
    e = _err(cur, ref)
    assert _kind(e, d) == "right", (e, e / d)          # the sliver is harmless
    assert cur[3] > 1.0 + ps._STACK_SUPERUNITY_BAR, cur[3]
    assert ps._stack_provably_passive(_st(d)) is True
    assert ps._sliver_screen(_st(d)) is not None       # conjunct (a) fires

    # ROUND 1: no arbiter -> refuse, and the message says the attribution
    # could not be measured.
    monkeypatch.setattr(ps, "_sliver_probe_solve", lambda *a, **k: None)
    refused, msg, _out, _w = _guarded(_st(d))
    assert refused, "round 1's path must refuse this correct solve"
    assert "NEAR-COINCIDENT-WALL SLIVER" in msg
    assert "could NOT be run on this path" in msg, msg[-600:]

    # ROUND 2: the arbiter runs, declines to attribute, and the solve returns.
    monkeypatch.undo()
    refused, _msg, out, warns = _guarded(_st(d))
    assert not refused, "round 2 must return it"
    assert out is not None and len(out) == 4
    assert any("energy not conserved" in w for w in warns), warns
    assert any("is NOT what moved this answer" in w for w in warns), warns


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
def test_the_round1_misses_are_refused_or_are_below_the_trigger():
    """Round 1 RETURNED eight rows in 660 that the continuity rule does not
    call correct.  Round 2 must refuse every one of them that reaches the
    trigger, and the ones it still returns must be exactly the ones whose
    super-unity is below it -- the guard's FLOOR, stated as a decision rather
    than as a margin.

    The floor is not an implementation choice: the family's CORRECT population
    reaches ``|R+T-1|`` = 1.10e-04 over 600 rows and three fixtures, so a
    trigger low enough to catch the last rows would sit ON that population --
    1.01x at 1e-4, where the closure-only form of the arbiter already refuses
    a correct row."""
    refs = {}
    caught, floor = [], []
    for deg, d in _ROUND1_MISSES:
        ref = refs.setdefault(deg, _raw(_stack(0.0, deg)))
        cur = _raw(_stack(d, deg))
        e = _err(cur, ref)
        assert _kind(e, d) != "right", (deg, d, e / d)   # the premise
        assert cur[3] <= 1.0 + ps._STACK_SUPERUNITY_BAR, (
            f"degree {deg}, delta {d:g} is not a round-1 miss on this build "
            f"(R+T = {cur[3]:.6g})")
        refused, msg, _out, _w = _guarded(_stack(d, deg))
        if refused:
            assert "NEAR-COINCIDENT-WALL SLIVER" in msg
            assert cur[3] > 1.0 + ps._SLIVER_TRIGGER_BAR, (deg, d, cur[3])
            caught.append((deg, d, e / d))
        else:
            assert cur[3] <= 1.0 + ps._SLIVER_TRIGGER_BAR, (
                f"degree {deg}, delta {d:g} reaches the trigger "
                f"(R+T-1 = {cur[3] - 1:.3e}) and was NOT refused")
            floor.append((deg, d, e / d))
    assert len(caught) >= 4, (caught, floor)
    # and the floor is where the trigger puts it, not lower
    assert all(f[2] < max(c[2] for c in caught) for f in floor), (caught, floor)


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
    """The two populations the two arbiter bars must separate, measured here:

    * SLIVER rows -- the super-unity must VANISH on the prescribed grid and
      the answer must MOVE far past the snap's own displacement;
    * TRUNCATION rows -- it must SURVIVE.

    Asserted as the verdict, plus one decade of separation on the quantity
    that carries the decision in each population."""
    sliver, trunc = [], []
    for deg, d in ((14, 1e-4), (14, 3e-5), (12, 3e-5), (20, 3e-5),
                   (16, 1e-5)):
        st = _stack(d, deg)
        cur = _raw(st)
        v, ev = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        assert v == "sliver", (deg, d, v, ev)
        sliver.append((ev["snapped_super_unity"], ev["move"] / ev["w_wide"]))
    for nsub, th, deg, d in ((1.5 + 0.05j, 1.2, 6, 1e-3),
                             (3.0 + 2.0j, 1.45, 6, 1e-3),
                             (1.5 + 0.05j, 1.3, 8, 3e-4)):
        st = _stack(d, deg, n_sup=2.5, n_sub=nsub, eps=12.0, theta=th, ffo=31)
        cur = _raw(st)
        got = ps._sliver_arbiter(st, cur[3], cur[1], cur[2], None)
        if got is None or cur[3] <= 1.0 + ps._SLIVER_TRIGGER_BAR:
            continue
        v, ev = got
        assert v == "truncation", (nsub, th, deg, d, v, ev)
        trunc.append((ev["snapped_super_unity"], ev["move"] / ev["w_wide"]))
    assert sliver and trunc, (sliver, trunc)
    # the closure bar sits between the two populations, with decades
    assert max(s[0] for s in sliver) <= ps._SLIVER_ATTRIB_CLOSURE / 10.0, sliver
    assert min(t[0] for t in trunc) >= ps._SLIVER_ATTRIB_CLOSURE * 10.0, trunc
    # and the move bar likewise
    assert min(s[1] for s in sliver) >= ps._SLIVER_MOVE_FACTOR * 3.0, sliver
    assert max(t[1] for t in trunc) <= ps._SLIVER_MOVE_FACTOR / 3.0, trunc


def test_the_trigger_sits_above_the_correct_populations_envelope():
    """TESTING_STANDARDS rule 5, and the reason the floor is where it is.  The
    trigger's job is to keep the arbiter off solves that are RIGHT, so what it
    must clear is the CORRECT population's own super-unity -- measured here
    across three degrees and both sides of the onset, not pinned."""
    envelope = 0.0
    n = 0
    for deg in (10, 14, 20):
        ref = _raw(_stack(0.0, deg))
        for d in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
            cur = _raw(_stack(d, deg))
            if _kind(_err(cur, ref), d) != "right":
                continue
            envelope = max(envelope, abs(cur[3] - 1.0))
            n += 1
    assert n >= 10, n
    assert envelope < ps._SLIVER_TRIGGER_BAR, (envelope,
                                               ps._SLIVER_TRIGGER_BAR)
    # and the warning bar is still a decade above the trigger, so the plain
    # warning's population is unchanged
    assert ps._STACK_SUPERUNITY_BAR >= 10.0 * ps._SLIVER_TRIGGER_BAR


def test_the_arbiter_costs_one_solve_and_only_on_a_triggered_stack():
    """The cost claim, executed: the probe is called exactly ONCE on a stack
    that trips both trigger conditions and NEVER on a healthy one."""
    calls = []
    real = ps._sliver_probe_solve

    def _count(*a, **k):
        calls.append(a[1])
        return real(*a, **k)

    ps._sliver_probe_solve = _count
    try:
        _guarded(_stack(1e-4, 14))            # in the band
        assert len(calls) == 1, calls
        calls.clear()
        for d in (3e-3, 1e-3, 3e-4, 0.0):     # correct, R+T = 1 to 1e-12
            _guarded(_stack(d, 14))
        _guarded(_stack(3e-6, 14, min_feature=_P * 1e-5))   # default snap
        assert calls == [], calls             # healthy: never probed
    finally:
        ps._sliver_probe_solve = real


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
    """The measured consequence: the same O-11 sliver on a 45-degree in-plane
    LC director read ``R+T`` = 2.18 and only WARNED under round 1.  Two-sided
    on three tensor classes -- in-plane director, OUT-OF-PLANE director and
    gyrotropic -- each against the sliver-FREE control of the same stack."""
    classes = {}
    classes["lc_in_plane"] = _uniaxial(np.pi / 4.0, "xy")
    classes["lc_out_of_plane"] = _uniaxial(np.pi / 6.0, "xz")
    G = np.eye(3, dtype=complex) * 4.0
    G[0, 1], G[1, 0] = 0.35j, -0.35j
    classes["gyrotropic"] = G
    fired = []
    for name, M in classes.items():
        st = _stack(3e-5, 14, eps=M)
        assert ps._stack_provably_passive(st) is True, name
        cur = _raw(st)
        assert cur[3] > 1.0 + ps._SLIVER_TRIGGER_BAR, (name, cur[3])
        refused, msg, _o, _w = _guarded(_stack(3e-5, 14, eps=M))
        assert refused and "NEAR-COINCIDENT-WALL SLIVER" in msg, name
        fired.append(name)
        # the control: the same tensor, no sliver
        refused2, msg2, out2, _w2 = _guarded(_stack(3e-3, 14, eps=M))
        assert not refused2, (name, (msg2 or "")[:300])
        assert out2 is not None
    assert len(fired) == 3, fired
    # The OUT-OF-PLANE director is refused on evidence a polarization-1-only
    # statistic cannot see, which is why the move is taken on BOTH: scored
    # against the exact delta -> 0 limit, pol 1 is 0.01x the physical shift
    # (i.e. "correct") while pol 0 is 316x.  Re-derived here.
    M = classes["lc_out_of_plane"]
    ref, cur = _raw(_stack(0.0, 14, eps=M)), _raw(_stack(3e-5, 14, eps=M))
    c = np.intersect1d(cur[0], ref[0])
    ia, ib = np.searchsorted(cur[0], c), np.searchsorted(ref[0], c)
    per_pol = [float(max(np.abs(cur[1][p][ia] - ref[1][p][ib]).max(),
                         np.abs(cur[2][p][ia] - ref[2][p][ib]).max()))
               for p in (0, 1)]
    assert per_pol[1] <= 10.0 * 3e-5, per_pol      # pol 1 looks CORRECT ...
    assert per_pol[0] > 100.0 * 3e-5, per_pol      # ... pol 0 is WRONG
    # ... and the NON-Hermitian payload, which no exact argument makes
    # passive, keeps the behaviour it had: warn and return.
    N = np.eye(3, dtype=complex) * 4.0
    N[0, 1] = 0.2
    assert ps._stack_provably_passive(_stack(3e-5, 14, eps=N)) is False
    refused3, msg3, out3, _w3 = _guarded(_stack(3e-5, 14, eps=N))
    assert not refused3, (msg3 or "")[:300]
    assert out3 is not None


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

    st = _stack(1e-4, 14)
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

    ps._sliver_probe_solve = _spy
    try:
        for mw in (1, 2, 4):
            seen.clear()
            with pytest.raises(ValueError, match="NEAR-COINCIDENT-WALL SLIVER"):
                _stack(1e-4, 14).solve_vs_wavelength(
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

    st = _stack(1e-4, 14)
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
        refused, msg, out, _w = _guarded(_conical(d))
        seen[d] = (kind, refused, cur[3])
        if kind == "wrong":
            assert refused, (d, cur[3])
            assert "NEAR-COINCIDENT-WALL SLIVER" in msg
        elif kind == "right":
            assert not refused, (d, cur[3], (msg or "")[:200])
            assert out is not None
    assert sum(1 for v in seen.values() if v[1]) >= 2, seen
    assert sum(1 for v in seen.values() if not v[1]) >= 2, seen
    # the conical path is where round 2's LOWER trigger pays: this row reads
    # super-unity BELOW round 1's bar and is wrong.
    assert seen[3e-5][0] == "wrong" and seen[3e-5][1], seen[3e-5]
    assert seen[3e-5][2] <= 1.0 + ps._STACK_SUPERUNITY_BAR, seen[3e-5]


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
        refused, msg, out, _w = _guarded(_st(d))
        seen.append((d, e / d, refused))
        if e > 100.0 * d:
            assert refused, (d, e / d, cur[3])
            assert "NEAR-COINCIDENT-WALL SLIVER" in msg
        elif e <= 10.0 * d:
            assert not refused, (d, e / d, (msg or "")[:200])
            assert out is not None
    assert any(v[2] for v in seen) and any(not v[2] for v in seen), seen


# ==========================================================================
# THE SWITCH still restores the pre-fix path, bit for bit
# ==========================================================================
def test_the_fail_before_switch_still_disarms_everything_round_2_added():
    """``PMM_SLIVER_GUARD = False`` is the contract: no refusal, no arbiter
    solve, no within-layer warning -- and the returned numbers are the ones
    the pre-fix library returned."""
    calls = []
    real = ps._sliver_probe_solve
    ps._sliver_probe_solve = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
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
    assert calls == [], calls
    assert out is not None and out2 is not None
    msgs = [str(w.message) for w in rec]
    assert not [m for m in msgs if "WITHIN-LAYER feature" in m], msgs
    assert not [m for m in msgs if "is NOT what moved this answer" in m], msgs
    ref = _raw(_stack(1e-4, 14))
    i = np.argsort(np.asarray(out[0]).ravel())
    assert np.array_equal(np.asarray(out[1])[:, i], ref[1])
