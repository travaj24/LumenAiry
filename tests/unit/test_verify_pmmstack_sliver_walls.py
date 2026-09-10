"""VERIFICATION of the O-11 ``PMMStack`` sliver refusal, and the two follow-ups
this verification shipped.

Companion to ``tests/unit/test_fix_pmmstack_sliver_walls.py``, which owns the
fix's own claims.  What is pinned HERE is what re-measuring that fix found:

* **follow-up 1** -- ``_pmm_union_grid``'s ``min_feature`` comparison now
  carries a round-off DEADBAND, so two pairs a caller made the same width are
  treated the same way on both sides (open item A).  Two-sided: the deadband is
  sized against wall round-off MEASURED here, the symmetric case is cured, and
  every case off the threshold is bit-identical to a verbatim re-implementation
  of the pre-fix loop.
* **the guard's two contracts, stated as decisions** -- it refuses no solve
  that agrees with the exact ``delta -> 0`` limit, and it returns no solve that
  violates the passivity theorem above its own bar.
* **the guard's measured FLOOR** -- the population the theorem cannot reach.
  The fix's own ladder shows 3.9x of gap above the bar; a denser walk of the
  SAME family finds wrong solves BELOW it.  Bounded, not asserted away.
* **the exemptions' measured cost** -- a thin feature owned by one layer is
  exempt by design, and that exemption is silent at widths where the same
  mechanism is already catastrophic.
* **the 2-D / mortar route** -- the pure stack's shared grid cannot express a
  sliver at all, the hybrid has no union grid, and the per-layer mortar carries
  a within-layer sliver without a silent wrong answer.

Everything is measured on the running build; the references are EXACT limits
(``delta -> 0``, a vanishing stripe), never a prior run's numbers.

Evidence: ``docs/audits/VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md`` and
``validation/probe_verify_sliver/``.
"""
import os

# Near-degenerate eigenproblems: pin one BLAS thread before numpy loads, the
# pattern of test_fix_pmmstack_sliver_walls / test_v5_13_0_pmm_tapered.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402
from lumenairy.elements.pmm._core import (  # noqa: E402
    _WALL_SNAP_DEADBAND,
    _pmm_union_grid,
)
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

# ---- the O-11 fixture, verbatim ------------------------------------------
_P, _WL, _THETA = 1.2e-6, 0.85e-6, 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505
_NO_SNAP = _P * 1e-12          # below the function's own 1e-9 fractional tol

# ---- a SECOND fixture, whose efficiency sensitivity to a wall is 4x the
# O-11 one's -- the arm that shows which bars are the structure's and which
# are the fixture's.
_Q = dict(P=0.9e-6, WL=0.62e-6, TH=0.21, DZ=70e-9, EH=2.0, EP=6.5,
          A=0.311, B=0.688, NSUB=1.46)


def _segs(a, b, eh=_EH, ep=_EP):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def _solve(delta, degree=14, *, guard=True, min_feature=_NO_SNAP, fx=None):
    f = fx or dict(P=_P, WL=_WL, TH=_THETA, DZ=_DZ, EH=_EH, EP=_EP,
                   A=_A0, B=_B0, NSUB=1.0)
    st = PMMStack(f["P"], n_superstrate=1.0, n_substrate=f["NSUB"],
                  degree=degree, min_feature=min_feature)
    for (a, b) in [(f["A"], f["B"]), (f["A"] - delta, f["B"] + delta)]:
        st.add_layer(f["DZ"], segments=_segs(a, b, f["EH"], f["EP"]))
    st.set_source(f["WL"], theta=f["TH"])
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot))


def _err(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


# ==========================================================================
# FOLLOW-UP 1 -- the min_feature round-off deadband (open item A)
# ==========================================================================
def _union_widths_prefix(layer_segments, min_feature):
    """A VERBATIM re-implementation of the pre-deadband snap loop -- the
    fail-before arm.  Only the comparison differs from the shipped one."""
    walls = {0.0, 1.0}
    owners_of = {0.0: set(), 1.0: set()}
    for li, segs in enumerate(layer_segments):
        w = np.asarray([float(s[0]) for s in segs], dtype=float)
        cw = np.concatenate([[0.0], np.cumsum(w)])
        cw[-1] = 1.0
        for x in cw:
            x = float(x)
            walls.add(x)
            owners_of.setdefault(x, set()).add(li)
    uw = np.array(sorted(walls))
    if uw.size <= 2:
        return np.diff(uw)
    tol = 1e-9
    keep, owners = [uw[0]], [owners_of.get(float(uw[0]), set())]
    for w in uw[1:]:
        if w - keep[-1] > tol:
            keep.append(w)
            owners.append(owners_of.get(float(w), set()))
        else:
            owners[-1] = owners[-1] | owners_of.get(float(w), set())
    if keep[-1] < uw[-1]:
        keep[-1] = uw[-1]
    if min_feature is not None and float(min_feature) > tol:
        mf = float(min_feature)                       # <-- the PRE-FIX test
        out_w, out_o = [keep[0]], [owners[0]]
        for w, ow in zip(keep[1:], owners[1:]):
            d = w - out_w[-1]
            if d < mf and 0.0 < out_w[-1] and w < 1.0 and not (out_o[-1] & ow):
                out_w[-1] = 0.5 * (out_w[-1] + w)
                out_o[-1] = out_o[-1] | ow
            else:
                out_w.append(w)
                out_o.append(ow)
        keep = out_w
    return np.diff(np.array(keep))


_PAIRS = ((_A0, _B0), (0.311, 0.688), (0.1907, 0.5533))


def _two_layer(a, b, d):
    return [_segs(a, b), _segs(a - d, b + d)]


def test_the_snap_threshold_treats_a_symmetric_pair_symmetrically():
    """FAIL-BEFORE and after, on the SAME call.  A caller who sets
    ``min_feature`` to the width their two collisions have gets two
    separations that straddle it by a fraction of an ULP -- which the strict
    comparison resolved by merging ONE of them.  Both are measured here."""
    straddle, asym_before, cases = 0, 0, 0
    for (a, b) in _PAIRS:
        for mf in (1e-6, 1e-5, 3e-5, 1e-4):
            segs = _two_layer(a, b, mf)
            sep_l = a - (a - mf)
            sep_r = (b + mf) - b
            # the premise: both separations are ``mf`` to within wall
            # round-off, i.e. INSIDE the deadband the fix introduces
            assert abs(sep_l - mf) <= _WALL_SNAP_DEADBAND, (sep_l, mf)
            assert abs(sep_r - mf) <= _WALL_SNAP_DEADBAND, (sep_r, mf)
            if (sep_l < mf) != (sep_r < mf):           # ... and they straddle
                straddle += 1
            before = _union_widths_prefix(segs, mf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                after, _e = _pmm_union_grid(segs, mf)
            if before.size == 4:                       # one of the pair merged
                asym_before += 1
                assert (sep_l < mf) != (sep_r < mf), (a, b, mf)
            assert after.size != 4, (a, b, mf, after)  # never asymmetric now
            cases += 1
    # MEASURED 2026-09-11, both builds: 7 of these 12 straddle and every one of
    # those 7 produced the 4-cell asymmetric grid before the deadband.
    assert straddle >= 4, straddle
    assert asym_before == straddle, (asym_before, straddle)
    assert cases == len(_PAIRS) * 4


def test_the_deadband_changes_nothing_off_the_threshold():
    """The other side: away from the threshold the shipped grid is BIT-equal
    to the pre-fix loop, across four ``min_feature`` values and four decades of
    ``delta`` either side of each."""
    same = diff = 0
    for (a, b) in _PAIRS:
        for mf in (1e-5, 3e-5, 1e-4, 1e-3):
            for f in np.geomspace(1e-2, 1e2, 60):
                d = float(mf * f)
                if abs(d - mf) <= _WALL_SNAP_DEADBAND:
                    continue
                segs = _two_layer(a, b, d)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    now, _e = _pmm_union_grid(segs, mf)
                if np.array_equal(now, _union_widths_prefix(segs, mf)):
                    same += 1
                else:
                    diff += 1
    assert diff == 0, diff
    assert same >= 600, same


def test_the_deadband_is_sized_above_the_wall_round_off_measured_here():
    """The constant's origin, re-derived on the running build: the deadband
    must exceed the worst ``|d - min_feature|`` a nominally-exact geometry can
    produce, with headroom, and stay far below the function's own dedup tol."""
    rng = np.random.default_rng(20260911)
    worst = 0.0
    for _ in range(4000):
        a = float(rng.uniform(0.05, 0.45))
        b = float(rng.uniform(0.55, 0.95))
        mf = float(10.0 ** rng.uniform(-6, -2))
        for sep in (a - (a - mf), (b + mf) - b):
            worst = max(worst, abs(sep - mf))
    # MEASURED 2026-09-11, both builds: 1.565e-16 over 120k layouts (0.70 ULP
    # of 1.0).  The deadband is 16 ULP, so it clears the round-off by ~23x and
    # sits 6 decades below the 1e-9 fractional dedup tol it must not reach.
    assert worst < _WALL_SNAP_DEADBAND / 5.0, (worst, _WALL_SNAP_DEADBAND)
    assert _WALL_SNAP_DEADBAND < 1e-9 / 1e4, _WALL_SNAP_DEADBAND


# ==========================================================================
# THE GUARD'S TWO CONTRACTS, as decisions
# ==========================================================================
_DENSE = [(deg, float(d)) for deg in (10, 14)
          for d in np.geomspace(3e-3, 1e-6, 24)]


def test_the_guard_refuses_no_solve_that_agrees_with_the_exact_limit():
    """FALSE-POSITIVE arm.  Over a dense two-degree walk, every row whose
    answer tracks the physical shift (within 10x the structure's own measured
    continuity) must come back untouched -- and bit-for-bit."""
    refs = {deg: _solve(0.0, deg, guard=False) for deg in (10, 14)}
    checked = 0
    for deg, d in _DENSE:
        pre = _solve(d, deg, guard=False)
        if _err(pre, refs[deg]) > 10.0 * d:
            continue
        post = _solve(d, deg, guard=True)          # must not raise
        for x, y in zip(pre[:3], post[:3]):
            assert np.array_equal(x, y), (deg, d)
        checked += 1
    assert checked >= 20, checked


def test_the_guard_returns_no_solve_that_violates_the_theorem_above_its_bar():
    """The guard's OWN contract, unconditional: on a provably passive stack
    with a lossless incidence medium, nothing that comes back may read
    super-unity above ``_STACK_SUPERUNITY_BAR``."""
    n = 0
    for deg, d in _DENSE:
        try:
            res = _solve(d, deg, guard=True)
        except ValueError as exc:
            assert "NEAR-COINCIDENT-WALL SLIVER" in str(exc), str(exc)[:120]
            continue
        assert res[3] <= 1.0 + ps._STACK_SUPERUNITY_BAR, (deg, d, res[3])
        n += 1
    assert n >= 20, n


def test_the_guard_has_a_measured_floor_the_theorem_cannot_reach():
    """The guard's residual, BOUNDED rather than asserted away.  Conjunct (b)
    is a passivity theorem, so a wrong answer that still closes to better than
    the bar is returned.  Scan for one; whatever the scan finds must stay
    inside the bound this test measures, and the snapped remedy must land on
    the continuity slope -- which is what proves the found row wrong."""
    deg = 14
    ref = _solve(0.0, deg, guard=False)
    assert abs(ref[3] - 1.0) < 1e-9, ref[3]        # premise: exact reference
    found, returned = [], 0
    for d in np.geomspace(3e-5, 1e-6, 60):
        d = float(d)
        try:
            res = _solve(d, deg, guard=True)
        except ValueError:
            continue                                # refused: the guard worked
        returned += 1
        e = _err(res, ref)
        if e > 100.0 * d:
            found.append((d, e, res[3] - 1.0))
    # the ladder must actually reach the guard: MEASURED 2026-09-11, both
    # builds, 57 of these 60 rows are REFUSED and 3 come back -- so a run that
    # returns nothing has stopped exercising the floor and must say so.
    assert returned >= 2, returned
    if not found:
        # The ladder is exhausted with NO miss -- a stronger guard than the one
        # measured on 2026-09-11.  Never a skip (TESTING_STANDARDS rule 4): the
        # claim becomes the unconditional one, that every returned row tracks
        # the physical shift, and this test then fails honestly if a LATER
        # change reopens the floor.
        for d in np.geomspace(3e-5, 1e-6, 60):
            d = float(d)
            try:
                res = _solve(d, deg, guard=True)
            except ValueError:
                continue
            assert _err(res, ref) <= 100.0 * d, (d, _err(res, ref))
        return
    worst = max(f[1] for f in found)
    # MEASURED 2026-09-11, BOTH builds: 8 unrefused wrong rows over 660 samples
    # (5 degrees x 60 deltas + a 120-point walk), worst absolute per-order
    # error 2.80e-03 and worst R+T-1 = +7.14e-03.  The bound is 3e-2, ~10x the
    # observed worst and still 2.4 decades below the 0.48-8.6 the REFUSED rows
    # carry -- so it fails honestly if the floor deepens and does not pin a
    # sampling artefact.
    assert worst < 3e-2, (worst, found)
    for d, _e, rt1 in found:
        assert rt1 <= ps._STACK_SUPERUNITY_BAR, (d, rt1)
    # and the remedy the refusal WOULD have prescribed proves them wrong
    d0, e0, _rt = max(found, key=lambda f: f[1])
    hit = ps._cross_layer_sliver(_two_layer(_A0, _B0, d0), _NO_SNAP / _P)
    assert hit is not None
    fixed = _solve(d0, deg, guard=True, min_feature=2.0 * hit[3] * _P)
    assert _err(fixed, ref) <= 2.0 * d0, (d0, _err(fixed, ref))
    assert e0 > 100.0 * _err(fixed, ref), (e0, _err(fixed, ref))


# ==========================================================================
# THE ATTRIBUTION HOLE (the fix audit's open item F) -- RE-PINNED AGAINST
# THE ROUND-2 ARBITER THAT CLOSED IT
# ==========================================================================
def test_a_TRUNCATION_super_unity_is_no_longer_blamed_on_the_sliver():
    """Conjunct (b) reads super-unity as a theorem violation.  On a provably
    passive stack it can equally be ORDINARY UNDER-CONVERGENCE -- a lossy
    substrate at a large angle and a modest degree is enough, no many-slice
    quasi-resonance required.  When such a stack also carries a HARMLESS sliver
    (well above the onset, the answer still on the physical shift) round 1
    refused it and blamed the sliver: 110 of 648 realistic staircase
    configurations, measured on both builds
    (``validation/probe_verify_sliver/v9_falsepos.py``).

    This is the test that pinned that defect.  Its own instruction was
    "re-pin it against the improvement, do not relax it", and round 2's
    arbiter is that improvement, so the SAME fixture now asserts the SAME
    three facts with the verdict inverted: the sliver is harmless here, the
    solve RETURNS, and the warning it returns under says a sliver is present
    and is NOT the cause.  Re-measured 2026-09-11: 0 of 648
    (``validation/probe_pmmstack_sliver_round2/r4_falsepos.py``)."""
    def _st(delta, degree, mf=_NO_SNAP, nl=2):
        st = PMMStack(_P, n_superstrate=2.5, n_substrate=1.5 + 0.05j,
                      degree=degree, min_feature=mf, far_field_orders=31)
        for k in range(nl):
            dd = delta * k / max(nl - 1, 1)
            st.add_layer(_DZ, segments=[(_A0 - dd, _EH),
                                        (_B0 + dd - (_A0 - dd), 12.0),
                                        (1.0 - (_B0 + dd), _EH)])
        st.set_source(_WL, theta=1.2)
        return st

    def _go(st, guard=True):
        was = ps.PMM_SLIVER_GUARD
        ps.PMM_SLIVER_GUARD = guard
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T, _J = st.solve()
            o = np.asarray(o).ravel()
            i = np.argsort(o)
            tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
            return (o[i], np.asarray(R)[1][i], np.asarray(T)[1][i],
                    float(np.max(tot)))
        finally:
            ps.PMM_SLIVER_GUARD = was

    d = 1e-3
    assert ps._stack_provably_passive(_st(d, 6)) is True
    # (i) the sliver is HARMLESS here: the answer tracks the exact delta -> 0
    #     limit to within the fix's own "correct" rule.
    ref, cur = _go(_st(0.0, 6), False), _go(_st(d, 6), False)
    c = np.intersect1d(cur[0], ref[0])
    ia, ib = np.searchsorted(cur[0], c), np.searchsorted(ref[0], c)
    e = float(max(np.abs(cur[1][ia] - ref[1][ib]).max(),
                  np.abs(cur[2][ia] - ref[2][ib]).max()))
    assert e <= 10.0 * d, (e, d)
    assert cur[3] > 1.0 + ps._STACK_SUPERUNITY_BAR, cur[3]
    # (ii) the geometric screen and the theorem BOTH still fire -- so round 1
    #      would refuse -- and the ARBITER is what declines to attribute.
    st = _st(d, 6)
    assert ps._sliver_screen(st) is not None
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o1, R1, T1, _J1 = _st(d, 6).solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    verdict, ev = ps._sliver_arbiter(st, cur[3], R1, T1, None)
    assert verdict == "truncation", (verdict, ev)
    assert ev["snapped_super_unity"] > ps._SLIVER_ATTRIB_CLOSURE, ev
    # (iii) the solve RETURNS, bit for bit the unguarded answer, under a
    #       warning that says the sliver is present and is not the cause.
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o2, R2, T2, _J2 = _st(d, 6).solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    i2 = np.argsort(np.asarray(o2).ravel())
    assert np.array_equal(np.asarray(R2)[1][i2], cur[1])
    assert np.array_equal(np.asarray(T2)[1][i2], cur[2])
    msgs = [str(w.message) for w in rec]
    assert any("energy not conserved" in m for m in msgs), msgs
    assert any("is NOT what moved this answer" in m for m in msgs), msgs
    # (iv) remedy (1) -- the one the message names FIRST, with a number --
    #      leaves the answer where it was, which is why the arbiter refuses to
    #      call it the cure ...
    mf = 2.0 * ps._cross_layer_sliver([L[1] for L in st._layers],
                                      float(st.min_feature) / _P)[3] * _P
    fixed = _go(_st(d, 6, mf=mf), True)
    assert abs(fixed[3] - cur[3]) < 1e-3, (fixed[3], cur[3])
    assert fixed[3] > 1.0 + ps._STACK_SUPERUNITY_BAR, fixed[3]
    # ... while remedy (4), degree, is the one that actually converges it.
    assert _go(_st(d, 12), True)[3] < 1.0 + 1e-5


# ==========================================================================
# THE REMEDY'S BAR IS THE STRUCTURE'S, NOT THE NUMBER 2
# ==========================================================================
def test_the_remedy_lands_on_the_structures_own_continuity_slope():
    """The snap moves each wall by at most ``delta``; what that costs the
    ANSWER is ``dR/dx * delta``, and ``dR/dx`` is a property of the device.
    Measured on TWO fixtures whose slopes differ by 4x: the snapped answer
    lands within the slope this test measures, on both -- while a fixed
    ``2 delta`` bar holds on one and fails on the other."""
    rows = []
    for fx in (None, _Q):
        P = (fx or {}).get("P", _P)
        ref = _solve(0.0, 14, guard=False, fx=fx)
        slope = max(_err(_solve(d, 14, guard=False, fx=fx), ref) / d
                    for d in (3e-3, 1e-3, 3e-4))
        try:
            _solve(3e-5, 14, guard=True, fx=fx)
            pytest.fail("the fixture must be inside the hazard band")
        except ValueError as exc:
            mf = float(str(exc).split("min_feature=")[1].split(" ")[0])
        fixed = _solve(3e-5, 14, guard=True, min_feature=mf, fx=fx)
        e = _err(fixed, ref)
        # the STRUCTURE's bar: the snap cannot cost more than the structure's
        # own sensitivity times the displacement, x2 for the two walls.
        assert e <= 2.0 * slope * 3e-5, (slope, e)
        assert abs(fixed[3] - 1.0) < 1e-5, fixed[3]
        rows.append((slope, e / 3e-5, P))
    slopes = [r[0] for r in rows]
    # the point of the second fixture: the sensitivities are decades apart in
    # ratio terms, so a FIXED bar cannot be scale-free.  MEASURED 2026-09-11:
    # 1.146 (O-11) and 4.442 (the 0.9 um fixture), err/delta 1.153 and 3.515.
    assert max(slopes) / min(slopes) > 2.0, slopes
    assert max(r[1] for r in rows) > 2.0, rows       # the 2*delta bar fails


# ==========================================================================
# THE EXEMPTIONS, AND WHAT THEY COST
# ==========================================================================
def test_a_thin_feature_owned_by_one_layer_is_exempt_and_that_is_not_free():
    """The ownership rule is deliberate ("a 1 nm liner is intentional"), and
    the mechanism does not care who owns the walls.  Both halves measured: the
    screen stays silent at every width, and by 1e-7 of a period the answer is
    catastrophically wrong on a stack the refusal never reaches.

    The reference is EXACT: as the liner width goes to zero the layer becomes
    the plain two-segment layer."""
    def _run(segs, deg=14):
        st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                      min_feature=_NO_SNAP)
        st.add_layer(_DZ, segments=segs)
        st.add_layer(_DZ, segments=segs)
        st.set_source(_WL, theta=_THETA)
        was = ps.PMM_SLIVER_GUARD
        ps.PMM_SLIVER_GUARD = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T, _J = st.solve()
        finally:
            ps.PMM_SLIVER_GUARD = was
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot))

    def _score(segs, deg):
        res = _run(segs, deg)
        r0 = _run([(0.30, _EH), (0.70, _EH)], deg)
        common = np.intersect1d(res[0], r0[0])
        ia, ib = np.searchsorted(res[0], common), np.searchsorted(r0[0], common)
        return float(max(np.abs(res[1][ia] - r0[1][ib]).max(),
                         np.abs(res[2][ia] - r0[2][ib]).max())), res[3]

    ref = _run([(0.30, _EH), (0.70, _EH)])
    assert abs(ref[3] - 1.0) < 1e-9, ref[3]
    bad = []
    for d in (1e-3, 1e-5, 1e-7):
        segs = [(0.30, _EH), (d, _EP), (0.70 - d, _EH)]
        # the screen is silent at EVERY width -- that is the ownership rule
        assert ps._cross_layer_sliver([segs, segs], _NO_SNAP / _P) is None, d
        for deg in (8, 12, 14, 16):
            e, tot = _score(segs, deg)              # and it never refuses
            if d == 1e-7:
                bad.append((deg, e, tot))
            else:
                assert e <= 10.0 * d, (d, deg, e)   # continuity holds up there
    # MEASURED 2026-09-11, both builds: at d = 1e-7 the degree ladder reads
    # errors up to 1.05 in absolute per-order efficiency and R+T from 0.57 to
    # 4.19 -- the pre-fix warn-and-return shape (and at some degrees not even
    # super-unity, so NOTHING fires), on a stack conjunct (a) exempts by
    # construction.  Bars: 0.1 is ~10x below the observed worst error and the
    # super-unity arm only needs ONE degree of the four.
    assert max(b[1] for b in bad) > 0.1, bad
    assert max(abs(b[2] - 1.0) for b in bad) > ps._STACK_SUPERUNITY_BAR, bad
    assert min(b[2] for b in bad) < 1.0, bad        # some degrees are SUB-unity


# ==========================================================================
# THE PER-LAYER WINDOW, AT GRID LEVEL
# ==========================================================================
def test_the_per_layer_window_IS_the_union_only_on_a_short_stack():
    """The caveat the refusal states, checked on the GRID rather than on the
    answer: at ``window_halfwidth = 1`` a 2-layer window is the whole union
    (bit-for-bit), a 3-layer stack's middle window is, and on a 5-layer stack
    no window is -- yet every window still carries the sliver, so the escape is
    from cross-STACK accumulation, not from this defect."""
    from lumenairy.elements.pmm._core import _perlayer_window_grids
    seen = {}
    for nlay in (2, 3, 5):
        segs = [_segs(_A0 - 1e-4 * i, _B0 + 1e-4 * i) for i in range(nlay)]
        uw, _e = _pmm_union_grid(segs, _NO_SNAP / _P)
        grids = _perlayer_window_grids(segs, _NO_SNAP / _P, halfwidth=1)
        eq = [bool(np.array_equal(np.asarray(g[0]), uw)) for g in grids]
        seen[nlay] = eq
        for g in grids:
            assert float(np.min(g[0])) <= 1.01e-4, (nlay, np.min(g[0]))
    assert all(seen[2]), seen[2]
    assert seen[3] == [False, True, False], seen[3]
    assert not any(seen[5]), seen[5]


# ==========================================================================
# THE 2-D STACKS AND THE MORTAR ROUTE
# ==========================================================================
_P2, _WL2 = 1.2, 0.85


def test_the_pure_stacks_shared_grid_cannot_express_a_sliver():
    """Aspect ratio exactly 1 at every lattice size, and the API refuses two
    layers on different lattices -- so no non-trivial union is ever formed."""
    for N in (8, 24, 129):
        cell = np.full((N, N), _EH + 0j)
        cell[N // 4:3 * N // 4, N // 4:3 * N // 4] = _EP
        st = PMM2DStackPure(_P2, n_modes=4, n_orders=1)
        st.add_layer(0.3, eps_cell=cell)
        w = np.full(N, _P2 / N)
        assert float(w.max() / w.min()) == 1.0, N
    cA = np.full((6, 6), _EH + 0j)
    cA[2:4, 2:4] = _EP
    cB = np.full((8, 8), _EH + 0j)
    cB[3:5, 3:5] = _EP
    st = PMM2DStackPure(_P2, n_modes=4, n_orders=1)
    st.add_layer(0.3, eps_cell=cA)
    with pytest.raises(ValueError, match="common"):
        st.add_layer(0.3, eps_cell=cB)


def test_the_hybrid_has_no_union_grid_and_keeps_the_plain_warning():
    """Different lattices per layer are legal on the Fourier-projected stack,
    so a cross-layer cell cannot exist; and its low-order super-unity WARNS
    rather than refusing -- which is why ``stack2d`` passes ``stack=None``."""
    cA = np.full((6, 6), _EH + 0j)
    cA[2:4, 2:4] = _EP
    cB = np.full((8, 8), _EH + 0j)
    cB[3:5, 3:5] = _EP
    sh = PMM2DStackHybrid(_P2, degree=7, n_orders=3)
    sh.add_layer(0.3, eps_cell=cA)
    sh.add_layer(0.3, eps_cell=cB)
    sh.set_source(_WL2, theta=0.2, phi=0.3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _o, R, T, _J = sh.solve()
    tot = np.real(np.asarray(R)).sum(axis=-1) + np.real(np.asarray(T)).sum(axis=-1)
    assert float(np.max(tot)) > 1.0 + ps._STACK_SUPERUNITY_BAR, tot
    msgs = [str(w.message) for w in rec]
    assert any("energy not conserved" in m for m in msgs), msgs
    assert not any("SLIVER" in m for m in msgs), msgs


def test_the_mortar_carries_a_within_layer_sliver_without_a_silent_wrong_answer():
    """The claim the fix audit's S7 makes about the in-flight mortar work,
    tested rather than assumed.  A caller CAN put two of one layer's own walls
    1e-5 of a period apart through ``x_walls`` -- the exact configuration the
    shared 1-D union grid turns catastrophic at 1e-4 -- and on the per-layer
    mortar the answer stays on the exact ``d -> 0`` limit (the stripe vanishes)
    and closes."""
    def _nu(walls, M):
        tile = np.array([[_EH] * 3, [_EP] * 3, [_EH] * 3], dtype=complex)
        st = PMM2DStackPure(_P2, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps_cell=tile,
                     x_walls=[w * _P2 for w in walls],
                     y_walls=[w * _P2 for w in walls])
        st.set_source(_WL2, theta=0.15)
        return st

    def _uni(M):
        st = PMM2DStackPure(_P2, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.30, eps=_EH, grid=1, n_modes=M)
        st.set_source(_WL2, theta=0.15)
        return st

    def _run(st):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = st.solve(jones=False)
        oo = np.asarray(o)
        sel = oo[:, 1] == 0
        i = np.argsort(oo[sel, 0])
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return (np.asarray(R)[1][sel][i], np.asarray(T)[1][sel][i],
                float(np.max(tot)))

    M = 5
    ref = _run(_uni(M))
    a0 = 0.2371
    for d in (1e-3, 1e-5):
        r = _run(_nu([a0, a0 + d], M))
        e = float(max(np.abs(r[0] - ref[0]).max(), np.abs(r[1] - ref[1]).max()))
        # continuity in d, and no theorem violation -- MEASURED 2026-09-11 at
        # err/d = 0.46 (1e-3) and 0.34 (1e-5), |R+T-1| <= 8e-7, both builds.
        assert e <= 5.0 * d, (d, e)
        assert abs(r[2] - 1.0) < 1e-4, (d, r[2])


def test_the_mortars_within_layer_sliver_fails_LOUDLY_when_it_fails():
    """The other side: it is not unconditionally safe.  At 1e-7 of a period the
    per-layer basis raises (``LinAlgError`` today -- an unhelpful message, but
    a refusal, not a silent wrong answer)."""
    tile = np.array([[_EH] * 3, [_EP] * 3, [_EH] * 3], dtype=complex)
    st = PMM2DStackPure(_P2, n_modes=5, n_orders=1, layer_grids="per-layer")
    a0 = 0.2371
    st.add_layer(0.30, eps_cell=tile,
                 x_walls=[a0 * _P2, (a0 + 1e-7) * _P2],
                 y_walls=[a0 * _P2, (a0 + 1e-7) * _P2])
    st.set_source(_WL2, theta=0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception):               # noqa: B017, PT011
            st.solve(jones=False)


# ==========================================================================
# THE M1 DISARM DOES NOT LEAK
# ==========================================================================
def test_the_m1_module_disarm_is_function_scoped_and_restores():
    """``tests/unit/test_m1_conditioning_guard.py`` throws
    ``PMM_SLIVER_GUARD`` off for its whole file.  Two properties matter and
    both are checked here: the switch is a module GLOBAL (so a leak would be
    silent), and this file sees it ARMED -- i.e. whatever ran before restored
    it."""
    assert ps.PMM_SLIVER_GUARD is True, (
        "PMM_SLIVER_GUARD arrived DISARMED -- an earlier test file leaked it")
    # and the refusal is live right now, on this file's own fixture
    with pytest.raises(ValueError, match="NEAR-COINCIDENT-WALL SLIVER"):
        _solve(1e-4, 14, guard=True)
