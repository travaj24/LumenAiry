"""M2 -- the per-layer WINDOW is an honest, tested contract.

Campaign: docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md (N-4, N-6,
T3-1, T3-2).  Evidence: docs/audits/PMM_M2_WINDOW_CONTRACT_2026_08_04.md.

What is pinned here
-------------------
* **N-4** -- one window helper replaces five verbatim copies
  (``stack.py`` x3, ``conical.py``, ``_jax_stack.py``).  Byte-identity is
  proved against a VERBATIM re-implementation of the pre-M2 loop, at
  tolerance 0.0, not by ``array_equal``.
* **N-4 continuity** -- at ``window_halfwidth >= nlay - 1`` every window is
  the whole stack, so the per-layer path must reproduce the SHARED union grid
  BIT-FOR-BIT.  A structural pin between the two grid paths that did not
  exist before.
* **N-6** -- the ``min_feature`` contract, correcting v5.32.0's "inert by
  construction": dormant at the library default, ACTIVE above it, bounded by
  ``min_feature/2``, bounding INTERIOR CROSS-LAYER PAIRS and not the minimum
  cell width -- and it is the accuracy lever on this path exactly as on the
  shared one.
* **T3-1** -- ``window_halfwidth = 2`` moves a converged answer only inside
  the mortar's own non-conforming residual band.
* **T3-2** -- the staircase is stationary in ``n_slice`` at 8/10/12 ONCE the
  geometry is representable, and demonstrably is NOT at the library default.

The audit-class device
----------------------
The exp21 coated-pillar out-coupler lives in a different repository, so the
device here is rebuilt from the parameters printed in
AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28 S4.4/S12: 700 nm pitch,
1310 nm, 2.0 deg sidewall, 310 nm tapered region, 5.00 nm conformal coat.
That reproduces the audit's measured collision exactly -- the per-slice wall
offset is ``(310/ns) tan(2 deg)`` nm, so at ``ns = 2`` a ridge wall lands
``5.4127 - 5.000 = 0.4127 nm`` from a neighbouring slice's coat wall, which
is the audit's 0.41 nm pair.
"""
import contextlib
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import _core as PC
from lumenairy.elements.pmm._core import (
    _perlayer_window_grids,
    _pmm_union_grid,
)

NM = 1e-9
PERIOD = 700 * NM
WL = 1310 * NM
SIDEWALL = np.deg2rad(2.0)
H1 = 310 * NM
COAT = 5.0 * NM
W_TOP = 340 * NM

EPS_CORE = (3.48 + 0j) ** 2
EPS_COAT = (1.76 + 0j) ** 2
EPS_GROOVE = 1.0 + 0j
N_SUP = 1.50
N_SUB = 1.50


# --------------------------------------------------------------- geometry --
def _coated_segments(w_core, coat=COAT):
    """Centred conformally-coated ridge -> (width_fraction, eps) list."""
    w_out = 0.5 * (w_core + 2.0 * coat) / PERIOD
    w_in = 0.5 * w_core / PERIOD
    g = 0.5 - w_out
    c = w_out - w_in
    return [(g, EPS_GROOVE), (c, EPS_COAT), (2 * w_in, EPS_CORE),
            (c, EPS_COAT), (g, EPS_GROOVE)]


def taper_layers(ns, coat=COAT):
    dz = H1 / ns
    out = []
    for k in range(ns):
        zeta = (k + 0.5) / ns
        a = 0.5 * W_TOP - zeta * H1 * np.tan(SIDEWALL)
        out.append((dz, _coated_segments(2.0 * a, coat)))
    return out


def build(ns, degree=8, grids="per-layer", min_feature=None, ffo=11,
          halfwidth=None, coat=COAT):
    kw = {}
    if min_feature is not None:
        kw["min_feature"] = min_feature
    if halfwidth is not None:
        kw["window_halfwidth"] = halfwidth
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                  degree=degree, far_field_orders=ffo, layer_grids=grids, **kw)
    for t, segs in taper_layers(ns, coat):
        st.add_layer(t, segments=segs)
    return st


def solve0(st, theta_deg=8.0):
    """(R_order0 both pols, |R+T-1|, Jones, orders, R)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(WL, theta=np.deg2rad(theta_deg)).solve()
    R, T, J, o = np.asarray(R), np.asarray(T), np.asarray(J), np.asarray(o)
    m0 = int(np.where(o == 0)[0][0])
    close = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
    return (float(np.real(R[0, m0])), float(np.real(R[1, m0])), close, J, o, R)


# --------------------------------------------------------------------------
# THE FORWARD-GROWTH REPAIR, and why three tests in this file are re-pinned
#
# ``docs/audits/FIX_UNION_GRID_2THREAD_2026_08_06.md``.  M2's collapse and the
# T3-4 silent-wrong family were both the SAME defect: an EVANESCENT mode
# carries exactly zero z-power, so the ``flux`` the propagating/evanescent cut
# scores it on is round-off; when a near-zero-width cross-layer cell pushes
# that round-off across the cut, the mode's DIRECTION is taken from the SIGN
# of the round-off, and half the time a GROWING mode enters the forward set.
# ``_core._forward_growth_flip`` now hands exactly those modes back to the
# ``Im(q)`` rule -- the same three-conjunct mask ``_mode_cut_growth`` already
# detected on -- so the forward set can no longer grow.
#
# CONSEQUENCE FOR THIS FILE.  The three tests below pinned "below the
# ``min_feature`` threshold the degree ladder COLLAPSES" as a fail-before.
# That collapse is now CURED at the library default, so those assertions had
# to move -- and their own messages said so ("this test must be re-pinned
# against the fix, not relaxed").  Every one of them is re-pinned the same
# way, and NO bar is weakened:
#
#   * the original collapse assertions are kept VERBATIM, scored with the
#     library's fail-before switch OFF -- the defect is still reproduced;
#   * the CURE is then asserted with the switch on, which is a new claim;
#   * the above-threshold cells are additionally asserted BIT-IDENTICAL with
#     the switch either way (tolerance 0.0) -- a null control that did not
#     exist here before;
#   * and the repaired ladder is scored against an INDEPENDENT reference
#     (RCWA, 141 orders) rather than only against itself, because
#     stationarity alone never proved correctness.
# --------------------------------------------------------------------------

#: The M3 suite's RCWA anchor for this device (141 orders), by ``n_slice`` --
#: the independent oracle that separates "stationary" from "right".
RCWA_R0 = {2: 0.1100920, 6: 0.1111090}

#: The below-threshold cells: ``(ns, min_feature)`` with ``min_feature`` under
#: M2's ``min(off, |coat - off|)`` rule, so the pre-repair library collapses on
#: them.  A FAMILY and not a cell, for the reason the four-name adjudication
#: records: WHICH cell collapses is a per-build, per-thread-count fact.
BELOW_THRESHOLD = ((2, None), (3, 0.5 * NM), (6, 1.5 * NM))

#: The above-threshold (cured) cells.  ``min_feature`` is over the rule's
#: threshold, so no sliver survives and the repair has nothing to repair.
ABOVE_THRESHOLD = ((2, 0.5 * NM), (3, 1.5 * NM), (6, 3.0 * NM))

_LADDER_DEGREES = (6, 8, 10, 12, 14)

#: The ABOVE-threshold pair ``test_min_feature_threshold_rule_predicts_
#: stationarity`` pins ITS null floor on -- one cell each side of the 1.5 nm
#: bar, and a subset of :data:`ABOVE_THRESHOLD` (same ladders, same cache).
_RULE_CURED = ((3, 1.5e-9), (6, 3.0e-9))


def _window_separations(ns):
    """``(off, |coat - off|)`` -- the +/-1 window's two cross-layer separations
    on the coated taper at ``n_slice = ns``, derived from the device rather
    than listed.  ``off = (H1/ns) tan(sidewall)`` is the per-slice lateral
    walk; the conformal coat puts the second wall ``|coat - off|`` away."""
    off = (H1 / ns) * np.tan(SIDEWALL)
    return off, abs(COAT - off)


#: ``n_slice`` values the N-6 rule is scored on, and the MARGIN FACTORS the
#: below / above cells are placed at on that cell's OWN separations.
#: ``x0.5`` of the smaller separation and ``x1.2`` of the LARGER one -- see
#: :func:`test_min_feature_threshold_rule_predicts_stationarity` for why the
#: larger one is the build-free boundary and for the measured envelopes.
_RULE_NS = (3, 6)
_RULE_BELOW_FACTOR, _RULE_ABOVE_FACTOR = 0.5, 1.2

#: The N-6 partition's two clouds, measured 2026-08-16 over 6 (build x BLAS
#: thread cap) environments -- WSL py3.12 numpy 2.5.1 / scipy 1.18.0, py3.12
#: numpy 1.26.4 / scipy 1.11.4, py3.11 numpy 2.4.6 / scipy 1.17.1, each at 1
#: and 4 threads -- and over ns = 2, 3, 6, 8:
#:
#:   below 0.5*min(off,|c-off|):  spread 0.685 .. 3.050   (COLLAPSE)
#:   above 1.2*max(off,|c-off|):  spread 0.00356 .. 0.00586 (STATIONARY)
#:
#: -- a 116x gap between the clouds.  The bars sit inside it with better than
#: a decade to the nearer cloud on each side.
_RULE_COLLAPSE_BAR, _RULE_STATIONARY_BAR = 0.1, 2e-2
_RULE_SEPARATION = 20.0

#: ... and the CURED pair, degrees and device of the SINGLE-REGION uncoated
#: taper, which pins its own.  A different DEVICE, so a different ladder.
_UNCOATED_CURED = ((6, 3.0e-9), (12, 1.5e-9))
_UNCOATED_DEGREES = (8, 10, 12, 14, 16)


@pytest.fixture
def growth_repair_off():
    """The library's fail-before switch for ``_forward_growth_flip``, restored
    on the way out.  ``False`` is the pre-fix selector, bit for bit.

    Both switches are saved and restored: ``PMM_FORWARD_GROWTH_PASSIVE`` (the
    2026-08-08 widening) is toggled by the tests below as well, and a test that
    left it off would silently change the next one's ladder."""
    prev = (PC.PMM_FORWARD_GROWTH_REPAIR, PC.PMM_FORWARD_GROWTH_PASSIVE)
    PC.PMM_FORWARD_GROWTH_REPAIR = False
    try:
        yield
    finally:
        (PC.PMM_FORWARD_GROWTH_REPAIR,
         PC.PMM_FORWARD_GROWTH_PASSIVE) = prev


def spread(v):
    """Peak-to-peak of a degree ladder, relative to its own mean."""
    v = np.asarray(v, dtype=float)
    return float((v.max() - v.min()) / abs(v.mean()))


#: The near-cut FAULT INJECTOR's scale (see :func:`near_cut_injector`), and
#: part of :data:`_LADDER_CACHE`'s key -- a cache that ignored it would serve an
#: un-injected ladder to an injected measurement, which is the same mistake as
#: ignoring the two switches.
_CUT_SCALE = 1.0


@contextlib.contextmanager
def near_cut_injector(scale):
    """``FIX_CI_ROUND2_PMM_2026_08_08`` S6's NEAR-CUT INJECTOR, in-tree.

    ``_core._mass_flux_threshold`` scaled by ``scale`` < 1, which pulls the
    propagating/evanescent cut DOWN under evanescent modes whose ROUND-OFF flux
    sits just below it.  Those modes are then classified PROPAGATING on
    round-off and handed the flux-SIGN direction rule -- which is exactly the CI
    runner's condition, produced on a build whose own round-off does not do it.

    It emulates the RUNNER, not the physics.  Nothing about the geometry, the
    materials, the operator or the eigenvalues changes: only where the cut lands
    relative to a spectrum of round-off fluxes, which is the one thing an
    OpenBLAS kernel is entitled to move.

    NESTS, and the cache key COMPOSES with it (2026-08-12).  ``scaled`` wraps
    whatever ``_mass_flux_threshold`` already is, so two nested injectors put
    the cut at the PRODUCT of their scales -- and :data:`_CUT_SCALE` has to
    carry that product or :data:`_LADDER_CACHE` keys an inner scale against an
    effective one and serves the wrong ladder.  That nesting is not
    hypothetical: :func:`_uncured_below_threshold`'s probe branch walks
    ``_INJECTOR_SCALES`` DOWNWARD inside the NOT-CURED fail-before's own
    upward injector, and on a build where the sibling test also reaches that
    branch un-injected the two would have collided on the same key."""
    global _CUT_SCALE
    orig, prev = PC._mass_flux_threshold, _CUT_SCALE

    def scaled(flux, W2, SVt, SVb, n, xp=np):
        return orig(flux, W2, SVt, SVb, n, xp) * scale

    PC._mass_flux_threshold = scaled
    _CUT_SCALE = prev * float(scale)
    try:
        yield
    finally:
        PC._mass_flux_threshold = orig
        _CUT_SCALE = prev


#: Ladder cache, keyed by ``(ns, min_feature, degrees, repair_flag,
#: passive_flag, injector_scale)``.  The three re-pinned tests below each score
#: the SAME ladders with the switches off and on, so without this the file
#: re-solves every cell four or five times.  BOTH switches are part of the key
#: because they are exactly what the answer depends on -- a cache that ignored
#: either would silently fake the null control.
_LADDER_CACHE = {}


def _coated_stack(ns, mf, degree):
    """The audit-class COATED taper -- the device of the N-6 sections."""
    return build(ns, degree=degree, min_feature=mf)


def _uncoated_stack(ns, mf, degree):
    """The SINGLE-REGION uncoated taper -- M5's device, no conformal coat, so
    the window's only cross-layer separation is the per-slice offset itself.
    Reproduces ``test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper``'s
    own two builders exactly, ``min_feature`` present or not."""
    layers = _uncoated_layers(ns)
    return _mk(layers, degree, 1) if mf is None else _mk_mf(layers, degree, mf)


def _ladder_rec(ns, mf, degrees=_LADDER_DEGREES, mk=None):
    """``(R0 ladder, raw n_grow, n_grow_post, non-passive rows)`` per rung on
    the device ``mk`` builds (default: the coated taper), with the T3-4 census
    armed.

    The three census columns are read off ``_MODE_CUT_CENSUS`` rather than
    re-derived, so what the null control below conditions on is the SHIPPED
    instrument and cannot drift from it:

    * ``n_grow`` -- the RAW DIAGNOSIS, what the bare selector would have done.
      ``_record_mode_cut`` is deliberately called on the pre-repair
      ``prop``/``q`` (``FIX_UNION_GRID_2THREAD_2026_08_06`` S4), so this column
      does NOT move with the switch, which is what makes it a switch-independent
      thing to condition on.  Asserted, not assumed, in
      :func:`_score_null_control`.
    * ``n_grow_post`` -- the RESIDUAL, what the selector the solve ACTUALLY used
      left growing in the forward set.
    * the non-passive row count -- the invariant "a forward mode of a passive
      layer may not grow" only applies where passivity is PROVEN, so a claim
      about the residual has to know this device's rows are.

    Arming the census changes no number: it decides only whether the
    instruments are computed."""
    mk = mk or _coated_stack
    key = (ns, mf, degrees, bool(PC.PMM_FORWARD_GROWTH_REPAIR),
           bool(PC.PMM_FORWARD_GROWTH_PASSIVE), _CUT_SCALE, mk.__name__)
    hit = _LADDER_CACHE.get(key)
    if hit is None:
        r0, raw, post, nonpas = [], [], [], []
        for d in degrees:
            PC._MODE_CUT_CENSUS = []
            try:
                v = solve0(mk(ns, mf, d))[0]
                rows = list(PC._MODE_CUT_CENSUS)
            finally:
                PC._MODE_CUT_CENSUS = None
            r0.append(v)
            raw.append(sum(int(r["n_grow"]) for r in rows))
            post.append(sum(int(r["n_grow_post"]) for r in rows))
            nonpas.append(sum(0 if r["passive"] else 1 for r in rows))
        hit = (np.array(r0, dtype=float), np.array(raw, dtype=int),
               np.array(post, dtype=int), np.array(nonpas, dtype=int))
        for arr in hit:
            arr.flags.writeable = False
        _LADDER_CACHE[key] = hit
    return hit


def coated_ladder(ns, mf, degrees=_LADDER_DEGREES):
    """Order-0 reflectance over the degree ladder on the coated taper."""
    return _ladder_rec(ns, mf, degrees)[0]


def _score_null_control(cells, refs, degrees=_LADDER_DEGREES, mk=None):
    """N-6's NULL CONTROL, CONDITIONED ON THE INSTRUMENT.  Returns its table.

    Shared by all THREE of this file's forward-growth null controls -- section
    (5) of the accuracy-lever test, the threshold-rule test's above-threshold
    pair, and the uncoated taper's cured pair (``mk`` selects the device).

    ``docs/audits/FIX_M2_NULL_CONTROL_2026_08_09.md``.  The claim this replaces
    conditioned on the ``min_feature`` THRESHOLD -- a fact about the GEOMETRY --
    while the repair conditions on the CENSUS, a fact about one runner's
    round-off.  Those are not the same set, and on the ubuntu py3.10 runner they
    came apart: an ABOVE-threshold cell still produced a near-cut growing mode,
    the repair redirected it (correctly -- the invariant applies wherever such a
    mode appears, cured geometry or not), and the test called the move a
    violation.

    So the null is re-stated where it was always true, on the instrument:

    * a rung whose census reads ZERO raw growing modes has NOTHING in the
      repair's mask, so the two selectors are the SAME ARRAY and the answer must
      be BIT-IDENTICAL, tolerance 0.0.  That is the true null, and on a build
      where every cured rung reads zero -- every one measured on this box and in
      WSL -- it is the original assertion on every cell, verbatim;
    * a rung that DOES read one may move, and then owes the two claims that make
      the move right: the shipped forward set no longer grows
      (``n_grow_post == 0``, a mode count, which no bar can be tuned into), and
      the answer ended CLOSER-OR-EQUAL to the reference in ``refs``.

    ``refs`` maps a cell to its reference value (a scalar, a per-rung array, or
    ``None`` where the test has no independent one).  Nothing here is a
    tolerance."""
    table = []
    for cell in cells:
        ns, mf = cell
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        on, raw_on, post_on, nonpas = _ladder_rec(ns, mf, degrees, mk)
        PC.PMM_FORWARD_GROWTH_REPAIR = False
        off, raw_off, _post_off, _np_off = _ladder_rec(ns, mf, degrees, mk)
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        assert list(raw_on) == list(raw_off), (
            f"ns={ns} min_feature={mf}: the RAW census moved with the "
            f"forward-growth switch ({list(raw_off)} -> {list(raw_on)}).  It is "
            f"supposed to be the pre-repair DIAGNOSIS on both settings "
            f"(_record_mode_cut is called with the raw prop/q by design), which "
            f"is the whole reason this control can condition on it")
        assert not int(np.sum(nonpas)), (
            f"ns={ns} min_feature={mf}: {int(np.sum(nonpas))} modal row(s) of "
            f"this lossless taper were NOT recognised PASSIVE, so the invariant "
            f"the moved-rung claims below rest on -- a forward mode of a passive "
            f"layer cannot grow along +z -- does not apply and they prove "
            f"nothing")
        ref = refs.get(cell)
        for i, deg in enumerate(degrees):
            moved = float(abs(float(on[i]) - float(off[i])))
            table.append((ns, mf, deg, int(raw_on[i]), int(post_on[i]),
                          float(off[i]), float(on[i]), moved))
            if not int(raw_on[i]):
                assert moved == 0.0, (
                    f"ns={ns} min_feature={mf} degree={deg}: a cell whose "
                    f"census reads ZERO raw growing modes moved {moved:.4g} "
                    f"when the forward-growth repair was switched.  With "
                    f"nothing in its mask the repair returns the historical "
                    f"selector's array bit for bit, so this is not a "
                    f"round-off question -- something outside the mask moved "
                    f"the answer")
                continue
            assert int(post_on[i]) == 0, (
                f"ns={ns} min_feature={mf} degree={deg}: the census reads "
                f"{int(raw_on[i])} raw growing mode(s) here, so the repair was "
                f"entitled to redirect them -- but the SHIPPED forward set "
                f"still grows {int(post_on[i])}.  A forward mode of a passive "
                f"layer cannot grow along +z at any distance from the cut, so "
                f"a survivor is a SECOND mechanism and must be diagnosed")
            if ref is None:
                continue
            r = float(np.asarray(ref).reshape(-1)[i] if np.ndim(ref) else ref)
            d_on, d_off = abs(float(on[i]) - r), abs(float(off[i]) - r)
            assert d_on <= d_off, (
                f"ns={ns} min_feature={mf} degree={deg}: redirecting "
                f"{int(raw_on[i])} growing mode(s) moved this rung AWAY from "
                f"the reference {r:.7f} ({d_off:.4g} -> {d_on:.4g}).  The "
                f"repair is allowed to move an answer that carries a growing "
                f"forward mode, but only towards the right one")
    return table


#: The degree-ladder SPREAD that separates a COLLAPSED ladder from a
#: stationary one.  THIS FILE'S OWN BAR since M2 and NOT moved by the
#: 2026-08-15 re-statement below
#: (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5, shape S6) -- it is
#: restated, never widened.  Measured on both mounts: collapsed ladders read
#: 1.23 / 1.34 / 1.76 / 2.28 / 2.38 at the shipped cut and never less than
#: 0.93 at any injected one, cured ladders read 0.00356 / 0.00406 / 0.00733,
#: so the bar sits 9.3x under the closest collapse and 13.6x over the loosest
#: cure.
_COLLAPSE_SPREAD = 0.1


def _reach_probe(ns, mf, degrees, mk):
    """``scale -> (spread, raw census)`` for one cell, with the cut scaled by
    ``scale`` RELATIVE TO WHERE IT ALREADY IS.

    ``near_cut_injector`` composes multiplicatively and :data:`_LADDER_CACHE`
    keys on the product, so a caller that already holds an injector (the
    NOT-CURED fail-before does) gets the composed cut and its own cache entry.
    ``scale = 1.0`` is a true no-op that hits the un-injected entry, so the
    FIRST rung of a reach walk costs no solve on any build."""
    def probe(scale):
        with near_cut_injector(scale):
            v, raw, _post, _nonpas = _ladder_rec(ns, mf, degrees, mk)
        return float(spread(v)), [int(x) for x in raw]
    return probe


def _collapse_reach(probe, bar=_COLLAPSE_SPREAD, scales=None):
    """``(hit, rows)`` -- the FIRST rung of the CUT LADDER at which the census
    is ARMED and the degree ladder has COLLAPSED, or ``(None, rows)`` when no
    rung reaches it.  ``rows`` is every rung walked, ``(scale, raw census,
    spread)``.

    NEW 2026-08-15, ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5
    (shape S6): this EXISTENCE claim is what replaces the per-build CONDITIONAL
    main CI refuted.  :func:`_uncured_below_threshold` carries the measured
    census readings on each build and the argument for why a reach over cut
    positions is build-free where a reading at ONE cut position is not.

    THE LADDER IS ``(1.0,) + _INJECTOR_SCALES`` and it starts at 1.0 for a
    reason: rung one is THE CUT AS THIS BUILD FINDS IT, so a build whose own
    round-off arms the collision is scored on its OWN reading and pays no extra
    solve, and the rungs below it are the cut positions a DIFFERENT build's
    round-off could have put a mode across (``FIX_CI_ROUND2_PMM_2026_08_08``
    S6's near-cut injector).  Nothing about the geometry, the materials, the
    operator or the eigenvalues moves along that ladder -- only where the cut
    lands, which is the one thing an OpenBLAS kernel is entitled to move.

    Both conjuncts are required at the SAME rung and neither is a tolerance
    dressed up: ``sum(raw)`` is a mode COUNT off the shipped instrument, and
    the spread bar is :data:`_COLLAPSE_SPREAD`, this file's own since M2.  A
    rung that scatters without an armed census is not this mechanism and does
    not count; nor does an armed census whose ladder never moves."""
    scales = ((1.0,) + _INJECTOR_SCALES) if scales is None else scales
    rows = []
    for scale in scales:
        sp, raw = probe(scale)
        rows.append((float(scale), [int(x) for x in raw], float(sp)))
        if sum(rows[-1][1]) and rows[-1][2] > bar:
            return rows[-1], rows
    return None, rows


def _uncured_below_threshold(ns, mf, layers, degrees, mk, off_nm, probe=None):
    """The threshold rule's NOT-CURED half, stated as a REACH OVER CUT
    POSITIONS.  Returns ``(spread at the cut as currently set, rows walked)``.

    ``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S5, and the third instance of
    ``FIX_M2_NULL_CONTROL_2026_08_09``'s finding.  ``"1.5 nm should NOT cure
    ns = 6"`` was asserted as a fact about the min_feature THRESHOLD -- the
    GEOMETRY -- by measuring the degree-ladder SPREAD.  But the spread only
    collapses when the near-cut collision actually FIRES, and whether a mode
    lands across the cut is a fact about one build's round-off.  The two came
    apart on the ubuntu py3.12 shard of main, which read the un-snapped cell at
    0.00406 -- bit-for-bit the value this file already pins for the same cell
    with the forward-growth REPAIR ON, i.e. the runner's pre-repair answer is
    the shipped answer, because its round-off never produced the growing mode
    the repair exists to redirect.  Nothing was silently wrong there; the
    premise was.

    **2026-08-15 -- THE SECOND ADJUDICATION, AND IT RETIRES THE CONDITIONAL.**
    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5 (shape S6).  The
    2026-08-12 form above partitioned on the census AT THE SHIPPED CUT and then
    asserted the CONDITIONAL *armed => collapsed*.  main CI, new pythons,
    refuted it on this very cell::

        ns=6 min_feature=1.5nm: census reads [0, 0, 0, 0, 2] raw growing
        mode(s) over degrees (8..16) ... but it spreads only 0.004055

    and the conditional was never a theorem.  ``_core._mode_cut_growth`` counts
    a mode that is ``prop``, within ``_MODE_CUT_MARGIN_WARN`` of the cut and
    growing; NOTHING in that count says how strongly the mode is EXCITED or how
    far it grows across a 51.7 nm slice, so an armed rung is free to be
    indistinguishable from a cured one -- which is what two modes on the LAST
    rung of the runner's ladder were.  ``n_grow`` is NECESSARY for the
    collapse, never SUFFICIENT for it.

    THE CENSUS IS A PER-BUILD READING THREE TIMES OVER -- mount, thread count
    and wheel.  Measured on this cell, repair OFF, degrees 8..16, over the cut
    ladder (2026-08-15, one BLAS thread; W = Windows py3.14.6 / numpy 2.4.4,
    M = WSL py3.12.3 / numpy 2.5.1, which is the numpy 2.5-era wheel the
    failing ubuntu job runs):

        cut     [W] raw census      spread    [M] raw census      spread
        x1e-4   [40,35,47,54,55]    1.33478   [29,32,42,42,55]    1.42007
        x1e-2   [10,26,28,43,67]    0.93247   [10,25,24,59,72]    1.35714
        x1/3    [0,1,3,4,6]         1.78619   [0,1,0,5,5]         2.38098
        x1      [0,0,1,1,1]         1.33841   [0,0,0,3,0]         2.38098
        ------- and upward, which is the runner's direction ----------------
        x3      [0,0,0,0,0]         0.004055  [0,0,0,1,0]         2.38098
        x5/x10  [0,0,0,0,0]         0.004055  [0,0,0,0,0]         0.004055

    -- the two mounts do not agree on WHICH rung arms, on HOW MANY modes it
    carries, or on where the collision disarms (x3 on W, x5 on M), and M does
    not agree with what the 2026-08-12 table recorded on the same mount at the
    same width ([0,0,1,1,1] spread 1.33841 there, on a numpy 2.4 wheel).  No
    reading here is universal.  What IS universal on both mounts, at every
    width and every wheel, is that SOMEWHERE on the cut ladder the census arms
    and the ladder collapses.

    So the half is re-stated on the two things that are true build by build:

    * the GEOMETRY premise, asserted rather than commented -- ``min_feature``
      below the per-slice offset must leave the window UNSNAPPED, read off the
      shipped snap accounting (``_snap_report``), which is deterministic and
      carries no BLAS dependence at all;
    * the MECHANISM as an EXISTENCE claim, which names no census reading:
      somewhere on the cut ladder that STARTS at this build's own cut and walks
      down ``_INJECTOR_SCALES`` there is a rung whose census is ARMED and whose
      degree ladder has COLLAPSED (:func:`_collapse_reach`).  A build whose
      round-off arms the collision -- both mounts, every width measured -- meets
      it on rung one, where the assertion is the 2026-08-12 one verbatim and
      costs no extra solve; a build whose round-off does not, or does so
      harmlessly as the runner's did, escalates until the cut sits where such a
      build's round-off would have put it.  If NO rung reaches it the test
      FAILS with its table: an un-snapped window that admits no collapse at any
      cut position REFUTES the rule, and that is the claim this half exists to
      make.

    ``probe`` overrides the measurement for the engineered fail-before
    (:func:`test_the_NOT_CURED_reach_is_scored_on_a_LADDER_of_cut_positions`),
    which drives this same decision against the census tables the two mounts
    and the ubuntu runner ACTUALLY produced -- including the states no build
    here manifests.  The default measures the device.
    """
    merged, _disp = _snap_report([s for _t, s in layers], mf / PERIOD,
                                 halfwidth=1)
    assert not merged, (
        f"ns={ns}: min_feature={mf * 1e9:.1f} nm is supposed to be BELOW this "
        f"cell's only cross-layer separation (off={off_nm:.3f} nm), so the "
        f"window must come back UNSNAPPED -- but the shipped accounting "
        f"merged {merged} pair(s).  The cell is mislabelled and every claim "
        f"below is about a different geometry than the one named")
    probe = _reach_probe(ns, mf, degrees, mk) if probe is None else probe
    hit, rows = _collapse_reach(probe)
    assert hit is not None, (
        f"ns={ns} min_feature={mf * 1e9:.1f} nm (off={off_nm:.3f} nm): NO rung "
        f"of the cut ladder "
        f"{tuple(float(f'{s:.4g}') for s in (1.0,) + _INJECTOR_SCALES)} "
        f"both ARMS the census and COLLAPSES the degree ladder over "
        f"{tuple(degrees)} -- walked "
        f"{[(float(f'{s:.4g}'), r, float(f'{p:.4g}')) for s, r, p in rows]}.  "
        f"An un-snapped window is supposed to ADMIT the collision at SOME cut "
        f"position; this device admits it at none, so either the mode-cut "
        f"instrument is dead (the same reading at every scale) or the "
        f"threshold rule itself is refuted here.  Widen the ladder and "
        f"adjudicate; do not delete this")
    if hit[0] != 1.0:
        print(f"\nM2 NOT-CURED reach [{mk.__name__} ns={ns} "
              f"min_feature={mf * 1e9:.1f} nm]: this build's own cut does NOT "
              f"collapse this cell, so the collision was reached at cut "
              f"x{hit[0]:.4g} (census {hit[1]}, spread {hit[2]:.4g}) -- walked "
              + ", ".join(f"x{s:.4g} raw={r} spread={p:.5g}"
                          for s, r, p in rows))
    return rows[0][2], rows


def _snap_report(layer_segments, mf_frac, halfwidth=1):
    """(merged pair count, max wall displacement as a period fraction) over
    every window -- parsed from the routine's own warning, so the test scores
    the SHIPPED accounting, not a re-derivation of it."""
    nlay = len(layer_segments)
    merged, disp = 0, 0.0
    for i in range(nlay):
        js = [j for j in range(i - halfwidth, i + halfwidth + 1)
              if 0 <= j < nlay]
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _pmm_union_grid([layer_segments[j] for j in js], mf_frac)
        for w in wl:
            txt = str(w.message)
            if "snapped" in txt:
                merged += int(txt.split("snapped ")[1].split(" pair")[0])
                disp = max(disp, float(
                    txt.split("displacement ")[1].split(" of")[0]))
    return merged, disp


# ======================================================================== N-4
def _window_grids_pre_m2(layer_segments, min_feature_frac):
    """VERBATIM re-implementation of the pre-M2 window loop, as it stood at
    stack.py:1654 / 1838 / 2923, conical.py:239 and _jax_stack.py:482 (commit
    d30f1ca).  The helper must reproduce this bit-for-bit at halfwidth 1."""
    nlay = len(layer_segments)
    grid_of = []
    for i in range(nlay):
        js = [j for j in (i - 1, i, i + 1) if 0 <= j < nlay]
        uw_i, rows_i = _pmm_union_grid([layer_segments[j] for j in js],
                                       min_feature_frac)
        grid_of.append((np.asarray(uw_i, dtype=float), rows_i[js.index(i)]))
    return grid_of


def test_n4_helper_is_byte_identical_to_the_five_verbatim_copies():
    # N-4 fail-before: the helper replaced five copies of this loop.  Compare
    # the GRIDS themselves (what all five sites consumed), at tolerance 0.0,
    # over the geometries the five sites see -- tapered staircases at several
    # n_slice, both min_feature regimes, and a non-taper control.
    cases = [taper_layers(ns) for ns in (1, 2, 3, 6, 9)]
    cases.append([(200e-9, [(0.35, 4.0 + 0j), (0.65, 1.0 + 0j)]),
                  (150e-9, [(0.50, 2.25 + 0j), (0.50, 1.0 + 0j)]),
                  (120e-9, [(0.35, 6.25 + 0.1j), (0.65, 1.0 + 0j)])])
    for layers in cases:
        segs = [s for _t, s in layers]
        for mf in (None, 1e-5, 1.5e-9 / PERIOD, 3.0e-9 / PERIOD):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = _window_grids_pre_m2(segs, mf)
                got = _perlayer_window_grids(segs, mf, 1)
            assert len(got) == len(ref)
            for (wa, ra), (wb, rb) in zip(ref, got):
                assert float(np.max(np.abs(wa - wb))) == 0.0
                assert len(ra) == len(rb)
                for ea, eb in zip(ra, rb):
                    assert float(np.max(np.abs(np.asarray(ea)
                                               - np.asarray(eb)))) == 0.0


def test_n4_default_solve_is_bit_identical_through_every_dispatch():
    # The helper is reached by solve() (classical), solve() at phi != 0
    # (conical) and solve_vs_wavelength (the sweep).  Each must still agree
    # BIT-FOR-BIT with the shared path on a CONFORMING stack, which is the
    # property the five copies were pinned on.
    lay = [(220e-9, [(0.30, 4.0 + 0j), (0.70, 1.0 + 0j)]),
           (180e-9, [(0.50, 2.25 + 0j), (0.50, 1.0 + 0j)])]

    def mk(grids):
        st = PMMStack(PERIOD, n_substrate=1.5, n_superstrate=1.0, degree=6,
                      far_field_orders=7, layer_grids=grids)
        for t, s in lay:
            st.add_layer(t, segments=s)
        return st

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, Rs, _T, Js = mk("shared").set_source(WL, theta=0.14).solve()
        _o, Rp, _T, Jp = mk("per-layer").set_source(WL, theta=0.14).solve()
        assert float(np.max(np.abs(np.asarray(Js) - np.asarray(Jp)))) == 0.0
        assert float(np.max(np.abs(np.asarray(Rs) - np.asarray(Rp)))) == 0.0
        # conical dispatch
        _o, Rs, _T, Js = mk("shared").set_source(
            WL, theta=0.14, phi=0.7).solve()
        _o, Rp, _T, Jp = mk("per-layer").set_source(
            WL, theta=0.14, phi=0.7).solve()
        assert float(np.max(np.abs(np.asarray(Js) - np.asarray(Jp)))) == 0.0
        # sweep dispatch
        wls = [1200e-9, 1310e-9]
        _o, Rs, _T, Js = mk("shared").solve_vs_wavelength(
            wls, angle=0.14, jones=True, max_workers=1)
        _o, Rp, _T, Jp = mk("per-layer").solve_vs_wavelength(
            wls, angle=0.14, jones=True, max_workers=1)
    assert float(np.max(np.abs(np.asarray(Js) - np.asarray(Jp)))) < 1e-12


def test_window_halfwidth_covering_the_stack_reproduces_shared_bit_exact():
    # CONTINUITY PIN (new in M2): at halfwidth >= nlay - 1 every window is the
    # whole stack, so the two grid paths are the same discretisation and the
    # per-layer answer must be BIT-IDENTICAL to the shared one -- mortar
    # bypassed everywhere, tolerance-at-0.0.  This is what makes
    # window_halfwidth a knob that DEGENERATES to a known reference rather
    # than a free parameter.
    for ns in (2, 3, 4):
        for deg in (6, 8):
            a = solve0(build(ns, degree=deg, halfwidth=max(1, ns - 1)))
            b = solve0(build(ns, degree=deg, grids="shared"))
            assert float(np.max(np.abs(a[3] - b[3]))) == 0.0
            assert float(np.max(np.abs(a[5] - b[5]))) == 0.0


def test_window_halfwidth_is_validated():
    for bad in (0, -1, 1.5, "2", None):
        with pytest.raises(ValueError, match="window_halfwidth"):
            PMMStack(PERIOD, layer_grids="per-layer", window_halfwidth=bad)
    # meaningless on the shared path -- there is no window
    with pytest.raises(ValueError, match="window_halfwidth"):
        PMMStack(PERIOD, layer_grids="shared", window_halfwidth=2)
    # the helper guards its own contract too
    with pytest.raises(ValueError, match="halfwidth"):
        _perlayer_window_grids([[(1.0, 1.0 + 0j)]], None, 0)


# ======================================================================== N-6
def test_min_feature_is_dormant_at_the_library_default():
    # N-6, half one of the contract: at the library default (period * 1e-5)
    # NOTHING is snapped -- 0 pairs, 0.0 displacement -- on the per-layer
    # windows AND on the shared union, at every n_slice.  This is the half of
    # the v5.32.0 claim that IS true, and it is what made the wrong half
    # (below) invisible.
    for ns in (2, 3, 6, 8, 10, 12):
        segs = [s for _t, s in taper_layers(ns)]
        mf_frac = (PERIOD * 1e-5) / PERIOD
        merged, disp = _snap_report(segs, mf_frac, halfwidth=1)
        assert merged == 0, f"ns={ns}: default min_feature snapped {merged}"
        assert disp == 0.0
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _pmm_union_grid(segs, mf_frac)
        assert not [w for w in wl if "snapped" in str(w.message)]
        # ... and PARSE-INDEPENDENTLY: dormant means the grids are the ones
        # that no-snapping-at-all produces.  (_snap_report reads the routine's
        # warning text, so on its own it would report "0 merged" if that text
        # ever changed; this check cannot.)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _perlayer_window_grids(segs, mf_frac, 1)
            b = _perlayer_window_grids(segs, None, 1)
        for (wa, _ra), (wb, _rb) in zip(a, b):
            assert wa.shape == wb.shape
            assert float(np.max(np.abs(wa - wb))) == 0.0


def test_min_feature_is_LIVE_on_the_per_layer_window():
    # N-6, the corrected half.  v5.32.0 said min_feature is "inert by
    # construction (there is no global union to snap)" on the per-layer path.
    # A window IS a union, and a tapered staircase's TIGHTEST collisions are
    # between ADJACENT slices -- exactly what a window contains.  At the
    # shared path's own recommended 1.5 nm the windows snap real walls.
    #
    # This test FAILS on the claim, not on the code: if a future change made
    # the per-layer path genuinely inert, this test must be deleted together
    # with the contract it pins.
    any_merged = 0
    for ns in (3, 6, 8, 10, 12):
        segs = [s for _t, s in taper_layers(ns)]
        merged, disp = _snap_report(segs, 1.5e-9 / PERIOD, halfwidth=1)
        any_merged += merged
        if merged:
            assert disp > 0.0
    assert any_merged > 0, (
        "min_feature = 1.5 nm merged nothing on any per-layer window -- the "
        "N-6 contract (the snap is LIVE here) no longer holds")
    # and the grids it produces genuinely differ from the default ones
    segs = [s for _t, s in taper_layers(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _perlayer_window_grids(segs, 1e-5, 1)
        b = _perlayer_window_grids(segs, 1.5e-9 / PERIOD, 1)
    assert any(len(wa) != len(wb) for (wa, _), (wb, _) in zip(a, b))


def test_min_feature_displacement_is_bounded_by_half():
    # N-6, the BOUND: a pair merges only when closer than min_feature and each
    # wall moves to their midpoint, so no wall moves further than
    # min_feature / 2.  Measured on the routine's own reported displacement.
    for mf_nm in (0.5, 1.5, 3.0, 6.0):
        mf = mf_nm * NM
        for ns in (2, 3, 6, 8, 12):
            segs = [s for _t, s in taper_layers(ns)]
            _m, disp = _snap_report(segs, mf / PERIOD, halfwidth=1)
            assert disp * PERIOD <= 0.5 * mf + 1e-18, (
                f"mf={mf_nm} nm ns={ns}: wall moved {disp * PERIOD / NM:.4g} "
                f"nm > mf/2")


def test_min_feature_bounds_pairs_not_the_minimum_cell_width():
    # N-6, the SCOPE of the bound -- the part a user will otherwise assume
    # wrongly.  min_feature bounds INTERIOR CROSS-LAYER pair separations; it
    # is not a floor on the cell width, because (a) the period boundary is
    # never dropped and (b) a close pair owned by ONE layer is that layer's
    # own thin feature and is never thinned.  Here the 5 nm coat is a
    # single-layer feature, so a 6 nm min_feature must NOT thin it.
    segs = [s for _t, s in taper_layers(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grids = _perlayer_window_grids(segs, 6.0e-9 / PERIOD, 1)
    smallest = min(float(np.min(w)) for w, _r in grids) * PERIOD
    assert smallest < 6.0 * NM, (
        "min_feature is not a floor on the cell width -- if this now holds, "
        "the contract changed and the docstring must change with it")
    # the single-layer 5 nm coat is never thinned: a cell of that width
    # survives a 6 nm min_feature on every window.
    for w, _r in grids:
        widths_nm = np.asarray(w) * PERIOD / NM
        assert np.any(np.abs(widths_nm - 5.0) < 0.5), (
            f"the 5 nm single-layer coat was thinned: {widths_nm}")


def test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too(
        growth_repair_off):
    """N-6, THE DECISIVE MEASUREMENT -- re-pinned against the fix.

    On the audit-class taper at ns = 2 the per-layer and shared grids are
    IDENTICAL (a 2-layer window is the full union), so this is a statement
    about the discretisation, not about the mortar.  At the library default
    the 0.4127 nm adjacent-slice collision survives, and BEFORE the
    forward-growth repair the degree ladder was not stationary: it held
    ~0.1106 to degree 10 and then jumped to 0.0617 / 0.6234, while
    ``|R+T-1|`` stayed ~1e-7 so ENERGY CLOSURE COULD NOT SEE IT (only order 0
    propagates here, which makes R + T = 1 nearly tautological -- the standing
    caution).

    **2026-08-06 -- THE COLLAPSE WAS THREAD-COUNT-DEPENDENT, AND IS NOW
    FIXED.**  This test was referred as a tenth name: run standalone at
    ``OPENBLAS_NUM_THREADS=2`` the ``spread(broken) > 0.5`` assertion read
    ``3.615e-03``.  That is not a tolerance miss -- it is the same coin flip
    the campaign hunts, one level down.  Holding code and geometry fixed and
    varying ONLY the pool [M, Windows + WSL], with the repair OFF::

        cell                       1 thread   2 threads
        ns=2 default (this one)    2.7605     0.0036153   <- migrated
        ns=3 min_feature 0.5 nm    2.6345     2.8101
        ns=6 min_feature 1.5 nm    2.2490     1.9665

    WHICH below-threshold cell collapses is a per-build fact; THAT the family
    contains one is not.  So the fail-before is scored over the FAMILY.

    What is asserted, all of it measured on both mounts at 1 and 2 threads:

      (1) with the switch OFF the defect still reproduces somewhere in the
          below-threshold family (the original bar, 0.5, verbatim);
      (2) conservation is blind to it -- the reason nothing caught it;
      (3) with the switch ON every one of those ladders is stationary;
      (4) and the repaired ladder is RIGHT, not merely stationary: it agrees
          with the CURED ladder and with the independent RCWA reference, which
          stationarity alone never proved;
      (5) the null floor on the above-threshold cells, conditioned on the
          CENSUS rather than on the min_feature threshold (2026-08-09): a cured
          rung the census reads ZERO raw growing modes on is BIT-IDENTICAL
          either way -- the proof the repair is not a global re-tune -- while
          one that does carry a growing mode may move, and then must end with
          none growing and no further from the RCWA anchor.  See the section;
      (6) 2026-08-08 -- and the repair's REACH is a fail-before of its own:
          the 2026-08-06 mask still leaves this device collapsing two rungs
          further up the SAME ladder, which is the mechanism that failed
          this test on CI.  See the section itself.
    """
    # ---- (1) THE DEFECT, on the switch, over the family ------------------
    off = {cell: spread(coated_ladder(*cell)) for cell in BELOW_THRESHOLD}
    collapsed = {c: s for c, s in off.items() if s > 0.5}
    assert collapsed, (
        "no below-threshold ladder COLLAPSES on this build with the "
        "forward-growth repair switched OFF.  The M2 fail-before is supposed "
        "to reproduce with the switch off on every build -- if the pre-repair "
        "defect has stopped manifesting entirely, re-pin this against "
        f"whatever changed rather than relaxing it.  spreads: "
        f"{ {k: round(v, 5) for k, v in off.items()} }")
    # (2) ... and it is energy-clean while wrong, which is why nothing caught
    #     it.  Scored on a cell that ACTUALLY collapsed on this build.
    ns_bad, mf_bad = max(collapsed, key=collapsed.get)
    assert max(solve0(build(ns_bad, degree=d, min_feature=mf_bad))[2]
               for d in (12, 14)) < 1e-5

    # ---- (3) THE FIX ------------------------------------------------------
    PC.PMM_FORWARD_GROWTH_REPAIR = True          # the fixture owns the restore
    on = {cell: spread(coated_ladder(*cell)) for cell in BELOW_THRESHOLD}
    for cell, s in on.items():
        assert s < 1e-2, (                       # measured worst 4.24e-03
            f"ns={cell[0]} min_feature={cell[1]}: the ladder still spreads "
            f"{s:.4g} with the forward-growth repair on (it spread "
            f"{off[cell]:.4g} with it off) -- the repair has stopped curing "
            f"the collapse this test exists to pin")
    # (4) RIGHT, not merely stationary: against the cured ladder on the same
    #     device AND against the RCWA anchor the M3 suite carries.  Measured
    #     1.1e-03 / 7.1e-03 against the cure and 8.2e-03 / 8.8e-03 against
    #     RCWA, at 1 and 2 threads on both mounts; with the repair OFF the
    #     same numbers are 4.6 and 5.0 -- three decades apart, so the 5 %
    #     bar (M3's own partition) is not a calibration.
    # (``cure_bar`` is each cell's OWN pre-existing bar: 5e-3 is this test's
    # historical ns=2 one, 2e-2 the sibling's ns=6 one.  Neither is relaxed;
    # they are simply not interchangeable -- the ns=6 cured ladder has always
    # spread 5.9e-03, which is over the ns=2 bar and under its own.)
    for ns, mf_cure, cure_bar in ((2, 0.5 * NM, 5e-3), (6, 3.0 * NM, 2e-2)):
        uncured, cured = coated_ladder(ns, None), coated_ladder(ns, mf_cure)
        rel_cure = float(np.max(np.abs(uncured / cured - 1.0)))
        rel_rcwa = float(np.max(np.abs(uncured / RCWA_R0[ns] - 1.0)))
        assert rel_cure < 0.05, (
            f"ns={ns}: the repaired default ladder disagrees with the CURED "
            f"ladder by {rel_cure:.4g} -- it is stationary on the wrong "
            f"answer, which is the failure mode stationarity cannot see")
        assert rel_rcwa < 0.05, (
            f"ns={ns}: the repaired default ladder reads {uncured[-1]:.6f} "
            f"against the RCWA reference {RCWA_R0[ns]:.6f} "
            f"({rel_rcwa:.4g} out) -- the independent oracle disagrees")
        # the cured ladder's OWN stationarity, each on its own historical bar
        assert spread(cured) < cure_bar
        assert all(abs(v - cured[-1]) < 1e-3 for v in cured)

    # ---- (5) THE NULL FLOOR, ON THE INSTRUMENT (2026-08-09) ---------------
    # WHY THIS SECTION MOVED (FIX_M2_NULL_CONTROL_2026_08_09).  It read "an
    # ABOVE-threshold cell moved when the forward-growth repair was switched"
    # on the ubuntu py3.10 shard of main -- rung 4 of the ns=2 / 0.5 nm ladder
    # moving 0.04886 -- while the IDENTICAL tree was green on branch CI and on
    # both mounts at every thread count.  The axis was the runner's OpenBLAS
    # kernel, and the mismatch was in the control's own premise: it
    # conditioned on the min_feature THRESHOLD, which is a fact about the
    # GEOMETRY, while the repair conditions on the CENSUS -- near-cut growing
    # modes -- which is a fact about one build's round-off.  An above-threshold
    # cell whose round-off flux happens to cross the cut still carries one, and
    # redirecting it is CORRECT: the invariant holds wherever such a mode
    # appears, cured geometry or not.
    #
    # So the null is re-stated where it was always true, on the instrument, and
    # nothing is relaxed -- on a build whose cured ladder reads zero raw
    # growing modes everywhere (every rung measured here and in WSL) this is
    # the ORIGINAL bit-identity assertion, on every cell.  The reference for
    # the moved half is the RCWA anchor this test already carries.  The
    # fail-before is
    # ``test_the_null_control_conditions_on_the_census_not_the_threshold``,
    # which drives a cured cell into carrying a growing mode with the near-cut
    # injector and reproduces the CI failure's magnitude to four figures.
    _score_null_control(
        ABOVE_THRESHOLD,
        {(ns, mf): RCWA_R0.get(ns) for ns, mf in ABOVE_THRESHOLD})

    # ---- (6) THE REACH, 2026-08-08 (FIX_CI_ROUND2_PMM_2026_08_08) ---------
    # WHY THIS SECTION EXISTS.  (3) above failed on ubuntu CI -- "the ladder
    # still spreads 0.4888 / 2.761 with the forward-growth repair on" -- while
    # passing on both mounts at all three thread counts.  The 2026-08-06 mask
    # additionally required the growing mode to sit inside a DECADE of the
    # classification cut, and that conjunct was there for gain media, not for
    # the physics; on CI's round-off the survivors sat further out and were
    # left growing.
    #
    # WHICH RUNG CARRIES A SURVIVOR IS A PER-POOL FACT, and this section was
    # written asserting it was not.  Measured on THIS box, ns = 2, library
    # default, degrees (10, 12, 14, 18, 20), passivity widening OFF:
    #
    #     OPENBLAS_NUM_THREADS   spread     survivors
    #     1                      3.75       degrees 18 (2 modes) and 20 (4)
    #     2                      2.15       degree 20 only
    #     24                     5.047e-04  NONE -- the 2026-08-06 mask
    #                                       already covers this pool's
    #
    # -- the same migration the four-name adjudication measured, one level
    # down.  So the fail-before half is ADJUDICATED (printed when the pool
    # does not carry it) and the CURE half is asserted unconditionally.  The
    # guaranteed fail-before for the widening lives in
    # ``test_pmm_m3_efficiency.py::test_t34_guard_fires_on_every_silent_wrong_
    # cell_of_this_build``, which scans BOTH n_slice families and widens its
    # degree set until it finds one -- and which passed at every thread count
    # here, including 24, where this ladder alone carries nothing.
    _REACH = (10, 12, 14, 18, 20)
    PC.PMM_FORWARD_GROWTH_REPAIR = True
    PC.PMM_FORWARD_GROWTH_PASSIVE = False
    try:
        s_narrow = spread(coated_ladder(2, None, _REACH))
    finally:
        PC.PMM_FORWARD_GROWTH_PASSIVE = True
    s_wide = spread(coated_ladder(2, None, _REACH))
    if s_narrow > 0.5:
        print(f"\nM2 reach: the 2026-08-06 mask leaves the ns=2 ladder over "
              f"degrees {_REACH} COLLAPSED on this pool (spread "
              f"{s_narrow:.4g}) and the widening cures it to {s_wide:.4g} -- "
              f"the CI-B mechanism, reproduced on this build.")
    else:
        print(f"\nM2 reach: the 2026-08-06 mask leaves the ns=2 ladder over "
              f"degrees {_REACH} stationary on this pool (spread "
              f"{s_narrow:.4g}), i.e. no beyond-decade survivor lands here -- "
              f"the widening is inert on this cell and its fail-before is "
              f"carried by the M3 wrong-cell scan instead.")
    # the CURE side is assertable on every pool, survivor or not: the ladder
    # the library actually SHIPS must be stationary AND right.  Where the pool
    # carried a survivor this is the cure (measured 3.75 -> 3.73e-03); where it
    # did not it is a plain stationarity claim on two extra rungs.
    assert s_wide < 1e-2, (
        f"the shipped ns=2 ladder spreads {s_wide:.4g} over degrees {_REACH} "
        f"(the 2026-08-06 mask spread {s_narrow:.4g} on the same rungs) -- a "
        f"forward mode of a passive layer cannot grow along +z at ANY "
        f"distance from the cut, so a surviving collapse is a SECOND "
        f"mechanism and must be diagnosed, not tolerated")
    rel_reach = float(np.max(np.abs(coated_ladder(2, None, _REACH)
                                    / RCWA_R0[2] - 1.0)))
    assert rel_reach < 0.05, (
        f"the widened ns=2 ladder over degrees {_REACH} reads {rel_reach:.4g} "
        f"from the RCWA reference -- stationary on the wrong answer")


#: Near-cut injector scales tried, in order, by the three fail-befores below.
#: 1/3 is ``FIX_CI_ROUND2_PMM_2026_08_08`` S6's own scale and is the one that
#: carries the CI condition on the COATED device in all five (mount x
#: thread-count) cells measured.  The two decades below it are there because
#: WHICH rung a cut position lands on migrates with the pool, and because the
#: UNCOATED taper is measurably harder to drive into the condition -- it needs
#: 1e-4, i.e. its spectrum has no round-off-flux mode within four decades of
#: where the cut sits.  That difference is a finding, not a calibration: the
#: single-region device has no coat/offset collision to inject them.
_INJECTOR_SCALES = (1.0 / 3.0, 1.0e-2, 1.0e-4)

#: The near-cut injector scales the NOT-CURED half's fail-before walks UPWARD,
#: and the derivation of where it stops.  ``_INJECTOR_SCALES`` scales the cut
#: DOWN to ARM a build that has no mode across it; this ladder scales it UP to
#: DISARM a build whose own round-off does, which is the condition the ubuntu
#: runners are in and this box is not.
#:
#: WHERE IT LANDS IS READ OFF THE INSTRUMENT, not searched for.
#: ``_core._mode_cut_growth`` counts a mode as growing only while it is
#: ``prop`` AND within ``_MODE_CUT_MARGIN_WARN`` of the cut, so multiplying the
#: cut by that constant provably puts EVERY mode the census is currently
#: reporting back under it.  ``W = _MODE_CUT_MARGIN_WARN`` is therefore the
#: guaranteed rung, and it is on this ladder by construction (asserted in
#: :func:`_first_disarming_scale`, so it moves if the library's constant does).
#: The remaining rungs are its geometric continuation, one per GENERATION of
#: modes a raised cut can pull INTO the warn window from above -- a mode
#: further out than ``W`` is not counted now and can be once the cut moves --
#: and the ladder is walked rather than jumped so the cheapest sufficient rung
#: is the one paid for.
#:
#: It STARTS at 1.0 -- the shipped cut, a true no-op that shares
#: :data:`_LADDER_CACHE`'s un-injected entry -- so the search has to escalate
#: on any build that is armed, which is the mechanism exercised locally.  3.0
#: sits ahead of ``W`` because at exactly that scale the reconditioned helper's
#: downward probe composes back to ``3 * (1/3) = 1.0``, i.e. onto the
#: un-injected ladder already in the cache: it is the one rung that costs
#: nothing to verify, and it is what makes this fail-before CHEAPER than the
#: frozen-x3 version it replaces (0.96 s against 1.92 s, M, one BLAS thread).
#:
#: The CEILING is derived twice over, and the two derivations meet exactly:
#:
#: * ``W^4 = 1e4 = 1 / min(_INJECTOR_SCALES)``.  The reconditioned helper
#:   probes BACK DOWN through ``_INJECTOR_SCALES`` to prove the collision is
#:   still REACHABLE, and those two injectors compose multiplicatively, so a
#:   disarming scale above 1e4 would put the shipped cut outside the probe's
#:   own reach and the mechanism claim could no longer be made.  Asserted, not
#:   commented -- four escalations of ``W`` is also four generations of
#:   pulled-in modes, which is far more than the mechanism admits;
#: * ``1e9 = 1 / 1e-9``, from ``_core._mass_flux_threshold``'s own
#:   ``max(1e-9 max|flux|, round-off floor)``: at that scale the cut sits AT
#:   the strongest mode's flux and NOTHING is classified propagating, so the
#:   injector has stopped emulating a build's round-off and started deleting
#:   the spectrum.  Measured exactly there -- ``n_prop`` reads 0 on every rung
#:   at x1e9 and nowhere below it.  1e4 is five decades inside it.
#:
#: Both are far above what the mechanism needs: ``_mode_cut_growth``'s own
#: calibration records every pathological mode sitting at 1.00-3.47x the cut
#: (against ~1e8x for a real amplifying mode), so one decade disarms them --
#: which is what ``W`` is, and what the ubuntu runners needed.
_DISARM_SCALES = (1.0, 3.0) + tuple(
    float(PC._MODE_CUT_MARGIN_WARN) ** k for k in (1, 2, 3, 4))


def _first_disarming_scale(ns, mf, degrees, mk):
    """``(scale, ladder, rows)`` -- the smallest :data:`_DISARM_SCALES` rung
    whose raw growing-mode census is ALL ZERO on this build, the order-0
    ladder it reads there, and the table walked to get there
    (``(scale, raw census, spread)`` per rung).

    ``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S5.1.  The NOT-CURED half's
    fail-before needs a build with NO mode across the cut -- that is the
    runner's condition -- and it used a FROZEN scale of 3.0 to produce one.
    3.0 disarms this cell on both mounts and did NOT on four runner jobs
    (``[0,0,0,1,1]`` on py3.11 / 3.12 / 3.13, ``[0,0,0,0,1]`` on py3.10), which
    is the same disease one layer up: a CUT POSITION frozen as a number.  Where
    the cut has to sit to clear a build's round-off is a fact about that build.

    So the scale is taken from the CENSUS instead of from a constant, and the
    census -- the shipped instrument -- decides when it is enough.  The
    guaranteed rung is not guessed: ``_mode_cut_growth`` only counts a mode
    while it is within ``_MODE_CUT_MARGIN_WARN`` of the cut, so that factor
    provably clears every mode the census is currently reporting.  The walk
    exists only to pay the CHEAPEST sufficient rung first (see
    :data:`_DISARM_SCALES`) and to catch the one thing the derivation does not
    cover -- a mode further out than the warn window being pulled INTO it by
    the raised cut.  Every build has a disarming scale (raising the cut can
    only move modes from propagating to evanescent, and ``n_grow`` counts only
    propagating ones), so failing to find one inside the derived ceiling is a
    real finding and is reported as one rather than passed over."""
    assert _DISARM_SCALES[-1] * min(_INJECTOR_SCALES) <= 1.0, (
        f"the disarm ladder tops out at {_DISARM_SCALES[-1]:.4g} while the "
        f"reconditioned helper probes back down by at most "
        f"{min(_INJECTOR_SCALES):.4g}: the two compose, so at that scale the "
        f"probe can no longer reach the shipped cut and the mechanism claim "
        f"below would be unreachable")
    assert float(PC._MODE_CUT_MARGIN_WARN) in _DISARM_SCALES, (
        f"the library counts a mode as growing only within "
        f"{PC._MODE_CUT_MARGIN_WARN:.4g} of the cut, so that factor is the "
        f"rung this walk is GUARANTEED to disarm at -- and it is not on "
        f"{tuple(float(f'{s:.4g}') for s in _DISARM_SCALES)}.  The ladder is "
        f"derived from that constant; if the constant moved, derive it again")
    rows = []
    for scale in _DISARM_SCALES:
        with near_cut_injector(scale):
            v, raw, _post, _nonpas = _ladder_rec(ns, mf, degrees, mk)
        rows.append((scale, [int(x) for x in raw], spread(v)))
        if not int(np.sum(raw)):
            return scale, v, rows
    assert False, (
        f"ns={ns} min_feature={mf * 1e9:.1f} nm: no near-cut injector scale in "
        f"{tuple(float(f'{s:.4g}') for s in _DISARM_SCALES)} DISARMED this "
        f"cell -- it still reads a raw growing mode at every rung "
        f"({rows}).  Raising the cut can only move modes from propagating to "
        f"evanescent and n_grow counts only propagating ones, so a census that "
        f"survives four decades of it is a second mechanism and must be "
        f"diagnosed.  Widen the ladder (and its ceiling derivation with it) "
        f"rather than deleting this")


def _injected_null_control(cells, degrees, mk, original_msg, match):
    """Shared body of the three injector-driven fail-befores.  Returns
    ``(table, scale)``.

    ``docs/audits/FIX_M2_NULL_CONTROL_2026_08_09.md``.  A test cannot change
    the runner's OpenBLAS kernel, but it CAN put the cut where that kernel's
    round-off put it.  So: walk :data:`_INJECTOR_SCALES` until one puts a raw
    growing mode on one of THESE cells, score the reconditioned control there,
    and pin the ORIGINAL premise FAILING on the same table.

      (a) the reconditioned control passes under the injector, and for the
          RIGHT REASON -- every rung it lets move carries a raw growing mode,
          ends with none growing, and ends no further from the cured answer;
      (b) ``original_msg`` -- the caller's own pre-2026-08-09 assertion, made
          VERBATIM on that same table -- FAILS;
      (c) the null half is not vacuous: rungs whose census reads ZERO are still
          scored BIT-IDENTICAL, and (c) is decided on the CENSUS reading, not
          on "did it move".  Those come apart at a deep scale, where a rung can
          carry a growing mode and still not move.

    The SCAN is repair-ON only -- ``n_grow`` is the pre-repair diagnosis and
    does not move with the switch -- so a scale that carries nothing costs one
    ladder, not two.  The reference is each cell's UN-INJECTED shipped ladder:
    the answer the library gives when the cut sits where this build's own
    round-off puts it, which is the only reference that exists for all three
    devices (``RCWA_R0`` covers two ``n_slice`` of one of them).

    Nothing about the geometry or the physics is injected -- the eigenvalues,
    the operator and the materials are untouched.  Only the cut moves, which is
    the one thing a BLAS kernel is entitled to move."""
    PC.PMM_FORWARD_GROWTH_REPAIR = True
    refs = {c: _ladder_rec(c[0], c[1], degrees, mk)[0] for c in cells}
    used, table = None, None
    for scale in _INJECTOR_SCALES:
        with near_cut_injector(scale):
            PC.PMM_FORWARD_GROWTH_REPAIR = True
            if not any(int(np.sum(_ladder_rec(c[0], c[1], degrees, mk)[1]))
                       for c in cells):
                continue
            # (a) the reconditioned control, scored under the injector
            used, table = scale, _score_null_control(cells, refs, degrees, mk)
        break
    assert used is not None, (
        f"no near-cut injector scale in "
        f"{tuple(float(f'{s:.4g}') for s in _INJECTOR_SCALES)} put a growing "
        f"mode on ANY rung of {list(cells)} on this build, so the 2026-08-09 "
        f"CI condition is not reproduced here and this test has stopped being "
        f"a fail-before.  Widen the scale ladder rather than deleting it -- "
        f"the condition is a CUT POSITION and every build has one.")

    moved = [r for r in table if r[7] > 0.0]
    # the rungs the control scored on its BIT-IDENTITY branch -- the census
    # reading, which is what the branch is chosen on, NOT "did it move"
    nulls = [r for r in table if r[3] == 0]
    print(f"\nM2 null control [{mk.__name__}], near-cut injector at {used:.4g}:"
          f" {len(moved)} of {len(table)} cured rungs moved, "
          f"{len(nulls)} scored bit-identical -- "
          + ", ".join(f"ns={r[0]} deg={r[2]} raw={r[3]} post={r[4]} "
                      f"{r[5]:.7f} -> {r[6]:.7f} (moved {r[7]:.4g})"
                      for r in moved))

    # (b) THE FAIL-BEFORE: the caller's original claim, verbatim, same table.
    with pytest.raises(AssertionError, match=match):
        for cell in cells:
            rows = [r for r in table if (r[0], r[1]) == cell]
            a = np.asarray([r[6] for r in rows], dtype=float)
            b = np.asarray([r[5] for r in rows], dtype=float)
            assert float(np.max(np.abs(a - b))) == 0.0, original_msg(*cell)

    # ... and it failed for the reason the fix names: every rung that moved
    # was carrying a growing mode the repair was entitled to redirect, and it
    # ended with none.  (Both are already asserted inside the scorer; restated
    # here so the fail-before is pinned to its DIAGNOSIS and not merely to the
    # fact that something moved.)
    assert all(r[3] > 0 and r[4] == 0 for r in moved), (
        f"a cured rung moved without carrying a raw growing mode, or ended "
        f"still growing: {[(r[0], r[2], r[3], r[4]) for r in moved]}")

    # (c) the null half is not vacuous -- the injector does not put a growing
    #     mode on EVERY cured rung, so the bit-identity claim was really scored
    #     on some of them.
    assert nulls, (
        f"the injector at {used:.4g} put a growing mode on every one of the "
        f"{len(table)} cured rungs, so the BIT-IDENTITY half of the control "
        f"was never exercised and (a) proves only that nothing raised.  Use a "
        f"shallower scale -- the point of this test is that the two branches "
        f"COEXIST on one ladder")
    assert all(r[7] == 0.0 for r in nulls), (
        f"a rung whose census reads zero moved: "
        f"{[(r[0], r[2], r[7]) for r in nulls if r[7]]}")
    return table, used


def test_the_null_control_conditions_on_the_census_not_the_threshold(
        growth_repair_off):
    """THE 2026-08-09 CI FAILURE, REPRODUCED ON THIS BUILD -- and the
    fail-before for section (5) of the test above.

    ``docs/audits/FIX_M2_NULL_CONTROL_2026_08_09.md``.

    ``test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too`` failed
    on the ubuntu py3.10 shard of main with

        ns=2 min_feature=5e-10: an ABOVE-threshold cell moved when the
        forward-growth repair was switched

    rung 4 of that ladder moving 0.04886, on a tree that was green on branch CI
    and on both mounts at every thread count.  A test cannot change the runner's
    OpenBLAS kernel, but it CAN put the cut where that kernel's round-off put
    it: ``_mass_flux_threshold`` scaled by 1/3 (``FIX_CI_ROUND2`` S6's near-cut
    injector) drives a CURED cell into carrying a near-cut growing mode on this
    box, and the resulting move is the CI number to four figures --

        [M, Windows, 1 thread]  ns=2, min_feature 0.5 nm, degree 12
        repair OFF  0.0616396      repair ON  0.1104990    moved  4.886e-02

    -- so the CI failure is not a mystery about ubuntu, it is what happens
    whenever the cut lands one notch lower than it does here.  WHICH rung
    carries it migrates with the pool, exactly as everything else in this family
    does, so this test SCANS rather than naming one:

        mount / OPENBLAS_NUM_THREADS   rung(s) moved at scale 1/3
        Windows 1                      ns=2 deg 12, ns=3 deg 10
        Windows 2                      ns=3 deg 12
        Windows 24                     ns=2 deg 12 + 14, ns=3 deg 10 + 14
        WSL 1                          ns=2 deg 12, ns=3 deg 10
        WSL 2                          ns=3 deg 12

    What is asserted:

      (a) the RECONDITIONED control passes under the injector, and passes for
          the RIGHT REASON -- every rung it lets move carries a raw growing
          mode, ends with none growing, and ends no further from the cured
          answer;
      (b) the ORIGINAL control -- bit-identity on every above-threshold rung --
          is re-run VERBATIM on that same table and FAILS.  That is the
          fail-before, and it is the CI failure itself;
      (c) the null half is not vacuous: the injector leaves rungs whose census
          reads ZERO, and those are still scored BIT-IDENTICAL.

    (c) is scored on the CENSUS reading and not on "did it move", and the
    difference is not pedantry -- at a deep injector scale every rung carries a
    growing mode while some still do not move (measured: scale 1e-3, ns=2
    degree 6, raw 2, moved 0.0), and a rung like that was scored by the
    moved-rung branch, not by the bit-identity one.  The first draft of (c)
    partitioned on the movement and would have called that rung a null.

    Nothing about the geometry or the physics is injected -- the eigenvalues,
    the operator and the materials are untouched.  Only the cut moves, which is
    the one thing a BLAS kernel is entitled to move.
    """
    _injected_null_control(
        ABOVE_THRESHOLD, _LADDER_DEGREES, _coated_stack,
        lambda ns, mf: (
            f"ns={ns} min_feature={mf}: an ABOVE-threshold cell moved when "
            f"the forward-growth repair was switched -- the repair is only "
            f"allowed to touch solves that put a growing mode in the forward "
            f"set, and a cured cell has none"),
        "ABOVE-threshold cell moved")


def test_min_feature_threshold_rule_predicts_stationarity(growth_repair_off):
    """N-6, the QUANTITATIVE form of the contract, and the reason it is a rule
    and not a heuristic.

    Inside a +/-1 window the only cross-layer walls are ADJACENT slices', so
    for a staircased taper with a conformal coat ``c`` and per-slice offset
    ``off = (thickness/ns) tan(sidewall)`` the window's cross-layer
    separations are EXACTLY ``{off, |c - off|}``.

    **WHICH OF THE TWO SEPARATIONS IS THE BUILD-FREE BOUNDARY -- CORRECTED
    2026-08-16 (S4, at-threshold comparison; docs/TESTING_STANDARDS.md).**
    This test asserted the contract as

        min_feature  >  min(off, |c - off|)      -> stationary

    and pinned ``(ns=3, 1.5 nm)`` as an ABOVE-threshold cell.  At ns = 3 the
    separations are ``off`` = 3.6085 nm and ``|c - off|`` = 1.3915 nm, so that
    cell sits 7.8 % above the ``min`` -- and on WSL py3.12 / numpy 2.5.1 /
    scipy 1.18.0 at 4 BLAS threads it does not behave like an above-threshold
    cell at all: spread 2.27008, i.e. a full collapse, against 0.0039957 on
    every other environment.  Clearing ``min`` merges the SMALLER sliver and
    leaves the LARGER separation unmerged, and whether that residual sliver
    collapses the ladder is decided by round-off.  Scanned at ns = 2, 3, 6, 8
    over 6 (build x thread-cap) environments::

        placement                     spread
        0.5 x min(off, |c - off|)     0.685 .. 3.050     COLLAPSE, every cell
        1.2 x min(off, |c - off|)     0.00356 .. 2.2701  BUILD-DEPENDENT
        0.9 x max(off, |c - off|)     0.00356 .. 2.2763  BUILD-DEPENDENT
        1.2 x max(off, |c - off|)     0.00356 .. 0.00586 STATIONARY, every cell
        1.5 x max(off, |c - off|)     0.00356 .. 0.00586 STATIONARY, every cell

    -- so the two-sided, build-free form of the contract is

        min_feature  <  min(off, |c - off|)   -> collapse   on every build
        min_feature  >  max(off, |c - off|)   -> stationary on every build

    with the band BETWEEN them build-dependent.  The cells are now DERIVED
    from each ``ns``'s own separations at a stated margin
    (:data:`_RULE_BELOW_FACTOR` / :data:`_RULE_ABOVE_FACTOR`) instead of listed
    in nm, so nothing sits 7.8 % from its own boundary again, and the claim is
    a PARTITION with a measured separation rather than two absolute bars.  The
    intermediate band is measured and PRINTED, not asserted -- it is exactly
    where the answer is a per-build fact.

    NOTE for the library, recorded not fixed here: ``_mode_cut_verdict``'s
    message tells the user to "raise min_feature above min(off, |coat - off|)".
    The measurement above says that advice is not sufficient on every build;
    ``max`` is.  That is a library-message finding and belongs in its own
    change, not in a test-hardening pass.

    **2026-08-06.**  The rule is now a statement about the library WITHOUT the
    forward-growth repair, so its four assertions are scored with the switch
    off -- verbatim, same cells, same bars.  What the repair adds is the other
    half: the threshold is no longer an ACCURACY CLIFF, because the
    below-threshold cells are stationary too once the forward set cannot grow.
    The cells these two ns values name do NOT migrate with the thread count
    (measured 2.63/2.81 and 2.25/1.97 at 1/2 threads, both mounts), which is
    why they can stay pinned while the ns=2 cell in the sibling above cannot.
    """
    # ---- the rule, on the pre-repair library (switch off via the fixture) --
    below, above, band = {}, {}, {}
    for ns in _RULE_NS:
        lo, hi = sorted(_window_separations(ns))
        mf_lo, mf_hi = _RULE_BELOW_FACTOR * lo, _RULE_ABOVE_FACTOR * hi
        below[(ns, mf_lo)] = spread(coated_ladder(ns, mf_lo))
        above[(ns, mf_hi)] = spread(coated_ladder(ns, mf_hi))
        # the band between the two separations, MEASURED and printed: this is
        # the region the corrected rule declines to predict, and printing it
        # is how a future build's migration is seen rather than tripped over.
        band[(ns, 1.2 * lo)] = spread(coated_ladder(ns, 1.2 * lo))
    for (ns, mf), s in below.items():
        assert s > _RULE_COLLAPSE_BAR, (
            f"ns={ns} min_feature={mf * 1e9:.4f} nm is BELOW that cell's "
            f"smaller cross-layer separation "
            f"{min(_window_separations(ns)) * 1e9:.4f} nm, so the PRE-REPAIR "
            f"degree ladder must COLLAPSE -- spread {s:.4g}")
    for (ns, mf), s in above.items():
        assert s < _RULE_STATIONARY_BAR, (
            f"ns={ns} min_feature={mf * 1e9:.4f} nm is ABOVE that cell's "
            f"LARGER cross-layer separation "
            f"{max(_window_separations(ns)) * 1e9:.4f} nm, so no sliver "
            f"survives and the ladder must be STATIONARY -- spread {s:.4g}")
    # ... and the two really are two clouds, not one with a bar inside it.
    assert min(below.values()) > _RULE_SEPARATION * max(above.values()), (
        f"the collapse/stationary partition is inside one population: "
        f"below {[round(v, 5) for v in below.values()]} vs above "
        f"{[round(v, 6) for v in above.values()]}")
    print("\nN-6 rule (pre-repair): below "
          + str({f"ns={k[0]}@{k[1] * 1e9:.3f}nm": round(v, 5)
                 for k, v in below.items()})
          + "  above "
          + str({f"ns={k[0]}@{k[1] * 1e9:.3f}nm": round(v, 6)
                 for k, v in above.items()})
          + "  [band, not asserted] "
          + str({f"ns={k[0]}@{k[1] * 1e9:.3f}nm": round(v, 5)
                 for k, v in band.items()}))

    # ---- and what the repair changes: the cliff, not the rule --------------
    PC.PMM_FORWARD_GROWTH_REPAIR = True          # the fixture owns the restore
    # the ABOVE-threshold side is untouched, bit for bit: there is no sliver
    # left, so there is no growing mode to repair.
    #
    # 2026-08-09 (FIX_M2_NULL_CONTROL_2026_08_09): scored through the shared
    # census-conditioned scorer, for the reason its sibling's section (5) was
    # re-stated.  "There is no growing mode to repair" is a claim about the
    # CENSUS, and this test was asserting it as a claim about the min_feature
    # THRESHOLD; the two come apart on a runner whose round-off puts a
    # near-cut mode on a cured cell (the referred failure), and this test
    # carried the identical premise on a different pair of cells.  On every
    # build measured both cells read ZERO raw growing modes on every rung, so
    # the assertion below IS the bit-identity one above, verbatim.  Its
    # fail-before is the test immediately following.
    _score_null_control(_RULE_CURED,
                        {c: RCWA_R0.get(c[0]) for c in _RULE_CURED})
    # the BELOW-threshold side no longer collapses (measured 3.79e-03 and
    # 4.24e-03 against the 0.1 the rule predicts without the repair -- a
    # 25x-fold separation, at every thread count on both mounts).
    for ns, mf in ((3, 0.5e-9), (6, 1.5e-9)):
        s = spread(coated_ladder(ns, mf))
        assert s < 1e-2, (
            f"ns={ns} min_feature={mf}: still spreads {s:.4g} with the "
            f"forward-growth repair on -- below the min_feature threshold is "
            f"supposed to have stopped being an accuracy cliff")


def test_the_threshold_rule_null_control_conditions_on_the_census_too(
        growth_repair_off):
    """The fail-before for ``test_min_feature_threshold_rule_predicts_
    stationarity``'s own null floor.

    ``docs/audits/FIX_M2_NULL_CONTROL_2026_08_09.md`` S7.  That test asserted
    bit-identity on ITS above-threshold pair with the same premise the referred
    CI failure refuted -- "there is no growing mode to repair" stated as a fact
    about the min_feature THRESHOLD rather than about the CENSUS.  It had not
    failed on any runner; it carried the identical exposure.

    Driven with the near-cut injector at ``FIX_CI_ROUND2`` S6's own 1/3, the
    exposure is real on this device, and it is the ``ns = 3`` cell that carries
    it (this test's pair does not include ``ns = 2``, so it is a DIFFERENT rung
    from the one CI found):

        mount / OPENBLAS_NUM_THREADS   rung(s) moved at scale 1/3
        Windows 1                      ns=3 deg 10  (raw 1, 0.6594 -> 0.1112)
        Windows 2                      ns=3 deg 12
        Windows 24                     ns=3 deg 10 + 14
        WSL 1                          ns=3 deg 10
        WSL 2                          ns=3 deg 12

    Everything else is the sibling's: (a) the reconditioned control passes for
    the right reason, (b) this test's ORIGINAL assertion -- "an above-threshold
    cell moved when the repair was switched" -- is re-run VERBATIM on the same
    table and fails, (c) the null half is not vacuous.
    """
    _injected_null_control(
        _RULE_CURED, _LADDER_DEGREES, _coated_stack,
        lambda ns, mf: (f"ns={ns} min_feature={mf}: an above-threshold cell "
                        f"moved when the repair was switched"),
        "above-threshold cell moved")


def test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper(
        growth_repair_off):
    # N-6, the GENERAL form -- and the answer to M5's escalation
    # (docs/audits/PMM_M5_2D_FEASIBILITY_2026_08_04.md S4), which reported the
    # same silent-wrong scatter on the SIMPLEST member of the class: one
    # region, lossless dielectric, 2-deg taper, NO coat.  There is no
    # coat/offset resonance at all there, so the collision cannot be blamed on
    # the coat -- and the rule still holds, because the window's only
    # cross-layer separation IS the per-slice offset itself:
    #
    #     off = (H / ns) * tan(sidewall)   ->   3.61 / 1.80 / 0.90 nm at
    #                                           ns = 3 / 6 / 12
    #
    # The load-bearing worry was that snapping the ONLY separation there would
    # cure the collapse by DELETING THE TAPER.  It does not: the cured ladders
    # land within 0.3% of the undisturbed ns=3 value and keep the correct
    # monotone n_slice trend (checked against RCWA in the M2 audit S9).
    degs = _UNCOATED_DEGREES

    def spread_u(ns, mf):
        # 2026-08-09: the local cache this test carried is now ``_LADDER_CACHE``
        # via ``_ladder_rec`` -- the SAME builders and the same degrees, plus
        # the census columns the re-pointed null control below reads.  The key
        # gains the device and the injector scale; it already carried both
        # switches, which is what a repair-vs-no-repair comparison needs.
        v = _ladder_rec(ns, mf, degs, _uncoated_stack)[0]
        return float((v.max() - v.min()) / abs(v.mean())), v

    # ns = 6, off = 1.804 nm: 1.5 nm is BELOW the threshold, 3.0 nm above it.
    #
    # 2026-08-12 (FIX_RUNNER_PINS_2026_08_12 S5): the NOT-CURED half goes
    # through the census-conditioned helper, for the reason the three null
    # controls in this file already did.  "1.5 nm does not cure ns=6" was being
    # measured as a SPREAD, but the spread only collapses where the near-cut
    # collision fires, which is a fact about the build's round-off and not
    # about the min_feature threshold; the ubuntu py3.12 shard read this cell
    # at 0.00406.  The helper asserts the geometric premise (the window really
    # is unsnapped) plus the mechanism where the instrument says it is armed,
    # and probes with the near-cut injector where it is not.
    #
    # 2026-08-15 (FIX_RUNNER_PINS_2_2026_08_15 S8.5, shape S6): that helper
    # partitioned on the census AT THE SHIPPED CUT and then asserted "armed =>
    # collapsed".  main CI read [0, 0, 0, 0, 2] with the ladder still at its
    # cured 0.004055 and failed here -- an armed census is NECESSARY for the
    # collapse, never SUFFICIENT.  The helper now makes the mechanism claim as
    # a REACH over a ladder of CUT POSITIONS that starts at this build's own
    # cut, which names no census reading; see its docstring for the readings
    # both mounts produce.
    s_lo, _rows = _uncured_below_threshold(6, 1.5e-9, _uncoated_layers(6),
                                           degs, _uncoated_stack, 1.804)
    s_hi, v_hi = spread_u(6, 3.0e-9)
    assert s_hi < 2e-2, f"3.0 nm should cure ns=6: {s_hi:.3g}"
    # ns = 12, off = 0.902 nm: 1.5 nm is already above the threshold
    s12, _v12 = spread_u(12, 1.5e-9)
    assert s12 < 2e-2, f"1.5 nm should cure ns=12 (off=0.902 nm): {s12:.3g}"

    # ... and the cured values are the SAME PHYSICS, not a flattened taper.
    # The check must hold min_feature FIXED across n_slice, or it compares
    # three different geometry treatments: at 3.0 nm the ns=3 rung (off =
    # 3.61 nm) is not snapped at all while ns >= 6 is, and the trend is
    # legitimately non-monotone for that reason alone.  At 6.0 nm every rung
    # is snapped, and then the n_slice trend is monotone increasing -- the
    # same direction and order as RCWA's on this device (M2 audit S9).
    vals = [spread_u(ns, 6.0e-9) for ns in (3, 6, 8, 12)]
    last = [v[-1] for _s, v in vals]
    for (sp, _v), ns in zip(vals, (3, 6, 8, 12)):
        assert sp < 2e-2, f"ns={ns} not degree-stationary at 6 nm: {sp:.3g}"
    assert last[0] < last[1] < last[2] < last[3], (
        f"cured n_slice trend not monotone at a FIXED min_feature: {last}")
    assert (max(last) - min(last)) / abs(np.mean(last)) < 2e-2, (
        f"cured n_slice values spread too far -- the snap may be flattening "
        f"the taper rather than removing a collision: {last}")

    # ---- 2026-08-06: the same re-pin as the two siblings ------------------
    # Everything above is the rule on the PRE-REPAIR library (the fixture
    # holds the switch off) and is unchanged, cells and bars.  What the
    # forward-growth repair adds, on the SIMPLEST member of the class -- one
    # region, lossless, no coat, so the collapse can never be blamed on a
    # coat/offset resonance:
    PC.PMM_FORWARD_GROWTH_REPAIR = True          # the fixture owns the restore
    # (a) the cured cells do not move a bit -- the null floor.
    #
    # 2026-08-09 (FIX_M2_NULL_CONTROL_2026_08_09): scored through the shared
    # census-conditioned scorer, the same re-statement its two siblings got.
    # "A cured cell has no growing mode" is a claim about the CENSUS and was
    # being asserted as a claim about the min_feature threshold.  On every
    # build measured both cells read ZERO raw growing modes on every rung, so
    # this IS the bit-identity assertion it replaces, verbatim.  There is no
    # independent reference on this device -- it has no RCWA anchor in this
    # file -- so the moved half is scored on ``n_grow_post`` alone, which is
    # the claim that does not need one.  Fail-before: the test below.
    _score_null_control(_UNCOATED_CURED, {c: None for c in _UNCOATED_CURED},
                        degs, _uncoated_stack)
    # (b) the BELOW-threshold cell stops collapsing (measured 4.06e-03 with
    #     the repair on against 1.34 with it off, both thread counts).
    s_lo_on, _v = spread_u(6, 1.5e-9)
    assert s_lo_on < 1e-2, (
        f"uncoated ns=6 at 1.5 nm still spreads {s_lo_on:.4g} with the "
        f"forward-growth repair on (it spread {s_lo:.4g} with it off)")


def test_the_threshold_rules_NOT_CURED_half_conditions_on_the_census_too(
        growth_repair_off):
    """The fail-before for ``test_threshold_rule_holds_on_a_SINGLE_REGION_
    uncoated_taper``'s own NOT-CURED half.

    ``docs/audits/FIX_RUNNER_PINS_2026_08_12.md`` S5.  Its two SIBLING null
    controls were re-stated on the census in ``FIX_M2_NULL_CONTROL_2026_08_09``;
    this half kept the identical premise on the SAME cell -- "1.5 nm does not
    cure ns=6" measured as a degree-ladder SPREAD -- and it is the one the
    ubuntu py3.12 shard of main actually failed, at 0.00406.

    The injector runs the OTHER WAY here, and that is the point.  The null
    controls need a mode PUSHED across the cut, so they scale the threshold
    DOWN; this claim needs the runner that has NO mode near the cut, so it
    scales UP and DISARMS the collision on a build whose own round-off arms it.
    It is still the runner being emulated and not the physics -- the geometry,
    the materials, the operator and the eigenvalues are untouched, and only
    where the cut lands moves, which is the one thing an OpenBLAS kernel is
    entitled to move.

    The emulation is EXACT, not merely qualitative: at x3 the cell reads
    0.00406 on both mounts, which is the CI assertion message's number to all
    four of its figures, and its census goes to all-zero::

        mount / OPENBLAS_NUM_THREADS   cut x1                cut x3
        Windows py3.14 default         [0,0,1,2,3]  1.33841  [0,0,0,0,0]  0.00406
        WSL py3.12 1                   [0,0,1,1,1]  1.33841  [0,0,0,0,0]  0.00406
        WSL py3.12 default             [0,0,1,2,3]  1.33841  [0,0,0,0,0]  0.00406

    (x10, x1e2, x1e4 and x1e6 all read 0.00406 too: once the near-cut mode is
    back under the cut there is nothing further for a higher cut to change.)

    2026-08-12, SECOND ADJUDICATION (same document, S5.1).  x3 was frozen as a
    NUMBER, and four runner jobs refused it: it left ``[0,0,0,1,1]`` on
    py3.11 / 3.12 / 3.13 and ``[0,0,0,0,1]`` on py3.10, i.e. the cut had to go
    HIGHER there than on either mount before that build's round-off flux fell
    under it.  It REPRODUCES locally at a reduction width the table above
    never drove -- WSL py3.12 at ``OPENBLAS_NUM_THREADS=4`` reads
    ``[0,0,1,2,1]`` spread 1.93102 at x1 and ``[0,0,0,1,0]`` spread 2.13121 at
    x3, i.e. x3 does NOT disarm there either, and x10 does.  So x3 is not a
    property of the mount, it is a property of the mount at a given width, and
    it was never anything but a number.

    This test's own message already named the treatment ("Raise the scale
    rather than deleting it: the condition is a CUT POSITION and every build
    has one"), so the scale is now SEARCHED -- ``_first_disarming_scale``
    walks :data:`_DISARM_SCALES` from the shipped cut upward until the SHIPPED
    census reads all zero, and the ceiling it stops at is derived twice over
    (from ``_INJECTOR_SCALES``' own reach and from
    ``_mass_flux_threshold``'s ``1e-9``), not chosen.  Nothing is frozen but
    the instrument.

    That also makes the search itself falsifiable HERE, where 3.0 works: this
    box is ARMED at the shipped cut, so the walk must escalate off its first
    rung.  A build already in the runner's condition stops at 1.0 and that
    self-check is skipped with it -- there is nothing to escalate away from
    there, and the three claims below are the whole point on such a build.

    **2026-08-15** (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5).  The
    escalation self-check used to ALSO require the rung the walk left behind to
    SCATTER, i.e. that an armed census and a collapsed ladder are the same
    event.  main CI refuted that -- armed on the last rung only, ladder still
    cured -- and it is the second of the two assertions this round repairs.
    See the block itself; the scatter claim survives build-free inside (a),
    which is now a REACH over cut positions rather than a reading at one.

    Three claims, the siblings':

    (a) the RECONDITIONED half passes under the injector, and passes for the
        right reason -- from the DISARMED cut it walks back down and finds a
        cut position where the collision both arms and collapses;
    (b) the ORIGINAL assertion, verbatim, on the SAME reading, FAILS;
    (c) the reach is not vacuous -- the disarmed reading really is the
        cured-looking one, so (a) is not passing on an armed ladder.
    """
    cell, off_nm = (6, 1.5e-9), 1.804
    layers = _uncoated_layers(cell[0])
    scale, v_off, rows = _first_disarming_scale(
        cell[0], cell[1], _UNCOATED_DEGREES, _uncoated_stack)
    print(f"\nM2 NOT-CURED fail-before [uncoated ns={cell[0]} "
          f"min_feature={cell[1] * 1e9:.1f} nm]: disarmed at cut x{scale:.4g} "
          "-- " + ", ".join(f"x{s:.4g} raw={r} spread={sp:.5g}"
                            for s, r, sp in rows))
    # THE CEILING, with teeth, scored against the SHIPPED answer.  Raising the
    # cut only emulates a runner while it stays a GAUGE change; pushed far
    # enough it deletes the spectrum instead, and the SPREAD cannot see that
    # (at x1e9 -- 1/1e-9, _mass_flux_threshold's own relative floor -- the
    # census reads n_prop = 0 on every rung and the ladder moves 7.7e-9, while
    # the spread still reads 0.0040554 to six figures).  So the disarmed
    # PRE-repair ladder is scored against the ladder the library SHIPS with
    # the forward-growth repair ON, which is the adjudication in one line: the
    # runner's build never produced the growing mode the repair exists to
    # redirect, so its pre-repair answer IS the shipped answer.  Measured
    # BIT-IDENTICAL (tolerance 0.0) at every disarming rung of the ladder.
    PC.PMM_FORWARD_GROWTH_REPAIR = True          # the fixture owns the restore
    v_on = _ladder_rec(cell[0], cell[1], _UNCOATED_DEGREES, _uncoated_stack)[0]
    PC.PMM_FORWARD_GROWTH_REPAIR = False
    assert np.array_equal(np.asarray(v_off), np.asarray(v_on)), (
        f"the ladder disarmed at cut x{scale:.4g} is not the ladder the "
        f"library ships with the forward-growth repair on: "
        f"{list(v_off)} against {list(v_on)}.  Disarming the census is "
        f"supposed to reach the SAME answer by never making the growing mode "
        f"the repair redirects -- a difference means the raised cut stopped "
        f"being a gauge change and started deleting propagating modes, and "
        f"then this is no longer an emulation of any runner")
    if len(rows) > 1:
        # This build's own round-off ARMS the cell, so the walk had to
        # escalate off its first rung.
        #
        # 2026-08-15 (docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md S8.5, shape
        # S6).  THIS CLAIM USED TO ADD "and the rung it left behind must also
        # SCATTER", i.e. that the census and the collapse are the same event.
        # They are not, and this was the SECOND of the two assertions main CI
        # failed: the runner read [0, 0, 0, 0, 2] at the SHIPPED cut -- armed,
        # on the last rung only -- with the ladder still at its cured 0.004055,
        # so the walk escalated off an armed rung that had not collapsed.  An
        # armed census is NECESSARY for the collapse and never SUFFICIENT
        # (_mode_cut_growth counts a near-cut growing mode; it says nothing
        # about how strongly that mode is excited), so the equivalence was a
        # per-build coincidence of the two mounts, which happen to arm and
        # collapse together at every width measured:
        #
        #     build / OPENBLAS_NUM_THREADS   cut x1        spread   disarm rung
        #     W py3.14 numpy 2.4.4 / 1       [0,0,1,1,1]   1.33841  x3
        #     M py3.12 numpy 2.5.1 / 1       [0,0,0,3,0]   2.38098  x10
        #     M py3.12 numpy 2.4.6 / 1       [0,0,1,1,1]   1.33841  x3
        #     M py3.12 numpy 2.4.6 / 4       [0,0,1,2,1]   1.93102  x10
        #     ubuntu, main CI 2026-08-15     [0,0,0,0,2]   0.004055 (armed and
        #                                                  NOT collapsed)
        #
        # (rows 1-2 measured 2026-08-15, rows 3-4 are the 2026-08-12 table on
        # the numpy 2.4 wheel -- the same mount and width reads a different
        # census on a different wheel, which is the axis stated without any
        # injector at all.)
        #
        # So what is asserted here is the STRUCTURAL fact the walk is built on
        # -- it escalates on the CENSUS, and therefore only ever off an ARMED
        # rung -- which is a self-check binding this claim to the loop
        # condition in _first_disarming_scale rather than a reading.  The
        # SCATTER is adjudicated and printed.  It is not lost: (a) below makes
        # it build-free, as a REACH over cut positions.
        prev_scale, prev_raw, prev_spread = rows[-2]
        assert int(np.sum(prev_raw)), (
            f"the disarm walk escalated past cut x{prev_scale:.4g}, whose "
            f"census reads {prev_raw} -- but the walk only continues off a rung "
            f"the census ARMS, so the search and this claim have come apart")
        print(f"M2 NOT-CURED fail-before: the last ARMED rung (cut "
              f"x{prev_scale:.4g}, census {prev_raw}) "
              + (f"collapses at spread {prev_spread:.5g}, this build's own "
                 f"reproduction of the mechanism"
                 if prev_spread > _COLLAPSE_SPREAD else
                 f"spreads only {prev_spread:.5g} -- armed WITHOUT collapsing, "
                 f"the runner's 2026-08-15 condition, reproduced here"))
    with near_cut_injector(scale):
        # (a) the reconditioned half, under the runner's condition
        s_lo, _rows = _uncured_below_threshold(
            cell[0], cell[1], layers, _UNCOATED_DEGREES, _uncoated_stack,
            off_nm)
        # (c) ... and it really is the disarmed, cured-looking reading
        assert s_lo < 1e-2, (
            f"the disarmed cell (cut x{scale:.4g}) spreads {s_lo:.4g}, which "
            f"is not the cured-looking reading the runner saw (0.00406), so "
            f"(a) proved only that nothing raised")
        # (b) THE FAIL-BEFORE: the original claim, verbatim, same reading
        with pytest.raises(AssertionError, match="should NOT cure"):
            assert s_lo > 0.1, (
                f"1.5 nm should NOT cure ns=6 (off=1.804 nm): {s_lo:.3g}")


#: The reading MAIN CI produced on 2026-08-15 for the uncoated ns = 6 / 1.5 nm
#: cell at the SHIPPED cut, degrees 8..16, repair off -- ARMED (two growing
#: modes, on the last rung only) and NOT COLLAPSED.  It is the state neither
#: mount produces at any cut position measured on it, and it is the state that
#: refuted "armed => collapsed".  Verbatim from the failure message:
#:
#:     ns=6 min_feature=1.5nm: census reads [0, 0, 0, 0, 2] raw growing
#:     mode(s) over degrees (8..16) ... spread only 0.004055
_CI_2026_08_15 = (0.004055, [0, 0, 0, 0, 2])

#: The census tables the NOT-CURED half's REACH is driven against, one per
#: BUILD, as ``(spread, raw census)`` per rung of the cut ladder
#: ``(1.0,) + _INJECTOR_SCALES`` -- plus the rung the reach must land on, or
#: ``None`` where it must refuse.  2026-08-15,
#: ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5 (shape S6).
#:
#: The first four are MEASURED (2026-08-15, degrees 8..16, repair off,
#: uncoated ns = 6 / 1.5 nm).  Rows 1-2 are the two mounts' own readings, which
#: arm AND collapse at their own cut -- there the reach is the 2026-08-12
#: assertion verbatim.  Row 3 is the RUNNER: its shipped-cut rung is
#: :data:`_CI_2026_08_15` and its deeper rungs are M's, which is the numpy 2.5
#: wheel that job runs.  Row 4 is a build with no near-cut mode at all, the
#: branch 2026-08-12 added.  Rows 5-6 are REFUTATIONS -- what a threshold rule
#: that has genuinely stopped holding looks like from here -- and the reach
#: must FAIL on them rather than pass quietly.
_REACH_CASES = (
    ("[W] py3.14 numpy 2.4.4 / 1 thread: armed and collapsed at its own cut",
     ((1.33841, [0, 0, 1, 1, 1]),), 1.0),
    ("[M] py3.12 numpy 2.5.1 / 1 thread: same, on a different rung and wheel",
     ((2.38098, [0, 0, 0, 3, 0]),), 1.0),
    ("ubuntu main CI 2026-08-15: ARMED at the shipped cut and NOT collapsed",
     (_CI_2026_08_15, (2.38098, [0, 1, 0, 5, 5])), 1.0 / 3.0),
    ("a build with no mode near the cut at all (the 2026-08-12 condition)",
     ((0.004055, [0, 0, 0, 0, 0]), (0.004055, [0, 0, 0, 0, 0]),
      (1.35714, [10, 25, 24, 59, 72])), 1.0e-2),
    ("REFUTED: the census arms at every cut position and nothing collapses",
     ((0.004055, [0, 0, 0, 0, 2]), (0.004055, [0, 1, 0, 5, 5]),
      (0.004055, [10, 25, 24, 59, 72]), (0.004055, [29, 32, 42, 42, 55])),
     None),
    ("REFUTED: the ladder scatters but the census never arms -- not this "
     "mechanism, so it does not count",
     ((2.38098, [0, 0, 0, 0, 0]), (2.38098, [0, 0, 0, 0, 0]),
      (1.35714, [0, 0, 0, 0, 0]), (1.42007, [0, 0, 0, 0, 0])), None),
)


def _canned_probe(readings):
    """``(probe, seen)`` -- a reach probe that serves a canned per-rung table
    and ASSERTS that the walk visits the cut ladder IN ORDER, starting at the
    SHIPPED cut.  So each case pins WHERE the reach looks as well as what it
    concludes, and a walk that silently reordered or skipped a rung -- or ran
    past the table it was given -- is a failure and not a pass."""
    seen = []

    def probe(scale):
        i = len(seen)
        assert i < len(readings), (
            f"the reach asked for rung {i + 1} (cut x{scale:.4g}) of a "
            f"{len(readings)}-rung table: it walked past the ladder this case "
            f"describes instead of stopping where the case says it should")
        want = ((1.0,) + _INJECTOR_SCALES)[i]
        assert scale == want, (
            f"the reach asked for cut x{scale:.4g} at rung {i + 1}, but the "
            f"ladder's rung {i + 1} is x{want:.4g}: the walk must start at the "
            f"cut THIS BUILD finds (x1, no injector, no extra solve) and step "
            f"down _INJECTOR_SCALES in order")
        seen.append(float(scale))
        return readings[i]
    return probe, seen


def test_the_NOT_CURED_reach_is_scored_on_a_LADDER_of_cut_positions():
    """THE 2026-08-15 CI FAILURE, AND EVERY OTHER BUILD'S CENSUS, SCORED
    AGAINST THE SHIPPED DECISION -- the engineered fail-before for
    :func:`_uncured_below_threshold`.

    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S8.5, shape S6 (FIXED-CENSUS
    READING).  The sibling above emulates the runner by moving the CUT on the
    real device, which is the right instrument for the states this box's
    round-off can be driven into.  It cannot reach the state main CI actually
    read -- ARMED at the shipped cut with the ladder still cured
    (:data:`_CI_2026_08_15`) -- because at every cut position measured here,
    from x1e-4 up to wherever the collision disarms, an armed census on this
    cell also collapses (2026-08-15, one BLAS thread, degrees 8..16, repair
    off):

        cut     [W] raw census      spread    [M] raw census      spread
        x1e-4   [40,35,47,54,55]    1.33478   [29,32,42,42,55]    1.42007
        x1e-2   [10,26,28,43,67]    0.93247   [10,25,24,59,72]    1.35714
        x1/3    [0,1,3,4,6]         1.78619   [0,1,0,5,5]         2.38098
        x1      [0,0,1,1,1]         1.33841   [0,0,0,3,0]         2.38098
        x3      [0,0,0,0,0]         0.004055  [0,0,0,1,0]         2.38098
        x5/x10  [0,0,0,0,0]         0.004055  [0,0,0,0,0]         0.004055

    That coincidence of the two mounts is exactly what the retired conditional
    was pinned on, so the states that DO NOT manifest here are supplied
    directly: :data:`_REACH_CASES` drives the shipped decision -- geometry
    premise and all -- against each build's own table through a canned probe.
    The runner's row is its failure verbatim, and no coverage is lost on any
    build, because the branch a build cannot reach physically is still scored
    here.

    What is asserted:

      (a) every MEASURED table reaches the collision, and at the RUNG the case
          names -- rung one where the build's own cut collapses (both mounts),
          the first injected rung where it does not (the runner), the third
          where no mode is near the cut at all (the 2026-08-12 condition);
      (b) both REFUTATION tables are REFUSED, loudly: a census that arms at
          every cut position without ever collapsing, and a ladder that
          scatters without the census ever arming.  Neither is the mechanism,
          and a reach that accepted either would pass on a device where the
          threshold rule had stopped holding;
      (c) the ORIGINAL 2026-08-12 form -- the conditional, verbatim -- is
          re-run on the runner's own reading and FAILS.  That is the CI
          failure, pinned in-tree and permanently;
      (d) the GEOMETRY premise still has teeth and is checked FIRST: an
          ABOVE-threshold min_feature is refused before any cut is probed at
          all (the probe raises if it is called).

    No solve runs here -- the tables ARE the measurement -- so the whole thing
    scores the decision that consumes them in milliseconds, and it is the one
    test of this family that costs the suite nothing on any build.
    """
    layers = _uncoated_layers(6)
    for name, readings, want in _REACH_CASES:
        probe, seen = _canned_probe(readings)
        if want is None:                             # (b)
            with pytest.raises(AssertionError, match="admits it at none"):
                _uncured_below_threshold(6, 1.5e-9, layers, _UNCOATED_DEGREES,
                                         _uncoated_stack, 1.804, probe=probe)
            assert len(seen) == len(readings), (
                f"{name}: the reach refused after {len(seen)} of "
                f"{len(readings)} rungs -- it must exhaust the whole cut "
                f"ladder before it may call the rule refuted")
            continue
        sp, rows = _uncured_below_threshold(                     # (a)
            6, 1.5e-9, layers, _UNCOATED_DEGREES, _uncoated_stack, 1.804,
            probe=probe)
        assert rows[-1][0] == want, (
            f"{name}: the reach landed on cut x{rows[-1][0]:.4g}, not on the "
            f"x{want:.4g} this table's collision lives at -- walked "
            f"{[(float(f'{s:.4g}'), r, float(f'{p:.4g}')) for s, r, p in rows]}")
        assert sum(rows[-1][1]) and rows[-1][2] > _COLLAPSE_SPREAD, (
            f"{name}: the reach accepted a rung that is not both ARMED and "
            f"COLLAPSED: {rows[-1]}")
        assert sp == readings[0][0], (
            f"{name}: the spread returned to the caller is {sp:.5g}, not the "
            f"{readings[0][0]:.5g} this build reads at its OWN cut -- the "
            f"reach may escalate to make its claim, but what it hands back "
            f"(and what the caller's cured/not-cured comparisons are made on) "
            f"must stay the un-injected reading")

    # (c) THE FAIL-BEFORE: the 2026-08-12 conditional, verbatim, on the
    #     runner's own reading.  "armed => collapsed" is the claim, and the
    #     runner is the counterexample -- an armed census says a near-cut
    #     growing mode EXISTS, never that it is excited enough to move the
    #     answer.
    sp_ci, raw_ci = _CI_2026_08_15
    ns, mf, off_nm, degrees = 6, 1.5e-9, 1.804, _UNCOATED_DEGREES
    with pytest.raises(AssertionError, match="census reads"):
        if sum(raw_ci):                          # the retired partition ...
            assert sp_ci > 0.1, (                # ... and its claim, verbatim
                f"ns={ns} min_feature={mf * 1e9:.1f} nm (off={off_nm:.3f} nm): "
                f"the census reads {raw_ci} raw growing mode(s) over degrees "
                f"{tuple(degrees)}, so the near-cut collision the un-snapped "
                f"window admits DID fire -- and the degree ladder must scatter "
                f"with it, but it spreads only {sp_ci:.3g} (measured 1.34).  A "
                f"collision that fires without moving the answer is a second "
                f"mechanism and must be diagnosed")

    # (d) the geometry premise is checked BEFORE any cut is probed
    def never(scale):
        raise AssertionError(
            f"the reach probed cut x{scale:.4g} on an ABOVE-threshold cell: "
            f"the geometry premise must be refused before anything is solved")

    with pytest.raises(AssertionError, match="must come back UNSNAPPED"):
        _uncured_below_threshold(6, 3.0e-9, layers, _UNCOATED_DEGREES,
                                 _uncoated_stack, 1.804, probe=never)


def test_the_uncoated_null_control_conditions_on_the_census_too(
        growth_repair_off):
    """The fail-before for the SINGLE-REGION uncoated taper's null floor --
    and the measurement that says that device was the least exposed of the
    three.

    ``docs/audits/FIX_M2_NULL_CONTROL_2026_08_09.md`` S7.  Same premise, same
    re-statement: "a cured cell has no growing mode" is a claim about the
    census, not about the min_feature threshold.

    WHAT IS DIFFERENT HERE, and it is worth the measurement.  The coated device
    reaches the CI condition at ``FIX_CI_ROUND2`` S6's own 1/3 scale.  This one
    does NOT -- not at 1/3, not at 1e-1, 3e-2, 1e-2, 3e-3, 1e-3 or 3e-4.  It
    takes **1e-4**, i.e. the cut has to fall four decades before any mode's
    round-off flux crosses it [M, Windows 1 thread, cells (6, 3 nm) and
    (12, 1.5 nm), degrees 8..16]::

        scale     rungs with a raw growing mode   moved
        1         0 / 10                          0
        1/3       0 / 10                          0
        1e-2      0 / 10                          0
        1e-3      0 / 10                          0
        1e-4      3 / 10                          3  (ns=6 deg 10 + 16,
                                                      ns=12 deg 10)

    That is what a single region with no conformal coat buys: there is no
    coat/offset collision to inject near-zero-width cross-layer cells, so the
    spectrum has no round-off-flux mode anywhere near the cut.  It is a
    statement about THIS device, not a bar, and it is why the scale ladder has
    to reach 1e-4 -- the fail-before still exists here, it is simply four
    decades further away.

    The three claims are the sibling's, unchanged.
    """
    _injected_null_control(
        _UNCOATED_CURED, _UNCOATED_DEGREES, _uncoated_stack,
        lambda ns, mf: (f"uncoated ns={ns} min_feature={mf}: a cured cell "
                        f"moved when the forward-growth repair was switched"),
        "a cured cell moved")


def _mk_mf(layers, degree, mf):
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                  degree=degree, far_field_orders=11,
                  layer_grids="per-layer", min_feature=mf)
    for t, s in layers:
        st.add_layer(t, segments=s)
    return st


# ======================================================================= T3-1
def _uncoated_layers(ns):
    """The taper with NO conformal coat: the window's only cross-layer
    separation is ``off``, which is >= 3.6 nm at ns = 3, so the snap is
    provably inert at the library default and ``window_halfwidth`` is the
    ONLY variable."""
    dz = H1 / ns
    out = []
    for k in range(ns):
        zeta = (k + 0.5) / ns
        a = 0.5 * W_TOP - zeta * H1 * np.tan(SIDEWALL)
        w = 2.0 * a / PERIOD
        g = 0.5 * (1.0 - w)
        out.append((dz, [(g, EPS_GROOVE), (w, EPS_CORE), (g, EPS_GROOVE)]))
    return out


def _fatcoat_layers(ns):
    """25 nm coat -> separations {off, |25 - off|}, min 1.35 nm at ns = 8."""
    dz = H1 / ns
    out = []
    for k in range(ns):
        zeta = (k + 0.5) / ns
        a = 0.5 * W_TOP - zeta * H1 * np.tan(SIDEWALL)
        out.append((dz, _coated_segments(2.0 * a, 25 * NM)))
    return out


def _mk(layers, degree, hw):
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                  degree=degree, far_field_orders=11,
                  layer_grids="per-layer", window_halfwidth=hw)
    for t, s in layers:
        st.add_layer(t, segments=s)
    return st


def _solve_screened(st):
    """``((solve0 tuple), n_growing)`` -- the answer plus the SHIPPED T3-4
    growth instrument's reading over the whole solve, read off the census
    (``_MODE_CUT_CENSUS``) rather than re-derived, so this screen scores the
    instrument the library ships and cannot drift from it.

    ``n_growing > 0`` means the flux cut put a mode that GROWS along +z into
    the forward set of some layer or half-space -- a physical contradiction,
    hence a solve whose modal classification is decided by round-off.  See
    ``_core._mode_cut_growth``."""
    PC._MODE_CUT_CENSUS = []
    try:
        out = solve0(st)
        rows = list(PC._MODE_CUT_CENSUS)
    finally:
        PC._MODE_CUT_CENSUS = None
    return out, sum(int(r["n_grow"]) for r in rows)


def test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band():
    # T3-1, CONFOUND-CONTROLLED, and SCREENED (2026-08-05).
    #
    # Confound 1 (M2).  Measuring halfwidth on the coated device at
    # min_feature = 3.0 nm is NOT a window measurement: a +/-2 window holds
    # FIVE layers, so it snaps a different pair set than a +/-1 window and the
    # geometry changes too.  The tell was that max|dJ| read 1.169e-02 at BOTH
    # degree 6 and degree 8 -- a discretisation effect decays with degree, a
    # geometry difference does not.  So the devices below are run where the
    # snap is PROVABLY inert (asserted).
    #
    # Confound 2 (PMM_FOURNAME_ADJUDICATION_2026_08_05 S2), and it is the
    # OPPOSITE of what "the snap is inert" sounds like.  Inert at the library
    # default means the device's tightest cross-layer separation SURVIVES:
    # 3.6085 nm on the uncoated ns = 3 device (harmless) but 1.3532 nm on the
    # 25 nm coat at ns = 8, which is a sliver, so THAT device carries the
    # silent-wrong classification defect T3-4 exists to find.  A cell whose
    # classification flips between the halfwidth-1 and halfwidth-2 runs
    # measures the flip, not the window: it read |dJ| = 6.710e-01 at degree 8
    # (N BLAS threads) and 1.375e+00 at degree 12 (1 thread) -- O(1), the
    # silent-wrong magnitude, not a tolerance miss.
    #
    # So each cell is SCREENED with the shipped instrument first and the
    # contract is asserted on the sound ones.  The unsound cells are not
    # ignored: they are exactly the coverage
    # ``test_pmm_m3_efficiency.py::test_t34_guard_fires_on_...`` pins.
    #
    # The contract is then COMPARATIVE, with no absolute bar to calibrate:
    # the window residual must be a fraction of the DEGREE-REFINEMENT residual
    # at the same rung, and must decay spectrally.  Measured on the uncoated
    # device, IDENTICAL to five digits at 1 and at N BLAS threads:
    #
    #   degree   |dJ| window   |dJ| degree->degree+2   ratio
    #   6        1.2977e-04    1.2317e-03             0.105
    #   8        3.1232e-05    4.8489e-04             0.064
    #   10       6.3825e-06    2.3257e-04             0.027
    full_ladders, screened = 0, []
    for name, layers in (("uncoated ns=3", _uncoated_layers(3)),
                         ("25 nm coat ns=8", _fatcoat_layers(8))):
        segs = [s for _t, s in layers]
        for hw in (1, 2):
            merged, _d = _snap_report(segs, (PERIOD * 1e-5) / PERIOD, hw)
            assert merged == 0, "the snap must be inert for this measurement"
        got = {}
        for deg in (6, 8, 10, 12):
            for hw in ((1,) if deg == 12 else (1, 2)):
                got[(deg, hw)] = _solve_screened(_mk(layers, deg, hw))
        dJs = []
        for deg in (6, 8, 10):
            (a, ga), (b, gb) = got[(deg, 1)], got[(deg, 2)]
            (c, gc) = got[(deg + 2, 1)]
            if ga or gb or gc:
                screened.append((name, deg, ga, gb, gc))
                continue
            m = np.isin(a[4], b[4])
            mb = np.isin(b[4], a[4])
            dJ = float(np.max(np.abs(a[3] - b[3])))
            dR = float(np.max(np.abs(a[5][:, m] - b[5][:, mb])))
            dJ_deg = float(np.max(np.abs(a[3] - c[3])))
            assert dJ < 0.5 * dJ_deg, (
                f"{name} deg={deg}: the window moved the answer by "
                f"{dJ:.3e}, which is NOT inside the discretisation's own "
                f"residual band ({dJ_deg:.3e} from degree {deg} to {deg + 2})")
            assert dR < 0.5 * dJ_deg, f"{name} deg={deg}: |dR| = {dR:.3e}"
            dJs.append(dJ)
        if len(dJs) == 3:
            # the window residual is a DISCRETISATION residual: it must decay
            assert dJs[1] < dJs[0] and dJs[2] < dJs[1], (
                f"{name}: window residual did not decay with degree: {dJs}")
            full_ladders += 1
    assert full_ladders >= 1, (
        "every cell of both devices was screened out as classification-"
        "unsound, so T3-1 was not measured at all.  Add a device whose "
        "cross-layer separations are all above the sliver scale (the "
        "uncoated ns = 3 taper is one) rather than relaxing the screen.  "
        f"screened: {screened}")


# ======================================================================= T3-2
def test_staircase_is_stationary_in_n_slice_at_ns_8_to_12():
    # T3-2.  A converged discretisation stops moving: at a representable
    # min_feature the order-0 efficiency must be stationary across
    # n_slice 8 / 10 / 12 (the staircase truncation is O(1/ns^2), so the
    # residual step 8 -> 12 is small), AND stationary in degree at each ns.
    # Conservation is reported alongside per the standing rule.
    vals, closes = [], []
    for ns in (8, 10, 12):
        r6 = solve0(build(ns, degree=6, min_feature=1.5e-9))
        r8 = solve0(build(ns, degree=8, min_feature=1.5e-9))
        vals.append((r6[0], r8[0]))
        closes += [r6[2], r8[2]]
        # degree-stationarity at fixed ns
        assert abs(r6[0] - r8[0]) < 5e-3 * abs(r8[0]), (
            f"ns={ns}: degree 6 vs 8 moved {abs(r6[0] - r8[0]):.3e}")
    flat = np.asarray([v[1] for v in vals], dtype=float)
    spread = float((flat.max() - flat.min()) / abs(flat.mean()))
    assert spread < 2e-2, f"ns 8/10/12 spread {spread:.3e} at degree 8"
    assert max(closes) < 1e-4


# =============================================================== null control
def test_conforming_and_untapered_stacks_are_immune_to_both_knobs():
    # NULL CONTROL.  A stack whose layers SHARE walls has conforming windows
    # (mortar bypassed) and no cross-layer pair to snap, so neither
    # window_halfwidth nor min_feature may move a single bit.
    lay = [(200e-9, [(0.35, 4.0 + 0j), (0.65, 1.0 + 0j)]),
           (150e-9, [(0.35, 2.25 + 0j), (0.65, 1.0 + 0j)]),
           (120e-9, [(0.35, 6.25 + 0.1j), (0.65, 1.0 + 0j)])]

    def mk(**kw):
        st = PMMStack(PERIOD, n_substrate=1.5, n_superstrate=1.0, degree=8,
                      far_field_orders=7, layer_grids="per-layer", **kw)
        for t, s in lay:
            st.add_layer(t, segments=s)
        return st

    base = solve0(mk())
    for kw in (dict(window_halfwidth=2), dict(window_halfwidth=3),
               dict(min_feature=1.5e-9), dict(min_feature=3.0e-9),
               dict(window_halfwidth=2, min_feature=3.0e-9)):
        got = solve0(mk(**kw))
        assert float(np.max(np.abs(base[3] - got[3]))) == 0.0, kw
        assert float(np.max(np.abs(base[5] - got[5]))) == 0.0, kw

    # ... and a VERTICAL UNTAPERED staircase (identical slices) likewise.
    stk = [(H1 / 4, _coated_segments(W_TOP))] * 4

    def mk2(**kw):
        st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                      degree=8, far_field_orders=11, layer_grids="per-layer",
                      **kw)
        for t, s in stk:
            st.add_layer(t, segments=s)
        return st

    base = solve0(mk2())
    for kw in (dict(window_halfwidth=2), dict(min_feature=3.0e-9)):
        got = solve0(mk2(**kw))
        assert float(np.max(np.abs(base[3] - got[3]))) == 0.0, kw


# ===========================================================================
# THE NINTH NAME -- the union grid answered differently at 2 BLAS threads
# docs/audits/FIX_UNION_GRID_2THREAD_2026_08_06.md
# ===========================================================================

#: The M1 audit staircase (``test_m1_conditioning_guard.py``): six lossless
#: 60 nm slices whose walls shift 4 nm per slice, solved CONICALLY.  It is the
#: device the ninth name was referred on -- the SHARED (union) grid returned
#: ``J00`` = -0.27216-0.09245j with ``|R+T-1|`` = 21.35 at
#: ``OPENBLAS_NUM_THREADS=2`` and -0.17118+0.00907j with ``|R+T-1|`` =
#: 6.65e-06 at 1 and at 24, on BOTH mounts.
_STAIR_WL = 700e-9
_STAIR_PERIOD = 1.0e-6
_STAIR = [(60e-9, [(0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j),
                   (0.35 + 0.004 * i, 4.0 + 0j),
                   (0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j)])
          for i in range(6)]


def _stair_solve(grids, ffo=7, degree=6):
    st = PMMStack(_STAIR_PERIOD, n_substrate=1.5, n_superstrate=1.0,
                  degree=degree, far_field_orders=ffo, layer_grids=grids)
    for t, segs in _STAIR:
        st.add_layer(t, segments=segs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(_STAIR_WL, theta=0.15, phi=0.6).solve()
    R, T = np.asarray(R), np.asarray(T)
    return (complex(np.asarray(J)[0, 0]),
            float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0))))


def test_the_passivity_widening_is_a_null_floor_and_an_invariant():
    """The 2026-08-08 widening's two CONTRACTS, pinned on the selector itself
    over 400 random modal spectra -- no device, no BLAS, no environment.

    ``docs/audits/FIX_CI_ROUND2_PMM_2026_08_08.md`` S1.3.

      (1) THE NULL FLOOR.  With ``passive`` False -- a gain medium, an
          off-diagonal tensor, an unknown element payload -- the shipped
          selector is BIT-IDENTICAL to the 2026-08-06 mask.  Compared against
          a VERBATIM re-implementation of that mask, at tolerance 0.0, not by
          ``array_equal`` on the answer.  Measured 0 differing trials of 400.
      (2) THE INVARIANT.  With ``passive`` True the returned forward set
          CANNOT contain a growing mode, at any distance from the cut.
          Measured 0 of 400 -- while the same spectra leave one under the
          2026-08-06 mask in 399 of 400, which is what makes (2) a measurement
          rather than a tautology about the generator.

    Random spectra are the right instrument here precisely because the CI
    failures were about round-off landing where the dev box's does not: this
    samples the whole (flux, q, thr) plane instead of one build's corner of it.
    """
    rng = np.random.default_rng(7)

    def mask_2026_08_06(flux, q, thr, prop):
        """VERBATIM re-implementation of the shipped 2026-08-06 selector."""
        flip = np.where(prop, flux < 0.0, q.imag < 0.0)
        qf = np.where(flip, -q, q)
        lim = np.where(np.isfinite(thr), PC._MODE_CUT_MARGIN_WARN * thr, 0.0)
        bad = (prop & (np.abs(flux) < lim)
               & (qf.imag < -PC._MODE_GROWTH_REL * np.abs(qf)))
        return np.where(bad, q.imag < 0.0, flip)

    def grows(flip, q):
        qf = np.where(flip, -q, q)
        return bool((qf.imag < -PC._MODE_GROWTH_REL * np.abs(qf)).any())

    n_diff, n_grew_wide, n_grew_narrow = 0, 0, 0
    for _trial in range(400):
        n = 40
        flux = rng.standard_normal(n) * 10.0 ** rng.integers(-20, 2, n)
        q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        m = rng.random(n) < 0.5
        # a purely-imaginary q is the pathological population: zero z-power,
        # so its flux is round-off wherever the cut happens to sit.
        q[m] = 1j * rng.standard_normal(int(m.sum()))
        thr = float(10.0 ** rng.integers(-20, -2))
        prop = np.abs(flux) > thr
        narrow = PC._forward_growth_flip(flux, q, thr, prop, np, False)
        wide = PC._forward_growth_flip(flux, q, thr, prop, np, True)
        if not np.array_equal(narrow, mask_2026_08_06(flux, q, thr, prop)):
            n_diff += 1
        n_grew_wide += grows(wide, q)
        n_grew_narrow += grows(narrow, q)
    assert n_diff == 0, (                        # (1)
        f"{n_diff} of 400 spectra: the selector with passivity NOT PROVEN is "
        f"no longer bit-identical to the 2026-08-06 mask, so the widening has "
        f"stopped being a null floor for gain / off-diagonal / unknown grids")
    assert n_grew_wide == 0, (                   # (2)
        f"{n_grew_wide} of 400 spectra leave a GROWING mode in the forward "
        f"set with passive=True -- the invariant is supposed to hold by "
        f"construction there")
    assert n_grew_narrow > 100, (                # (2) is not a tautology
        f"only {n_grew_narrow} of 400 spectra grow under the 2026-08-06 mask "
        f"(measured 399), so this generator no longer reaches the population "
        f"the widening exists for and the claim above proves nothing")


#: Spectral degrees the staircase scan walks.  ``_stair_solve``'s OTHER knob,
#: ``ffo``, was measured INERT for this purpose -- 5 / 7 / 9 / 11 give a
#: bit-identical census on both mounts, because far-field orders shape the
#: OUTPUT orders, not the per-layer eigenproblem the selector reads.  Degree
#: does move it, because it changes which double roots ``q^2`` the union grid
#: carries and therefore where the ~1e-10 LAPACK splitting lands.
#:
#: MEASURED 2026-08-16, repair off, union grid, RAW growing modes in the
#: forward set (``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S12):
#:
#:     degree                    4    5    6    7    8
#:     W py3.14 / np2.4.4 / 1    0    0    1    6    3
#:     M py3.12 / np2.5.1 / 1    0    0    6    6    2
#:     py3.12 CI shard           -    -    0    -    -
#:
#: -- and the SHIPPED selector leaves 0 at every one of those cells on both
#: mounts.  Degree 6 is the M1 audit's own staircase and stays first so the
#: documented cell is scored before anything else.
#: Degrees 4 and 5 were MEASURED and EXCLUDED: they read 0 on BOTH mounts,
#: so they cost a union-grid solve each and can never contribute the
#: natural reproduction.  6/7/8 all manifest on both mounts, so the scan
#: still triples the (b) coverage the single-degree form had.
_STAIR_SCAN_DEGREES = (6, 7, 8)
#: ``_stair_solve``'s own default degree, which the observable arm (c)
#: reuses out of the scan instead of re-solving.  Asserted rather than
#: assumed so a change to either constant is a failure, not a silent
#: comparison of two different devices.
assert _stair_solve.__defaults__[1] in _STAIR_SCAN_DEGREES


def _stair_census(degree):
    """``(rows, J00, closure)`` for one union-grid staircase solve, with the
    T3-4 census spy armed.  Each row is
    ``(flux, q, thr, prop, site, passive)`` exactly as the selector saw it."""
    seen = []
    orig = PC._record_mode_cut

    def spy(flux, q, thr, prop, mats, site, patterned=None,
            flip=None, passive=None):
        seen.append((np.asarray(flux).copy(), np.asarray(q).copy(),
                     float(thr), np.asarray(prop).copy(), site,
                     PC._grid_is_passive(mats) if passive is None else passive))
        return orig(flux, q, thr, prop, mats, site, patterned,
                    flip=flip, passive=passive)

    PC._record_mode_cut = spy
    PC._MODE_CUT_CENSUS = []
    try:
        j, close = _stair_solve("shared", degree=degree)
    finally:
        PC._MODE_CUT_CENSUS = None
        PC._record_mode_cut = orig
    return seen, j, close


def _forward_grown(rows, flip_fn, near_only):
    """Modes the given selector leaves GROWING in the forward set, summed over
    every row.  ``near_only`` restores the 2026-08-06 ``|flux| < 10 thr``
    conjunct (used for the RAW count, which is what that reading meant)."""
    total = 0
    for flux, q, thr, prop, _site, _pas in rows:
        flip = flip_fn(flux, q, thr, prop)
        qf = np.where(flip, -q, q)
        bad = prop & (qf.imag < -PC._MODE_GROWTH_REL * np.abs(qf))
        if near_only:
            bad = bad & (np.abs(flux) < PC._MODE_CUT_MARGIN_WARN * thr)
        total += int(np.count_nonzero(bad))
    return total


def _raw_flip(flux, q, thr, prop):
    """The historical selector, verbatim: ``where(prop, flux < 0, Im q < 0)``."""
    return np.where(prop, flux < 0.0, q.imag < 0.0)


def _shipped_flip(flux, q, thr, prop):
    return PC._forward_growth_flip(flux, q, thr, prop, np, True)


def _engineer_null_flux_misfiling(rows):
    """Build the ninth name's input OUT OF THIS BUILD'S OWN modal rows.

    ``docs/audits/FIX_RUNNER_PINS_2_2026_08_15.md`` S12.  The defect is not a
    property of a fixture; it is a property of the RULE.  A ``prop``-flagged
    mode of a lossless layer that carries no z-power is scored on a ``flux``
    that is pure round-off, and the historical rule takes its DIRECTION from
    the SIGN of that round-off -- so whenever the sign points against
    ``Im(q)``, a growing mode enters the forward set.  WHICH member of a
    double root gets the unlucky sign is exactly what a LAPACK build chooses,
    which is why the natural manifestation is per-build (see
    ``_STAIR_SCAN_DEGREES``) and why it cannot be asserted as a universal.

    So the sign is INJECTED rather than waited for.  Everything else is this
    build's own solve: the real ``q`` spectrum, the real ``prop`` flags, the
    real ``thr`` from :func:`_mass_flux_threshold`.  Only the round-off flux
    of the modes that COULD grow is replaced, with a magnitude at half the
    cut's own warn margin -- i.e. squarely inside the band where the rule is
    provably reading noise, so both the RAW conjunct and the repair's own
    ``|flux| < margin * thr`` guard apply.

    Returns EVERY row that carries a candidate, injected -- so the
    demonstration is as strong as the device allows rather than resting on
    whichever row happened to come first -- or ``[]`` if this device has no
    propagating mode that either direction would grow, in which case the
    mechanism cannot be exercised on it at all and the caller says so.
    """
    out = []
    for flux, q, thr, prop, site, _pas in rows:
        if not np.isfinite(thr) or thr <= 0.0:
            continue
        rel = PC._MODE_GROWTH_REL * np.abs(q)
        grow_if_kept = q.imag < -rel            # forward = +q already grows
        grow_if_flipped = (-q).imag < -rel      # forward = -q would grow
        cand = prop & (grow_if_kept | grow_if_flipped)
        if not np.any(cand):
            continue
        eps = 0.5 * PC._MODE_CUT_MARGIN_WARN * thr
        # the RAW rule is ``flip = flux < 0`` and ``qf = -q if flip else q``,
        # so give each candidate the sign that makes the FORWARD pick the
        # growing one.
        f = np.array(flux, dtype=float, copy=True)
        f = np.where(cand & grow_if_kept, eps, f)
        f = np.where(cand & grow_if_flipped, -eps, f)
        out.append((f, q, thr, prop, f"injected:{site}", True))
    return out


def test_the_forward_set_cannot_grow_on_the_union_grid_conical_staircase():
    """THE NINTH NAME, pinned as an INVARIANT rather than as a number.

    The referred defect: this device's SHARED (union) grid returned a
    completely different answer at ``OPENBLAS_NUM_THREADS=2`` -- ``J00``
    moving 1.4e-01 and ``|R+T-1|`` going from 6.65e-06 to 21.35 -- at every
    truncation, on both mounts, with the per-layer path unaffected.

    The cause is NOT a null-space draw (the ``Hsup`` least squares that M1's
    ``_guarded_lstsq`` watches) and not conditioning.  It is the modal
    forward/backward CLASSIFICATION.  The union grid carries exactly-double
    roots ``q^2``; LAPACK splits each into a pair ~1e-10 apart whose ``sqrt``
    lands one member at ``Im(q) < 0``; both members are EVANESCENT and so
    carry exactly zero z-power, which means the ``flux`` the cut scores them
    on is pure round-off.  Whichever member's round-off happens to cross the
    cut is handed the FLUX-SIGN direction rule, and when that sign points
    against ``Im(q)`` a GROWING mode enters the forward set.  Measured on the
    substrate half-space row of this device, repair off::

        threads  n_growing (whole solve)  J00                       |R+T-1|
        1        1                        -0.17117932+0.00906676j   6.654e-06
        2        8                        -0.27216039-0.09244619j   2.135e+01
        24       1                        -0.17117932+0.00906676j   6.651e-06

    A test cannot vary the BLAS pool in-process, so what is pinned here is the
    INVARIANT the repair restores, which is thread-count-independent by
    construction and is checked on every modal row of the solve:

      (a) the RAW selector puts at least one growing mode in the forward set
          on this device -- the defect still reproduces (fail-before);
      (b) the REPAIRED selector puts none, on any row (the fix);
      (c) and the observable follows: the union grid agrees with the
          per-layer path, which is the cross-path oracle that was TRUE at 1
          thread and FALSE at 2 before the fix.

    **2026-08-08 (FIX_CI_ROUND2_PMM_2026_08_08).**  (b) is now scored WITHOUT
    the ``|flux| < 10 thr`` conjunct the 2026-08-06 form carried, i.e. against
    the invariant itself -- "no forward mode of a passive layer grows", at ANY
    distance from the cut -- because the cut's decade was measured to be the
    thing that let two cells through (M2's ns=2 ladder at degrees 18 and 20,
    where survivors sit at 15.8-23.6 x the cut).  The bar is REMOVED from the
    claim, not moved: this is the strictly stronger statement.  The RAW count
    (a) keeps the old conjunct so it still means what it meant.
    """
    # ---- the SCAN.  Every rung is scored, so (b) is now checked at five
    #      truncations instead of one; the RAW column is only REPORTED.
    census, raw_by_deg, obs = {}, {}, {}
    for deg in _STAIR_SCAN_DEGREES:
        rows, j_deg, close_deg = _stair_census(deg)
        obs[deg] = (j_deg, close_deg)
        assert rows, f"degree {deg}: the census spy never saw a modal solve"
        # the device is lossless, so every row must be recognised PASSIVE --
        # if it is not, the widened branch never runs and (b) proves nothing.
        assert all(r[5] for r in rows), (
            f"degree {deg}: a row of this lossless staircase was not "
            f"recognised as PASSIVE, so _forward_growth_flip's widened branch "
            f"was never exercised: {[(r[4], r[5]) for r in rows]}")
        census[deg] = rows
        raw_by_deg[deg] = _forward_grown(rows, _raw_flip, True)
        # (b) THE FIX, UNCONDITIONAL and at every truncation: a forward mode
        #     of a passive layer cannot grow along +z -- anywhere, not merely
        #     inside the cut's decade.
        fixed = _forward_grown(rows, _shipped_flip, False)
        assert fixed == 0, (
            f"degree {deg}: the repaired selector STILL leaves {fixed} "
            f"growing mode(s) in the forward set (raw: {raw_by_deg[deg]}) -- "
            f"_forward_growth_flip is supposed to make that impossible by "
            f"construction")
        # the instrument agrees with the raw count, so the guard's DIAGNOSIS
        # channel and the 2026-08-06 mask can never disagree about which modes
        # are affected ...
        assert raw_by_deg[deg] == sum(
            PC._mode_cut_growth(f, q, t, p)[0] for f, q, t, p, _s, _x in rows)
        # ... and the RESIDUAL instrument agrees with (b).
        assert 0 == sum(
            PC._mode_cut_growth_post(
                f, q, PC._forward_growth_flip(f, q, t, p, np, True))
            for f, q, t, p, _s, _x in rows)

    # ---- (a) THE FAIL-BEFORE, stated so it cannot depend on which member of
    #      a double root this build's round-off happens to put across the cut.
    #
    #      The historical form asserted ``raw >= 1`` on ONE truncation of ONE
    #      device.  A py3.12 CI shard read 0 there and failed -- correctly, in
    #      the sense that nothing was wrong: that build's eig simply handed the
    #      unlucky sign to nobody.  The natural manifestation is per-build (see
    #      the table at _STAIR_SCAN_DEGREES: 1 here, 6 on the other mount, 0 on
    #      the shard, same degree, same source), so it is REPORTED across the
    #      scan and ASSERTED only where it is build-free -- on an input this
    #      build's own solve supplies and this test signs.
    injected = _engineer_null_flux_misfiling(
        [r for deg in _STAIR_SCAN_DEGREES for r in census[deg]])
    assert injected, (
        "no propagating mode of this staircase would grow in EITHER "
        "direction, so the round-off band the ninth name lives in does not "
        "exist on this device at any scanned degree -- the fixture, not the "
        "selector, has changed")
    raw_i = _forward_grown(injected, _raw_flip, True)
    fixed_i = _forward_grown(injected, _shipped_flip, False)
    assert raw_i >= 1, (
        f"the RAW selector filed NO growing mode forward even with the "
        f"round-off flux signed against Im(q) across {len(injected)} of "
        f"this device's own modal rows -- the injector no longer reaches the "
        f"defect, so this fail-before is vacuous")
    assert fixed_i == 0, (
        f"the repaired selector left {fixed_i} growing mode(s) forward on the "
        f"injected round-off band -- the repair does not close the ninth name")

    # ... and the NATURAL reproduction, reported rather than required.  A
    # build that shows it at NO degree is a fact about that build's eig, not a
    # regression; the injected arm above is what keeps the claim alive there.
    natural = sorted(d for d, n in raw_by_deg.items() if n >= 1)
    print(f"[forward-set] raw growing modes by degree: {raw_by_deg} -- "
          f"natural reproduction at degree(s) {natural or 'none'}")

    # (c) reuses the SCAN's own degree-6 union solve -- ``_stair_solve``'s
    # default degree IS 6, so re-solving it would be the same numbers twice.
    j_union, close_union = obs[6]
    # (c) the observable: the two grid paths now agree.  Measured 3.89e-04
    #     relative at 1, 2 and 24 threads on both mounts; before the fix the
    #     same comparison read 8.36e-01 at 2 threads -- a 2150x separation,
    #     so the 1e-2 bar is not a calibration.  (The residual 3.89e-04 is
    #     the mortar's own non-conforming band, which M2 owns.)
    j_pl, close_pl = _stair_solve("per-layer")
    rel = abs(j_union - j_pl) / abs(j_pl)
    assert rel < 1e-2, (
        f"the union grid returns J00={j_union!r} and the per-layer path "
        f"{j_pl!r} ({rel:.3g} relative) on the same device -- the two grid "
        f"paths have stopped agreeing, which is the ninth name's observable")
    assert close_union < 1e-3, (
        f"the union-grid conical staircase closes at {close_union:.3e}: the "
        f"forward set is mis-assembled again (it read 21.35 at two BLAS "
        f"threads before the forward-growth repair)")
