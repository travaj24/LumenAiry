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
    OpenBLAS kernel is entitled to move."""
    global _CUT_SCALE
    orig, prev = PC._mass_flux_threshold, _CUT_SCALE

    def scaled(flux, W2, SVt, SVb, n, xp=np):
        return orig(flux, W2, SVt, SVb, n, xp) * scale

    PC._mass_flux_threshold = scaled
    _CUT_SCALE = float(scale)
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
    separations are EXACTLY ``{off, |c - off|}``.  The contract is

        min_feature  >  min(off, |c - off|)

    and below that threshold the PRE-REPAIR degree ladder collapses.  MEASURED
    to predict stationary-vs-collapse correctly on every cell of
    ns x min_feature = {2, 3, 6, 8} x {0.5, 1.5, 3.0} nm.  Two ns values are
    pinned here (one each side of the 1.5 nm bar) to keep the suite quick; the
    full 4 x 3 matrix is in the M2 audit.

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
    # ns = 3: off = 3.6085, |c - off| = 1.3915 nm -> threshold 1.3915 nm
    assert spread(coated_ladder(3, 0.5e-9)) > 0.1, \
        "0.5 nm is BELOW the ns=3 threshold"
    assert spread(coated_ladder(3, 1.5e-9)) < 1e-2, \
        "1.5 nm is ABOVE the ns=3 threshold"
    # ns = 6: off = 1.8042, |c - off| = 3.1958 nm -> threshold 1.8042 nm
    assert spread(coated_ladder(6, 1.5e-9)) > 0.1, \
        "1.5 nm is BELOW the ns=6 threshold"
    assert spread(coated_ladder(6, 3.0e-9)) < 2e-2, \
        "3.0 nm is ABOVE the ns=6 threshold"

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

    # ns = 6, off = 1.804 nm: 1.5 nm is BELOW the threshold, 3.0 nm above it
    s_lo, _ = spread_u(6, 1.5e-9)
    s_hi, v_hi = spread_u(6, 3.0e-9)
    assert s_lo > 0.1, f"1.5 nm should NOT cure ns=6 (off=1.804 nm): {s_lo:.3g}"
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
        j_union, close_union = _stair_solve("shared")
    finally:
        PC._MODE_CUT_CENSUS = None
        PC._record_mode_cut = orig
    assert seen, "the census spy never saw a modal solve: nothing was measured"
    # the device is lossless, so every row must be recognised PASSIVE -- if it
    # is not, the widened branch never runs and (b) below proves nothing.
    assert all(row[5] for row in seen), (
        "a row of this lossless staircase was not recognised as PASSIVE, so "
        "_forward_growth_flip's widened branch was never exercised: "
        f"{[(r[4], r[5]) for r in seen]}")

    def grown(flip_fn, near_only):
        """Modes the given selector leaves GROWING in the forward set, over
        every row of the solve.  ``near_only`` restores the 2026-08-06
        conjunct (used for the RAW count, which is what that reading meant)."""
        total = 0
        for flux, q, thr, prop, _site, _pas in seen:
            flip = flip_fn(flux, q, thr, prop)
            qf = np.where(flip, -q, q)
            bad = prop & (qf.imag < -PC._MODE_GROWTH_REL * np.abs(qf))
            if near_only:
                bad = bad & (np.abs(flux) < PC._MODE_CUT_MARGIN_WARN * thr)
            total += int(np.count_nonzero(bad))
        return total

    raw = grown(lambda f, q, t, p: np.where(p, f < 0.0, q.imag < 0.0), True)
    fixed = grown(
        lambda f, q, t, p: PC._forward_growth_flip(f, q, t, p, np, True), False)
    # (a) the defect still reproduces: the raw selector grows something here
    assert raw >= 1, (
        "the RAW selector puts NO growing mode in the forward set on the M1 "
        "audit staircase, so the ninth name no longer reproduces on this "
        "build and this test has stopped being a fail-before.  Re-pin it "
        "against whatever changed rather than deleting it.")
    # (b) THE FIX: a forward mode of a passive layer cannot grow along +z --
    #     anywhere, not merely inside the cut's decade.
    assert fixed == 0, (
        f"the repaired selector STILL leaves {fixed} growing mode(s) in the "
        f"forward set (raw: {raw}) -- _forward_growth_flip is supposed to "
        f"make that impossible by construction")
    # ... and the instrument agrees with the raw count, so the guard's
    # DIAGNOSIS channel and the 2026-08-06 mask can never disagree about which
    # modes are affected.
    assert raw == sum(PC._mode_cut_growth(f, q, t, p)[0]
                      for f, q, t, p, _s, _pas in seen)
    # ... and the RESIDUAL instrument the guard now speaks on agrees with (b):
    # what the shipped selector leaves is what _mode_cut_growth_post reports.
    assert 0 == sum(
        PC._mode_cut_growth_post(
            f, q, PC._forward_growth_flip(f, q, t, p, np, True))
        for f, q, t, p, _s, _pas in seen)

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
