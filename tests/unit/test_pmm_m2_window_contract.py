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


@pytest.fixture
def growth_repair_off():
    """The library's fail-before switch for ``_forward_growth_flip``, restored
    on the way out.  ``False`` is the pre-fix selector, bit for bit."""
    prev = PC.PMM_FORWARD_GROWTH_REPAIR
    PC.PMM_FORWARD_GROWTH_REPAIR = False
    try:
        yield
    finally:
        PC.PMM_FORWARD_GROWTH_REPAIR = prev


def spread(v):
    """Peak-to-peak of a degree ladder, relative to its own mean."""
    v = np.asarray(v, dtype=float)
    return float((v.max() - v.min()) / abs(v.mean()))


#: Ladder cache, keyed by ``(ns, min_feature, degrees, repair_flag)``.  The
#: three re-pinned tests below each score the SAME ladders with the switch off
#: and on, so without this the file re-solves every cell four or five times.
#: The repair flag is part of the key because it is exactly what the answer
#: depends on -- a cache that ignored it would silently fake the null control.
_LADDER_CACHE = {}


def coated_ladder(ns, mf, degrees=_LADDER_DEGREES):
    """Order-0 reflectance over the degree ladder on the coated taper."""
    key = (ns, mf, degrees, bool(PC.PMM_FORWARD_GROWTH_REPAIR))
    hit = _LADDER_CACHE.get(key)
    if hit is None:
        hit = np.array([solve0(build(ns, degree=d, min_feature=mf))[0]
                        for d in degrees], dtype=float)
        hit.flags.writeable = False
        _LADDER_CACHE[key] = hit
    return hit


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
      (5) the above-threshold cells are BIT-IDENTICAL either way -- the null
          floor, and the proof the repair is not a global re-tune.
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

    # ---- (5) THE NULL FLOOR: cured cells do not move a bit ----------------
    for ns, mf in ABOVE_THRESHOLD:
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        a = coated_ladder(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = False
        b = coated_ladder(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        assert float(np.max(np.abs(a - b))) == 0.0, (
            f"ns={ns} min_feature={mf}: an ABOVE-threshold cell moved when "
            f"the forward-growth repair was switched -- the repair is only "
            f"allowed to touch solves that put a growing mode in the forward "
            f"set, and a cured cell has none")


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
    for ns, mf in ((3, 1.5e-9), (6, 3.0e-9)):
        a = coated_ladder(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = False
        b = coated_ladder(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        assert float(np.max(np.abs(a - b))) == 0.0, (
            f"ns={ns} min_feature={mf}: an above-threshold cell moved when "
            f"the repair was switched")
    # the BELOW-threshold side no longer collapses (measured 3.79e-03 and
    # 4.24e-03 against the 0.1 the rule predicts without the repair -- a
    # 25x-fold separation, at every thread count on both mounts).
    for ns, mf in ((3, 0.5e-9), (6, 1.5e-9)):
        s = spread(coated_ladder(ns, mf))
        assert s < 1e-2, (
            f"ns={ns} min_feature={mf}: still spreads {s:.4g} with the "
            f"forward-growth repair on -- below the min_feature threshold is "
            f"supposed to have stopped being an accuracy cliff")


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
    degs = (8, 10, 12, 14, 16)

    cache = {}

    def spread_u(ns, mf):
        # cached on the repair flag too -- see ``_LADDER_CACHE``
        key = (ns, mf, bool(PC.PMM_FORWARD_GROWTH_REPAIR))
        v = cache.get(key)
        if v is None:
            v = np.asarray([solve0(_mk(_uncoated_layers(ns), d, 1)
                                    if mf is None else
                                    _mk_mf(_uncoated_layers(ns), d, mf))[0]
                            for d in degs], dtype=float)
            cache[key] = v
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
    for ns, mf in ((6, 3.0e-9), (12, 1.5e-9)):
        _s_on, v_on = spread_u(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = False
        _s_off, v_off = spread_u(ns, mf)
        PC.PMM_FORWARD_GROWTH_REPAIR = True
        assert float(np.max(np.abs(np.asarray(v_on)
                                   - np.asarray(v_off)))) == 0.0, (
            f"uncoated ns={ns} min_feature={mf}: a cured cell moved when the "
            f"forward-growth repair was switched")
    # (b) the BELOW-threshold cell stops collapsing (measured 4.06e-03 with
    #     the repair on against 1.34 with it off, both thread counts).
    s_lo_on, _v = spread_u(6, 1.5e-9)
    assert s_lo_on < 1e-2, (
        f"uncoated ns=6 at 1.5 nm still spreads {s_lo_on:.4g} with the "
        f"forward-growth repair on (it spread {s_lo:.4g} with it off)")


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
    """
    seen = []
    orig = PC._record_mode_cut

    def spy(flux, q, thr, prop, mats, site, patterned=None):
        seen.append((np.asarray(flux).copy(), np.asarray(q).copy(),
                     float(thr), np.asarray(prop).copy(), site))
        return orig(flux, q, thr, prop, mats, site, patterned)

    PC._record_mode_cut = spy
    PC._MODE_CUT_CENSUS = []
    try:
        j_union, close_union = _stair_solve("shared")
    finally:
        PC._MODE_CUT_CENSUS = None
        PC._record_mode_cut = orig
    assert seen, "the census spy never saw a modal solve: nothing was measured"

    def grown(flip_fn):
        """Modes the given selector leaves GROWING in the forward set, over
        every row of the solve -- scored with the SHIPPED bars."""
        total = 0
        for flux, q, thr, prop, _site in seen:
            flip = flip_fn(flux, q, thr, prop)
            qf = np.where(flip, -q, q)
            near = np.abs(flux) < PC._MODE_CUT_MARGIN_WARN * thr
            total += int(np.count_nonzero(
                prop & near
                & (qf.imag < -PC._MODE_GROWTH_REL * np.abs(qf))))
        return total

    raw = grown(lambda f, q, t, p: np.where(p, f < 0.0, q.imag < 0.0))
    fixed = grown(lambda f, q, t, p: PC._forward_growth_flip(f, q, t, p))
    # (a) the defect still reproduces: the raw selector grows something here
    assert raw >= 1, (
        "the RAW selector puts NO growing mode in the forward set on the M1 "
        "audit staircase, so the ninth name no longer reproduces on this "
        "build and this test has stopped being a fail-before.  Re-pin it "
        "against whatever changed rather than deleting it.")
    # (b) THE FIX: a forward mode of a passive layer cannot grow along +z
    assert fixed == 0, (
        f"the repaired selector STILL leaves {fixed} growing mode(s) in the "
        f"forward set (raw: {raw}) -- _forward_growth_flip is supposed to "
        f"make that impossible by construction")
    # ... and the instrument agrees with the raw count, so the guard and the
    # repair can never disagree about which modes are affected.
    assert raw == sum(PC._mode_cut_growth(f, q, t, p)[0]
                      for f, q, t, p, _s in seen)

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
