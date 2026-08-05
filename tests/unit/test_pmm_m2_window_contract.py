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


def test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too():
    # N-6, the DECISIVE measurement, with the library default as the
    # fail-before.  On the audit-class taper at ns = 2 the per-layer and
    # shared grids are IDENTICAL (a 2-layer window is the full union), so this
    # is a statement about the discretisation, not about the mortar.
    #
    # At the default the 0.4127 nm adjacent-slice collision survives, and the
    # degree ladder is NOT stationary: it holds ~0.1106 to degree 10 and then
    # jumps to 0.0617 / 0.6234 -- while |R+T-1| stays ~1e-7, so ENERGY CLOSURE
    # CANNOT SEE IT (only order 0 propagates here, which makes R + T = 1
    # nearly tautological -- the standing caution).
    #
    # Snapping the collision away makes the same ladder stationary to 4e-4
    # relative across degree 6..14.
    degs = (6, 8, 10, 12, 14)
    broken = [solve0(build(2, degree=d))[0] for d in degs]
    fixed = [solve0(build(2, degree=d, min_feature=0.5e-9))[0] for d in degs]

    def spread(v):
        v = np.asarray(v, dtype=float)
        return float((v.max() - v.min()) / abs(v.mean()))

    # fail-before: the default is NOT stationary (measured spread ~1.4)
    assert spread(broken) > 0.5, (
        "the library-default ladder is expected to COLLAPSE on this device "
        "(the M2 fail-before).  If this now passes, the underlying "
        "sliver-mode defect has been fixed and this test must be re-pinned "
        "against the fix, not relaxed.")
    # and it is energy-clean while wrong -- the reason nothing caught it
    assert max(solve0(build(2, degree=d))[2] for d in (12, 14)) < 1e-5
    # after: stationary
    assert spread(fixed) < 5e-3
    assert all(abs(v - fixed[-1]) < 1e-3 for v in fixed)


def test_min_feature_threshold_rule_predicts_stationarity():
    # N-6, the QUANTITATIVE form of the contract, and the reason it is a rule
    # and not a heuristic.  Inside a +/-1 window the only cross-layer walls
    # are ADJACENT slices', so for a staircased taper with a conformal coat
    # ``c`` and per-slice offset ``off = (thickness/ns) tan(sidewall)`` the
    # window's cross-layer separations are EXACTLY ``{off, |c - off|}``.  The
    # contract is therefore
    #
    #     min_feature  >  min(off, |c - off|)
    #
    # and below that threshold the degree ladder collapses.  MEASURED to
    # predict stationary-vs-collapse correctly on every cell of
    # ns x min_feature = {2, 3, 6, 8} x {0.5, 1.5, 3.0} nm.  Two ns values are
    # pinned here (one each side of the 1.5 nm bar) to keep the suite quick;
    # the full 4 x 3 matrix is in the M2 audit.
    degs = (6, 8, 10, 12, 14)

    def spread(ns, mf):
        v = np.asarray([solve0(build(ns, degree=d, min_feature=mf))[0]
                        for d in degs], dtype=float)
        return float((v.max() - v.min()) / abs(v.mean()))

    # ns = 3: off = 3.6085, |c - off| = 1.3915 nm -> threshold 1.3915 nm
    assert spread(3, 0.5e-9) > 0.1, "0.5 nm is BELOW the ns=3 threshold"
    assert spread(3, 1.5e-9) < 1e-2, "1.5 nm is ABOVE the ns=3 threshold"
    # ns = 6: off = 1.8042, |c - off| = 3.1958 nm -> threshold 1.8042 nm
    assert spread(6, 1.5e-9) > 0.1, "1.5 nm is BELOW the ns=6 threshold"
    assert spread(6, 3.0e-9) < 2e-2, "3.0 nm is ABOVE the ns=6 threshold"


def test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper():
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

    def spread_u(ns, mf):
        v = np.asarray([solve0(_mk(_uncoated_layers(ns), d, 1)
                                if mf is None else
                                _mk_mf(_uncoated_layers(ns), d, mf))[0]
                        for d in degs], dtype=float)
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


def test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band():
    # T3-1, CONFOUND-CONTROLLED.  Measuring halfwidth on the coated device at
    # min_feature = 3.0 nm is NOT a window measurement: a +/-2 window holds
    # FIVE layers, so it snaps a different pair set than a +/-1 window and the
    # geometry changes too.  The tell was that max|dJ| read 1.169e-02 at BOTH
    # degree 6 and degree 8 -- a discretisation effect decays with degree, a
    # geometry difference does not.
    #
    # Run instead where the snap is PROVABLY inert (asserted below), so the
    # window is the only variable.  The envelope then behaves the way a
    # discretisation residual must: it DECAYS SPECTRALLY with degree, which is
    # the comparative form the repo prefers over an absolute bar.
    #
    # Measured, both builds, max over the healthy cells of two devices:
    #     degree 6 -> 2.7e-4     degree 8 -> 8.9e-5     degree 10 -> 6.9e-6
    for layers in (_uncoated_layers(3), _fatcoat_layers(8)):
        segs = [s for _t, s in layers]
        for hw in (1, 2):
            merged, _d = _snap_report(segs, (PERIOD * 1e-5) / PERIOD, hw)
            assert merged == 0, "the snap must be inert for this measurement"
        dJs = []
        for deg in (6, 8, 10):
            a = solve0(_mk(layers, deg, 1))
            b = solve0(_mk(layers, deg, 2))
            m = np.isin(a[4], b[4])
            mb = np.isin(b[4], a[4])
            dJ = float(np.max(np.abs(a[3] - b[3])))
            dR = float(np.max(np.abs(a[5][:, m] - b[5][:, mb])))
            assert dJ < 3e-4, f"deg={deg}: |dJ| = {dJ:.3e}"
            assert dR < 3e-4, f"deg={deg}: |dR| = {dR:.3e}"
            dJs.append(dJ)
        # the window residual is a DISCRETISATION residual: it must decay
        assert dJs[1] < dJs[0] and dJs[2] < dJs[1], (
            f"window residual did not decay with degree: {dJs}")


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
