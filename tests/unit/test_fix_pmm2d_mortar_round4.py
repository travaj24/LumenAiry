"""ROUND 4 of the PURE staggered 2-D PMM per-layer-grid (L2 mortar) work --
the two P3 defects and the two durability flags of
``docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md``, fixed and gated.

Fix doc: ``docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md``.

TERMS, before they are used.

* **MORTAR** -- the L2 projection that couples two adjacent per-layer element
  grids.  It exists on an interface only where the two layers' wall arrays
  DIFFER; where they coincide the interface is the plain square modal match.
* **A MORTARED AXIS** -- an axis (x or y) on which some adjacent pair of layers
  carries DIFFERENT wall arrays.  :meth:`StagGridOps.key` is the fingerprint
  PAIR ``(x, y)``, so the x axis is mortared iff the stack shows more than one
  x fingerprint; and if it does, some ADJACENT pair differs on x, because a
  sequence whose entries are not all equal has two neighbours that are not.
* **THE BAND** -- ``[_STAG_MIN_SEG_FRAC, _STAG_SLIVER_BAND_FRAC)`` = 1e-3 ..
  3e-2 of the period.  Below it a stack is REFUSED by the width contract;
  inside it the solve proceeds and WARNS; above it it is silent.
* **PROMOTION** -- ``_modes_as_general`` rewrites a symmetric IN-PLANE region's
  modes as ``(W, V, lam, W, -V, -lam)`` so they can enter the generalized
  cascade an out-of-plane or slanted layer forces.  An interface is ASYMMETRIC
  when EXACTLY ONE of its sides is promoted.

WHAT IS GATED.

* **DEFECT 1** (wording) round 3 said the generalized mortar operand is
  rank-deficient "whenever either side is an in-plane region promoted".  It
  needs the interface to be ASYMMETRIC; with BOTH sides promoted the operand is
  healthy.  Gated as a MEASUREMENT (the near-null direction is not localised)
  and as a WORDING check on the string a user is actually shown.
* **DEFECT 2** (false positive) the band warning scanned BOTH axes of every
  grid while asking about the STACK, so a stack whose layers differ on x and
  share the y wall array warned about a narrow y segment.  Gated two-sidedly:
  the false positive stops, every warning that fired for a real reason fires
  identically, and the answer is BIT-IDENTICAL across the change.
* **S14** the two SAMPLE-scoped bars in the round-3 gate file are restated
  there, family-scoped; this file gates the population they are restated from.

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS**
(2026-09-11), both readings stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

Per ``docs/TESTING_STANDARDS.md`` every population is RE-MEASURED on the
running build rather than pinned, and the new behaviour carries a FAIL-BEFORE
arm: :func:`_round3_rule` restores round 3's collapsed stack-level condition
IN PROCESS, so both arms of every comparison run in one interpreter with
everything but the change held fixed.
"""
import os

# The band comparisons below read twelve figures of a reflectance; pin ONE
# BLAS thread before numpy is imported.  (In a MULTI-FILE run this is a no-op
# -- numpy is already imported -- so the shell must set them too.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import pathlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as _sp  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C, StagGridOps  # noqa: E402

# ---- the VERIFICATION's own DEFECT-2 fixture, knob for knob ---------------
# (VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md S8 DEFECT 2 /
# validation/probe_verify_mortar_round3/v5_falsepos.py fp2).  DIMENSIONLESS:
# the period and the wavelength are in the same units, which is all that
# matters -- do not drive it with a metre wavelength.
_P, _WL, _TH = 1.07, 0.79, 0.31
_YC = 0.585


# ==========================================================================
# helpers
# ==========================================================================
@contextlib.contextmanager
def _round3_rule():
    """FAIL-BEFORE arm: round 3's condition, restored in process.

    Round 3 called ``_warn_stag_sliver_band(gof, force_mortar or
    len({g.key() for g in gof}) > 1)`` and then scanned BOTH axes.  The key is
    the fingerprint PAIR, so ``len({key}) > 1`` is exactly "at least one axis
    differs"; collapsing the per-axis helper to that reproduces round 3
    EXACTLY, in the same interpreter as the round-4 arm."""
    orig = _sp._stag_mortared_axes

    def collapsed(grids, force=False):
        m = bool(force) or len({g.key() for g in grids}) > 1
        return (m, m)

    _sp._stag_mortared_axes = collapsed
    try:
        yield
    finally:
        _sp._stag_mortared_axes = orig


def _solved(st):
    """Solve, hashing the answer and recording the band warnings the solve
    raises -- the axis and the width each one names."""
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    h = hashlib.sha256()
    for arr in (o, R, T):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    band = [w for w in ws if issubclass(w.category, UserWarning)
            and "degradation band" in str(w.message)]
    k = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
    axes, widths = [], []
    for w in band:
        m = str(w.message)
        axes.append("x" if " on the x axis" in m
                    else "y" if " on the y axis" in m else None)
        widths.append(float(m.split("has a segment ", 1)[1]
                            .split(" of the period", 1)[0]))
    return {"hash": h.hexdigest(), "R00": float(R[1, k]),
            "n_warn": len(band), "axes": axes, "widths": widths,
            "closure": abs(float(R[1].sum() + T[1].sum()) - 1.0)}


def _both_arms(build):
    """``(round3, round4)`` readings of the SAME stack, built twice so neither
    arm can see the other's state."""
    with _round3_rule():
        pre = _solved(build())
    return pre, _solved(build())


def _uniform_xy_differ(fy, M=5, *, xconf=False):
    """The DEFECT-2 fixture: UNIFORM permittivity in both layers, layers
    differing on x and sharing the y wall array EXACTLY.  The y walls sit
    inside a region of constant permittivity, so the DEVICE cannot depend on
    ``fy`` at all and every deviation is numerical."""
    yws = [(_YC - fy / 2) * _P, (_YC + fy / 2) * _P]
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(2.5)),
                 x_walls=[0.25 * _P, 0.60 * _P], y_walls=yws)
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(3.5)),
                 x_walls=([0.25 * _P, 0.60 * _P] if xconf
                          else [0.31 * _P, 0.66 * _P]),
                 y_walls=yws)
    st.set_source(_WL, theta=_TH, phi=0.0)
    return st


def _closing_taper(n_slices, M=4):
    """A taper whose tip CLOSES, sliced by the midpoint rule the shipped
    builder uses: slice ``i`` of ``n`` carries a pillar ``w_bottom (1 - (i +
    0.5) / n)`` wide, so the narrowest sampled width is ``w_bottom / (2 n)``.
    Every slice is its OWN grid, so x is mortared; y is shared and ordinary."""
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    yws = [0.27 * _P, 0.61 * _P]
    for i in range(n_slices):
        w = 0.5 * (1.0 - (i + 0.5) / n_slices)
        tile = np.full((3, 3), _C(2.1))
        tile[1, 1] = _C(6.0)
        st.add_layer(0.04, eps_cell=tile,
                     x_walls=[(0.5 - w / 2) * _P, (0.5 + w / 2) * _P],
                     y_walls=yws)
    st.set_source(_WL, theta=_TH, phi=0.0)
    return st


def _delta_sweep(delta, M=4):
    """The ROUND-2 sweep (``test_the_mortars_own_algebra_is_exact_at_every_
    wall_separation``): three ALL-HOST layers whose x walls DIFFER, the middle
    one carrying the sliver, on a SHARED y wall array.

    ``M`` = 4 rather than the round-2 gate's 5: what is asserted here is
    GEOMETRIC -- which axis warns and at what width -- and the band helper
    reads wall arrays, not modes.  The round-2 gate itself still runs `M` = 5
    and still raises this warning; that is checked by the suite run, not
    here."""
    st = PMM2DStackPure(1.2, n_modes=M, n_orders=1, layer_grids="per-layer")
    yw = [0.27 * 1.2, 0.61 * 1.2]
    for xw in ([0.21 * 1.2, 0.68 * 1.2],
               [(0.5 - delta / 2) * 1.2, (0.5 + delta / 2) * 1.2],
               [0.33 * 1.2, 0.79 * 1.2]):
        st.add_layer(0.06, eps=2.25, x_walls=xw, y_walls=yw)
    st.set_source(0.85, theta=0.23, phi=0.0)
    return st


def _mixed(fx, fy, *, share_y=True, M=4):
    """x walls DIFFER and carry a segment ``fx`` wide; y walls are IDENTICAL
    in both layers and carry a NARROWER segment ``fy``.  The global narrowest
    is on y; the axis that carries a cross-grid projection is x."""
    yws = [(_YC - fy / 2) * _P, (_YC + fy / 2) * _P]
    yws2 = yws if share_y else [(0.40 - fy / 2) * _P, (0.40 + fy / 2) * _P]
    tile = np.full((3, 3), _C(2.1))
    tile[1, 1] = _C(6.0)
    st = PMM2DStackPure(_P, n_modes=M, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.10, eps_cell=tile,
                 x_walls=[(0.44 - fx / 2) * _P, (0.44 + fx / 2) * _P],
                 y_walls=yws)
    st.add_layer(0.10, eps_cell=tile,
                 x_walls=[0.31 * _P, 0.66 * _P], y_walls=yws2)
    st.set_source(_WL, theta=_TH, phi=0.0)
    return st


def _grids_of(st, M=4):
    return [StagGridOps(st.period_x, st.period_y, L["wx"], L["wy"], M,
                        1.0 + 0j, 1.0 + 0j) for L in st._layers]


def _narrow(grids, axes):
    return _ts._stag_band_narrowest(grids, axes)


def _round2_module():
    """The ROUND-2 gate file's geometry battery, loaded BY PATH.

    ``import test_fix_pmm2d_mortar_round2`` works only when pytest happens to
    have put ``tests/unit`` on ``sys.path``, which it does not when this file
    is run alone.  Loading it by path keeps the census reading the SAME
    battery the round-2 and round-3 gates do."""
    import importlib.util  # noqa: PLC0415

    path = pathlib.Path(__file__).with_name("test_fix_pmm2d_mortar_round2.py")
    spec = importlib.util.spec_from_file_location("_pmm2d_r2_gate_r4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _census_battery():
    """The ORDINARY per-layer geometries the library builds, through the
    PUBLIC builders: the round-2 battery plus the two classes the VERIFY
    round-3 census found closer to the band's edge than anything in it -- a
    high-duty pillar and a fine uniform lattice."""
    out = dict(_round2_module()._shipped_geometry_battery())
    for name, duty in (("duty_0.9_pillar", 0.9), ("duty_0.7_pillar", 0.7)):
        a = 0.5 - duty / 2
        st = PMM2DStackPure(1.2, n_modes=4, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.10, eps_cell=np.full((3, 3), _C(6.0)),
                     x_walls=[a * 1.2, (a + duty) * 1.2],
                     y_walls=[a * 1.2, (a + duty) * 1.2])
        st.add_layer(0.10, eps=2.0)
        out[name] = st
    st = PMM2DStackPure(1.2, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.10, eps_cell=np.full((16, 16), _C(6.0)))
    st.add_layer(0.10, eps=2.0)
    out["uniform_lattice_N16"] = st
    return out


# ==========================================================================
# DEFECT 2 -- the false positive stops, and the answer does not move
# ==========================================================================
def test_a_non_mortared_axis_no_longer_raises_the_band_warning():
    """DEFECT 2, the FALSE-POSITIVE half, two-sided against round 3's own rule.

    The stack's two layers differ on x and share the y wall array EXACTLY, and
    the y walls sit inside a region of constant permittivity.  Round 3 warned
    about the narrow y segment; there is no cross-grid projection on y, so
    there is nothing there to warn about.

    THE ANSWER IS THE EVIDENCE, and it is asserted twice.  (1) The two arms
    return a BIT-IDENTICAL answer -- the warning is a pure observer, so the
    change cannot move a shipped number.  (2) On this device the band-width y
    segment and an ORDINARY (3e-1) one give the SAME reflectance: MEASURED
    2026-09-11, ``R00`` = **0.247088457739** at both, i.e. the band/ordinary
    ratio is **1.000000000000 to twelve places**, a relative move of
    **1.5322e-13**, against the "about 4-5x the error" the message claimed.
    Over the full 2.5-decade ladder 3e-1 .. 1.2e-3 the move stays at
    1.7e-14 .. 8.5e-13 (WIN) and 6.4e-15 .. 1.3e-13 (WSL)
    (``validation/probe_fix_mortar_round4/p1_axis_band_{win,wsl}.json``).
    Asserted at 1e-6, which is seven decades over the measurement and five
    decades under the claim.
    """
    pre_wide, post_wide = _both_arms(lambda: _uniform_xy_differ(3.0e-1))
    pre_band, post_band = _both_arms(lambda: _uniform_xy_differ(9.0e-3))

    # (1) round 3 warned on the band rung and named the axis with no mortar
    assert pre_wide["n_warn"] == 0, pre_wide
    assert pre_band["n_warn"] == 1 and pre_band["axes"] == ["y"], pre_band
    # (2) round 4 is silent on BOTH -- the DEFECT is gone
    assert post_wide["n_warn"] == 0, post_wide
    assert post_band["n_warn"] == 0, post_band
    # (3) and the answers are BIT-IDENTICAL across the change, both rungs
    assert pre_wide["hash"] == post_wide["hash"]
    assert pre_band["hash"] == post_band["hash"]
    # (4) there was nothing to warn about: the band rung and the ordinary rung
    # are the same answer.  DECISION: "the device does not depend on this
    # width".  MEASURED 1.5322e-13 relative (WIN), and 6.4e-15 .. 1.3e-13 over
    # the whole 2.5-decade ladder on WSL; asserted at 1e-6, seven decades over
    # the measurement.  The closure at this rung reads 4.855e-10.
    rel = abs(post_band["R00"] - post_wide["R00"]) / abs(post_wide["R00"])
    assert rel < 1e-6, (rel, post_wide["R00"], post_band["R00"])
    assert post_band["closure"] < 1e-4, post_band["closure"]
    # (5) a stack conforming on BOTH axes was silent before and after -- the
    # round-3 behaviour this must not disturb
    # (M = 4 here: the claim is "no mortar anywhere -> no warning", which is a
    # geometry reading, and M = 5 costs 5x for nothing)
    pre_c, post_c = _both_arms(
        lambda: _uniform_xy_differ(9.0e-3, 4, xconf=True))
    assert pre_c["n_warn"] == 0 and post_c["n_warn"] == 0, (pre_c, post_c)
    assert pre_c["hash"] == post_c["hash"]


def test_every_band_warning_that_fired_for_a_real_reason_still_fires():
    """DEFECT 2, the TRUE-POSITIVE half.  A fix that silences the warning
    everywhere would pass the test above; this is the other side of it.

    Both surfaces the round-3 census identifies as reaching the band carry
    their narrow segment on the axis that IS mortared:

      * a taper whose tip CLOSES -- every slice is its own grid, so x is
        mortared, and the narrowest sampled width is ``w_bottom / (2 n)``.
        MEASURED: 8 slices is 3.1250e-02, 1.04x OUTSIDE the edge and silent;
        9 slices is 2.7778e-02, inside it and warning on the x axis;
      * the ROUND-2 ``delta`` sweep, whose middle layer's x walls close to
        1e-2 of the period on a SHARED y array.

    Each is asserted against round 3's own rule as well as round 4's: the
    count, the AXIS and the WIDTH must be identical, and so must the answer.
    """
    edge = _ts._STAG_SLIVER_BAND_FRAC
    for n, expect in ((8, 0), (9, 1)):
        pre, post = _both_arms(lambda n=n: _closing_taper(n))
        assert post["n_warn"] == expect, (n, post)
        assert (pre["n_warn"], pre["axes"], pre["widths"]) == \
               (post["n_warn"], post["axes"], post["widths"]), (n, pre, post)
        assert pre["hash"] == post["hash"], n
        if expect:
            assert post["axes"] == ["x"], (n, post)
            # the width is the midpoint rule's, DERIVED rather than pinned
            # ... at the precision the message PRINTS (``{frac:.3e}``, four
            # significant figures), not at machine precision
            pred = 0.5 / (2 * n)
            assert abs(post["widths"][0] - pred) < 1e-3 * pred, post
            assert _ts._STAG_MIN_SEG_FRAC <= post["widths"][0] < edge, post
    # ... and 8 slices is only 1.04x outside, so the silence is a decision the
    # constant makes, not a wide margin
    assert 1.0 < (0.5 / 16.0) / edge < 1.1, edge

    for delta, expect in ((0.30, 0), (1.0e-2, 1)):
        pre, post = _both_arms(lambda d=delta: _delta_sweep(d))
        assert post["n_warn"] == expect, (delta, post)
        assert (pre["n_warn"], pre["axes"], pre["widths"]) == \
               (post["n_warn"], post["axes"], post["widths"]), (delta, pre,
                                                                post)
        assert pre["hash"] == post["hash"], delta
        if expect:
            assert post["axes"] == ["x"], (delta, post)
            assert abs(post["widths"][0] - delta) < 1e-3 * delta, post


def test_a_stack_narrow_on_both_axes_names_the_axis_that_carries_the_mortar():
    """DEFECT 2, the MIXED case -- the one that shows the fix selects an axis
    rather than merely suppressing the warning.

    The stack is narrow on BOTH axes: 9e-3 on x, where the two layers' wall
    arrays DIFFER, and 3e-3 on y, where they are identical.  The global
    narrowest is the y one, so round 3 named y at 3.000e-03; the axis that
    carries a cross-grid projection is x, and round 4 names x at 9.000e-03.
    ONE warning either way -- the warning is per SOLVE, not per axis.

    The CONTROL is the same two widths with the y arrays made to differ too:
    then y really is mortared, it really is the narrowest, and BOTH rules name
    it.  Without that arm "names x" would be indistinguishable from "always
    names x"."""
    pre, post = _both_arms(lambda: _mixed(9.0e-3, 3.0e-3))
    assert pre["n_warn"] == 1 and pre["axes"] == ["y"], pre
    assert abs(pre["widths"][0] - 3.0e-3) < 1e-6, pre
    assert post["n_warn"] == 1 and post["axes"] == ["x"], post
    assert abs(post["widths"][0] - 9.0e-3) < 1e-6, post
    assert pre["hash"] == post["hash"]

    # CONTROL: y mortared as well -> the narrowest is the right answer, and
    # both rules give it
    pre2, post2 = _both_arms(lambda: _mixed(9.0e-3, 3.0e-3, share_y=False))
    assert pre2["axes"] == ["y"] and post2["axes"] == ["y"], (pre2, post2)
    assert abs(post2["widths"][0] - 3.0e-3) < 1e-6, post2
    assert pre2["hash"] == post2["hash"]

    # and with x ORDINARY the stack is silent under round 4 while round 3
    # warned about the conforming y -- the false positive again, on a stack
    # whose mortared axis is nowhere near the band
    pre3, post3 = _both_arms(lambda: _mixed(2.0e-1, 3.0e-3))
    assert pre3["n_warn"] == 1 and pre3["axes"] == ["y"], pre3
    assert post3["n_warn"] == 0, post3
    assert pre3["hash"] == post3["hash"]


def test_the_mortared_axis_reading_is_two_sided_and_per_axis():
    """The DECISION :func:`_stag_mortared_axes` takes, on all four corners
    plus the test instrument, read off the wall arrays with no solve.

    It is the whole content of the fix: an axis counts when adjacent layers'
    wall arrays on it DIFFER, and conformity is per axis."""
    wa = [0.25 * _P, 0.60 * _P]
    wb = [0.31 * _P, 0.66 * _P]

    def _st(x0, y0, x1, y1):
        st = PMM2DStackPure(_P, n_modes=4, n_orders=1,
                            layer_grids="per-layer")
        for xw, yw in ((x0, y0), (x1, y1)):
            st.add_layer(0.10, eps_cell=np.full((3, 3), _C(2.5)),
                         x_walls=xw, y_walls=yw)
        return st

    cases = {
        (False, False): _st(wa, wa, wa, wa),      # fully conforming
        (True, False): _st(wa, wa, wb, wa),       # x differs only
        (False, True): _st(wa, wa, wa, wb),       # y differs only
        (True, True): _st(wa, wa, wb, wb),        # both differ
    }
    for want, st in cases.items():
        got = _ts._stag_mortared_axes(_grids_of(st))
        assert got == want, (want, got)
        # ... and the ``force_mortar`` test instrument makes both axes live,
        # because it drives the mortar algebra where the grids coincide
        assert _ts._stag_mortared_axes(_grids_of(st), True) == (True, True)

    # a THREE-layer stack whose ends agree on x and whose middle does not:
    # the set-of-fingerprints reading and "some ADJACENT pair differs" are the
    # same statement, and this is the case that could separate them
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, layer_grids="per-layer")
    for xw in (wa, wb, wa):
        st.add_layer(0.10, eps_cell=np.full((3, 3), _C(2.5)),
                     x_walls=xw, y_walls=wa)
    assert _ts._stag_mortared_axes(_grids_of(st)) == (True, False)

    # the narrowest-segment search follows the axes it is given, and skipping
    # every axis returns "nothing to score" rather than a spurious 1.0 hit
    g = _grids_of(_st(wa, [0.50 * _P, 0.51 * _P], wb, [0.50 * _P, 0.51 * _P]))
    assert _narrow(g, (True, True))[1][1] == "y"
    assert _narrow(g, (True, False))[1][1] == "x"
    assert _narrow(g, (False, False)) == (1.0, None)


def test_no_ordinary_geometry_lands_in_the_band_on_a_mortared_axis():
    """The FALSE-POSITIVE census, per AXIS, and the SCOPE the round-3 gate
    states as a family claim over a sample (VERIFY round 3 S11.1, S14).

    Two things are asserted, and the second is the one round 3 could not say.

    (1) The per-axis reading can only ever be WIDER than the round-3 reading --
    it searches a SUBSET of the axes -- so the change cannot ADD a warning to
    any geometry.  That is asserted geometry by geometry, not argued.

    (2) The narrowest ORDINARY geometry the library builds is **5.0000e-02, a
    duty-0.9 pillar** (a 5 %-wide trench either side), which is **1.67x** above
    the 3e-2 edge; a 16-cell uniform lattice is 6.2500e-02 = 2.08x.  The
    round-3 gate's ``worst > 3.0 * edge`` is a property of the ROUND-2 BATTERY,
    which contains neither.  0 ordinary geometries land in the band on either
    reading, on both builds, so the edge stands -- and the margin is recorded
    here at what it actually is."""
    edge = _ts._STAG_SLIVER_BAND_FRAC
    lo = _ts._STAG_MIN_SEG_FRAC
    battery = _census_battery()
    ordinary, closing = {}, {}
    for name, st in battery.items():
        g = _grids_of(st)
        live = _ts._stag_mortared_axes(g)
        all_axes = _ts._stag_band_narrowest(g, (True, True))[0]
        per_axis = _ts._stag_band_narrowest(g, live)[0]
        # (1) MONOTONE by construction, asserted per geometry
        assert per_axis >= all_axes - 1e-15, (name, per_axis, all_axes)
        (closing if "closing" in name else ordinary)[name] = (all_axes,
                                                              per_axis)
    assert len(ordinary) >= 12, sorted(ordinary)
    for name, (a, p) in ordinary.items():
        assert not (lo <= a < edge), (name, a)          # round-3 reading
        assert not (lo <= p < edge), (name, p)          # round-4 reading
    # (2) the margin, RE-MEASURED, and the geometry that sets it
    worst = min(p for _a, p in ordinary.values())
    who = [n for n, (_a, p) in ordinary.items() if p == worst]
    # MEASURED 5.0000e-02 = 1.67x the edge (a duty-0.9 pillar); the round-2
    # battery alone reads 1.0937e-01 = 3.6x.  Asserted at 1.25x, i.e. with the
    # measured margin stated and 1.33x of slack over the assertion -- a
    # duty-0.94 pillar would cross it, and that is the honest scope.
    assert worst > 1.25 * edge, (worst, edge, who, ordinary)
    assert worst < 3.0 * edge, (worst, who,
                                "the wider census must reach INSIDE the "
                                "round-3 gate's battery-scoped 3x margin, or "
                                "this gate is not testing the scope")
    # the CLOSING taper is the surface the band exists for, and it reaches it
    # from NINE slices: 8 is 3.1250e-02 (1.04x outside), 9 is 2.7778e-02
    assert closing, "the census must contain a taper whose tip closes"
    assert any(lo <= p < edge for _a, p in closing.values()), closing


# ==========================================================================
# DEFECT 1 -- the mechanism is the ASYMMETRY, and the wording now says so
# ==========================================================================
def test_the_promoted_side_bar_is_scoped_to_the_asymmetric_interface():
    """DEFECT 1.  Round 3 derived ``_MORTAR_RESID_REFUSE`` from "the operand is
    rank-deficient by construction whenever ONE side is a promoted in-plane
    region", which a reader takes as "either side".  The near-null space needs
    the interface to be ASYMMETRIC -- EXACTLY ONE promoted side.

    Nothing shipped behaves differently, so what this gate protects is the
    WORDING a later reader derives a bar from, and it checks the string a user
    is actually SHOWN as well as the docstrings:

      * the refusal message the guarded solve raises;
      * :data:`_MORTAR_RESID_REFUSE`'s derivation;
      * :func:`_guarded_mortar_solve`'s ``screen='residual'`` bullet;
      * the site comment inside the generalized mortar itself.

    MEASURED, and it is why the wording matters: with BOTH sides promoted
    ``s_min/s_max`` reads 2.631e-05 / 2.416e-08 / 4.295e-07 at ``M`` = 4/5/6
    against the ASYMMETRIC interface of the SAME solve at 1.356e-10 /
    8.371e-12 / 2.068e-12 -- five decades apart.  The measurement itself is
    gated by
    ``test_a_generalized_mortar_with_both_sides_promoted_is_not_rank_deficient``
    in ``tests/unit/test_verify_pmm2d_mortar_round3.py``; this gate is about
    the text, which is the thing that was wrong.
    """
    # the derivation and the user-facing message must not say "either", and
    # must say EXACTLY ONE
    src = pathlib.Path(_pc.__file__).read_text(encoding="utf-8",
                                               errors="replace")
    head = src.split("_MORTAR_RESID_REFUSE = 1e-6")[0]
    deriv = head[head.rindex("#: REFUSE bar for the GENERALIZED"):]
    assert "EXACTLY ONE side" in deriv, deriv[:400]
    assert "ASYMMETRIC" in deriv
    assert "BOTH sides promoted" in deriv, "the correction must state the " \
                                           "measurement it corrects"
    guard_doc = inspect.getdoc(_pc._guarded_mortar_solve)
    assert "EXACTLY ONE promoted in-plane side" in guard_doc, guard_doc
    site_src = inspect.getsource(_pc._interface_smatrix_general_mortar_2d)
    assert "EXACTLY ONE side is an in-plane region promoted" in site_src

    # the REFUSAL MESSAGE, read off a real refusal rather than off the source
    A = np.eye(4, dtype=complex)
    A[:, 1] = A[:, 0]                       # exactly singular by construction
    B = np.arange(16, dtype=complex).reshape(4, 4) + 1.0
    try:
        # scipy's own "Diagonal number k is exactly zero" LinAlgWarning is the
        # POINT of this operand; swallow it so it does not leak into the run
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _pc._guarded_mortar_solve(A, B, "probe site", screen="residual")
    except Exception as e:                  # noqa: BLE001
        msg = str(e)
    else:
        raise AssertionError("an exactly singular operand with an "
                             "out-of-range right-hand side must be REFUSED")
    assert "RANK-DEFICIENT BY CONSTRUCTION whenever EXACTLY ONE side" in msg, \
        msg


def test_the_band_documents_that_no_warning_is_not_no_degradation():
    """The release note the round-3 verification asked for, gated where a
    reader meets it.

    The band's upper edge is set by the FALSE-POSITIVE census, not by where
    the accuracy loss begins: on the verification's independent fixture the
    error has already grown 13.5x (``M`` = 7) / 22.9x (``M`` = 8) by the time a
    segment reaches the edge, and grows only a further 1.24x from there to the
    width contract.  A user reading "no warning" as "no degradation" would be
    wrong, and until round 4 the documentation invited that reading."""
    doc = inspect.getdoc(_ts._warn_stag_sliver_band)
    assert "13.5x" in doc and "22.9x" in doc, doc
    assert "1.24x" in doc, doc
    assert '"no warning"' in doc and '"no degradation"' in doc, doc
    assert "NOT where the floor starts" in doc.replace("\n", " "), doc

    # the constant's own comment carries the same statement and the census
    # scope, and those are ``#:`` comments, so they are read from the source
    src = pathlib.Path(_ts.__file__).read_text(encoding="utf-8",
                                               errors="replace")
    block = src.split("_STAG_SLIVER_BAND_FRAC = 3.0e-2")[0]
    block = block[block.rindex("#: UPPER EDGE of the SILENTLY-DEGRADED band"):]
    for token in ("5.0000e-02", "duty-0.9", "1.67x", "SAMPLE-SCOPED",
                  "13.5x", "22.9x", "1.24x"):
        assert token in block, (token, block[-2500:])

    # and the CHANGELOG says it too, inside the round-3 paragraph rather than
    # under a second header
    ch = pathlib.Path(_ts.__file__).parents[3] / "CHANGELOG.md"
    if ch.exists():
        txt = ch.read_text(encoding="utf-8", errors="replace")
        unrel = txt.split("## [", 2)[1]
        assert "ROUND 4" in unrel, "the round-4 note must sit inside the " \
                                   "existing Unreleased block"
        for token in ("13.5x", "1.24x", "5.0000e-02", "EXACTLY ONE"):
            assert token in unrel, token
