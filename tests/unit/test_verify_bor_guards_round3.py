"""INDEPENDENT VERIFICATION of ROUND 3 of the 5.45.1 BOR/EME guards.

Companion to ``docs/audits/VERIFY_BOR_GUARDS_ROUND3_2026_09_12.md``, which
re-measures `docs/audits/FIX_BOR_GUARDS_ROUND3_2026_09_12.md`'s three
behaviour changes on fixtures written for the verification.  GAP 2 (the split
of the passivity predicate), GAP 3 (a non-finite ``q_excess`` formed from a
real spectrum is HOT) and the bit identity against ``1ac6de7e`` all reproduced
as claimed, so nothing here re-states them; the file exists for the one place
the re-measurement found a gap and for the one scope constant whose margin on
an independent fixture is thinner than on the fixture it was measured on.

WHAT IS HERE

* **D-V1 (P3)** -- ``_BOR_FRAC_DEADBAND`` closes the width comparison's tie
  only where the MESH reproduces the requested width at or below the bar.
  Whether it does is a property of the wall coordinates, not of the library,
  and on 63 of 169 ``(Rbig, wall)`` combinations it reproduces it ABOVE, by up
  to 156,767 ULP -- four decades outside a 16-ULP deadband.  On those
  geometries the same physical liner still decides two ways AT THE EDGE, and
  round 3 OPENED that split rather than closing it: on ``1ac6de7e`` all three
  positions read ``ok`` there (every comparison was strict and every
  representation failed it); on round 3 the axis reads ``warn_own`` -- which is
  the correct reading -- and the other two still read ``ok``.  Round 3 does
  close the geometry's two non-finite-``q_excess`` rungs (GAP 3), so the ladder
  goes 2 of 9 disagreeing to 1 of 9; the rung that survives is the one this
  file is about.  Two gates: the arithmetic premise, unconditional, and the
  decision, ``xfail(strict=True)``.

* **GAP 5's knee, on a second fixture.**  The shipped scope
  ``_CUTOFF_ENERGY_FLOOR_MULT = 100`` separates the two closure populations on
  this verification's own near-cutoff fixture too -- but the highest rung that
  exceeds 1e-6 sits at 91.7x the channel floor here against the fix round's
  79x, i.e. the SCOPE has 1.09x of headroom on an independently written
  fixture.  The BAR it scopes still has 21x.  Asserted on the second fixture so
  the claim stops being a property of one geometry, which is the same lesson
  GAP 5 itself was filed for.

PREMISE GATING, IN ONE DIRECTION ONLY.  A claim that a pathology REPRODUCES on
the running arm is gated with its reading (the CI pool is a random per-job mix
of EPYC 9V74 and 7763 with older wheels, and a kernel is entitled to a cleaner
spectrum); every claim that an INVARIANT holds is unconditional.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pytest

from lumenairy.elements.bor import _orient as _or
from lumenairy.elements.bor import _sem_contract as _sc
from lumenairy.elements.bor.bor_stack import BORStack

# =========================================================================== #
#  D-V1 -- the width deadband is narrower than the mesh's reproduction error  #
# =========================================================================== #
#: THE SECOND ``Rbig``, and why it is this one.
#:
#: ``verdict``'s width arms compare ``w_min_own_frac`` -- a difference of two
#: MESH BREAKPOINTS divided by ``Rbig`` -- against exact decimal literals.  For
#: an interior liner the mesh holds ``r`` and ``fl(r + w)``, so the width it
#: reproduces is ``fl(r + w) - r``, which is ``w`` only up to the rounding of a
#: sum of numbers of order ``Rbig``; for an outer liner it is
#: ``Rbig - fl(Rbig - w)``.  Round 3's ``_BOR_FRAC_DEADBAND`` (16 ULP relative,
#: 16.78 ULP of the 1e-6 bar) closes the comparison's tie only where that
#: reproduction lands AT OR BELOW the bar.
#:
#: MEASURED, pure IEEE-754 arithmetic over 13 ``Rbig`` values x 11 interior
#: wall positions + the outer wall
#: (``validation/probe_verify_bor_round3/v3_liner.py`` part ``repr``,
#: Windows py3.14 and WSL py3.12, four loaded kernels, 2026-09-12): **63 of 169
#: combinations reproduce the edge width ABOVE the bar**, the worst by
#: **156,767 ULP** -- and ``_below`` answers ``False`` on every one of them.
#: Nine of the thirteen ``Rbig`` values contain positions that split.
#:
#: ``Rbig`` = 24 (the fix round's) reproduces it 80,088 ULP BELOW the bar,
#: which is why its ladder reads 0 of 9 disagreements.  ``Rbig`` = 12.5 with an
#: interior wall at 5.0 reproduces it 156,767 ULP ABOVE, and is the worst
#: combination in that grid.
_DV1_RBIG = 12.5
_DV1_WALL = 5.0
_DV1_N, _DV1_M, _DV1_DEGREE, _DV1_K0 = 96, 2, 6, 3.5
_DV1_EPS_HI, _DV1_EPS_LO = 5.0, 2.2


def _reproduced_frac(Rbig, wall, w):
    """What the MESH's ``np.diff`` sees for a liner of requested width ``w``
    whose inner wall is at ``wall``: ``(fl(wall + w) - wall) / Rbig``.  Pure
    arithmetic -- no solve, no BLAS, identical on every arm."""
    return ((float(wall) + float(w)) - float(wall)) / float(Rbig)


def test_the_width_deadband_does_not_reach_the_meshs_reproduction_error():
    """THE PREMISE OF D-V1, stated as arithmetic and asserted unconditionally.

    ``_BOR_FRAC_DEADBAND`` is a RELATIVE band around the bar; the quantity it
    has to absorb is the mesh's reproduction error of the requested width,
    which is relative to ``Rbig / w`` and therefore ENORMOUS by comparison.
    This gate measures both on the running build and asserts the inequality
    that makes D-V1 possible -- so if a future round sizes the deadband from
    the reproduction error instead, this gate fires and is re-derived.

    Build-free: every number here is IEEE-754 arithmetic on the wall
    coordinates.  It cannot move with the BLAS kernel, the thread count or the
    interpreter.
    """
    bar = float(_sc._BOR_MIN_ELEM_FRAC)
    reach = float(_sc._BOR_FRAC_DEADBAND) * bar        # absolute, at the bar
    w = bar * _DV1_RBIG
    fa = _reproduced_frac(_DV1_RBIG, _DV1_WALL, w)
    err = fa - bar
    assert err > 0.0, (
        "the premise moved: Rbig=%g with a wall at %g now reproduces the edge "
        "width at %.17g, which is NOT above the %.0e bar -- re-derive D-V1's "
        "geometry from validation/probe_verify_bor_round3/v3_liner.py part "
        "'repr'" % (_DV1_RBIG, _DV1_WALL, fa, bar))
    assert err > reach, (
        "the deadband now reaches the reproduction error (error %.4e = %.0f "
        "ULP of the bar, deadband %.4e = %.2f ULP): D-V1 may be closed -- "
        "re-measure the three-position ladder at Rbig=%g and retire the xfail "
        "below" % (err, err / np.spacing(bar), reach,
                   reach / np.spacing(bar), _DV1_RBIG))
    assert not _sc._below(fa, bar), (
        "_below answered True at %.4e above the bar, so the deadband is no "
        "longer 16 ULP -- re-derive" % (err,))
    # and the axis representation, which is exact, decides the other way
    assert _sc._below(w / _DV1_RBIG, bar), (
        "the axis representation (w - 0) no longer resolves to the "
        "informative side of the bar, which is round 3's own GAP 4 fix")


def _dv1_record(position, w_frac):
    """One layer's mesh record for a caller-prescribed liner of relative width
    ``w_frac``, at the axis / on the interior wall / at the outer wall of the
    D-V1 geometry."""
    Rbig, w = _DV1_RBIG, w_frac * _DV1_RBIG
    segs = {"axis": [(w, _DV1_EPS_HI), (Rbig, _DV1_EPS_LO)],
            "middle": [(_DV1_WALL, _DV1_EPS_HI), (_DV1_WALL + w, _DV1_EPS_LO),
                       (Rbig, _DV1_EPS_LO)],
            "outer": [(Rbig - w, _DV1_EPS_LO), (Rbig, _DV1_EPS_HI)]}[position]
    st = BORStack(Rbig=Rbig, m=_DV1_M, N=_DV1_N, n_superstrate=1.2,
                  n_substrate=1.7, basis="sem", degree=_DV1_DEGREE)
    st.add_layer(0.6, segments=segs)
    st.set_source(k0=_DV1_K0)
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev
    assert recs, "%s liner produced no mesh report" % (position,)
    return recs[0]


@pytest.mark.xfail(strict=True, reason=(
    "VERIFY round 3, D-V1: _BOR_FRAC_DEADBAND (16 ULP of the bar) is four "
    "decades narrower than the mesh's own reproduction error of the requested "
    "width (up to 156,767 ULP), so on the 63 of 169 (Rbig, wall) combinations "
    "that reproduce it ABOVE the bar the three positions still disagree.  At "
    "Rbig=12.5 the axis reads warn_own and the interior and outer walls read "
    "ok; on 1ac6de7e all three read ok, so round 3 introduced the split on "
    "this geometry while removing it from its own."))
def test_the_liner_verdict_agrees_at_all_three_positions_at_a_second_rbig():
    """ROUND 3's GAP 4 CLAIM, on a second geometry: *the same physical liner
    decides identically at the axis, in the interior and at the outer wall*.

    It holds at ``Rbig`` = 24, where the mesh reproduces the edge width BELOW
    the bar and the exact-tie axis representation is the only one the strict
    comparison excluded.  It does not hold where the mesh reproduces it ABOVE:
    the deadband is sized to the BAR (16 ULP) and the error is sized to
    ``Rbig / w``.

    MEASURED, this geometry, ``Rbig`` = 12.5, interior wall 5.0, a liner
    exactly ``_BOR_MIN_ELEM_FRAC`` of ``Rbig`` wide, 2026-09-12
    (``validation/probe_verify_bor_round3/v3_ladder_*.json``, geometry B):

        arm (build x loaded kernel x threads)        axis / middle / outer
        Windows py3.14  Haswell      t1 and t4       warn_own / ok / ok
        Windows py3.14  Sandybridge  t1              warn_own / ok / ok
        Windows py3.14  Katmai       t1              warn_own / ok / ok
        WSL py3.12      Haswell      t1 and t4       warn_own / ok / ok
        WSL py3.12      Sandybridge  t1              warn_own / ok / ok
        WSL py3.12      Katmai       t1              warn_own / ok / ok

    Eight arms, one reading: the split is arithmetic, not backward error.  The
    same geometry on ``1ac6de7e``, six arms (both builds x Haswell /
    Sandybridge / Katmai), reads ``ok`` on all three positions -- so this is a
    behaviour round 3 OPENED here while closing it at its own ``Rbig``.  Over
    the whole nine-rung ladder round 3 still improves this geometry -- it
    closes the two non-finite-``q_excess`` rungs (GAP 3), 2 of 9 disagreeing
    down to 1 of 9 -- and the rung that survives is this one.

    CONSEQUENCE.  Bounded, and the same as GAP 4's: a ``UserWarning`` present
    or absent, never a wrong returned number -- which is why this is P3.  The
    remedy is to size the deadband from the record itself (the reproduction
    error of a width ``w`` against a wall at ``r`` is about
    ``np.spacing(r) / w`` relative) rather than from the bar.

    PREMISE-GATED on all three positions being spectrally HOT, which is what
    makes the width comparison the only thing left to decide the verdict.
    """
    w_frac = float(_sc._BOR_MIN_ELEM_FRAC)
    recs = {p: _dv1_record(p, w_frac) for p in ("axis", "middle", "outer")}
    cold = {p: r["q_excess"] for p, r in recs.items()
            if not (np.isfinite(r["q_excess"])
                    and r["q_excess"] > _sc._BOR_Q_EXCESS)}
    if cold:
        pytest.skip(
            "premise absent on this arm: %s not spectrally hot (q_excess %s), "
            "so the width comparison is not what decides the verdict here"
            % (sorted(cold), {k: "%.4g" % v for k, v in cold.items()}))
    got = {p: _sc.verdict(r) for p, r in recs.items()}
    assert len(set(got.values())) == 1, (
        "the same %.0e-of-Rbig liner gets different verdicts depending on "
        "which end of the domain it sits against: %s (reproduced widths %s, "
        "bar %.0e, deadband %.2f ULP)"
        % (w_frac, got,
           {p: "%.17g" % r["w_min_own_frac"] for p, r in recs.items()},
           _sc._BOR_MIN_ELEM_FRAC,
           _sc._BOR_FRAC_DEADBAND * _sc._BOR_MIN_ELEM_FRAC
           / np.spacing(float(_sc._BOR_MIN_ELEM_FRAC))))


# =========================================================================== #
#  GAP 5's knee, on a fixture written for this verification                    #
# =========================================================================== #
#: The near-cutoff fixture, INDEPENDENT of the gate's own.
#: ``test_fix_bor_multilayer_guards`` measures Rbig 24 / N 120 / n 1.41 and a
#: (3.0, 0.5, 2.45, 1.41) ring layer; this is Rbig 19 / N 96 / n 1.63 and a
#: (2.7, 0.45, 2.10, 1.63) ring layer between two 0.35-thick slabs.
_C_RBIG, _C_N, _C_NREF = 19.0, 96, 1.63
_C_EPS = _C_NREF ** 2
_C_RINGS = (2.7, 0.45, 2.10, 1.63)

#: The two shipped constants this gate re-measures, read from the module that
#: owns them so nothing here pins a copy.
_C_FLOOR = float(_or._BOR_CHANNEL_REAL_FLOOR)

#: SCOPE: the multiple of the channel gate's real floor at which the closure
#: leaves its round-off floor.  ``test_fix_bor_multilayer_guards.
#: _CUTOFF_ENERGY_FLOOR_MULT`` ships 100.0 and records the knee at 79x on its
#: own fixture.  MEASURED HERE over m 0..3 x idx 1..3 x 13 rungs = 156 solves
#: (``validation/probe_verify_bor_round3/v4_family_win_Haswell_t1.json``,
#: 2026-09-12): the shallow population (108 rows at or above 100x) reads
#: **4.707754e-07** and the deep population (48 rows) **2.888043e-05** -- 1.79
#: decades apart, and NO row at or above 100x exceeds 1e-6.  But the HIGHEST
#: rung anywhere in the grid that does exceed 1e-6 sits at **91.66x**, so the
#: scope has 1.09x of headroom on this fixture against 1.27x on the fix
#: round's.  The BAR it scopes keeps 21.2x (1e-5 / 4.7078e-07) here and 14.5x
#: there; the deep bound keeps 69.3x (2e-3 / 2.8880e-05) here and 11.2x there.
#:
#: ORIGIN OF THE THREE NUMBERS.  All three are the values
#: ``tests/unit/test_fix_bor_multilayer_guards.py`` ships
#: (``_CUTOFF_ENERGY_FLOOR_MULT``, ``_CUTOFF_LADDER_BAR``, ``_CUTOFF_DEEP_BAR``),
#: restated here rather than imported: this gate exists to ask whether those
#: numbers survive a SECOND fixture, so importing them would make the two gates
#: move together and the question unanswerable.  If that module changes any of
#: them, this gate keeps the old value, diverges, and the divergence is the
#: report -- which is the intended behaviour, and the reason each carries its
#: own measured margin above (21.2x and 69.3x on this fixture).
_C_ENERGY_FLOOR_MULT = 100.0
_C_SHALLOW_BAR = 1.0e-5
_C_DEEP_BAR = 2.0e-3


def _c_modes(m, k0):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, _C_RBIG, _C_N,
                       lambda r: np.full_like(r, _C_EPS, dtype=complex),
                       float(k0), staggered=True)


def _c_gamma(m, idx):
    L = _c_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _C_EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx]) if g.size > idx else None


def _c_stack(m, k0):
    s = BORStack(_C_RBIG, m, n_substrate=_C_NREF, n_superstrate=_C_NREF,
                 N=_C_N, basis="fd")
    s.add_layer(0.35, eps=_C_EPS)
    s.add_layer(0.6, rings=_C_RINGS)
    s.add_layer(0.35, eps=_C_EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def _c_rungs():
    """The ladder, bounded below by the R/T channel gate's own floor times the
    shipped ``_CUTOFF_LADDER_FLOOR_MULT`` (10), derived rather than pinned."""
    out = []
    for e_ in range(8, 40):
        dl = 10.0 ** (-e_ / 2.0)
        if _C_NREF * np.sqrt(dl) < _C_FLOOR * 10.0:
            break
        out.append(dl)
    return out


@pytest.mark.parametrize("m,idx", [(0, 1), (1, 1)])
def test_the_near_cutoff_populations_separate_on_a_second_fixture(m, idx):
    """GAP 5's shipped scope and its two bars, re-measured on a fixture that
    shares no numbers with the gate's own.

    THE POINT.  GAP 5 was filed because a margin measured over ``m`` alone was
    quoted as a family property while the axis that actually moved it
    (``idx``) was never swept.  The remedy round 3 shipped sweeps that axis --
    on ONE geometry.  This gate carries the same three claims on a second one,
    so the separation stops being a property of a single Rbig / N / index
    triple.  ``(m, idx)`` = (0, 1) and (1, 1) are the two ladders whose deep
    rungs are worst here (2.888e-05 and 4.010e-06).

    THREE CLAIMS, in decreasing strength, and the first two are the durable
    ones -- they are integers:

    1. the R/T channel count is ONE number over the whole ladder;
    2. it is ``idx + 1``;
    3. the closure is inside 1e-5 wherever the marginal channel sits at or
       above 100x the channel gate's real floor, and inside 2e-3 below that.

    BARS AND THEIR MARGINS, measured on this fixture over the full
    m 0..3 x idx 1..3 grid, Windows py3.14 / Haswell / 1 thread, 2026-09-12:
    shallow envelope 4.707754e-07 (bar 1e-5, **21.2x**), deep envelope
    2.888043e-05 (bound 2e-3, **69.3x**).  Both bars therefore clear their
    measured envelope on BOTH fixtures, which is the property that matters;
    the SCOPE constant is the thinner of the three (1.09x here).
    """
    g = _c_gamma(m, idx)
    assert g is not None, (
        "the m=%d spectrum of this fixture has no radial cutoff at index %d "
        "-- the grid changed and the ladder must be re-derived" % (m, idx))
    rungs = _c_rungs()
    assert len(rungs) >= 12, (
        "the ladder collapsed to %d rungs -- re-derive it" % (len(rungs),))
    counts, worst_shallow, worst_deep = set(), 0.0, 0.0
    knee = _C_FLOOR * _C_ENERGY_FLOOR_MULT
    for dl in rungs:
        k0 = g / (_C_NREF * np.sqrt(1.0 - dl))
        res = _c_stack(m, k0)
        counts.add(int(np.size(res["R"])))
        en = np.asarray(res["energy"], float)
        if en.size:
            c = float(np.max(np.abs(en - 1.0)))
            if _C_NREF * np.sqrt(dl) >= knee:
                worst_shallow = max(worst_shallow, c)
            else:
                worst_deep = max(worst_deep, c)
    assert len(counts) == 1, (
        "m=%d, radial cutoff index %d, SECOND fixture (Rbig %g, N %d, n %g): "
        "the R/T channel count moves over the near-cutoff ladder: %s"
        % (m, idx, _C_RBIG, _C_N, _C_NREF, sorted(counts)))
    assert counts == {idx + 1}, (
        "m=%d, radial cutoff index %d, SECOND fixture: the ladder drives the "
        "%d-th channel to cutoff so the solve must report %d channels; it "
        "reports %s" % (m, idx, idx + 1, idx + 1, sorted(counts)))
    assert worst_shallow < _C_SHALLOW_BAR, (
        "m=%d, idx=%d, SECOND fixture: worst lossless closure %.4e on the "
        "rungs at qn >= %.0fx the channel floor, past the %.0e bar (measured "
        "family envelope on this fixture 4.7078e-07 over 12 ladders x 9 rungs "
        "-- 21.2x)" % (m, idx, worst_shallow, _C_ENERGY_FLOOR_MULT,
                       _C_SHALLOW_BAR))
    assert worst_deep < _C_DEEP_BAR, (
        "m=%d, idx=%d, SECOND fixture: worst lossless closure %.4e on the DEEP "
        "rungs (qn < %.0fx the channel floor), past the %.0e bound (measured "
        "envelope on this fixture 2.8880e-05 over 12 ladders x 4 rungs -- "
        "69.3x).  The residual is the marginal channel's flux normalisation, "
        "so start at its own |flux| census"
        % (m, idx, worst_deep, _C_ENERGY_FLOOR_MULT, _C_DEEP_BAR))


# =========================================================================== #
#  D-V3 -- GAP 3 also converts a RETURNED answer into a RAISE, which nothing   #
#  else pins                                                                   #
# =========================================================================== #
#: THE POPULATION.  ``verdict``'s ``refuse`` arm is
#: ``_below(w_min_union_frac, _BOR_MIN_ELEM_FRAC) and hot``, so GAP 3's change
#: to ``hot`` reaches it: a MANUFACTURED union cell narrower than the bar whose
#: spectrum came back non-finite read ``warn_manufactured`` on ``1ac6de7e`` and
#: reads ``refuse`` on round 3 -- and with ``BOR_SEM_MESH_GUARD`` at its default
#: ``True`` that is a ``BORSemMeshError`` where the caller previously received
#: an answer.
#:
#: MEASURED, this geometry, Windows py3.14 / Haswell / 1 thread, 2026-09-12:
#:
#:     layer 0 wall     layer 1 wall     w_min_union_frac   q_excess   1ac6de7e -> round 3
#:     1e-9 x Rbig      2e-9 x Rbig      1e-09              inf        returned -> BORSemMeshError
#:     2e-9 x Rbig      3e-9 x Rbig      1e-09              inf        returned -> BORSemMeshError
#:     1e-8 x Rbig      1.1e-8 x Rbig    1e-09              1.87e+08   BORSemMeshError (both)
#:
#: The new behaviour is the RIGHT one -- a manufactured sliver on a spectrum
#: that has blown up is exactly what ``refuse`` exists for, and returning it was
#: the defect.  What is wrong is the round's rating of the cost: it records the
#: old reading as "a missing ``UserWarning``, never a wrong returned number",
#: and this population shows it was also a RETURNED ANSWER where the contract
#: had to refuse.  No identity battery in round 3 or in this verification
#: contains the population, so this gate is the only thing pinning it.
#:
#: MARGIN.  The two conjuncts this gate needs are four decades clear of their
#: bars on every arm measured: the manufactured union cell is 1e-09 of ``Rbig``
#: against the 1e-06 ``_BOR_MIN_ELEM_FRAC`` edge (3 decades), and ``q_excess``
#: reads ``inf`` on Haswell/Katmai/Haswell-t4 and 1.87e+08 where it is finite,
#: against the 1e+04 ``_BOR_Q_EXCESS`` bar (4.3 decades).  So the REFUSAL is
#: unconditional here and the non-finite reading is not: the gate asserts the
#: refusal always and reads the ratio only to say which branch produced it.
_DV3_A, _DV3_B = 1.0e-9, 2.0e-9


def _dv3_stack():
    st = BORStack(Rbig=_DV1_RBIG, m=_DV1_M, N=_DV1_N, n_superstrate=1.2,
                  n_substrate=1.7, basis="sem", degree=_DV1_DEGREE)
    st.add_layer(0.6, segments=[(_DV3_A * _DV1_RBIG, _DV1_EPS_HI),
                                (_DV1_RBIG, _DV1_EPS_LO)])
    st.add_layer(0.6, segments=[(_DV3_B * _DV1_RBIG, _DV1_EPS_HI),
                                (_DV1_RBIG, _DV1_EPS_LO)])
    st.set_source(k0=_DV1_K0)
    return st


def test_a_manufactured_sliver_on_a_blown_up_spectrum_is_refused_not_returned():
    """GAP 3's change reaches the ``refuse`` arm, not only ``warn_own`` --
    asserted as the DECISION a caller sees.

    Two layers ask for axis walls one vacuum-wavelength-ratio apart, so the
    union mesh MANUFACTURES a cell neither layer's own segment list asked for,
    narrower than ``_BOR_MIN_ELEM_FRAC`` of ``Rbig``; at that width the
    spectrum blows up and ``q_excess`` comes back non-finite.  Before round 3
    ``hot`` was ``False`` there, the verdict was ``warn_manufactured``, and
    ``BORStack.solve`` RETURNED a number built on that spectrum.  Round 3 reads
    a formed non-finite ratio as HOT, so the verdict is ``refuse`` and the
    guard raises.

    Two claims, BOTH UNCONDITIONAL -- and deliberately so, because the refusal
    does not depend on which branch of ``hot`` produced it:

    1. the mesh record's two conjuncts are the ones the mechanism names -- the
       union width is below ``_BOR_MIN_ELEM_FRAC`` (1e-09 of ``Rbig`` against
       the 1e-06 bar, 3 decades) and the ratio was MEASURABLE;
    2. the verdict is ``refuse`` and, with the guard at its default, the caller
       gets ``BORSemMeshError``.

    A cleaner LAPACK is entitled to return a FINITE ``q_excess`` here; on such
    an arm the refusal still holds by magnitude (1.87e+08 measured, against the
    1e+04 bar -- 4.3 decades), which the gate asserts as well.  So the ratio's
    finiteness is READ, never gated on.
    """
    st = _dv3_stack()
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev
    assert recs, "the fixture produced no mesh report -- re-derive it"
    rec = recs[0]
    assert _sc._below(rec["w_min_union_frac"], _sc._BOR_MIN_ELEM_FRAC), (
        "the premise moved: the union cell this fixture manufactures is "
        "%.4e of Rbig, not below the %.0e bar, so the refuse arm is not "
        "reached" % (rec["w_min_union_frac"], _sc._BOR_MIN_ELEM_FRAC))
    assert rec.get("q_measurable", False), (
        "the ratio was never FORMED on this arm (q_measurable False), so this "
        "fixture is not the population D-V3 is about")
    assert _sc.verdict(rec) == "refuse", (
        "a manufactured union cell %.4e of Rbig wide with q_excess = %r is "
        "not being refused (verdict %r) -- on 1ac6de7e that reading was "
        "warn_manufactured and the solve RETURNED"
        % (rec["w_min_union_frac"], rec["q_excess"], _sc.verdict(rec)))
    if np.isfinite(rec["q_excess"]):
        # a cleaner kernel: the refusal above still holds, by magnitude
        assert rec["q_excess"] > _sc._BOR_Q_EXCESS
    st2 = _dv3_stack()
    with pytest.raises(_sc.BORSemMeshError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st2.solve()
