"""Niche C11 -- the decentred beam's ray-fit branch is ARBITRATED, not guessed.

Audit ``docs/audits/C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md``.

**What was wrong.**  ``_DECENTRE_GATE_W_FRAC = 0.05`` (niche C1) selects, per
element call, between the historical CONCENTRIC ray fit (disc about the grid
origin with radius ``frbf * sqrt(2 c^2 + w^2)``, hard NaN sample mask, order
``newton_poly_order``) and the D1/D7 OFF-CENTRE one (disc about the beam,
weighted restriction, order ``_DECENTRED_FIT_POLY_ORDER``).  0.05 w was chosen
to kill a discontinuity at NULL decentre and its own note says so -- it was
never the crossover.  Measured, the crossover is at **0.55 w** on a synthetic
f/3 N-BK7 singlet, at **0 w** on an f/6 one, and anywhere in **0.46-0.69 w**
across design 121's six groups: it is not a property of the decentre at all,
but of how much aberration the concentric fit's lower order leaves over the
beam, so no single constant can be right for every design.

**What replaces it.**  At the fit site the rays are already traced, so both
candidates are BUILT and COMPARED there: fit the OPL each way and score it
against the traced samples themselves, weighted by the beam's own intensity.
Smaller weighted rms wins; an exact tie keeps the concentric one.  See
:data:`~lumenairy.elements._lens_traced.DECENTRED_FIT_ARBITER`.

**What is pinned here.**  Every numeric bar below is a RATIO between two arms
measured in the same process on the same fixture, or an exact-arithmetic
identity -- there is no absolute bar on a BLAS-dependent magnitude anywhere.

1. the flag ships ``True``, and ``False`` restores the pure-gate selector (a
   real fail-before: the two states differ on the f/3 fixture and agree on the
   f/6 one, so the switch is not a no-op and not a blanket change either);
2. the niche-C1 NULL contract survives untouched in BOTH states -- below the
   gate the concentric path is byte-identical, which is what C1 exists for;
3. the arbiter moves the STEP off the gate: across 0.05 w the f/3 fixture's
   field jump falls ~100x, because the branch no longer changes there;
4. the crossover is design-dependent -- same beam, same gate, two lenses, two
   different answers;
5. the candidate that is SCORED is the candidate that is APPLIED (order and
   restriction both), which is what makes it an arbiter rather than a coin
   toss;
6. the scoring function ranks a fit that reproduces the traced map above one
   that does not, and refuses to let an inadmissible candidate win;
7. the restriction helper reproduces D1's weight formula and D7's order
   step-down exactly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lt
from lumenairy.elements._lens_traced import (
    _Cheb2DEvaluator,
    _decentred_fit_restriction,
    _decentred_fit_score,
)

_WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent',
            on_aperture_beam='silent')
_MIN_FREE_GIB = 3.0


def _ram_guard(need=_MIN_FREE_GIB):
    try:
        import psutil
    except ImportError:
        return
    free = psutil.virtual_memory().available / (1024 ** 3)
    if free < need:
        pytest.skip(f"needs ~{need} GiB available, saw {free:.1f}")


def _singlet(R1, R2, d, glass, ap, name):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


# The niche-C1 fixtures, unchanged: an f/6 and an f/3 N-BK7 singlet, beam
# 1.0 mm, ``fit_radius_beam_factor=2`` so the disc is live, ``ray_subsample=8``
# so D7's order-10 raise clears its own sample-count step-down.
_N, _DX, _W = 512, 30e-6, 1.0e-3
_SLOW = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'c11-f6')
_FAST = _singlet(30e-3, -30e-3, 3.0e-3, 'N-BK7', 10e-3, 'c11-f3')


def _gauss(n, dx, w, cx=0.0, cy=0.0):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) / w ** 2)
                  ).astype(np.complex128)


def _apply(c, tell, arbiter, presc=_SLOW, **kw):
    """One element call with the beam physically at ``(c, 0)``.  ``tell`` is
    the decentre the element is TOLD about (``None`` -> the grid origin, i.e.
    the historical concentric arm on the same physical field).

    ``arbiter`` is POSITIONAL and mandatory: the flag ships ``False``, and a
    test that exercised the arbiter by inheriting the default would silently
    stop exercising it the moment that default moved -- in either direction.
    Every arm below states which side it is on."""
    opts = dict(prescription=presc, wavelength=_WL, dx=_DX, ray_subsample=8,
                n_workers=1, fit_radius_beam_factor=2.0, carrier=np.inf,
                beam_centre=(0.0, 0.0) if tell is None else tell, **_TKW)
    opts.update(kw)
    old = _lt.DECENTRED_FIT_ARBITER
    _lt.DECENTRED_FIT_ARBITER = bool(arbiter)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                _gauss(_N, _DX, _W, c, 0.0), **opts))
    finally:
        _lt.DECENTRED_FIT_ARBITER = old


def _step(A, B):
    pk = float(np.abs(A).max())
    return float(np.abs(A - B).max()) / pk


# ===========================================================================
# 1 -- the flag, and its fail-before
# ===========================================================================
def test_the_arbiter_ships_off_as_an_opt_in():
    """It is OPT-IN, and that is a decision recorded in the constant's own
    note: on design 121 it improves four of five tilted orders and makes one
    worse by 0.026 points, which is a judgement about a design rather than a
    library fact.

    Turning it off must be a NO-OP, not merely a fall-back: with the flag
    ``False`` the arbiter's scoring function is never called at all, on a call
    that would otherwise reach it."""
    assert _lt.DECENTRED_FIT_ARBITER is False
    calls = []
    orig = _lt._decentred_fit_score

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    _lt._decentred_fit_score = spy
    try:
        c = 0.2 * _W
        _apply(c, (c, 0.0), False, presc=_FAST)
        assert calls == [], calls
        _apply(c, (c, 0.0), True, presc=_FAST)
        assert len(calls) == 2, calls
    finally:
        _lt._decentred_fit_score = orig


def test_turning_the_arbiter_off_is_a_real_fail_before_and_not_a_blanket_change():
    """``False`` restores the pure ``_DECENTRE_GATE_W_FRAC`` selector.  That is
    only a meaningful switch if it CHANGES something -- and only a meaningful
    arbiter if it does not change everything.

    On the f/3 singlet at 0.2 w the arbiter overrides the gate (the concentric
    fit reproduces the traced map better there), so the two states differ.  On
    the f/6 singlet at the SAME decentre it agrees with the gate, so they are
    bit-identical.  One fixture proves the switch is live, the other proves the
    arbiter is not simply reverting the branch."""
    _ram_guard()
    c = 0.2 * _W
    fast_on = _apply(c, (c, 0.0), True, presc=_FAST)
    fast_off = _apply(c, (c, 0.0), False, presc=_FAST)
    assert not np.array_equal(fast_on, fast_off)
    slow_on = _apply(c, (c, 0.0), True, presc=_SLOW)
    slow_off = _apply(c, (c, 0.0), False, presc=_SLOW)
    assert np.array_equal(slow_on, slow_off), float(
        np.abs(slow_on - slow_off).max())


# ===========================================================================
# 2 -- the niche-C1 null contract, in BOTH states
# ===========================================================================
_NULL_OFFSETS = [('1e-9 px', 1e-9 * _DX), ('0.4 px', 0.4 * _DX),
                 ('1 px', 1.0 * _DX), ('0.02 w', 0.02 * _W),
                 ('0.049 w', 0.049 * _W)]


@pytest.mark.parametrize('presc, tag', [(_SLOW, 'f6'), (_FAST, 'f3')],
                         ids=['f6', 'f3'])
@pytest.mark.parametrize('label, c', _NULL_OFFSETS,
                         ids=[o[0] for o in _NULL_OFFSETS])
def test_the_c1_null_contract_survives_the_arbiter(label, c, presc, tag):
    """BELOW the C1 gate the arbiter is inert BY CONSTRUCTION (it is gated on
    ``_beam_decentred``), so the concentric path stays byte-identical -- both
    to the origin-referenced arm, which is C1's own pin, and across the flag,
    which is the new statement.  If this ever fails the arbiter has leaked
    below the null gate and C1's whole finding is back."""
    _ram_guard()
    ref = _apply(c, None, True, presc=presc)
    got = _apply(c, (c, 0.0), True, presc=presc)
    off = _apply(c, (c, 0.0), False, presc=presc)
    assert np.array_equal(got, ref), float(np.abs(got - ref).max())
    assert np.array_equal(got, off), float(np.abs(got - off).max())


# ===========================================================================
# 3 -- the STEP moves off the gate
# ===========================================================================
def test_the_arbiter_takes_the_discontinuity_off_the_gate_on_the_fast_singlet():
    """Niche C1 relocated a branch-flip discontinuity from 1e-9 px to 0.05 w;
    it did not remove it.  On the f/3 fixture the field still jumps as the
    beam crosses 0.05 w, and nothing physical happens there -- 0.05 w is a
    constant, not a feature of the lens.

    Scored as a RATIO in one process: the jump across the gate with the
    arbiter off, against the jump across the gate with it on, and against the
    geometry's OWN smoothness measured the same way at 0.10 w (where no
    selector changes its mind in either state).  The arbiter's job is to make
    the first look like the third."""
    _ram_guard()
    g = _lt._DECENTRE_GATE_W_FRAC * _W
    rel = 1e-6

    def jump(centre, arbiter):
        a = _apply(centre * (1 - rel), (centre * (1 - rel), 0.0), arbiter,
                   presc=_FAST)
        b = _apply(centre * (1 + rel), (centre * (1 + rel), 0.0), arbiter,
                   presc=_FAST)
        return _step(a, b)

    smooth = jump(0.10 * _W, True)
    with_gate = jump(g, False)
    with_arb = jump(g, True)
    # the gate's step is LIVE -- far above what the same measurement reads
    # where no branch changes
    assert with_gate > 10.0 * smooth, (with_gate, smooth)
    # ... and the arbiter removes it, to within the geometry's own smoothness
    assert with_arb < 0.1 * with_gate, (with_arb, with_gate)
    assert with_arb < 3.0 * smooth, (with_arb, smooth)


# ===========================================================================
# 4 -- the crossover is a property of the DESIGN
# ===========================================================================
def test_the_crossover_is_design_dependent_not_a_constant():
    """Same beam, same decentre, same gate, same fit_radius_beam_factor -- two
    lenses, two different branches.  This is why a constant cannot be right:
    at 0.2 w the f/3 singlet's concentric fit is the better one and the f/6
    singlet's is not.

    Stated as an exact identity rather than a threshold: when the arbiter takes
    the concentric candidate, the returned field is bit-identical to the arm
    that was TOLD the beam is on the grid origin (which is that candidate, by
    construction); when it takes the off-centre one it is not."""
    _ram_guard()
    c = 0.2 * _W
    for presc, expect_concentric in ((_FAST, True), (_SLOW, False)):
        ref = _apply(c, None, True, presc=presc)
        got = _apply(c, (c, 0.0), True, presc=presc)
        assert np.array_equal(got, ref) is expect_concentric, (
            presc['name'], float(np.abs(got - ref).max()))


def test_the_f6_fixture_still_routes_to_the_weighted_raised_order_path():
    """The live half of niche C1's
    ``test_a_genuine_decentre_still_routes_to_the_weighted_raised_order_path``
    (which is era-pinned at ``DECENTRED_FIT_ARBITER = False`` there, because
    its spy counts the arbiter's TRIAL fits as well as the applied ones).

    Same claim, scoped to the fits the Newton inversion is actually handed --
    the LAST three ``_Cheb2DEvaluator`` builds of the call."""
    _ram_guard()
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((int(order), weights is not None))

    _Cheb2DEvaluator.__init__ = spy
    try:
        for c in (0.06 * _W, 0.2 * _W, 1.0 * _W):
            seen.clear()
            _apply(c, (c, 0.0), True, presc=_SLOW)
            applied = seen[-3:]
            assert applied and all(
                o == _lt._DECENTRED_FIT_POLY_ORDER for o, _w in applied), seen
            assert all(w for _o, w in applied), 'the disc lost its weights'
    finally:
        _Cheb2DEvaluator.__init__ = orig


# ===========================================================================
# 5 -- the candidate SCORED is the candidate APPLIED
# ===========================================================================
@pytest.mark.parametrize('presc', [_SLOW, _FAST], ids=['f6', 'f3'])
def test_the_scored_candidate_is_the_applied_candidate(presc):
    """An arbiter that scores one configuration and applies another is a coin
    toss.  Both candidates are built through ``_decentred_fit_restriction``,
    and so is the applied fit, so the (order, weighted) pair of the three
    APPLIED fits must equal that of the WINNING trial fit -- and the winner is
    read from the library's own scores, not assumed."""
    _ram_guard()
    c = 0.2 * _W
    builds, scores = [], []
    o_ev, o_sc = _Cheb2DEvaluator.__init__, _lt._decentred_fit_score

    def spy_ev(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
        o_ev(self, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        builds.append((int(order), weights is not None))

    def spy_sc(xs, opl, wg, disc, wts, order):
        v = o_sc(xs, opl, wg, disc, wts, order)
        scores.append(v)
        return v

    _Cheb2DEvaluator.__init__ = spy_ev
    _lt._decentred_fit_score = spy_sc
    try:
        _apply(c, (c, 0.0), True, presc=presc)
    finally:
        _Cheb2DEvaluator.__init__ = o_ev
        _lt._decentred_fit_score = o_sc
    # exactly two trials (off-centre, then concentric) then the three fits the
    # Newton loop is given -- the arbiter costs ONE extra OPL fit per branch
    assert len(scores) == 2, scores
    assert len(builds) == 5, builds
    s_off, s_conc = scores
    winner = builds[1] if s_conc <= s_off else builds[0]
    assert builds[-3:] == [winner] * 3, (builds, scores)


# ===========================================================================
# 6 -- the scoring function itself
# ===========================================================================
def test_the_score_ranks_the_fit_that_reproduces_the_traced_map():
    """A degree-2 map is spanned exactly by an order-2 basis and not by an
    order-1 one, so the score must put the first far below the second -- and
    the ratio is between two arms of the same measurement, not a bar."""
    n = 41
    xs = np.linspace(-1.0e-3, 1.0e-3, n)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    opl = 1e-3 + 0.3 * X + 0.7 * Y + 2.0e2 * (X * X - 0.5 * Y * Y) + 5e1 * X * Y
    disc = (X ** 2 + Y ** 2) <= (0.8e-3) ** 2
    wgt = np.exp(-2.0 * (X ** 2 + Y ** 2) / (0.5e-3) ** 2)
    good = _decentred_fit_score(xs, opl, wgt, disc, None, 2)
    poor = _decentred_fit_score(xs, opl, wgt, disc, None, 1)
    assert good < 1e-6 * poor, (good, poor)
    # and the weighted restriction is scored on the same footing
    w_in = np.where(disc, 1.0, 1e-4)
    assert _decentred_fit_score(xs, opl, wgt, disc, w_in, 2) < 1e-6 * poor


def test_an_inadmissible_candidate_cannot_win():
    """``inf`` on no usable weight, so a candidate that cannot be scored is
    never preferred over one that can."""
    n = 21
    xs = np.linspace(-1.0e-3, 1.0e-3, n)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    disc = np.ones_like(X, dtype=bool)
    opl = np.full_like(X, np.nan)
    assert not np.isfinite(
        _decentred_fit_score(xs, opl, np.ones_like(X), disc, None, 2))
    assert not np.isfinite(
        _decentred_fit_score(xs, X.copy(), np.zeros_like(X), disc, None, 2))


# ===========================================================================
# 7 -- the restriction helper is D1 + D7, exactly
# ===========================================================================
def test_the_restriction_helper_is_the_d1_weight_and_the_d7_step_down():
    n = 60
    xs = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    disc = ((X - 30) ** 2 + (Y - 30) ** 2) <= 20.0 ** 2
    n_in = int(disc.sum())
    n_out = int(disc.size) - n_in
    w, order = _decentred_fit_restriction(disc, True, 6, 10)
    expect = float(np.sqrt(_lt._FIT_DISC_OUTSIDE_WEIGHT_REL * n_in / n_out))
    assert np.array_equal(w, np.where(disc, 1.0, expect))
    assert order == 10 and (10 + 1) * (10 + 2) * 3 // 2 <= n_in
    # the hard-mask candidate keeps the caller's order and takes no weights
    w0, order0 = _decentred_fit_restriction(disc, False, 6, 10)
    assert w0 is None and order0 == 6
    # ... and a disc too thin to constrain order 10 steps DOWN, never below
    # the caller's own order
    thin = ((X - 30) ** 2 + (Y - 30) ** 2) <= 7.0 ** 2
    _w1, order1 = _decentred_fit_restriction(thin, True, 6, 10)
    assert 6 < order1 < 10
    assert (order1 + 1) * (order1 + 2) * 3 // 2 <= int(thin.sum())
    assert (order1 + 2) * (order1 + 3) * 3 // 2 > int(thin.sum())
    # ... and it never steps below the caller's own order, even when the disc
    # cannot constrain that either (the raise is zeroed out, not inverted)
    tiny = ((X - 30) ** 2 + (Y - 30) ** 2) <= 1.0 ** 2
    _w2, order2 = _decentred_fit_restriction(tiny, True, 6, 10)
    assert order2 == 6
