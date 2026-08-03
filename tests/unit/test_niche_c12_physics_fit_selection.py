"""Niche C12 -- the decentred beam's ray-fit branch is DERIVED, not compared.

Audit ``docs/audits/C12_PHYSICS_FIT_SELECTION_2026_08_03.md``.

**What niche C11 left.**  ``DECENTRED_FIT_ARBITER`` builds both candidate ray
fits and keeps the one with the smaller beam-weighted OPL residual.  That
works, but it is a measurement: it cannot say WHY the crossover sits at 0.57 w
on an f/3 singlet and at 0 w on an f/6 one, and it says nothing about any
decentre other than the one in front of it.

**The derivation.**  The traced OPL is a fixed function of the ENTRANCE
position -- moving the beam moves neither it nor the launch grid.  A
total-degree-``m`` least-squares fit reproduces the degree-``<= m`` part
exactly, so each candidate's residual is IDENTICALLY the residual of fitting
its own spectral tail ``W_>m``.  The tail is decentre-free, so the whole
``u``-dependence is geometric: the concentric disc is sized from the
ORIGIN-referenced second moment and inflates by ``rho = sqrt(1 + 2 u^2)``,
while the off-centre disc and the beam translate together.  Each shell scales
as ``s^n``, so the concentric residual runs as ``rho^m_eff`` with ``m_eff`` the
tail's spectral first moment, and the crossover follows in closed form.  No
fitted constant appears anywhere.

**What is pinned here.**  Every numeric bar is a RATIO between two arms
measured in the same process on the same fixture, or an exact-arithmetic
identity -- there is no absolute bar on a BLAS-dependent magnitude anywhere.

1. the flag ships ``False`` and is a genuine no-op in that state -- the
   crossover is never even computed;
2. the fail-before restores BOTH earlier eras bit for bit (the C11 arbiter and
   the v5.32 gate);
3. the score-weight floor ships inert, and is the bare beam intensity at 0.0;
4. the spectral first moment is the inflation law's exponent, as exact
   arithmetic on a synthetic spectrum;
5. the closed-form crossover INVERTS the inflation law exactly, and its
   degenerate cases return what the docstring says;
6. predictor and arbiter are the same decision when the model falls back --
   which is what makes a disagreement mean something;
7. the spectral-tail identity ``(I - Pi_m) W == (I - Pi_m) W_>m`` holds on a
   real traced map, to the resolution of the box fit;
8. on the f/3 fixture the predictor keeps the concentric branch at a decentre
   where the arbiter has already switched -- an exact bitwise identity at a
   decentre BISECTED in this process, and the direction the fit-domain-free
   oracle agrees with on both builds measured;
9. on the f/6 fixture the two agree, so 8 is not a blanket difference;
10. a disagreement is NEVER silent, and the warning names both score pairs;
11. niche C1's null contract survives in every flag state;
12. the candidate that is scored is still the candidate that is applied.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lt
from lumenairy.elements._lens_traced import (
    _Cheb2DEvaluator,
    _decentred_fit_crossover,
    _decentred_fit_restriction,
    _decentred_fit_score,
    _decentred_fit_score_weight,
    _decentred_fit_spectral_moment,
    _decentred_fit_spectrum,
)

_WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent',
            on_aperture_beam='silent')


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


# niche C1's own fixtures, unchanged.
_N, _DX, _W = 512, 30e-6, 1.0e-3
_SLOW = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'c12-f6')
_FAST = _singlet(30e-3, -30e-3, 3.0e-3, 'N-BK7', 10e-3, 'c12-f3')

#: The two selectors' crossovers on the f/3 fixture are BLAS-dependent and the
#: band between them is narrow, so it is BISECTED here rather than pinned.
#: Measured (``c12_oracle_crossover.py``, both builds):
#:
#:     build            oracle    arbiter   predictor
#:     Windows / MKL    0.5715    0.5453    0.5717
#:     Linux / OpenBLAS 0.5906    0.5555    0.5717
#:
#: -- the predictor's own number is build-invariant to four digits because it
#: comes from the spectrum rather than from a difference of two nearly-equal
#: least-squares residuals, and the ordering arbiter < predictor <= oracle
#: holds on both.  A test that pinned 0.555 passed on MKL and failed on
#: OpenBLAS; the band is measured in-process instead.
_XOVER_CACHE = {}


def _gauss(n, dx, w, cx=0.0, cy=0.0):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) / w ** 2)
                  ).astype(np.complex128)


def _apply(c, predictor, arbiter, presc=_FAST, tell=None, **kw):
    """One element call with the beam physically at ``(c, 0)``.

    ``predictor`` and ``arbiter`` are POSITIONAL and mandatory: both flags ship
    ``False``, and an arm that inherited a default would silently stop
    exercising what it names the moment that default moved.
    """
    opts = dict(prescription=presc, wavelength=_WL, dx=_DX, ray_subsample=8,
                n_workers=1, fit_radius_beam_factor=2.0, carrier=np.inf,
                beam_centre=(c, 0.0) if tell is None else tell, **_TKW)
    opts.update(kw)
    old = (_lt.DECENTRED_FIT_PREDICTOR, _lt.DECENTRED_FIT_ARBITER)
    _lt.DECENTRED_FIT_PREDICTOR = bool(predictor)
    _lt.DECENTRED_FIT_ARBITER = bool(arbiter)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                _gauss(_N, _DX, _W, c, 0.0), **opts))
    finally:
        _lt.DECENTRED_FIT_PREDICTOR, _lt.DECENTRED_FIT_ARBITER = old


def _picks_concentric(u, predictor, arbiter, presc=_FAST):
    """Which branch one element call actually took, read from the library's own
    decision quantities rather than inferred from the field."""
    scores, ustar = [], []
    o_s, o_x = _lt._decentred_fit_score, _lt._decentred_fit_crossover

    def s_spy(*a):
        v = o_s(*a)
        scores.append(float(v))
        return v

    def x_spy(*a):
        v = o_x(*a)
        ustar.append((float(a[0]), float(v)))
        return v

    _lt._decentred_fit_score = s_spy
    _lt._decentred_fit_crossover = x_spy
    try:
        _apply(u * _W, predictor, arbiter, presc=presc)
    finally:
        _lt._decentred_fit_score = o_s
        _lt._decentred_fit_crossover = o_x
    if predictor:
        return bool(ustar and ustar[-1][0] <= ustar[-1][1])
    return bool(len(scores) >= 2 and scores[1] <= scores[0])


def _crossover(predictor, arbiter, presc=_FAST, lo=0.06, hi=1.5, n=11):
    """Bisect one selector's own crossover on ``presc``.

    ``lo`` sits ABOVE niche C1's null gate (0.05 w): below it no selector runs
    at all, and a bisection started there would report a choice never made.
    """
    key = (predictor, arbiter, presc['name'])
    if key in _XOVER_CACHE:
        return _XOVER_CACHE[key]
    if not _picks_concentric(lo, predictor, arbiter, presc):
        _XOVER_CACHE[key] = 0.0
        return 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if _picks_concentric(mid, predictor, arbiter, presc):
            lo = mid
        else:
            hi = mid
    _XOVER_CACHE[key] = 0.5 * (lo + hi)
    return _XOVER_CACHE[key]


# ===========================================================================
# 1-3  the switches
# ===========================================================================
def test_the_predictor_ships_off_and_is_never_even_computed():
    """Flag-off is a path NOT TAKEN, not a result discarded."""
    assert _lt.DECENTRED_FIT_PREDICTOR is False
    seen = []
    orig = _lt._decentred_fit_crossover

    def spy(*a, **k):
        seen.append(a)
        return orig(*a, **k)

    _lt._decentred_fit_crossover = spy
    try:
        # a decentre well above the C1 null gate, on the arbiter arm, i.e. a
        # call that reaches the branch decision by every other route
        _apply(0.30 * _W, False, True)
        assert seen == []
    finally:
        _lt._decentred_fit_crossover = orig


def test_the_fail_before_restores_both_earlier_eras_bit_for_bit():
    """``predictor=False`` is the C11 arbiter exactly; ``arbiter=False`` on top
    of it is the v5.32 gate exactly -- and the two eras genuinely differ on
    this fixture, so neither identity is vacuous."""
    c = 0.30 * _W
    arb = _apply(c, False, True)
    gate = _apply(c, False, False)
    assert not np.array_equal(arb, gate)          # the eras differ
    assert np.array_equal(_apply(c, False, True), arb)
    assert np.array_equal(_apply(c, False, False), gate)


def test_the_score_weight_floor_ships_inert():
    xs = (np.arange(24) - 12) * 1e-4
    bare = np.exp(-2.0 * ((xs[:, None] - 3e-4) ** 2 + xs[None, :] ** 2)
                  / (1e-3 ** 2))
    assert _lt._DECENTRED_FIT_SCORE_FLOOR == 0.0
    assert np.array_equal(
        _decentred_fit_score_weight(xs, 3e-4, 0.0, 1e-3), bare)
    floored = _decentred_fit_score_weight(xs, 3e-4, 0.0, 1e-3, floor=1e-3)
    assert np.array_equal(floored, np.maximum(bare, 1e-3))
    assert float(floored.min()) == 1e-3


# ===========================================================================
# 4-5  the algebra of the law
# ===========================================================================
def test_the_spectral_first_moment_is_the_inflation_exponent():
    """``m_eff`` is ``d log T / d log rho`` at ``rho = 1``, so on a spectrum
    with ONE shell above the order it must return that shell's degree exactly,
    and on two shells the energy-weighted mean of the two."""
    S = np.zeros(13)
    S[8] = 1.0
    assert _decentred_fit_spectral_moment(S, 6, 0.5) == pytest.approx(8.0)
    # two shells, chosen so the weights are exactly 1:1 at sigma = 0.5
    S = np.zeros(13)
    S[8] = 1.0
    S[10] = 1.0 / 0.5 ** 2
    assert _decentred_fit_spectral_moment(S, 6, 0.5) == pytest.approx(9.0)
    # ... and the documented fall-back when the tail carries nothing
    assert _decentred_fit_spectral_moment(np.zeros(13), 6, 0.5) == 8.0
    assert _decentred_fit_spectral_moment(None, 6, 0.5) == 8.0


def test_the_crossover_inverts_the_inflation_law_exactly():
    """``u*`` is defined by ``E_conc(u*) = E_off`` under ``E ~ rho^m``, so
    propagating ``E_conc`` from ``u`` to ``u*`` must land on ``E_off``."""
    for u, ec, eo, m in ((0.3, 1e-12, 4e-12, 8.0),
                         (0.9, 5e-11, 2e-12, 7.25),
                         (0.05, 2e-13, 2e-13, 8.0)):
        us = _decentred_fit_crossover(u, ec, eo, m)
        rho = np.sqrt(1.0 + 2.0 * u * u)
        rho_s = np.sqrt(1.0 + 2.0 * us * us)
        assert ec * (rho_s / rho) ** m == pytest.approx(eo, rel=1e-12)
    # the documented degenerate answers
    assert _decentred_fit_crossover(0.3, 0.0, 1e-12, 8.0) == 0.0
    assert _decentred_fit_crossover(0.3, 1e-12, 0.0, 8.0) == 0.0
    assert _decentred_fit_crossover(0.3, 1e-12, np.inf, 8.0) == np.inf
    assert np.isnan(_decentred_fit_crossover(0.3, 1e-12, 1e-12, 0.0))
    assert np.isnan(_decentred_fit_crossover(np.nan, 1e-12, 1e-12, 8.0))


def test_the_closed_form_and_the_raw_comparison_are_one_decision():
    """``u <= u*`` and ``E_conc <= E_off`` are the SAME test for any positive
    exponent -- which is what makes a disagreement between the predictor and
    the arbiter a statement about the MODEL rather than about arithmetic."""
    rng = np.random.default_rng(20260803)
    for _ in range(400):
        u = float(rng.uniform(1e-3, 2.0))
        ec = float(10.0 ** rng.uniform(-14, -9))
        eo = float(10.0 ** rng.uniform(-14, -9))
        m = float(rng.uniform(4.0, 12.0))
        us = _decentred_fit_crossover(u, ec, eo, m)
        assert bool(u <= us) == bool(ec <= eo)


# ===========================================================================
# 6-7  the spectral tail, on a real traced map
# ===========================================================================
def _capture_traced_opl(c, presc=_FAST):
    """The launch axes and the UNMASKED traced OPL of one element call."""
    seen = []
    orig = _Cheb2DEvaluator.__init__

    class _Stop(Exception):
        pass

    def spy(zelf, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(zelf, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((np.asarray(xs_in), np.asarray(values)))
        if len(seen) >= 3:
            raise _Stop()

    _Cheb2DEvaluator.__init__ = spy
    old = (_lt._DECENTRE_GATE_PIXELS, _lt._DECENTRE_GATE_W_FRAC)
    _lt._DECENTRE_GATE_PIXELS = 0.0
    _lt._DECENTRE_GATE_W_FRAC = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                la.apply_real_lens_traced(
                    _gauss(_N, _DX, _W, c, 0.0), prescription=presc,
                    wavelength=_WL, dx=_DX, ray_subsample=8, n_workers=1,
                    fit_radius_beam_factor=2.0, carrier=np.inf,
                    beam_centre=(c, 0.0), **_TKW)
            except _Stop:
                pass
    finally:
        _Cheb2DEvaluator.__init__ = orig
        _lt._DECENTRE_GATE_PIXELS, _lt._DECENTRE_GATE_W_FRAC = old
    assert len(seen) >= 3
    return seen[0][0], seen[2][1]


def test_the_spectral_tail_carries_the_whole_candidate_residual():
    """``(I - Pi_m) W == (I - Pi_m) W_>m``: fitting the tail alone must
    reproduce the residual of fitting the map, because the fit annihilates the
    degree-``<= m`` part exactly.  Comparative -- a RATIO of two residuals
    measured in the same process, with the band set by what the order-``q`` box
    fit itself leaves over (also measured here, in the same units)."""
    c = 0.30 * _W
    xs, opl = _capture_traced_opl(c)
    r2 = xs[:, None] ** 2 + xs[None, :] ** 2
    disc_c = r2 <= (2.0 * np.sqrt(2 * c * c + _W * _W)) ** 2
    wgt = _decentred_fit_score_weight(xs, c, 0.0, _W)
    wc, oc = _decentred_fit_restriction(disc_c, False, 6, 10)
    _S, tails, _resid = _decentred_fit_spectrum(xs, opl, 14, (oc,),
                                                weight=wgt)
    full = _decentred_fit_score(xs, opl, wgt, disc_c, wc, oc)
    tail = _decentred_fit_score(xs, tails[oc], wgt, disc_c, wc, oc)
    gap = _decentred_fit_score(xs, opl - tails[oc], wgt, disc_c, wc, oc)
    assert np.isfinite(full) and full > 0.0
    # the two residuals agree to within what is provably left out -- the
    # beyond-q content that the same restriction cannot absorb
    assert abs(full - tail) <= 2.0 * gap
    # ... and that gap is itself small against the residual, i.e. the fixture's
    # spectrum IS resolved at this order (design 121's is not; audit S3.4)
    assert gap < full


def test_the_shell_spectrum_of_a_rotationally_symmetric_lens_is_even():
    """A sanity pin on the spectrum itself: a centred spherical singlet's OPL
    is even in both launch coordinates, so every ODD total-degree shell must be
    at round-off against its even neighbours."""
    xs, opl = _capture_traced_opl(0.30 * _W)
    S, _t, _r = _decentred_fit_spectrum(xs, opl, 12, ())
    even = np.array([S[n] for n in range(2, 13, 2)])
    odd = np.array([S[n] for n in range(3, 12, 2)])
    assert float(odd.max()) < 1e-9 * float(even.max())


# ===========================================================================
# 8-10  the two selectors, and the warning
# ===========================================================================
def test_the_predictor_holds_the_concentric_branch_longer_than_the_arbiter():
    """The two selectors' crossovers are BISECTED in this process (they are
    BLAS-dependent; see the ``_XOVER_CACHE`` note), and the predictor's must be
    the LARGER -- which is the direction the fit-domain-free spline oracle
    backs on both builds measured.

    The consequence is then pinned as an exact bitwise identity rather than as
    a threshold: at a decentre inside the band, the predictor's field IS the
    forced-concentric arm and the arbiter's is not.
    """
    u_arb = _crossover(False, True)
    u_pred = _crossover(True, False)
    assert u_arb > 0.0 and u_pred > 0.0
    assert u_pred > u_arb
    c = 0.5 * (u_arb + u_pred) * _W
    # the reference arm: the gate itself told to keep the concentric branch at
    # every decentre, which is what "concentric" means with no selector at all
    old = _lt._DECENTRE_GATE_W_FRAC
    _lt._DECENTRE_GATE_W_FRAC = float('inf')
    try:
        conc = _apply(c, False, False)
    finally:
        _lt._DECENTRE_GATE_W_FRAC = old
    assert np.array_equal(_apply(c, True, False), conc)
    assert not np.array_equal(_apply(c, False, True), conc)


def test_the_two_selectors_agree_on_the_slow_fixture():
    """On the f/6 fixture the off-centre branch wins from the first pixel by
    both routes, so the band above is a band, not a blanket difference."""
    assert _crossover(True, False, presc=_SLOW) == 0.0
    assert _crossover(False, True, presc=_SLOW) == 0.0
    for u in (0.20, 0.55, 1.00):
        c = u * _W
        assert np.array_equal(_apply(c, True, False, presc=_SLOW),
                              _apply(c, False, True, presc=_SLOW))


def test_a_disagreement_is_never_silent_and_names_both_scores():
    c = 0.5 * (_crossover(False, True) + _crossover(True, False)) * _W
    old = (_lt.DECENTRED_FIT_PREDICTOR, _lt.DECENTRED_FIT_ARBITER)
    _lt.DECENTRED_FIT_PREDICTOR, _lt.DECENTRED_FIT_ARBITER = True, False
    try:
        with pytest.warns(RuntimeWarning, match='PREDICTOR') as rec:
            la.apply_real_lens_traced(
                _gauss(_N, _DX, _W, c, 0.0), prescription=_FAST,
                wavelength=_WL, dx=_DX, ray_subsample=8, n_workers=1,
                fit_radius_beam_factor=2.0, carrier=np.inf,
                beam_centre=(c, 0.0), **_TKW)
    finally:
        _lt.DECENTRED_FIT_PREDICTOR, _lt.DECENTRED_FIT_ARBITER = old
    msg = ' '.join(str(w.message) for w in rec)
    for token in ('modelled OPL residuals', "arbiter's own measured residuals",
                  'crossover u*', 'spectral exponent m_eff',
                  'CONCENTRIC', 'OFF-CENTRE'):
        assert token in msg


# ===========================================================================
# 11-12  the contracts C11 and C1 already hold, re-checked in the new state
# ===========================================================================
@pytest.mark.parametrize('presc', [_SLOW, _FAST])
@pytest.mark.parametrize('frac', [1e-9, 0.02, 0.049])
def test_the_c1_null_contract_survives_with_the_predictor_on(presc, frac):
    """Below niche C1's null gate the predictor is inert BY CONSTRUCTION (it is
    gated on ``_beam_decentred``), so the field must be byte-identical to the
    origin-referenced arm and across every flag state."""
    c = frac * _W
    ref = _apply(c, False, False, presc=presc, tell=(0.0, 0.0))
    for pred, arb in ((False, False), (False, True), (True, False),
                      (True, True)):
        assert np.array_equal(_apply(c, pred, arb, presc=presc), ref)


def test_the_candidate_that_is_scored_is_still_the_one_applied():
    """With the predictor engaged the fit site builds MORE trials -- the
    spectral surrogate's two, plus the resolution probe -- so the guarantee
    that the APPLIED fits carry the winning trial's ``(order, weighted)`` pair
    is re-pinned from the end of the build list, which is the end the claim is
    about."""
    seen = []
    orig = _Cheb2DEvaluator.__init__

    def spy(zelf, xs_in, ys_in, values, order=6, xp=None, weights=None):
        orig(zelf, xs_in, ys_in, values, order=order, xp=xp, weights=weights)
        seen.append((int(order), weights is not None))

    c = 0.5 * (_crossover(False, True) + _crossover(True, False)) * _W
    _Cheb2DEvaluator.__init__ = spy
    try:
        _apply(c, True, False)
    finally:
        _Cheb2DEvaluator.__init__ = orig
    # the last three builds are the x_out / y_out / opl fits the Newton
    # inversion is handed, and they must agree with each other
    applied = seen[-3:]
    assert len(set(applied)) == 1
    # the predictor picks CONCENTRIC on this fixture at this decentre (test 8),
    # and the concentric candidate is the hard mask at the caller's own order
    assert applied[0] == (6, False)
    # ... and it really did cost extra builds, i.e. the model ran
    assert len(seen) > 5
