"""GATE-COVERAGE WITNESS for the fit-domain restrictions (fix D5, 2026-08-06).

WHY THIS FILE EXISTS
--------------------
Three guards in ``_lens_traced.py`` used to be spelled ``... and newton_fit
!= 'spline'``.  When ``newton_fit='auto'`` was flipped to resolve to spline on
the CPU (2026-08-05, commit ``6dfc79d``) the default path crossed all three at
once and SILENTLY retired:

  * the fit-domain disc restriction -- i.e. ``fit_radius_beam_factor``, the
    remedy for the aperture:beam cliff;
  * the niche-C11 ``DECENTRED_FIT_ARBITER`` (shipped ``True``);
  * the niche-C12 ``DECENTRED_FIT_PREDICTOR`` and D7's order raise.

The flip was justified purely on parallel speed-up.  Nothing failed loudly:
the guards simply stopped running, and the 15 tests that noticed
(``c6_fit_guard`` 2, ``c11_decentred_fit_arbiter`` 7,
``c12_physics_fit_selection`` 6) read like calibration drift rather than like
a feature going missing.

That default flip has since been REVERTED (``'auto'`` -> polynomial).  This
file is not about the revert -- it is the thing that would have caught it, and
that will catch the next one.

WHAT IS ASSERTED, AND WHY IT IS NOT A NUMERIC PIN
-------------------------------------------------
One invariant, in two halves, per feature:

  A. on the SHIPPED default basis the feature must be LIVE -- toggling it must
     change the returned field.  A basis flip that makes it inert fails here.
  B. on a basis that CANNOT honour it, the call must SAY SO.  So the only way
     to make a feature inert is to also announce it, and (per the third test)
     the default path must never be announcing -- which together mean a
     feature can never again go missing quietly.

No tolerance, no measured magnitude, no BLAS-dependent quantity: every bar is
``array_equal`` or the presence of a warning.

THE ADJUDICATION BEHIND (B) -- the spline basis is not being let off lightly:
the restriction has exactly two implementations and neither exists for
``RectBivariateSpline`` (no ``weights`` argument; one NaN in its data array
makes ``.ev()`` return NaN at the grid CENTRE -- pinned below).  And the
mechanism the restriction controls -- a global least-squares fit whose every
coefficient sees every sample -- is a property of the polynomial basis.  What
is NOT claimed is that spline is safe past the cliff: measured on the E4
corrected relay it returns an ALL-ZERO exit field there, which is precisely
why the inertness must be announced rather than assumed benign.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_N, _DX = 256, 12e-6
_W = 0.60e-3


def _singlet(ap):
    return {'name': 's', 'aperture_diameter': ap, 'thicknesses': [3.0e-3],
            'surfaces': [
                {'radius': 18e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -18e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _field(cx=0.0):
    x = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-((X - cx) ** 2 + Y ** 2) / _W ** 2).astype(np.complex128)


def _apply(**kw):
    """One element call, warnings recorded, on a geometry whose aperture is
    2.5x the beam diameter -- i.e. PAST the aperture:beam cliff, which is the
    regime the restriction exists for."""
    kw.setdefault('on_undersample', 'silent')
    kw.setdefault('on_noncollimated', 'off')
    cx = kw.pop('cx', 0.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = np.asarray(la.apply_real_lens_traced(
            _field(cx), prescription=_singlet(2.5 * 2.0 * _W),
            wavelength=_WL, dx=_DX, ray_subsample=4, n_workers=1, **kw))
    return out, [str(w.message) for w in rec]


#: The announcement's own text, matched on rather than on a category, so a
#: sibling RuntimeWarning (the aperture:beam cliff guard fires in this fixture
#: too) cannot be mistaken for it.
_ANNOUNCE = 'cannot honour'


# ===========================================================================
# The code fact the whole adjudication rests on
# ===========================================================================
def test_a_nan_masked_grid_destroys_a_rect_bivariate_spline():
    """The disc restriction's historical form is a hard NaN mask outside the
    disc.  On the spline basis that is not a restriction, it is a demolition:
    ONE non-finite sample propagates through the banded solve and the
    interpolant returns NaN at the grid CENTRE, far from the masked corner.

    (The other form, D1's least-squares weights, has no home either --
    ``RectBivariateSpline`` takes no ``w``.  Both halves are checked, so the
    claim "this restriction cannot be applied on this basis" is a property of
    the code and not an opinion.)
    """
    from scipy.interpolate import RectBivariateSpline
    x = np.linspace(-1.0, 1.0, 21)
    Z = np.add.outer(x ** 2, x ** 2)
    assert np.isfinite(RectBivariateSpline(x, x, Z).ev(0.0, 0.0))
    Zm = Z.copy()
    Zm[0, 0] = np.nan                       # one masked corner sample
    v = RectBivariateSpline(x, x, Zm).ev(np.array([0.0]), np.array([0.0]))
    assert not np.isfinite(v).all(), (
        'a NaN-masked sample no longer poisons RectBivariateSpline -- if that '
        'is really true, the D5 adjudication must be redone: the disc '
        'restriction could then be applied on the spline basis after all')
    import inspect
    assert 'w' not in inspect.signature(RectBivariateSpline.__init__).parameters


# ===========================================================================
# A -- every fit-domain feature is LIVE on the shipped default basis
# ===========================================================================
def test_the_disc_restriction_is_live_on_the_shipped_default_basis():
    """``fit_radius_beam_factor`` must CHANGE the answer with whatever
    ``newton_fit='auto'`` resolves to.  This is the assertion the 2026-08-05
    basis flip would have failed."""
    a, _ = _apply(fit_radius_beam_factor=None, on_aperture_beam='silent')
    b, _ = _apply(fit_radius_beam_factor=2.0, on_aperture_beam='silent')
    assert not np.array_equal(a, b), (
        "fit_radius_beam_factor is INERT on the resolved default basis -- the "
        "aperture:beam cliff remedy is not running by default")


# niche C11's own f/3 fixture, restated here so this witness shares no helper
# with the file it is guarding: an f/3 N-BK7 singlet, beam 1.0 mm at 0.2 w
# off axis, ``fit_radius_beam_factor=2`` so the disc is live and
# ``ray_subsample=8`` so D7's order raise clears its sample-count step-down.
# C11 measured the arbiter to pick CONCENTRIC there where the constant gate
# picks off-centre, which is what makes the flag observable at all.
_C11_N, _C11_DX, _C11_W = 512, 30e-6, 1.0e-3
_C11_F3 = {'name': 'd5-f3', 'aperture_diameter': 10e-3,
           'thicknesses': [3.0e-3], 'surfaces': [
               {'radius': 30e-3, 'glass_before': 'air',
                'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                'conic_y': None, 'aspheric_coeffs': None,
                'aspheric_coeffs_y': None},
               {'radius': -30e-3, 'glass_before': 'N-BK7',
                'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                'conic_y': None, 'aspheric_coeffs': None,
                'aspheric_coeffs_y': None}]}


def _apply_c11(arbiter, **kw):
    c = 0.2 * _C11_W
    x = (np.arange(_C11_N) - _C11_N // 2) * _C11_DX
    E = np.exp(-(((x[None, :] - c) ** 2 + x[:, None] ** 2) / _C11_W ** 2)
               ).astype(np.complex128)
    opts = dict(prescription=_C11_F3, wavelength=_WL, dx=_C11_DX,
                ray_subsample=8, n_workers=1, fit_radius_beam_factor=2.0,
                carrier=np.inf, beam_centre=(c, 0.0),
                on_undersample='silent', on_noncollimated='silent',
                on_aperture_beam='silent')
    opts.update(kw)
    old = (LT.DECENTRED_FIT_ARBITER, LT.DECENTRED_FIT_PREDICTOR)
    LT.DECENTRED_FIT_ARBITER = bool(arbiter)
    LT.DECENTRED_FIT_PREDICTOR = False
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = np.asarray(la.apply_real_lens_traced(E, **opts))
        return out, [str(w.message) for w in rec]
    finally:
        LT.DECENTRED_FIT_ARBITER, LT.DECENTRED_FIT_PREDICTOR = old


def test_the_arbiter_is_live_on_the_shipped_default_basis():
    """``DECENTRED_FIT_ARBITER`` ships ``True`` (niche C11).  On the resolved
    default basis, flipping it must change a DECENTRED call -- otherwise the
    flag is documentation for a code path nobody takes."""
    assert LT.DECENTRED_FIT_ARBITER is True, (
        'this witness is written against the shipped-ON arbiter; if the '
        'default changed, the C11 decision changed and this file needs a '
        'deliberate update, not a green tick')
    a, _ = _apply_c11(True)
    b, _ = _apply_c11(False)
    assert not np.array_equal(a, b), (
        'DECENTRED_FIT_ARBITER is a no-op on the resolved default basis -- '
        'niche C11 has been silently retired')


def test_the_arbiter_is_announced_when_the_basis_cannot_honour_it():
    """... and on ``newton_fit='spline'`` it really is a no-op -- announced,
    naming the flag, instead of quietly gone.  This is the exact regression
    the 2026-08-05 default flip shipped."""
    a, msgs = _apply_c11(True, newton_fit='spline')
    b, _ = _apply_c11(False, newton_fit='spline')
    assert np.array_equal(a, b), (
        'the arbiter now runs on the spline basis -- if that is intended, the '
        'D5 adjudication has changed and this file must be redone')
    assert [m for m in msgs
            if 'cannot honour' in m and 'DECENTRED_FIT_ARBITER' in m], msgs


# ===========================================================================
# B -- a basis that cannot honour a restriction must SAY SO
# ===========================================================================
@pytest.mark.parametrize('knob,extra', [
    ('fit_radius_beam_factor', dict(fit_radius_beam_factor=2.0)),
    ('decentred_fit_poly_order', dict(fit_radius_beam_factor=2.0,
                                      decentred_fit_poly_order=10)),
])
def test_an_inert_fit_domain_knob_is_announced_not_silent(knob, extra):
    """On ``newton_fit='spline'`` the knob really is inert -- and the call
    says so, naming the knob.  Both halves matter: the first is what makes it
    a real bypass, the second is what stops the next one being silent."""
    kw = dict(newton_fit='spline', on_aperture_beam='silent', **extra)
    a, msgs = _apply(**kw)
    b, _ = _apply(newton_fit='spline', on_aperture_beam='silent',
                  fit_radius_beam_factor=None)
    assert np.array_equal(a, b), (
        f'{knob} is no longer inert on the spline basis -- if it now applies, '
        f'the D5 adjudication has changed and this file must be redone')
    hits = [m for m in msgs if 'cannot honour' in m and knob in m]
    assert hits, (
        f'the spline basis silently ignored {knob}: {msgs}')


def test_the_inert_knob_can_be_made_fatal_and_acknowledged():
    """``on_fit_domain_basis`` is a real three-way disposition, validated the
    same way its siblings are."""
    with pytest.raises(ValueError, match='cannot honour'):
        _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
               on_aperture_beam='silent', on_fit_domain_basis='error')
    _, msgs = _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
                     on_aperture_beam='silent', on_fit_domain_basis='silent')
    assert not [m for m in msgs if 'cannot honour' in m]


# ===========================================================================
# FINDING V4 -- the knob itself must be validated
#
# ``on_fit_domain_basis`` shipped (in the D5 fix) as the ONLY string mode knob
# in ``_lens_traced.py`` with no membership gate.  Measured on that build:
#
#   on_fit_domain_basis='warm'   : NO RAISE   announcements=1
#   on_fit_domain_basis='Error'  : NO RAISE   announcements=1
#   on_fit_domain_basis=None     : NO RAISE   announcements=1
#   on_fit_domain_basis=1        : NO RAISE   announcements=1
#   on_fit_domain_basis=''       : NO RAISE   announcements=1
#   on_fit_domain_basis='ignore' : NO RAISE   announcements=1
#
# Every one of those fell through to the 'warn' branch.  Two consequences,
# and the second is the one with teeth:
#
#   1. a caller who asked for the combination to be FATAL ('Error') got a
#      RuntimeWarning and a returned field -- and past the aperture:beam
#      cliff that field can be ALL ZERO (see the module docstring), which is
#      exactly the silent-wrong-answer path the knob exists to close;
#   2. 'ignore' -- the vocabulary every ``on_*`` knob in
#      ``lumenairy/propagators/carrier.py`` uses for suppression -- was
#      accepted AND INERT, i.e. it neither silenced nor complained.
#
# The valid set is now ('warn', 'error', 'silent'), with 'ignore' and 'off'
# accepted as aliases for 'silent' (the same collision ``on_noncollimated``
# resolves with ``_NONCOL_ALIASES``).  Everything else raises at ENTRY.
# ===========================================================================
_VALID_FDB = ('warn', 'error', 'silent')
_FDB_ALIASES = ('ignore', 'off')


@pytest.mark.parametrize('bad', [
    'warm', 'Error', 'ERROR', 'Silent', 'SILENT', 'Warn', 'erorr',
    'warn ', ' warn', '', 'none', 'None', None, 1, 0, True, 1.0,
    ('warn',), ['warn'], b'warn', {'k': 'warn'},
])
def test_a_junk_fit_domain_basis_raises_and_names_the_valid_set(bad):
    """The D4 defect class, applied to the knob D5 added.

    Case matters ('Error' is not 'error'), whitespace matters, and non-strings
    raise rather than being coerced -- the same contract the ``gap_kernel`` and
    ``on_noncollimated`` gates hold.  The message must name the knob AND the
    whole valid set, so the caller does not have to read the source to find
    out what to pass instead."""
    with pytest.raises(ValueError) as ei:
        _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
               on_aperture_beam='silent', on_fit_domain_basis=bad)
    msg = str(ei.value)
    assert 'on_fit_domain_basis' in msg, msg
    for good in _VALID_FDB:
        assert repr(good) in msg or f"'{good}'" in msg, (
            f'the refusal does not name the valid value {good!r}: {msg}')


def test_the_junk_gate_fires_before_any_propagation():
    """AT ENTRY, like its siblings -- not after a minute of ray tracing, and
    not only on the branch that happens to read the knob.

    The announcement only fires on the spline basis with an inert fit-domain
    knob; a gate wired next to that read would leave every OTHER call silently
    accepting junk.  So the refusal must also fire on the DEFAULT basis, where
    the knob is never consulted."""
    with pytest.raises(ValueError, match='on_fit_domain_basis'):
        _apply(on_fit_domain_basis='Error')          # default basis, no spline
    with pytest.raises(ValueError, match='on_fit_domain_basis'):
        _apply(newton_fit='polynomial', on_fit_domain_basis='ignore_me')


@pytest.mark.parametrize('mode', _VALID_FDB)
def test_each_valid_mode_behaves_as_named(mode):
    """'warn' warns and returns, 'error' RAISES where the announcement would
    have fired, 'silent' returns with nothing said.  Pre-gate, all three of
    these behaved as 'warn' for any misspelling of any of them."""
    kw = dict(newton_fit='spline', fit_radius_beam_factor=2.0,
              on_aperture_beam='silent', on_fit_domain_basis=mode)
    if mode == 'error':
        with pytest.raises(ValueError, match=_ANNOUNCE):
            _apply(**kw)
        return
    out, msgs = _apply(**kw)
    hits = [m for m in msgs if _ANNOUNCE in m]
    assert out is not None and np.isfinite(out).all()
    if mode == 'warn':
        assert len(hits) == 1, (
            f"'warn' must announce exactly once, got {len(hits)}: {msgs}")
        assert 'fit_radius_beam_factor' in hits[0], hits[0]
    else:
        assert not hits, f"'silent' still announced: {msgs}"


@pytest.mark.parametrize('alias', _FDB_ALIASES)
def test_the_carrier_style_silencers_are_honoured_as_aliases(alias):
    """``carrier.py``'s ``_check_guard_action`` spells suppression 'ignore';
    this signature's siblings spell it 'silent'.  One knob, two house styles
    to guess between -- so both are honoured, and honoured means SILENCED.

    Before the gate, 'ignore' was accepted and inert: it produced the warning
    it was asked to suppress, with no diagnostic.  Rejecting it outright would
    also have been honest; silently doing the opposite was not."""
    out, msgs = _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
                       on_aperture_beam='silent', on_fit_domain_basis=alias)
    assert not [m for m in msgs if _ANNOUNCE in m], (
        f'on_fit_domain_basis={alias!r} was accepted but still announced: '
        f'{msgs}')
    # ...and it means exactly 'silent', bit for bit
    ref, _ = _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
                    on_aperture_beam='silent', on_fit_domain_basis='silent')
    assert np.array_equal(out, ref)


def test_the_default_is_warn_and_silent_on_the_default_basis():
    """Two halves of one claim, because the gate must not have changed either.

    The signature default is 'warn' (so a caller who passes nothing gets the
    announcement when it applies), and on the SHIPPED default basis nothing is
    announced at all (there is nothing inert to announce).  The second half is
    what makes 'the default path is quiet' a fact rather than a side effect of
    the knob defaulting to a silencer."""
    import inspect
    sig = inspect.signature(la.apply_real_lens_traced)
    assert sig.parameters['on_fit_domain_basis'].default == 'warn'
    assert LT._traced_kwarg_defaults()['on_fit_domain_basis'] == 'warn', (
        'the chain-forwarded default drifted from the signature default')
    for kw in ({}, dict(fit_radius_beam_factor=2.0, on_aperture_beam='silent'),
               dict(newton_fit='auto', fit_radius_beam_factor=2.0,
                    on_aperture_beam='silent'),
               dict(decentred_fit_poly_order=6, on_aperture_beam='silent')):
        out, msgs = _apply(**kw)
        assert np.isfinite(out).all()
        assert not [m for m in msgs if _ANNOUNCE in m], (
            f'the default on_fit_domain_basis announced on {kw}: {msgs}')


#: Knobs on ``apply_real_lens_traced`` whose signature default is a string but
#: which an ALL-DEFAULT call does not refuse junk for.  Measured, not assumed:
#:
#:   on_undersample : IS validated -- but at :7404, INSIDE the branch that only
#:       runs when the undersampling condition trips.  A call that is well
#:       sampled never reaches the check, and the knob's other two reads are
#:       ``!= 'silent'``, so junk behaves as 'warn'.
#:   caustic_band   : never read unless ``caustic=`` is requested; on the
#:       default path it is accepted and inert.
#:
#: Both are PRE-EXISTING and outside finding V4 (which named
#: ``on_fit_domain_basis``).  They are recorded here rather than left for the
#: next sweep to rediscover.  This ledger may only ever SHRINK -- adding a name
#: to it is how a new ungated knob would be smuggled past the test below, so
#: doing that instead of writing the gate is the failure mode to watch for.
_KNOWN_UNGATED = frozenset({'on_undersample', 'caustic_band'})


def test_no_new_string_mode_knob_ships_without_a_gate():
    """The reason V4 existed at all: ``on_fit_domain_basis`` was the one knob
    in this signature nobody had gated, in the very commit that gated the
    others.  So sweep the whole signature rather than naming knobs one at a
    time -- the NEXT one added without a gate fails here, which is the only
    version of this test that keeps working.

    Scope: parameters whose SIGNATURE DEFAULT is a string, i.e. the
    enum-shaped ones.  A junk value must raise ``ValueError`` on an
    all-default call -- not warn, not fall through to whichever branch the
    equality test happens to miss."""
    import inspect
    sig = inspect.signature(la.apply_real_lens_traced)
    ungated = set()
    detail = []
    for pname, p in sig.parameters.items():
        if not isinstance(p.default, str):
            continue
        try:
            _apply(**{pname: 'zzz_not_a_valid_mode'})
        except ValueError:
            continue
        except Exception as exc:                     # noqa: BLE001
            ungated.add(pname)
            detail.append(f'{pname} raised {type(exc).__name__}, not '
                          f'ValueError')
        else:
            ungated.add(pname)
            detail.append(f'{pname} accepted junk silently')
    assert 'on_fit_domain_basis' not in ungated, (
        'on_fit_domain_basis lost its entry gate -- finding V4 is back: a '
        "caller asking for 'error' would get a warning and a returned field")
    assert ungated <= _KNOWN_UNGATED, (
        'a string mode knob on apply_real_lens_traced does not refuse an '
        'unknown value, and is not in the disclosed ledger: '
        + '; '.join(sorted(detail)))


def test_the_default_basis_never_emits_the_inapplicable_guard():
    """The other half of the invariant, and the reason it cannot be satisfied
    by wiring the announcement and leaving the feature dead: on the DEFAULT
    basis the guard must never fire.  So 'feature live' and 'feature
    announced' are mutually exclusive there, and a future flip has to pick
    one -- neither of which is silent."""
    for kw in (dict(fit_radius_beam_factor=2.0),
               dict(fit_radius_beam_factor=2.0,
                    decentred_fit_poly_order=10)):
        _, msgs = _apply(on_aperture_beam='silent', **kw)
        assert not [m for m in msgs if 'cannot honour' in m], (
            f'the SHIPPED default basis cannot honour {kw} -- a fit-domain '
            f'guard has been retired from the default path: {msgs}')


def test_an_inert_restriction_no_longer_silences_the_aperture_beam_guard():
    """The live half of the defect, not just the bookkeeping.

    The aperture:beam cliff warning is skipped whenever a ray-fit disc will be
    applied, because the disc IS the remedy.  On a basis that cannot apply it
    the remedy is not in force -- so before this fix, passing the (inert)
    ``fit_radius_beam_factor`` SILENCED the warning about exactly the failure
    it had stopped preventing.  Now it does not.
    """
    _, msgs = _apply(newton_fit='spline', fit_radius_beam_factor=2.0,
                     on_aperture_beam='warn', on_fit_domain_basis='silent')
    assert [m for m in msgs if 'aperture:beam ratio' in m], (
        'an INERT fit_radius_beam_factor still silences the aperture:beam '
        f'cliff guard on the spline basis: {msgs}')
    # ... and on the basis that CAN apply it, the disc still suppresses the
    # warning, because there the remedy really is in force (no behaviour
    # change on the default path).
    _, msgs_poly = _apply(newton_fit='polynomial',
                          fit_radius_beam_factor=2.0, on_aperture_beam='warn')
    assert not [m for m in msgs_poly if 'aperture:beam ratio' in m]
