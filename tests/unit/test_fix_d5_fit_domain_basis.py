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

WHAT THE BYPASS IS, AFTER 2026-08-12 -- READ THIS BEFORE EDITING (A) OR (B)
--------------------------------------------------------------------------
The bypass did NOT retire, and it did not stay what it was either.  It NARROWED
to exactly one consumer, and the announcement narrowed with it.

``_fit_domain_basis_ok`` used to answer two different questions at once:

  Q1  can THIS BASIS restrict ITS OWN forward fit to the requested region?
  Q2  is there a requested region at all?

The adjudication above answers Q1, and it stands unchanged: no, not for
``RectBivariateSpline``, and that is a fact about the code.  But the flag was
also answering Q2, and Q2 is a property of the BEAM and the traced samples --
not of the interpolant.  So a SECOND consumer that could honour the region
never saw one: the inverse-characteristic model
(``lumenairy.elements._lens_imap``) is a global total-degree Chebyshev in EXIT
coordinates, i.e. exactly the mechanism the restriction exists to control, and
it was handed the disc on the polynomial basis and the whole launch square on
the spline basis.  Measured consequence: the two backends described DIFFERENT
maps, and the shipped backend-symmetry guard in
``test_niche_c6_stationary_phase_launch`` read 1.0600e-02 against its 5e-04
bar.

Since 2026-08-12 the region is resolved BASIS-INDEPENDENTLY whenever a
consumer that can honour it will run, and each consumer honours it iff it can:

    forward fit, polynomial basis   YES  -- NaN mask or D1 weights
    forward fit, spline basis       NO   -- the adjudication above, unchanged
    the inverse-map model           YES  -- always, on either basis

So on the spline basis ``fit_radius_beam_factor`` is inert IF AND ONLY IF the
model is not being built on that call, which on the shipped default
(``TRACED_INVERSE_MAP = False``) is every call -- which is why (A) and (B)
below are unchanged and still pass.  ``test_the_model_gets_the_same_fit_domain
_on_either_basis`` and its siblings pin the other half, and the announcement
now names its own scope so it cannot claim "no ray-fit-domain guard at all"
on a call where the model is applying one.

WHY NOT INSTEAD MAKE THE SPLINE FORWARD FIT HONOUR IT (the route not taken):
``RectBivariateSpline`` needs a full NaN-free tensor grid, so the only
restriction it can express is a rectangular SUB-LATTICE, and no rectangle is a
disc -- pinned in ``test_no_rectangular_sub_lattice_can_express_the_disc``.  A
sub-lattice restriction would therefore leave the two bases handing the model
different sample sets ANYWAY, i.e. it cannot fix the defect even in principle,
while additionally moving every spline consumer.
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


# ===========================================================================
# FIX FIT-DOMAIN SYMMETRY (2026-08-12)
#
# The bypass narrowed to the FORWARD FIT.  These pin the other half: the
# inverse-characteristic model gets the SAME fit domain on either basis, the
# route that could not have worked is refuted as a code fact, and the build
# cache cannot carry one basis's acceptance to the other.
# ===========================================================================
def _apply_imap(basis, **kw):
    """The same call as ``_apply``, with the inverse-characteristic model
    FORCED ON for this call only (``inverse_map=True`` overrides the module
    flag), and its guard record returned."""
    rec = {}
    kw.setdefault('on_aperture_beam', 'silent')
    kw.setdefault('on_fit_domain_basis', 'silent')
    out, msgs = _apply(newton_fit=basis, inverse_map=True, _imap_out=rec, **kw)
    return out, msgs, rec


#: What "the same fit domain" MEANS, in the model's own recorded numbers: the
#: sample set it fitted, the exit box it normalises in, and -- the one that is
#: a statement about the ANSWER rather than about the setup -- its own held-out
#: error.  All three moved together when the domain was basis-dependent
#: (2 809 disc samples against 32 761 launch-square ones on niche C6's
#: fixture), which is exactly why the backends stopped describing the same map.
_SAME_DOMAIN_KEYS = ('n_alive', 'n_fit_samples', 'n_detj_census',
                     'exit_half_mm', 'exit_centre_mm', 'n_terms',
                     'parity_map_opl_waves', 'parity_map_pos_m',
                     'fit_resid_opl_waves', 'fit_resid_x_in_m')


@pytest.mark.parametrize('extra', [
    pytest.param({}, id='concentric'),
    # ...and OFF AXIS, which is the branch the fix had to reach through the
    # niche-C11 arbiter: the concentric candidate's radius comes from the
    # ORIGIN-referenced moment, and that moment used to be measured only on a
    # basis that could apply the disc -- so on the spline basis there was no
    # second candidate and the arbiter could not have picked the same branch.
    pytest.param(dict(cx=0.35e-3, beam_centre=(0.35e-3, 0.0)), id='decentred'),
])
def test_the_model_gets_the_same_fit_domain_on_either_basis(extra):
    """``newton_fit`` is an interpolant choice, so it must not change what the
    inverse-map model is FITTED TO.

    This is the assertion the D5 bypass used to fail: the polynomial basis
    applied the ray-fit disc and the spline basis did not, so the model saw a
    disc on one and the whole launch square on the other.  Everything checked
    here is the model's OWN record, not a field tolerance -- the sample set,
    the exit normalisation box, and its own held-out error against the traced
    landings.
    """
    _a, _m, ra = _apply_imap('polynomial', fit_radius_beam_factor=2.0, **extra)
    _b, _m2, rb = _apply_imap('spline', fit_radius_beam_factor=2.0, **extra)
    assert ra.get('gate_open') and rb.get('gate_open'), (ra, rb)
    for k in _SAME_DOMAIN_KEYS:
        assert k in ra and ra[k] is not None, (k, ra)
        assert ra[k] == rb[k], (
            f'the inverse-map model saw a different {k} on the two '
            f'newton_fit backends ({ra[k]!r} vs {rb[k]!r}) -- the fit domain '
            f'has become basis-dependent again')


def test_the_disc_is_live_for_the_model_even_where_the_basis_cannot_fit_it():
    """The half of the bypass that DID retire, stated as behaviour.

    On the spline basis ``fit_radius_beam_factor`` cannot reach the forward
    bicubic -- but it does reach the model, so on a call that builds one the
    knob is NOT inert.  Both halves are asserted, because "the knob now does
    something" is only meaningful next to "and it does the RIGHT something":
    the restricted model must be the same object the polynomial basis builds.
    """
    _a, _m, r_on = _apply_imap('spline', fit_radius_beam_factor=2.0)
    _b, _m2, r_off = _apply_imap('spline', fit_radius_beam_factor=None)
    assert r_on['n_alive'] < r_off['n_alive'], (
        'fit_radius_beam_factor no longer restricts the inverse-map model on '
        'the spline basis -- the fit-domain symmetry fix has been reverted: '
        f'{r_on["n_alive"]} vs {r_off["n_alive"]} samples')
    _c, _m3, r_poly = _apply_imap('polynomial', fit_radius_beam_factor=2.0)
    assert r_on['n_alive'] == r_poly['n_alive']


def test_no_rectangular_sub_lattice_can_express_the_disc():
    """THE ROUTE NOT TAKEN, refuted as arithmetic rather than as an opinion.

    ``RectBivariateSpline`` needs a full NaN-free tensor grid, so the only
    fit-domain restriction that basis could express is a rectangular
    SUB-LATTICE of the launch lattice.  No rectangle is a disc: any
    sub-lattice either keeps launch nodes the disc excludes or drops nodes the
    disc keeps.  So restricting the spline that way would leave the two bases
    handing the inverse-map model DIFFERENT sample sets anyway -- it cannot
    close the defect even in principle, which is why the domain was made
    basis-independent at the consumer instead.
    """
    n = 181
    xs = np.linspace(-1.0, 1.0, n)
    R = 0.30
    disc = (xs[:, None] ** 2 + xs[None, :] ** 2) <= R ** 2
    assert disc.any() and not disc.all()
    for half in np.linspace(0.02, 1.0, 40):
        keep = np.abs(xs) <= half
        rect = keep[:, None] & keep[None, :]
        assert not np.array_equal(rect, disc), (
            'a rectangular sub-lattice reproduced the fit disc exactly -- if '
            'that is really possible the D5 route adjudication must be redone')
    # ...and the best one still misses by a wide margin, in both directions.
    best = None
    for half in np.linspace(0.02, 1.0, 200):
        keep = np.abs(xs) <= half
        rect = keep[:, None] & keep[None, :]
        sym = int((rect ^ disc).sum())
        if best is None or sym < best[0]:
            best = (sym, int((rect & ~disc).sum()), int((disc & ~rect).sum()))
    assert best[1] > 0 or best[2] > 0
    assert best[0] > 0.10 * int(disc.sum()), best


def test_a_cached_model_cannot_carry_an_acceptance_across_bases():
    """The build cache keys on the MODEL; G8's verdict is a property of the
    PAIR (model, incumbent).

    Once the fit domain is basis-independent the two backends build the SAME
    model, so without the incumbent in the key the second call would inherit
    the first's acceptance and the returned field would depend on the ORDER
    the two calls were made in.  Measured on niche C6's fixture before the tag
    was added: 1.0600e-02 backend spread one way round and 0.0 the other, from
    the same two calls.  Neither number is wrong here -- an ORDER-DEPENDENT
    answer is.
    """
    def verdicts(order):
        v = {}
        for b in order:
            _f, _m, r = _apply_imap(b, fit_radius_beam_factor=2.0)
            v[b] = (bool(r.get('engaged')), r.get('refused'))
        return v

    fwd = verdicts(('polynomial', 'spline'))
    rev = verdicts(('spline', 'polynomial'))
    assert fwd == rev, (
        'the inverse-map build cache carried one newton_fit backend\'s G8 '
        f'verdict to the other: {fwd} run one way, {rev} the other')
