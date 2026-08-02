"""The v5.32 ray-density HALO self-check
(``lumenairy.elements._lens_traced._RD_HALO_AMAX_TOL`` and friends).

WHY IT EXISTS.  ``amplitude_model='ray_density'`` already had a post-hoc energy
self-check, and that check is a scalar power sum.  It cannot see the OTHER way
this mode fails: the fitted entrance->exit map is Newton-inverted outside its
own data support, the inversion finds a spurious root, and the ray-density
amplitude hands that root real power -- depositing a LOBE at a radius no traced
ray reaches.  Measured on design 121 (docs/audits/
ENERGY_CONSERVATION_AUDIT_2026_07_31.md S2.4), a library change moved that
defect's total-power signature from 1.001058 (visibly wrong) to 0.999371
(inside every absolute band) while the lobe stayed at 3.4e-03 of the input
power at 77 % of peak.  A criterion on total power alone called it fixed.

WHAT IS PINNED HERE.

  1. It is a DIAGNOSTIC, not a filter: the returned array is byte-identical
     whether the check runs or not.  If this ever fails, the check has started
     changing physics.
  2. It does not fire on clean calls -- including a decentred beam, which is
     the case a grid-referenced (rather than centroid-referenced) radius would
     get wrong.
  3. It DOES fire on a real manufactured lobe, on a fixture that needs no
     proprietary asset: the ``REMAP_STATIONARY_PHASE_FIT_GUARD`` regression
     recorded in that flag's own note.  Both directions are pinned, so this is
     a fail-before and not only a fail-after.
  4. The radius really is the traced ray support: shrinking the factor below 1
     brings real light inside the halo annulus and the check notices.
  5. The message carries the numbers a caller needs to act, and the policy
     knob silences it.

The bound's CALIBRATION (173 element calls over the P2 design battery, the
synthetic C6 ghost fixtures and design 121's fan) lives in the constant's own
note; this file pins only that the shipped constants sit where that
calibration says, so a future edit to them has to come back through it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(r1, r2, th, z, ap, glass='N-BK7'):
    """Biconvex singlet followed by a free leg ``z`` to the exit plane."""
    return {'name': 'halo_singlet', 'aperture_diameter': ap,
            'surfaces': [_surf(r1, 'air', glass), _surf(r2, glass, 'air'),
                         _flat()],
            'thicknesses': [th, z]}


def _field(n, dx, w, rc, alpha, cx=0.0, cy=0.0):
    """Gaussian on a converging carrier sphere plus an ``alpha (r/w)^4``
    residual -- the same construction the C6 fixtures use."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


# The CLEAN fixture: small, well conditioned, exercised by most tests here.
# The grid half-width is 5.76 mm against a beam radius of 0.9 mm -- 6.4 w -- so
# the bound circle ``1.25 x r_hull`` (``r_hull`` is about ``3 w`` times the
# group's magnification) sits well inside the grid and the check has a genuine
# annulus to measure.  A grid narrower than about 4 w makes it DECLINE, by
# design; ``test_declines_when_only_the_corners_are_left`` pins that.
_CLEAN = dict(n=384, dx=30e-6, w=0.9e-3, rc=-0.20, alpha=3.0,
              r1=200e-3, r2=-200e-3, th=4e-3, z=5e-3, ap=11e-3)

# The DEFECTIVE fixture: ``probe_ghost_synthetic.py``'s 'medium, finer grid'
# cell, where turning REMAP_STATIONARY_PHASE_FIT_GUARD on takes the halo from
# 6.9e-06 to 4.6e-02 of peak.  Kept as small as still reproduces it.
_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)

_BASE_KW = dict(wavelength=_WL, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                ray_subsample=4, fit_radius_beam_factor=2.0)


def _call(spec, launch=False, guard=False, policy=None, factor=None,
          tol=None, cx=0.0, cy=0.0, bound=None, **over):
    """One element call with every halo constant and both C6 flags controlled
    and restored.  Returns ``(field, [halo warning messages])``.

    ``bound`` controls niche C8's ``REMAP_INVERSE_SUPPORT_BOUND`` (``None``
    leaves the shipped default).  It exists for ONE test - the fit-guard
    regression below, whose stimulus is a manufactured lobe that C8 removes at
    source - and every other test here runs at the shipped default, so this
    file still scores the shipped path."""
    E = _field(spec['n'], spec['dx'], spec['w'], spec['rc'], spec['alpha'],
               cx=cx, cy=cy)
    presc = _singlet(spec['r1'], spec['r2'], spec['th'], spec['z'], spec['ap'])
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.RAY_DENSITY_HALO_CHECK, LT._RD_HALO_RADIUS_FACTOR,
           LT._RD_HALO_AMAX_TOL, LT.REMAP_INVERSE_SUPPORT_BOUND)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    if bound is not None:
        LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    if policy is not None:
        LT.RAY_DENSITY_HALO_CHECK = policy
    if factor is not None:
        LT._RD_HALO_RADIUS_FACTOR = float(factor)
    if tol is not None:
        LT._RD_HALO_AMAX_TOL = float(tol)
    try:
        kw = dict(_BASE_KW)
        kw['dx'] = spec['dx']
        kw.update(over)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=spec['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.RAY_DENSITY_HALO_CHECK, LT._RD_HALO_RADIUS_FACTOR,
         LT._RD_HALO_AMAX_TOL, LT.REMAP_INVERSE_SUPPORT_BOUND) = old
    msgs = [str(w.message) for w in wl
            if 'HALO self-check FAILED' in str(w.message)]
    return F, msgs


@pytest.fixture(scope='module')
def _warm():
    """Both members of every byte-identity pair must sit on the same side of
    the traced pipeline's first-call ulp boundary (the W9 determinism
    calibration)."""
    for _ in range(2):
        _call(_CLEAN)
    return True


# ---------------------------------------------------------------------------
# 1. The shipped constants are where the calibration put them
# ---------------------------------------------------------------------------
def test_policy_defaults_to_warn():
    """Warning-only by design: a far lobe does not invalidate the returned
    field's core metrics, so refusing the call would be worse than saying so."""
    assert LT.RAY_DENSITY_HALO_CHECK == 'warn'


def test_constants_sit_inside_the_measured_gap():
    """Calibrated separation at the shipped factor (constant's own note): the
    worst CLEAN reading over 173 element calls is 4.622e-05 and the mildest
    CONFIRMED defect is 5.684e-03.  The tolerance must sit strictly between
    them with real margin on both sides, and the radius factor must stay in
    the window where the upsample spill has died but real defects have not."""
    assert 4.622e-05 * 4.0 < LT._RD_HALO_AMAX_TOL < 5.684e-03 / 4.0
    assert 1.0 < LT._RD_HALO_RADIUS_FACTOR <= 1.5
    assert LT._RD_HALO_AMP_CONTOUR >= 4.0


# ---------------------------------------------------------------------------
# 2. It is a DIAGNOSTIC: it may not move a single bit of the returned field
# ---------------------------------------------------------------------------
def test_check_is_field_neutral(_warm):
    """'warn' vs 'silent' must be byte-identical.  'silent' also skips the
    hull reduction entirely, so this pins that the skipped work really was
    diagnostic-only."""
    a, _ = _call(_CLEAN, policy='warn')
    b, _ = _call(_CLEAN, policy='silent')
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_check_is_field_neutral_when_it_fires(_warm):
    """The interesting half: a call that DOES trip the check must return the
    same array as one that does not.  Driven by the tolerance so the two runs
    differ in nothing but whether the warning is emitted."""
    a, ma = _call(_CLEAN, tol=-1.0)
    b, mb = _call(_CLEAN, tol=1.0)
    assert ma and not mb
    assert np.array_equal(a, b), float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# 3. No false positives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('launch', [False, True])
def test_silent_on_a_clean_call(_warm, launch):
    """Neither the shipped launch nor the C6 one may trip it on a well
    conditioned singlet."""
    _F, msgs = _call(_CLEAN, launch=launch)
    assert msgs == [], msgs


def test_silent_on_a_decentred_beam(_warm):
    """The support radius and the halo annulus are both referenced to the
    TRACED EXIT CENTROID, so a beam sitting off the grid centre must read the
    same as one sitting on it.  A grid-referenced radius would fire here."""
    _F, msgs = _call(_CLEAN, cx=1.5 * _CLEAN['w'], cy=-1.0 * _CLEAN['w'],
                     beam_centre=(1.5 * _CLEAN['w'], -1.0 * _CLEAN['w']))
    assert msgs == [], msgs


def test_silent_in_screen_mode(_warm):
    """The check belongs to the ray-density amplitude and must not appear in
    ``amplitude_model='screen'``, which has no ray-tube amplitude to inflate."""
    _F, msgs = _call(_CLEAN, amplitude_model='screen',
                     preserve_input_phase=True, tol=-1.0)
    assert msgs == [], msgs


# ---------------------------------------------------------------------------
# 4. FAIL-BEFORE: it fires on a real manufactured lobe
# ---------------------------------------------------------------------------
def test_fires_on_the_fit_guard_regression(_warm):
    """``REMAP_STATIONARY_PHASE_FIT_GUARD``'s own note records that the flag
    REVERSES on some fixtures -- the hard mask is clean and the weighted branch
    ghosts.  This is one of them, and it is exactly the class of defect the
    energy self-check cannot see.  Both directions are pinned: guard off is
    silent, guard on warns, and the energy band is quiet in BOTH.

    2026-08-01 (niche C8).  The stimulus this test needs -- a manufactured lobe
    -- is what ``REMAP_INVERSE_SUPPORT_BOUND`` removes AT SOURCE, so the
    detector correctly reports nothing when it is on.  The check itself is
    unchanged and no bar is moved: the fires-on-a-lobe arm is now measured with
    the bound OFF (the library state this check was calibrated in), and the
    bound ON is asserted as well -- the lobe is gone and the check is silent,
    which is the outcome that would otherwise have silently replaced a true
    positive with a green test."""
    Fa, ma = _call(_GHOST, launch=True, guard=False, bound=False)
    Fb, mb = _call(_GHOST, launch=True, guard=True, bound=False)
    assert ma == [], ma
    assert mb, 'the halo check missed a confirmed manufactured lobe'
    # niche C8 removes the lobe, so the SAME call is silent with it on -- and
    # the field really is quieter, not merely unreported.
    Fc, mc = _call(_GHOST, launch=True, guard=True, bound=True)
    assert mc == [], mc
    assert float(np.abs(Fc).max()) <= float(np.abs(Fb).max()) * (1 + 1e-12)
    assert float((np.abs(Fc) ** 2).sum()) < float((np.abs(Fb) ** 2).sum())
    # ... and the total-power band would NOT have caught it, which is the
    # whole argument for this check existing.
    p_in = float((np.abs(_field(_GHOST['n'], _GHOST['dx'], _GHOST['w'],
                                _GHOST['rc'], _GHOST['alpha'])) ** 2).sum())
    lo = 1.0 - (LT._RD_ENERGY_DEFICIT_BASE
                + LT._RD_ENERGY_DEFICIT_PER_SUB * (4 - 1))
    hi = 1.0 + LT._RD_ENERGY_GAIN_TOL
    for F in (Fa, Fb):
        assert lo <= float((np.abs(F) ** 2).sum()) / p_in <= hi


def test_declines_when_the_support_exceeds_the_grid(_warm):
    """SCOPE (d) of the constant's note, pinned.  When ``factor * r_hull``
    passes the grid's half-diagonal there is no annulus, so there is nothing
    to test -- and the check must DECLINE silently rather than error, divide
    by an empty max, or report a zero as a pass.  This is not hypothetical:
    it is what design 121's production readout leg does, where the last group
    is re-run on a fine grid narrower than its own exit fan."""
    _F, msgs = _call(_CLEAN, factor=50.0, tol=-1.0)
    assert msgs == [], msgs


def test_declines_when_only_the_corners_are_left(_warm):
    """The HARDER half of SCOPE (d), and the one that needed measuring.  At a
    factor where the bound circle is outside the grid's half-width but inside
    its half-diagonal, an annulus still EXISTS -- it is four corner slivers -
    and a naive implementation happily reports a number from it.  Measured
    twice, that number is unreliable in both directions: on design 121's
    (-2,0) readout leg it read 4.5e-04 with the fit guard off and 1.4e-03 with
    it on, on two fields that are both defective; on niche D6's tilted-leg
    retrace it read 0.841 of peak at the grid corner diagonally opposite the
    beam.  So the check must decline here too, not merely when the annulus is
    empty."""
    _F, msgs = _call(_CLEAN, factor=2.40, tol=-1.0)
    assert msgs == [], msgs
    # ... and the corners really are still there, i.e. this test is not
    # passing for the trivial reason the previous one does.
    n, dx = _CLEAN['n'], _CLEAN['dx']
    assert (n / 2) * dx * np.sqrt(2.0) > 2.40 * 2.6e-3 > (n / 2) * dx


def test_fires_when_the_radius_is_driven_inside_the_beam(_warm):
    """The radius is derived from the exact traced ray support, not chosen.
    Drive the factor well below 1 and REAL light falls in the annulus, so the
    check must fire on the clean fixture too -- which is what proves the
    radius is the ray support and not some fixed number."""
    _F, msgs = _call(_CLEAN, factor=0.30)
    assert msgs, 'the halo radius is not tracking the traced ray support'


# ---------------------------------------------------------------------------
# 5. The message is actionable, and the knob works
# ---------------------------------------------------------------------------
def test_message_reports_radius_amplitude_and_power(_warm):
    _F, msgs = _call(_CLEAN, tol=-1.0)
    assert len(msgs) == 1
    m = msgs[0]
    for token in ('amax_halo', 'exact-ray exit support radius',
                  'traced exit centroid', 'g_halo',
                  'RAY_DENSITY_HALO_CHECK'):
        assert token in m, token


def test_policy_silent_suppresses(_warm):
    _F, msgs = _call(_CLEAN, tol=-1.0, policy='silent')
    assert msgs == [], msgs


def test_is_a_runtimewarning_and_never_an_error(_warm):
    """Deliberately not an error under any policy -- the returned field's core
    metrics survive a far lobe, and erroring would break legitimate coarse-grid
    workflows the way the exit-NA guard is documented not to.  The category
    matters too: ``RuntimeWarning`` is what the energy self-check uses, so a
    caller filtering one filters both."""
    E = _field(_CLEAN['n'], _CLEAN['dx'], _CLEAN['w'], _CLEAN['rc'],
               _CLEAN['alpha'])
    presc = _singlet(_CLEAN['r1'], _CLEAN['r2'], _CLEAN['th'], _CLEAN['z'],
                     _CLEAN['ap'])
    kw = dict(_BASE_KW)
    kw['dx'] = _CLEAN['dx']
    old = LT._RD_HALO_AMAX_TOL
    LT._RD_HALO_AMAX_TOL = -1.0
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=_CLEAN['rc'], **kw))
    finally:
        LT._RD_HALO_AMAX_TOL = old
    hits = [w for w in wl if 'HALO self-check FAILED' in str(w.message)]
    assert len(hits) == 1
    assert hits[0].category is RuntimeWarning
    assert np.isfinite(F).all()
