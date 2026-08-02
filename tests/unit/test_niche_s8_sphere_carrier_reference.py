"""Regression pins for the chain-level EXACT-SPHERE carrier reference
(``propagate_traced_carrier_chain(carrier_reference='sphere')``, audit
AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8).

Why it exists: the carrier machinery references the paraxial PARABOLA
``r^2/(2R)`` while a traced element's ray launch / carrier eikonal references
the EXACT sphere ``S(R)``.  The two differ by ``+k r^4/(8R^3)``, which is
several radians at even a modest NA over a long carrier leg (design-121:
+3.36 rad at r=w at the first group vertex, emitter NA 0.104 over 45.9 mm).
With ``preserve_input_phase=False`` the element re-imposes its own spherical
reference, so the chain's exit carries only the LAST group's own contribution
and the design's distributed correction is discarded (design-121: +3.11 rad
r^4, 0.347 rad rms, best-focus EE6 79.7%).  With ``'remap'`` the residual IS
carried -- but then the parabola's spurious r^4 is carried too (+4.30 rad).
``carrier_reference='sphere'`` band-limits the parabola out of the hand-off so
the carried content is the physical residual; together with
``amplitude_model='ray_density'`` + ``preserve_input_phase='remap'`` the
design-121 exit lands on the full-train ray oracle's design floor (r^4 -0.13,
rms 0.015 vs the oracle's 0.018) and the focus on FWHM 3.55 um / EE3 88.4 /
EE6 99.3, dx-flat over N = 1024..4096.

These pins are CI-safe (synthetic singlet, no external assets):

* the conversion factor's analytic form, unit modulus, band-limit taper and
  collimated-carrier no-op;
* the mixed-convention-over-the-beam warning;
* the API guard;
* ``'parabola'`` (default) is byte-identical to omitting the kwarg;
* ``'sphere'`` is a NO-OP when the element discards the input phase
  (the measured design-121 property that made the earlier "conversion alone"
  experiments read as null results);
* ``'sphere'`` is EXACTLY equivalent (under ``'remap'``) to pre-referencing
  the input field to the exact sphere by hand -- i.e. it applies the
  conversion with the right sign at the right boundary.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as CM
from lumenairy.propagators.carrier import _exact_sphere_eikonal, _sphere_parab_conversion

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


def _singlet(R1, R2, d, glass, ap, name='s'):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


# --------------------------------------------------------------------------
# the conversion factor itself
# --------------------------------------------------------------------------

def test_conversion_factor_analytic_and_unit_modulus():
    """Inside the band limit the factor is exactly
    ``exp(sign*i*k*(S(R) - r^2/2R))``; it is unit-modulus everywhere."""
    N, dx, R = 128, 20e-6, -30e-3
    f = _sphere_parab_conversion((N, N), dx, _WL, R, +1)
    assert f.shape == (N, N)
    np.testing.assert_allclose(np.abs(f), 1.0, atol=1e-12)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    diff = _exact_sphere_eikonal((N, N), dx, dx, _WL, R) - r2 / (2.0 * R)
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    core = np.sqrt(r2) < 0.5 * r_safe          # taper is identically 1 here
    np.testing.assert_allclose(f[core], np.exp(1j * _K0 * diff[core]),
                               rtol=0, atol=1e-9)
    # the -1 sign is the exact inverse
    g = _sphere_parab_conversion((N, N), dx, _WL, R, -1)
    np.testing.assert_allclose(f * g, 1.0, atol=1e-9)


def test_conversion_factor_band_limited_taper():
    """Beyond ``r_safe`` the factor rolls off to identity (1+0j), so the
    aliasing guard band carries no phase.

    2026-08-02 (niche C9): the taper is no longer the shipped behaviour -- the
    conversion is applied EXACTLY on the whole grid.  The assertion below is
    unchanged word for word; it is now scored on the FAIL-BEFORE arm
    (``SPHERE_PARAB_CONVERSION_EXACT = False``), which is the library state it
    was calibrated in.  The shipped state is asserted alongside rather than
    instead, so this file still pins BOTH sides of the switch.  Full record:
    ``docs/audits/D121_FINAL_CLOSURE_2026_08_02.md`` and
    ``tests/unit/test_niche_c9_sphere_parab_exact_conversion.py``."""
    N, dx, R = 512, 40e-6, -8e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    x = (np.arange(N) - N / 2) * dx
    rr = np.sqrt(x[None, :] ** 2 + x[:, None] ** 2)
    assert rr.max() > 1.2 * r_safe, 'test grid must reach past r_safe'
    _old = CM.SPHERE_PARAB_CONVERSION_EXACT
    try:
        CM.SPHERE_PARAB_CONVERSION_EXACT = False
        f = _sphere_parab_conversion((N, N), dx, _WL, R, +1)
        np.testing.assert_allclose(f[rr > r_safe], 1.0 + 0.0j, atol=1e-12)
        # ... and the shipped arm carries the exact phase out there instead.
        CM.SPHERE_PARAB_CONVERSION_EXACT = True
        g = _sphere_parab_conversion((N, N), dx, _WL, R, +1)
        np.testing.assert_allclose(np.abs(g), 1.0, atol=1e-12)
        assert np.abs(g[rr > r_safe] - 1.0).max() > 0.5, (
            'the shipped conversion must NOT be identity beyond r_safe')
        # the two agree exactly inside the old onset, so the change is
        # confined to the annulus
        core = rr < 0.75 * r_safe
        assert core.any()
        CM.SPHERE_PARAB_CONVERSION_EXACT = False
        assert np.array_equal(
            g[core], _sphere_parab_conversion((N, N), dx, _WL, R, +1)[core])
    finally:
        CM.SPHERE_PARAB_CONVERSION_EXACT = _old


@pytest.mark.parametrize('R', [np.inf, -np.inf, 0.0])
def test_conversion_none_for_collimated_or_degenerate(R):
    assert _sphere_parab_conversion((16, 16), 1e-5, _WL, R, +1) is None


def test_conversion_warns_when_band_limit_reaches_into_beam():
    """The conversion tapered off where the beam still carries power is a MIXED
    carrier convention over the skirt; warn."""
    N, dx, R = 256, 40e-6, -4e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    with pytest.warns(RuntimeWarning, match='band-limit radius'):
        _sphere_parab_conversion((N, N), dx, _WL, R, +1, w_beam=r_safe)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # no warning when comfortable
        _sphere_parab_conversion((N, N), dx, _WL, R, +1, w_beam=0.1 * r_safe)


def test_conversion_guard_tests_the_taper_ONSET_not_r_safe():
    """2026-07-31: the guard tests ``0.75*r_safe`` (where the cos^2 roll-off
    STARTS), not ``r_safe``.

    The old ``r_safe < 2*w`` form could not detect the case the warning's own
    message describes.  Design 121's last two carrier planes sit at
    ``r_safe = 2.18 w`` -- clear of that threshold, so silent -- while the
    taper ONSET is at 1.63 w, inside the beam, worth a measured 1.41 EE3
    points on the (-4,-2) congruence (docs/audits/
    APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S2).

    The regime this pins is ``0.375 r_safe < w <= 0.5 r_safe``: warned about
    now, silent before.  Design 121 sits at ``w = 0.459 r_safe``, inside it.

    2026-08-02 (niche C9): the TRIGGER is unchanged and so is everything this
    test measures; only the message was reworded (there is no taper to have an
    onset any more), so the ``match`` phrase moved from ``taper ONSET`` to the
    stable ``band-limit radius`` the sibling test above already uses.  No
    threshold, radius or assertion changed."""
    N, dx, R = 256, 40e-6, -4e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    with pytest.warns(RuntimeWarning, match='band-limit radius'):
        _sphere_parab_conversion((N, N), dx, _WL, R, +1,
                                 w_beam=0.459 * r_safe)
    # ... and just outside it the guard is still silent, so the threshold
    # moved rather than being removed.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _sphere_parab_conversion((N, N), dx, _WL, R, +1,
                                 w_beam=0.36 * r_safe)


def test_conversion_guard_is_warning_only():
    """The threshold change must not touch the returned factor -- the taper
    itself is unchanged, only its detection."""
    N, dx, R = 256, 40e-6, -4e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    quiet = _sphere_parab_conversion((N, N), dx, _WL, R, +1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loud = _sphere_parab_conversion((N, N), dx, _WL, R, +1,
                                        w_beam=0.459 * r_safe)
    assert np.array_equal(quiet, loud)


# --------------------------------------------------------------------------
# chain-level semantics
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def _chain_setup():
    """A single-group chain with a NON-negligible parabola/sphere gap:
    R_in = 60 mm at w = 4.5 mm (NA 0.075) -> k w^4/(8 R^3) ~ 1.1 rad."""
    N, dx = 512, 30e-6
    w, R_in = 4.5e-3, 60e-3
    presc = _singlet(60.0e-3, -60.0e-3, 3.0e-3, 'N-BK7', 14.0e-3, 'sph')
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2).astype(np.complex128)
    gap = 0.0          # zero gap: the boundary conversion is then EXACTLY a
    #                    pre-reference of the input (no transport in between)
    return N, dx, R_in, w, presc, env, gap


def _run(setup, *, carrier_reference='parabola', env=None, **tkw):
    N, dx, R_in, w, presc, env0, gap = setup
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env0 if env is None else env,
            [{'prescription': presc, 'gap_before': gap}],
            _WL, dx, r_in=R_in, ray_subsample=8, n_workers=1,
            carrier_reference=carrier_reference,
            traced_kwargs=dict(on_undersample='silent',
                               on_noncollimated='silent', **tkw))
    return np.asarray(res.field)


def test_chain_carrier_reference_guard(_chain_setup):
    with pytest.raises(ValueError, match='carrier_reference'):
        _run(_chain_setup, carrier_reference='exact-sphere')


def test_chain_default_is_validated_config_and_legacy_is_reachable(
        _chain_setup):
    """v5.29 default flip (audit S8): omitting every knob must be bit-for-bit
    the validated configuration (carrier_reference='sphere' + ray_density +
    'remap'; since S12 the chain also defaults remap_sampling='full' and
    fit_radius_beam_factor=2.0 -- both arms of this pin inherit those
    non-overridden defaults, so the equality tests the merge, not the full
    triple), and the documented LEGACY escape hatch
    (carrier_reference='parabola' + screen amplitude + preserve_input_phase
    =True) must be bit-for-bit the pre-flip chain."""
    N, dx, R_in, w, presc, env0, gap = _chain_setup
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        default = np.asarray(la.propagate_traced_carrier_chain(
            env0, [{'prescription': presc, 'gap_before': gap}], _WL, dx,
            r_in=R_in, ray_subsample=8, n_workers=1,
            traced_kwargs=dict(on_undersample='silent',
                               on_noncollimated='silent')).field)
    explicit = _run(_chain_setup, carrier_reference='sphere',
                    amplitude_model='ray_density',
                    preserve_input_phase='remap')
    assert np.array_equal(default, explicit)
    # the legacy escape hatch routes through the pre-flip branches (screen
    # amplitude, wave-pair phase, parabola hand-offs) and must therefore
    # differ from the new default on this setup -- guards against the
    # default merge silently not reaching the per-group calls
    legacy = _run(_chain_setup, carrier_reference='parabola',
                  amplitude_model='screen', preserve_input_phase=True)
    assert not np.array_equal(default, legacy)


def test_sphere_reference_is_noop_when_input_phase_discarded(_chain_setup):
    """With ``preserve_input_phase=False`` the element re-imposes its own
    spherical reference and ignores the input wavefront, so the conversion
    cannot change the result (design-121: identical to 3 digits).  This is
    why 'conversion alone' experiments read as null results -- the fix needs
    the carry as well."""
    kw = dict(amplitude_model='ray_density', preserve_input_phase=False)
    a = _run(_chain_setup, carrier_reference='parabola', **kw)
    b = _run(_chain_setup, carrier_reference='sphere', **kw)
    num = np.linalg.norm(b - a)
    den = np.linalg.norm(a)
    assert num / den < 1e-9, num / den


def test_sphere_reference_exit_conversion_is_physically_invariant():
    """The EXIT-side conversion (sign -1) and the convert-back before the
    parabola-referenced final leg must cancel exactly, so with the element
    ignoring the input phase the returned PHYSICAL field is unchanged even
    with a gap and a nonzero final distance.  Measured margin: 6e-16 here vs
    5.9e-2 if the exit-side sign is flipped."""
    N, dx = 512, 30e-6
    w, R_in = 4.5e-3, 60e-3
    presc = _singlet(60.0e-3, -60.0e-3, 3.0e-3, 'N-BK7', 14.0e-3, 'sph2')
    x = (np.arange(N) - N // 2) * dx
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
                 ).astype(np.complex128)
    out = {}
    for cr in ('parabola', 'sphere'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out[cr] = np.asarray(la.propagate_traced_carrier_chain(
                env, [{'prescription': presc, 'gap_before': 20e-3}],
                _WL, dx, r_in=R_in, ray_subsample=8, n_workers=1,
                carrier_reference=cr, final_distance=15e-3,
                traced_kwargs=dict(on_undersample='silent',
                                   on_noncollimated='silent',
                                   amplitude_model='ray_density',
                                   preserve_input_phase=False)).field)
    rel = (np.linalg.norm(out['sphere'] - out['parabola'])
           / np.linalg.norm(out['parabola']))
    assert rel < 1e-10, rel


def test_sphere_reference_equals_prereferenced_input_under_remap(
        _chain_setup):
    """'sphere' must apply the (S - parabola) conversion with the RIGHT sign
    at the element boundary: with a zero gap it is exactly equivalent to
    handing the PARABOLA chain an input envelope pre-multiplied by the same
    band-limited factor.  A sign or reference error breaks this."""
    N, dx, R_in, w, presc, env0, gap = _chain_setup
    kw = dict(amplitude_model='ray_density', preserve_input_phase='remap')
    f = _sphere_parab_conversion((N, N), dx, _WL, R_in, +1)
    a = _run(_chain_setup, carrier_reference='sphere', **kw)
    b = _run(_chain_setup, carrier_reference='parabola', env=env0 * f, **kw)
    num = np.linalg.norm(b - a)
    den = np.linalg.norm(a)
    assert num / den < 1e-9, num / den
    # ... and it is NOT the same as leaving the parabola in place (the
    # conversion actually does something at this NA)
    c = _run(_chain_setup, carrier_reference='parabola', **kw)
    assert np.linalg.norm(c - a) / den > 1e-3, (
        np.linalg.norm(c - a) / den)
