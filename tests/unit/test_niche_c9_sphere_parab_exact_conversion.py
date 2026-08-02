"""Niche C9 (2026-08-02) -- the parabola <-> exact-sphere carrier-convention
conversion is applied EXACTLY, with no ``cos^2`` band-limit taper.

Record: ``docs/audits/D121_FINAL_CLOSURE_2026_08_02.md``.

WHAT THE CHANGE IS.  ``_sphere_parab_conversion`` used to multiply its exact
factor ``exp(sign*i*k*(S(R) - r^2/2R))`` by a ``cos^2`` roll-off from
``0.75*r_safe`` to ``r_safe = (|R|^3 lambda/dx)^(1/3)``.  The taper was a
Nyquist guard on the FACTOR, but the factor is applied pointwise and its
product is consumed pointwise -- ``apply_real_lens_traced`` de-chirps against
the same exact sphere, so the chain's entrance conversion and the element's own
reference are an identity PAIR.  The taper broke that pair, leaving an ALIASED
``exp(-i k (S - r^2/2R)(1 - T))`` in the annulus where the beam still carries
power, which the chain then transported across the next gap.

DISCIPLINE.  Every numeric assertion below is a COMPARATIVE ENVELOPE -- a ratio
between two arms measured in the same process on the same fixture, with the
measured headroom in the comment -- never an absolute bar on a BLAS-dependent
magnitude.  The two exceptions are exact-arithmetic identities
(``np.array_equal`` on a closed form, and unit modulus of ``exp(i*phi)``),
which are elementwise ``exp``/``sqrt`` and carry no BLAS dependence at all.
"""
import numpy as np
import pytest

from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.propagators import carrier as CM
from lumenairy.propagators.carrier import _exact_sphere_eikonal, _sphere_parab_conversion

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


@pytest.fixture
def exact_off():
    """Run the body with the fail-before switch thrown, then restore."""
    old = CM.SPHERE_PARAB_CONVERSION_EXACT
    CM.SPHERE_PARAB_CONVERSION_EXACT = False
    try:
        yield
    finally:
        CM.SPHERE_PARAB_CONVERSION_EXACT = old


def _grid(n, dx):
    x = (np.arange(n) - n / 2) * dx
    return x[None, :] ** 2 + x[:, None] ** 2


def _tapered_reference(n, dx, R, sign):
    """The historical factor, recomputed here term for term."""
    r2 = _grid(n, dx)
    diff = _exact_sphere_eikonal((n, n), dx, dx, _WL, R) - r2 / (2.0 * R)
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    t = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
    return np.exp(sign * 1j * _K0 * diff * np.cos(0.5 * np.pi * t) ** 2)


# --------------------------------------------------------------------------
# the flag, and that it really is a switch
# --------------------------------------------------------------------------

def test_the_default_is_exact():
    assert CM.SPHERE_PARAB_CONVERSION_EXACT is True


def test_shipped_factor_is_the_exact_closed_form_everywhere():
    """No taper: the factor equals ``exp(sign*i*k*(S - r^2/2R))`` on EVERY
    pixel, not only inside the band limit.  Exact arithmetic (elementwise
    ``exp`` of a closed form), so ``array_equal`` is the right assertion."""
    n, dx, R = 512, 40e-6, -8e-3
    r2 = _grid(n, dx)
    assert np.sqrt(r2).max() > 1.2 * (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0), \
        'the fixture grid must reach past r_safe or this proves nothing'
    diff = _exact_sphere_eikonal((n, n), dx, dx, _WL, R) - r2 / (2.0 * R)
    for sign in (+1, -1):
        f = _sphere_parab_conversion((n, n), dx, _WL, R, sign)
        assert np.array_equal(f, np.exp(sign * 1j * _K0 * diff))


def test_the_fail_before_restores_the_taper_bit_for_bit(exact_off):
    """``SPHERE_PARAB_CONVERSION_EXACT = False`` reproduces the historical
    tapered factor exactly -- the contract every pre-C9 result rests on."""
    n, dx, R = 512, 40e-6, -8e-3
    for sign in (+1, -1):
        f = _sphere_parab_conversion((n, n), dx, _WL, R, sign)
        assert np.array_equal(f, _tapered_reference(n, dx, R, sign))
    # and the historical taper's own defining property, kept word for word
    # from ``test_niche_s8::test_conversion_factor_band_limited_taper``
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    rr = np.sqrt(_grid(n, dx))
    f = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
    np.testing.assert_allclose(f[rr > r_safe], 1.0 + 0.0j, atol=1e-12)


def test_the_two_arms_are_byte_identical_inside_the_onset(exact_off):
    """The taper is identically 1 inside ``0.75*r_safe``, so the change cannot
    touch the beam core -- it acts only in the annulus.  Measured on design
    121's worst plane the onset sits at 1.42-1.64 beam radii."""
    n, dx, R = 512, 40e-6, -8e-3
    off = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
    CM.SPHERE_PARAB_CONVERSION_EXACT = True
    on = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
    core = np.sqrt(_grid(n, dx)) < 0.75 * (abs(R) ** 3 * _WL / dx) ** (1 / 3)
    assert core.any()
    assert np.array_equal(on[core], off[core])
    assert not np.array_equal(on, off), 'the fixture must exercise the annulus'


@pytest.mark.parametrize('R', [np.inf, -np.inf, 0.0])
def test_collimated_or_degenerate_still_returns_none(R):
    assert _sphere_parab_conversion((16, 16), 1e-5, _WL, R, +1) is None


# --------------------------------------------------------------------------
# the property the change is FOR: the +1/-1 pair is an identity everywhere
# --------------------------------------------------------------------------

def test_the_pair_is_a_pointwise_identity_over_the_whole_grid():
    """``+1`` then ``-1`` at the same ``(R, dx, centre)`` must be exactly 1.
    Unit-modulus arithmetic; the bar is double-precision rounding, not a
    physics tolerance."""
    n, dx, R = 512, 40e-6, -8e-3
    for cen in ((0.0, 0.0), (1.9e-3, -0.6e-3)):
        f = _sphere_parab_conversion((n, n), dx, _WL, R, +1, centre=cen)
        g = _sphere_parab_conversion((n, n), dx, _WL, R, -1, centre=cen)
        np.testing.assert_allclose(f * g, 1.0, atol=1e-12)


def test_the_conversion_recovers_the_physical_field_far_better_than_the_taper(
        exact_off):
    """THE POINT OF THE CHANGE, as a measurement.

    Build a field that IS an exact sphere with a smooth residual and store the
    SPHERE-referenced envelope of it -- which is what the chain carries under
    the shipping ``carrier_reference='sphere'``.  The chain then reconstructs
    it exactly as ``propagate_traced_carrier_chain`` does before an element
    call: ``carrier_referenced_reconstruct`` (times the PARABOLA) followed by
    the ``+1`` conversion.  That composite must return the field it started
    from.  Comparative envelope -- the exact arm's error over the tapered
    arm's, on one fixture, in one process."""
    n, dx, R, w = 512, 40e-6, -8e-3, 4.0e-3
    r2 = _grid(n, dx)
    S = _exact_sphere_eikonal((n, n), dx, dx, _WL, R)
    resid = 0.3 * (r2 / w ** 2) ** 2                    # smooth, sub-wave
    E = np.exp(-r2 / w ** 2) * np.exp(1j * _K0 * (S + resid * _WL))
    env_s = E * np.exp(-1j * _K0 * S)                   # sphere-referenced

    def err():
        cf = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
        back = env_s * np.exp(1j * _K0 * r2 / (2.0 * R)) * cf
        return float(np.abs(back - E).max() / np.abs(E).max())

    e_taper = err()
    CM.SPHERE_PARAB_CONVERSION_EXACT = True
    e_exact = err()
    # measured on this fixture: exact 0 (it is an algebraic identity), tapered
    # ~2 (the annulus phase is wrong by up to pi).  A 1e-6 ratio bar leaves
    # every decade of that separation in hand and is scale-free.
    assert e_exact < 1e-6 * max(e_taper, 1e-300), (
        f'exact {e_exact:.3e} vs tapered {e_taper:.3e}')
    assert e_taper > 1e-2, (
        f'the fixture must exercise the annulus (tapered err {e_taper:.3e})')


def test_the_element_hand_off_pair_is_consistent(exact_off):
    """The chain hands ``apply_real_lens_traced`` a field converted to the
    exact-sphere convention, and the element immediately de-chirps against the
    SAME exact sphere.  That round trip must return the stored envelope.

    Scored as the amplitude-weighted p99.9 wrapped nearest-neighbour phase step
    of the recovered envelope -- the sampling statistic this campaign quotes --
    because a residual that saturates at ``pi`` is exactly what makes the
    element's launch directions meaningless."""
    n, dx, R, w = 512, 40e-6, -8e-3, 4.0e-3
    r2 = _grid(n, dx)
    env = np.exp(-r2 / w ** 2).astype(np.complex128)     # smooth, real
    S = _exact_sphere_eikonal((n, n), dx, dx, _WL, R)

    def step_p999():
        cf = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
        E_full = env * np.exp(1j * _K0 * r2 / (2.0 * R)) * cf
        rec = E_full * np.exp(-1j * _K0 * S)             # what the element does
        ph, a = np.angle(rec), np.abs(rec)
        mx = max(float(a.max()), 1e-300)

        def _w(x):
            return (x + np.pi) % (2 * np.pi) - np.pi

        st = np.concatenate([
            (np.abs(_w(ph[:, 1:] - ph[:, :-1]))
             * np.minimum(a[:, 1:], a[:, :-1]) / mx).ravel(),
            (np.abs(_w(ph[1:] - ph[:-1]))
             * np.minimum(a[1:], a[:-1]) / mx).ravel()])
        return float(np.percentile(st, 99.9))

    s_taper = step_p999()
    CM.SPHERE_PARAB_CONVERSION_EXACT = True
    s_exact = step_p999()
    # the exact arm recovers ``env`` identically, so its residual step is 0;
    # the tapered arm leaves the aliased quartic.  Ratio bar with the whole
    # measured separation in hand.
    assert s_exact < 1e-6 * max(s_taper, 1e-300), (
        f'exact {s_exact:.3e} rad vs tapered {s_taper:.3e} rad')
    assert s_taper > 0.1, (
        f'the fixture must put the band limit inside the beam '
        f'(tapered step {s_taper:.3e} rad)')


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------

def test_the_guard_still_fires_and_is_warning_only():
    """Trigger and phrase are unchanged (``band-limit radius``); the returned
    array does not depend on whether it fired."""
    n, dx, R = 256, 40e-6, -8e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    quiet = _sphere_parab_conversion((n, n), dx, _WL, R, +1)
    with pytest.warns(RuntimeWarning, match='band-limit radius'):
        loud = _sphere_parab_conversion((n, n), dx, _WL, R, +1,
                                        w_beam=0.459 * r_safe)
    assert np.array_equal(quiet, loud)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _sphere_parab_conversion((n, n), dx, _WL, R, +1,
                                 w_beam=0.36 * r_safe)


def test_the_guard_message_no_longer_claims_a_taper():
    n, dx, R = 256, 40e-6, -8e-3
    r_safe = (abs(R) ** 3 * _WL / dx) ** (1.0 / 3.0)
    with pytest.warns(RuntimeWarning) as rec:
        _sphere_parab_conversion((n, n), dx, _WL, R, +1,
                                 w_beam=0.459 * r_safe)
    msg = str(rec[0].message)
    assert 'band-limit radius' in msg
    assert 'taper' not in msg.lower(), msg
    assert 'stays' in msg and 'exact' in msg.lower(), msg


# --------------------------------------------------------------------------
# it must not disturb anything that does not use the conversion
# --------------------------------------------------------------------------

def test_the_element_alone_is_byte_identical_across_the_flag(exact_off):
    """``apply_real_lens_traced`` never calls this conversion -- the chain
    applies it on the way in and out.  So an element call must be
    byte-identical across the switch, which also pins that the flag has no
    accidental second reader."""
    n, dx, w, rc = 256, 4.0e-6, 200e-6, -0.02
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(rc) * (np.sqrt(r2 + rc ** 2) - abs(rc))
    E = (np.exp(-r2 / w ** 2) * np.exp(1j * _K0 * S)).astype(np.complex128)
    _gb, _ga = ['air', 'N-BK7'], ['N-BK7', 'air']
    presc = {'name': 'c9-singlet', 'aperture_diameter': 0.5e-3,
             'thicknesses': [1.0e-3], 'surfaces': [
                 {'radius': 3.1e-3, 'glass_before': _gb[0],
                  'glass_after': _ga[0], 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None},
                 {'radius': -3.1e-3, 'glass_before': _gb[1],
                  'glass_after': _ga[1], 'conic': 0.0, 'radius_y': None,
                  'conic_y': None, 'aspheric_coeffs': None,
                  'aspheric_coeffs_y': None}]}
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, carrier=rc,
              ray_subsample=4, n_workers=1, parallel_amp=False,
              on_undersample='silent', on_noncollimated='silent')
    a = np.asarray(apply_real_lens_traced(E, **kw))
    CM.SPHERE_PARAB_CONVERSION_EXACT = True
    b = np.asarray(apply_real_lens_traced(E, **kw))
    assert np.array_equal(a, b) and a.dtype == b.dtype
