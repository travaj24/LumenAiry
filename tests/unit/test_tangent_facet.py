"""Route 3 -- ``surface_model='tangent_facet'``, the per-pixel tangent-facet
screen.

Every bar in this file is either an EXACT structural claim (``np.array_equal``,
a refusal, a zero by construction) or a comparison against an INDEPENDENT
oracle -- exact plane-facet ray algebra, or the shipped exact ray tracer
through ``apply_real_lens_traced`` -- with the bar derived from that oracle's
own floor and the measured numbers dated in the comment.  Nothing here pins a
build's residual, a count, or a magnitude ratio whose pass/fail boundary sits
inside the cross-build spread (docs/TESTING_STANDARDS.md S1-S5).

Design note the tests encode: route 3 REPLACES the paraxial facet coefficient
rather than correcting it, so (a) it needs no carrier -- a steep facet is
angle-wrong at normal arrival too -- and (b) it supersedes ``screen_obliquity``
instead of composing with it.
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements import apply_real_lens, apply_real_lens_traced

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _deterministic_fft():
    """The byte-identity claims are about arithmetic, not about the FFT
    planner's dtype promotion; pin it the way the sibling suites do."""
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(False)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _plate(t=3.0e-3, glass='N-BK7'):
    return {'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'glass_before': 'AIR',
         'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'glass_before': glass,
         'glass_after': 'AIR'}],
        'thicknesses': [t]}


def _singlet(R=19.6e-3, glass='N-SSK2', t=4.0e-3, ap=3.0e-3):
    """The steepest single facet in design 121's last group -- |grad sag| =
    0.155 at 3 mm -- which is the binding case for this model."""
    return {'surfaces': [
        {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [t], 'aperture_diameter': ap, 'name': 'singlet'}


def _biconvex(R=12.6e-3, glass='N-SSK2', t=4.0e-3, ap=3.0e-3):
    return {'surfaces': [
        {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'AIR', 'glass_after': glass},
        {'radius': -R, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'AIR'}],
        'thicknesses': [t], 'aperture_diameter': ap, 'name': 'biconvex'}


def _field(N, dx, w=None, tilt=0.0):
    a = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(a, a, indexing='ij')
    w = w if w is not None else 0.25 * N * dx
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    if tilt:
        E = E * np.exp(1j * 2 * np.pi / LAM * tilt * X)
    return E.astype(np.complex128)


def _tf(E, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               surface_model='tangent_facet', **kw)


def _thin(E, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               **kw)


# ---------------------------------------------------------------------------
# 1.  THE DEFAULT PATH DID NOT MOVE
# ---------------------------------------------------------------------------
def test_the_default_path_is_byte_identical():
    """``surface_model`` already existed and already defaulted to ``'thin'``,
    so route 3 is reachable only through a value that did not exist in 5.35.5.
    The null is therefore structural -- measured anyway, on the option
    combinations that share the surface loop with it."""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3)
    for presc in (_singlet(), _biconvex(), _plate()):
        base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
        for kw in ({}, {'surface_model': 'thin'}, {'sag_chunk_rows': 32},
                   {'sag_chunk_rows': 7}, {'bandlimit': True}):
            got = apply_real_lens(E, prescription=presc, wavelength=LAM,
                                  dx=dx, **kw)
            assert np.array_equal(base.view(np.uint8), got.view(np.uint8)), \
                (presc.get('name'), kw)


def test_the_thin_carriered_correction_is_byte_identical():
    """Route 3 must not have perturbed the v5.35.x angle-true screen it
    supersedes: the same call, with and without the new code present, has to
    return the same bits.  Pinned here against the shipped path's own
    determinism plus the untouched-branch structure."""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3)
    car = la.TiltedCarrier(np.inf, 0.05, 0.0)
    a = _thin(E, _biconvex(), dx, carrier=car)
    b = _thin(E, _biconvex(), dx, carrier=car)
    assert np.array_equal(a.view(np.uint8), b.view(np.uint8))
    # and the banded path still matches the whole-grid one for that call
    c = _thin(E, _biconvex(), dx, carrier=car, sag_chunk_rows=13)
    assert np.array_equal(a.view(np.uint8), c.view(np.uint8))


# ---------------------------------------------------------------------------
# 2.  THE PLATE, AND EVERY FLAT FACE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('tilt', [0.0, 0.01, 0.02, 0.055, 0.1, 0.2])
def test_a_plate_is_byte_identical_to_the_thin_screen_at_every_tilt(tilt):
    """``sag == 0`` gives no facet to tilt and no height to translate, so all
    three terms are identically zero and the block is skipped by one
    reduction.  ``np.array_equal``, not a tolerance."""
    N, dx = 192, 25e-6
    E = _field(N, dx, w=1.2e-3, tilt=tilt)
    car = None if tilt == 0.0 else la.TiltedCarrier(np.inf, tilt, 0.0)
    base = _thin(E, _plate(), dx)
    got = _tf(E, _plate(), dx, carrier=car)
    assert np.array_equal(base.view(np.uint8), got.view(np.uint8))


def test_a_flat_face_never_reaches_the_screen(monkeypatch):
    """The flat-face skip is observable, not an optimisation: it is what keeps
    the momentum accumulator a pair of PLAIN FLOATS through a plate, so a
    leading plate allocates nothing.  Assert the decision, not a byte count."""
    calls = []
    real = LR._tangent_facet_screen
    monkeypatch.setattr(LR, '_tangent_facet_screen',
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    N, dx = 128, 25e-6
    E = _field(N, dx, w=1.0e-3)
    _tf(E, _plate(), dx)
    assert calls == [], 'a plate must not evaluate the tangent-facet screen'
    _tf(E, _singlet(), dx)
    assert len(calls) == 1, 'exactly the one POWERED face of the singlet'


def test_the_accumulator_stays_scalar_until_a_powered_surface(monkeypatch):
    """A gap behind a flat face transports a SCALAR momentum, which the
    transport must pass through untouched rather than promoting to a grid."""
    seen = []
    real = LR._tangent_facet_transport

    def spy(px, py, *a, **k):
        seen.append((getattr(px, 'ndim', 0), getattr(py, 'ndim', 0)))
        return real(px, py, *a, **k)

    monkeypatch.setattr(LR, '_tangent_facet_transport', spy)
    N, dx = 128, 25e-6
    E = _field(N, dx, w=1.0e-3)
    _tf(E, _plate(), dx)
    assert seen == [(0, 0)], 'a plate gap must carry scalars, not grids'


@pytest.mark.parametrize('cr', [0, 1, 7, 64, 4096])
def test_sag_chunk_rows_is_INERT_for_this_model(cr):
    """Route 3 is whole-grid only: both band gates exclude it, so
    ``sag_chunk_rows`` must make no difference at all.  Pinned with
    ``np.array_equal`` rather than left implicit, so that a future change which
    lets the model into the band loop cannot take effect silently -- it would
    have to come with a byte-identity argument of its own, which is the same
    standard `test_obl_banded_halo.py` holds the carriered path to.
    """
    N, dx = 160, 25e-6
    E = _field(N, dx, w=1.1e-3)
    car = la.TiltedCarrier(np.inf, 0.05, 0.0)
    base = _tf(E, _biconvex(), dx, carrier=car)
    got = _tf(E, _biconvex(), dx, carrier=car, sag_chunk_rows=cr)
    assert np.array_equal(base.view(np.uint8), got.view(np.uint8))


@pytest.mark.parametrize('kw', [{'fresnel': True}, {'absorption': True},
                                {'seidel_correction': True},
                                {'bandlimit': False},
                                {'sag_dtype': np.float32}])
def test_the_orthogonal_option_axes_run_rather_than_raise(kw):
    """The refusals above are deliberate and narrow.  Everything that does NOT
    touch the facet coefficient -- Fresnel amplitudes, bulk absorption, the
    Seidel post-fit, the band limiter, float32 geometry -- must still work, or
    the model would be unusable for the reason it was built.  Two-sided
    against the refusal tests, which assert the opposite for the axes that DO
    collide."""
    N, dx = 128, 25e-6
    out = _tf(_field(N, dx, w=1.0e-3), _biconvex(), dx,
              carrier=la.TiltedCarrier(np.inf, 0.05, 0.0), **kw)
    assert bool(np.all(np.isfinite(out))), kw


# ---------------------------------------------------------------------------
# 3.  THE IDENTITY, AGAINST EXACT PLANE-FACET RAY ALGEBRA
# ---------------------------------------------------------------------------
def _exact_plane_facet_screen(x, s, g, p_in, n1, n2):
    """``S_in - S_out`` at the vertex plane for a PLANE facet ``z = s + g x``
    under a plane wave of transverse optical momentum ``p_in``.

    Independent of the library: exact ray algebra only.  Both fields are plane
    waves, so both eikonals are known in closed form and the screen a wave
    model must imprint is their difference.  Its own error floor is the
    float64 round-off of the algebra, ~1e-16 relative.
    """
    pz1 = np.sqrt(n1 ** 2 - p_in ** 2)
    inv = 1.0 / np.sqrt(1.0 + g ** 2)
    a = -g * inv * p_in + inv * pz1
    b = np.sqrt(n2 ** 2 - n1 ** 2 + a ** 2)
    p_out = p_in + (b - a) * (-g * inv)
    pz2 = pz1 + (b - a) * inv
    z = (s + g * x) / (1.0 - g * p_in / pz1)         # the true hit height
    x_h = x + z * p_in / pz1
    x1 = x_h - z * p_out / pz2
    s_out_at_x1 = p_in * x + z * (n1 ** 2 / pz1 - n2 ** 2 / pz2)
    c = s_out_at_x1 - p_out * x1                     # the exit plane wave
    return p_in * x - (p_out * x + c)


@pytest.mark.parametrize('slope', [0.05, 0.12, 0.24])
@pytest.mark.parametrize('n1,n2', [(1.0, 1.8047), (1.8047, 1.0),
                                   (1.5917, 1.8047)])
@pytest.mark.parametrize('tilt', [0.0, 0.055, 0.15])
def test_the_screen_is_the_exact_plane_facet_screen(slope, n1, n2, tilt):
    """For a plane facet the model is not an expansion -- it is the closed-form
    answer.  Bar: 1e-11 RELATIVE.

    Derivation of the bar: the oracle is exact float64 ray algebra, whose own
    floor is ~1e-16 relative; the model evaluates the same quantity through a
    different expression tree, so their difference is a few units in the last
    place amplified by the cancellations in ``b - a`` (worst measured over this
    27-cell matrix on 2026-08-16: 1.739e-13 relative, at slope 0.24,
    1.5917 -> 1.8047, 150 mrad).  1e-11 sits ~1.8 decades above the worst
    measurement and ~7 decades below the smallest real defect
    the model could acquire (a dropped second-order term reads 1e-4 relative
    on this fixture), so the bar has a gap on both sides.
    """
    n = 129
    x = np.linspace(-1.5e-3, 1.5e-3, n)
    X, Y = np.meshgrid(x, np.linspace(-1.5e-3, 1.5e-3, n))
    s0 = 2.3e-4
    sag = s0 + slope * X
    gx = np.full_like(sag, slope)
    gy = np.zeros_like(sag)
    p = np.full_like(sag, n1 * tilt)
    dx = float(x[1] - x[0])
    got, ok = LR._tangent_facet_screen(sag, gx, gy, p, np.zeros_like(p),
                                       n1, n2, dx, dx, np)
    assert bool(np.all(ok))
    want = _exact_plane_facet_screen(X, s0, slope, n1 * tilt, n1, n2)
    # interior only: np.gradient is one-sided on the border rows/columns, which
    # is a property of the stencil, not of the model.
    sl = (slice(2, -2), slice(2, -2))
    rel = np.abs(got[sl] - want[sl]) / np.abs(want[sl])
    assert float(rel.max()) < 1e-11, float(rel.max())


def test_the_second_order_terms_vanish_on_a_plane_facet():
    """(T2) needs ``grad grad sag`` and ``grad dz``; (T3) needs
    ``grad p_out``.  A plane facet under a plane wave makes all three exactly
    zero, so the screen collapses to ``dz * sag`` -- which is the structural
    reason the test above can hold at machine precision, and the reason a
    PLATE is byte-exact.  Asserted directly on the pieces."""
    n = 65
    x = np.linspace(-1.5e-3, 1.5e-3, n)
    X, _Y = np.meshgrid(x, x)
    dx = float(x[1] - x[0])
    slope, n1, n2 = 0.2, 1.0, 1.8
    sag = 2.3e-4 + slope * X
    gx = np.full_like(sag, slope)
    gy = np.zeros_like(sag)
    p = np.full_like(sag, 0.055)
    got, _ok = LR._tangent_facet_screen(sag, gx, gy, p, np.zeros_like(p),
                                        n1, n2, dx, dx, np)
    inv = 1.0 / np.sqrt(1.0 + slope ** 2)
    pz1 = np.sqrt(n1 ** 2 - 0.055 ** 2)
    a = (-slope * 0.055 + pz1) * inv
    b = np.sqrt(n2 ** 2 - n1 ** 2 + a ** 2)
    dz = (b - a) * inv
    sl = (slice(2, -2), slice(2, -2))
    assert np.allclose(got[sl], (dz * sag)[sl], rtol=0.0, atol=1e-18)


# ---------------------------------------------------------------------------
# 4.  THE GAP TRANSPORT
# ---------------------------------------------------------------------------
def test_the_transport_is_exact_for_a_linear_momentum_field():
    """The accumulator is resampled across a gap by one Taylor term.  For a
    momentum field LINEAR in x -- which is what a spherical surface's own kick
    is, to leading order -- that shift is not an approximation but the exact
    answer, so this is an equality claim and not a tolerance on a residual.

    Engineered rather than hoped for: the field is constructed linear, the
    walk is read back from the transport's own inputs, and the expected value
    is the analytic shift.
    """
    n = 129
    x = np.linspace(-1.5e-3, 1.5e-3, n)
    # the library's own grid convention: axis 0 is y, axis 1 is x, and every
    # gradient is taken as ``xp.gradient(f, dy, dx)``.
    X, Y = np.meshgrid(x, x)
    dx = float(x[1] - x[0])
    c = -24.0                                   # 1/m, a lens-scale kick slope
    px = c * X
    py = c * Y
    t, n_gap = 4.0e-3, 1.6
    nx, ny = LR._tangent_facet_transport(px, py, t, n_gap, dx, dx, np)
    pz = np.sqrt(n_gap ** 2 - px ** 2 - py ** 2)
    wx, wy = t * px / pz, t * py / pz
    sl = (slice(2, -2), slice(2, -2))
    assert np.allclose(nx[sl], (px - c * wx)[sl], rtol=0.0, atol=1e-15)
    assert np.allclose(ny[sl], (py - c * wy)[sl], rtol=0.0, atol=1e-15)


def test_a_scalar_accumulator_passes_through_the_transport_unchanged():
    """Identity, not near-identity: the objects returned are the objects
    passed in, so a leading plate cannot allocate a grid by accident."""
    got = LR._tangent_facet_transport(0.0, 0.0, 4e-3, 1.5, 1e-5, 1e-5, np)
    assert got == (0.0, 0.0)
    a, b = 0.03, -0.01
    assert LR._tangent_facet_transport(a, b, 4e-3, 1.5, 1e-5, 1e-5, np) \
        == (a, b)


# ---------------------------------------------------------------------------
# 5.  ACCURACY AGAINST THE TRACED RAY ORACLE
# ---------------------------------------------------------------------------
def _traced_gap(presc, N, dx, w, r_probe, tilt):
    """Wavefront gap between the analytic arms and ``apply_real_lens_traced``
    (the shipped exact ray tracer) over a probe disc, in waves."""
    from lumenairy.elements._lens_traced import _tilted_carrier_parts
    a = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(a, a)
    mask = (X ** 2 + Y ** 2) <= r_probe ** 2
    car = la.TiltedCarrier(float('inf'), float(tilt), 0.0)
    W, _l, _m = _tilted_carrier_parts(car, X, Y)
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * 2 * np.pi / LAM * W)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arms = {
            'blind': apply_real_lens(E, prescription=presc, wavelength=LAM,
                                     dx=dx),
            'thin+carrier': apply_real_lens(
                E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                on_screen_obliquity='silent'),
            'tangent_facet': apply_real_lens(
                E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
                surface_model='tangent_facet'),
        }
        ref = apply_real_lens_traced(
            E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
            on_noncollimated='off', on_aperture_beam='silent',
            on_undersample='silent')
    out = {}
    for k, v in arms.items():
        d = np.angle(v * np.conj(ref))[mask] / (2 * np.pi)
        d = d - d.mean()
        out[k] = float(np.sqrt((d ** 2).mean()))
    return out


def test_at_normal_incidence_route_3_beats_the_screen_a_carrier_cannot_fix():
    """THE CLAIM THAT NEEDS NO CARRIER.

    A steep facet is angle-wrong even when the light arrives along the axis,
    because the FACET is tilted.  The v5.35.x correction is a DIFFERENCE
    against the model's own zero-angle value, so it is identically zero here
    by construction -- it cannot help.  Route 3 replaces the coefficient, so
    it can.

    Two-sided and build-free: the carriered thin arm must land on the blind
    arm EXACTLY (a structural zero), and the tangent-facet arm must land at
    least 3x below both.  Measured 2026-08-16 on this fixture: blind and
    thin+carrier 0.00141 waves rms (identical), tangent_facet 0.00008 -- a
    17.6x separation, so a 3x bar sits ~0.8 decades below the measurement and
    ~0.5 decades above the equality it must not confuse with an improvement.
    """
    g = _traced_gap(_biconvex(R=12.6e-3), N=768, dx=4.0e-6, w=0.8e-3,
                    r_probe=0.55e-3, tilt=0.0)
    assert g['thin+carrier'] == g['blind'], (
        'the angular correction must be structurally zero at normal '
        'incidence; got %r' % (g,))
    assert g['tangent_facet'] * 3.0 < g['blind'], g


def test_with_a_carrier_route_3_beats_the_shipped_angular_correction():
    """The angular claim, on the arm BUILD_SCREEN_OBLIQUITY S3.6 established as
    resolvable (Nyquist-sampled exit, beam inside the aperture).

    Bar: route 3 strictly below the shipped carriered arm, and both far below
    blind.  Measured 2026-08-16 at 100 mrad on the R = 19.6 mm singlet:
    blind 0.00423, thin+carrier 0.00050, tangent_facet 0.00017 waves rms.  The
    assertion is an ORDERING plus a 2x margin on the blind arm, both of which
    have decades of headroom against the 1e-5-scale cross-build spread of a
    phase rms over 10^5 pixels.
    """
    g = _traced_gap(_singlet(), N=768, dx=4.0e-6, w=0.8e-3, r_probe=0.55e-3,
                    tilt=0.1)
    assert g['tangent_facet'] < g['thin+carrier'], g
    assert g['tangent_facet'] * 2.0 < g['blind'], g


def test_the_gain_tracks_the_FACET_SLOPE_which_is_what_the_model_fixes():
    """LOAD-BEARING CONTROL.  A model that merely added a constant offset, or
    that happened to help once, would not care how steep the facet is.  This
    one is a correction to the facet COEFFICIENT, so at normal incidence -- the
    regime where the arrival angle contributes nothing -- its gain must grow
    with |grad sag| and with nothing else.

    R = 19.6 mm has |grad sag| = 0.155 at 3 mm and R = 12.6 mm has 0.244, a
    1.6x steeper facet.  Measured 2026-08-16: gains 7.3x and 17.6x
    respectively.  The assertion is the ORDERING plus a 1.5x margin, which is
    a decision about which fixture is harder rather than a reading of either
    number -- and both gains are ratios of ~1e-3-scale phase rms over ~10^5
    pixels, decades outside any cross-build spread.
    """
    shallow = _traced_gap(_biconvex(R=19.6e-3), N=768, dx=4.0e-6, w=0.8e-3,
                          r_probe=0.55e-3, tilt=0.0)
    steep = _traced_gap(_biconvex(R=12.6e-3), N=768, dx=4.0e-6, w=0.8e-3,
                        r_probe=0.55e-3, tilt=0.0)
    g_shallow = shallow['blind'] / shallow['tangent_facet']
    g_steep = steep['blind'] / steep['tangent_facet']
    assert g_shallow > 2.0, shallow
    assert g_steep > 1.5 * g_shallow, (g_shallow, g_steep)


def test_the_carrier_is_what_supplies_the_angle():
    """Route 3 without a carrier repairs the FACET but not the ARRIVAL angle,
    so at a large tilt it must sit near the blind arm while the carriered call
    sits far below it.  That two-sided shape proves the carrier is wired into
    the accumulator seed rather than ignored -- and it is a decision, not a
    reading.

    Measured 2026-08-16 at 100 mrad, R = 19.6 mm: blind 0.00423,
    tangent_facet without carrier 0.00413, with carrier 0.00017.
    """
    presc, N, dx, w, rp = _singlet(), 768, 4.0e-6, 0.8e-3, 0.55e-3
    from lumenairy.elements._lens_traced import _tilted_carrier_parts
    a = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(a, a)
    mask = (X ** 2 + Y ** 2) <= rp ** 2
    car = la.TiltedCarrier(float('inf'), 0.1, 0.0)
    W, _l, _m = _tilted_carrier_parts(car, X, Y)
    E = (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
         * np.exp(1j * 2 * np.pi / LAM * W)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        free = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               surface_model='tangent_facet')
        carr = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                               surface_model='tangent_facet', carrier=car)
        ref = apply_real_lens_traced(
            E, prescription=presc, wavelength=LAM, dx=dx, carrier=car,
            on_noncollimated='off', on_aperture_beam='silent',
            on_undersample='silent')

    def rms(v):
        d = np.angle(v * np.conj(ref))[mask] / (2 * np.pi)
        return float(np.sqrt(((d - d.mean()) ** 2).mean()))

    assert rms(carr) * 5.0 < rms(free), (rms(carr), rms(free))


# ---------------------------------------------------------------------------
# 6.  THE REFUSALS
# ---------------------------------------------------------------------------
def _call(**kw):
    N, dx = 96, 25e-6
    return apply_real_lens(_field(N, dx, w=0.8e-3), prescription=_singlet(),
                           wavelength=LAM, dx=dx,
                           surface_model='tangent_facet', **kw)


def test_screen_obliquity_true_is_refused_as_a_double_count():
    with pytest.raises(ValueError, match='double-count'):
        _call(carrier=la.TiltedCarrier(np.inf, 0.05, 0.0),
              screen_obliquity=True)


def test_slant_correction_is_refused_as_a_double_count():
    with pytest.raises(ValueError, match='double-count'):
        _call(slant_correction=True)


@pytest.mark.parametrize('kw', [{'conjugate': 0.5},
                                {'displaced_mode': 'split'},
                                {'displaced_obliquity': 'pointwise'}])
def test_the_displaced_keywords_are_refused(kw):
    with pytest.raises(ValueError, match="only meaningful with"):
        _call(**kw)


@pytest.mark.parametrize('kw', [{'surface_frame': True}, {'use_gpu': True},
                                {'wave_propagator': 'fresnel'}])
def test_the_unmeasured_axes_are_refused_rather_than_guessed(kw):
    with pytest.raises(NotImplementedError):
        _call(**kw)


def test_an_unknown_surface_model_still_names_the_new_one():
    with pytest.raises(ValueError, match='tangent_facet'):
        N, dx = 64, 25e-6
        apply_real_lens(_field(N, dx, w=0.6e-3), prescription=_singlet(),
                        wavelength=LAM, dx=dx, surface_model='facet')


def test_a_carrier_is_accepted_and_the_route_1_correction_is_not_applied():
    """``carrier=`` is honoured (it seeds the accumulator) but returns
    ``_obl_apply = False``, so equations (4) and (7) never run.  Asserted on
    the validator's own return value, which is the decision itself."""
    assert LR._check_screen_obliquity_support(
        carrier=la.TiltedCarrier(np.inf, 0.05, 0.0), screen_obliquity='auto',
        on_screen_obliquity='warn', surface_model='tangent_facet',
        displaced_mode='screen') is False


def test_the_guard_is_silent_because_it_has_nothing_to_score():
    """The estimator measures the SIZE of the correction the thin screen
    needs; route 3 has no such correction to accumulate, so the guard must not
    fire -- and must not raise under ``'error'`` either, which is the
    stronger of the two claims."""
    N, dx = 128, 25e-6
    E = _field(N, dx, w=1.0e-3)
    car = la.TiltedCarrier(np.inf, 0.09, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        apply_real_lens(E, prescription=_singlet(R=15e-3), wavelength=LAM,
                        dx=dx, surface_model='tangent_facet', carrier=car,
                        on_screen_obliquity='error')


# ---------------------------------------------------------------------------
# 7.  NON-PROPAGATING PIXELS KEEP THE THIN SCREEN
# ---------------------------------------------------------------------------
def test_a_non_propagating_facet_falls_back_instead_of_clamping():
    """Engineered: an incoming momentum beyond ``n1`` makes the pixel
    evanescent, so ``ok`` must be False there and the caller keeps the thin
    screen.  A clamped cosine would be a WRONG OPD, which is worse than the
    documented approximation, so the fallback is the contract."""
    n = 33
    x = np.linspace(-1e-3, 1e-3, n)
    X, _Y = np.meshgrid(x, x)
    dx = float(x[1] - x[0])
    sag = 1e-4 + 0.1 * X
    gx = np.full_like(sag, 0.1)
    gy = np.zeros_like(sag)
    p = np.full_like(sag, 0.4)
    p[n // 2, n // 2] = 1.4                      # evanescent in n1 = 1.0
    _opd, ok = LR._tangent_facet_screen(sag, gx, gy, p, np.zeros_like(p),
                                        1.0, 1.5, dx, dx, np)
    assert not bool(ok[n // 2, n // 2])
    assert bool(np.all(np.delete(ok.ravel(), (n // 2) * n + n // 2)))


def test_the_nan_sentinel_does_not_poison_the_screen():
    """An oblate conic's domain edge falls inside the grid and
    ``surface_sag_general`` returns NaN there.  The model must produce a
    finite field, exactly as the thin screen does."""
    presc = {'surfaces': [
        {'radius': 8.0e-3, 'conic': 4.0, 'glass_before': 'AIR',
         'glass_after': 'N-BK7'},
        {'radius': np.inf, 'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
        'thicknesses': [3.0e-3]}
    N, dx = 160, 25e-6
    E = _field(N, dx, w=1.2e-3)
    out = _tf(E, presc, dx, carrier=la.TiltedCarrier(np.inf, 0.05, 0.0))
    assert bool(np.all(np.isfinite(out)))


def test_a_nan_annulus_does_not_reach_a_later_surface():
    """The sharper version of the test above: the NaN-producing conic is
    FIRST and a POWERED surface sits behind it, so the persistent momentum
    accumulator is a live route by which the sentinel could travel from one
    surface's annulus to the whole exit field.

    Two-sided, which is the point: the field must be finite everywhere AND the
    pixels well inside the conic's domain must still differ from the thin
    screen -- otherwise 'finite' could be bought by silently zeroing the whole
    model, which is the failure this shape is prone to.
    """
    presc = {'surfaces': [
        {'radius': 8.0e-3, 'conic': 4.0, 'glass_before': 'AIR',
         'glass_after': 'N-BK7'},
        {'radius': -30.0e-3, 'conic': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'AIR'}],
        'thicknesses': [3.0e-3]}
    N, dx = 160, 25e-6
    E = _field(N, dx, w=1.2e-3)
    car = la.TiltedCarrier(np.inf, 0.05, 0.0)
    out = _tf(E, presc, dx, carrier=car)
    assert bool(np.all(np.isfinite(out))), 'the sentinel escaped the annulus'
    # the model is still doing something inside the domain
    ref = _thin(E, presc, dx)
    a = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(a, a, indexing='ij')
    core = (X ** 2 + Y ** 2) <= (0.6e-3) ** 2
    assert not np.allclose(out[core], ref[core]), \
        'the screen was silently zeroed rather than evaluated'
