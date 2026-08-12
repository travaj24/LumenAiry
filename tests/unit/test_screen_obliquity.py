"""The screen-obliquity correction and its accuracy guard (v5.35.0).

``apply_real_lens`` models each surface as a thin phase screen on the
surface's VERTEX plane, so the shipped ``(n2-n1)*sag`` OPD is the OPD of a
ray crossing the sag at NORMAL incidence: the screen is angle-blind by
``~ sag * theta**2``, which is exactly the bound the function's own docstring
has always quoted.  The angular-spectrum steps BETWEEN the screens carry the
gaps' angular optical path exactly (a plane-parallel plate is machine-exact at
every tilt -- ``docs/audits/BUILD_ANGLE_AWARE_LENS_2026_08_11.md``), so the
sag screen is the only angle-blind piece and the only thing corrected here.

The closed form is the AXIAL-TRANSLATION IDENTITY: moving a plane facet a
height ``s`` onto the vertex plane changes the exit-referenced eikonal by
exactly ``s * (n1 cos(alpha_in) - n2 cos(alpha_out))`` with the angles taken
to the Z-AXIS.  ``apply_real_lens(carrier=...)`` imprints that MINUS its
normal-incidence value, so it is identically zero for a plate and identically
zero without a carrier.

What is pinned here:

* the PLATE ZERO -- byte-identical output at every tilt (a zero-sag element
  has nothing to correct, and the shipped model is already exact there);
* the CARRIER-FREE BYTE-NULL, and the zero-angle-carrier byte-null;
* the closed form against EXACT RAY TRACES on a single spherical surface --
  the correction must cut the screen model's own exit-plane angular error by
  two orders of magnitude, and NEGATING it must land on almost exactly DOUBLE
  the blind error (the sign control the refutation used to kill the previous
  design);
* the guard: fires on a steep surface at a large carrier angle, silent
  without a carrier and at a small angle, 'error' raises, 'silent' suppresses,
  and it fires even when the correction is switched OFF;
* the kwarg validators.

Determinism note: the model does an FFT-based ASM propagation through the
glass between surfaces, so byte-equality across separate calls needs FFT plan
determinism -- the fixture pins auto-promote off (matching
``test_slant_chunk_byte_identical``).
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace

LAM = 1.31e-6


@pytest.fixture(autouse=True)
def _deterministic_fft():
    prev = la.get_fft_auto_promote()
    la.set_fft_auto_promote(False)
    yield
    la.set_fft_auto_promote(prev)


# ---------------------------------------------------------------------------
# prescriptions
# ---------------------------------------------------------------------------
def _plate(thickness=25.4e-3, glass='N-BK7'):
    return {'name': 'plate', 'thicknesses': [thickness], 'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': 'air', 'glass_after': glass},
        {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': glass, 'glass_after': 'air'}]}


def _singlet(R=19.6e-3, glass='N-SSK2', thickness=4e-3, aperture=3e-3):
    """A FAST curved face (sag 231 um at 3 mm, |grad sag| = 0.155) followed by
    a flat one -- the steepest single facet in design 121's last group, which
    is the binding case for this correction."""
    return {'name': 'singlet', 'aperture_diameter': aperture,
            'thicknesses': [thickness], 'surfaces': [
                {'radius': R, 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': 'air', 'glass_after': glass},
                {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
                 'glass_before': glass, 'glass_after': 'air'}]}


def _field(N, dx, L=0.0, M=0.0, w=None):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(1j * 2 * np.pi / LAM * (L * X + M * Y))
    if w is not None:
        E = E * np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    return E.astype(np.complex128)


# ---------------------------------------------------------------------------
# the EXACT-RAY oracle, and the screen model traced as a ray system
# ---------------------------------------------------------------------------
def _sag_and_dsag(h_sq, R):
    """Sphere sag and ``d sag / d(h^2)`` (the prescriptions here are conic-free
    spheres, so the closed form is short and exact)."""
    if not np.isfinite(R):
        return np.zeros_like(h_sq), np.zeros_like(h_sq)
    c = 1.0 / R
    sq = np.sqrt(np.maximum(1.0 - c ** 2 * h_sq, 1e-300))
    D = 1.0 + sq
    return c * h_sq / D, c / D + c * h_sq * c ** 2 / (2.0 * sq * D ** 2)


def _exact_eikonal(presc, x0, y0, L, M):
    """The oracle: the shipped exact ray trace from the entrance plane to the
    exit VERTEX plane, with the input plane wave's own eikonal added."""
    p = {k: v for k, v in presc.items() if k != 'aperture_diameter'}
    surfs = surfaces_from_prescription(p)
    n_exit = float(get_glass_index(surfs[-1].glass_after, LAM))
    rays = _make_bundle(x=x0, y=y0, L=np.full(x0.size, L),
                        M=np.full(y0.size, M), wavelength=LAM)
    fin = trace(rays, surfs, LAM, output_filter='last').image_rays
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    return (fin.x + fin.L * t, fin.y + fin.M * t,
            L * x0 + M * y0 + fin.opd + n_exit * t,
            np.asarray(fin.alive, dtype=bool))


def _screen_eikonal(presc, x0, y0, L, M, corrected):
    """``apply_real_lens``'s OWN model traced as a Hamiltonian ray system:
    zero-thickness sag screens (``Lam -= OPD``, ``p -= grad OPD``) separated by
    homogeneous slabs (``x += t p / pz``, ``Lam += t n^2 / pz``, the Legendre
    transform of the ASM kernel).  ``corrected`` adds the library's own closed
    form so the test exercises the shipped expression, not a copy of it;
    ``corrected=-1`` SUBTRACTS it (the sign control)."""
    from lumenairy.elements._lens_real import _screen_obliquity_delta
    surfaces, thick = presc['surfaces'], presc['thicknesses']
    idx = [(float(get_glass_index(s['glass_before'], LAM)),
            float(get_glass_index(s['glass_after'], LAM))) for s in surfaces]
    x, y = x0.copy(), y0.copy()
    px, py = np.full_like(x, L), np.full_like(y, M)
    lam_e = L * x + M * y
    p0x, p0y = np.zeros_like(x), np.zeros_like(y)
    for i, surf in enumerate(surfaces):
        n1, n2 = idx[i]
        sag, dsag = _sag_and_dsag(x ** 2 + y ** 2, surf['radius'])
        gx, gy = 2.0 * x * dsag, 2.0 * y * dsag
        opd = (n2 - n1) * sag
        gox, goy = (n2 - n1) * gx, (n2 - n1) * gy
        if corrected:
            d = _screen_obliquity_delta(sag, gx, gy, p0x, p0y, L, M,
                                        n1, n2, np)
            if corrected == -1:            # the SIGN control
                d = -d
            opd = opd + d
            # the screen's gradient leg through the sag factor (the momentum
            # factors vary on the pupil scale, the sag on the surface scale)
            fac = np.where(sag == 0.0, 0.0, d / np.where(sag == 0.0, 1.0, sag))
            gox, goy = gox + fac * gx, goy + fac * gy
        lam_e = lam_e - opd
        px, py = px - gox, py - goy
        p0x, p0y = p0x - (n2 - n1) * gx, p0y - (n2 - n1) * gy
        if i < len(surfaces) - 1:
            t = float(thick[i])
            pz = np.sqrt(np.maximum(n2 ** 2 - px ** 2 - py ** 2, 1e-300))
            x, y = x + t * px / pz, y + t * py / pz
            lam_e = lam_e + t * n2 ** 2 / pz
    return x, y, lam_e, px, py


def _angular_error_waves(presc, r_pupil, L, M, corrected, n=41):
    """The COMMON-MODE-controlled exit-plane angular error of the screen model,
    piston and tilt removed, in waves.  ``D(theta) - D(0)`` with
    ``D = Lam_model - Lam_exact`` referenced at the EXACT ray's exit point --
    everything angle-independent (the model's documented normal-incidence
    accuracy ceiling) cancels."""
    t = np.linspace(-1.0, 1.0, n)
    PX, PY = np.meshgrid(r_pupil * t, r_pupil * t, indexing='ij')
    keep = (PX ** 2 + PY ** 2) <= r_pupil ** 2
    x0, y0 = PX[keep], PY[keep]
    d = []
    for (l_, m_) in ((L, M), (0.0, 0.0)):
        xe, ye, le, alive = _exact_eikonal(presc, x0, y0, l_, m_)
        xm, ym, lm, pxm, pym = _screen_eikonal(presc, x0, y0, l_, m_,
                                               corrected)
        assert alive.all()
        d.append(lm + pxm * (xe - xm) + pym * (ye - ym) - le)
    g = d[0] - d[1]
    A = np.stack([np.ones_like(x0), x0, y0], axis=1)
    c, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(np.sqrt(((g - A @ c) ** 2).mean())) / LAM


# ---------------------------------------------------------------------------
# (a) THE PLATE ZERO
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('tilt', [0.0, 0.01, 0.02, 0.0415, 0.10])
def test_plane_plate_correction_is_exactly_zero(tilt):
    """A plane-parallel plate has zero sag, so the correction is identically
    zero -- BYTE-identical, not merely small.  This is the control that killed
    the refuted entrance-referenced design, which was wrong by 2.77 waves
    here."""
    N, dx = 128, 20e-6
    presc = _plate()
    E = _field(N, dx, L=tilt)
    base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    got = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                          carrier=la.TiltedCarrier(float('inf'), tilt, 0.0))
    assert np.array_equal(base.view(np.uint8), got.view(np.uint8))


# ---------------------------------------------------------------------------
# (b) THE BYTE-NULLS
# ---------------------------------------------------------------------------
def test_carrier_free_call_is_byte_identical():
    """No carrier -> no angle -> the shipped screens, bit for bit.  Also pins
    that the new keywords at their defaults change nothing."""
    N, dx = 256, 25e-6
    presc = _singlet()
    E = _field(N, dx, w=1.2e-3)
    base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    for kw in ({'carrier': None},
               {'carrier': None, 'screen_obliquity': 'auto'},
               {'screen_obliquity': False},
               {'on_screen_obliquity': 'error'}):
        got = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                              **kw)
        assert np.array_equal(base.view(np.uint8), got.view(np.uint8)), kw


def test_zero_angle_carrier_is_byte_identical():
    """A carrier with zero direction cosines is a zero-angle congruence, and
    the correction is a DIFFERENCE against the zero-angle screen, so it must
    vanish exactly -- not to 1e-16."""
    N, dx = 256, 25e-6
    presc = _singlet()
    E = _field(N, dx, w=1.2e-3)
    base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    got = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                          carrier=la.TiltedCarrier(float('inf'), 0.0, 0.0))
    assert np.array_equal(base.view(np.uint8), got.view(np.uint8))


def test_screen_obliquity_false_leaves_the_field_alone():
    """``screen_obliquity=False`` computes the guard's estimate but must not
    touch the screens."""
    N, dx = 256, 25e-6
    presc = _singlet()
    E = _field(N, dx, w=1.2e-3)
    car = la.TiltedCarrier(float('inf'), 0.055, 0.0)
    base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        off = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                              carrier=car, screen_obliquity=False)
        on = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                             carrier=car, screen_obliquity=True)
    assert np.array_equal(base.view(np.uint8), off.view(np.uint8))
    assert not np.array_equal(base.view(np.uint8), on.view(np.uint8))


# ---------------------------------------------------------------------------
# (c) THE CLOSED FORM AGAINST EXACT RAYS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('R,glass,tilt,floor', [
    (50e-3, 'N-BK7', 0.05487, 100.0),
    (19.6e-3, 'N-SSK2', 0.05487, 50.0),
    (19.6e-3, 'N-SSK2', 0.10, 40.0),
    (-30e-3, 'N-SF11', 0.05487, 50.0),
])
def test_closed_form_beats_the_blind_screen_against_exact_rays(
        R, glass, tilt, floor):
    """The decisive control: the screen model's own exit-plane angular error
    against EXACT ray traces, with the common-mode subtraction that isolates
    the angular part.  The closed form must cut it by at least ``floor``x.

    Measured reductions at 3 mm on the campaign's ladder are 100-700x for a
    single curved facet (the residual is a DIFFERENT defect -- the screen's
    angle-blind deflection kick -- which no sag screen carries); the bars here
    are set well inside those so the pin tests the physics, not the digits."""
    presc = _singlet(R=R, glass=glass)
    blind = _angular_error_waves(presc, 3e-3, tilt, 0.0, corrected=False)
    fixed = _angular_error_waves(presc, 3e-3, tilt, 0.0, corrected=True)
    assert blind > 1e-3, f'case is too easy to discriminate: {blind:.2e} w'
    assert fixed * floor < blind, (
        f'blind {blind:.6f} w rms -> corrected {fixed:.6f} w rms '
        f'({blind / max(fixed, 1e-30):.1f}x, wanted >= {floor}x)')


def test_the_sign_is_load_bearing():
    """The refuted entrance-referenced design was "wrong by twice the term".
    Run that same test here: NEGATE the correction and the exit-plane angular
    error must land on almost exactly DOUBLE the shipped screen's, which is the
    signature of a term that otherwise cancels the defect exactly."""
    presc = _singlet(R=19.6e-3)
    blind = _angular_error_waves(presc, 3e-3, 0.0549, 0.0, corrected=False)
    fixed = _angular_error_waves(presc, 3e-3, 0.0549, 0.0, corrected=True)
    flipped = _angular_error_waves(presc, 3e-3, 0.0549, 0.0, corrected=-1)
    assert fixed < blind / 50.0
    assert abs(flipped / blind - 2.0) < 0.05, (
        f'blind {blind:.6f} -> flipped {flipped:.6f} '
        f'({flipped / blind:.3f}x, wanted 2.00x)')


def test_correction_scales_as_the_square_of_the_angle():
    """The leading term is ``sag * (n2-n1)/(2 n1 n2) * theta**2``, so doubling
    the carrier angle must quadruple the correction -- the signature that
    separates it from the (refuted) entrance-referenced piston, which is the
    SAME size but the opposite sign and is already carried by the gaps."""
    from lumenairy.elements._lens_real import _screen_obliquity_delta
    n1, n2, sag = 1.0, 1.6, 2.0e-4
    gx = gy = 0.0
    d1 = _screen_obliquity_delta(sag, gx, gy, 0.0, 0.0, 0.02, 0.0, n1, n2, np)
    d2 = _screen_obliquity_delta(sag, gx, gy, 0.0, 0.0, 0.04, 0.0, n1, n2, np)
    assert d1 > 0.0                       # a positive-sag n1<n2 facet ADDS OPD
    assert abs(d2 / d1 - 4.0) < 0.01
    # and the leading coefficient itself
    pred = sag * (n2 - n1) / (2.0 * n1 * n2) * 0.02 ** 2
    assert abs(d1 / pred - 1.0) < 1e-3


def test_correction_is_zero_for_zero_sag_at_any_angle():
    """Equation (4) is proportional to the sag, so a flat facet contributes
    exactly nothing however oblique the ray -- the per-surface statement
    behind the plate zero."""
    from lumenairy.elements._lens_real import _screen_obliquity_delta
    for q in (0.0, 0.05, 0.3):
        d = _screen_obliquity_delta(0.0, 0.1, -0.05, 0.02, 0.0, q, 0.0,
                                    1.0, 1.7, np)
        assert float(d) == 0.0


# ---------------------------------------------------------------------------
# (d) THE GUARD
# ---------------------------------------------------------------------------
_GUARD_N, _GUARD_DX = 256, 25e-6


def _steep_case():
    """A fast surface at a large carrier angle: sag 0.42 mm over a 5 mm
    pupil, 90 mrad.  The estimator reads 0.129 waves rms there, so the guard
    fires with the correction ON (budgeted residual 0.051 w) as well as OFF.

    Design 121's own group 5 -- the campaign's binding case -- reads 0.26
    waves and fires the same way; it is not used here because a unit test
    must not depend on the local .zmx.  See
    ``validation/repro_traced_carrier_121/screen_obliquity_derive.py guard``.
    """
    return (_singlet(R=15e-3, aperture=5e-3),
            la.TiltedCarrier(float('inf'), 0.09, 0.0))


def test_guard_fires_on_the_steep_large_angle_case():
    N, dx = _GUARD_N, _GUARD_DX
    presc, car = _steep_case()
    E = _field(N, dx, w=1.5e-3)
    with pytest.warns(RuntimeWarning, match='angle-blind'):
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                        carrier=car)


def test_guard_fires_even_with_the_correction_switched_off():
    """The estimator is the diagnostic, not the fix: turning the correction
    off must not silence the warning -- it makes it larger."""
    N, dx = _GUARD_N, _GUARD_DX
    presc, car = _steep_case()
    E = _field(N, dx, w=1.5e-3)
    with pytest.warns(RuntimeWarning, match='NOT applied'):
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                        carrier=car, screen_obliquity=False)


def test_guard_is_silent_without_a_carrier():
    N, dx = _GUARD_N, _GUARD_DX
    presc, _car = _steep_case()
    E = _field(N, dx, w=1.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)


def test_guard_is_silent_at_a_small_carrier_angle():
    """1 mrad on the same steep surface is 8100x below the 90 mrad case
    (the term is quadratic), i.e. far inside the lambda/20 tolerance."""
    N, dx = _GUARD_N, _GUARD_DX
    presc, _car = _steep_case()
    E = _field(N, dx, w=1.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                        carrier=la.TiltedCarrier(float('inf'), 1e-3, 0.0))


def test_guard_is_silent_on_a_plane_plate_at_a_large_angle():
    """Zero sag -> zero estimate, so the guard must not fire on the one
    element the shipped model is already EXACT on."""
    N, dx = 128, 20e-6
    E = _field(N, dx, L=0.0415)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        apply_real_lens(E, prescription=_plate(), wavelength=LAM, dx=dx,
                        carrier=la.TiltedCarrier(float('inf'), 0.0415, 0.0))


def test_guard_policies():
    N, dx = _GUARD_N, _GUARD_DX
    presc, car = _steep_case()
    E = _field(N, dx, w=1.5e-3)
    kw = dict(prescription=presc, wavelength=LAM, dx=dx, carrier=car)
    with pytest.raises(ValueError, match='angle-blind'):
        apply_real_lens(E, on_screen_obliquity='error', **kw)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        apply_real_lens(E, on_screen_obliquity='silent', **kw)


# ---------------------------------------------------------------------------
# (e) THE VALIDATORS
# ---------------------------------------------------------------------------
def test_screen_obliquity_true_without_a_carrier_raises():
    N, dx = 64, 30e-6
    with pytest.raises(ValueError, match='needs carrier'):
        apply_real_lens(_field(N, dx), prescription=_singlet(),
                        wavelength=LAM, dx=dx, screen_obliquity=True)


def test_carrier_with_displaced_surface_model_raises():
    """The displaced path is ALREADY angle-aware through ``conjugate=`` -- it
    modifies the same per-surface sag OPD with true ray cosines -- so stacking
    the two would double-count.  Refuse rather than degrade."""
    N, dx = 64, 30e-6
    with pytest.raises(ValueError, match='double-count'):
        apply_real_lens(_field(N, dx), prescription=_singlet(),
                        wavelength=LAM, dx=dx, surface_model='displaced',
                        carrier=la.TiltedCarrier(float('inf'), 0.02, 0.0))


@pytest.mark.parametrize('kw,match', [
    ({'on_screen_obliquity': 'shout'}, 'on_screen_obliquity'),
    ({'screen_obliquity': 'yes'}, 'screen_obliquity must be'),
])
def test_bad_policy_values_raise(kw, match):
    N, dx = 64, 30e-6
    with pytest.raises(ValueError, match=match):
        apply_real_lens(_field(N, dx), prescription=_singlet(),
                        wavelength=LAM, dx=dx, **kw)


# ---------------------------------------------------------------------------
# (f) CARRIER VOCABULARY
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('carrier', [
    0.5,                                   # signed on-axis conjugate [m]
    'auto',                                # a fit of E_in's own phase
])
def test_other_carrier_vocabularies_run_and_change_the_field(carrier):
    """``carrier=`` takes the traced path's whole vocabulary, not just a
    TiltedCarrier: a scalar conjugate and an 'auto' fit both resolve to a
    (L, M) field and both produce a non-trivial correction."""
    N, dx = 256, 25e-6
    presc = _singlet()
    E = _field(N, dx, L=0.02, w=1.2e-3)
    base = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                              carrier=carrier)
    assert np.all(np.isfinite(got))
    assert not np.array_equal(base.view(np.uint8), got.view(np.uint8))


def test_explicit_wavefront_carrier_matches_the_equivalent_tilt():
    """An explicit wavefront ndarray describing a uniform tilt must give the
    same correction as the TiltedCarrier that describes it."""
    N, dx = 256, 25e-6
    presc = _singlet()
    tilt = 0.03
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = _field(N, dx, L=tilt, w=1.2e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                            carrier=la.TiltedCarrier(float('inf'), tilt, 0.0))
        b = apply_real_lens(E, prescription=presc, wavelength=LAM, dx=dx,
                            carrier=(tilt * X).astype(np.float64))
    assert np.allclose(np.angle(a * np.conj(b)), 0.0, atol=2e-6)
