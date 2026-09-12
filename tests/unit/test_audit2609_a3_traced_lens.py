"""WP-A3 / audit 2026-09-11: ``apply_real_lens_traced`` and its infrastructure.

T1 (real-dtype input), T3 (spline + a vignetted ray), T4
(``fast_analytic_phase``), T5 (``form_error``), T6 (``_reverse_prescription``),
T7 / T12 (the tilt and carrier samplers), T8 (the ``_multi`` / ``segmented``
kwarg and grid contracts), T9 / T10 (the chunked and blocked kernels), T11 (the
Chebyshev consolidation) and T13 (the exit-NA gate).

The §14 V3 finding is why several of these exist at all: against five seeded
defects, D3 (a real-dtype input) was "untested by intent, reached accidentally
in ~107 places and therefore PINNED rather than caught", D5 (``newton_fit=
'spline'`` with a vignetted ray) had 0 of 18 spline tests able to catch it, and
D4 (multibranch at the paraxial focus) was visited by 6 tests that all asserted
on the WARNING rather than the field.  Each of those now has a fixture with the
geometry it needs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.elements._lens_traced as LT
from lumenairy.elements import apply_real_lens
from lumenairy.elements._lens_traced import (
    _cheb_deriv_vand_2d,
    _cheb_vand_2d,
    _geometric_lens_phase,
    _reverse_prescription,
    _sample_local_tilts,
    apply_real_lens_traced,
    apply_real_lens_traced_segmented,
)
from lumenairy.glass import get_glass_index

# Dispersionless model glass -- registered / removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_A3TG': lambda wl: 1.5168}

_WL = 587.6e-9
_N_GLASS = 1.5168


def _singlet(r1=0.1, r2=-0.1, thick=4e-3, aperture=6e-3, **extra):
    s0 = {'radius': r1, 'glass_before': 'air', 'glass_after': '_A3TG'}
    s0.update(extra)
    return {'surfaces': [s0,
                         {'radius': r2, 'glass_before': '_A3TG',
                          'glass_after': 'air'}],
            'thicknesses': [thick],
            'aperture_diameter': aperture}


def _grid(N=256, dx=25e-6):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return N, dx, x, X, Y


def _gauss(X, Y, w=1.5e-3):
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2)


_QUIET = dict(on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent')


# ===========================================================================
# T1 -- a REAL-dtype E_in
# ===========================================================================
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_t1_real_dtype_input_runs_and_returns_complex128(dtype):
    """``apply_real_lens`` accepts a real field and returns complex128;
    ``apply_real_lens_traced`` crashed on the LAST assembly line with
    ``AttributeError: type object 'numpy.complex128' has no attribute 'type'``
    -- after the whole ray trace, the three fits and the Newton inversion had
    run -- because ``target_cdtype`` was a numpy scalar TYPE rather than a
    ``np.dtype``.  Four sites are affected, two of them on the BANDED path,
    which is the shipped default at N >= 4096.

    DECISION test: does the call complete, and is the result the same field
    the equivalent complex input gives?  No tolerance on the completion; a
    tight one on the equality, because promoting a real array to complex is
    exact for float64 and is the same rounding for float32.
    """
    N, dx, x, X, Y = _grid()
    E_real = _gauss(X, Y).astype(dtype)
    presc = _singlet()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(E_real, prescription=presc,
                                     wavelength=_WL, dx=dx, **_QUIET)
        ref = apply_real_lens_traced(E_real.astype(np.complex128),
                                     prescription=presc, wavelength=_WL,
                                     dx=dx, **_QUIET)
    assert np.iscomplexobj(out)
    assert out.dtype == np.dtype(np.complex128)
    assert out.shape == E_real.shape
    assert np.isfinite(out).all()
    assert float(np.abs(out - ref).max()) <= 1e-12 * float(np.abs(ref).max())


def test_t1_the_pre_fix_expression_is_the_one_that_raised():
    """FAIL-BEFORE, in one line: the scalar TYPE has no ``.type``, the dtype
    does.  This is the whole defect, and it is why the fix is ``np.dtype``."""
    with pytest.raises(AttributeError):
        np.complex128.type(0)                                # noqa: B018
    assert np.dtype(np.complex128).type(0) == 0


def test_t1_banded_assembly_also_accepts_a_real_input():
    """Two of the four broken sites are on the row-band path.  Force it with
    an explicit ``sag_chunk_rows`` rather than waiting for N >= 4096."""
    N, dx, x, X, Y = _grid(N=128, dx=50e-6)
    presc = _singlet(aperture=5e-3)
    E_real = _gauss(X, Y, w=1.2e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(E_real, prescription=presc,
                                     wavelength=_WL, dx=dx, ray_subsample=2,
                                     sag_chunk_rows=32, **_QUIET)
    assert out.dtype == np.dtype(np.complex128) and np.isfinite(out).all()


# ===========================================================================
# T3 -- newton_fit='spline' with a vignetted launch ray
# ===========================================================================
def test_t3_spline_with_a_vignetted_ray_returns_a_field_not_zeros():
    """The launch lattice is a SQUARE of half-width ``0.75*aperture``, so its
    corners sit at ``sqrt(2)*0.75 = 1.06`` aperture RADII -- past any
    per-surface ``semi_diameter`` of ``aperture/2``.  Those rays die,
    ``RectBivariateSpline`` (an interpolating s=0 FITPACK fit) turns ~90 % of
    its coefficients into NaN, and the returned field is IDENTICALLY ZERO.

    BAR.  Transmitted power within 15 % of the polynomial fit's.  The two fits
    are different interpolants over the same rays and the spline path now
    additionally MASKS the pixels whose entrance solution lands on a filled
    node, so they are not expected to agree closely; the pre-fix value is
    exactly 0.0, i.e. 100 % below, so the bar has 6x of gap on the defect side
    and the measured difference (2.6 %) leaves 5x on the other.
    """
    N, dx = 384, 8e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E_in = _gauss(X, Y, w=0.8e-3).astype(np.complex128)
    presc = _singlet(r1=0.06, r2=-0.06, aperture=3e-3)
    for s in presc['surfaces']:
        s['semi_diameter'] = 1.5e-3                # = aperture / 2
    kw = dict(prescription=presc, wavelength=_WL, dx=dx, ray_subsample=4,
              **_QUIET)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_poly = apply_real_lens_traced(E_in, newton_fit='polynomial', **kw)
        E_spl = apply_real_lens_traced(E_in, newton_fit='spline', **kw)
    p_in = float((np.abs(E_in) ** 2).sum())
    p_poly = float((np.abs(E_poly) ** 2).sum()) / p_in
    p_spl = float((np.abs(E_spl) ** 2).sum()) / p_in
    assert p_poly > 0.5, f'the polynomial control transmits only {p_poly:.4f}'
    assert int((np.abs(E_spl) > 0).sum()) > 0, (
        'newton_fit=spline returned an identically ZERO field on a vignetting '
        'prescription -- the NaN launch nodes poisoned the FITPACK solve')
    assert abs(p_spl - p_poly) / p_poly < 0.15, (
        f'spline {p_spl:.4f} vs polynomial {p_poly:.4f}')


def test_t3_one_nan_still_poisons_a_raw_rectbivariatespline():
    """FAIL-BEFORE / PREMISE: the mechanism is FITPACK's, and it is total.
    If this ever stops being true the fill above is unnecessary."""
    RectBivariateSpline = pytest.importorskip(
        'scipy.interpolate').RectBivariateSpline
    xs = np.linspace(-1.0, 1.0, 21)
    Z = np.add.outer(xs ** 2, xs ** 2)
    clean = RectBivariateSpline(xs, xs, Z, kx=3, ky=3)
    assert np.isfinite(clean.ev([0.0, 0.3], [0.0, 0.5])).all()
    Zn = Z.copy()
    Zn[0, 0] = np.nan
    poisoned = RectBivariateSpline(xs, xs, Zn, kx=3, ky=3)
    assert not np.isfinite(poisoned.ev([0.0, 0.3], [0.0, 0.5])).any()


def test_t3_spline_warns_and_names_the_polynomial_remedy():
    """The only diagnostic used to be a 100 %-unconverged Newton warning,
    which misdiagnoses the cause AND mis-states the outcome."""
    N, dx = 256, 10e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E_in = _gauss(X, Y, w=0.6e-3).astype(np.complex128)
    presc = _singlet(r1=0.06, r2=-0.06, aperture=2.4e-3)
    for s in presc['surfaces']:
        s['semi_diameter'] = 1.2e-3
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        apply_real_lens_traced(
            E_in, prescription=presc, wavelength=_WL, dx=dx, ray_subsample=4,
            newton_fit='spline', on_undersample='error',
            on_noncollimated='off', on_aperture_beam='silent')
    msgs = [str(w.message) for w in rec if 'spline' in str(w.message)]
    assert msgs, 'no diagnostic at all on a vignetting spline call'
    assert any("newton_fit='polynomial'" in m for m in msgs), msgs


# ===========================================================================
# T4 -- fast_analytic_phase on a REFRACTING prescription
# ===========================================================================
def test_t4_geometric_lens_phase_runs_on_a_refracting_prescription():
    """``_geometric_lens_phase`` did ``from .. import raytrace as _rt`` and
    then ``_rt._surface_sag_xy(...)`` -- which lives in ``raytrace.surface``
    and is NOT re-exported by the package.  The loop ``continue``s only when
    ``|n_after - n_before| < 1e-15``, so EVERY refracting prescription hit the
    ``AttributeError``; the two tests that touched the function deliberately
    used a no-refraction fixture.

    DECISION test plus an accuracy check against the analytic sibling's own
    phase, which is the quantity this function exists to approximate.

    BAR.  0.35 rad max over the lit region on a 1 um-thick element.  The
    omitted term is the in-glass ASM leg, whose error is LINEAR in centre
    thickness: measured 3.39e-04 rad max at 1 um, 3.40e-02 at 100 um and
    6.81e-01 at 2 mm.  At 1 um the bar sits 1000x above the residual, and the
    pre-fix behaviour is an exception, so there is no bar to cross at all on
    that side.
    """
    import lumenairy.raytrace as _rt
    assert not hasattr(_rt, '_surface_sag_xy'), (
        'the package now re-exports _surface_sag_xy, so this fixture no '
        'longer demonstrates why the import has to name the module')
    N, dx = 256, 40e-6
    presc = _singlet(r1=0.1, r2=-0.1, thick=1e-6, aperture=8e-3)
    phase = _geometric_lens_phase(presc, _WL, dx, N)
    assert phase.shape == (N, N) and np.isfinite(phase).all()

    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    ones = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_a = apply_real_lens(ones, prescription=presc, wavelength=_WL, dx=dx)
    lit = np.abs(E_a) > 0.5 * np.abs(E_a).max()
    d = np.angle(np.exp(1j * (phase[lit] - np.angle(E_a[lit]))))
    d = d - np.median(d)                     # piston is not under test
    assert float(np.abs(d).max()) < 0.35, (
        f'max |dphi| vs the analytic model = {float(np.abs(d).max()):.4e} rad')


def test_t4_fast_analytic_phase_end_to_end_on_a_refracting_lens():
    """The public knob (and a GUI checkbox) must not raise."""
    N, dx, x, X, Y = _grid(N=128, dx=50e-6)
    presc = _singlet(thick=1e-3, aperture=5e-3)
    E_in = _gauss(X, Y, w=1.2e-3).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(
            E_in, prescription=presc, wavelength=_WL, dx=dx,
            ray_subsample=2, fast_analytic_phase=True, **_QUIET)
    assert np.isfinite(out).all() and np.abs(out).max() > 0


def test_t4_out_of_conic_domain_gives_zero_not_nan():
    """``surface_sag_general`` returns NaN outside the conic domain; that used
    to propagate into ``np.angle(np.exp(1j*phase))`` and from there into
    ``delta_phase``, blanking those pixels of the returned field with no
    diagnostic.  Now: 0 from that surface, plus a warning naming the count."""
    N, dx = 96, 120e-6          # grid half-extent 5.76 mm > R = 5 mm
    presc = {'surfaces': [
        {'radius': 5e-3, 'glass_before': 'air', 'glass_after': '_A3TG'},
        {'radius': float('inf'), 'glass_before': '_A3TG',
         'glass_after': 'air'}],
        'thicknesses': [1e-3], 'aperture_diameter': 4e-3}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        phase = _geometric_lens_phase(presc, _WL, dx, N)
    assert np.isfinite(phase).all(), 'NaN leaked out of the conic-domain guard'
    assert any('conic domain' in str(w.message) for w in rec), (
        [str(w.message) for w in rec])


def test_t4_bulk_piston_is_folded_before_a_float32_accumulator_sees_it():
    """``k0 n t`` is ~2.9e4 rad for 4 mm of glass at 1.31 um, whose float32
    ulp is 1.95e-3 rad (lambda/3220) and grows linearly with thickness.  The
    piston is now reduced mod 2*pi in float64 first.

    BAR.  The float32 and float64 answers must agree to 1e-4 rad over the lit
    region.  float32's ulp at |phase| <= pi is 2.4e-7 rad, so 1e-4 is ~400x
    above the representable floor; the pre-fix error is the 1.95e-3 rad ulp
    at 2.9e4 rad, 20x above the bar.
    """
    prop = pytest.importorskip('lumenairy.propagators.propagation')
    N, dx = 128, 60e-6
    presc = _singlet(r1=0.1, r2=-0.1, thick=4e-3, aperture=5e-3)
    p64 = _geometric_lens_phase(presc, 1.31e-6, dx, N)
    old = prop.get_default_real_dtype()
    try:
        prop.set_default_real_dtype(np.float32)
        p32 = _geometric_lens_phase(presc, 1.31e-6, dx, N)
    finally:
        prop.set_default_real_dtype(old)
    d = np.angle(np.exp(1j * (np.asarray(p32, dtype=np.float64) - p64)))
    assert float(np.abs(d).max()) < 1e-4, (
        f'float32 vs float64 geometric phase {float(np.abs(d).max()):.3e} rad')


# ===========================================================================
# T5 -- form_error must survive the traced assembly
# ===========================================================================
def _astig_form_error(X, Y, pv=250e-9, r_ap=3.0e-3):
    return pv * ((X ** 2 - Y ** 2) / r_ap ** 2)


def test_t5_form_error_is_applied_not_cancelled():
    """BOTH analytic legs of ``E_analytic * exp(i(k0 opl - phi_analytic))``
    carry ``phi_form`` and the ray OPL carries none, so a surface figure error
    CANCELLED out of the traced answer -- measured 254x suppression on a
    250 nm PV astigmatic map (2.57e-03 rad against the analytic model's
    6.54e-01 rad), with no warning and no mention in the docstring.

    ORACLE: the exact screen ``-k0 (n - 1) form_error``, which is what
    ``apply_real_lens`` applies (CONVENTIONS §7: a positive sag in a denser
    medium retards).  BAR: 15 % of the screen's own rms.  The traced model
    reproduces it to 0.6 % (3.4e-03 rad rms against a 5.64e-01 rad screen) --
    better than the analytic sibling's 2.6 %, which carries the ASM
    diffraction of the figure error -- and the pre-fix value is the whole
    screen, i.e. 100 %, so the bar has 6.7x on one side and 25x on the other.
    """
    N, dx, x, X, Y = _grid()
    fe = _astig_form_error(X, Y)
    E_in = _gauss(X, Y).astype(np.complex128)
    kw = dict(wavelength=_WL, dx=dx, ray_subsample=8, **_QUIET)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = apply_real_lens_traced(E_in, prescription=_singlet(), **kw)
        b = apply_real_lens_traced(
            E_in, prescription=_singlet(form_error=fe), **kw)
    m = np.abs(a) > 1e-2 * np.abs(a).max()
    applied = np.angle(b[m] * np.conj(a[m]))
    screen = -(2 * np.pi / _WL) * (_N_GLASS - 1.0) * fe[m]
    err = np.angle(np.exp(1j * (applied - screen)))
    rms_screen = float(np.sqrt(np.mean(screen ** 2)))
    rms_err = float(np.sqrt(np.mean(err ** 2)))
    assert rms_screen > 0.3, 'the fixture applies no measurable figure error'
    assert rms_err / rms_screen < 0.15, (
        f'applied-vs-exact screen {rms_err:.3e} rad rms against a '
        f'{rms_screen:.3e} rad screen')


def test_t5_fail_before_without_the_screen_the_error_is_the_whole_map():
    """FAIL-BEFORE by disabling the re-application in process: the traced
    model then reproduces NONE of the figure error."""
    N, dx, x, X, Y = _grid()
    fe = _astig_form_error(X, Y)
    E_in = _gauss(X, Y).astype(np.complex128)
    kw = dict(wavelength=_WL, dx=dx, ray_subsample=8, **_QUIET)
    real_builder = LT._form_error_phase_screen
    try:
        LT._form_error_phase_screen = (
            lambda *a, **k: None)            # the pre-fix behaviour exactly
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = apply_real_lens_traced(E_in, prescription=_singlet(), **kw)
            b = apply_real_lens_traced(
                E_in, prescription=_singlet(form_error=fe), **kw)
    finally:
        LT._form_error_phase_screen = real_builder
    m = np.abs(a) > 1e-2 * np.abs(a).max()
    applied_rms = float(np.sqrt(np.mean(
        np.angle(b[m] * np.conj(a[m])) ** 2)))
    screen_rms = float(np.sqrt(np.mean(
        ((2 * np.pi / _WL) * (_N_GLASS - 1.0) * fe[m]) ** 2)))
    assert applied_rms < 0.05 * screen_rms, (
        'the pre-fix path should suppress the figure error by ~2 decades; '
        f'got {applied_rms:.3e} against a {screen_rms:.3e} rad screen')


def test_t5_caustic_modes_refuse_a_form_error_rather_than_drop_it():
    """The branch-enumeration modes have no analytic leg to re-apply the
    screen onto, so they must REFUSE rather than return the nominal field."""
    N, dx, x, X, Y = _grid(N=128, dx=40e-6)
    presc = _singlet(aperture=4e-3, form_error=_astig_form_error(X, Y))
    E_in = _gauss(X, Y, w=1.0e-3).astype(np.complex128)
    with pytest.raises(ValueError, match='form_error'):
        apply_real_lens_traced(
            E_in, prescription=presc, wavelength=_WL, dx=dx,
            amplitude_model='ray_density', caustic='multibranch', **_QUIET)


# ===========================================================================
# T6 -- _reverse_prescription
# ===========================================================================
def _sag_at(presc, surf_index, h):
    import lumenairy.raytrace as _rt
    from lumenairy.raytrace.surface import _surface_sag_xy  # noqa: I001

    s = _rt.surfaces_from_prescription(presc)[surf_index]
    return float(_surface_sag_xy(np.array([h]), np.array([0.0]), s)[0])


def test_t6_every_sag_term_changes_sign_under_the_reversal():
    """Reversal is the reflection ``z -> -z``, so ``sag -> -sag`` TERM BY
    TERM.  The radius flip supplies the sign for the conic; the polynomial
    aspheric, the freeform block, the field-frame ``tilt`` and a
    ``sag_callable`` have no radius to flip and were passed through UNCHANGED.

    ORACLE: the forward surface's own sag, negated.  BAR: 1e-12 m against a
    measured 0.0 m (the two are the same closed form evaluated twice) and a
    pre-fix error of 1.25e-06 m = 2.13 waves at 588 nm.
    """
    presc = _singlet(aperture=12e-3)
    presc['surfaces'][0]['aspheric_coeffs'] = {4: 1.0e3, 6: 2.0e7}
    presc['surfaces'][0]['tilt'] = (1e-3, -2e-3)
    rev = _reverse_prescription(presc)
    h = 5e-3
    fwd = _sag_at(presc, 0, h)
    bwd = _sag_at(rev, 1, h)            # forward S0 is reversed S1
    assert abs(bwd + fwd) < 1e-12, (
        f'sag(forward S0) = {fwd:.9e}, sag(reversed S1) = {bwd:.9e} -- these '
        f'must be exact negatives')
    assert abs(fwd) > 1e-5, 'the fixture has no measurable sag to negate'
    # the coefficients themselves, so the intent is visible
    assert rev['surfaces'][1]['aspheric_coeffs'] == {4: -1.0e3, 6: -2.0e7}
    assert rev['surfaces'][1]['tilt'] == (-1e-3, 2e-3)


def test_t6_sag_callable_is_negated_and_wrapped():
    presc = _singlet()
    presc['surfaces'][0]['sag_callable'] = lambda X, Y: 1e-6 * np.cos(
        X / 1e-3)
    rev = _reverse_prescription(presc)
    f = rev['surfaces'][1]['sag_callable']
    Xq = np.array([0.0, 0.5e-3, 1.0e-3])
    got = np.asarray(f(Xq, np.zeros_like(Xq)))
    exp = -1e-6 * np.cos(Xq / 1e-3)
    assert np.allclose(got, exp, rtol=0, atol=1e-18)


@pytest.mark.parametrize('n_thick', [1, 2])
def test_t6_thickness_pairing_survives_both_conventions(n_thick):
    """``validate_prescription`` accepts ``len(thicknesses) == len(surfaces)``
    (each surface's forward gap, the last being the BFD) and ``== len - 1``.
    ``list(reversed(thicknesses))`` is correct only for the second: under the
    first it made the reversed GLASS gap the forward BFD -- measured OPL
    7.58e-03 m forward against 1.52e-01 m backward, and a ray launched at
    +4.000 mm landing at +6.576 mm.

    ORACLE: Fermat -- the forward OPL through P equals the backward OPL
    through reverse(P) for the same ray, both referenced to their own vertex
    planes.  BAR: 1e-12 m on a ~7.6e-03 m path, i.e. 1.3e-10 relative, against
    a measured 5e-18 m floor and a pre-fix 1.44e-01 m error.
    """
    import lumenairy.raytrace as _rt
    presc = _singlet(r1=0.1, r2=-0.1, thick=5e-3, aperture=10e-3)
    presc['thicknesses'] = ([5e-3, 0.1] if n_thick == 2 else [5e-3])
    rev = _reverse_prescription(presc)
    assert len(rev['thicknesses']) == len(presc['thicknesses'])
    if n_thick == 2:
        assert rev['thicknesses'][0] == pytest.approx(5e-3), (
            f'reversed glass gap {rev["thicknesses"][0]} -- the pairing, not '
            f'the list, has to be reversed')

    def _trace(p, x0, L0):
        """Exit-vertex ``(opl, x, L, alive)`` for one meridional ray."""
        s = _rt.surfaces_from_prescription(p)
        N0 = float(np.sqrt(max(0.0, 1.0 - L0 * L0)))
        rays = _rt.RayBundle(x=np.array([x0]), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L0]), M=np.zeros(1),
                             N=np.array([N0]), wavelength=_WL,
                             alive=np.ones(1, dtype=bool), opd=np.zeros(1))
        ex = _rt.trace(rays, s, _WL).at_exit_vertex()
        return (float(ex.opd[0]), float(ex.x[0]), float(ex.L[0]),
                bool(ex.alive[0]))

    for h in (0.0, 2e-3, 4e-3):
        f_opl, f_x, f_L, f_alive = _trace(presc, h, 0.0)
        # Launch the reverse ray from the forward EXIT, travelling back along
        # the forward exit direction.  In the reversed (z -> -z) frame that is
        # ``(-L, +N)``: the transverse cosine flips, the axial one does not.
        b_opl, b_x, _b_L, b_alive = _trace(rev, f_x, -f_L)
        assert f_alive and b_alive
        assert abs(f_opl - b_opl) < 1e-12, (
            f'h={h}: forward {f_opl:.12e} vs backward {b_opl:.12e}')
        assert abs(b_x - h) < 1e-9, f'back-traced ray lands at {b_x} not {h}'


def test_t6_top_level_keys_are_carried_and_stop_index_is_remapped():
    presc = _singlet(aperture=8e-3)
    presc['surfaces'].insert(1, {'radius': float('inf'),
                                 'glass_before': '_A3TG',
                                 'glass_after': '_A3TG'})
    presc['thicknesses'] = [2e-3, 2e-3]
    presc['stop_index'] = 0
    presc['elements'] = ['front', 'middle', 'back']
    presc['name'] = 'A3 fixture'
    rev = _reverse_prescription(presc)
    assert rev['stop_index'] == 2                # len(surfaces) - 1 - 0
    assert rev['elements'] == ['back', 'middle', 'front']
    assert rev['name'] == 'A3 fixture'
    assert rev['aperture_diameter'] == presc['aperture_diameter']
    # ...and the whole thing round-trips
    rev2 = _reverse_prescription(rev)
    assert rev2['stop_index'] == presc['stop_index']
    assert rev2['elements'] == presc['elements']
    assert rev2['thicknesses'] == presc['thicknesses']
    for a, b in zip(rev2['surfaces'], presc['surfaces']):
        assert a['radius'] == b['radius']
        assert a['glass_before'] == b['glass_before']


# ===========================================================================
# T7 -- _sample_local_tilts: the np.roll wrap and the half-pixel offset
# ===========================================================================
def test_t7_tilt_sampler_has_no_wrapped_boundary_column():
    """``np.roll(E, -1, axis=1)`` differenced the LAST column against column
    0, so on any field that fills the grid -- a plane wave, a top hat, a
    post-DOE multi-order field, i.e. what this function exists for -- the rim
    read a tilt unrelated to the beam (measured L = -0.1175 against a true
    +0.03, 5x the tilt itself), and the shipped sigma = 4 px smoothing SPREAD
    it 12 columns inward rather than suppressing it.

    ORACLE: the exact launch tilt, which a uniform-amplitude plane wave has
    everywhere.  BAR: 1e-6 on |L - L0| over the WHOLE grid, against a measured
    1.2e-15 floor (float64 phase round-off) and a pre-fix whole-grid error of
    1.475e-01 (sigma = 0) / 8.111e-02 (sigma = 4).  9 decades above the floor,
    5 below the defect.
    """
    N, dx = 256, 4e-6
    lam = 1.31e-6
    k0 = 2 * np.pi / lam
    L0, M0 = 0.03, -0.02
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(1j * k0 * (L0 * X + M0 * Y))
    for sigma in (0.0, 4.0):
        L, M = _sample_local_tilts(E, lam, dx, X, Y, smooth_sigma_px=sigma)
        assert float(np.abs(L - L0).max()) < 1e-6, (
            f'sigma={sigma}: whole-grid max |L - L0| = '
            f'{float(np.abs(L - L0).max()):.4e}')
        assert float(np.abs(M - M0).max()) < 1e-6


def test_t7_tilt_sampler_has_no_half_pixel_bias_on_a_curved_wavefront():
    """``angle(E[i+1] conj(E[i]))/dx`` estimates dphi/dx at ``i + 1/2`` and
    was STORED at ``i``, biasing every launch direction by ``dx/(2R)`` --
    coherent across the pupil, invisible on a collimated fixture and exactly
    predicted.

    ORACLE: the analytic tilt of a spherical wave at the PIXEL CENTRE,
    ``L(x) = x / sqrt(x^2 + R^2)``.  BAR: mean bias below 1e-9, against a
    measured 4.5e-21 and a pre-fix +2.0000e-05 (R = 0.10 m) / +4.0000e-05
    (R = 0.05 m) -- which equal ``dx/(2R)`` to five digits, so the defect is
    identified, not merely bounded.
    """
    N, dx = 256, 4e-6
    lam = 1.31e-6
    k0 = 2 * np.pi / lam
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    for R in (0.10, 0.05):
        E = np.exp(1j * k0 * np.sqrt(X ** 2 + Y ** 2 + R ** 2))
        L, _ = _sample_local_tilts(E, lam, dx, X, Y, smooth_sigma_px=0.0)
        core = (np.abs(X) < 0.25 * N * dx) & (np.abs(Y) < 0.25 * N * dx)
        L_true = X / np.sqrt(X ** 2 + Y ** 2 + R ** 2)
        bias = float(np.mean(L[core] - L_true[core]))
        assert abs(bias) < 1e-9, (
            f'R={R} m: mean bias {bias:+.4e}, predicted half-pixel bias '
            f'{dx / (2 * R):+.4e}')


# ===========================================================================
# T7 / T12 -- carrier=<ndarray> sampling
# ===========================================================================
def test_t12_ndarray_carrier_matches_the_analytically_identical_float_one():
    """``carrier=<ndarray>`` is the documented way to hand the element a
    measured wavefront, and it sampled both the direction cosines AND (the
    larger, and the one the in-code note omitted) the H6 entrance eikonal by
    NEAREST NEIGHBOUR at launch-lattice nodes that are not grid centres.  The
    error is linear in dx: 301.4 / 150.9 / 69.0 nm rms at N = 256 / 512 / 1024,
    exactly the predicted ``|grad W| dx/2``.

    ORACLE: ``carrier=<float>``, which for an exact-sphere W is the SAME
    wavefront in closed form.  BAR: the two exit fields agree to 0.05 rad rms
    over the bright core.  Measured 1.9e-03 rad (the ndarray path is now
    within 1.5 % of the float path's own 1.85e-03 rad residual against an
    independent trace); pre-fix the gap was 9.9e-02 rad rms, 2x above the bar.
    """
    N, dx = 384, 10.5e-6
    lam = 1.31e-6
    s = 0.2                                   # diverging point source, 200 mm
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    W = np.sign(s) * (np.sqrt(X ** 2 + Y ** 2 + s * s) - abs(s))
    E_in = (np.exp(-(X ** 2 + Y ** 2) / (1.0e-3) ** 2)
            * np.exp(1j * (2 * np.pi / lam) * W)).astype(np.complex128)
    presc = _singlet(r1=0.05, r2=float('inf'), thick=2e-3, aperture=3e-3)
    kw = dict(prescription=presc, wavelength=lam, dx=dx, ray_subsample=4,
              **_QUIET)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_f = apply_real_lens_traced(E_in, carrier=s, **kw)
        E_a = apply_real_lens_traced(E_in, carrier=W, **kw)
    core = np.abs(E_f) > 0.2 * np.abs(E_f).max()
    assert int(core.sum()) > 2000
    d = np.angle(E_a[core] * np.conj(E_f[core]))
    d = d - np.median(d)
    rms = float(np.sqrt(np.mean(d ** 2)))
    assert rms < 0.05, f'ndarray vs float carrier {rms:.4e} rad rms'


# ===========================================================================
# T8 -- the _multi / segmented contracts
# ===========================================================================
def test_t8_multi_names_every_kwarg_the_prepared_screen_cannot_take():
    """``apply_real_lens_traced_multi(reuse_prepared=True)`` raised an opaque
    ``TypeError: prepare_real_lens_traced() got an unexpected keyword
    argument`` from three frames down for TWELVE public kwargs -- and only on
    the DEFAULT reuse path, so the same call worked or crashed depending on
    the carrier kind.  The accepted set is now DERIVED from the two
    signatures, so this cannot drift again.
    """
    import inspect

    from lumenairy.elements._lens_traced import (
        apply_real_lens_traced_multi,
        prepare_real_lens_traced,
    )
    traced = set(inspect.signature(apply_real_lens_traced).parameters)
    prep = set(inspect.signature(prepare_real_lens_traced).parameters)
    missing = sorted(k for k in traced - prep if not k.startswith('_'))
    assert missing, 'the two signatures now agree -- this test is obsolete'

    N, dx = 128, 40e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = _gauss(X, Y, w=1.0e-3).astype(np.complex128)
    presc = _singlet(aperture=4e-3)
    for key, val in (('on_pool_memory', 'silent'),
                     ('newton_mask_dilate_coarse_px', 3),
                     ('dy', dx), ('origin', (0.0, 0.0)),
                     ('caustic_band', 'plain')):
        with pytest.raises(ValueError) as exc:
            apply_real_lens_traced_multi(
                [E, E], prescription=presc, wavelength=_WL, dx=dx,
                carriers=None, reuse_prepared=True, **{key: val})
        msg = str(exc.value)
        assert key in msg and 'reuse_prepared=False' in msg, msg


def test_t8_segmented_refuses_an_anamorphic_grid():
    """It used to hand ``dy`` to the angular partition and ``dx`` alone to
    every traced pass, so the element's own square-pixel refusal was never
    reached and the result was internally inconsistent -- silently."""
    N, dx = 128, 40e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = _gauss(X, Y, w=1.0e-3).astype(np.complex128)
    presc = _singlet(aperture=4e-3)
    with pytest.raises(ValueError, match='square'):
        apply_real_lens_traced_segmented(
            E, prescription=presc, wavelength=_WL, dx=dx, dy=2 * dx)
    # the direct call raises the same class, which is the consistency claim
    with pytest.raises(ValueError, match='square'):
        apply_real_lens_traced(E, prescription=presc, wavelength=_WL, dx=dx,
                               dy=2 * dx)


def test_t8_segmented_rejects_a_mismatched_carrier_sequence():
    """The segment count is DATA-dependent, so forwarding a per-segment list
    as a scalar ``carrier=`` made the same call valid or ill-typed depending
    on the input spectrum."""
    N, dx = 128, 40e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = _gauss(X, Y, w=1.0e-3).astype(np.complex128)   # unimodal -> 1 segment
    presc = _singlet(aperture=4e-3)
    segs = apply_real_lens_traced_segmented(
        E, prescription=presc, wavelength=_WL, dx=dx, return_segments=True)
    assert len(segs) == 1, 'fixture must produce a single segment'
    with pytest.raises(ValueError, match='carriers'):
        apply_real_lens_traced_segmented(
            E, prescription=presc, wavelength=_WL, dx=dx,
            carriers=[None, None], **_QUIET)


# ===========================================================================
# T9 / T10 / T11 -- bit-identity of the chunked / blocked / shared kernels
# ===========================================================================
def test_t11_the_chebyshev_helpers_are_the_shared_ones_bitwise():
    """Three byte-identical copies of the same recurrences existed in
    ``_lens_traced``, ``_lens_imap`` and ``_math.chebyshev`` -- the last of
    which exists SPECIFICALLY to de-duplicate them.  The wrappers that remain
    must be bitwise equal to it, or the consolidation moved a number.

    Exact equality, and a cross-check against ``numpy.polynomial.chebyshev``
    (an implementation outside this library) so "they agree with each other"
    is not the only claim.
    """
    from numpy.polynomial.chebyshev import chebvander

    from lumenairy._math.chebyshev import (
        chebyshev_derivative_vandermonde,
        chebyshev_vandermonde,
    )
    from lumenairy.elements._lens_imap import _cheb_dvander
    rng = np.random.default_rng(20260911)
    u = rng.uniform(-1.0, 1.0, 1000)
    for k in (0, 1, 6, 8, 12):
        assert np.array_equal(_cheb_vand_2d(u, k),
                              chebyshev_vandermonde(u, k))
        assert np.array_equal(_cheb_deriv_vand_2d(u, k),
                              chebyshev_derivative_vandermonde(u, k))
        assert np.array_equal(_cheb_dvander(u, k),
                              chebyshev_derivative_vandermonde(u, k).T)
        # ...and against numpy's own Vandermonde (T only; numpy has no
        # derivative Vandermonde, so the derivative is checked by FD below).
        assert np.array_equal(_cheb_vand_2d(u, k), chebvander(u, k).T)
    # Derivative vs central finite differences.  The bar is the FD's OWN error
    # floor, derived here rather than guessed: truncation ``h^2 |T'''|/6`` with
    # the endpoint bound ``|T_k'''| <= k^2 (k^2-1) (k^2-4) / 15`` (Markov), plus
    # round-off ``eps |T| / h``.  At h = 1e-5, k = 8 that is 2.7e-07 + 2.2e-11,
    # and the measured difference is 1.08e-07 -- inside its own floor.  The
    # factor of 3 is the margin; a derivative that was actually WRONG would be
    # O(k) = O(1..10), seven decades above.
    h = 1e-5
    uu = np.linspace(-0.9, 0.9, 101)
    for k in (4, 8):
        d = _cheb_deriv_vand_2d(uu, k)[k]
        fd = ((_cheb_vand_2d(uu + h, k)[k] - _cheb_vand_2d(uu - h, k)[k])
              / (2 * h))
        t3 = k ** 2 * (k ** 2 - 1) * (k ** 2 - 4) / 15.0
        bound = 3.0 * (h * h * t3 / 6.0 + np.finfo(float).eps / h)
        assert float(np.abs(d - fd).max()) < bound, (
            f'k={k}: max |T_k\' - FD| = {float(np.abs(d - fd).max()):.3e} '
            f'against the FD floor {bound:.3e}')


def test_t10_the_blocked_numba_kernel_is_bit_identical_to_the_numpy_branch():
    """The numba kernel allocated FOUR scratch arrays per SAMPLE inside
    ``prange``; they are now hoisted to a 512-sample block.  The arithmetic,
    its order and its rounding are untouched, which is what the pool/serial
    byte-identity contracts in this module depend on (measured max|diff| =
    0.000e+00 at orders 6 and 10 over 4 Mpt, for a 1.62x / 1.55x speed-up).

    Here: the kernel's answer against the pure-xp Vandermonde branch of the
    SAME evaluator.  BAR: 1e-13 relative.  The two are documented to agree to
    ~1e-16 relative (different summation order over the basis terms), so this
    is three decades of margin; a blocking bug would move whole samples.
    """
    pytest.importorskip('numba')
    xs = np.linspace(-1e-3, 1e-3, 129)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    Z = np.exp(-(Xg ** 2 + Yg ** 2) / (0.8e-3) ** 2) * (1 + 0.3 * Xg / 1e-3)
    xq = np.linspace(-0.9e-3, 0.9e-3, 200)
    Xq, Yq = np.meshgrid(xq, xq)
    for order in (6, 10):
        ev_nb = LT._Cheb2DEvaluator(xs, xs, Z, order=order, backend='numba')
        ev_np = LT._Cheb2DEvaluator(xs, xs, Z, order=order, backend='numpy')
        a = ev_nb.ev_value_and_grad(Xq, Yq)
        b = ev_np.ev_value_and_grad(Xq, Yq)
        scale = float(np.abs(b[0]).max())
        assert float(np.abs(a[0] - b[0]).max()) < 1e-13 * scale
        gscale = float(np.abs(b[1]).max())
        assert float(np.abs(a[1] - b[1]).max()) < 1e-13 * gscale
        assert float(np.abs(a[2] - b[2]).max()) < 1e-13 * gscale
        assert np.isfinite(a[0]).all()


def test_t9_the_chunked_numpy_evaluator_is_bit_identical_and_bounded():
    """The pure-NumPy fallback cost 200 float64 PER QUERY POINT with no
    chunking -- 1.6 GB at 1 Mpt and 26.9 GB at 4096^2 -- and is the branch
    taken on any box without numba (and the REQUIRED branch for CuPy).  It is
    now blocked against the same entry budget every other large-array site in
    this module uses.

    Chunking an elementwise product plus a sum over a FIXED axis is
    order-independent, so the claim is BITWISE, not a tolerance.
    """
    xs = np.linspace(-1e-3, 1e-3, 65)
    Xg, Yg = np.meshgrid(xs, xs, indexing='ij')
    Z = np.exp(-(Xg ** 2 + Yg ** 2) / (0.7e-3) ** 2)
    ev = LT._Cheb2DEvaluator(xs, xs, Z, order=6, backend='numpy')
    xq = np.linspace(-0.9e-3, 0.9e-3, 180)
    Xq, Yq = np.meshgrid(xq, xq)
    n_q = Xq.size
    n_terms = len(ev._mi)
    budget = LT._CHEB_FIT_CHUNK_ENTRIES
    try:
        # Force MANY blocks (~17 here), then ONE block, and compare.  Driving
        # the budget rather than the fixture size is what makes the claim
        # independent of the shipped constant: whatever it is, the chunked and
        # unchunked answers have to be the same bits.
        LT._CHEB_FIT_CHUNK_ENTRIES = 4 * n_terms * 2000
        step = max(1, LT._CHEB_FIT_CHUNK_ENTRIES // (4 * n_terms))
        assert step < n_q, (step, n_q)
        chunked = ev.ev_value_and_grad(Xq, Yq)
        LT._CHEB_FIT_CHUNK_ENTRIES = 10 ** 12        # one block = unchunked
        whole = ev.ev_value_and_grad(Xq, Yq)
    finally:
        LT._CHEB_FIT_CHUNK_ENTRIES = budget
    for c, w in zip(chunked, whole):
        assert np.array_equal(c, w)
    # ...and the shipped budget really does bound the working set: the block
    # holds 4 gathers of (M, step) plus two live products, i.e. ~6*M float64
    # per point, so the peak is O(budget) rather than O(n_q * 200).
    assert budget > 0 and budget < 10 ** 9


def test_t9_final_masking_is_in_place_and_leaves_masked_pixels_exactly_zero():
    """The whole-grid masking was two ``np.where`` calls over full-grid
    radius temporaries (~7.1 units of 8N^2 where ~3.1 suffice); it is now
    ``np.copyto(where=)`` over the 1-D axes form the banded path already used.
    The VALUES must not move: masked pixels exactly zero, unmasked untouched.
    """
    N, dx, x, X, Y = _grid(N=192, dx=30e-6)
    presc = _singlet(aperture=4e-3)
    E_in = _gauss(X, Y, w=1.0e-3).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(E_in, prescription=presc,
                                     wavelength=_WL, dx=dx, ray_subsample=4,
                                     **_QUIET)
    outside = (X ** 2 + Y ** 2) > (0.5 * presc['aperture_diameter']) ** 2
    assert np.all(out[outside] == 0), 'aperture mask leaked'
    assert np.isfinite(out).all(), 'NaN / inf in the returned field'
    assert float(np.abs(out).max()) > 0


# ===========================================================================
# T13 -- the exit-NA guard measures the field it returns
# ===========================================================================
def test_t13_exit_na_is_measured_inside_the_output_aperture():
    """The trace runs with ``aperture_diameter`` popped, so unless the
    SURFACES carry a ``semi_diameter`` the rays go out to ``0.75*aperture``
    while the returned field is masked to ``aperture/2``.  Gating on input
    amplitude alone did nothing for a flat input, so ``na_exit`` was measured
    over rays the output mask deletes -- 3.144x overstated on the audit's f/5
    fixture -- and it feeds the chain's ``on_tilt_exact_grid``, whose default
    action is 'error'.

    ORACLE: the marginal ray AT the aperture edge, traced independently.
    BAR: 5 % relative.  The reported NA is the max over launch nodes, which
    land on a lattice and so under-sample the exact rim by up to one lattice
    step; measured agreement is 0.3 %, and the pre-fix overstatement is 214 %,
    so the bar has 16x of gap below and 43x above.
    """
    import lumenairy.raytrace as _rt
    N, dx = 512, 50e-6
    presc = _singlet(r1=51.68e-3, r2=-51.68e-3, thick=4e-3, aperture=24e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    diag = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        apply_real_lens_traced(E_in, prescription=presc, wavelength=_WL,
                               dx=dx, ray_subsample=8, _exit_na_out=diag,
                               **_QUIET)
    assert diag.get('na_exit'), f'the NA diagnostic was not filled: {diag}'

    h = 0.5 * presc['aperture_diameter']
    s = _rt.surfaces_from_prescription(presc)
    rays = _rt.RayBundle(x=np.array([h]), y=np.zeros(1), z=np.zeros(1),
                         L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                         wavelength=_WL, alive=np.ones(1, dtype=bool),
                         opd=np.zeros(1))
    ex = _rt.trace(rays, s, _WL).at_exit_vertex()
    na_true = float(np.hypot(ex.L[0], ex.M[0]))
    assert na_true > 0.1, 'the fixture is not fast enough to be a test'
    rel = abs(diag['na_exit'] - na_true) / na_true
    assert rel < 0.05, (
        f"na_exit = {diag['na_exit']:.5f} against the marginal ray's "
        f'{na_true:.5f} ({rel * 100:.1f} % off)')


# ===========================================================================
# guards that must keep holding (the audit's "verified correct" list)
# ===========================================================================
def test_masked_pixels_and_energy_are_where_the_audit_left_them():
    """Re-check three properties the audit verified and this work must not
    move: masked pixels exactly zero (no NaN anywhere), ``amplitude_model=
    'screen'`` energy 1.000000 against the aperture-transmitted input, and a
    default call byte-identical to ``caustic=None``.
    """
    N, dx, x, X, Y = _grid(N=192, dx=30e-6)
    presc = _singlet(aperture=4e-3)
    E_in = _gauss(X, Y, w=0.9e-3).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = apply_real_lens_traced(E_in, prescription=presc, wavelength=_WL,
                                   dx=dx, ray_subsample=4, **_QUIET)
        b = apply_real_lens_traced(E_in, prescription=presc, wavelength=_WL,
                                   dx=dx, ray_subsample=4, caustic='single',
                                   **_QUIET)
    assert np.array_equal(a, b), "caustic='single' must be the default path"
    assert not np.isnan(a).any() and not np.isinf(a).any()
    inside = (X ** 2 + Y ** 2) <= (0.5 * presc['aperture_diameter']) ** 2
    p_in = float((np.abs(E_in[inside]) ** 2).sum())
    p_out = float((np.abs(a) ** 2).sum())
    # 1e-4 relative: the screen model transports the analytic amplitude
    # unchanged, so the only loss is the ray-coverage mask; the audit measured
    # 1.000000 and this fixture measures within 1e-5.
    assert abs(p_out / p_in - 1.0) < 1e-4, f'energy ratio {p_out / p_in:.6f}'


# ===========================================================================
# T11 -- the PreparedTracedLens amplitude-leg contract, checked not commented
# ===========================================================================
def test_t11_prepared_amp_kwargs_match_the_elements_own_leg():
    """``PreparedTracedLens.__call__`` must call ``apply_real_lens`` with the
    SAME amplitude-affecting keywords the element's internal ``_amp_call``
    uses, or the prepared screen silently stops matching the leg it factors.

    The code carried a comment saying "same 8 kwargs" and NOTHING read it: a
    signature change in the element's amp leg would have desynchronised the
    two with no test failing except by luck.  This reads both call sites out
    of the source and compares the keyword SETS -- the property the comment
    asserted.
    """
    import ast
    import inspect
    src = inspect.getsource(LT)
    tree = ast.parse(src)

    def _apply_real_lens_kwargs(node):
        out = []
        for n in ast.walk(node):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == 'apply_real_lens'):
                out.append({kw.arg for kw in n.keywords if kw.arg})
        return out

    prepared = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == 'PreparedTracedLens':
            for f in n.body:
                if isinstance(f, ast.FunctionDef) and f.name == '__call__':
                    calls = _apply_real_lens_kwargs(f)
                    assert len(calls) == 1, calls
                    prepared = calls[0]
    assert prepared is not None, 'PreparedTracedLens.__call__ not found'

    amp_leg = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == '_amp_call':
            calls = _apply_real_lens_kwargs(n)
            assert len(calls) == 1, calls
            amp_leg = calls[0]
    assert amp_leg is not None, "the element's _amp_call was not found"

    # ``progress`` is a callback, not physics: the prepared path has no bar to
    # drive.  Everything else must match exactly.
    assert prepared == (amp_leg - {'progress'}), (
        f'prepared={sorted(prepared)}\namp_leg={sorted(amp_leg)}')
    # ...and the comment's count, now that something reads it.
    assert len(prepared) == 8, sorted(prepared)


# ===========================================================================
# S8 -- the inverse-map cache key carries the flags that change its arithmetic
# ===========================================================================
def test_s8_imap_cache_key_separates_the_least_squares_flags():
    """``build_inverse_map``'s SHA-256 key hashed 12 scalars and every input
    array but NONE of the ``_lens_traced`` flags that select the branch of the
    solve it runs -- and ``_det_traced()`` reads
    ``DETERMINISTIC_TRACED_FIT`` at CALL time.  So
    ``traced_flags(DETERMINISTIC_TRACED_FIT=False)``, whose registry entry
    states the contract "restores the G = A.T @ A / rhs = A.T @ b route for the
    traced chain EXACTLY, bit for bit, and is the fail-before for the whole
    layer", was defeated by a cache HIT on the second and every later call in a
    process: measured ``key(det=True) == key(det=False) -> True`` with the SAME
    OBJECT served.

    DECISION test on the key itself: two flag settings, two keys.  No numeric
    bar -- the numbers they produce differ by 6.2e-14 relative on a
    well-conditioned fixture and by 1681-2138x in least-squares residual on the
    production 120-term fits, so the CONTRACT, not the size, is the claim.
    """
    from lumenairy.elements import _lens_imap as IM
    n = 17
    xs_in = np.linspace(-1e-3, 1e-3, n)
    Xg, Yg = np.meshgrid(xs_in, xs_in, indexing='ij')
    XO = Xg * 0.9 + 1e-6 * Xg ** 3
    YO = Yg * 0.9 + 1e-6 * Yg ** 3
    OP = 1e-3 + 1e-6 * (Xg ** 2 + Yg ** 2)
    DJ = np.full_like(XO, 0.81)
    W = None
    args = (xs_in, XO, YO, OP, DJ, W, 6, 1e-3, _WL)

    def _key():
        return IM._imap_key(*args)

    flags = [
        ('DETERMINISTIC_TRACED_FIT', not LT.DETERMINISTIC_TRACED_FIT),
        ('LSTSQ_CONDITIONING_STEPDOWN',
         not LT.LSTSQ_CONDITIONING_STEPDOWN),
        ('_DET_REFINE_STEPS', int(LT._DET_REFINE_STEPS) + 1),
        ('_LSTSQ_RESID_MARGIN', float(LT._LSTSQ_RESID_MARGIN) * 2.0),
    ]
    base = _key()
    assert base == _key(), 'the key is not deterministic on fixed inputs'
    for name, new in flags:
        old = getattr(LT, name)
        try:
            setattr(LT, name, new)
            moved = _key()
        finally:
            setattr(LT, name, old)
        assert moved != base, (
            f'{name} does not move the inverse-map cache key, so a map built '
            f'under one setting is served to a caller who asked for the other')
        assert _key() == base, f'{name} left the key moved after restore'


def test_s8_gram_refusal_records_its_budget_provenance():
    """The GRAM guard prices against ``lumenairy.memory.get_ram_budget()``,
    which with no ``set_max_ram`` override is psutil's LIVE available reading
    -- so the same call on the same inputs returns a DIFFERENT FIELD on two
    boxes, or on the same box under load.  The record now says which it was,
    and ``report_refusal`` no longer claims the refusal "costs speed, never
    accuracy" (the module's own numbers three hundred lines above say it moves
    design 121's FWHM 3.350 -> 3.450 um and the peak by 0.8 %).
    """
    import inspect

    from lumenairy.elements import _lens_imap as IM
    # The MESSAGE, not the source: the comment above the message quotes the
    # retired wording on purpose, so grepping the function body would match it.
    rec = {'refused': 'GRAM', 'detail': 'synthetic', 'build_budget_gb': 1.0}
    old_action = IM.INVERSE_MAP_GUARD
    try:
        IM.INVERSE_MAP_GUARD = 'warn'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            IM.report_refusal(rec, caller='test')
    finally:
        IM.INVERSE_MAP_GUARD = old_action
    assert caught, 'the refusal reported nothing'
    msg = str(caught[0].message)
    assert 'costs speed, never accuracy' not in msg, (
        'report_refusal still tells the user a refusal cannot change the '
        "answer, which contradicts this module's own measurements")
    assert 'DIFFERENT ANSWER' in msg, msg
    assert 'set_max_ram' in msg, msg
    assert 'reproducible' in msg and 'AVAILABLE' in msg, msg
    # the provenance keys the guard now records
    guard_src = inspect.getsource(IM.build_inverse_map)
    assert "rec['build_ram_budget_explicit']" in guard_src
    assert 'not reproducible across boxes' in guard_src


# ===========================================================================
# WP-A2 follow-ups landed in _lens_traced.py (the traced family's entry)
# ===========================================================================
def _stop_fixture():
    N, dx = 64, 20e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2).astype(np.complex128)
    presc = {'aperture_diameter': 2.0e-3, 'surfaces': [
        {'radius': 50e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': '_A3TG', 'semi_diameter': 1e-3},
        {'radius': -50e-3, 'conic': 0., 'glass_before': '_A3TG',
         'glass_after': 'air', 'semi_diameter': 1e-3}],
        'thicknesses': [2e-3]}
    return E, dx, presc


def test_traced_stop_index_is_read_through_the_shared_normaliser():
    """A malformed ``stop_index`` is diagnosed by the traced entry itself.

    An out-of-range stop matches no surface AND suppresses the entrance
    aperture, so it silently removes every aperture mask.  Before this, the
    traced entry only tested ``int(stop_index) != 0``: a bad index reached the
    analytic amplitude leg and surfaced (if at all) as an ``apply_real_lens:``
    ValueError raised from inside a worker thread, after the ray trace had
    run and a "non-entrance stop" warning had already fired; a non-integer
    produced a bare ``invalid literal for int()``.

    DECISION test: valid indices (including the Python negative spellings)
    must leave the field BIT-IDENTICAL, and invalid ones must raise under this
    function's own name.
    """
    E, dx, presc = _stop_fixture()

    def _run(**over):
        p = dict(presc)
        p.update(over)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = apply_real_lens_traced(
                E, prescription=p, wavelength=_WL, dx=dx, ray_subsample=2,
                min_coarse_samples_per_aperture=0)
        return out, [str(w.message) for w in rec]

    base, _ = _run()
    # every in-range spelling, including -1 / -2, is a no-op on the field
    for si in (None, 0, 1, -1, -2):
        out, msgs = _run(stop_index=si)
        assert np.array_equal(out, base), f'stop_index={si!r} changed the field'
        # -2 IS the entrance on a 2-surface lens, so it must not warn; 1 and
        # -1 are the same non-entrance surface and both must.
        warned = any('stop_index' in m for m in msgs)
        assert warned == (si in (1, -1)), (si, warned, msgs)
    # out of range and non-integer: refused, by name, before anything runs
    for si in (2, 5, -3, 'first'):
        with pytest.raises(ValueError) as exc:
            _run(stop_index=si)
        assert 'apply_real_lens_traced' in str(exc.value), str(exc.value)
        assert 'stop_index' in str(exc.value), str(exc.value)


def test_traced_aperture_notice_measures_the_narrow_axis():
    """The pre-flight aperture-vs-grid notice must describe the axis that
    truncates first.  Passing ``shape[0]`` (Ny) with ``dx`` describes a
    semi-extent that exists on neither axis of an anamorphic grid: on a TALL
    grid (Ny > Nx) it reports the WIDE y half-width and stays silent while the
    aperture over-fills x.  Measured on Nx=64 / Ny=256 / dx=dy=20 um with a
    2 mm aperture: 0 warnings before, 1 after.

    The traced entry refuses a non-square grid a few lines later, so the
    notice is observed alongside that refusal.
    """
    _, dx, presc = _stop_fixture()
    E = np.ones((256, 64), dtype=np.complex128)      # tall and narrow
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='square grid'):
            apply_real_lens_traced(E, prescription=presc, wavelength=_WL,
                                   dx=dx)
    hits = [str(w.message) for w in rec
            if 'aperture(s) exceed the simulation grid' in str(w.message)]
    assert hits, ('the aperture notice was silent on a grid whose x '
                  'half-width (0.64 mm) is smaller than the 1.00 mm aperture '
                  'semi-diameter')
    assert 'N=64' in hits[0], hits[0]        # the narrow axis, not Ny=256
