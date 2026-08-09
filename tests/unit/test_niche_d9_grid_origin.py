"""niche D9 -- ``apply_real_lens_traced(origin=(x0, y0))``: a CHIEF-RAY-CENTRED
wave grid, and the exact final leg that stops paying for the optical axis.

**The problem.**  A tilted congruence's beam sits at its chief ray.  Until D9
``apply_real_lens_traced`` built its grid symmetrically about the OPTICAL AXIS,
so :func:`~lumenairy.propagators.carrier._fine_trace_group_exit` had to size ONE
grid that spans both -- ``2*max(|x_c|,|y_c|) + window_factor*w`` where the beam
alone needs ``window_factor*w``.  Measured on design 121's order (-4,-2): chief
ray 3.02 mm off axis against an entrance beam radius 3.12 mm, so the window grew
12.50 -> 18.54 mm (1.48x linear, 2.2x memory), forcing ``n_fine`` 16384 (17.2 GB
at four complex128 work arrays) where ~12288 would have done, AND forcing the
chain grid to satisfy ``N*cur_dx >= 18.54 mm`` at the last group -- which is what
drove N=8192 and its ~24 GB working set.  Both terms are the axis, not the beam.

**The fix.**  ``origin=(x0, y0)`` states that the grid's CENTRE PIXEL sits at
that point in the element's own (optical-axis) frame: index ``(i, j)`` is the
physical point ``(x0 + (j - N/2) dx, y0 + (i - N/2) dy)``.  The element, its
aperture and the ray launch do not move.

**What this file pins.**

1.  ``origin=(0, 0)`` is BYTE-IDENTICAL to omitting it.  Non-negotiable: it is
    the whole safety property of a change that touches ~20 coordinate sites in a
    4000-line function.

2.  THE EQUIVALENCE ORACLE.  The same physical beam, traced twice: once on a
    large AXIS-centred grid at ``origin=(0, 0)``, once on a grid a quarter the
    area centred on the chief ray.  Over the physical region they share the two
    exit fields must agree.  **Measured: 2.7e-9 max and 2.9e-10 rms of the peak
    amplitude** (512 vs 256 at dx = 4 um, chief ray (0.512, 0.304) mm, beam
    ``w`` = 0.10 mm, i.e. a 5.1-``w`` half-window).  That is the test that
    proves no coordinate site was missed: a site left un-shifted moves a
    physical quantity by ``|origin|`` = 0.6 mm = 150 pixels, which no tolerance
    this side of "completely different field" absorbs.

    The residual is NOT numerical noise, it is the small grid's own INPUT
    truncation, and it is identified by scaling: at half-windows of 2.6 / 3.7 /
    5.1 / 6.4 beam radii the full-window agreement is 1.7e-3 / 1.9e-6 / 2.7e-9 /
    5.9e-10.  The outliers at 2.6 w are pixels whose traced ray comes from an
    ENTRANCE height the smaller grid does not represent, so
    ``map_coordinates(..., cval=0)`` correctly gives them no amplitude while the
    larger grid still carries the beam skirt there.  That is a property of the
    window, not of the origin plumbing.

3.  REFUSAL outside the validated carrier regime, because the analytic
    ``apply_real_lens`` amplitude leg has no origin of its own.

4.  The tilted ``_fine_trace_group_exit`` window is now ``window_factor * w``.
"""

import warnings

import numpy as np
import pytest

from lumenairy import get_glass_index
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import (
    TiltedCarrier,
    _tilted_carrier_parts,
)
from lumenairy.propagators.carrier import (
    _envelope_amp_radius,
    _fine_trace_group_exit,
)

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_GLASS = 'N-BK7'
_F = 3.0e-3
_APER = 3.0e-3
_THICK = 1.5e-3
_DX = 4.0e-6
_SUB = 4

# The chief ray, in whole ``ray_subsample * dx`` steps.  Both quantisations are
# deliberate and are properties of the ORACLE, not of the feature:
#   * a whole-pixel offset makes the two grids sample the SAME physical lattice,
#     so the comparison needs no interpolation;
#   * a whole-COARSE-cell offset makes the two ``X[::sub, ::sub]`` Newton
#     lattices coincide as well.  Break that and the two runs interpolate the
#     ray-density amplitude from lattices offset by one fine pixel, which on
#     this fixture reads 1.7e-3 -- a real O(h^2) upsample difference, not a
#     missed site (verified by sweeping the offset).
_XC, _YC = 128 * _DX, 76 * _DX
_RCAR = 40.0e-3                      # diverging carrier at the entrance
_TL, _TM = 0.012, -0.007             # a genuinely 2-D tilt: L != M, x0 != y0
_CAR = TiltedCarrier(_RCAR, _TL, _TM, _XC, _YC)


def _surface(radius, glass_before, glass_after, conic, clear_aperture=None):
    s = {'radius': float(radius), 'glass_before': glass_before,
         'glass_after': glass_after, 'conic': float(conic),
         'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    if clear_aperture is not None:
        s['clear_aperture'] = float(clear_aperture)
    return s


def _prescription(aperture=_APER, exit_clear_aperture=None):
    """Flat entrance + ``K = -n^2`` conic exit -- the exact Fermat singlet."""
    n = float(get_glass_index(_GLASS, _WL))
    return {'name': 'D9 singlet', 'aperture_diameter': aperture,
            'thicknesses': [_THICK],
            'surfaces': [_surface(np.inf, 'air', _GLASS, 0.0),
                         _surface(-(n - 1.0) * _F, _GLASS, 'air', -n * n,
                                  exit_clear_aperture)]}


def _field(n, origin, w, resid=True):
    """The SAME physical field, sampled on a grid whose centre pixel is at
    ``origin``: a Gaussian about the chief ray times the exact tilted-carrier
    eikonal, plus (optionally) a small smooth residual eikonal so the C6
    stationary-phase launch has something non-trivial to fit."""
    ax = (np.arange(n) - n / 2) * _DX + origin[0]
    ay = (np.arange(n) - n / 2) * _DX + origin[1]
    X, Y = np.meshgrid(ax, ay)
    u, v = X - _CAR.x0, Y - _CAR.y0
    W, _, _ = _tilted_carrier_parts(_CAR, X, Y)
    ph = _K0 * W
    if resid:
        ph = ph + _K0 * 2.0e-8 * ((u / w) ** 3 + 0.6 * (u / w) * (v / w) ** 2)
    return (np.exp(-(u ** 2 + v ** 2) / (w * w))
            * np.exp(1j * ph)).astype(np.complex128)


def _carrier_kw(**extra):
    """The VALIDATED carrier-regime configuration -- the only one ``origin``
    supports, and the one ``propagate_traced_carrier_chain`` uses."""
    kw = dict(prescription=_prescription(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1,
              amplitude_model='ray_density', preserve_input_phase='remap',
              carrier=_CAR, on_undersample='silent',
              on_aperture_beam='silent', on_noncollimated='off')
    kw.update(extra)
    return kw


def _run(n, origin, w, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return apply_real_lens_traced(_field(n, origin, w), origin=origin,
                                      **_carrier_kw(**extra))


# ===========================================================================
# 1.  BYTE-IDENTITY -- origin=(0, 0) is the historical path, bit for bit.
# ===========================================================================
@pytest.mark.parametrize('n,extra', [
    (192, {}),
    (192, {'remap_sampling': 'full'}),
    (192, {'fit_radius_beam_factor': 2.0}),
    (192, {'ray_subsample': 2}),
    (160, {'newton_fit': 'spline'}),
])
def test_origin_zero_is_byte_identical(n, extra):
    """Passing ``origin=(0.0, 0.0)`` must return the SAME BITS as omitting it,
    across the sampling / fit-domain / inversion-backend knobs that reach the
    coordinate sites the origin threads through."""
    E = _field(n, (0.0, 0.0), 0.10e-3)
    kw = _carrier_kw(**extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        without = apply_real_lens_traced(E, **kw)
        with_zero = apply_real_lens_traced(E, origin=(0.0, 0.0), **kw)
    assert np.array_equal(without, with_zero), (
        'origin=(0,0) is not byte-identical: max |dE| = '
        f'{float(np.abs(without - with_zero).max()):.3e}')


def test_origin_zero_is_byte_identical_on_the_screen_amplitude_path():
    """The DEFAULT (``amplitude_model='screen'``, ``preserve_input_phase=True``)
    path is the one every existing caller uses, and it is the one the origin
    REFUSES -- so the only thing to pin is that adding the parameter did not
    perturb it."""
    E = _field(192, (0.0, 0.0), 0.10e-3)
    kw = dict(prescription=_prescription(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, carrier=_CAR,
              on_undersample='silent', on_aperture_beam='silent',
              on_noncollimated='off')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        without = apply_real_lens_traced(E, **kw)
        with_zero = apply_real_lens_traced(E, origin=(0.0, 0.0), **kw)
    assert np.array_equal(without, with_zero)


# ===========================================================================
# 2.  THE EQUIVALENCE ORACLE.
# ===========================================================================
@pytest.mark.parametrize('w,sampling,tol_max,tol_rms', [
    (0.10e-3, 'lattice', 1.0e-7, 1.0e-8),   # measured 2.7e-9 / 2.9e-10
    (0.08e-3, 'lattice', 1.0e-7, 1.0e-8),   # measured 5.9e-10 / 5.3e-11
    # ``remap_sampling='full'`` is the CHAIN's setting and it routes the
    # transported residual through TWO further entrance-coordinate -> pixel
    # conversions (the sub>1 upsampled pullback and the sub==1 one) that
    # 'lattice' never touches.  Byte-identity alone cannot see them, because at
    # origin=(0,0) a missing shift is a no-op; only the oracle can.
    (0.10e-3, 'full', 1.0e-7, 1.0e-8),
])
def test_chief_ray_centred_grid_reproduces_the_axis_centred_one(
        w, sampling, tol_max, tol_rms):
    """A LARGE axis-centred grid at ``origin=(0, 0)`` and a grid a QUARTER of
    its area centred on the chief ray must return the same exit field over the
    physical region they share.

    Measured on this fixture (``w`` = 0.10 mm): **max |dE| = 2.7e-9 and
    rms |dE| = 2.9e-10 of the peak amplitude**, over all 256x256 shared pixels.
    The tolerances are set two orders above that so the test pins the CONCLUSION
    ("every coordinate site moved together") rather than the last digit of a
    least-squares fit.

    A single un-shifted site displaces a physical quantity by |origin| =
    0.60 mm = 150 pixels -- the resulting field is not close in any norm.
    """
    n_a, n_b = 512, 256
    Ea = _run(n_a, (0.0, 0.0), w, remap_sampling=sampling)
    Eb = _run(n_b, (_XC, _YC), w, remap_sampling=sampling)
    # grid-b index j  <->  grid-a index j + off:  (j - n_b/2) dx + XC
    #                                          == (j + off - n_a/2) dx
    off_x = int(round(_XC / _DX)) + n_a // 2 - n_b // 2
    off_y = int(round(_YC / _DX)) + n_a // 2 - n_b // 2
    shared = Ea[off_y:off_y + n_b, off_x:off_x + n_b]

    peak = float(np.abs(shared).max())
    assert peak > 0.1, 'fixture produced no beam'
    err = np.abs(shared - Eb)
    e_max = float(err.max()) / peak
    e_rms = float(np.sqrt((err ** 2).mean())) / peak
    assert e_max < tol_max, f'max |dE|/peak = {e_max:.3e}'
    assert e_rms < tol_rms, f'rms |dE|/peak = {e_rms:.3e}'
    # and the beam has not merely been reproduced -- it is where it should be
    p_a = float((np.abs(shared) ** 2).sum())
    p_b = float((np.abs(Eb) ** 2).sum())
    assert abs(p_b / p_a - 1.0) < 1e-6, (p_a, p_b)


def test_the_oracle_would_catch_a_missed_site():
    """Negative control: the tolerance above is not vacuous.  Feed the
    chief-ray-centred call an input sampled as if the grid were STILL
    axis-centred -- the single mistake a missed coordinate site makes -- and the
    comparison must fail by orders of magnitude."""
    w, n_a, n_b = 0.10e-3, 512, 256
    Ea = _run(n_a, (0.0, 0.0), w)
    E_wrong = _field(n_b, (0.0, 0.0), w)       # <- the frame mix-up
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eb = apply_real_lens_traced(E_wrong, origin=(_XC, _YC),
                                    **_carrier_kw())
    off_x = int(round(_XC / _DX)) + n_a // 2 - n_b // 2
    off_y = int(round(_YC / _DX)) + n_a // 2 - n_b // 2
    shared = Ea[off_y:off_y + n_b, off_x:off_x + n_b]
    peak = float(np.abs(shared).max())
    e_max = float(np.abs(shared - Eb).max()) / peak
    assert e_max > 1e-2, (
        f'the control did not diverge (max |dE|/peak = {e_max:.3e}); the '
        f'oracle above is not measuring what it claims to')


# ===========================================================================
# 3.  REFUSAL -- what cannot be proved is not offered.
# ===========================================================================
def test_origin_refuses_the_screen_amplitude_model():
    """``amplitude_model='screen'`` returns the ANALYTIC envelope as its
    magnitude, and that leg builds the element about the grid centre.  Under a
    decentred origin it would return an on-axis element's diffraction pattern
    with no symptom, so it is refused."""
    E = _field(160, (_XC, _YC), 0.10e-3)
    with pytest.raises(NotImplementedError) as e:
        apply_real_lens_traced(
            E, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, carrier=_CAR,
            amplitude_model='screen', origin=(_XC, _YC))
    msg = str(e.value)
    for token in ('ray_density', 'remap', 'apply_real_lens', 'origin=(0, 0)'):
        assert token in msg, f'refusal is missing {token!r}:\n{msg}'


def test_origin_refuses_preserve_input_phase_true():
    """Same reason, the other half: with ``preserve_input_phase=True`` the exit
    field is ``E_analytic * exp(i delta)``, so the analytic leg's ENVELOPE (not
    merely its zero set) survives into the answer."""
    E = _field(160, (_XC, _YC), 0.10e-3)
    with pytest.raises(NotImplementedError) as e:
        apply_real_lens_traced(
            E, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, carrier=_CAR,
            amplitude_model='ray_density', preserve_input_phase=True,
            origin=(_XC, _YC))
    assert 'ray_density' in str(e.value) and 'remap' in str(e.value)


def test_origin_refuses_the_multibranch_caustic_route():
    """``caustic='multibranch'`` hands the whole call to a different function
    that builds its own axis-centred grids."""
    E = _field(160, (_XC, _YC), 0.10e-3)
    with pytest.raises(NotImplementedError, match='caustic'):
        apply_real_lens_traced(
            E, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, carrier='auto',
            amplitude_model='ray_density', preserve_input_phase='remap',
            caustic='multibranch', origin=(_XC, _YC))


def test_origin_refuses_the_noncollimated_delegate_fallback():
    """``on_noncollimated='delegate'`` returns ``apply_real_lens(E_in)``
    directly -- a model with no origin at all."""
    E = _field(160, (_XC, _YC), 0.10e-3)
    with pytest.raises(NotImplementedError, match='delegate'):
        apply_real_lens_traced(
            E, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, carrier=_CAR,
            amplitude_model='ray_density', preserve_input_phase='remap',
            on_noncollimated='delegate', origin=(_XC, _YC))


@pytest.mark.parametrize('bad', [(np.nan, 0.0), (0.0, np.inf), 3.0,
                                 (1.0, 2.0, 3.0), 'x'])
def test_origin_is_validated(bad):
    E = _field(64, (0.0, 0.0), 0.10e-3)
    with pytest.raises(ValueError, match='origin'):
        apply_real_lens_traced(
            E, prescription=_prescription(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, origin=bad)


def test_the_ordinary_prescription_has_an_empty_analytic_zero_set():
    """The premise of the whole ``origin`` restriction, measured rather than
    argued: for a prescription whose only mask is an ENTRANCE
    ``aperture_diameter``, the analytic amplitude leg has NO exact zeros at
    all -- the ASM through glass fills the shadow back in -- so under
    ``ray_density`` + ``remap`` it contributes literally nothing to the
    returned field and the decentred origin has nothing to couple to.

    Measured here: ``min |apply_real_lens(E_in)|`` over the whole grid.  If a
    future change makes the analytic leg produce exit-plane zeros for a plain
    prescription, the guard below stops being a formality and this test is
    where that shows up first."""
    from lumenairy.elements import apply_real_lens
    E = _field(192, (_XC, _YC), 0.10e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A = np.abs(apply_real_lens(E, prescription=_prescription(),
                                   wavelength=_WL, dx=_DX))
    assert int((A == 0.0).sum()) == 0, (
        f'{int((A == 0.0).sum())} exact zeros; min |A| = {float(A.min()):.3e}')


def test_the_analytic_amplitude_support_check_refuses_a_clipped_beam():
    """The one coupling the origin cannot remove, and the reason the feature is
    restricted to ``ray_density`` + ``remap``: the analytic amplitude leg has no
    origin, so it places the element's masks about the GRID centre.  It reaches
    the answer only through its ZERO SET (the ray-density swap divides its
    modulus out), which makes the coupling MEASURABLE -- and it is measured, and
    refused.

    Triggered here with a ``clear_aperture`` on the LAST surface, which is the
    configuration that actually produces exact zeros: that mask lands on the
    exit plane with no propagation after it to fill the shadow in.  The ray leg
    applies the same stop in ABSOLUTE coordinates and the analytic leg applies
    it about the decentred grid centre, so the two disagree over a crescent --
    measured deletions on this fixture: 1.3e-04 % of the exit power at
    ``clear_aperture`` = 1.20 mm, 0.134 % at 0.90 mm, 0.583 % at 0.80 mm.

    The chief ray is nearer the axis here than in the oracle above, on purpose:
    the stop has to pass the beam in the ABSOLUTE frame (or there is no exit
    power for the analytic mask to delete) while still cutting it in the GRID
    frame.  That is the whole geometry of the defect."""
    n, w = 256, 0.25e-3
    cx, cy = 0.20e-3, 0.12e-3
    car = TiltedCarrier(_RCAR, 0.010, -0.006, cx, cy)
    ax = (np.arange(n) - n / 2) * _DX + cx
    ay = (np.arange(n) - n / 2) * _DX + cy
    X, Y = np.meshgrid(ax, ay)
    W, _, _ = _tilted_carrier_parts(car, X, Y)
    E = (np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (w * w))
         * np.exp(1j * _K0 * W)).astype(np.complex128)
    with pytest.raises(NotImplementedError) as e:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_traced(
                E, origin=(cx, cy),
                **_carrier_kw(carrier=car, prescription=_prescription(
                    aperture=2.0e-3, exit_clear_aperture=0.90e-3)))
    msg = str(e.value)
    for token in ('ORIGIN_AMP_SUPPORT_CHECK', 'zero set', 'apply_real_lens'):
        assert token in msg, f'refusal is missing {token!r}:\n{msg}'


# ===========================================================================
# 4.  The caller: the tilted retrace window is the ON-AXIS window.
# ===========================================================================
def _gaussian_env(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def test_tilted_fine_trace_window_is_the_on_axis_window():
    """``_fine_trace_group_exit`` used to size the tilted retrace to
    ``2*max(|x_c|,|y_c|) + window_factor*w`` so ONE axis-centred grid held both
    the optical axis and the displaced beam.  With the grid centred on the chief
    ray the first term is gone, and what is left is exactly what an ON-AXIS beam
    of the same size costs -- however far off axis it sits.

    The returned field spans ``n_fine * dx_fine``, which is the window itself
    (``dx_fine = win / n_fine`` by construction), so it is read directly."""
    n, cur_dx, w_beam = 256, 10.0e-6, 0.20e-3
    x_c, y_c = 0.50e-3, 0.30e-3
    wf = 4.0
    presc = _prescription(aperture=2.0e-3)
    env = _gaussian_env(n, cur_dx, w_beam)
    w = _envelope_amp_radius(env, cur_dx, cur_dx)
    r_out = -1.2e-3
    na_exit = w / abs(r_out)
    call_kw = dict(amplitude_model='ray_density',
                   preserve_input_phase='remap', parallel_amp=False,
                   on_undersample='silent', on_noncollimated='off',
                   on_aperture_beam='silent')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E, dx_fine = _fine_trace_group_exit(
            env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, r_out, na_exit,
            window_factor=wf, n_fine_cap=4096,
            centre=(x_c, y_c), tilt=(0.010, 0.005),
            on_tilt_exact_grid='ignore')

    win = float(np.shape(E)[-1]) * dx_fine
    want_new = wf * w
    want_old = 2.0 * max(abs(x_c), abs(y_c)) + wf * w
    # the crop quantises to an even number of co-moving pixels
    assert abs(win - want_new) <= 2.0 * cur_dx, (
        f'window {win * 1e3:.4f} mm is not the on-axis window '
        f'{want_new * 1e3:.4f} mm')
    assert win < want_old - 2.0 * cur_dx, (
        f'window {win * 1e3:.4f} mm still carries the D6 axis term '
        f'({want_old * 1e3:.4f} mm)')


def test_the_tilted_window_no_longer_grows_with_the_chief_ray_offset():
    """The property that actually buys the memory: sweep the chief-ray offset
    and the window must not move.  Pre-D9 it grew as ``2*max(|x_c|, |y_c|)``."""
    n, cur_dx, w_beam = 192, 10.0e-6, 0.20e-3
    presc = _prescription(aperture=2.0e-3)
    env = _gaussian_env(n, cur_dx, w_beam)
    r_out = -1.2e-3
    na_exit = _envelope_amp_radius(env, cur_dx, cur_dx) / abs(r_out)
    call_kw = dict(amplitude_model='ray_density',
                   preserve_input_phase='remap', parallel_amp=False,
                   on_undersample='silent', on_noncollimated='off',
                   on_aperture_beam='silent')
    wins = []
    for x_c in (0.10e-3, 0.40e-3, 0.70e-3):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E, dxf = _fine_trace_group_exit(
                env, np.inf, cur_dx, presc, _WL, 4, 1, call_kw, r_out,
                na_exit, window_factor=4.0, n_fine_cap=4096,
                centre=(x_c, 0.0), tilt=(0.008, 0.0),
                on_tilt_exact_grid='ignore')
        wins.append(float(np.shape(E)[-1]) * dxf)
    assert max(wins) - min(wins) < 1e-12, (
        f'the retrace window still tracks the chief-ray offset: {wins}')
