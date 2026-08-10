"""Niche C1 -- consolidation of the five open findings the design-121 (D1-D7)
adversarial verifiers left behind.

Roadmap ``docs/audits/ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md``.
Everything here is SELF-CONTAINED: synthetic N-BK7 singlets and in-memory
``.zmx`` text built inline, no prescription asset, and every physical claim is
scored against an oracle that shares no code with the thing under test (an
exact skew ray trace from :mod:`lumenairy.raytrace`, a local bicubic-spline
map, or the loader's own pre-D4 window rule re-derived by hand).

The five items, and what each pins:

1. **A null decentre flipped the whole ray fit.**  ``_beam_decentred`` was
   ``bool(_bcx or _bcy)``, so ANY nonzero beam centre -- including a
   sub-pixel or numerically tiny one -- swapped the historical concentric HARD
   MASK for D1's off-centre WEIGHTED solve and D7's raised fit order.
   Measured, that moved the returned field by 8.3e-6 of peak at 1e-9 PIXELS of
   decentre.  Now gated on ``max(_DECENTRE_GATE_PIXELS * dx,
   _DECENTRE_GATE_W_FRAC * w)``; below the gate the concentric path is
   BYTE-IDENTICAL, above it the weighted + raised-order path runs unchanged.
2. **A test that could not fail.**  D1's
   ``test_zero_tilt_reproduces_the_scalar_chain`` claimed
   ``TiltedCarrier(R, 0, 0)`` "routes through the scalar path exactly", but
   ``_parse_chain_carrier`` reports ``tilted=False`` for it, so both arms were
   literally the same code path.  That pin is kept (correctly described, at its
   own site) and joined here by a genuinely TILTED one: a SKEW congruence
   (both ``L`` and ``M`` nonzero -- D1's own relay tilts in x only and asserts
   ``y_c_out == 0``) scored against an inline exact SKEW ray trace, with three
   demonstrated fail-before switches.
3. **D4's lens-window widening re-opened the v5.17.1 (P3-42) no-STOP
   aperture-DIAM pollution.**  Investigated, REPRODUCED (12.000 -> 100.000 mm,
   8.33x, on an ordinary DOE layout) and fixed at root: the no-STOP fallback
   reads the GLASS/MIRROR span only.
4. **The tilted exact-leg guard measured the wrong NA.**
   ``on_tilt_exact_grid`` was sized from the chain's PARAXIAL
   ``w_in/|R_out|``, not the element's own measured exit NA (design 121:
   0.4052 vs 0.4780), so it stayed silent on a leg
   ``apply_real_lens_traced`` itself warned was under-sampled.  A second,
   DECISIVE test now runs on the measured NA, with a power budget calibrated
   so the demonstrably-converged shipped configuration is still accepted.
5. **An unreachable remedy and a silent-typo hole.**  ``on_readout_window`` /
   ``readout_window_tol`` were not in ``_OUTPUT_GRID_PASSTHROUGH``, so the
   remedy the readout guard's own message prescribes could not be reached from
   the D2 orchestrator the 121 fan uses; and ``focus_readout`` had no key
   whitelist, so a typo'd ``'on_readout_windo'`` was silently accepted and the
   caller kept the hard ``'error'`` default.
"""
from __future__ import annotations

import dataclasses
import os
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced as _lt
from lumenairy.elements._lens_traced import _Cheb2DEvaluator
from lumenairy.io.prescriptions_zemax import load_zemax_zmx
from lumenairy.propagators import carrier as _c
from lumenairy.propagators.carrier import (
    _FOCUS_READOUT_KEYS,
    _OUTPUT_GRID_PASSTHROUGH,
    _group_abcd,
    _parse_chain_carrier,
    _tilt_obliquity,
)
from lumenairy.raytrace import Surface, make_ray, trace
from lumenairy.raytrace.trace import surfaces_from_prescription

_WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent',
            on_aperture_beam='silent')
_MIN_FREE_GIB = 3.0


def _ram_guard(need=_MIN_FREE_GIB):
    try:
        import psutil
    except ImportError:
        return
    free = psutil.virtual_memory().available / (1024 ** 3)
    if free < need:
        pytest.skip(f"needs ~{need} GiB available, saw {free:.1f}")


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


def _gauss(n, dx, w, cx=0.0, cy=0.0):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(((x[None, :] - cx) ** 2 + (x[:, None] - cy) ** 2) / w ** 2)
                  ).astype(np.complex128)


# ===========================================================================
# ITEM 1 -- the decentre gate: a NULL decentre must not flip the ray fit
# ===========================================================================
# Geometry: an f/6 N-BK7 singlet, ``fit_radius_beam_factor=2`` so the disc is
# live, launch radius 7.5 mm, beam radius 1.0 mm.  ``ray_subsample=8`` leaves
# ~4k coarse samples inside the disc, enough to carry D7's order-10 raise.
_D_N, _D_DX, _D_W = 512, 30e-6, 1.0e-3
_D_PRESC = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'c1-gate')
_D_FAST = _singlet(30e-3, -30e-3, 3.0e-3, 'N-BK7', 10e-3, 'c1-gate-fast')


def _apply_dec(c, tell, presc=_D_PRESC, carrier_form=False, **kw):
    """One element call with the beam physically at ``(c, 0)``.

    ``tell`` is the decentre the element is TOLD about (``None`` -> the grid
    origin).  ``carrier_form=True`` states it through a ``TiltedCarrier``
    instead of ``beam_centre``, which is the route the chain uses.
    """
    opts = dict(prescription=presc, wavelength=_WL, dx=_D_DX,
                ray_subsample=8, n_workers=1, fit_radius_beam_factor=2.0,
                **_TKW)
    if carrier_form:
        opts['carrier'] = la.TiltedCarrier(np.inf, 0.0, 0.0,
                                           *(tell or (0.0, 0.0)))
        if tell is None:
            opts['beam_centre'] = (0.0, 0.0)
    else:
        opts['carrier'] = np.inf
        opts['beam_centre'] = (0.0, 0.0) if tell is None else tell
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            _gauss(_D_N, _D_DX, _D_W, c, 0.0), **opts))


# Offsets that are physically NULL: sub-ulp, sub-pixel, one pixel, and two
# beam-relative offsets inside the 0.05 w gate.
_NULL_OFFSETS = [
    ('1e-9 px', 1e-9 * _D_DX),
    ('0.4 px', 0.4 * _D_DX),
    ('1 px', 1.0 * _D_DX),
    ('0.02 w', 0.02 * _D_W),
    ('0.049 w', 0.049 * _D_W),
]
# Offsets that are GENUINE: just past the gate, and the D6/D7 working range.
_REAL_OFFSETS = [('0.06 w', 0.06 * _D_W), ('0.2 w', 0.2 * _D_W),
                 ('1.0 w', 1.0 * _D_W)]


@pytest.mark.parametrize('label, c', _NULL_OFFSETS, ids=[o[0] for o in _NULL_OFFSETS])
def test_a_null_decentre_keeps_the_concentric_path_byte_identically(label, c):
    """BITWISE, not "to a tolerance": the whole finding is that the branch flip
    is a DISCONTINUITY, so the fix has to be exact equality with the
    origin-referenced arm -- which includes measuring the beam radius about the
    grid ORIGIN, as the historical path does.

    Measured against a pristine tree at HEAD (bd408bd), same call: the
    off-centre branch it took instead moved the field by 7.8e-8 of peak on this
    f/6 singlet and by **8.3e-6 at 1e-9 pixels** on the f/3 one below.
    """
    _ram_guard()
    ref = _apply_dec(c, None)
    got = _apply_dec(c, (c, 0.0))
    assert np.array_equal(got, ref), float(np.abs(got - ref).max())


def test_the_null_gate_holds_on_the_faster_singlet_and_through_a_carrier():
    """Same claim on the geometry where the pre-C1 step was largest (8.3e-6 of
    peak), and through the ``TiltedCarrier`` route the chain actually uses --
    a ``TiltedCarrier`` STATES the chief-ray position, so it feeds the same
    gate."""
    _ram_guard()
    c = 1e-9 * _D_DX
    a = _apply_dec(c, None, presc=_D_FAST)
    b = _apply_dec(c, (c, 0.0), presc=_D_FAST)
    assert np.array_equal(a, b), float(np.abs(a - b).max())
    # and via the carrier's own (x0, y0)
    c2 = 0.02 * _D_W
    p = _apply_dec(c2, None, carrier_form=True)
    q = _apply_dec(c2, (c2, 0.0), carrier_form=True)
    assert np.array_equal(p, q), float(np.abs(p - q).max())


@pytest.mark.parametrize('label, c', _REAL_OFFSETS, ids=[o[0] for o in _REAL_OFFSETS])
def test_a_genuine_decentre_still_routes_to_the_weighted_raised_order_path(
        label, c):
    """The gate is a floor, not a retreat: past it, D1's weighted restriction
    and D7's raised order still engage.  Verified two ways -- the returned
    field differs from the concentric arm, and the fit ORDER handed to
    ``_Cheb2DEvaluator`` is ``_DECENTRED_FIT_POLY_ORDER`` with weights.

    ERA-PINNED at ``DECENTRED_FIT_ARBITER = False`` (niche C11, 2026-08-03),
    which is the library state this case was calibrated in and the pure-gate
    selector's own pin.  Nothing here is relaxed and no threshold moved -- the
    assertions are the originals, word for word.  What changed under them is
    the INSTRUMENT: the C11 arbiter builds BOTH candidate fits before choosing,
    so this spy now collects two trial ``_Cheb2DEvaluator`` builds it was never
    written to see, and the ``all(...)`` over every build is no longer a
    statement about the fit the Newton inversion is handed.  The same claim,
    scoped to the applied fits and run on the SHIPPED default, is
    ``test_niche_c11::test_the_f6_fixture_still_routes_to_the_weighted_raised_order_path``
    -- which passes, so the claim itself is intact on this fixture; it is the
    counting that stopped being valid."""
    _ram_guard()
    # 5.32.1: the era pin needs BOTH flags now.  The selector block is entered
    # on ``ARBITER or PREDICTOR``, so pinning only the arbiter stopped being an
    # era pin the moment ``DECENTRED_FIT_PREDICTOR`` shipped ``True`` -- the
    # spy would collect the predictor's trial builds instead of the arbiter's,
    # and the count would be wrong for a different reason.
    _arb = (_lt.DECENTRED_FIT_ARBITER, _lt.DECENTRED_FIT_PREDICTOR)
    _lt.DECENTRED_FIT_ARBITER = False
    _lt.DECENTRED_FIT_PREDICTOR = False
    try:
        ref = _apply_dec(c, None)
        got = _apply_dec(c, (c, 0.0))
        assert not np.array_equal(got, ref)
        seen = []
        orig = _Cheb2DEvaluator.__init__

        def spy(self, xs_in, ys_in, values, order=6, xp=None, weights=None):
            orig(self, xs_in, ys_in, values, order=order, xp=xp,
                 weights=weights)
            seen.append((int(order), weights is not None))

        _Cheb2DEvaluator.__init__ = spy
        try:
            _apply_dec(c, (c, 0.0))
            assert seen and all(o == _lt._DECENTRED_FIT_POLY_ORDER
                                for o, _w in seen), seen
            assert all(w for _o, w in seen), \
                'the off-centre disc lost its weights'
            seen.clear()
            _apply_dec(c, None)
            assert seen and all(o == 6 for o, _w in seen), seen
            assert not any(w for _o, w in seen), \
                'the concentric disc used weights'
        finally:
            _Cheb2DEvaluator.__init__ = orig
    finally:
        _lt.DECENTRED_FIT_ARBITER, _lt.DECENTRED_FIT_PREDICTOR = _arb


def test_the_gate_is_the_documented_max_of_the_two_floors():
    """Straddle the threshold from both sides at the SAME physical beam, so the
    only thing that changes is which branch runs.

    ``max(0.5 * dx, 0.05 * w)``: here ``w`` = 1.0 mm dominates (0.05 w = 50 um
    against 0.5 dx = 15 um), and on a coarsely-sampled beam the pixel term
    takes over -- both sides are checked."""
    _ram_guard()
    w_meas = float(_lt._input_beam_amp_radius(
        _gauss(_D_N, _D_DX, _D_W), _D_DX, _D_DX))
    gate = max(_lt._DECENTRE_GATE_PIXELS * _D_DX,
               _lt._DECENTRE_GATE_W_FRAC * w_meas)
    assert gate == pytest.approx(_lt._DECENTRE_GATE_W_FRAC * w_meas)
    below, above = 0.97 * gate, 1.03 * gate
    ref_b = _apply_dec(below, None)
    assert np.array_equal(_apply_dec(below, (below, 0.0)), ref_b)
    ref_a = _apply_dec(above, None)
    assert not np.array_equal(_apply_dec(above, (above, 0.0)), ref_a)
    # the PIXEL floor: a beam sampled at only ~5 px of radius makes 0.05 w
    # sub-pixel, and the pitch term is then the binding one
    n2, dx2, w2 = 128, 40e-6, 0.2e-3
    presc2 = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 4e-3, 'c1-coarse')
    w2_meas = float(_lt._input_beam_amp_radius(
        _gauss(n2, dx2, w2), dx2, dx2))
    assert _lt._DECENTRE_GATE_W_FRAC * w2_meas < _lt._DECENTRE_GATE_PIXELS * dx2

    def small(c, tell):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                _gauss(n2, dx2, w2, c, 0.0), prescription=presc2,
                wavelength=_WL, dx=dx2, carrier=np.inf, ray_subsample=2,
                n_workers=1, fit_radius_beam_factor=2.0,
                beam_centre=(0.0, 0.0) if tell is None else tell,
                min_coarse_samples_per_aperture=0, **_TKW))

    c_sub = 0.4 * dx2
    assert np.array_equal(small(c_sub, (c_sub, 0.0)), small(c_sub, None))


def test_the_pre_c1_selector_is_the_fail_before_switch():
    """Setting both floors to 0.0 restores ``bool(_bcx or _bcy)`` exactly, and
    the null-decentre pin above then FAILS -- so it is a real pin, not a
    tautology.  Measured step on this geometry: 7.8e-8 of peak at 1e-9 px."""
    _ram_guard()
    c = 1e-9 * _D_DX
    ref = _apply_dec(c, None)
    p0, w0 = _lt._DECENTRE_GATE_PIXELS, _lt._DECENTRE_GATE_W_FRAC
    try:
        _lt._DECENTRE_GATE_PIXELS = 0.0
        _lt._DECENTRE_GATE_W_FRAC = 0.0
        broken = _apply_dec(c, (c, 0.0))
    finally:
        _lt._DECENTRE_GATE_PIXELS = p0
        _lt._DECENTRE_GATE_W_FRAC = w0
    step = float(np.abs(broken - ref).max()) / float(np.abs(ref).max())
    assert not np.array_equal(broken, ref)
    assert step > 1e-9, step
    # and with the gate back the same call is exact again
    assert np.array_equal(_apply_dec(c, (c, 0.0)), ref)


def test_the_on_axis_contract_is_untouched():
    """``beam_centre=None`` / ``(0, 0)`` and a CENTRED TiltedCarrier stay
    identical -- the D1 contract this gate must not disturb."""
    _ram_guard()
    kw = dict(prescription=_D_PRESC, wavelength=_WL, dx=_D_DX,
              ray_subsample=8, n_workers=1, fit_radius_beam_factor=2.0,
              **_TKW)
    env = _gauss(_D_N, _D_DX, _D_W)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(la.apply_real_lens_traced(env, carrier=np.inf, **kw))
        b = np.asarray(la.apply_real_lens_traced(
            env, carrier=np.inf, beam_centre=(0.0, 0.0), **kw))
        d = np.asarray(la.apply_real_lens_traced(
            env, carrier=la.TiltedCarrier(np.inf, 0.0, 0.0, 0.0, 0.0), **kw))
    assert np.array_equal(a, b)
    assert np.array_equal(a, d)


# ===========================================================================
# ITEM 2 -- a genuinely TILTED pin, with a demonstrated fail-before
# ===========================================================================
_S_N, _S_DX, _S_W = 1024, 12e-6, 1.0e-3
_S_L, _S_M = 0.044, 0.026          # SKEW: both components, hypot 0.0511 rad,
_S_GAP = 25e-3                     # each inside lambda/(2 dx) = 0.0546
_S_DXO, _S_NOUT = 0.05e-6, 1024
_S_TOL = 0.30e-6                   # 1.6 % of the measured 19.1 um spot FWHM
_S_gA = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'gA')
_S_gB = _singlet(60e-3, -60e-3, 3.0e-3, 'N-BK7', 10e-3, 'gB')
_S_GROUPS = [{'prescription': _S_gA, 'gap_before': 0.0},
             {'prescription': _S_gB, 'gap_before': _S_GAP}]


def _skew_paraxial(obliq=True):
    """The D1 closure by hand in BOTH transverse axes: ``R`` by the Moebius
    law, ``(x_c, L)`` and ``(y_c, M)`` as ordinary paraxial rays through each
    group, and the exact ``1/cos(theta)`` obliquity on each free leg.
    ``obliq=False`` is the fully-paraxial advance ``L z``, kept so the test can
    show it DISCRIMINATES the obliquity term."""
    A1, B1, C1, D1 = _group_abcd(_S_gA, _WL)
    A2, B2, C2, D2 = _group_abcd(_S_gB, _WL)
    xA, LA = B1 * _S_L, D1 * _S_L
    yA, MA = B1 * _S_M, D1 * _S_M
    ob = _tilt_obliquity(LA, MA, 'c1') if obliq else 1.0
    R_g = A1 / C1 + _S_GAP
    xg, yg = xA + LA * _S_GAP * ob, yA + MA * _S_GAP * ob
    R_B = (A2 * R_g + B2) / (C2 * R_g + D2)
    xB, LB = A2 * xg + B2 * LA, C2 * xg + D2 * LA
    yB, MB = A2 * yg + B2 * MA, C2 * yg + D2 * MA
    fd = -R_B
    ob2 = _tilt_obliquity(LB, MB, 'c1') if obliq else 1.0
    return dict(fd=fd, R_B=R_B, x=xB + LB * fd * ob2, y=yB + MB * fd * ob2,
                L_B=LB, M_B=MB)


def _skew_surfaces(fd):
    sa = surfaces_from_prescription(_S_gA)
    sb = surfaces_from_prescription(_S_gB)
    sa[-1] = dataclasses.replace(sa[-1], thickness=_S_GAP)
    sb[-1] = dataclasses.replace(sb[-1], thickness=fd)
    return sa + sb + [Surface(radius=np.inf, conic=0.0, semi_diameter=np.inf,
                              glass_before='air', glass_after='air',
                              is_mirror=False, thickness=0.0, label='img')]


def _exact_skew_chief(fd):
    """THE ORACLE: an exact SKEW ray trace of the chief ray (entrance height 0,
    direction cosines ``(L, M)``) through the real spherical surfaces.  No
    paraxial approximation, no propagator, no carrier machinery, no FFT."""
    r = trace(make_ray(0.0, 0.0, _S_L, _S_M, wavelength=_WL),
              _skew_surfaces(fd), _WL).image_rays
    return float(r.x[0]), float(r.y[0])


def _spot(F, dxo, cx, cy, ee_r=6e-6):
    I = np.abs(np.asarray(F)) ** 2
    tot = float(I.sum())
    n = I.shape[-1]
    ax = (np.arange(n) - n / 2) * dxo
    gx = float((I.sum(axis=0) * ax).sum() / tot)
    gy = float((I.sum(axis=1) * ax).sum() / tot)
    iy, _ = np.unravel_index(int(np.argmax(I)), I.shape)
    row = I[iy]
    idx = np.where(row >= 0.5 * row.max())[0]
    fwhm = float(idx[-1] - idx[0] + 1) * dxo
    rr = np.hypot(ax[None, :] - gx, ax[:, None] - gy)
    return dict(x=cx + gx, y=cy + gy, fwhm=fwhm, peak=float(I.max()),
                ee=float(I[rr <= ee_r].sum() / tot), power=tot * dxo * dxo)


def _run_skew(r_in, cx, cy, nout=_S_NOUT):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(
            _gauss(_S_N, _S_DX, _S_W), _S_GROUPS, _WL, _S_DX, r_in=r_in,
            ray_subsample=8, n_workers=1, final_distance=_SK['fd'],
            focus_readout=dict(dx_out=_S_DXO, N_out=nout,
                               centre_out=(cx, cy)),
            traced_kwargs=_TKW)


_SK = _skew_paraxial()
_SK_PAR = _skew_paraxial(obliq=False)


@pytest.fixture(scope='module')
def _skew_runs():
    _ram_guard()
    xe, ye = _exact_skew_chief(_SK['fd'])
    tilted = _run_skew(la.TiltedCarrier(np.inf, _S_L, _S_M), xe, ye)
    on_axis = _run_skew(np.inf, 0.0, 0.0)
    return dict(xe=xe, ye=ye, tilted=tilted, on_axis=on_axis,
                spot=_spot(tilted.field, _S_DXO, xe, ye),
                spot0=_spot(on_axis.field, _S_DXO, 0.0, 0.0))


def test_the_zero_tilt_pin_covers_the_REDUCTION_not_the_tilted_branch():
    """The honest statement of what ``TiltedCarrier(R, 0, 0)`` pins.

    ``_parse_chain_carrier`` reports ``tilted=False`` for it, so the dataclass
    entry point and a bare float take literally the SAME code path -- the pin
    is that the new entry point does not perturb the shipped scalar default,
    NOT that the tilted transport is right.  D1's
    ``test_zero_tilt_reproduces_the_scalar_chain`` is that pin; the tilted
    branch is covered by the skew tests below.  Asserted here so the two can
    never drift apart silently."""
    R = 0.06
    for spec in (la.TiltedCarrier(R, 0.0, 0.0), (R, 0.0, 0.0),
                 (R, 0.0, 0.0, 0.0, 0.0)):
        out = _parse_chain_carrier(spec, 'c1')
        assert out[:5] == (R, 0.0, 0.0, 0.0, 0.0)
        assert out[5] is False, f"{spec!r} claims to be tilted"
    # ... and anything with a real tilt or offset does NOT reduce
    for spec in (la.TiltedCarrier(R, 1e-6, 0.0),
                 la.TiltedCarrier(R, 0.0, 0.0, 1e-9, 0.0)):
        assert _parse_chain_carrier(spec, 'c1')[5] is True


def test_a_skew_tilted_congruence_lands_on_the_exact_skew_ray_trace(
        _skew_runs):
    """THE tilted-branch acceptance: the measured WAVE centroid of a
    (44, 26) mrad SKEW congruence must land on the exact skew ray trace of its
    own chief ray, in BOTH transverse axes.

    Measured on this fixture: exact trace (+1705.888, +1008.025) um, wave
    centroid **0.004 um** away, the chain's own analytic predictor 0.167 um
    away, and the fully-PARAXIAL advance (``L z`` instead of
    ``z L / cos(theta)``) **1.053 um** away -- 3.5x the 0.30 um tolerance, so
    the test can see the obliquity term at all."""
    r = _skew_runs
    d_wave = float(np.hypot(r['spot']['x'] - r['xe'], r['spot']['y'] - r['ye']))
    d_pred = float(np.hypot(_SK['x'] - r['xe'], _SK['y'] - r['ye']))
    d_par = float(np.hypot(_SK_PAR['x'] - r['xe'], _SK_PAR['y'] - r['ye']))
    # the test must DISCRIMINATE: a paraxial-advance chain misses by > 2 tol
    assert d_par > 2.0 * _S_TOL, d_par
    assert d_wave < _S_TOL, (d_wave, r['spot'], r['xe'], r['ye'])
    assert d_pred < _S_TOL, d_pred
    # both axes are genuinely off axis (this is a SKEW congruence, not D1's
    # meridional one) and the chain tracks the y half too
    assert abs(r['ye']) > 1.0e-3 and abs(r['xe']) > 1.0e-3
    st = r['tilted'].stages
    assert st[0]['y_c_out'] != 0.0 and st[0]['M_out'] != 0.0
    # niche C3: the chain's tracked chief ray is now an EXACT trace through
    # each group's surfaces, so it is pinned against this file's own exact
    # skew oracle -- NOT against the paraxial-ABCD composition ``_SK``, which
    # that same oracle already reports as ~0.17 um wrong here.  ``_SK`` is
    # kept below as the fail-before witness for exactly that gap.
    assert st[-1]['x_c'] == pytest.approx(r['xe'], abs=1.0e-9)
    assert st[-1]['y_c'] == pytest.approx(r['ye'], abs=1.0e-9)
    # ... and the ABCD it replaced really was measurably off, so this pin
    # cannot be satisfied by both conventions at once.
    assert d_pred > 20.0 * float(np.hypot(st[-1]['x_c'] - r['xe'],
                                          st[-1]['y_c'] - r['ye']) + 1e-18)


def test_the_skew_congruence_reaches_the_on_axis_diffraction_limit(_skew_runs):
    """Same relay, same beam, 51 mrad off axis in a skew direction: the spot
    must be the on-axis spot BROADENED BY THIS RELAY'S OWN COMA and no more,
    and carry the same power.  Measured: FWHM 19.20 vs 18.95 um, EE6 30.4 vs
    30.7 %, window power within 0.15 %.

    UPDATED 2026-07-30 (niche C5).  Two uncorrected singlets are not
    diffraction-limited at 51 mrad: a 2-D exact skew ray trace of this same
    Gaussian bundle reads a geometric rms spot radius of **1.719 um** off
    axis against 0.419 um on axis, which in quadrature with the 18.95 um
    on-axis FWHM predicts 19.38 um.  The measured 19.20 um sits between the
    two.  Pre-C5 this read 18.95 um -- EXACTLY the on-axis width -- because
    the tilted carrier reference was an on-axis sphere plus a linear ramp,
    which is not a wavefront and suppressed the coma.  So the bracket below
    replaces an equality that only the artefact could satisfy."""
    s, s0 = _skew_runs['spot'], _skew_runs['spot0']
    assert (s0['fwhm'] - 0.5 * _S_DXO <= s['fwhm']
            <= s0['fwhm'] + 10.0 * _S_DXO), (s['fwhm'], s0['fwhm'])
    assert s['ee'] == pytest.approx(s0['ee'], abs=0.01)
    assert s['power'] == pytest.approx(s0['power'], rel=0.02)
    # and it is the ABCD/Gaussian width, not merely "the same as on axis"
    na = _skew_runs['tilted'].stages[1]['w'] / abs(_SK['R_B'])
    fwhm_gauss = np.sqrt(2 * np.log(2)) * _WL / (np.pi * na)
    assert s['fwhm'] == pytest.approx(fwhm_gauss, rel=0.10), (s['fwhm'],
                                                              fwhm_gauss)


_BREAKS = [
    # (label, carrier-module attribute, replacement)
    ('obliquity->1', '_tilt_obliquity', lambda L, M, fn: 1.0),
    ('no tilt ramp', '_tilt_ramp', lambda *a, **k: None),
    ('no chief-ray shift', '_shift_envelope',
     lambda env, x, y, dx: env),
]


@pytest.mark.parametrize('label, attr, repl', _BREAKS,
                         ids=[b[0] for b in _BREAKS])
def test_breaking_the_tilted_path_moves_the_skew_spot_off_the_oracle(
        _skew_runs, label, attr, repl):
    """FAIL-BEFORE, demonstrated rather than asserted.  Each switch disables
    one mechanism of the tilted branch and the spot must then MISS the exact
    skew ray trace by more than the tolerance the passing test uses.

    Measured misses (against 0.004 um for the shipped path and a 0.30 um
    tolerance): the exact free-leg obliquity ``z L / cos(theta)`` -> 1.204 um;
    the tilt ramp -> 12.731 um AND the peak intensity collapses to 3e-6 of the
    shipped path's; the band-limited chief-ray shift into the element's axis
    frame -> 2.342 um (peak 0.976).  Peak intensity is used for the second
    clause rather than an encircled fraction because it is window-independent
    and the broken arms need a wider readout window to stay on the grid."""
    _ram_guard()
    orig = getattr(_c, attr)
    try:
        setattr(_c, attr, repl)
        broken = _run_skew(la.TiltedCarrier(np.inf, _S_L, _S_M),
                           _skew_runs['xe'], _skew_runs['ye'], nout=2048)
    finally:
        setattr(_c, attr, orig)
    sb = _spot(broken.field, _S_DXO, _skew_runs['xe'], _skew_runs['ye'])
    d = float(np.hypot(sb['x'] - _skew_runs['xe'], sb['y'] - _skew_runs['ye']))
    assert d > 2.0 * _S_TOL, (
        f"the {label} fail-before switch stopped failing: {d * 1e6:.4f} um "
        f"against the passing arm's "
        f"{np.hypot(_skew_runs['spot']['x'] - _skew_runs['xe'], _skew_runs['spot']['y'] - _skew_runs['ye']) * 1e6:.4f} um")
    if attr == '_tilt_ramp':
        # dropping the ramp does not merely displace the spot -- it destroys it
        assert sb['peak'] < 0.1 * _skew_runs['spot']['peak'], (
            sb['peak'], _skew_runs['spot']['peak'])
    else:
        # the other two break the POSITION while leaving a real spot behind,
        # which is what makes them the subtle regressions worth pinning
        assert sb['peak'] > 0.9 * _skew_runs['spot']['peak'], (
            sb['peak'], _skew_runs['spot']['peak'])


# ===========================================================================
# ITEM 3 -- the no-STOP aperture must not read an out-of-span DGRATING
# ===========================================================================
_ZHDR = [
    'VERS 210000 0 123 0 0', 'MODE SEQ', 'NAME c1_item3',
    'UNIT MM X W X CM MR CPMM', 'ENPD 10.0', 'WAVM 1 1.310000 1.0', 'PWAV 1',
]


def _zload(lines):
    fd, path = tempfile.mkstemp(suffix='.zmx', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            rx = load_zemax_zmx(path)
    finally:
        os.unlink(path)
    return rx, [str(w.message) for w in wl]


def _surf(n, *body):
    return [f'SURF {n}'] + list(body)


def _doe(diam='6.5', disz='12.0'):
    return ['  COMM DOE', '  TYPE DGRATING', '  CURV 0.0',
            '  PARM 1 0.00879', '  PARM 2 -4', f'  DISZ {disz}',
            f'  DIAM {diam}']


def _gfront(disz='3.0', diam='6.0', stop=False):
    out = ['  TYPE STANDARD', '  CURV 0.01 0 0 0 0 ""', f'  DISZ {disz}',
           '  GLAS SILICA 0 0 1.5 50.0', f'  DIAM {diam}']
    return (['  STOP'] + out) if stop else out


def _gback(disz='40.0', diam='6.0'):
    return ['  TYPE STANDARD', '  CURV -0.01 0 0 0 0 ""', f'  DISZ {disz}',
            f'  DIAM {diam}']


def _flat(disz='0.0', diam='3.0'):
    return ['  TYPE STANDARD', '  CURV 0.0', f'  DISZ {disz}', f'  DIAM {diam}']


_OBJ = _surf(0, '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY',
             '  DIAM 5.0')

# The glass is 6.0 mm semi-diameter throughout, so the correct no-STOP
# aperture is 12.000 mm in every layout below.
_GLASS_APERTURE_MM = 12.0

_LAYOUTS = {
    # the D4 suite's own fixture: collimated -> DGRATING -> singlet, no STOP.
    # Pre-D4 the DOE was outside the window; post-D4 it is inside, and its own
    # DIAM 6.5 became the aperture (13.000 mm).
    'doe_first': (_ZHDR + _OBJ + _surf(1, *_doe())
                  + _surf(2, *_gfront()) + _surf(3, *_gback())
                  + _surf(4, *_flat()) + ['BLNK']),
    # the v5.17.1 (P3-42) failure verbatim: the widening also drags in an
    # ordinary dummy reference plane, whose Zemax default DIAM is huge.
    'doe_first_big_dummy': (
        _ZHDR + _OBJ + _surf(1, *_doe(disz='6.0'))
        + _surf(2, *_flat(disz='6.0', diam='50.0'))
        + _surf(3, *_gfront()) + _surf(4, *_gback())
        + _surf(5, *_flat()) + ['BLNK']),
    'doe_last_big_dummy': (
        _ZHDR + _OBJ + _surf(1, *_gfront())
        + _surf(2, *_gback(disz='6.0'))
        + _surf(3, *_flat(disz='6.0', diam='50.0'))
        + _surf(4, *_doe(disz='9.0'))
        + _surf(5, *_flat(disz='5.0')) + _surf(6, *_flat()) + ['BLNK']),
}


@pytest.mark.parametrize('name', sorted(_LAYOUTS))
def test_an_out_of_span_dgrating_does_not_become_the_no_stop_aperture(name):
    """The no-STOP fallback is a ``max`` over the imported window, and D4
    widened that window to span a DGRATING outside the glass.  Measured before
    the fix: ``doe_first`` 12.000 -> **13.000 mm** (the DOE's own DIAM 6.5),
    and both big-dummy layouts 12.000 -> **100.000 mm (8.33x)** -- which is
    exactly what the v5.17.1 comment three lines above the changed code says
    must not happen.

    Consumer-visible: ``aperture_diameter`` sizes ``launch_radius``
    (0.5 * ap * 1.5 -> 9.00 mm becomes 75.00 mm), the Newton bound, the
    out-of-domain threshold and the ``min_coarse_samples_per_aperture`` floor.
    Measured on the widened number, ``ray_subsample=8`` at dx = 60 um SILENCES
    that floor (416.7 coarse samples across the fake 100 mm aperture instead of
    25.0 across the real 12 mm, where it correctly RAISES), and the
    ``fit_radius_beam_factor`` restriction is abandoned for want of in-disc
    samples -- i.e. two guards go inert at once."""
    rx, msgs = _zload(_LAYOUTS[name])
    assert len(rx['diffractives']) == 1, rx['diffractives']
    assert rx['aperture_diameter'] * 1e3 == pytest.approx(
        _GLASS_APERTURE_MM, rel=1e-12), rx['aperture_diameter'] * 1e3
    assert any('GLASS/MIRROR span' in m for m in msgs), msgs


def test_a_stop_wins_and_a_doe_free_file_is_untouched():
    """The pollution was no-STOP-only, as the v5.17.1 comment says, and the
    fix must not touch anything else.  A declared STOP still wins outright,
    and a file with no DGRATING keeps the historical whole-window ``max``
    (here a 50 mm dummy inside the glass span -> 100.000 mm, deliberately
    unchanged: that is the pre-existing heuristic, not this item's bug)."""
    with_stop = (_ZHDR + _OBJ + _surf(1, *_doe(disz='6.0'))
                 + _surf(2, *_flat(disz='6.0', diam='50.0'))
                 + _surf(3, *_gfront(stop=True)) + _surf(4, *_gback())
                 + _surf(5, *_flat()) + ['BLNK'])
    rx, msgs = _zload(with_stop)
    assert rx['aperture_diameter'] * 1e3 == pytest.approx(12.0, rel=1e-12)
    assert not any('GLASS/MIRROR span' in m for m in msgs)
    no_doe = (_ZHDR + _OBJ + _surf(1, *_flat(disz='6.0', diam='50.0'))
              + _surf(2, *_gfront()) + _surf(3, *_gback())
              + _surf(4, *_flat()) + ['BLNK'])
    rx2, msgs2 = _zload(no_doe)
    assert rx2['diffractives'] == []
    assert rx2['aperture_diameter'] * 1e3 == pytest.approx(12.0, rel=1e-12)
    assert not any('GLASS/MIRROR span' in m for m in msgs2)
    # and the dummy really is only excluded when the DOE widening pulled it in:
    # put it INSIDE the glass span of a DOE-free file and it still counts
    inside = (_ZHDR + _OBJ + _surf(1, *_gfront(disz='3.0'))
              + _surf(2, *_gback(disz='6.0'))
              + _surf(3, *_flat(disz='6.0', diam='50.0'))
              + _surf(4, *_gfront(disz='3.0')) + _surf(5, *_gback())
              + _surf(6, *_flat()) + ['BLNK'])
    rx3, _ = _zload(inside)
    assert rx3['aperture_diameter'] * 1e3 == pytest.approx(100.0, rel=1e-12)


def test_a_diffractive_only_file_still_gets_an_aperture():
    """No glass at all -> there is no glass span to fall back to, so the whole
    imported window is used (and the load still succeeds, which it did not
    pre-D4: ``ValueError: No glass/mirror surfaces found``)."""
    doe_only = (_ZHDR + _OBJ + _surf(1, *_doe(disz='6.0'))
                + _surf(2, *_doe(diam='7.5', disz='6.0'))
                + _surf(3, *_flat()) + ['BLNK'])
    rx, msgs = _zload(doe_only)
    assert len(rx['diffractives']) == 2
    assert rx['aperture_diameter'] * 1e3 == pytest.approx(15.0, rel=1e-12)
    assert not any('GLASS/MIRROR span' in m for m in msgs)


def test_an_explicit_surface_range_is_still_the_callers_own_window():
    """``surface_range`` is honoured as given, aperture included -- unchanged,
    because the caller asked for that window (the D4 warning about a DGRATING
    it excludes is a separate contract and stays)."""
    rx, _ = _zload(_LAYOUTS['doe_first_big_dummy'])
    assert rx['aperture_diameter'] * 1e3 == pytest.approx(12.0, rel=1e-12)
    lines = _LAYOUTS['doe_first_big_dummy']
    fd, path = tempfile.mkstemp(suffix='.zmx', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            rx2 = load_zemax_zmx(path, surface_range=(1, 4))
    finally:
        os.unlink(path)
    assert rx2['aperture_diameter'] * 1e3 == pytest.approx(100.0, rel=1e-12)


# ===========================================================================
# ITEM 4 -- the tilted exact-leg guard, sourced from the MEASURED exit NA
# ===========================================================================
# The paraxial pre-check's blind spot: a group whose measured exit NA is far
# above w_in/|R_out|.  This singlet reads paraxial 0.0520 vs MEASURED 0.3407
# (6.6x), so D6's pre-check is silent at every grid below.
_G4 = _singlet(8.0e-3, -8.0e-3, 1.5e-3, 'N-BK7', 3.2e-3, 'c1-blindspot')
_G4_GROUPS = [{'prescription': _G4, 'gap_before': 0.0}]
_N4, _DX4, _W4, _TILT4 = 2048, 2.0e-6, 0.40e-3, 0.02
_FD4 = 8.0e-3


def _run4(cap, **kw):
    opts = dict(final_leg='exact', on_tilt_exact_grid='warn',
                on_multi_congruence='ignore', on_decentred_fit='ignore',
                on_rs_fine_clamp='ignore', on_ram_cap='ignore')
    opts.update(kw)
    return la.propagate_traced_carrier_chain(
        _gauss(_N4, _DX4, _W4), _G4_GROUPS, _WL, _DX4,
        r_in=la.TiltedCarrier(np.inf, _TILT4, 0.0),
        focus_readout=dict(dx_out=0.05e-6, N_out=512, window_factor=4.0,
                           n_fine_cap=cap),
        final_distance=_FD4, ray_subsample=2, n_workers=1,
        traced_kwargs=_TKW, **opts)


def test_the_element_reports_the_measured_exit_na_and_the_aliased_power():
    """The element already measured its exit NA for its own Nyquist warning;
    ``_exit_na_out`` just hands it back.  Checked against the definition
    recomputed here from an independent trace of the same rays."""
    _ram_guard()
    diag: dict = {}
    dx = 4.0e-6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens_traced(
            _gauss(512, dx, _W4), prescription=_G4, wavelength=_WL, dx=dx,
            carrier=np.inf, ray_subsample=2, n_workers=1,
            min_coarse_samples_per_aperture=0, _exit_na_out=diag, **_TKW)
    assert set(diag) >= {'na_exit', 'dx', 'na_nyquist',
                         'power_frac_above_nyquist', 'n_rays'}
    assert diag['dx'] == pytest.approx(dx, rel=1e-12)
    assert diag['na_nyquist'] == pytest.approx(_WL / (2.0 * dx), rel=1e-12)
    assert 0.0 < diag['na_exit'] < 1.0
    assert 0.0 <= diag['power_frac_above_nyquist'] <= 1.0
    # a much FINER grid carries more NA, so the aliased fraction must not rise
    diag2: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens_traced(
            _gauss(2048, dx / 4, _W4), prescription=_G4, wavelength=_WL,
            dx=dx / 4, carrier=np.inf, ray_subsample=2, n_workers=1,
            min_coarse_samples_per_aperture=0, _exit_na_out=diag2, **_TKW)
    assert diag2['na_nyquist'] > diag['na_nyquist']
    assert diag2['power_frac_above_nyquist'] <= \
        diag['power_frac_above_nyquist'] + 1e-12
    # and it is PURELY diagnostic: passing the sink changes no output
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(la.apply_real_lens_traced(
            _gauss(512, dx, _W4), prescription=_G4, wavelength=_WL, dx=dx,
            carrier=np.inf, ray_subsample=2, n_workers=1,
            min_coarse_samples_per_aperture=0, **_TKW))
        b = np.asarray(la.apply_real_lens_traced(
            _gauss(512, dx, _W4), prescription=_G4, wavelength=_WL, dx=dx,
            carrier=np.inf, ray_subsample=2, n_workers=1,
            min_coarse_samples_per_aperture=0, _exit_na_out={}, **_TKW))
    assert np.array_equal(a, b)


def test_the_chain_reports_the_measured_na_next_to_the_paraxial_one():
    """``stages[last]`` carries the quantity the guard now decides on, so the
    margin is visible without catching a warning -- the same reason D3 put
    ``na_exit`` there.  On this stand-in the two NAs differ by 6.6x, which is
    the whole finding."""
    _ram_guard()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = _run4(2048)
    st = [s for s in res.stages if s.get('exact_final')][0]
    for k in ('na_exit', 'na_exit_measured', 'na_grid_nyquist',
              'exit_power_above_nyquist'):
        assert k in st, sorted(st)
    assert st['na_exit_measured'] > 3.0 * st['na_exit'], (
        st['na_exit_measured'], st['na_exit'])
    assert st['na_grid_nyquist'] == pytest.approx(
        _WL / (2.0 * st['dx']), rel=1e-12)
    # this grid CARRIES the measured NA, so nothing aliases and no guard fires
    assert st['na_grid_nyquist'] > st['na_exit_measured']
    assert st['exit_power_above_nyquist'] < _c._TILT_EXACT_NA_POWER_TOL


def test_the_measured_na_guard_closes_the_paraxial_pre_checks_blind_spot():
    """The hole, and its closure.  At ``n_fine_cap=192`` the retrace grid
    carries NA 0.0786 against a MEASURED exit NA 0.3408 -- 23 % of it, with
    2.11e-2 of the exit power aliasing -- while the PARAXIAL pre-check is
    silent, because ``lambda/(2 * 0.0520)`` = 12.6 um is coarser than the
    3.13 um... 8.33 um ``dx_fine`` this sweep produces.  So pre-C1 the leg ran
    with no diagnostic at all."""
    _ram_guard()
    # 'warn' first, to read the numbers the refusal is made of
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res = _run4(192)
    st = [s for s in res.stages if s.get('exact_final')][0]
    assert st['exit_power_above_nyquist'] > _c._TILT_EXACT_NA_POWER_TOL
    hits = [str(w.message) for w in wl if 'MEASURED exit NA' in str(w.message)]
    assert hits, [str(w.message)[:90] for w in wl]
    # the message must name BOTH NAs, the aliased fraction and the budget
    assert 'ALIASES' in hits[0] and 'NA=0.34' in hits[0], hits[0]
    assert 'w_in/|R_out| = 0.052' in hits[0], hits[0]
    assert '2.110 %' in hits[0] and 'tolerance 1 %' in hits[0], hits[0]
    # the PARAXIAL pre-check cannot see it: its own message never appears
    assert not [w for w in wl
                if 'merely NYQUIST-sampling the exit sphere' in str(w.message)]
    # ... and the default is a refusal
    with pytest.raises(RuntimeError, match='MEASURED exit NA'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _run4(192, on_tilt_exact_grid='error')
    # 'ignore' silences it entirely
    with warnings.catch_warnings(record=True) as wl2:
        warnings.simplefilter('always')
        _run4(192, on_tilt_exact_grid='ignore')
    assert not [w for w in wl2 if 'MEASURED exit NA' in str(w.message)]


def test_the_guard_stays_silent_where_the_grid_carries_the_measured_na():
    """Tightening the guard must not start refusing what is demonstrably fine.
    At ``n_fine_cap`` 2048/1024 this leg's grid carries NA 0.4192 against the
    measured 0.3407, aliased fraction 4.25e-07, and the guard must not fire --
    the same shape as the shipped design-121 headline, which measures
    7.97e-04 against the 1e-2 budget."""
    _ram_guard()
    for cap in (2048, 1024):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            res = _run4(cap, on_tilt_exact_grid='error')
        st = [s for s in res.stages if s.get('exact_final')][0]
        assert st['exit_power_above_nyquist'] < 1e-4, st
        assert not [w for w in wl if 'MEASURED exit NA' in str(w.message)]


def test_the_tolerance_is_the_fail_before_switch_for_item_4():
    """``_TILT_EXACT_NA_POWER_TOL = inf`` restores D6's paraxial-only guard, so
    the refusing configuration above runs silently again.  A guard whose
    fail-before cannot be demonstrated is not a guard."""
    _ram_guard()
    tol0 = _c._TILT_EXACT_NA_POWER_TOL
    try:
        _c._TILT_EXACT_NA_POWER_TOL = np.inf
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            _run4(192, on_tilt_exact_grid='error')
        assert not [w for w in wl if 'MEASURED exit NA' in str(w.message)]
    finally:
        _c._TILT_EXACT_NA_POWER_TOL = tol0
    with pytest.raises(RuntimeError, match='MEASURED exit NA'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _run4(192, on_tilt_exact_grid='error')


def test_the_on_axis_exact_leg_is_not_disposed_by_the_tilted_guard():
    """The post-check is TILT-ONLY, like the pre-check it joins: an on-axis
    exact leg on a grid that cannot carry its measured NA keeps the on-axis
    F-D behaviour (warn-only) and must not raise under the default
    ``on_tilt_exact_grid='error'``."""
    _ram_guard()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            _gauss(_N4, _DX4, _W4), _G4_GROUPS, _WL, _DX4, r_in=np.inf,
            focus_readout=dict(dx_out=0.05e-6, N_out=512, window_factor=4.0,
                               n_fine_cap=192),
            final_distance=_FD4, ray_subsample=2, n_workers=1,
            traced_kwargs=_TKW, final_leg='exact',
            on_tilt_exact_grid='error', on_multi_congruence='ignore',
            on_rs_fine_clamp='ignore', on_ram_cap='ignore')
    st = [s for s in res.stages if s.get('exact_final')][0]
    assert st['exit_power_above_nyquist'] > _c._TILT_EXACT_NA_POWER_TOL
    assert not [w for w in wl if 'MEASURED exit NA' in str(w.message)]


# ===========================================================================
# ITEM 5 -- the readout-window remedy's reachability, and the typo hole
# ===========================================================================
_G5 = _singlet(6.0e-3, -6.0e-3, 1.2e-3, 'N-BK7', 2.4e-3, 'c1-item5')
_G5_GROUPS = [{'prescription': _G5, 'gap_before': 0.0}]
_N5, _DX5, _W5 = 256, 12e-6, 0.30e-3
# A tilted congruence entering ON AXIS leaves at x_c_out = B L != 0, so the
# exact readout's crop about the chief ray has only N*dx - 2|x_c_out| to work
# with while it wants window_factor * w_exit -- the clamp the guard measures.
# Measured truncation at window_factor=3.0: 0.557 % (L=0.10), 1.876 % (0.15),
# 4.317 % (0.20), against the 1e-4 default tolerance.
_L5, _WF5 = 0.15, 3.0


def _multi5(**og):
    d = dict(dx_out=0.2e-6, N_out=256, window_factor=_WF5)
    d.update(og)
    return la.propagate_traced_carrier_chain_multi(
        [{'field': _gauss(_N5, _DX5, _W5),
          'carrier': la.TiltedCarrier(np.inf, _L5, 0.0)}],
        _G5_GROUPS, _WL, _DX5, output_grid=d, final_distance=5.0e-3,
        final_leg='exact', ray_subsample=4, n_workers=1, traced_kwargs=_TKW,
        on_multi_congruence='ignore', on_readout_clip='ignore',
        on_replica='ignore', readout_tile=None,
        # This stand-in is deliberately a coarse leg (its whole job is to make
        # the READOUT-WINDOW clamp bite), so item 4's measured-NA guard fires
        # here too -- 88.4 % of its exit power aliases.  Silence that one so
        # these tests are about the readout-window guard alone.
        on_tilt_exact_grid='ignore')


def test_the_readout_window_remedy_is_reachable_from_the_multi_orchestrator():
    """The guard's own message prescribes ``on_readout_window='warn'`` (or a
    larger ``readout_window_tol``).  Pre-C1 neither key was in
    ``_OUTPUT_GRID_PASSTHROUGH``, so from ``..._multi`` -- the entry point the
    design-121 fan uses -- the prescribed remedy raised ``ValueError:
    output_grid has unknown key(s) ['on_readout_window']`` instead.  Measured
    truncation on this stand-in: 1.876 % of the field's power."""
    _ram_guard()
    for k in ('on_readout_window', 'readout_window_tol'):
        assert k in _OUTPUT_GRID_PASSTHROUGH
    # the DEFAULT is still the hard refusal
    with pytest.raises(RuntimeError, match='TRUNCATES'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _multi5()
    # the prescribed remedy now works, and says what it accepted
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        r = _multi5(on_readout_window='warn')
    assert np.isfinite(np.asarray(r.field)).all()
    hits = [str(w.message) for w in wl if 'TRUNCATES' in str(w.message)]
    assert hits, [str(w.message)[:80] for w in wl]
    assert '%' in hits[0]
    # the OTHER prescribed remedy -- raise the tolerance above the loss
    with warnings.catch_warnings(record=True) as wl2:
        warnings.simplefilter('always')
        _multi5(readout_window_tol=0.05)
    assert not [w for w in wl2 if 'TRUNCATES' in str(w.message)]
    # explicit 'error' and 'ignore' both reach the readout too
    with pytest.raises(RuntimeError, match='TRUNCATES'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _multi5(on_readout_window='error')
    with warnings.catch_warnings(record=True) as wl3:
        warnings.simplefilter('always')
        _multi5(on_readout_window='ignore')
    assert not [w for w in wl3 if 'TRUNCATES' in str(w.message)]


def test_a_typo_in_focus_readout_raises_instead_of_keeping_the_default():
    """``focus_readout`` had NO key whitelist, so ``'on_readout_windo'`` was
    silently accepted and the caller kept the hard ``'error'`` default while
    reading as a downgrade.  The accepted set is now named in the error."""
    kw = dict(final_distance=5.0e-3, final_leg='exact', ray_subsample=4,
              n_workers=1, traced_kwargs=_TKW)
    for bad in ('on_readout_windo', 'nonsense', 'N_Out', 'dxout'):
        with pytest.raises(ValueError, match='focus_readout has unknown'):
            la.propagate_traced_carrier_chain(
                _gauss(_N5, _DX5, _W5), _G5_GROUPS, _WL, _DX5, r_in=np.inf,
                focus_readout={'dx_out': 0.2e-6, 'N_out': 256, bad: 'warn'},
                **kw)
    with pytest.raises(ValueError, match='focus_readout must be a dict'):
        la.propagate_traced_carrier_chain(
            _gauss(_N5, _DX5, _W5), _G5_GROUPS, _WL, _DX5, r_in=np.inf,
            focus_readout=[('dx_out', 0.2e-6)], **kw)


def test_the_focus_readout_whitelist_is_exactly_what_the_chain_consumes():
    """The whitelist must not be a hand-maintained second list: it is
    ``{dx_out, N_out, centre_out}`` plus ``_OUTPUT_GRID_PASSTHROUGH``, so the
    two entry points cannot drift apart.  Every accepted key is also checked
    to actually pass validation."""
    assert _FOCUS_READOUT_KEYS == (
        {'dx_out', 'N_out', 'centre_out'} | set(_OUTPUT_GRID_PASSTHROUGH))
    assert {'on_readout_window', 'readout_window_tol', 'window_factor',
            'n_fine_cap', 'on_n_fine_cap', 'bandlimit',
            'standoff'} <= _FOCUS_READOUT_KEYS
    sample = {'dx_out': 0.2e-6, 'N_out': 64, 'centre_out': (0.0, 0.0),
              'standoff': 1e-4, 'bandlimit': True, 'window_factor': 6.0,
              'n_fine_cap': 512, 'max_fine_launch_points': 512,
              'ram_budget': None, 'dx_fine': None, 'N_fine': None,
              'on_readout_window': 'ignore', 'readout_window_tol': 1e-3,
              # v5.33.2 (AUDIT_TRACED_MEMORY_2026_08_09 row 10): ``n_fine_cap``
              # now caps the exact READOUT's internal grid too, so its
              # disposition knob reaches the chain through the same whitelist.
              # This fixture's final leg is paraxial, so the key is DROPPED by
              # ``_par_kw`` -- which is the other half of the contract.
              'on_n_fine_cap': 'warn',
              # D3 (2026-08-06): the readouts' own replica guard reaches the
              # chain through the same whitelist.  'ignore' here because this
              # fixture's window is wider than one period and the subject of
              # the test is the WHITELIST, not the guard.
              'on_replica': 'ignore'}
    assert set(sample) == _FOCUS_READOUT_KEYS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.propagate_traced_carrier_chain(
            _gauss(_N5, _DX5, _W5), _G5_GROUPS, _WL, _DX5, r_in=np.inf,
            focus_readout=sample, final_distance=5.0e-3,
            final_leg='paraxial', ray_subsample=4, n_workers=1,
            traced_kwargs=_TKW)


def test_output_grid_still_refuses_unknown_keys_and_the_defaults_do_not_move():
    """Adding two keys must not open ``output_grid`` up, and a run that
    supplies NEITHER must be bitwise what it was: the same run with the
    documented defaults spelled out."""
    _ram_guard()
    with pytest.raises(ValueError, match='output_grid has unknown key'):
        _multi5(on_readout_windo='warn')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = _multi5(window_factor=5.0, on_readout_window='ignore')
        b = _multi5(window_factor=5.0, on_readout_window='ignore',
                    readout_window_tol=1e-4)
        c = _multi5(window_factor=5.0)
    assert np.array_equal(np.asarray(a.field), np.asarray(b.field))
    assert np.array_equal(np.asarray(a.field), np.asarray(c.field))


# ===========================================================================
# NICHE C2 ITEM 5 -- a DGRATING gap that runs through GLASS
# ===========================================================================
# ``gap_before`` / ``gap_after`` are raw axial thicknesses and the chain
# transports them through AIR, so a grating ruled on a substrate would be
# placed at the wrong optical distance (t - t/n per glass leg) with no symptom
# in the output.  The importer's docstring claimed gap_before was "always the
# true distance from the previous optical element"; it is true only while the
# intervening legs are free space.
#
# The refusal deliberately lives at the POINT OF USE, not at import: a .zmx
# containing such a grating still loads (its surfaces, glasses and geometry are
# all correct -- only the DOE gap bookkeeping is not), and only handing that
# DOE entry to the chain raises.

def _glass_doe_zmx(glass='SILICA', disz='3.0'):
    """Grating ruled on the FRONT face of a `disz` mm `glass` substrate."""
    return _ZHDR + (
        _surf(0, '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY',
              '  DIAM 5.0')
        + _surf(1, '  TYPE DGRATING', '  CURV 0.0', '  PARM 1 0.00879',
                '  PARM 2 -4', f'  DISZ {disz}',
                f'  GLAS {glass} 0 0 1.5 50.0', '  DIAM 5.0')
        + _surf(2, '  TYPE STANDARD', '  CURV -0.005', '  DISZ 50.0',
                '  DIAM 5.0')
        + _surf(3, '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0',
                '  DIAM 5.0')
        + ['BLNK'])


def test_a_grating_gap_through_glass_is_flagged_at_import_not_refused():
    """The FILE still imports -- only the gap bookkeeping is suspect."""
    rx, msgs = _zload(_glass_doe_zmx())
    assert rx['surfaces'], "the surfaces must still import"
    d = rx['diffractives']
    assert len(d) == 1
    media = d[0].get('gap_media')
    assert media, "the non-free-space gap must be recorded on the entry"
    assert any(m['glass'].lower() != 'air' for m in media)
    assert any(abs(m['thickness'] - 3.0e-3) < 1e-12 for m in media), media
    assert any('does NOT lie in free space' in m for m in msgs), msgs


def test_the_chain_refuses_that_entry_rather_than_misplacing_the_grating():
    rx, _ = _zload(_glass_doe_zmx())
    spec = rx['diffractives'][0]
    with pytest.raises(NotImplementedError) as ei:
        _c._normalise_doe_entry({'doe': spec}, 0, 1.31e-6, 'fn')
    m = str(ei.value)
    assert 'free space' in m and 't - t/n' in m
    assert 'SILICA' in m.upper()


def test_overriding_the_offending_gap_is_the_documented_way_through():
    """FAIL-BEFORE / PASS-AFTER: the refusal names the gap to override, and
    supplying it on the entry (reduced to a free-space distance by the caller)
    lets the same grating through."""
    rx, _ = _zload(_glass_doe_zmx())
    spec = rx['diffractives'][0]
    bad = {m['gap'] for m in spec['gap_media']}
    assert bad, "fixture must actually carry a flagged gap"
    entry = {'doe': spec}
    entry.update({g: 3.0e-3 / 1.4585 for g in bad})   # t/n, done by the caller
    out = _c._normalise_doe_entry(entry, 0, 1.31e-6, 'fn')
    assert out is not None


def test_a_free_space_grating_is_untouched_by_the_c2_check():
    """Byte-level fail-safe: the ordinary air-gapped DOE gains no marker and
    parses exactly as before."""
    rx, _ = _zload(
        _ZHDR
        + _surf(0, '  TYPE STANDARD', '  CURV 0.0', '  DISZ INFINITY',
                '  DIAM 5.0')
        + _surf(1, '  TYPE STANDARD', '  CURV 0.01', '  DISZ 2.0',
                '  GLAS SILICA 0 0 1.5 50.0', '  DIAM 5.0')
        + _surf(2, '  TYPE STANDARD', '  CURV -0.005', '  DISZ 8.0',
                '  DIAM 5.0')
        + _surf(3, *_doe(diam='6.5', disz='12.0'))
        + _surf(4, '  TYPE STANDARD', '  CURV 0.0', '  DISZ 0.0',
                '  DIAM 5.0')
        + ['BLNK'])
    d = rx['diffractives']
    assert len(d) == 1
    assert d[0].get('gap_media') is None, d[0].get('gap_media')
    assert _c._normalise_doe_entry({'doe': d[0]}, 0, 1.31e-6, 'fn') is not None
