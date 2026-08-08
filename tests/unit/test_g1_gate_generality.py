"""G1 item 1: generality of the H2 sag-screen aberration gate.

The pre-G1 gate summed ``k * r_beam^4 / (2|R|^3)`` per powered surface using the
INPUT beam radius at EVERY surface and only the SPHERICAL r^4 term.  G1 extends
it three ways, all byte-identical for a single spherical surface with a
collimated real input:

  (a) conic / aspheric: the spherical r^4 SAG coefficient ``1/(8R^3)`` is
      replaced by the true ``c4 = (1+k)/(8R^3) + A4`` (item 1a);
  (b) multi-element: a paraxial marginal-ray y-nu trace gives the true beam
      radius AT each surface, not the input radius everywhere (item 1b);
  (c) converging / diverging input: the marginal ray starts at the measured
      signed input slope (item 1c).

Oracle backing: the 2026-07-19 displaced measurement (docs/audit_real_lens_
displaced_2026_07_19.md) shows the analytic ``thin`` (phase_screen) model is
58-123% wrong vs the Debye oracle on the doublet / asphere / meniscus fast
regime, so routing those AWAY from phase_screen is the correct (accurate)
outcome.  A false trip only costs speed (the ray members are never wrong).
"""
from __future__ import annotations

import numpy as np

from lumenairy.propagators.fga import (
    _ABERRATION_MAX_RAD,
    _input_marginal_slope,
    _sag_screen_aberration_rad,
    _system_na,
    _universal_route,
)
from lumenairy.raytrace import surfaces_from_prescription

_WL = 1.31e-6
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {
    _n: (lambda vv: (lambda wl: vv))(_v)
    for _n, _v in {'_G1_1p5168': 1.5168, '_G1_1p62': 1.62,
                   '_G1_1p70': 1.70}.items()}


def _gauss(N, dx, w0, R_in=None):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
    if R_in is not None:
        k = 2 * np.pi / _WL
        return (amp * np.exp(1j * k * (X ** 2 + Y ** 2) / (2 * R_in))
                ).astype(np.complex128)
    return amp.astype(np.complex128)


def _spherical_only_input_radius(E, dx, presc):
    """The PRE-G1 estimate: r_beam^4 * k/(2R^3) at the INPUT radius on every
    surface, ignoring conic/asphere -- the fail-before reference."""
    a2 = np.abs(E) ** 2
    tot = a2.sum()
    N = E.shape[0]
    xs = (np.arange(N) - N // 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    r_beam = np.sqrt(2 * np.sum(a2 * (Xg ** 2 + Yg ** 2)) / tot)
    k = 2 * np.pi / _WL
    tot_est = 0.0
    for s in surfaces_from_prescription(presc):
        R = float(getattr(s, 'radius', np.inf))
        if not np.isfinite(R) or R == 0:
            continue
        tot_est += k * r_beam ** 4 / (2 * abs(R) ** 3)
    return tot_est


def _biconvex(R, t, semi, glass='_G1_1p5168', asph=None, conic=0.0):
    return {'wavelength': _WL, 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R, 'conic': conic, 'aspheric_coeffs': asph,
                 'thickness': t, 'glass_before': 'air', 'glass_after': glass,
                 'semi_diameter': semi},
                {'radius': -R, 'thickness': 0.0, 'glass_before': glass,
                 'glass_after': 'air', 'semi_diameter': semi}],
            'thicknesses': [t], 'stop_index': 0}


# ---------------------------------------------------------------------------
# Byte-identity: a plain spherical prescription with a collimated real input
# is unchanged (spherical branch + single-surface height == r_beam).
# ---------------------------------------------------------------------------
def test_spherical_single_powered_surface_byte_identical():
    """A plano-convex (one powered surface) with a collimated real input keeps
    h == r_beam and takes the spherical branch, so the estimate equals the
    pre-G1 input-radius formula EXACTLY."""
    presc = {'wavelength': _WL, 'aperture_diameter': 20e-3,
             'surfaces': [
                 {'radius': 25.8e-3, 'thickness': 5e-3, 'glass_before': 'air',
                  'glass_after': '_G1_1p5168', 'semi_diameter': 10e-3},
                 {'radius': np.inf, 'thickness': 0.0,
                  'glass_before': '_G1_1p5168', 'glass_after': 'air',
                  'semi_diameter': 10e-3}],
             'thicknesses': [5e-3], 'stop_index': 0}
    N, dx = 256, 2 * 11.5e-3 / 256
    E = _gauss(N, dx, 6e-3)
    est = _sag_screen_aberration_rad(E, dx, dx, presc, _WL)
    ref = _spherical_only_input_radius(E, dx, presc)
    assert abs(est - ref) < 1e-9 * ref, (est, ref)


def test_collimated_real_input_has_zero_marginal_slope():
    """A real / globally-phased (flat-wavefront) field measures u_in == 0
    exactly, so a collimated input's height trace starts at r_beam."""
    N, dx = 256, 8e-6
    E = _gauss(N, dx, 0.5e-3)
    a2 = np.abs(E) ** 2
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    r_beam = np.sqrt(2 * np.sum(a2 * (X ** 2 + Y ** 2)) / a2.sum())
    # a genuinely real-valued field (imag exactly 0) -> u_in == 0.0 EXACTLY
    # (this is the byte-identity guarantee for the default real-Gaussian input)
    assert _input_marginal_slope(E, a2, X, Y, dx, dx, _WL, r_beam) == 0.0
    # a globally-phased field is still flat; the FP residual is negligible
    phased = E * np.exp(1j * 0.7)
    assert abs(_input_marginal_slope(phased, np.abs(phased) ** 2, X, Y, dx, dx,
                                     _WL, r_beam)) < 1e-12


def test_marginal_slope_recovers_input_curvature_on_resolved_grid():
    """On a grid that RESOLVES the input carrier fringe, the LS estimator
    recovers u_in ~ r_beam / R for a diverging (R>0) and converging (R<0)
    input.  (On a coarse grid the fringe aliases and u_in falls back toward 0 =
    the safe collimated height trace -- documented in the gate.)"""
    N, dx = 768, 8e-6                       # fine grid: resolves R=+/-60mm @ w0=1mm
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    for R_in in (60e-3, -60e-3):
        E = _gauss(N, dx, 1.0e-3, R_in=R_in)
        a2 = np.abs(E) ** 2
        r_beam = np.sqrt(2 * np.sum(a2 * (X ** 2 + Y ** 2)) / a2.sum())
        u = _input_marginal_slope(E, a2, X, Y, dx, dx, _WL, r_beam)
        expect = r_beam / R_in
        assert np.sign(u) == np.sign(expect)
        assert abs(u - expect) < 0.1 * abs(expect), (u, expect)


# ---------------------------------------------------------------------------
# (a) conic / aspheric r^4 term.
# ---------------------------------------------------------------------------
def test_conic_asphere_changes_estimate_vs_spherical():
    """FAIL-BEFORE / PASS-AFTER: a correcting asphere (A4 opposing the base
    sphere) LOWERS the G1 estimate below the spherical-only value; an
    aberrating asphere RAISES it.  The pre-G1 gate (spherical-only) is blind to
    both."""
    N, dx = 256, 2 * 11.5e-3 / 256
    E = _gauss(N, dx, 6e-3)
    base = {'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 25.8e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_G1_1p5168', 'semi_diameter': 10e-3},
                {'radius': np.inf, 'thickness': 0.0,
                 'glass_before': '_G1_1p5168', 'glass_after': 'air',
                 'semi_diameter': 10e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}
    sph = _sag_screen_aberration_rad(E, dx, dx, base, _WL)
    # c4_sphere = 1/(8R^3); an A4 = -1/(8R^3) nulls the surface r^4 sag -> ~0
    R = 25.8e-3
    A4_null = -1.0 / (8.0 * R ** 3)
    p_null = {**base, 'surfaces': [dict(base['surfaces'][0],
                                        aspheric_coeffs={4: A4_null}),
                                   base['surfaces'][1]]}
    est_null = _sag_screen_aberration_rad(E, dx, dx, p_null, _WL)
    assert est_null < 0.05 * sph, (est_null, sph)          # nulled surface ~0
    # an A4 adding to the base doubles c4 -> ~2x the spherical estimate
    p_add = {**base, 'surfaces': [dict(base['surfaces'][0],
                                       aspheric_coeffs={4: 1.0 / (8.0 * R ** 3)}),
                                  base['surfaces'][1]]}
    est_add = _sag_screen_aberration_rad(E, dx, dx, p_add, _WL)
    assert 1.8 * sph < est_add < 2.2 * sph, (est_add, sph)
    # a conic k<0 (reduces the r^4 sag) lowers the estimate below the sphere
    p_conic = {**base, 'surfaces': [dict(base['surfaces'][0], conic=-0.6),
                                    base['surfaces'][1]]}
    est_conic = _sag_screen_aberration_rad(E, dx, dx, p_conic, _WL)
    assert est_conic < sph, (est_conic, sph)


# ---------------------------------------------------------------------------
# (b) multi-element per-surface height trace.
# ---------------------------------------------------------------------------
def test_multielement_height_trace_below_input_radius_estimate():
    """FAIL-BEFORE / PASS-AFTER: on a cemented doublet the marginal ray
    CONVERGES through the elements, so the true per-surface height is below the
    input radius.  The G1 height-trace estimate is therefore LOWER than the
    pre-G1 input-radius-everywhere sum."""
    presc = {'wavelength': _WL, 'aperture_diameter': 40e-3,
             'surfaces': [
                 {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                  'glass_after': '_G1_1p5168', 'semi_diameter': 20e-3},
                 {'radius': -44.5e-3, 'thickness': 2.5e-3,
                  'glass_before': '_G1_1p5168', 'glass_after': '_G1_1p62',
                  'semi_diameter': 20e-3},
                 {'radius': -129e-3, 'thickness': 0.0,
                  'glass_before': '_G1_1p62', 'glass_after': 'air',
                  'semi_diameter': 20e-3}],
             'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}
    N, dx = 384, 1.15 * 40e-3 / 384
    E = _gauss(N, dx, 8e-3)
    est = _sag_screen_aberration_rad(E, dx, dx, presc, _WL)
    ref = _spherical_only_input_radius(E, dx, presc)
    assert est < ref, (est, ref)               # heights shrink -> lower estimate
    assert est > 0.7 * ref                      # but not wildly (thin doublet)


# ---------------------------------------------------------------------------
# (c) converging / diverging input.
# ---------------------------------------------------------------------------
def test_converging_and_diverging_inputs_differ():
    """The signed input-slope start makes a converging vs a diverging input
    give DIFFERENT per-surface heights (hence estimates) through the same lens
    -- item 1c.  A collimated input sits between them.  Uses a grid that
    RESOLVES the input carrier (small beam, fine dx) so u_in is measured."""
    presc = _biconvex(51.68e-3, 5e-3, 6e-3)     # semi 6mm for the 1mm beam
    N, dx = 768, 8e-6                           # resolves R=+/-60mm @ w0=1mm
    est_col = _sag_screen_aberration_rad(_gauss(N, dx, 1e-3), dx, dx, presc, _WL)
    est_div = _sag_screen_aberration_rad(_gauss(N, dx, 1e-3, R_in=60e-3),
                                         dx, dx, presc, _WL)
    est_con = _sag_screen_aberration_rad(_gauss(N, dx, 1e-3, R_in=-60e-3),
                                         dx, dx, presc, _WL)
    assert est_div != est_con                   # signed slope distinguishes them
    # converging input -> heights shrink faster -> smaller estimate than diverging
    assert est_con < est_div
    assert abs(est_div - est_con) > 1e-2 * est_col


# ---------------------------------------------------------------------------
# Matrix routing: every fast (high-NA) matrix design routes to a ray member;
# the low-NA aberrated corner is steered off phase_screen by the gate.
# ---------------------------------------------------------------------------
def _route(E, presc, opd):
    return _universal_route(E, presc, _WL, _dx_for(presc, E), _dx_for(presc, E),
                            opd, 0.12, 3.0, None, 0.06, _ABERRATION_MAX_RAD)


def _dx_for(presc, E):
    return 1.15 * presc['aperture_diameter'] / E.shape[0]


def test_matrix_fast_designs_never_route_to_phase_screen():
    """M1-M4 (fast, high-NA) and M6 (diverging) must route to a ray member at
    both the vertex and the paraxial image -- never the analytic sag-screen,
    which the displaced measurement proved 58-123% wrong on this regime."""
    designs = {
        'M1': (_doublet(), 8e-3, None, 88.95e-3),
        'M2': (_asphere(), 6e-3, None, 46.63e-3),
        'M4': (_biconvex(20.5e-3, 6e-3, 12e-3), 9e-3, None, 18.79e-3),
        'M6': (_biconvex(51.68e-3, 5e-3, 12e-3), 5e-3, 150e-3, 74.79e-3),
    }
    for name, (presc, w0, R_in, zimg) in designs.items():
        N = 384
        dx = _dx_for(presc, np.empty((N, N)))
        E = _gauss(N, dx, w0, R_in=R_in)
        assert _system_na(presc, _WL) > 0.12, name       # all genuinely fast
        for opd in (0.0, zimg):
            r = _universal_route(E, presc, _WL, dx, dx, opd, 0.12, 3.0,
                                 None, 0.06, _ABERRATION_MAX_RAD)
            assert r != 'phase_screen', (name, opd, r)
            assert r in ('traced', 'fga'), (name, opd, r)


def test_low_na_aberrated_asphere_gate_override():
    """The gate override in its OWN regime (NA < the 0.12 threshold): a low-NA
    lens that WOULD hit the phase_screen shortcut is steered off it when the
    conic/aspheric r^4 term makes it aberrated, and stays on phase_screen when
    a correcting asphere makes it benign -- the conic/aspheric extension
    deciding the route."""
    # slow biconvex (large R) -> low NA; big beam -> aberrated
    presc = _biconvex(120e-3, 4e-3, 12e-3)
    N, dx = 256, 1.15 * 24e-3 / 256
    E = _gauss(N, dx, 10e-3)
    assert _system_na(presc, _WL) < 0.12
    # spherical: aberrated (big beam, moderate R) -> trips -> off phase_screen
    est_sph = _sag_screen_aberration_rad(E, dx, dx, presc, _WL)
    assert est_sph > _ABERRATION_MAX_RAD
    r = _universal_route(E, presc, _WL, dx, dx, 0.3e-3, 0.12, 3.0, None, 0.06,
                         _ABERRATION_MAX_RAD)
    assert r != 'phase_screen'
    # add a strong CORRECTING asphere on BOTH surfaces (null the r^4 sag) ->
    # estimate collapses below budget -> the gate no longer overrides -> the
    # low-NA shortcut (phase_screen) is taken again.
    R = 120e-3
    A4n = -1.0 / (8.0 * R ** 3)
    p_corr = {**presc, 'surfaces': [
        dict(presc['surfaces'][0], aspheric_coeffs={4: A4n}),
        dict(presc['surfaces'][1], aspheric_coeffs={4: -A4n})]}
    est_corr = _sag_screen_aberration_rad(E, dx, dx, p_corr, _WL)
    assert est_corr < est_sph                  # the asphere lowered the estimate
    assert est_corr < _ABERRATION_MAX_RAD      # below budget now
    r2 = _universal_route(E, p_corr, _WL, dx, dx, 0.3e-3, 0.12, 3.0, None, 0.06,
                          _ABERRATION_MAX_RAD)
    assert r2 == 'phase_screen', (est_corr, r2)


def _doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3,
            'surfaces': [
                {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                 'glass_after': '_G1_1p5168', 'semi_diameter': 20e-3},
                {'radius': -44.5e-3, 'thickness': 2.5e-3,
                 'glass_before': '_G1_1p5168', 'glass_after': '_G1_1p62',
                 'semi_diameter': 20e-3},
                {'radius': -129e-3, 'thickness': 0.0, 'glass_before': '_G1_1p62',
                 'glass_after': 'air', 'semi_diameter': 20e-3}],
            'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _asphere():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 25.8e-3, 'conic': -0.6, 'aspheric_coeffs': {4: 4.0e3},
                 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_G1_1p5168', 'semi_diameter': 10e-3},
                {'radius': np.inf, 'thickness': 0.0,
                 'glass_before': '_G1_1p5168', 'glass_after': 'air',
                 'semi_diameter': 10e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
