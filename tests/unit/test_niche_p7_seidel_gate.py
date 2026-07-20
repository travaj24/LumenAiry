"""P7 (niche N8): Seidel-based true-spherical-aberration dispatcher gate.

The pre-P7 aberration gate (``_sag_screen_aberration_rad``) is a per-surface
``k * h^4 * 4 * |c4|`` MAGNITUDE bound on the THIN sag-screen's obliquity error.
It cannot see cross-surface cancellation, so an aplanatic conic / balanced asphere
whose SYSTEM spherical aberration is nulled still trips it -- a FALSE POSITIVE that
is safe but pays for the slow ray members.

P7 adds ``_seidel_sa_wfe_rad``: the SIGNED system Seidel S1 (third-order spherical)
sum along the SAME paraxial marginal-ray trace, at the field's measured input
congruence, including conic / aspheric contributions.  A nulled system reads ~0.
When -- at LOW NA, where the 3rd-order term dominates -- a c4 trip is paired with a
Seidel W_rms inside a conservative near-diffraction-limited budget (lambda/20 RMS,
Strehl ~0.91) AND every surface was classifiable AND the element is within the
displaced envelope, the router reclaims
the plane onto the obliquity-correct ``'displaced'`` analytic model (the thin
screen mis-renders the strong corrected sag departure; displaced does not).

INDEPENDENT re-check (the audit's core worry -- internal agreement is not truth):
the base-sphere S1 reproduces ``lumenairy.seidel_coefficients`` to machine
precision, and the base+conic/aspheric total reproduces the r^4 coefficient of a
lumenairy-free exact ray-traced wavefront (``validation/oracles/debye_oracle_v3.py``
``wavefront_r4``) to <2% on the near-paraxial designs.  (An f/0.85 extreme where
3rd-order theory itself breaks down disagrees by ~16-20%, but that regime is high
NA and never reaches this low-NA gate.)

SAFETY (measured, not asserted from internal estimates): the SA-nulled conic routes
to displaced AND the displaced field is within ~2% of the diffraction oracle; every
genuinely-aberrated M1-M6 design routes to a ray member.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

from lumenairy.glass import GLASS_REGISTRY
from lumenairy.propagators.fga import (
    _ABERRATION_MAX_RAD,
    _SEIDEL_RMS_C,
    _SEIDEL_SA_MAX_RAD,
    _displaced_compatible,
    _sag_screen_aberration_rad,
    _seidel_sa_wfe_rad,
    _system_na,
    _universal_route,
    apply_real_lens_universal,
)
from lumenairy.raytrace import seidel_coefficients, surfaces_from_prescription

_WL = 1.31e-6
K = 2.0 * np.pi / _WL
for _n, _v in {'_P7_1p5168': 1.5168, '_P7_1p62': 1.62, '_P7_1p70': 1.70}.items():
    GLASS_REGISTRY.setdefault(_n, (lambda vv: (lambda wl: vv))(_v))

# ---- lumenairy-free oracle (installed, out of default pytest collection) ---
_ORACLE_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / 'validation' / 'oracles' / 'debye_oracle_v3.py')
_spec = importlib.util.spec_from_file_location('debye_oracle_v3', _ORACLE_PATH)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


# ---------------------------------------------------------------------------
# M1-M6 designs (the plan's gate-decision matrix)
# ---------------------------------------------------------------------------
def M1_doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3, 'surfaces': [
        {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
         'glass_after': '_P7_1p5168', 'semi_diameter': 20e-3},
        {'radius': -44.5e-3, 'thickness': 2.5e-3, 'glass_before': '_P7_1p5168',
         'glass_after': '_P7_1p62', 'semi_diameter': 20e-3},
        {'radius': -129e-3, 'thickness': 0.0, 'glass_before': '_P7_1p62',
         'glass_after': 'air', 'semi_diameter': 20e-3}],
        'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def M2_asphere():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 25.8e-3, 'conic': -0.6, 'aspheric_coeffs': {4: 4.0e3},
         'thickness': 5e-3, 'glass_before': 'air', 'glass_after': '_P7_1p5168',
         'semi_diameter': 10e-3},
        {'radius': np.inf, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def M3_meniscus():
    return {'wavelength': _WL, 'aperture_diameter': 16e-3, 'surfaces': [
        {'radius': 18e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': '_P7_1p70', 'semi_diameter': 8e-3},
        {'radius': 30e-3, 'thickness': 0.0, 'glass_before': '_P7_1p70',
         'glass_after': 'air', 'semi_diameter': 8e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}


def M4_fastbiconvex():
    R, t, semi = 20.5e-3, 6e-3, 12e-3
    return {'wavelength': _WL, 'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': R, 'thickness': t, 'glass_before': 'air',
         'glass_after': '_P7_1p5168', 'semi_diameter': semi},
        {'radius': -R, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


def M5_biconcave():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': -50e-3, 'thickness': 3e-3, 'glass_before': 'air',
         'glass_after': '_P7_1p5168', 'semi_diameter': 10e-3},
        {'radius': 50e-3, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [3e-3], 'stop_index': 0}


def M6_f5singlet(semi=6e-3):
    R, t = 51.68e-3, 5e-3
    return {'wavelength': _WL, 'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': R, 'thickness': t, 'glass_before': 'air',
         'glass_after': '_P7_1p5168', 'semi_diameter': semi},
        {'radius': -R, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


_MATRIX = {'M1': (M1_doublet, 8e-3), 'M2': (M2_asphere, 4e-3),
           'M3': (M3_meniscus, 3e-3), 'M4': (M4_fastbiconvex, 8e-3),
           'M5': (M5_biconcave, 4e-3), 'M6': (M6_f5singlet, 5e-3)}

# Low-NA biconvex + the SA-nulling front CONIC (oracle root-found; grid-
# independent because it is a collimated ray property).  conic that zeroes the
# 3rd-order system spherical aberration at the collimated conjugate:
_LOWNA_R, _LOWNA_T, _LOWNA_SEMI, _LOWNA_W0 = 60e-3, 4e-3, 6e-3, 4e-3
_CONIC_NULL = -6.4611


def lowna_biconvex(conic_front=0.0):
    R, t, semi = _LOWNA_R, _LOWNA_T, _LOWNA_SEMI
    return {'wavelength': _WL, 'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': R, 'conic': conic_front, 'thickness': t, 'glass_before': 'air',
         'glass_after': '_P7_1p5168', 'semi_diameter': semi},
        {'radius': -R, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [t], 'stop_index': 0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _n(g):
    return 1.0 if g == 'air' else float(GLASS_REGISTRY[g](_WL))


def _gauss(N, dx, w0, R_in=None):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
    if R_in is None:
        return amp.astype(np.complex128)
    return (amp * np.exp(1j * K * (X ** 2 + Y ** 2) / (2 * R_in))).astype(np.complex128)


def _r_beam(E, dx):
    a2 = np.abs(E) ** 2
    N = E.shape[0]
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return math.sqrt(2.0 * float(np.sum(a2 * (X ** 2 + Y ** 2)) / a2.sum()))


def _gate_a4(E, dx, presc):
    """The gate's IMPLIED r^4 wavefront coefficient (magnitude) -- |S1|/(8 r^4)
    with |S1| recovered from the returned W_rms.  Normalization-free (per-h^4)."""
    wfe, _cl = _seidel_sa_wfe_rad(E, dx, dx, presc, _WL)
    rb = _r_beam(E, dx)
    s1 = wfe / (K * _SEIDEL_RMS_C)
    return s1 / (8.0 * rb ** 4)


def _paraxial_image_lastvertex(presc, R_in):
    """Paraxial image distance from the LAST surface vertex [m] (marginal y-nu)."""
    surfs, th = presc['surfaces'], presc['thicknesses']
    h, u = 1.0, (1.0 / R_in if R_in is not None else 0.0)
    for i, s in enumerate(surfs):
        R = s['radius']
        n1, n2 = _n(s['glass_before']), _n(s['glass_after'])
        phi = 0.0 if (R == 0 or not np.isfinite(R)) else (n2 - n1) / R
        u = (n1 * u - h * phi) / n2
        if i < len(surfs) - 1:
            h = h + u * th[i]
    return -h / u


def _oracle_surfs(presc, z_img_lastvtx=None):
    """Oracle surface list; the LAST surface's thickness carries the (last-vertex
    -> image) distance when ``z_img_lastvtx`` is given (the oracle's convention)."""
    surfs = []
    for i, s in enumerate(presc['surfaces']):
        if i == len(presc['surfaces']) - 1 and z_img_lastvtx is not None:
            t = z_img_lastvtx
        else:
            t = presc['thicknesses'][i] if i < len(presc['thicknesses']) else 0.0
        idx = _n(s['glass_after'])
        asph = s.get('aspheric_coeffs') or {}
        surfs.append({'radius_mm': (s['radius'] * 1e3 if np.isfinite(s['radius'])
                                    and s['radius'] != 0 else 0),
                      'thickness_mm': t * 1e3,
                      'conic': float(s.get('conic', 0.0) or 0.0),
                      'aspheric_coeffs_si': {str(int(p)): float(a)
                                             for p, a in asph.items()},
                      'index': 'air' if idx == 1.0 else idx})
    return surfs


def _oracle_a4(presc, R_in, w0):
    job = {'wavelength_um': _WL * 1e6, 'r_beam_mm': w0 * 1e3,
           'R_in_mm': None if R_in is None else R_in * 1e3,
           'pop': {'w0_mm': w0 * 1e3}, 'surfaces': _oracle_surfs(presc)}
    return _oracle.wavefront_r4(job)


def _route(E, presc, opd, thr=_SEIDEL_SA_MAX_RAD):
    dx = 1.15 * presc['aperture_diameter'] / E.shape[0]
    return _universal_route(E, presc, _WL, dx, dx, opd, 0.12, 3.0, None, 0.06,
                            _ABERRATION_MAX_RAD, thr)


# ===========================================================================
# ESTIMATOR CORRECTNESS -- independent re-check vs the oracle wavefront r^4
# ===========================================================================
def test_base_sphere_s1_matches_seidel_coefficients():
    """The gate's implied r^4 coefficient for a plain-spherical design reproduces
    lumenairy.seidel_coefficients (both are the third-order S1) to <0.5% -- the
    base-sphere walk is byte-faithful to the trusted Seidel machinery."""
    for fn, w0 in [(M1_doublet, 4e-3), (M6_f5singlet, 4e-3),
                   (M4_fastbiconvex, 6e-3)]:
        presc = fn()
        N = 512
        dx = 1.15 * presc['aperture_diameter'] / N
        E = _gauss(N, dx, w0)
        a4_gate = _gate_a4(E, dx, presc)
        surfs = surfaces_from_prescription(presc)
        sc, _ = seidel_coefficients(surfs, _WL, object_distance=np.inf)
        r_stop = surfs[0].semi_diameter
        a4_sc = abs(float(sc['total']['S1'])) / (8.0 * r_stop ** 4)
        assert abs(a4_gate - a4_sc) / a4_sc < 5e-3, (fn.__name__, a4_gate, a4_sc)


def test_seidel_wrms_matches_oracle_r4_collimated():
    """INDEPENDENT re-check: the gate's Seidel W_rms reproduces the r^4 coefficient
    of the lumenairy-free exact ray-traced wavefront (debye_oracle_v3.wavefront_r4)
    to <2% on the near-paraxial designs (M1/M2/M3/M6, collimated).  This is the
    audit's demanded 'internal agreement is not truth' gate -- the estimator is
    validated against an oracle, not against itself."""
    for name, (fn, _w) in _MATRIX.items():
        if name in ('M4', 'M5'):        # M4 high-NA (below); M5 virtual image
            continue
        presc, w0 = fn(), 3e-3
        N = 640
        dx = 1.15 * presc['aperture_diameter'] / N
        E = _gauss(N, dx, w0)
        a4_gate = abs(_gate_a4(E, dx, presc))
        a4_orc = abs(_oracle_a4(presc, None, w0)['a4'])
        assert abs(a4_gate - a4_orc) / a4_orc < 0.02, (name, a4_gate, a4_orc)


def test_aspheric_sign_and_magnitude_track_oracle():
    """The conic / aspheric r^4 term enters S1 with the CORRECT sign: a conic k<0
    (or an opposing A4) that REDUCES the true spherical aberration also reduces the
    gate estimate, tracking the oracle; a same-sign A4 raises both.  Kills a sign
    flip in the aspheric contribution (which would break cancellation)."""
    w0, N = 2e-3, 512
    base = M2_asphere()
    base['surfaces'][0]['conic'] = 0.0
    base['surfaces'][0]['aspheric_coeffs'] = None       # plain sphere R=25.8
    dx = 1.15 * base['aperture_diameter'] / N
    E = _gauss(N, dx, w0)
    R = 25.8e-3

    def variant(conic, A4):
        p = M2_asphere()
        p['surfaces'][0]['conic'] = conic
        p['surfaces'][0]['aspheric_coeffs'] = {4: A4} if A4 else None
        return p

    g0 = abs(_gate_a4(E, dx, base))
    o0 = abs(_oracle_a4(base, None, w0)['a4'])
    # conic k<0 REDUCES the surface r^4 sag -> lowers spherical -> both drop
    p_c = variant(-0.6, 0.0)
    assert abs(_gate_a4(E, dx, p_c)) < g0
    assert abs(_oracle_a4(p_c, None, w0)['a4']) < o0
    # a same-sign A4 (adds to the base r^4 sag c4=1/(8R^3)) RAISES both ~2x
    p_a = variant(0.0, 1.0 / (8.0 * R ** 3))
    assert abs(_gate_a4(E, dx, p_a)) > 1.6 * g0
    assert abs(_oracle_a4(p_a, None, w0)['a4']) > 1.6 * o0
    # and the gate tracks the oracle magnitude within 3% on each aspheric variant
    for p in (base, p_c, p_a):
        gg = abs(_gate_a4(E, dx, p))
        oo = abs(_oracle_a4(p, None, w0)['a4'])
        assert abs(gg - oo) / oo < 0.03, (gg, oo)


def test_extreme_na_underestimate_is_high_na_out_of_gate_scope():
    """On an f/0.85 extreme (M4) the 3rd-order Seidel UNDER-estimates the exact r^4
    coefficient by ~16-20% (higher-order SA the estimator cannot see), the honest
    trust limit.  It is harmless here: M4 is high NA (> the 0.12 gate), so the
    aberration gate is never consulted -- it routes to rays on the NA branch."""
    presc, w0 = M4_fastbiconvex(), 8e-3
    N = 640
    dx = 1.15 * presc['aperture_diameter'] / N
    E = _gauss(N, dx, w0)
    a4_gate = abs(_gate_a4(E, dx, presc))
    a4_orc = abs(_oracle_a4(presc, None, w0)['a4'])
    assert a4_gate < a4_orc                       # 3rd-order underestimates
    assert 0.75 < a4_gate / a4_orc < 0.90         # ~16-20% low
    assert _system_na(presc, _WL) > 0.12          # high NA -> gate not consulted


def test_unclassifiable_surface_falls_back_to_c4():
    """A surface the Seidel walk cannot classify (a mirror) sets all_classified
    False AND retains the conservative c4 magnitude bound as its contribution --
    so the rescue is refused (the plan's c4 fallback)."""
    p = {'wavelength': _WL, 'aperture_diameter': 20e-3, 'surfaces': [
        {'radius': 50e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': 'MIRROR', 'semi_diameter': 10e-3},
        {'radius': -50e-3, 'thickness': 0.0, 'glass_before': 'air',
         'glass_after': 'air', 'semi_diameter': 10e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}
    N, dx = 256, 1.15 * 20e-3 / 256
    E = _gauss(N, dx, 4e-3)
    wfe, classified = _seidel_sa_wfe_rad(E, dx, dx, p, _WL)
    assert classified is False
    assert wfe > 0.0                              # c4 fallback contributed
    assert _displaced_compatible(p) is False      # mirror blocks displaced route


def test_empty_and_malformed_never_rescue():
    assert _seidel_sa_wfe_rad(np.zeros((8, 8), complex), 1e-6, 1e-6,
                              M6_f5singlet(), _WL) == (0.0, False)
    E = _gauss(128, 1e-4, 4e-3)
    assert _seidel_sa_wfe_rad(E, 1e-4, 1e-4, {'surfaces': 'bogus'}, _WL) == (0.0, False)


def test_displaced_compatibility_guard():
    assert _displaced_compatible(M2_asphere()) is True
    assert _displaced_compatible(M1_doublet()) is True
    bad_dec = M6_f5singlet()
    bad_dec['surfaces'][0]['decenter'] = (0.5e-3, 0.0)
    assert _displaced_compatible(bad_dec) is False
    bad_bic = M6_f5singlet()
    bad_bic['surfaces'][0]['radius_y'] = 40e-3
    assert _displaced_compatible(bad_bic) is False


# ===========================================================================
# HEADLINE (plan gate (c)): SA-nulled conic routes FAST; non-nulled -> rays
# ===========================================================================
def test_sa_nulled_conic_routes_to_displaced_non_nulled_to_rays():
    """The false-positive the c4 bound cannot resolve: a low-NA biconvex whose
    front CONIC zeroes the 3rd-order system SA has LARGE per-surface |c4| (the c4
    bound trips) but ~0 SYSTEM spherical aberration.  The Seidel gate reclaims it
    onto the fast 'displaced' analytic model, while the non-nulled twin (same base,
    conic=0) stays on the ray members."""
    nulled = lowna_biconvex(_CONIC_NULL)
    twin = lowna_biconvex(0.0)
    N = 512
    dx = 1.15 * nulled['aperture_diameter'] / N
    E = _gauss(N, dx, _LOWNA_W0)
    # both are low NA and both TRIP the c4 obliquity bound (false positive):
    assert _system_na(nulled, _WL) < 0.12
    assert _sag_screen_aberration_rad(E, dx, dx, nulled, _WL) > _ABERRATION_MAX_RAD
    assert _sag_screen_aberration_rad(E, dx, dx, twin, _WL) > _ABERRATION_MAX_RAD
    # Seidel discriminates: nulled ~0 (< Marechal budget), twin well above it:
    wn, cn = _seidel_sa_wfe_rad(E, dx, dx, nulled, _WL)
    wt, ct = _seidel_sa_wfe_rad(E, dx, dx, twin, _WL)
    assert cn and ct
    assert wn < _SEIDEL_SA_MAX_RAD < wt
    # routing at the vertex AND the paraxial image:
    for opd in (0.0, 57.4e-3):
        assert _route(E, nulled, opd) == 'displaced', opd
        assert _route(E, twin, opd) in ('traced', 'fga'), opd


def test_seidel_rescue_only_at_low_na():
    """The rescue is LOW-NA only (where 3rd-order dominates; 5th-order ~ NA^6 is
    negligible).  The same nulled figure scaled to HIGH NA is NOT rescued to
    displaced -- it takes the ray members like any high-NA design."""
    # scale the nulled biconvex to high NA (short focal / big aperture)
    hp = {'wavelength': _WL, 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': 20e-3, 'conic': _CONIC_NULL, 'thickness': 6e-3,
         'glass_before': 'air', 'glass_after': '_P7_1p5168', 'semi_diameter': 12e-3},
        {'radius': -20e-3, 'thickness': 0.0, 'glass_before': '_P7_1p5168',
         'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [6e-3], 'stop_index': 0}
    N = 512
    dx = 1.15 * hp['aperture_diameter'] / N
    E = _gauss(N, dx, 8e-3)
    assert _system_na(hp, _WL) > 0.12
    assert _route(E, hp, 0.0) != 'displaced'
    assert _route(E, hp, 0.0) in ('traced', 'fga')


def test_false_positive_reduction_vs_c4_bound():
    """MEASURED false-positive reduction: over the SA-null pair the c4 bound trips
    (routes to rays) on BOTH, while the Seidel gate reclaims the nulled one to the
    fast analytic model -- exactly one now-correctly-fast case, zero new rays."""
    N = 512
    n_flipped = 0
    for conic, expect in ((_CONIC_NULL, 'displaced'), (0.0, 'rays')):
        p = lowna_biconvex(conic)
        dx = 1.15 * p['aperture_diameter'] / N
        E = _gauss(N, dx, _LOWNA_W0)
        # c4 bound alone would route BOTH to rays (it trips on both):
        assert _sag_screen_aberration_rad(E, dx, dx, p, _WL) > _ABERRATION_MAX_RAD
        r = _route(E, p, 0.0)
        if r == 'displaced':
            n_flipped += 1
            assert expect == 'displaced'
        else:
            assert r in ('traced', 'fga') and expect == 'rays'
    assert n_flipped == 1


# ===========================================================================
# SAFETY on the M1-M6 x {collim, div, conv} matrix (pinned regression)
# ===========================================================================
@pytest.mark.parametrize('name', list(_MATRIX))
def test_matrix_aberrated_designs_never_route_to_analytic(name):
    """SAFETY pin: every genuinely-aberrated M1-M6 design (high NA and/or large
    system SA) routes to a RAY member at the vertex and near its image -- NEVER the
    thin 'phase_screen' NOR the Seidel-rescued 'displaced'.  The Seidel gate does
    not wrongly reclaim any aberrated matrix design."""
    fn, w0 = _MATRIX[name]
    presc = fn()
    N = 448
    dx = 1.15 * presc['aperture_diameter'] / N
    for R_in in (None, 150e-3, -150e-3):
        E = _gauss(N, dx, w0, R_in)
        for opd in (0.0, 30e-3, 80e-3):
            r = _route(E, presc, opd)
            assert r in ('traced', 'fga', 'gbd'), (name, R_in, opd, r)


def test_defaults_unchanged_on_benign_and_high_na():
    """The Seidel gate is additive: the benign low-NA path still routes to the thin
    'phase_screen' (byte-identical dispatch), and a high-NA aberrated design still
    routes to rays -- the pre-P7 decisions are preserved."""
    # benign low-NA small beam -> phase_screen (thin), unchanged
    benign = M6_f5singlet(semi=5e-3)
    N, dx = 256, 1.15 * benign['aperture_diameter'] / 256
    Eb = _gauss(N, dx, 0.5e-3)
    assert _system_na(benign, _WL) < 0.12
    assert _sag_screen_aberration_rad(Eb, dx, dx, benign, _WL) < _ABERRATION_MAX_RAD
    assert _route(Eb, benign, 0.0) == 'phase_screen'
    # high-NA aberrated -> rays (unchanged; Seidel gate not consulted at high NA)
    fast = M4_fastbiconvex()
    dx2 = 1.15 * fast['aperture_diameter'] / N
    Ef = _gauss(N, dx2, 8e-3)
    assert _route(Ef, fast, 0.0) in ('traced', 'fga')


# ===========================================================================
# SLOW -- oracle-backed SAFETY: the rescued displaced field is actually accurate
# ===========================================================================
@pytest.mark.slow
def test_sa_null_genuinely_nulled_and_displaced_route_accurate():
    """The SAFETY property the plan demands, measured against the oracle (not the
    internal estimate): (1) the baked conic genuinely nulls the wavefront -- the
    lumenairy-free oracle r^4 coefficient is < 1% of the non-nulled twin's; and
    (2) the plane the Seidel gate reclaims to 'displaced' is within 10% of the
    diffraction oracle EE80 -- so the fast analytic route is accurate, not a silent
    wrong answer.  (The thin sag-screen the pre-P7 'phase_screen' member would use
    reads ~40% here -- routing to displaced, not thin, is load-bearing.)"""
    nulled = lowna_biconvex(_CONIC_NULL)
    twin = lowna_biconvex(0.0)
    w0 = _LOWNA_W0
    # (1) independent oracle confirms the null
    a4_n = abs(_oracle_a4(nulled, None, w0)['a4'])
    a4_t = abs(_oracle_a4(twin, None, w0)['a4'])
    assert a4_n < 0.01 * a4_t, (a4_n, a4_t)

    # (2) end-to-end: apply_real_lens_universal picks displaced and is accurate.
    # The dispatcher returns the field at the LAST vertex -> propagate by the
    # last-vertex image distance (not the front-vertex distance).
    N, dx = 4096, 6e-6                        # hw=12.29mm=3.07 w0; Nyquist ok
    z_img = _paraxial_image_lastvertex(nulled, None)
    E0 = _gauss(N, dx, w0)
    Eexit, member = apply_real_lens_universal(
        E0, prescription=nulled, wavelength=_WL, dx=dx, return_method=True)
    assert member == 'displaced'
    job = {'wavelength_um': _WL * 1e6, 'R_in_mm': None, 'window_um': 200.0,
           'rho_max_um': 400.0, 'pop': {'w0_mm': w0 * 1e3},
           'surfaces': _oracle_surfs(nulled, z_img_lastvtx=z_img)}
    huy = _oracle.evaluate(job)['huy_EE80_um'] * 1e-6
    ee = _wave_ee80(np.asarray(Eexit), dx, z_img)
    assert abs(ee - huy) / huy < 0.10, (ee, huy, member)


def _wave_ee80(E_exit, dx, z_img, win=200e-6):
    import lumenairy as la
    E = la.angular_spectrum_propagate(E_exit, z_img, _WL, dx)
    I = np.abs(E) ** 2
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    j, i = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(x - x[i], x - x[j])
    r = np.sqrt(X ** 2 + Y ** 2)
    m = r <= win
    Iw, rw, Pw = I[m], r[m], I[m].sum()
    rb = np.linspace(0, win, 601)
    cum = np.array([Iw[rw <= rr].sum() for rr in rb]) / Pw
    return float(np.interp(0.8, cum, rb))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
