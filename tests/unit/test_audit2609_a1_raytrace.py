"""WP-A1 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §4 -- ray-tracing
findings R1..R7.

Each section states its ORACLE, the PRE-FIX measured value (so the test
demonstrably fails on the old code), and the derivation of its bar.

* **R1** ``opd_fan_data`` had no reference sphere: wrong SIGN and 3.1x the
  magnitude at f/4.
* **R2** off-axis fans carried a spurious ``-y sin(theta)`` launch-plane
  tilt (1880 waves of artefact against 38 waves of real aberration at 5
  deg).
* **R3** ``seidel_coefficients`` ignored ``conic`` and every aspheric
  coefficient (a parabolic mirror reported -12.2 um of spherical).
* **R4** the conic intersection used the ray-SPHERE discriminant as its
  miss test, killing every ray with ``h > |R|`` on a conic.
* **R5** the diffraction-order kick omitted the medium index (50 %
  direction error into N-BK7).
* **R6** ``rays_from_field`` aliased above HALF the grid Nyquist and gave
  edge rays half the direction cosine; ``_transfer`` teleported grazing
  rays.
* **R7** the P3 bundle.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.glass import get_glass_index
from lumenairy.raytrace import (
    RAY_EVANESCENT,
    RAY_MISSED_SURFACE,
    RAY_OK,
    Surface,
    apply_doe_phase_traced,
    field_of_view,
    opd_fan_data,
    rays_from_field,
    raytrace_system,
    seidel_coefficients,
    trace,
    trace_summary,
)
from lumenairy.raytrace.intersection import _transfer
from lumenairy.raytrace.surface import RayBundle

WL = 587.6e-9
WL_IR = 1.31e-6


def _axial_bundle(ys, wl=WL):
    n = len(ys)
    return RayBundle(x=np.zeros(n), y=np.asarray(ys, float), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=wl, alive=np.ones(n, bool), opd=np.zeros(n))


# ===========================================================================
# R1 -- the reference sphere
# ===========================================================================

def _exact_plano_convex_wfe(R1, t, n, sd, n_rays=41):
    """INDEPENDENT exact on-axis wavefront error of a plano-convex singlet.

    Written straight from the ray equations -- ray/sphere intersection,
    vector Snell, straight leg to the Gaussian image point (Welford §4's
    "optical path from the object to the Gaussian image point") -- and
    shares no code with ``lumenairy.raytrace``.  The paraxial focus uses
    the thick-lens closed forms ``f = R1/(n-1)`` and
    ``BFL = f (1 - (n-1) t / (n R1))``, which the audit verified
    ``system_abcd`` against to 0.00e+00.

    Returns ``(y0, W_waves, f, bfl)``.
    """
    f = R1 / (n - 1.0)
    bfl = f * (1.0 - (n - 1.0) * t / (n * R1))
    y0 = np.linspace(-sd, sd, n_rays)
    # Surface 1: sphere of radius R1, vertex at z = 0, centre (0, 0, R1).
    t1 = R1 - np.sign(R1) * np.sqrt(R1 ** 2 - y0 ** 2)
    y1, z1 = y0, t1
    opl = 1.0 * t1
    ny, nz = y1 / R1, (z1 - R1) / R1              # unit outward normal
    dM, dN = np.zeros_like(y0), np.ones_like(y0)  # incoming direction
    mu = 1.0 / n
    ci = -(dM * ny + dN * nz)
    ct = np.sqrt(1.0 - mu ** 2 * (1.0 - ci ** 2))
    M1 = mu * dM + (mu * ci - ct) * ny
    N1 = mu * dN + (mu * ci - ct) * nz
    # Surface 2: flat, at z = t.
    t2 = (t - z1) / N1
    y2 = y1 + M1 * t2
    opl = opl + n * t2
    # Straight leg from the exit face to the Gaussian image point.
    opl = opl + np.sqrt(y2 ** 2 + bfl ** 2)
    return y0, (opl - opl[n_rays // 2]) / WL, f, bfl


def _plano_convex_surfaces(R1, t, sd, bfl):
    return [Surface(radius=R1, semi_diameter=sd, glass_before='air',
                    glass_after='N-BK7', thickness=t, is_stop=True),
            Surface(radius=np.inf, semi_diameter=sd, glass_before='N-BK7',
                    glass_after='air', thickness=bfl),
            Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                    glass_after='air', label='image')]


@pytest.mark.parametrize('sd, tag, prefix_value, prefix_error', [
    (12.5e-3, 'f/4', +36.1936, 47.9073),
    (2.5e-3, 'f/20', +0.0545, 0.0726),
    (1.0e-3, 'f/50', +0.0014, 0.0019),
])
def test_r1_opd_fan_matches_an_independent_exact_wavefront(
        sd, tag, prefix_value, prefix_error):
    """``opd_fan_data`` == the exact reference-sphere wavefront error.

    ORACLE: :func:`_exact_plano_convex_wfe` above (independent code).

    PRE-FIX (repro/RAYTRACE/p8_opdfan.py, p9_opdfan_confirm.py): the
    function returned the OPL to each ray's own intercept, i.e.
    ``+36.194`` waves at f/4 where the truth is ``-11.714`` -- the wrong
    SIGN and 3.1x the magnitude, the residual matching ``eps sin(theta')``
    to 3 decimals at f/20 (0.0726 w) and f/50 (0.0019 w).

    BAR: |lib - oracle| < 0.05 waves.  Derivation -- the two use different
    (both legitimate) reference-sphere tangent points: the library's
    sphere passes through the EXIT PUPIL, the oracle's leg starts at the
    last refracting surface.  They differ by the second-order
    ``n eps^2 / 2 * (1/R_last - 1/R_xp)``, which for this fixture is
    0.010 waves at f/4.  Measured: 0.0067 (f/4), 1.4e-6 (f/20), 3e-10
    (f/50) -- the bar sits 0.9 decades above the worst measurement and
    3.5 decades BELOW the pre-fix error (47.9 waves at f/4).
    """
    t = 3.6e-3
    n = float(get_glass_index('N-BK7', WL))
    y0, W, f, bfl = _exact_plano_convex_wfe(51.68e-3, t, n, sd)
    surfs = _plano_convex_surfaces(51.68e-3, t, sd, bfl)
    py, opd_y, px, opd_x = opd_fan_data(surfs, WL, sd, 0.0, 41)

    lib = float(opd_y[-1])
    ora = float(W[-1])
    assert abs(lib - ora) < 0.05, (
        f'{tag}: opd_fan_data {lib:+.6f} w vs exact oracle {ora:+.6f} w')
    # And demonstrably NOT the pre-fix value.
    assert abs(lib - prefix_value) > 0.5 * prefix_error, (
        f'{tag}: still reporting the pre-fix (no-reference-sphere) value '
        f'{prefix_value:+.4f} w')
    if tag == 'f/4':
        # The headline: the SIGN was wrong, not just the magnitude.
        assert lib < 0.0 and prefix_value > 0.0


def test_r1_cross_checks_against_the_seidel_minus_s1_over_8_relation():
    """The corrected wavefront agrees with the INDEPENDENT Seidel sum.

    ORACLE: third-order theory gives ``W(rho=1) = -S1/8`` in this
    module's sign convention (``code = -S_Welford``), computed by
    :func:`seidel_coefficients`, which shares no code with the ray fans.

    BAR: |W_ray - (-S1/8)/lambda| < 1.0 wave on an 11.7-wave total, i.e.
    < 9 %.  Derivation -- the gap is genuine FIFTH-order aberration:
    a rho^2/rho^4/rho^6 fit of the exact oracle gives a6 = -0.41 waves
    and a2 = -0.003 waves, so third-order theory cannot be closer than
    ~0.4 waves here.  Measured gap 0.402 waves.  Pre-fix the same
    comparison was off by 47.5 waves (a factor 118 outside this bar) and
    of the opposite sign.
    """
    t, sd = 3.6e-3, 12.5e-3
    n = float(get_glass_index('N-BK7', WL))
    _, _, f, bfl = _exact_plano_convex_wfe(51.68e-3, t, n, sd)
    surfs = _plano_convex_surfaces(51.68e-3, t, sd, bfl)
    py, opd_y, _, _ = opd_fan_data(surfs, WL, sd, 0.0, 41)
    sd_dict, _ = seidel_coefficients(surfs, WL, field_angle=1e-6)
    w_seidel = -sd_dict['total']['S1'] / 8.0 / WL
    assert abs(float(opd_y[-1]) - w_seidel) < 1.0
    assert np.sign(float(opd_y[-1])) == np.sign(w_seidel)


def test_r1_aberration_free_parabola_is_unchanged():
    """CONTROL: the function was right exactly when it did not matter.

    An on-axis parabolic mirror is stigmatic (eps == 0), so the missing
    reference sphere contributed nothing.  Pre-fix PV = 7.085e-11 waves;
    BAR 1e-9 waves -- three decades of headroom over both the pre- and
    post-fix value (measured 7.012e-11), and 11 decades below the
    +64.502-wave error the same code produced on the SPHERICAL mirror.
    """
    surfs = [Surface(radius=-200e-3, conic=-1.0, semi_diameter=25e-3,
                     glass_before='air', glass_after='air', is_mirror=True,
                     is_stop=True, thickness=-100e-3),
             Surface(radius=np.inf, semi_diameter=np.inf,
                     glass_before='air', glass_after='air')]
    py, opd_y, _, _ = opd_fan_data(surfs, WL, 25e-3, 0.0, 21)
    assert float(np.nanmax(opd_y) - np.nanmin(opd_y)) < 1e-9


def test_r1_reference_sphere_radius_kwarg_is_validated():
    surfs = _plano_convex_surfaces(51.68e-3, 3.6e-3, 12.5e-3, 97.6e-3)
    for bad in (-1.0, 0.0, np.nan):
        with pytest.raises(ValueError, match='reference_sphere_radius'):
            opd_fan_data(surfs, WL, 12.5e-3, 0.0, 11,
                         reference_sphere_radius=bad)
    # np.inf selects the reference-PLANE limit and must still run.
    _, w, _, _ = opd_fan_data(surfs, WL, 12.5e-3, 0.0, 11,
                              reference_sphere_radius=np.inf)
    assert np.all(np.isfinite(w))


# ===========================================================================
# R2 -- the off-axis launch-plane tilt
# ===========================================================================

@pytest.mark.parametrize('fa_deg, prefix_linear', [
    (0.5, -185.64), (2.0, -744.06), (5.0, -1879.86)])
def test_r2_offaxis_fan_carries_no_launch_plane_tilt(fa_deg, prefix_linear):
    """The fitted LINEAR term of an off-axis OPD fan must be ~0.

    ORACLE: a linear term in an OPD fan is tilt, i.e. a transverse image
    shift, and the fan is referenced to a chief ray of its own
    orientation -- so it is zero by construction for a correctly
    referenced wavefront.  The pre-fix value was exactly
    ``-y_max sin(theta)/lambda`` (ratio -1.0000 at 0.5 deg), which
    identifies the mechanism: ``make_fan`` launched on the z = 0 PLANE
    with ``opd = 0`` instead of on the incident WAVEFRONT.

    BAR: |linear| < 1.0 wave.  Derivation -- the residual is the
    higher-order coupling between the eikonal seed and the real
    aberration, which grows as theta^3; measured +0.01 / +0.06 / +0.15
    waves at 0.5 / 2 / 5 deg, so the bar has 0.8 decades of headroom at
    the worst field and sits 3.3 decades below the pre-fix artefact.
    """
    t, sd = 3.6e-3, 12.5e-3
    n = float(get_glass_index('N-BK7', WL))
    _, _, _, bfl = _exact_plano_convex_wfe(51.68e-3, t, n, sd)
    surfs = _plano_convex_surfaces(51.68e-3, t, sd, bfl)
    fa = np.radians(fa_deg)
    py, opd_y, _, _ = opd_fan_data(surfs, WL, sd, fa, 41)
    ok = np.isfinite(opd_y)
    c = np.polyfit(py[ok], opd_y[ok], 4)
    linear = float(c[3])
    assert abs(linear) < 1.0, (
        f'field {fa_deg} deg: fan still carries {linear:+.2f} waves of '
        f'launch-plane tilt (pre-fix {prefix_linear:+.2f})')
    assert abs(linear - prefix_linear) > 0.5 * abs(prefix_linear)
    # The real aberration must survive: the quartic term stays at the
    # on-axis level (-11.9 waves), not the pre-fix +37.2.
    assert -20.0 < float(c[0]) < -5.0


def test_r2_on_axis_is_bit_identical_to_the_plane_seed():
    """``L == M == 0`` makes the eikonal identically zero.

    Every on-axis result in the library is therefore untouched by R2.
    """
    from lumenairy.raytrace.trace import _make_bundle
    a = _make_bundle([0.0, 1e-3], [2e-3, -3e-3], [0.0, 0.0], [0.0, 0.0], WL)
    b = _make_bundle([0.0, 1e-3], [2e-3, -3e-3], [0.0, 0.0], [0.0, 0.0], WL,
                     opd_seed='eikonal')
    assert np.array_equal(a.opd, b.opd)
    assert np.all(a.opd == 0.0)


def test_r2_make_bundle_default_is_the_legacy_plane_seed():
    """The shared launcher keeps ``opd = 0``; only the fans opt in.

    ``_make_bundle`` feeds ~20 consumers, two of which add their own
    entrance eikonal downstream (``_lens_traced``'s v5.25.1 H6
    ``_carrier_W_fn``) and would double-count it.
    """
    from lumenairy.raytrace.trace import _make_bundle
    b = _make_bundle([1e-3], [2e-3], [0.05], [0.02], WL)
    assert float(b.opd[0]) == 0.0
    e = _make_bundle([1e-3], [2e-3], [0.05], [0.02], WL, opd_seed='eikonal')
    assert float(e.opd[0]) == pytest.approx(1e-3 * 0.05 + 2e-3 * 0.02,
                                            rel=1e-15)
    with pytest.raises(ValueError, match='opd_seed'):
        _make_bundle([0.0], [0.0], [0.0], [0.0], WL, opd_seed='nope')


# ===========================================================================
# R3 -- conic / aspheric Seidel terms
# ===========================================================================

@pytest.mark.parametrize('k, expect_um', [
    (0.0, -12.2047), (-0.5, -6.1023), (-1.0, 0.0), (-1.5, +6.1016)])
def test_r3_conic_mirror_seidel_tracks_the_real_ray_rho4_fit(k, expect_um):
    """``-S1/8`` must follow the conic constant.

    ORACLE: a rho^2/rho^4/rho^6 least-squares fit of the REAL-RAY optical
    path to the paraxial focus (built here from ``trace`` + an explicit
    geometric leg, independent of the Seidel algebra).  For k = -1 the
    paraboloid is exactly stigmatic at infinite conjugate, so the oracle
    is analytically 0.

    PRE-FIX: ``S1 = +9.765625e-05`` for k = 0, -0.5, -1.0 AND -1.5 alike
    -- the conic constant had zero occurrences in ``seidel.py`` -- so the
    parabola was reported as -12.207 um of spherical.

    BAR: |(-S1/8) - a4| < 0.05 um (on values up to 12.2 um, i.e. 0.4 %).
    Derivation -- the residual is genuine fifth order: the same fit's
    rho^6 coefficient is 0.02-0.06 um here.  Measured 0.0023 / 0.0012 /
    1.4e-14 / 0.0019 um across the four conics, giving 1.3 decades of
    headroom; the pre-fix error for k = -1 was 12.207 um, 2.4 decades
    outside the bar.
    """
    R = -200e-3
    ms = [Surface(radius=R, conic=k, semi_diameter=25e-3, glass_before='air',
                  glass_after='air', is_mirror=True, is_stop=True,
                  thickness=0.0)]
    sd_dict, _ = seidel_coefficients(ms, WL, field_angle=1e-6)
    s1_um = -sd_dict['total']['S1'] / 8.0 * 1e6

    # Real-ray oracle: OPL to the paraxial focus at z = -100 mm.
    ys = np.linspace(1e-9, 25e-3, 61)
    base = [Surface(radius=R, conic=k, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', is_mirror=True,
                    thickness=0.0)]
    im = trace(_axial_bundle(ys), base, WL).image_rays
    P = np.stack([im.x, im.y, im.z], axis=-1)
    Pimg = np.array([0.0, 0.0, -100e-3])
    seg = np.linalg.norm(Pimg[None, :] - P, axis=-1)
    W = (im.opd + seg) - (im.opd[0] + np.linalg.norm(Pimg - P[0]))
    rho = ys / 25e-3
    A = np.stack([rho ** 2, rho ** 4, rho ** 6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    a4_um = coef[1] * 1e6

    assert abs(s1_um - a4_um) < 0.05, (
        f'k={k}: Seidel -S1/8 = {s1_um:+.4f} um vs real-ray a4 = '
        f'{a4_um:+.4f} um')
    assert abs(a4_um - expect_um) < 0.05
    if k == -1.0:
        # The parabola must come out EXACTLY aberration-free.  BAR 1e-12 m
        # on S1: the terms cancel algebraically, so only float rounding of
        # a ~1e-4 quantity survives; measured |S1| = 1.4e-20 m.
        assert abs(sd_dict['total']['S1']) < 1e-12


@pytest.mark.parametrize('A4, tol_pct', [
    (0.0, 0.5), (-250.0, 0.5), (-1000.0, 0.5), (-2000.0, 0.5), (500.0, 0.5)])
def test_r3_aspheric_a4_seidel_tracks_the_real_ray_rho4_fit(A4, tol_pct):
    """``-S1/8`` must follow the A4 aspheric coefficient.

    Same real-ray rho^4 oracle as the conic test.  PRE-FIX:
    ``S1 = +5.320645e-05`` for A4 = 0, -250, -500, -1000 AND -2000 m^-3
    alike, while the real rho^4 coefficient swung -6.639 -> -0.333 ->
    +18.613 um (a SIGN change).

    BAR: 0.5 % relative.  Derivation -- Welford §8.5's aspheric term is
    third-order-exact, so the residual is the system's genuine fifth
    order; measured 0.14-0.29 % across this set (the near-aplanatic
    A4 = -500 point, where the rho^4 term has almost cancelled, reads
    2.9 % of 0.33 um = 0.0096 um absolute and is excluded from the
    relative bar for that reason -- it is covered by the separate
    absolute assertion below).  Pre-fix the same comparison was off by
    up to 1900 %.
    """
    sd = 12.5e-3
    surfs = [Surface(radius=51.68e-3,
                     aspheric_coeffs=({4: A4} if A4 else None),
                     semi_diameter=sd, glass_before='air',
                     glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=sd, glass_before='N-BK7',
                     glass_after='air', thickness=0.0)]
    sd_dict, _ = seidel_coefficients(surfs, WL, field_angle=1e-6)
    s1_um = -sd_dict['total']['S1'] / 8.0 * 1e6

    n = float(get_glass_index('N-BK7', WL))
    f = 51.68e-3 / (n - 1.0)
    bfl = f * (1.0 - (n - 1.0) * 3.6e-3 / (n * 51.68e-3))
    s2 = [Surface(radius=s.radius, conic=s.conic,
                  aspheric_coeffs=s.aspheric_coeffs, semi_diameter=np.inf,
                  glass_before=s.glass_before, glass_after=s.glass_after,
                  is_mirror=s.is_mirror, thickness=s.thickness)
          for s in surfs]
    ys = np.linspace(1e-9, sd, 61)
    im = trace(_axial_bundle(ys), s2, WL).image_rays
    Pimg = np.array([0.0, 0.0, bfl])
    P = np.stack([im.x, im.y, im.z], axis=-1)
    seg = np.sign(bfl - im.z) * np.linalg.norm(Pimg[None, :] - P, axis=-1)
    W = (im.opd + seg) - (im.opd[0] + abs(bfl - im.z[0]))
    rho = ys / sd
    A = np.stack([rho ** 2, rho ** 4, rho ** 6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    a4_um = coef[1] * 1e6

    assert abs(s1_um - a4_um) <= tol_pct / 100.0 * abs(a4_um) + 1e-9, (
        f'A4={A4}: Seidel -S1/8 = {s1_um:+.4f} um vs real-ray a4 = '
        f'{a4_um:+.4f} um')
    # Absolute floor that also covers the near-aplanatic case.
    assert abs(s1_um - a4_um) < 0.05


def test_r3_near_aplanatic_asphere_is_not_reported_as_aberrated():
    """A4 = -500 m^-3 nearly aplanatises the singlet.

    PRE-FIX: the library reported 6.65 um of spherical for a lens whose
    real rho^4 coefficient is -0.33 um -- a 20x over-statement on the
    exact design point an optimiser would be searching for.
    BAR: |-S1/8| < 1.0 um (measured 0.342 um); the pre-fix 6.651 um is
    6.7x outside it.
    """
    sd = 12.5e-3
    surfs = [Surface(radius=51.68e-3, aspheric_coeffs={4: -500.0},
                     semi_diameter=sd, glass_before='air',
                     glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=sd, glass_before='N-BK7',
                     glass_after='air', thickness=0.0)]
    sd_dict, _ = seidel_coefficients(surfs, WL, field_angle=1e-6)
    assert abs(-sd_dict['total']['S1'] / 8.0 * 1e6) < 1.0


def test_r3_higher_order_and_non_rotational_geometry_warns():
    """Silence is what let the omission survive; these now warn.

    Biconic / freeform / field-frame geometry and a paraxial-power-changing
    A2 warn because no rotationally-symmetric third-order expansion exists
    for them.  A6/A8 warn too (VERIFY-WP-A1 open item 6): their third-order
    contribution is genuinely ZERO, so the returned sums stay correct, but
    a caller who tuned an A6 term and saw the sums not move deserves to be
    told the coefficient was read and dropped -- the A6 arm below is
    therefore an assertion about DISCLOSURE, not about numbers, and
    ``test_r3_a6_does_not_change_the_sums`` pins the numbers separately.
    """
    biconic = [Surface(radius=51.68e-3, radius_y=60e-3, semi_diameter=10e-3,
                       glass_before='air', glass_after='N-BK7',
                       thickness=3.6e-3, is_stop=True),
               Surface(radius=np.inf, semi_diameter=10e-3,
                       glass_before='N-BK7', glass_after='air')]
    with pytest.warns(RuntimeWarning, match='radius_y'):
        seidel_coefficients(biconic, WL, field_angle=1e-6)

    a2 = [Surface(radius=51.68e-3, aspheric_coeffs={2: 1.0},
                  semi_diameter=10e-3, glass_before='air',
                  glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
          Surface(radius=np.inf, semi_diameter=10e-3, glass_before='N-BK7',
                  glass_after='air')]
    with pytest.warns(RuntimeWarning, match='paraxial power'):
        seidel_coefficients(a2, WL, field_angle=1e-6)

    # A4 + A6: A4 is handled silently, A6 is named in the warning.
    # (Pre-VERIFY this arm asserted the opposite -- that A4/A6 must NOT
    # warn.  It pinned the silence that open item 6 asked us to remove, so
    # it is inverted here, not deleted.)
    plain = [Surface(radius=51.68e-3, aspheric_coeffs={4: -500.0, 6: 1e6},
                     semi_diameter=10e-3, glass_before='air',
                     glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=10e-3,
                     glass_before='N-BK7', glass_after='air')]
    with pytest.warns(RuntimeWarning, match=r'aspheric_coeffs\[6\]'):
        seidel_coefficients(plain, WL, field_angle=1e-6)

    # A pure A4 asphere (the case the R3 fix HANDLES) must stay silent.
    a4_only = [Surface(radius=51.68e-3, aspheric_coeffs={4: -500.0},
                       semi_diameter=10e-3, glass_before='air',
                       glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
               Surface(radius=np.inf, semi_diameter=10e-3,
                       glass_before='N-BK7', glass_after='air')]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        seidel_coefficients(a4_only, WL, field_angle=1e-6)
    # ...and so must a plain conic, which the same fix handles.
    conic = [Surface(radius=51.68e-3, conic=-1.0, semi_diameter=10e-3,
                     glass_before='air', glass_after='N-BK7',
                     thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=10e-3,
                     glass_before='N-BK7', glass_after='air')]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        seidel_coefficients(conic, WL, field_angle=1e-6)


def test_r3_a6_does_not_change_the_sums():
    """The A6 warning is DISCLOSURE ONLY -- the numbers must not move.

    VERIFY-WP-A1 open item 6 asked for a warning, not a contribution: A6
    generates fifth- and higher-order aberration, whose third-order
    coefficient is exactly zero, so ``seidel_coefficients`` must return
    bit-identical sums with and without it.

    BAR: bit-identical (``np.array_equal``).  Derivation -- ``A6`` enters
    no expression in the per-surface loop at all; the only code path it
    touches is the warning list.  Any difference whatsoever would mean the
    higher-order coefficient had leaked into an arithmetic branch.
    Measured with A6 = 1e9 (10^6 x a realistic value, chosen so a leak of
    even 1e-9 relative weight would show): all five sums equal.
    """
    def _sys(asph):
        return [Surface(radius=51.68e-3, aspheric_coeffs=asph,
                        semi_diameter=10e-3, glass_before='air',
                        glass_after='N-BK7', thickness=3.6e-3,
                        is_stop=True),
                Surface(radius=np.inf, semi_diameter=10e-3,
                        glass_before='N-BK7', glass_after='air')]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        base, _ = seidel_coefficients(_sys({4: -500.0}), WL,
                                      field_angle=1e-3)
        with_a6, _ = seidel_coefficients(_sys({4: -500.0, 6: 1e9}), WL,
                                         field_angle=1e-3)
    for key in ('S1', 'S2', 'S3', 'S4', 'S5'):
        assert np.array_equal(base[key], with_a6[key]), (
            f'{key} moved when A6 = 1e9 was added: {base[key]} -> '
            f'{with_a6[key]}')


# ===========================================================================
# R4 -- the exact conic intersection
# ===========================================================================

def _conic_sag(h, R, k):
    """Closed-form conic sag -- the independent oracle for R4."""
    return (h * h / R) / (1.0 + np.sqrt(1.0 - (1.0 + k) * h * h / R ** 2))


@pytest.mark.parametrize('R, k, hs, label', [
    (10.84e-3, -0.6, [8e-3, 10e-3, 10.83e-3, 10.9e-3, 11.4e-3, 15e-3, 17e-3],
     'Thorlabs-class aspheric condenser'),
    (50e-3, -1.0, [10e-3, 60e-3, 100e-3], 'paraboloid'),
    (50e-3, -2.0, [10e-3, 60e-3, 120e-3], 'hyperboloid'),
])
def test_r4_conic_rays_beyond_the_base_radius_survive_and_land_exactly(
        R, k, hs, label):
    """Rays with ``h > |R|`` hit the conic and must not be reported missed.

    ORACLE: the closed-form conic sag :func:`_conic_sag`; a collimated
    axial ray launched from ``z = 0`` reaches the surface at exactly
    ``t = sag(h)``.

    PRE-FIX (repro/RAYTRACE/p1b_intersect.py, p12_conicdomain.py,
    repro/orch/verify_rt_glass.py): the Newton branch seeded from the
    ray-SPHERE quadratic and used ITS discriminant as the miss test.  A
    sphere of radius R only exists for ``h <= |R|``, so at R = 10.84 mm,
    k = -0.6 (conic valid to h = 17.14 mm) the rays at h = 10.9 and
    11.4 mm came back ``alive=False, error_code=3, t=0`` against true
    sags of 6.186249 and 6.863647 mm; a parabola R = 50 mm at h = 60 mm
    returned t = 0 where a brentq root-find gives 36.000000 mm.

    BAR: |z - sag| < 1e-15 m.  Derivation -- the Spencer-Murty ``t = e/q``
    form is algebraically exact for a conic, so only float rounding of a
    ~1e-2 m quantity survives (~1e-18 m); measured max 8.7e-19 m, three
    decades inside the bar and 16 decades below the pre-fix error (the
    whole sag).
    """
    surf = [Surface(radius=R, conic=k, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', thickness=0.0)]
    h = np.asarray(hs)
    im = trace(_axial_bundle(h, WL_IR), surf, WL_IR).image_rays
    assert np.all(im.alive), (
        f'{label}: rays killed at h/|R| = '
        f'{h[~np.asarray(im.alive)] / abs(R)}')
    assert np.all(im.error_code == RAY_OK)
    assert np.max(np.abs(im.z - _conic_sag(h, R, k))) < 1e-15
    assert np.max(h) > abs(R), 'fixture must exceed the base radius'


def test_r4_true_misses_are_still_killed():
    """A ray that genuinely misses the conic must still die.

    An OBLATE ellipsoid (k > 0) has a bounded domain ``h < |R|/sqrt(1+k)``
    and a genuinely closed surface; a ray outside it has no real root.
    """
    R, k = 20e-3, 2.0
    h_lim = abs(R) / np.sqrt(1.0 + k)         # 11.547 mm
    surf = [Surface(radius=R, conic=k, semi_diameter=np.inf,
                    glass_before='air', glass_after='air', thickness=0.0)]
    h = np.array([5e-3, 11e-3, 11.6e-3, 15e-3])
    im = trace(_axial_bundle(h, WL_IR), surf, WL_IR).image_rays
    assert list(np.asarray(im.alive)) == [True, True, False, False]
    assert np.all(im.error_code[2:] == RAY_MISSED_SURFACE)
    assert h_lim == pytest.approx(11.547e-3, rel=1e-3)


def test_r4_spherical_surfaces_are_unaffected():
    """conic == 0: the conic quadratic is exactly ``R`` x the sphere one.

    BAR: bit-identical alive mask, and |z - sag| < 1e-15 m against the
    closed form (measured 1.7e-18 m).  The audited spherical fast path
    (3.3e-16 m vs a Spencer-Murty root over 480k random rays) is on a
    SEPARATE branch that this change does not touch at all.
    """
    R = 30e-3
    surf = [Surface(radius=R, conic=0.0, aspheric_coeffs={4: 1.0},
                    semi_diameter=np.inf, glass_before='air',
                    glass_after='air', thickness=0.0)]
    h = np.array([0.0, 5e-3, 15e-3, 25e-3])
    im = trace(_axial_bundle(h, WL_IR), surf, WL_IR).image_rays
    assert np.all(im.alive)
    expect = _conic_sag(h, R, 0.0) + 1.0 * h ** 4
    assert np.max(np.abs(im.z - expect)) < 1e-15


def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_r4_jax_backend_agrees_with_numpy_on_the_conic_domain():
    """The JAX twin had the identical false miss; both are fixed.

    PRE-FIX both backends returned ``alive = [T T T F F]``; the analytic
    ADRT (``differential._adrt_step``, which already solved the exact
    conic) kept all five, which is how the audit localised the defect.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax

    pres = {'surfaces': [
        {'radius': 10.84e-3, 'conic': -0.6, 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [5e-3, 0.0]}
    surfs = [Surface(radius=10.84e-3, conic=-0.6, semi_diameter=np.inf,
                     glass_before='air', glass_after='N-BK7', thickness=5e-3),
             Surface(radius=np.inf, semi_diameter=np.inf,
                     glass_before='N-BK7', glass_after='air')]
    h = np.array([8e-3, 10e-3, 10.8e-3, 10.9e-3, 11.4e-3])
    npo = trace(_axial_bundle(h, WL_IR), surfs, WL_IR).image_rays
    st = make_jax_ray_state(np.zeros(5), h, np.zeros(5),
                            np.zeros(5), np.zeros(5), np.ones(5))
    jo = trace_jax(st, pres, WL_IR)
    assert np.all(np.asarray(jo.alive))
    assert np.array_equal(np.asarray(jo.alive), np.asarray(npo.alive))
    # BAR 1e-17 m: the audited NumPy<->JAX parity floor with x64 is
    # 6.9e-18 m on position / 2.8e-17 m on OPL; measured here 1.7e-18 m
    # and 8.7e-18 m.
    assert np.max(np.abs(np.asarray(jo.y) - npo.y)) < 1e-17
    assert np.max(np.abs(np.asarray(jo.opd) - npo.opd)) < 1e-16


# ===========================================================================
# R5 -- the diffraction-order kick carries the medium index
# ===========================================================================

def test_r5_grating_kick_into_glass_obeys_the_grating_equation():
    """``n2 L' = n1 L + m lambda / Lambda``.

    ORACLE: the grating equation itself (conservation of the tangential
    wavevector across the interface), evaluated by hand.

    PRE-FIX (repro/RAYTRACE/p11_misc.py): the library returned
    ``L = 0.26200000`` inside N-BK7 where the grating equation gives
    ``0.17425045`` -- ratio 1.503583, EXACTLY n(N-BK7), i.e. a 50 %
    direction error (15.2 deg instead of 10.0 deg).

    BAR: 1e-12 relative.  Derivation -- the fix is one extra divide, so
    the residual is the glass-index lookup's own rounding (~1e-16
    relative); measured 0.0.  The pre-fix value is 5.0e-1 relative away,
    11 decades outside.
    """
    lam, period, m = WL_IR, 5e-6, 1
    n2 = float(get_glass_index('N-BK7', lam))
    surfs = [Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                     glass_after='N-BK7', thickness=1e-3),
             Surface(radius=np.inf, semi_diameter=np.inf,
                     glass_before='N-BK7', glass_after='air')]
    rb = RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                   L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                   wavelength=lam, alive=np.ones(1, bool), opd=np.zeros(1))
    res = trace(rb, surfs, lam, output_filter='all',
                surface_diffraction={0: (m, 0, period, np.inf)})
    L_in_glass = float(res.rays_at(0).L[0])
    expect = m * lam / (n2 * period)
    assert abs(L_in_glass - expect) < 1e-12 * abs(expect)
    assert abs(L_in_glass - m * lam / period) > 0.4 * abs(expect)


def test_r5_grating_opl_term_is_index_independent():
    """The phase-screen OPL is ``m lambda x / Lambda`` -- NOT divided by n2.

    ORACLE: the grating is a phase discontinuity ``Phi = 2 pi m x /
    Lambda``, whose optical-path equivalent is ``Phi / k0``.  Its
    transverse gradient must equal ``n2 L' - n1 L``, which is the whole
    point of the R5 fix; dividing the OPL by n2 as well would break that
    identity.  BAR 1e-15 relative (measured 0.0).
    """
    lam, period, m, x0 = WL_IR, 5e-6, 1, 2e-3
    surfs = [Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                     glass_after='N-BK7', thickness=0.0)]
    rb = RayBundle(x=np.array([x0]), y=np.zeros(1), z=np.zeros(1),
                   L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                   wavelength=lam, alive=np.ones(1, bool), opd=np.zeros(1))
    res = trace(rb, surfs, lam, surface_diffraction={0: (m, 0, period,
                                                         np.inf)})
    got = float(res.image_rays.opd[0])
    expect = m * lam * x0 / period
    assert abs(got - expect) < 1e-15 * abs(expect)


def test_r5_apply_doe_phase_traced_n_medium_kwarg():
    """The standalone helper gained ``n_medium`` (default 1.0 = legacy)."""
    lam, period, m = WL_IR, 5e-6, 1
    rb = RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                   L=np.zeros(1), M=np.zeros(1), N=np.ones(1),
                   wavelength=lam, alive=np.ones(1, bool), opd=np.zeros(1))
    air = apply_doe_phase_traced(rb, m, 0, period_x=period,
                                 period_y=np.inf)
    assert float(air.L[0]) == pytest.approx(m * lam / period, rel=1e-15)
    n2 = float(get_glass_index('N-BK7', lam))
    glass = apply_doe_phase_traced(rb, m, 0, period_x=period,
                                   period_y=np.inf, n_medium=n2)
    assert float(glass.L[0]) == pytest.approx(m * lam / (n2 * period),
                                              rel=1e-15)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match='n_medium'):
            apply_doe_phase_traced(rb, m, 0, period_x=period,
                                   period_y=np.inf, n_medium=bad)


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_r5_jax_doe_kick_matches_numpy_in_glass():
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax

    lam, period, m = WL_IR, 5e-6, 1
    pres = {'surfaces': [
        {'radius': float('inf'), 'glass_before': 'air',
         'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7',
         'glass_after': 'air'}],
        'thicknesses': [1e-3, 0.0]}
    st = make_jax_ray_state(np.zeros(1), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.ones(1))
    out = trace_jax(st, pres, lam,
                    surface_diffraction={0: (m, 0, period, np.inf)})
    n2 = float(get_glass_index('N-BK7', lam))
    # After the second (glass -> air) surface the ray refracts back out;
    # check the direction cosine is the air-side grating angle, i.e. the
    # kick was applied with 1/n2 inside.  BAR 1e-12 relative.
    assert abs(float(out.L[0]) - m * lam / period) < 1e-12 * m * lam / period
    assert n2 > 1.4


# ===========================================================================
# R6 -- rays_from_field estimator + _transfer grazing guard
# ===========================================================================

@pytest.mark.parametrize('L_true', [0.05, 0.10, 0.15, 0.20, 0.24])
def test_r6_direction_estimator_is_exact_to_the_full_grid_nyquist(L_true):
    """No aliasing below the GRID's own Nyquist ``lambda / (2 dx)``.

    ORACLE: a pure tilted plane wave ``exp(i k (L x))`` -- the recovered
    L is analytically L_true whenever the grid can represent it.

    PRE-FIX (repro/RAYTRACE/p7_fromfield.py): the two-pixel estimator
    ``arg(E[j+1] conj(E[j-1])) / (2 dx)`` was unambiguous only to HALF
    that (``lambda / (4 dx)`` = 0.125 here) and wrapped silently above:
    0.150 -> -0.100, 0.200 -> -0.050, 0.300 -> +0.050, 0.490 -> -0.010.

    BAR: 1e-12 absolute on L.  Derivation -- the symmetrised one-pixel
    circular mean is analytically exact for a plane wave, so only the
    ``np.angle`` round-off of a unit-modulus product survives (~1e-16);
    measured max |dL| = 3e-16.  The pre-fix error at L = 0.15 was 0.25,
    twelve decades outside.
    """
    lam, dx, N = 1e-6, 2e-6, 64
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    E = np.exp(1j * 2 * np.pi / lam * L_true * X)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=200,
                           placement='uniform')
    assert abs(float(np.median(rays.L)) - L_true) < 1e-12
    assert L_true < lam / (2 * dx) + 1e-12   # inside the grid Nyquist


def test_r6_edge_rays_get_the_full_direction_cosine():
    """Boundary columns/rows are no longer halved.

    PRE-FIX: ``clip(ix+1)`` collapsed onto ``ix`` on the last column while
    the divisor stayed ``2 dx``, so edge rays read exactly HALF the
    correct cosine -- measured ratio 0.5000 over 128 edge rays.

    BAR: |L_edge / L_true - 1| < 1e-12 (measured 0.0); the pre-fix ratio
    0.5 is twelve decades outside.
    """
    lam, dx, N, L_true = 1e-6, 2e-6, 64, 0.05
    x = (np.arange(N) - N // 2) * dx
    X, _ = np.meshgrid(x, x)
    E = np.exp(1j * 2 * np.pi / lam * L_true * X)
    rays = rays_from_field(E, dx=dx, wavelength=lam,
                           n_rays=N * N, placement='uniform')
    edge = (np.abs(rays.x - x[0]) < 0.5 * dx) | \
           (np.abs(rays.x - x[-1]) < 0.5 * dx)
    assert int(edge.sum()) >= 64, 'fixture must contain edge rays'
    assert np.max(np.abs(rays.L[edge] / L_true - 1.0)) < 1e-12


def test_r6_converging_wave_still_focuses():
    """CONTROL: the audit's "verified correct" focusing result is preserved.

    A converging ``exp(-i k R)`` wave must produce rays crossing z = +f
    with ~zero rms radius.  The audit measured 0.001 nm; BAR 0.01 nm
    (measured 0.001 nm).  A naive amplitude-WEIGHTED symmetrised sum
    would have degraded this to 16.3 nm, which is why the estimator
    normalises the two one-pixel phasors before averaging.
    """
    lam, dx, N, f = 1e-6, 2e-6, 128, 5e-3
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X ** 2 + Y ** 2 + f ** 2)
    E = np.exp(-1j * 2 * np.pi / lam * R)
    rays = rays_from_field(E, dx=dx, wavelength=lam, n_rays=400,
                           placement='uniform')
    t = f / rays.N
    rx = rays.x + rays.L * t
    ry = rays.y + rays.M * t
    rms = float(np.sqrt(np.mean(rx ** 2 + ry ** 2)))
    assert rms < 1e-11, f'rms at +f = {rms * 1e9:.3f} nm'


def test_r6_transfer_kills_grazing_rays_instead_of_teleporting_them():
    """``_transfer`` had no grazing guard: the "immortal phantom".

    PRE-FIX (repro/RAYTRACE/p11_misc.py): a bundle at ``z = 1e-4`` with
    ``N = 0`` came out of ``_transfer(10 mm, n=1)`` as ``z=[0 0]``,
    ``alive=[True True]``, ``opd=[0 0]``, ``error_code=[0 0]`` -- the ray
    was teleported one gap downstream with zero optical path and stayed
    alive.  ``_intersect_surface``'s flat branch has carried the same
    guard since R-4; this is the sibling site.
    """
    b = RayBundle(x=np.zeros(2), y=np.zeros(2), z=np.full(2, 1e-4),
                  L=np.array([1.0, 0.0]), M=np.array([0.0, 1.0]),
                  N=np.zeros(2), wavelength=WL,
                  alive=np.ones(2, bool), opd=np.zeros(2))
    _transfer(b, 10e-3, 1.0)
    assert not b.alive.any()
    assert np.all(b.error_code == RAY_MISSED_SURFACE)
    assert np.all(b.z == 1e-4), 'grazing rays must NOT be teleported to z=0'
    assert np.all(b.opd == 0.0)


def test_transfer_and_flat_intersect_split_nan_from_grazing():
    """``N = nan`` gets RAY_NAN (4); a finite grazing ``N`` gets 3.

    VERIFY-WP-A1 open item 5.  Both fail the same ``|N| > tol`` test in
    ``vertex_plane_transfer_t`` (``abs(nan) > tol`` is False), so before
    the split a NaN direction cosine was reported as "missed the surface"
    at all three sites that share the kernel -- ``_transfer``, the flat
    branch of ``_intersect_surface``, and ``exit_vertex_transfer``.
    ``RAY_NAN`` is the code the Newton branch already uses for exactly
    this condition, so the diagnostic vocabulary is now consistent across
    the whole module.

    BAR: exact error codes at both sites.  Discrete decisions, no
    tolerance; and the healthy third ray proves the classification is
    per-ray rather than a whole-bundle ``np.where``.
    """
    from lumenairy.raytrace.intersection import _intersect_surface
    from lumenairy.raytrace.surface import RAY_NAN

    def _probe():
        return RayBundle(x=np.zeros(3), y=np.zeros(3), z=np.full(3, 1e-4),
                         L=np.array([1.0, 0.0, 0.0]),
                         M=np.array([0.0, 1.0, 0.0]),
                         N=np.array([np.nan, 0.0, 1.0]), wavelength=WL,
                         alive=np.ones(3, bool), opd=np.zeros(3),
                         error_code=np.zeros(3, dtype=np.uint8))

    b = _probe()
    with np.errstate(invalid='ignore'):
        _transfer(b, 10e-3, 1.0)
    assert list(b.alive) == [False, False, True]
    assert int(b.error_code[0]) == RAY_NAN
    assert int(b.error_code[1]) == RAY_MISSED_SURFACE
    assert int(b.error_code[2]) == 0

    b = _probe()
    flat = Surface(radius=np.inf, glass_before='air', glass_after='air',
                   thickness=0.0)
    with np.errstate(invalid='ignore'):
        _intersect_surface(b, flat, n_medium=1.0)
    assert list(b.alive) == [False, False, True]
    assert int(b.error_code[0]) == RAY_NAN
    assert int(b.error_code[1]) == RAY_MISSED_SURFACE
    assert int(b.error_code[2]) == 0


def test_r6_transfer_is_unchanged_for_ordinary_rays():
    """BAR: bit-identical.  The guard must not touch normal propagation."""
    b = RayBundle(x=np.array([1e-3]), y=np.array([-2e-3]),
                  z=np.array([1e-4]), L=np.array([0.1]),
                  M=np.array([-0.2]), N=np.array([np.sqrt(1 - 0.01 - 0.04)]),
                  wavelength=WL, alive=np.ones(1, bool), opd=np.zeros(1))
    t_expect = (10e-3 - 1e-4) / b.N[0]
    x0, y0, n = float(b.x[0]), float(b.y[0]), 1.5
    _transfer(b, 10e-3, n)
    assert bool(b.alive[0])
    assert float(b.z[0]) == 0.0
    assert float(b.opd[0]) == n * t_expect
    assert float(b.x[0]) == x0 + 0.1 * t_expect
    assert float(b.y[0]) == y0 + (-0.2) * t_expect


def test_r6_opd_phase_unwrapped_option():
    """R7 sibling: ``opd`` is the WRAPPED phase unless asked otherwise.

    PRE-FIX (and still the default): a converging wave with 0.8 waves of
    true OPL spread returned ``opd`` covering the whole
    [-lambda/2, +lambda/2] wrap interval.  BAR -- the wrapped seed must
    stay inside (-lambda/2, +lambda/2] and the unwrapped one must exceed
    it (measured spread 0.82 um = 0.82 waves).
    """
    lam, dx, N, f = 1e-6, 2e-6, 64, 5e-3
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-1j * 2 * np.pi / lam * np.sqrt(X ** 2 + Y ** 2 + f ** 2))
    w = rays_from_field(E, dx=dx, wavelength=lam, n_rays=400,
                        placement='uniform')
    u = rays_from_field(E, dx=dx, wavelength=lam, n_rays=400,
                        placement='uniform', opd_phase='unwrapped')
    assert float(np.ptp(w.opd)) <= lam + 1e-15
    assert float(np.ptp(u.opd)) > 0.5 * lam
    with pytest.raises(ValueError, match='opd_phase'):
        rays_from_field(E, dx=dx, wavelength=lam, n_rays=4,
                        opd_phase='nope')


# ===========================================================================
# R7 -- the P3 bundle
# ===========================================================================

def test_r7_refract_error_code_is_first_failure_wins():
    """``_refract``'s comment promised it; the ``np.where`` was unconditional.

    Currently unreachable through ``trace`` (``alive => RAY_OK`` holds by
    construction), so this exercises ``_refract`` directly with a
    hand-built bundle that carries a prior diagnosis -- the "live trap"
    the audit described.
    """
    from lumenairy.raytrace.intersection import _refract
    surf = Surface(radius=np.inf, glass_before='N-BK7', glass_after='air')
    b = RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                  L=np.array([0.95]), M=np.zeros(1),
                  N=np.array([np.sqrt(1 - 0.95 ** 2)]),
                  wavelength=WL, alive=np.ones(1, bool), opd=np.zeros(1),
                  error_code=np.array([2], dtype=np.uint8))  # RAY_APERTURE
    _refract(b, surf, float(get_glass_index('N-BK7', WL)), 1.0)
    assert not bool(b.alive[0]), 'fixture must TIR'
    assert int(b.error_code[0]) == 2, (
        'RAY_TIR must not relabel a ray that already carried RAY_APERTURE')


def test_r7_trace_summary_reports_evanescent_losses(capsys):
    """The printed breakdown must sum to the reported lost count.

    PRE-FIX: ``[TIR=, aperture=, miss=, nan=]`` omitted RAY_EVANESCENT
    entirely, so a ``surface_diffraction`` order going evanescent
    vanished from the diagnostic that exists to explain it.
    """
    lam, period = WL, 0.3e-6      # lambda / period > 1 -> evanescent
    surfs = [Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                     glass_after='air', thickness=1e-3),
             Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                     glass_after='air')]
    rb = RayBundle(x=np.zeros(3), y=np.zeros(3), z=np.zeros(3),
                   L=np.zeros(3), M=np.zeros(3), N=np.ones(3),
                   wavelength=lam, alive=np.ones(3, bool), opd=np.zeros(3))
    res = trace(rb, surfs, lam,
                surface_diffraction={0: (1, 0, period, np.inf)})
    assert np.all(res.image_rays.error_code == RAY_EVANESCENT)
    trace_summary(res)
    out = capsys.readouterr().out
    assert 'evanescent=3' in out
    assert 'unclassified' not in out


def test_r7_raytrace_system_does_not_mutate_the_surface_list_it_returns():
    """The last surface is CLONED with the image distance, not rewritten.

    Pre-fix ``surfaces[-1].thickness = image_distance`` mutated the object
    in place -- the exact pattern ``trace_prescription`` was moved away
    from in v4.13.2 (audit P1-NEW-J).  Calling twice on the same element
    list must give the same answer.
    """
    elements = [{'type': 'lens', 'f': 0.1, 'aperture_diameter': 0.02},
                {'type': 'propagate', 'z': 0.05}]
    r1, s1 = raytrace_system(elements, WL)
    r2, s2 = raytrace_system(elements, WL)
    assert [s.thickness for s in s1] == [s.thickness for s in s2]
    assert np.array_equal(r1.image_rays.y, r2.image_rays.y)


def test_r7_field_of_view_finite_conjugate_uses_the_sensor():
    """The finite-conjugate branch is a field of view again.

    PRE-FIX it returned ``arctan((aperture/2) / object_distance)``, an
    object-space APERTURE half-angle independent of the sensor -- a
    numerical aperture wearing the wrong name.

    ORACLE: Gaussian imaging, ``m = -f / (s_obj - f)``, so a sensor of
    half-height ``h'`` sees an object half-height ``h = h' / |m|`` and a
    half-field ``arctan(h / s_obj)``.
    """
    p = {'name': 't', 'aperture_diameter': 25e-3,
         'surfaces': [{'radius': 51.68e-3, 'glass_before': 'air',
                       'glass_after': 'N-BK7'},
                      {'radius': np.inf, 'glass_before': 'N-BK7',
                       'glass_after': 'air'}],
         'thicknesses': [3.6e-3, 0.0],
         'object_distance': 0.5}
    n = float(get_glass_index('N-BK7', WL))
    f = 51.68e-3 / (n - 1.0)          # thin-lens EFL of a plano-convex
    h_img = 4.8e-3
    theta, h_obj = field_of_view(p, WL, sensor_half_height_m=h_img)
    # BAR 2 %: the library's EFL is the THICK-lens value, which differs
    # from the thin-lens oracle by (n-1) t / (n R1) = 0.7 % here; the
    # magnification inherits that.  Measured 0.7 %.
    m_t = -f / (0.5 - f)
    assert h_obj == pytest.approx(abs(h_img / m_t), rel=0.02)
    assert theta == pytest.approx(np.arctan(h_obj / 0.5), rel=1e-12)
    # The aperture proxy is 25 mrad and must NOT be what we get.
    assert abs(theta - np.arctan(12.5e-3 / 0.5)) > 0.01

    # Without a sensor the legacy proxy survives, but it WARNS.
    with pytest.warns(RuntimeWarning, match='APERTURE half-angle'):
        theta2, h2 = field_of_view(p, WL)
    assert theta2 == pytest.approx(np.arctan(12.5e-3 / 0.5), rel=1e-12)


def test_r7_layout_module_states_it_holds_no_layout_geometry():
    """The docstring no longer implies a 2-D layout renderer lives here."""
    from lumenairy.raytrace import layout
    doc = layout.__doc__ or ''
    assert 'no layout GEOMETRY' in doc
    assert 'plot_lens_layout' in doc
    assert not hasattr(layout, 'plot_layout')


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_r7_adrt_jax_single_pass_matches_the_numpy_dual_path():
    """``_adrt_jax`` now uses ``jacfwd(..., has_aux=True)``: one pass.

    Pre-fix it walked the whole prescription TWICE (``jacfwd(_state)``
    then ``vmap(_full)``) -- a free ~2x.  BAR: Jacobian 1e-15 absolute,
    OPL / position bit-identical to the NumPy dual backend.  Derivation --
    the two backends evaluate the same closed form in different float
    orders, so ~1e-16 relative on Jacobian entries of order 1 is the
    floor; measured 3.3e-16 (Jacobian), 0.0 (opd, x), 1.4e-17 (ux).
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.raytrace import ray_transfer_jacobian_analytic
    surfs = [Surface(radius=50e-3, conic=-0.5, glass_before='air',
                     glass_after='N-BK7', thickness=4e-3),
             Surface(radius=-60e-3, glass_before='N-BK7', glass_after='air',
                     thickness=50e-3)]
    x = np.array([0.0, 2e-3, 5e-3])
    y = np.array([0.0, -1e-3, 4e-3])
    ux = np.array([0.0, 0.01, -0.03])
    uy = np.array([0.0, -0.02, 0.02])
    a = ray_transfer_jacobian_analytic(x, y, ux, uy, surfs, WL_IR)
    j = ray_transfer_jacobian_analytic(jnp.asarray(x), jnp.asarray(y),
                                       jnp.asarray(ux), jnp.asarray(uy),
                                       surfs, WL_IR)
    assert np.max(np.abs(np.asarray(j.jacobian) - a.jacobian)) < 1e-15
    assert np.max(np.abs(np.asarray(j.opd) - a.opd)) < 1e-18
    assert np.max(np.abs(np.asarray(j.x) - a.x)) < 1e-18
