"""W4c: the three ``analysis/`` consumers of W4/W4b's conventions.

W4 fixed ``compute_pupils`` (geometric pupil coordinates) and W4b fixed
``system_abcd`` (geometric ``bfl``/``ffl``, ``efl`` deliberately left
REDUCED).  Both left flagged, measured, unfixed consumers in
``analysis/``; this file closes them.

C1  ``field.py`` -- the f-tan(theta) DISTORTION REFERENCE
    ``h_paraxial = efl * tan(theta)`` (and the matching
    ``distortion_grid`` ``paraxial_x``/``paraxial_y``) needs the
    OBJECT-SPACE focal length ``n_obj * efl``, because ``efl`` is the
    reduced ``1/Phi``.  ORACLE: a real ray at theta = 1e-7 rad through
    the same surface list with the image plane at the focus gives the
    reference slope ``dh/dtan(theta)`` free of any finite-field
    aberration.  Measured on an N-BK7 object space: slope
    +5.373730037e-02 m, reproduced by ``n_obj * efl`` to 7.7e-16, while
    ``efl`` alone is 34.07% low -- reported as a spurious
    +51.680024% = 100*(n_obj - 1) distortion at EVERY field angle,
    including the theta -> 0 limit where a real system must read zero.
    After: +0.000014% at 0.1 deg.  Air object spaces are bit-identical
    (``n_obj`` is exactly 1.0).

C2  ``image_plane_wfe.py`` -- the GAUSS CONJUGATE SOLVE
    ``1/v = 1/efl - 1/u`` is the air-only form; with a reduced ``efl``
    and geometric ``u``/``v`` the general form is
    ``n_obj/u + n_img/v = 1/efl``.  ORACLE: a real ray from the axial
    object point at ``object_distance``, axis crossing read past the
    last surface (global z, the frame ``img_d_m`` is consumed in).
    Measured:

        prescription      img_d_m pre-fix  exact          error
        air control       +121.819095 mm   +121.819095 mm  1.8e-12
        N-BK7 image space  +41.569590 mm    +70.557940 mm  -41.1%
        N-BK7 object space +34.023215 mm    +35.549003 mm   -4.3%

    The two errors are NOT a common factor -- ``n_obj`` and ``n_img``
    enter the equation differently -- which is why both must be
    threaded.  The misplaced reference sphere also dominated the
    reported wavefront: PV 114.821866 -> 53.017548 waves on the
    image-immersed design.  After: both match the oracle to <= 5.3e-12.

C3  ``field.py`` -- ``bfl`` USED AS A GLOBAL-Z DISTANCE
    ``bfl`` is an along-the-ray ("unfolded") distance (W4b / S11-1) but
    a local-frame ``Surface.thickness`` is a GLOBAL-Z step, and
    ``_propagate_rms``'s ``z_to`` is a global-z position.  After an ODD
    number of mirrors those differ by a sign, so five
    ``_append_image_plane`` sites and the ``field_aberration_sweep``
    focus search all placed the image plane on the WRONG SIDE of the
    last vertex -- pre-existing in air, nothing to do with immersion.
    ORACLE: a collimated probe must land ON AXIS at the appended plane,
    and the folded system's spot must equal its own mirrorless control's.
    Measured on an air singlet + flat fold (bfl = +37.536224 mm, focus
    at global z = -37.536224 mm, so the plane sat 2*|bfl| = 75.07 mm
    away): the probe launched at 1.0e-7 m arrived 1.525460e-07 m off
    axis (1.53x its launch height) instead of ~6e-19 m, and
    ``_propagate_rms`` read 3.946987e-03 m instead of 2.130362e-05 m --
    a 185x error, where 2.130362e-05 m is EXACTLY the mirrorless
    control's own value.  The fix is S11-1's documented mapping
    (``(-1) ** n_mirrors``, the same expression
    ``test_niche_s11_sibling_deferred`` uses when it writes
    ``surf[-1].thickness = (-1.0) ** n_mir * fod.bfl``), taken from its
    single source ``raytrace.seidel._mirror_parity_sign``.
    Bit-identical for every EVEN mirror parity, mirrorless included.

CONSUMER AUDIT (done before editing; this is the ``analysis`` sweep
W4b said was needed)

* ``_append_image_plane`` -- 5 call sites, ALL inside ``field.py``
  (``distortion_vs_field``, ``distortion_grid``,
  ``spot_diagram_vs_field``, ``footprint_per_surface``,
  ``relative_illumination``).  No callers anywhere else in the library.
  The mapping therefore lives INSIDE the helper (one edit, all five
  sites) and its ``image_distance`` argument is documented as
  along-the-ray, matching ``bfl``, so the two can't drift.
* ``h_paraxial`` / ``distortion_pct`` / ``max_distortion_pct`` /
  ``paraxial_x`` / ``paraxial_y`` -- read only by
  ``ui/distortion_dock.py``, which plots them without re-deriving or
  compensating, plus the re-exported dataclasses.  No compensating
  factor anywhere.
* ``eval_image_plane_wfe``'s ``img_d_m`` -- consumed internally by
  ``_radius_for``, ``cz``, ``_chief_image_xy``, the ``best_rms`` /
  ``best_pv`` searches and the ``dof`` estimate, and returned as
  ``ImagePlaneWFE.img_d_m`` / ``img_d_m_paraxial``; read externally by
  ``ui/analysis.py`` (display) and the test suite.  All uses are
  geometric axial distances in the trace's own frame.
* ``field_aberration_sweep``'s ``dz_search = abs(bfl)/20`` needs the
  MAGNITUDE only and is deliberately untouched.

FLAGGED, MEASURED, NOT FIXED (both outside these three findings)

F1  ``_append_image_plane``'s WORLD-frame branch places the plane along
    ``last.world_R[:, 2]``.  ``world_surfaces_from_prescription``
    re-aligns that axis only at a COORD-BREAK, never at a mirror, so
    for a folded world-frame list WITHOUT the re-aligning coord-break
    the same sign bug bites: measured on a 1-mirror world list, the
    marginal probe landed at x = +5.5954e-04 m with ``+bfl`` and
    -7.94e-08 m with ``(-1)**m * bfl``.  But a prescription that DOES
    include the re-aligning coord-break has ``z_hat`` already along the
    ray, where mapping would be wrong -- and the surface list alone
    cannot distinguish the two.  Deciding that needs the world-frame
    convention's own audit; the local-frame branch (every call site in
    practice) is what W4c fixes.
F2  ``eval_image_plane_wfe`` on a FOLDED prescription is broken beyond
    C2: ``pp_image_z`` is unfolded while ``img_d_m`` is global z, and
    ``N_chief < 0`` inverts every ``1/N_chief`` factor.  Measured on an
    air singlet + flat fold: ``r_sphere_m`` comes back NEGATIVE
    (-4.288895e-02 m) and the WFE reads 321.00 waves PV / 100.17 waves
    RMS for a system that is a few waves unfolded.  Not a one-line
    sign -- the whole ray-sphere geometry needs a folded-frame review.

Tolerance: ``_TOL = 1e-9`` relative.  Oracle floor measured at 5.3e-12
(the C2 finite-conjugate root-find), so ~190x above the floor and ~7
orders below every defect above.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.field import (
    _append_image_plane,
    _entrance_pupil_aim,
    _propagate_rms,
    distortion_grid,
    distortion_vs_field,
    field_aberration_sweep,
)
from lumenairy.glass import get_glass_index
from lumenairy.raytrace import (
    RayBundle,
    Surface,
    first_order_data,
    surfaces_from_prescription,
    system_abcd,
    trace,
)

_WL = 587.5618e-9
_WLW = 587.56e-9          # the wavelength the image_plane_wfe pins use
_TOL = 1e-9
_HP = 1e-7                # probe launch height / paraxial probe scale


# ======================================================================
# Independent local re-implementations of the conversion factors.  The
# oracles must not import the code they judge (campaign rule), and this
# also lets the value pins below fail on their own NUMBERS against a
# pre-fix build rather than on an ImportError.
# ======================================================================
def _loc_parity_sign(surfaces):
    m = sum(1 for s in surfaces if s.is_mirror and not s.is_coordbrk)
    return -1.0 if (m % 2) else 1.0


def _loc_n_obj(surfaces, wl=_WL):
    return abs(float(get_glass_index(surfaces[0].glass_before, wl)))


def _loc_n_img(surfaces, wl=_WL):
    last = surfaces[-1]
    glass = (last.glass_before if (last.is_mirror and not last.is_coordbrk)
             else last.glass_after)
    return abs(float(get_glass_index(glass, wl)))


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


# ======================================================================
# Shared exact real-ray machinery (no paraxial/ABCD code involved)
# ======================================================================
def _ray(y0, u0, wl):
    y0 = np.atleast_1d(np.asarray(y0, float))
    th = np.arctan(float(u0))
    z = np.zeros_like(y0)
    return RayBundle(x=z.copy(), y=y0.copy(), z=z.copy(), L=z.copy(),
                     M=np.full_like(y0, np.sin(th)),
                     N=np.full_like(y0, np.cos(th)),
                     wavelength=wl, alive=np.ones_like(y0, bool),
                     opd=z.copy())


def _state_at(surfaces, wl, idx, y0, u0):
    b = trace(_ray(y0, u0, wl), surfaces, wl).ray_history[idx]
    assert bool(b.alive[0]), f"oracle ray died at surface {idx}"
    return float(b.y[0]), float(b.M[0] / b.N[0])


# ----------------------------------------------------------------------
# C1 designs + oracle
# ----------------------------------------------------------------------
def _d_c1(glass_before='N-BK7'):
    """Positive-EFL triplet-ish stack whose LAST surface is CURVED, so
    ``distortion_vs_field`` appends its own image plane at ``bfl``
    instead of treating the design's tail as the image plane."""
    return [
        Surface(radius=-60e-3, glass_before=glass_before, glass_after='air',
                thickness=8e-3, semi_diameter=6e-3, is_stop=True),
        Surface(radius=60e-3, glass_before='air', glass_after='N-SF2',
                thickness=4e-3, semi_diameter=np.inf),
        Surface(radius=-60e-3, glass_before='N-SF2', glass_after='air',
                thickness=0.0, semi_diameter=np.inf),
    ]


def _d_c1_immersed():
    return _d_c1('N-BK7')


def _d_c1_air():
    return _d_c1('air')


def _oracle_reference_slope(surfaces, wl):
    """``dh_image / dtan(theta)`` from a REAL ray at a vanishing field.

    That derivative IS the f-tan(theta) reference height's slope, by
    definition, and at theta = 1e-7 rad no finite-field aberration
    survives.  Uses the same entrance-pupil aiming and the same
    image-plane placement the function under test uses, so it measures
    the reference height and nothing else.
    """
    _M, _efl, bfl, _ffl = system_abcd(surfaces, wl)
    surfs = _append_image_plane(surfaces, bfl)
    ep_z, _ = _entrance_pupil_aim(surfs, wl, 0.0)
    th = 1e-7
    u = float(np.tan(th))
    y0 = -float(np.sin(th)) * ep_z / float(np.cos(th))
    y_img, _u = _state_at(surfs, wl, len(surfaces), y0, u)
    return y_img / u


# 12-significant-digit oracle values
_C1_SLOPE_IMMERSED = +5.373730037e-02
_C1_SLOPE_AIR = +4.693343922e-02


class TestW4cDistortionReferenceHeight:
    """C1: the f-tan(theta) reference needs the OBJECT-space focal
    length, not the reduced ``efl``."""

    def test_reference_slope_oracle_self_check(self):
        """The oracle must reproduce ``n_obj * efl`` -- that identity is
        what makes it ground truth for the reference height."""
        for builder, want in ((_d_c1_air, _C1_SLOPE_AIR),
                              (_d_c1_immersed, _C1_SLOPE_IMMERSED)):
            surfaces = builder()
            slope = _oracle_reference_slope(surfaces, _WL)
            assert _rel(slope, want) < _TOL, (
                f"oracle slope {slope!r} vs pinned {want!r}")
            _M, efl, _b, _f = system_abcd(surfaces, _WL)
            n_obj = _loc_n_obj(surfaces)
            assert _rel(n_obj * efl, slope) < _TOL
            if n_obj != 1.0:
                # ...and the reduced efl demonstrably is NOT the slope
                assert _rel(efl, slope) > 0.3

    @pytest.mark.parametrize('builder,slope', [
        (_d_c1_air, _C1_SLOPE_AIR), (_d_c1_immersed, _C1_SLOPE_IMMERSED)])
    def test_h_paraxial_is_the_real_reference_height(self, builder, slope):
        surfaces = builder()
        d = distortion_vs_field(surfaces, _WL, 0.4, n_points=5)
        for i in range(1, len(d.theta_deg)):
            want = slope * np.tan(np.radians(d.theta_deg[i]))
            assert _rel(d.h_paraxial[i], want) < _TOL, (
                f"theta={d.theta_deg[i]}deg: h_paraxial "
                f"{d.h_paraxial[i]!r} != f_obj*tan {want!r}")

    def test_distortion_vanishes_at_vanishing_field(self):
        """The discriminator: a real system has NO distortion as
        theta -> 0.  Pre-fix an N-BK7 object space reported a constant
        +51.680024% = 100*(n_obj - 1) at every field, including the
        smallest sampled one."""
        surfaces = _d_c1_immersed()
        d = distortion_vs_field(surfaces, _WL, 0.4, n_points=5)
        assert abs(d.distortion_pct[1]) < 1e-3, (
            f"distortion at {d.theta_deg[1]}deg = {d.distortion_pct[1]!r}%; "
            f"it must vanish as theta -> 0.  Pre-fix: +51.680024%.")
        n_obj = _loc_n_obj(surfaces)
        assert _rel(abs(d.distortion_pct[1]), 100.0 * (n_obj - 1.0)) > 0.9, (
            "the 100*(n_obj-1) pre-fix signature is back")

    def test_air_control_distortion_is_bit_identical(self):
        """``n_obj`` is exactly 1.0 for an air object space, so the extra
        multiply is an IEEE no-op: assert the reference equals the raw
        ``efl * tan`` bitwise."""
        surfaces = _d_c1_air()
        assert _loc_n_obj(surfaces) == 1.0
        d = distortion_vs_field(surfaces, _WL, 0.4, n_points=5)
        _M, efl, _b, _f = system_abcd(surfaces, _WL)
        raw = efl * np.tan(np.radians(d.theta_deg))
        assert np.array_equal(d.h_paraxial, raw), (
            "air-object h_paraxial moved off the pre-fix efl*tan values")

    @pytest.mark.parametrize('builder', [_d_c1_air, _d_c1_immersed])
    def test_distortion_grid_uses_the_same_reference(self, builder):
        surfaces = builder()
        n_obj = _loc_n_obj(surfaces)
        _M, efl, _b, _f = system_abcd(surfaces, _WL)
        g = distortion_grid(surfaces, _WL, 0.4, n_grid=3)
        want = n_obj * efl * np.tan(np.radians(g.theta_x_deg))
        assert np.allclose(g.paraxial_x[0, :], want, rtol=_TOL, atol=0)
        assert np.allclose(g.paraxial_y[:, 0], want, rtol=_TOL, atol=0)
        if n_obj == 1.0:
            assert np.array_equal(g.paraxial_x[0, :],
                                  efl * np.tan(np.radians(g.theta_x_deg)))
        else:
            assert _rel(g.paraxial_x[0, -1],
                        efl * np.tan(np.radians(g.theta_x_deg[-1]))) > 0.3


# ----------------------------------------------------------------------
# C2 prescriptions + oracle
# ----------------------------------------------------------------------
def _sd(r, gb, ga):
    return {'radius': r, 'conic': 0.0, 'aspheric_coeffs': None,
            'glass_before': gb, 'glass_after': ga}


def _rx(name, surfs, thicks, obj_d=500e-3):
    return {'name': name, 'aperture_diameter': 12e-3,
            'object_distance': obj_d, 'field_max_m': 5e-3,
            'surfaces': surfs, 'thicknesses': thicks}


def _rx_air():
    p = la.make_singlet(R1=51.5e-3, R2=float('inf'), d=4e-3,
                        glass='N-BK7', aperture=12e-3)
    p['object_distance'] = 500e-3
    p['field_max_m'] = 5e-3
    return p


def _rx_immersed_image():
    """Air object space, N-BK7 IMAGE space."""
    return _rx('W4cImmersedImage',
               [_sd(51.5e-3, 'air', 'N-BK7'), _sd(-80e-3, 'N-BK7', 'air'),
                _sd(120e-3, 'air', 'N-BK7')], [4e-3, 10e-3, 0.0])


def _rx_immersed_object():
    """N-BK7 OBJECT space, air image space."""
    return _rx('W4cImmersedObject',
               [_sd(-60e-3, 'N-BK7', 'air'), _sd(60e-3, 'air', 'N-SF2'),
                _sd(-60e-3, 'N-SF2', 'air')], [8e-3, 4e-3, 0.0])


def _oracle_img_d(rx, wl=_WLW):
    """Real-ray paraxial image distance for the FINITE conjugate.

    A ray leaves the axial object point at ``object_distance`` in front
    of surface 0 and reaches height ``_HP`` there (slope ``_HP/d``); its
    axis crossing past the last surface is read in GLOBAL z, the frame
    ``img_d_m`` is consumed in.
    """
    surfaces = surfaces_from_prescription(rx)
    d = float(rx['object_distance'])
    u = _HP / d
    y, uo = _state_at(surfaces, wl, len(surfaces) - 1, _HP, u)
    return -y / uo


_C2 = {
    'air': dict(builder=_rx_air, exact=+1.218190954e-01,
                prefix=+1.218190954e-01, immersed=False),
    'immersed_image': dict(builder=_rx_immersed_image, exact=+7.055793985e-02,
                           prefix=+4.156959023e-02, immersed=True),
    'immersed_object': dict(builder=_rx_immersed_object,
                            exact=+3.554900347e-02,
                            prefix=+3.402321540e-02, immersed=True),
}


class TestW4cGaussConjugateSolve:
    """C2: ``n_obj/u + n_img/v = 1/efl``, not the air-only form."""

    @pytest.mark.parametrize('name', sorted(_C2))
    def test_img_d_m_matches_the_exact_real_ray_oracle(self, name):
        spec = _C2[name]
        rx = spec['builder']()
        wfe = la.eval_image_plane_wfe(rx, _WLW, n_pupil=11,
                                      image_plane='paraxial',
                                      sphere_tangent='vertex')
        ex = _oracle_img_d(rx)
        assert _rel(ex, spec['exact']) < _TOL, "pinned oracle drifted"
        assert _rel(wfe.img_d_m, ex) < _TOL, (
            f"{name}: img_d_m {wfe.img_d_m!r} vs exact real-ray "
            f"{ex!r} (rel {_rel(wfe.img_d_m, ex):.3e}); pre-fix was "
            f"{spec['prefix']!r}.")
        assert _rel(wfe.img_d_m_paraxial, ex) < _TOL

    @pytest.mark.parametrize('name', ['immersed_image', 'immersed_object'])
    def test_prefix_air_only_values_are_rejected(self, name):
        spec = _C2[name]
        rx = spec['builder']()
        wfe = la.eval_image_plane_wfe(rx, _WLW, n_pupil=11,
                                      image_plane='paraxial',
                                      sphere_tangent='vertex')
        assert _rel(wfe.img_d_m, spec['prefix']) > 0.02, (
            f"{name}: img_d_m {wfe.img_d_m!r} is the pre-fix air-only "
            f"Gauss value {spec['prefix']!r}.")

    def test_the_two_index_errors_are_not_a_common_factor(self):
        """Mechanism: ``n_obj`` and ``n_img`` enter the equation
        DIFFERENTLY, so no single scale factor can repair both -- which
        is why the fix threads them separately.  Same glass on both
        designs, yet -41.1% vs -4.3%."""
        r_img = _rel(_C2['immersed_image']['prefix'],
                     _C2['immersed_image']['exact'])
        r_obj = _rel(_C2['immersed_object']['prefix'],
                     _C2['immersed_object']['exact'])
        assert r_img > 0.35 and r_obj < 0.10, (r_img, r_obj)
        assert abs(r_img - r_obj) > 0.25

    def test_air_conjugates_are_bit_identical(self):
        """Structural + empirical: both factors are exactly 1.0, and the
        reported value equals the air-only expression rebuilt here."""
        rx = _rx_air()
        surfaces = surfaces_from_prescription(rx)
        assert _loc_n_obj(surfaces, _WLW) == 1.0
        assert _loc_n_img(surfaces, _WLW) == 1.0
        fod = first_order_data(surfaces, _WLW)
        u_pp = float(rx['object_distance']) + fod.pp_object_z
        pre_fix = 1.0 / ((1.0 / fod.efl) - (1.0 / u_pp)) + fod.pp_image_z
        wfe = la.eval_image_plane_wfe(rx, _WLW, n_pupil=11,
                                      image_plane='paraxial',
                                      sphere_tangent='vertex')
        assert wfe.img_d_m == pre_fix, (
            f"air img_d_m {wfe.img_d_m!r} != the pre-fix expression "
            f"{pre_fix!r}; air must be bit-identical.")

    def test_explicit_img_d_m_still_bypasses_the_solve(self):
        """Regression guard: passing ``img_d_m`` must skip the Gauss
        branch entirely on every design."""
        rx = _rx_immersed_image()
        wfe = la.eval_image_plane_wfe(rx, _WLW, n_pupil=9, img_d_m=55e-3,
                                      image_plane='paraxial',
                                      sphere_tangent='vertex')
        assert wfe.img_d_m == 55e-3


# ----------------------------------------------------------------------
# C3 designs + oracle
# ----------------------------------------------------------------------
def _d_c3_plain():
    """Mirrorless control."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=6e-3, is_stop=True),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=25e-3, semi_diameter=np.inf),
    ]


def _d_c3_folded(t_sign=+1.0):
    """The SAME optics with a flat fold appended: ODD parity."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=6e-3, is_stop=True),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=10e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=t_sign * 15e-3, semi_diameter=np.inf,
                is_mirror=True),
    ]


def _d_c3_two_mirrors():
    """EVEN parity: must be bit-identical (no mapping)."""
    return [
        Surface(radius=50e-3, glass_before='air', glass_after='N-BK7',
                thickness=5e-3, semi_diameter=6e-3, is_stop=True),
        Surface(radius=-50e-3, glass_before='N-BK7', glass_after='air',
                thickness=10e-3, semi_diameter=np.inf),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=-8e-3, semi_diameter=np.inf, is_mirror=True),
        Surface(radius=np.inf, glass_before='air', glass_after='air',
                thickness=7e-3, semi_diameter=np.inf, is_mirror=True),
    ]


_C3 = {
    'plain': _d_c3_plain,
    'folded': _d_c3_folded,
    'folded_negt': lambda: _d_c3_folded(-1.0),
    'two_mirrors_even': _d_c3_two_mirrors,
}


def _probe_height_at_appended_plane(surfaces, wl, image_distance):
    """Height of a collimated probe at the appended image plane.

    Zero (to roundoff) iff the plane really is at the paraxial focus --
    the operational definition of a correctly placed image plane.
    """
    surfs = _append_image_plane(surfaces, image_distance)
    y, _u = _state_at(surfs, wl, len(surfaces), _HP, 0.0)
    return y


class TestW4cBflAsGlobalZDistance:
    """C3: the unfolded -> global-z mapping on every ``bfl`` consumer."""

    @pytest.mark.parametrize('name', sorted(_C3))
    def test_appended_image_plane_lands_on_the_focus(self, name):
        surfaces = _C3[name]()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        y = _probe_height_at_appended_plane(surfaces, _WL, bfl)
        assert abs(y) < 1e-4 * _HP, (
            f"{name}: a collimated probe launched at {_HP} m arrives "
            f"{y!r} m off axis at the plane appended at bfl = {bfl!r}; "
            f"the plane is not at the focus.")

    @pytest.mark.parametrize('name', ['folded', 'folded_negt'])
    def test_the_unmapped_placement_demonstrably_missed(self, name):
        """Pre-fix behaviour, rebuilt in-process: writing ``+bfl``
        straight into the thickness puts the plane 2*|bfl| away."""
        surfaces = _C3[name]()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        assert _loc_parity_sign(surfaces) == -1.0
        # the mapping is inside _append_image_plane, so pass s*bfl to
        # reproduce the pre-fix (unmapped) thickness
        y_bad = _probe_height_at_appended_plane(
            surfaces, _WL, _loc_parity_sign(surfaces) * bfl)
        assert abs(y_bad) > 0.5 * _HP, (
            f"{name}: the unmapped placement should miss the focus by "
            f"~1.5x the launch height; got {y_bad!r}")

    def test_folded_spot_equals_the_mirrorless_control(self):
        """Independent cross-check with no oracle: folding a system with
        a FLAT mirror cannot change its spot size.  Pre-fix the folded
        RMS read 3.946987e-03 m vs the control's 2.130362e-05 m."""
        rms = {}
        for name in ('plain', 'folded', 'folded_negt'):
            surfaces = _C3[name]()
            _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
            s = _loc_parity_sign(surfaces)
            t = np.linspace(-1.0, 1.0, 9)
            rb = RayBundle(x=np.zeros(9), y=t * 4e-3, z=np.zeros(9),
                           L=np.zeros(9), M=np.zeros(9), N=np.ones(9),
                           wavelength=_WL, alive=np.ones(9, bool),
                           opd=np.zeros(9))
            res = trace(rb, surfaces, _WL)
            rms[name] = _propagate_rms(res, s * bfl)
        assert _rel(rms['folded'], rms['plain']) < 1e-9, rms
        assert _rel(rms['folded_negt'], rms['plain']) < 1e-9, rms
        assert _rel(rms['plain'], 2.130362e-05) < 1e-6
        # and the pre-fix (unmapped) centre is demonstrably far off
        surfaces = _C3['folded']()
        _M, _efl, bfl, _ffl = system_abcd(surfaces, _WL)
        t = np.linspace(-1.0, 1.0, 9)
        rb = RayBundle(x=np.zeros(9), y=t * 4e-3, z=np.zeros(9),
                       L=np.zeros(9), M=np.zeros(9), N=np.ones(9),
                       wavelength=_WL, alive=np.ones(9, bool),
                       opd=np.zeros(9))
        res = trace(rb, surfaces, _WL)
        assert _propagate_rms(res, bfl) / rms['plain'] > 100.0

    @pytest.mark.parametrize('name', ['plain', 'two_mirrors_even'])
    def test_even_parity_is_bit_identical(self, name):
        """No mapping for even parity: the appended thickness must equal
        ``image_distance`` exactly."""
        surfaces = _C3[name]()
        assert _loc_parity_sign(surfaces) == +1.0
        d = 12.34e-3
        out = _append_image_plane(surfaces, d)
        assert out[-2].thickness == d, (
            f"{name}: thickness {out[-2].thickness!r} != {d!r}")

    @pytest.mark.parametrize('name', ['folded', 'folded_negt'])
    def test_odd_parity_thickness_carries_the_documented_sign(self, name):
        surfaces = _C3[name]()
        d = 12.34e-3
        out = _append_image_plane(surfaces, d)
        assert out[-2].thickness == -d
        # ...and it is S11-1's own expression
        n_mir = sum(1 for s in surfaces if s.is_mirror)
        assert out[-2].thickness == (-1.0) ** n_mir * d

    def test_field_aberration_sweep_runs_on_a_folded_system(self):
        """The ``_propagate_rms`` site: pre-fix the whole +-|bfl|/20
        search window sat on the wrong side of the vertex, so the
        reported focus shifts were meaningless.  After the fix the folded
        system's on-axis shift matches its mirrorless control's."""
        out = {}
        for name in ('plain', 'folded'):
            surfaces = _C3[name]()
            r = field_aberration_sweep(surfaces, _WL, [0.0, 0.5],
                                       semi_aperture=4e-3, n_fan=9, n_z=41)
            out[name] = (float(r.sagittal_focus_shift[0]),
                         float(r.tangential_focus_shift[0]))
        assert np.isfinite(out['folded'][0])
        assert abs(out['folded'][0] - out['plain'][0]) < 5e-5, out
        assert abs(out['folded'][1] - out['plain'][1]) < 5e-5, out


# ----------------------------------------------------------------------
# Cross-cutting: the conversion factors come from ONE source
# ----------------------------------------------------------------------
def test_w4c_factors_come_from_raytrace_seidel():
    """Both analysis modules import the factors from
    ``raytrace.seidel`` rather than re-deriving them -- the R-1 / S11-1
    drift lesson.  Pinned structurally so a future edit cannot quietly
    fork a second copy of either convention."""
    import lumenairy.analysis.field as _f
    import lumenairy.analysis.image_plane_wfe as _i
    from lumenairy.raytrace import seidel as _s
    assert _f._mirror_parity_sign is _s._mirror_parity_sign
    assert _f._object_space_index is _s._object_space_index
    assert _i._object_space_index is _s._object_space_index
    assert _i._image_space_index is _s._image_space_index
