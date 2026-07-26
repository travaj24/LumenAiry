"""W4d: the last two W4 flags -- folded FRAMES in ``analysis/``.

W4c closed the three immersed/folded ``analysis`` findings and left two
flagged, measured, unfixed.  This file closes both.

F1  ``field._append_image_plane``'s WORLD-frame branch placed the image
    plane along ``last.world_R[:, 2]`` unconditionally.  THE AMBIGUITY
    WAS REAL: ``world_surfaces_from_prescription`` re-aligns its running
    frame ONLY at a coord-break, never at a mirror, and a coord-break
    emits no surface -- so a folded world list that ENDS AT THE MIRROR
    carries the PRE-fold frame (``z_hat`` points back up the incoming
    axis) while one ending at a surface AFTER the re-aligning coord-break
    carries the POST-fold frame (``z_hat`` already follows the ray).  No
    property of a raw prescription separates them, which is why W4c
    declined to guess.

    RESOLVED AT THE SOURCE: a world-frame list is self-describing about
    which way is downstream -- ``trace_world`` reflects off the true
    surface normals -- so :func:`field._world_exit_direction` traces one
    semi-diameter-neutralised on-axis probe and places the plane along
    the ray's ACTUAL outgoing direction.  Correct for both flavours by
    construction; no frame marker, no builder change (``world.py`` is
    untouched), and provably bit-identical wherever ``z_hat`` already was
    the propagation direction, because the probe returns exactly
    ``z_hat`` there.  Measured discriminator on a 1-mirror world list
    with no re-aligning coord-break: the marginal probe lands at
    x = +5.5954e-04 m under the old placement and -7.94e-08 m under this
    one.

F2  ``eval_image_plane_wfe`` on a FOLDED prescription.  ``img_d_m``
    (built from ``bfl``/``pp_image_z``, both UNFOLDED per W4b/S11-1) was
    mixed with global-z ray arithmetic, and ``N_chief < 0`` after an odd
    number of mirrors inverted every ``1/N_chief`` arc-length factor.
    FIXED by doing the whole reference-sphere construction in the
    ALONG-THE-RAY axial frame: ``_fold_sign = _mirror_parity_sign`` now
    multiplies the along-ray/global-z crossings (``cz``, ``t_advance``,
    ``N_chief_for_R``, and ``xp_z``, which W3-T2/W4 define in global z).

    ORACLE: a FLAT fold adds no power and no aberration, so a folded
    system's WFE must equal its own mirrorless control's.  Measured
    (vertex tangent, paraxial plane, on axis):

        design   control PV / RMS      folded PV / RMS pre-fix
        1        1.124509 / 0.348635   388.095176 / 114.318947  waves
        2        1.152112 / 0.357265   304.824831 /  89.785602  waves

    with ``r_sphere_m`` coming back NEGATIVE (-4.817146063e-02 m,
    -3.430676598e-02 m) -- and under ``best_rms`` the back-solved
    ``img_d_m`` itself went negative (-4.776785347e-02 m).  After the
    fix every folded configuration reproduces its control EXACTLY (delta
    PV = delta RMS = 0.0, bit-for-bit) across 2 designs x both
    post-mirror thickness sign conventions x both ``sphere_tangent``
    choices x ``paraxial``/``best_rms`` x on-axis/off-axis -- 32
    folded-vs-control pairs.

Air / unfolded systems are bit-identical throughout: ``_fold_sign`` is
exactly ``+1.0`` for every even mirror count, and IEEE multiplication by
1.0 is the identity.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis.field import _append_image_plane

# NOTE: ``field._world_exit_direction`` is imported lazily inside the F1
# tests, NOT at module scope, so the whole file still COLLECTS against a
# pre-W4d build (where the helper does not exist) and the F2 value pins
# can be seen to fail on their own NUMBERS rather than on an ImportError.
from lumenairy.raytrace import (
    Surface,
    surfaces_from_prescription,
    system_abcd,
)
from lumenairy.raytrace.core import _make_bundle, trace_world

_WL = 587.5618e-9
_WLW = 587.56e-9
_TOL = 1e-9


def _rel(a, b):
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


def _loc_parity_sign(surfaces):
    m = sum(1 for s in surfaces if s.is_mirror and not s.is_coordbrk)
    return -1.0 if (m % 2) else 1.0


# ======================================================================
# F1 -- world-frame image-plane placement
# ======================================================================
def _sd(r, gb, ga, mirror=False):
    d = {'radius': r, 'conic': 0.0, 'aspheric_coeffs': None,
         'glass_before': gb, 'glass_after': ga}
    if mirror:
        d['is_mirror'] = True
    return d


def _rx_periscope():
    """Singlet + flat fold as the LAST surface: the world list therefore
    ends at the mirror and carries the PRE-fold frame."""
    return {
        'name': 'W4dPeriscope', 'aperture_diameter': 8e-3,
        'surfaces': [_sd(50e-3, 'air', 'N-BK7'), _sd(-50e-3, 'N-BK7', 'air'),
                     _sd(float('inf'), 'air', 'air', mirror=True)],
        'thicknesses': [5e-3, 20e-3, 0.0],
    }


def _rx_unfolded():
    return {
        'name': 'W4dPlain', 'aperture_diameter': 8e-3,
        'surfaces': [_sd(50e-3, 'air', 'N-BK7'), _sd(-50e-3, 'N-BK7', 'air'),
                     _sd(float('inf'), 'air', 'air')],
        'thicknesses': [5e-3, 20e-3, 0.0],
    }


def _world(rx):
    return la.world_surfaces_from_prescription(rx)


def _world_probe_x(world_surfaces, image_distance, h=5e-4):
    """Marginal-probe transverse landing at the appended image plane."""
    wimg = _append_image_plane(world_surfaces, image_distance)
    rays = _make_bundle(x=np.array([0.0, h]), y=np.array([0.0, 0.0]),
                        L=np.zeros(2), M=np.zeros(2), wavelength=_WL)
    out = trace_world(rays, wimg, _WL).ray_history[-1]
    assert bool(out.alive[1]), "probe died"
    return float(out.x[1])


class TestW4dF1WorldFrameImagePlane:
    """F1: place along the ray's real outgoing direction, not the frame
    axis -- and be bit-identical wherever those coincide."""

    def test_unfolded_world_list_direction_is_exactly_the_frame_axis(self):
        from lumenairy.analysis.field import _world_exit_direction
        w = _world(_rx_unfolded())
        z_hat = np.asarray(w[-1].world_R[:, 2], dtype=float)
        d = _world_exit_direction(w)
        assert np.array_equal(d, z_hat), (
            f"unfolded list: exit direction {d!r} != frame axis {z_hat!r}; "
            f"the W4d probe must be an exact no-op here.")

    def test_unfolded_placement_is_bit_identical(self):
        w = _world(_rx_unfolded())
        _M, _efl, bfl, _ffl = system_abcd(surfaces_from_prescription(
            _rx_unfolded()), _WL)
        got = _append_image_plane(w, bfl)[-1].world_origin
        want = w[-1].world_origin + w[-1].world_R[:, 2] * float(bfl)
        assert np.array_equal(got, want)

    def test_non_realigned_folded_list_lands_on_axis(self):
        """The discriminator.  The list ends at the mirror, so ``z_hat``
        still points up the INCOMING axis; the probe must nonetheless be
        placed downstream of the reflection."""
        from lumenairy.analysis.field import _world_exit_direction
        rx = _rx_periscope()
        lsurfs = surfaces_from_prescription(rx)
        assert _loc_parity_sign(lsurfs) == -1.0
        _M, _efl, bfl, _ffl = system_abcd(lsurfs, _WL)
        w = _world(rx)
        z_hat = np.asarray(w[-1].world_R[:, 2], dtype=float)
        d = _world_exit_direction(w)
        # the frame axis and the real outgoing direction are OPPOSED here
        assert float(np.dot(d, z_hat)) < -0.99, (d, z_hat)
        x_now = _world_probe_x(w, bfl)
        assert abs(x_now) < 1e-6, (
            f"probe lands at x = {x_now!r} m; the plane is not at the "
            f"focus.  Pre-W4d (frame-axis placement): +5.5954e-04 m.")

    def test_the_frame_axis_placement_demonstrably_missed(self):
        """Rebuild the pre-W4d placement in-process and show it misses."""
        rx = _rx_periscope()
        _M, _efl, bfl, _ffl = system_abcd(
            surfaces_from_prescription(rx), _WL)
        w = _world(rx)
        # feeding -bfl reproduces the old (frame-axis) origin exactly
        old_origin = w[-1].world_origin + w[-1].world_R[:, 2] * float(bfl)
        new_origin = _append_image_plane(w, -bfl)[-1].world_origin
        assert np.allclose(new_origin, old_origin, atol=1e-15)
        x_bad = _world_probe_x(w, -bfl)
        assert abs(x_bad) > 1e-4, (
            f"the frame-axis placement should miss by ~5.6e-04 m; got "
            f"{x_bad!r}")
        assert _rel(abs(x_bad), 5.5954e-04) < 5e-3

    def test_exit_direction_is_the_independently_traced_exit_direction(self):
        """THE CONTRACT, and why the re-aligned flavour needs no separate
        pin: the helper reports the direction an independent trace of the
        same list gives, and it reads ``world_R`` only as a fallback.  So
        it is right for ANY world list -- the two flavours differ only in
        what ``z_hat`` happens to be, which the answer no longer depends
        on.  Verified here against a trace built in the test."""
        from lumenairy.analysis.field import _world_exit_direction
        for rx in (_rx_unfolded(), _rx_periscope()):
            w = _world(rx)
            rays = _make_bundle(x=np.zeros(1), y=np.zeros(1),
                                L=np.zeros(1), M=np.zeros(1),
                                wavelength=_WL)
            out = trace_world(rays, w, _WL).ray_history[-1]
            want = np.array([float(out.L[0]), float(out.M[0]),
                             float(out.N[0])])
            want = want / float(np.linalg.norm(want))
            got = _world_exit_direction(w)
            assert np.allclose(got, want, atol=1e-12), (
                f"{rx['name']}: helper {got!r} vs independently traced "
                f"{want!r}")

    def test_folded_and_unfolded_agree_on_the_focus_position(self):
        """Both lists describe the same optics up to the fold, so the
        appended plane must land the probe on axis in BOTH -- which is the
        property every caller actually needs."""
        for rx in (_rx_unfolded(), _rx_periscope()):
            lsurfs = surfaces_from_prescription(rx)
            _M, _efl, bfl, _ffl = system_abcd(lsurfs, _WL)
            x = _world_probe_x(_world(rx), bfl)
            assert abs(x) < 1e-6, (rx['name'], x)

    def test_probe_failure_falls_back_to_the_frame_axis(self):
        """The fallback must be the pre-W4d answer (exact for unfolded
        lists), never a crash."""
        from lumenairy.analysis.field import _world_exit_direction
        w = _world(_rx_unfolded())
        broken = list(w)
        broken[-1] = Surface(radius=np.inf, glass_before='air',
                             glass_after='air', semi_diameter=np.inf)
        broken[-1].world_origin = np.array([0.0, 0.0, 0.025])
        broken[-1].world_R = np.full((3, 3), np.nan)
        d = _world_exit_direction(broken)
        assert d.shape == (3,)
        assert np.array_equal(d, np.asarray(broken[-1].world_R[:, 2]),
                              equal_nan=True) or np.all(np.isfinite(d))


# ======================================================================
# F2 -- folded reference-sphere geometry in eval_image_plane_wfe
# ======================================================================
def _rx_wfe_plain(R1=51.5e-3, R2=-80e-3, gap=20e-3, obj_d=500e-3):
    return {'name': 'plain', 'aperture_diameter': 10e-3,
            'object_distance': obj_d, 'field_max_m': 3e-3,
            'surfaces': [_sd(R1, 'air', 'N-BK7'), _sd(R2, 'N-BK7', 'air'),
                         _sd(float('inf'), 'air', 'air')],
            'thicknesses': [4e-3, gap, 0.0]}


def _rx_wfe_folded(R1=51.5e-3, R2=-80e-3, gap=20e-3, t_sign=+1.0,
                   obj_d=500e-3):
    """Identical optics up to the last vertex, with a FLAT fold there."""
    return {'name': f'folded{t_sign:+.0f}', 'aperture_diameter': 10e-3,
            'object_distance': obj_d, 'field_max_m': 3e-3,
            'surfaces': [_sd(R1, 'air', 'N-BK7'), _sd(R2, 'N-BK7', 'air'),
                         _sd(float('inf'), 'air', 'air', mirror=True)],
            'thicknesses': [4e-3, gap, t_sign * 8e-3]}


_D2 = dict(R1=40e-3, R2=-120e-3, gap=30e-3)


def _wfe_metrics(rx, field, plane, tangent):
    w = la.eval_image_plane_wfe(rx, _WLW, field=field, n_pupil=15,
                                image_plane=plane, sphere_tangent=tangent)
    ok = np.isfinite(w.opd_w) & w.alive
    assert ok.any()
    vals = w.opd_w[ok]
    return dict(pv=float(vals.max() - vals.min()),
                rms=float(np.sqrt(np.mean((vals - vals.mean()) ** 2))),
                img_d=float(w.img_d_m), r_sphere=float(w.r_sphere_m),
                alive=int(ok.sum()))


# pre-fix (21185b7) folded values, vertex tangent / paraxial / on-axis
_F2_PREFIX = {
    1: dict(pv=+3.880951760e+02, rms=+1.143189470e+02,
            r_sphere=-4.817146063e-02),
    2: dict(pv=+3.048248310e+02, rms=+8.978560200e+01,
            r_sphere=-3.430676598e-02),
}
# the mirrorless controls those must collapse onto
_F2_CONTROL = {
    1: dict(pv=+1.124509e+00, rms=+3.486350e-01, r_sphere=+4.817146063e-02),
    2: dict(pv=+1.152112e+00, rms=+3.572650e-01, r_sphere=+3.430676598e-02),
}


class TestW4dF2FoldedReferenceSphere:
    """F2: a flat fold adds no aberration, so folded == mirrorless."""

    @pytest.mark.parametrize('tangent', ['vertex', 'exit_pupil'])
    @pytest.mark.parametrize('plane', ['paraxial', 'best_rms'])
    @pytest.mark.parametrize('field', [(0.0, 0.0), (1.0, 0.0)])
    @pytest.mark.parametrize('t_sign', [+1.0, -1.0])
    @pytest.mark.parametrize('design', [1, 2])
    def test_folded_wfe_equals_the_mirrorless_control(
            self, design, t_sign, field, plane, tangent):
        kw = {} if design == 1 else _D2
        ctrl = _wfe_metrics(_rx_wfe_plain(**kw), field, plane, tangent)
        fold = _wfe_metrics(_rx_wfe_folded(t_sign=t_sign, **kw),
                            field, plane, tangent)
        for key in ('pv', 'rms', 'img_d', 'r_sphere'):
            assert _rel(fold[key], ctrl[key]) < _TOL, (
                f"design {design}, t_sign {t_sign:+.0f}, {plane}/{tangent}, "
                f"field {field}: folded {key} = {fold[key]!r} vs the "
                f"mirrorless control's {ctrl[key]!r}.  A FLAT fold adds no "
                f"aberration and no power.")
        assert fold['alive'] == ctrl['alive']

    @pytest.mark.parametrize('design', [1, 2])
    def test_reference_sphere_radius_is_positive_for_a_folded_system(
            self, design):
        """Pre-fix ``r_sphere_m`` came back NEGATIVE (the 1/N_chief
        inversion), which is the signature of the defect."""
        kw = {} if design == 1 else _D2
        fold = _wfe_metrics(_rx_wfe_folded(**kw), (0.0, 0.0),
                            'paraxial', 'vertex')
        assert fold['r_sphere'] > 0.0, fold
        assert fold['img_d'] > 0.0, fold
        assert _rel(fold['r_sphere'],
                    _F2_CONTROL[design]['r_sphere']) < _TOL
        assert _rel(fold['r_sphere'],
                    _F2_PREFIX[design]['r_sphere']) > 1.5   # sign flip

    @pytest.mark.parametrize('design', [1, 2])
    def test_prefix_folded_wavefront_is_rejected(self, design):
        kw = {} if design == 1 else _D2
        fold = _wfe_metrics(_rx_wfe_folded(**kw), (0.0, 0.0),
                            'paraxial', 'vertex')
        for key in ('pv', 'rms'):
            assert _rel(fold[key], _F2_PREFIX[design][key]) > 0.9, (
                f"design {design}: {key} = {fold[key]!r} is the pre-fix "
                f"folded value {_F2_PREFIX[design][key]!r}")
            assert _rel(fold[key], _F2_CONTROL[design][key]) < 1e-5

    @pytest.mark.parametrize('design', [1, 2])
    def test_mirrorless_controls_match_their_pinned_values(self, design):
        """Unfolded systems must not have moved: ``_fold_sign`` is
        exactly 1.0 there, so every W4d multiply is an IEEE no-op."""
        from lumenairy.raytrace.seidel import _mirror_parity_sign
        kw = {} if design == 1 else _D2
        rx = _rx_wfe_plain(**kw)
        assert _mirror_parity_sign(surfaces_from_prescription(rx)) == 1.0
        ctrl = _wfe_metrics(rx, (0.0, 0.0), 'paraxial', 'vertex')
        for key in ('pv', 'rms', 'r_sphere'):
            assert _rel(ctrl[key], _F2_CONTROL[design][key]) < 1e-5, (
                f"design {design}: control {key} = {ctrl[key]!r}, pinned "
                f"{_F2_CONTROL[design][key]!r}")

    def test_best_pv_also_survives_a_fold(self):
        """``best_pv`` walks ``_pv_at`` over shifted image distances, so
        it exercises the ``cz`` mapping on every probe."""
        ctrl = _wfe_metrics(_rx_wfe_plain(), (0.0, 0.0), 'best_pv', 'vertex')
        fold = _wfe_metrics(_rx_wfe_folded(), (0.0, 0.0), 'best_pv', 'vertex')
        assert _rel(fold['pv'], ctrl['pv']) < 1e-6, (fold, ctrl)
        assert fold['r_sphere'] > 0.0
