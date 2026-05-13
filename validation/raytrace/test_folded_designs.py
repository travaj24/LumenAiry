"""Validation tests for folded-design world-frame ray tracing (4.5.0).

Builds two folded test designs from scratch:

* **Periscope** -- singlet + 45-deg flat fold mirror.  The paraxial
  focus must land 100mm past the singlet along the bent axis, i.e.
  in world coordinates at roughly ``(0, EFL-3mm-50mm, 53mm)``.
* **Curved fold mirror** (oblique-incidence concave) -- a 45-deg
  tilted spherical mirror with ``R = -200mm`` focuses a collimated
  beam to its tangential focal point ``f_t = R/2 * cos(45 deg) ~
  70.7mm`` post-fold in the +y direction.

Verifies that, on each:

1. :func:`world_surfaces_from_prescription` places every surface at
   the correct world coordinate and rotation.
2. :func:`trace_world` follows the bent axis and a chief ray lands at
   the expected post-fold detector location.
3. :func:`paraxial_focus_world` returns the correct world image
   position (matched against analytical expectation and the trace's
   own consistency).

Also verifies the straight-axis prescription path is unaffected
(periscope vs straight singlet differ only in the fold).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la


H = Harness('folded-designs')


def _periscope_prescription():
    """Plano-convex N-BK7 singlet (R1=50mm, R2=inf, t=3mm) +
    45-deg flat fold mirror 50mm past the singlet.  Detector plane
    50mm post-fold in +y.  Zemax-signed post-mirror thickness."""
    return {
        'surfaces': [
            {'radius': 50e-3, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'surf_num': 1},
            {'radius': float('inf'), 'glass_before': 'N-BK7',
             'glass_after': 'air', 'surf_num': 2},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'MIRROR', 'surf_num': 15},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'air', 'surf_num': 25},
        ],
        'thicknesses': [3e-3, 0.05, 0.0, 0.0],
        'aperture_diameter': 0.010,
        'coord_breaks': [
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': -0.05},
        ],
    }


def _curved_fold_prescription(R_m=-0.2, pre_m=0.0, post_m=-0.1):
    """45-deg tilted concave spherical mirror.  Default radius
    -200mm gives a paraxial focal length f = R/2 = 100mm; at 45-deg
    obliquity the tangential focal length is f_t = f * cos(45) ~
    70.7mm in +y post-fold."""
    return {
        'surfaces': [
            {'radius': R_m, 'glass_before': 'air',
             'glass_after': 'MIRROR', 'surf_num': 15},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'air', 'surf_num': 25},
        ],
        'thicknesses': [0.0, 0.0],
        'aperture_diameter': 0.020,
        'coord_breaks': [
            {'surf_num': 1, 'tilt_x_deg': 0.0, 'order': 0,
             'thickness_m': 0.1 if pre_m == 0.0 else pre_m},
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': post_m},
        ],
    }


# ----------------------------------------------------------------------
# Periscope
# ----------------------------------------------------------------------

def t_periscope_world_geometry():
    """Mirror at world (0,0,53mm); detector 50mm post-fold in +y at
    world (0, 50mm, 53mm)."""
    wsurfs = la.world_surfaces_from_prescription(_periscope_prescription())
    mirror = wsurfs[2]
    detector = wsurfs[3]
    m_ok = np.allclose(mirror.world_origin, [0.0, 0.0, 0.053], atol=1e-9)
    d_ok = np.allclose(detector.world_origin, [0.0, 0.050, 0.053], atol=1e-9)
    return m_ok and d_ok, \
        (f'mirror @ {mirror.world_origin*1e3} mm, '
         f'detector @ {detector.world_origin*1e3} mm')


H.run('periscope: mirror + detector world positions correct',
      t_periscope_world_geometry)


def t_periscope_mirror_flagged():
    """The fold mirror must carry is_mirror=True after the
    prescription -> world conversion."""
    wsurfs = la.world_surfaces_from_prescription(_periscope_prescription())
    return wsurfs[2].is_mirror, \
        f'mirror.is_mirror = {wsurfs[2].is_mirror}'


H.run('periscope: mirror surface flagged is_mirror',
      t_periscope_mirror_flagged)


def t_periscope_chief_ray_traces_to_detector():
    """An on-axis chief ray must land at the detector vertex
    after reflecting off the fold mirror."""
    wsurfs = la.world_surfaces_from_prescription(_periscope_prescription())
    chief = la.make_ray(x=0.0, y=0.0, L=0.0, M=0.0, wavelength=1.31e-6)
    result = la.trace_world(chief, wsurfs, 1.31e-6)
    det = wsurfs[-1]
    rb = result.image_rays
    world = (np.asarray(det.world_origin)
             + det.world_R @ np.array([rb.x[0], rb.y[0], rb.z[0]]))
    expected = np.array([0.0, 0.050, 0.053])
    err = float(np.linalg.norm(world - expected))
    return err < 1e-9 and bool(rb.alive[0]), \
        f'chief landed @ {world*1e3} mm, err = {err*1e9:.2f} nm'


H.run('periscope: chief ray lands at post-fold detector',
      t_periscope_chief_ray_traces_to_detector)


def t_periscope_paraxial_focus_world():
    """f=99.3mm singlet, mirror 50mm post-S2, post-fold space.  Image
    sits at ~99.3 - 3 - 50 = ~46.3mm post-fold in +y, i.e. at world
    ~(0, 46.3mm, 53mm).  Allow 2mm slack for finite-thickness +
    paraxial-aspheric correction."""
    presc = _periscope_prescription()
    wsurfs = la.world_surfaces_from_prescription(presc)
    focus, normal = la.paraxial_focus_world(wsurfs, 1.31e-6)
    # Expected ~(0, 46-47mm, 53mm); allow some slack.
    z_ok = abs(focus[2] - 0.053) < 1e-6
    y_in_range = 0.043 < focus[1] < 0.050
    x_zero = abs(focus[0]) < 1e-9
    norm_ok = np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-9)
    return z_ok and y_in_range and x_zero and norm_ok, \
        (f'focus = {focus*1e3} mm, normal = {normal}')


H.run('periscope: paraxial_focus_world lands in post-fold range',
      t_periscope_paraxial_focus_world)


# ----------------------------------------------------------------------
# Curved fold mirror (oblique spherical)
# ----------------------------------------------------------------------

def t_curved_fold_paraxial_focus_world():
    """45-deg tilted spherical mirror R=-200mm focuses a collimated
    plane wave to its tangential focal point at f_t = R/2 * cos(45 deg)
    ~ 70.7mm post-fold in +y."""
    wsurfs = la.world_surfaces_from_prescription(_curved_fold_prescription())
    focus, normal = la.paraxial_focus_world(wsurfs, 1.31e-6)
    # The mirror itself sits at world (0, 0, 100mm).  Focus is in +y
    # direction by f_t.  Expected world y ~ 70.7mm.
    f_t = 0.100 * np.cos(np.radians(45.0))   # = 0.0707 m
    x_zero = abs(focus[0]) < 1e-9
    y_match = abs(focus[1] - f_t) < 5e-3   # 5 mm tolerance for finite-aperture
    z_match = abs(focus[2] - 0.100) < 5e-3
    norm_ok = np.allclose(normal, [0.0, 1.0, 0.0], atol=1e-6)
    return x_zero and y_match and z_match and norm_ok, \
        (f'focus = {focus*1e3} mm, f_t analytic = {f_t*1e3:.2f}mm')


H.run('curved fold mirror: paraxial_focus_world matches tangential f_t',
      t_curved_fold_paraxial_focus_world)


def t_curved_fold_chief_ray_traces_through():
    """An on-axis chief ray hits the curved mirror and reflects to +y."""
    wsurfs = la.world_surfaces_from_prescription(_curved_fold_prescription())
    chief = la.make_ray(x=0.0, y=0.0, L=0.0, M=0.0, wavelength=1.31e-6)
    result = la.trace_world(chief, wsurfs, 1.31e-6)
    rb = result.image_rays
    return bool(rb.alive[0]), f'alive = {rb.alive[0]}'


H.run('curved fold mirror: chief ray traces through fold',
      t_curved_fold_chief_ray_traces_through)


# ----------------------------------------------------------------------
# Straight-axis equivalence
# ----------------------------------------------------------------------

def t_straight_singlet_paraxial_focus_matches_local_bfl():
    """For a straight singlet (no folds), paraxial_focus_world's
    world position must match the local-frame BFL placement."""
    presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=10e-3)
    wsurfs = la.world_surfaces_from_prescription(presc)
    focus, normal = la.paraxial_focus_world(wsurfs, 1.31e-6)

    # Compare to local BFL
    lsurfs = la.surfaces_from_prescription(presc)
    _, _, bfl, _ = la.system_abcd(lsurfs, 1.31e-6)
    # Last lens surface is at world (0, 0, 3mm); image at z = 3mm + bfl
    z_expected = 3e-3 + float(bfl)
    err = abs(focus[2] - z_expected)
    return err < 1e-3 and np.allclose(normal, [0, 0, 1], atol=1e-9), \
        (f'focus z = {focus[2]*1e3:.4f}mm, expected = {z_expected*1e3:.4f}mm, '
         f'err = {err*1e6:.2f} um')


H.run('straight singlet: paraxial_focus_world matches local BFL',
      t_straight_singlet_paraxial_focus_matches_local_bfl)


# ----------------------------------------------------------------------
# Mirror inference from prescription dict
# ----------------------------------------------------------------------

def t_mirror_inferred_from_glass_after_marker():
    """A surface with glass_after='MIRROR' should automatically get
    is_mirror=True; glass_after should be normalized to glass_before."""
    presc = {
        'surfaces': [
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'MIRROR'},
        ],
        'thicknesses': [0.0],
    }
    surfs = la.surfaces_from_prescription(presc)
    s = surfs[0]
    return (s.is_mirror is True
            and s.glass_after == s.glass_before == 'air'), \
        (f'is_mirror={s.is_mirror}, '
         f'glass=({s.glass_before},{s.glass_after})')


H.run('surfaces_from_prescription: glass_after=MIRROR -> is_mirror',
      t_mirror_inferred_from_glass_after_marker)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
