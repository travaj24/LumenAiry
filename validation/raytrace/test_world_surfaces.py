"""Validation tests for :func:`lumenairy.world_surfaces_from_prescription`
(4.4.0).

Verifies:

* Straight (un-folded) prescriptions produce identity rotations and
  cumulative-along-z origins, and the resulting world trace gives
  bit-identical image-plane ray positions to the local-frame
  :func:`trace`.
* Coord-break tilts rotate the local +z axis as expected.
* Coord-break decenters shift the origin in the current local frame.
* The ``order`` field (Zemax PARM 6: 0 = decenter then tilt; 1 = tilt
  then decenter) is honoured.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from _harness import Harness

import numpy as np

import lumenairy as la


H = Harness('world-surfaces')


def _singlet():
    return la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                              glass='N-BK7', aperture=10e-3)


# ----------------------------------------------------------------------
# Straight prescription -> identity rotations
# ----------------------------------------------------------------------

def t_straight_singlet_identity_rotation():
    surfs = la.world_surfaces_from_prescription(_singlet())
    ok = all(np.allclose(s.world_R, np.eye(3)) for s in surfs)
    return ok, f'all rotations identity: {ok}'


H.run('straight singlet: identity world_R', t_straight_singlet_identity_rotation)


def t_straight_singlet_cumulative_z_origin():
    surfs = la.world_surfaces_from_prescription(_singlet())
    z0 = float(surfs[0].world_origin[2])
    z1 = float(surfs[1].world_origin[2])
    # Singlet has 3 mm thickness between S1 and S2.
    return (abs(z0) < 1e-12 and abs(z1 - 3e-3) < 1e-12), \
        f'S0 z = {z0*1e3} mm, S1 z = {z1*1e3} mm (expected 0 and 3)'


H.run('straight singlet: world_origin advances along +z',
      t_straight_singlet_cumulative_z_origin)


def t_straight_trace_matches_local():
    """World trace and local trace must give identical image-plane ray
    positions for a straight (un-folded) prescription."""
    presc = _singlet()
    wsurfs = la.world_surfaces_from_prescription(presc)
    lsurfs = la.surfaces_from_prescription(presc)
    rays = la.make_rings(4e-3, 3, 8, 0.0, 1.31e-6)
    r_world = la.trace_world(rays, wsurfs, 1.31e-6)
    r_local = la.trace(rays, lsurfs, 1.31e-6)
    max_diff = float(np.max(np.abs(
        r_world.image_rays.x - r_local.image_rays.x)))
    return max_diff < 1e-12, \
        f'max |x_world - x_local| = {max_diff:.3e}'


H.run('straight singlet: world trace == local trace',
      t_straight_trace_matches_local)


# ----------------------------------------------------------------------
# Coord-break tilt
# ----------------------------------------------------------------------

def t_tilt_x_90deg_rotates_z_to_neg_y():
    """A 90° tilt about the local x-axis should swing the z-axis to -y
    (right-hand rule)."""
    presc = _singlet()
    folded = dict(presc)
    folded['coord_breaks'] = [{
        'surf_num': 99, 'decenter_x_m': 0.0, 'decenter_y_m': 0.0,
        'tilt_x_deg': 90.0, 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
        'order': 0, 'thickness_m': 0.0,
    }]
    folded['surfaces'] = list(folded['surfaces']) + [{
        'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
        'surf_num': 100,
    }]
    folded['thicknesses'] = list(folded['thicknesses']) + [0.0]
    surfs = la.world_surfaces_from_prescription(folded)
    z_axis = surfs[-1].world_R[:, 2]
    expected = np.array([0.0, -1.0, 0.0])
    err = float(np.max(np.abs(z_axis - expected)))
    return err < 1e-12, \
        f'post-tilt z-axis = {z_axis}, expected {expected}'


H.run('coord-break: 90 deg tilt-x rotates +z to -y',
      t_tilt_x_90deg_rotates_z_to_neg_y)


def t_tilt_advances_origin_along_new_z():
    """A 90° tilt-x followed by a 20 mm cb-thickness should put the
    post-cb origin at y = -20 mm (since +z rotated to -y)."""
    presc = _singlet()
    folded = dict(presc)
    folded['coord_breaks'] = [{
        'surf_num': 99, 'decenter_x_m': 0.0, 'decenter_y_m': 0.0,
        'tilt_x_deg': 90.0, 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
        'order': 0, 'thickness_m': 0.02,
    }]
    folded['surfaces'] = list(folded['surfaces']) + [{
        'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air',
        'surf_num': 100,
    }]
    folded['thicknesses'] = list(folded['thicknesses']) + [0.0]
    surfs = la.world_surfaces_from_prescription(folded)
    origin = surfs[-1].world_origin
    # Expected: starts at z=3mm after the singlet, tilts 90 about x at
    # that point, then advances 20mm along the new local +z (which is
    # world -y).  Final origin = (0, -20mm, 3mm).
    expected = np.array([0.0, -0.020, 0.003])
    err = float(np.max(np.abs(origin - expected)))
    return err < 1e-12, \
        f'origin after fold = {origin*1e3} mm, expected {expected*1e3} mm'


H.run('coord-break: thickness advances origin along new +z',
      t_tilt_advances_origin_along_new_z)


# ----------------------------------------------------------------------
# Coord-break decenter
# ----------------------------------------------------------------------

def t_decenter_shifts_origin():
    """A pure decenter (no tilt, order=0) shifts the post-cb origin by
    the decenter vector in the pre-tilt frame.  At the start of the
    chain that frame is identity, so the shift is in world coords."""
    presc = _singlet()
    folded = dict(presc)
    folded['coord_breaks'] = [{
        'surf_num': 0,   # before any optical surface
        'decenter_x_m': 0.001, 'decenter_y_m': 0.002,
        'tilt_x_deg': 0.0, 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
        'order': 0, 'thickness_m': 0.0,
    }]
    surfs = la.world_surfaces_from_prescription(folded)
    # The first optical surface should now sit at (1, 2, 0) mm.
    expected_xy = np.array([0.001, 0.002])
    got_xy = surfs[0].world_origin[:2]
    err = float(np.max(np.abs(got_xy - expected_xy)))
    return err < 1e-12, \
        f'first-surface xy = {got_xy*1e3} mm, expected {expected_xy*1e3} mm'


H.run('coord-break: pure decenter shifts origin', t_decenter_shifts_origin)


# ----------------------------------------------------------------------
# order field (PARM 6) — tilt-first vs decenter-first
# ----------------------------------------------------------------------

def t_order_field_changes_origin_for_combined_break():
    """A decenter + tilt with order=0 (decenter first, then tilt) vs
    order=1 (tilt first, then decenter) produces different origins
    when the rotation is non-trivial."""
    base = _singlet()
    cb0 = {
        'surf_num': 0, 'decenter_x_m': 0.001, 'decenter_y_m': 0.0,
        'tilt_x_deg': 0.0, 'tilt_y_deg': 90.0, 'tilt_z_deg': 0.0,
        'thickness_m': 0.0,
    }
    p_dec_first = dict(base, coord_breaks=[dict(cb0, order=0)])
    p_tilt_first = dict(base, coord_breaks=[dict(cb0, order=1)])
    s0 = la.world_surfaces_from_prescription(p_dec_first)[0]
    s1 = la.world_surfaces_from_prescription(p_tilt_first)[0]
    delta = float(np.linalg.norm(s0.world_origin - s1.world_origin))
    return delta > 1e-6, \
        f'order-0 vs order-1 origin diff = {delta*1e3:.3f} mm'


H.run('coord-break: order=0 vs order=1 give different origins',
      t_order_field_changes_origin_for_combined_break)


if __name__ == '__main__':
    import sys
    sys.exit(H.summary())
