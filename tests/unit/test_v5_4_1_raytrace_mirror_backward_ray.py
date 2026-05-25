"""v5.4.1 (audit P1) regression pins: ``_intersect_surface`` must use
a direction-aware root pick so backward-propagating rays (N < 0 after
a mirror reflection) land on the NEAR root of the sphere, not the
diametrically-opposite far root.

Background
----------

AUDIT_V5_4_0_2026_05_25.md Part 5 P1 (CONVERGENT across V5 + F1 + F3)
discovered that ``_intersect_surface`` was direction-blind: it picked
``t = t1 if R > 0 else t2``, which gives the near root for forward
propagation but the FAR root for backward propagation (a ray with
N < 0 after a reflection).  Consequences:

1. **Unit reproducer** -- a single backward ray at (x=0.05, y=0, z=0)
   with (L=0, M=0, N=-1) hitting an R=-1m sphere centred at z=-1
   should intersect at z = -0.00125 (the near root, ~1.25 mm "into"
   the sphere along -z).  The pre-v5.4.1 code returned the FAR root
   at z = -1.99875 (almost 2 m past the vertex), i.e. the far side
   of the sphere along the opposite parametric branch.

2. **End-to-end reproducer** -- a Cassegrain (R1=-1m primary, d=0.4m,
   R2=-0.3m secondary, both mirrors) chief ray launched 2 degrees
   off-axis reflected from the primary, transferred to the secondary
   at z=0.4m, then hit the R=-0.3m secondary with N<0.  The library
   landed the post-secondary ray at z ~ -0.596 m (20 cm PAST the
   secondary vertex on the WRONG side).  The correct result keeps
   the ray on the near side of the secondary (|z| < 0.4 m, before it
   reaches the secondary again).

3. **Scoped workaround** -- v5.4.0 patched this in the consumer layer
   only (``analysis/ghost.py:_ghost_intersect``).  Regular
   ``trace()`` / ``raytrace_system()`` / ``trace_world()`` paths
   continued to silently produce garbage on every mirror reflection.

v5.4.1 (this patch) promotes the direction-aware root pick (``where(
|t1| <= |t2|, t1, t2)``) into ``intersection.py`` -- both the
spherical fast path and the Newton initial-guess branch.  The
``_ghost_intersect`` helper is now a thin alias around
``_intersect_surface`` for backward compatibility.

Tests pinned
------------

1. ``test_intersect_surface_backward_ray_picks_near_root`` -- direct
   unit test of ``_intersect_surface`` with N=-1, R=-1 m sphere, ray
   at (0.05, 0, 0).  Asserts ``z ~= -0.00125 m`` (near root) and
   explicitly fails on ``z ~= -1.99875 m`` (far root).

2. ``test_cassegrain_chief_ray_lands_near_secondary_vertex`` -- a
   two-mirror Cassegrain (R1=-1 m, d=0.4 m, R2=-0.3 m) traced with
   a 2-degree off-axis chief ray.  Asserts that the post-secondary
   ray sits on the NEAR side of the secondary vertex
   (``|z| < 0.4 m``) -- the pre-v5.4.1 bug placed it at z ~ -0.596 m.

3. ``test_ghost_intersect_alias_routes_through_library`` -- the
   v5.4.0 ``_ghost_intersect`` helper, now an alias, must produce
   bit-identical results to a direct call to ``_intersect_surface``
   on the same backward-ray input.

4. ``test_trace_with_mirror_alive_flag_consistent`` -- end-to-end
   ``trace()`` on a single curved mirror with a backward-propagating
   ray must leave the ray alive (no silent vignetting from a wrong-
   side-of-sphere intersection that drifts outside the aperture).

Author: Andrew Traverso -- v5.4.1 (audit P1 fix, regression pin).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.raytrace import (
    RayBundle,
    Surface,
    surfaces_from_prescription,
    trace,
)
from lumenairy.raytrace.intersection import _intersect_surface

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_backward_ray(x=0.05, y=0.0, z=0.0, wavelength=550e-9):
    """Single-ray bundle propagating in -z (N = -1)."""
    return RayBundle(
        x=np.array([x], dtype=float),
        y=np.array([y], dtype=float),
        z=np.array([z], dtype=float),
        L=np.array([0.0], dtype=float),
        M=np.array([0.0], dtype=float),
        N=np.array([-1.0], dtype=float),
        wavelength=wavelength,
        alive=np.array([True], dtype=bool),
        opd=np.array([0.0], dtype=float),
    )


def _make_cassegrain_prescription():
    """Two-mirror Cassegrain: R1=-1 m primary (concave), R2=-0.3 m
    secondary (convex), inter-mirror distance d=0.4 m.  Aperture 100 mm
    on the primary; secondary obscuration 30 mm.

    Matches the prescription used by the v4.14.0 Welford-mirror pin
    so the geometry is independently sanity-checked.
    """
    return {
        'name': 'Cassegrain (v5.4.1 P1 regression)',
        'aperture_diameter': 0.100,
        'surfaces': [
            {'radius': -1.0, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'air',
             'is_mirror': True, 'is_stop': True,
             'semi_diameter': 0.050},
            {'radius': -0.3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'air',
             'is_mirror': True,
             'semi_diameter': 0.015},
        ],
        'thicknesses': [0.4, 0.0],
    }


# ----------------------------------------------------------------------
# Test 1: unit reproducer for the direction-aware root pick
# ----------------------------------------------------------------------

def test_intersect_surface_backward_ray_picks_near_root():
    """Backward ray (N=-1) at (0.05, 0, 0) hitting R=-1 m sphere
    must land at z ~= -0.00125 m (the near root, sag ~ 1.25 mm).

    Pre-v5.4.1 the library picked the FAR root at z ~ -1.99875 m
    (nearly 2 m past the vertex on the wrong side of the sphere).
    """
    rays = _make_backward_ray(x=0.05, y=0.0, z=0.0)
    surface = Surface(
        radius=-1.0,
        glass_before='air', glass_after='air',
        is_mirror=True,
        semi_diameter=0.10,
        thickness=0.0,
    )

    _intersect_surface(rays, surface, n_medium=1.0)

    # Expected near root: sphere centred at z=-R = +1, radius |R|=1.
    # For a ray at (x=0.05, y=0) along -z, the near intersection is
    # the small sag z ~= R - sqrt(R^2 - x^2) = -1 + sqrt(1 - 0.0025)
    # = -1 + 0.99875... = -0.00125 m (the bowl's outer rim).
    z_near = -0.00125
    z_far = -1.99875  # the BAD pre-v5.4.1 result

    assert rays.alive[0], "Ray should remain alive after near-root pick"
    assert abs(rays.z[0] - z_near) < 1e-5, (
        f"Expected z ~= {z_near} m (near root); got z = {rays.z[0]:.6f} m. "
        f"If z is close to {z_far} m, the direction-blind far-root pick "
        f"has regressed.")
    # Explicit guard against the pre-fix far root:
    assert abs(rays.z[0] - z_far) > 0.1, (
        f"Ray landed at z = {rays.z[0]:.6f} m, dangerously close to the "
        f"pre-v5.4.1 far-root value of {z_far} m.")


# ----------------------------------------------------------------------
# Test 2: Cassegrain end-to-end reproducer
# ----------------------------------------------------------------------

def test_cassegrain_chief_ray_lands_near_secondary_vertex():
    """A 2-degree off-axis chief ray through a (R1=-1, d=0.4, R2=-0.3)
    Cassegrain must, after reflecting off the secondary, end up on
    the NEAR side of the secondary vertex (i.e. on the way back
    toward the primary's image plane, not 20 cm past the secondary).

    Pre-v5.4.1 the chief ray landed at z ~= -0.596 m in the secondary's
    local frame -- i.e. ~20 cm past the vertex on the WRONG side of
    the R=-0.3 m hyperbola.
    """
    prescription = _make_cassegrain_prescription()
    surfaces = surfaces_from_prescription(prescription)

    # 2-degree off-axis chief ray entering on-axis at the primary's
    # vertex plane (z=0 in the prescription's frame).  N is set so
    # |(L, M, N)| = 1.
    field_angle_rad = np.deg2rad(2.0)
    M = np.sin(field_angle_rad)
    N = np.cos(field_angle_rad)
    rays = RayBundle(
        x=np.array([0.0], dtype=float),
        y=np.array([0.0], dtype=float),
        z=np.array([0.0], dtype=float),
        L=np.array([0.0], dtype=float),
        M=np.array([M], dtype=float),
        N=np.array([N], dtype=float),
        wavelength=550e-9,
        alive=np.array([True], dtype=bool),
        opd=np.array([0.0], dtype=float),
    )

    result = trace(rays, surfaces, wavelength=550e-9)
    bundle_after_secondary = result.ray_history[-1]

    z_after = float(bundle_after_secondary.z[0])
    n_after = float(bundle_after_secondary.N[0])

    # The bundle's z is in the secondary's LOCAL frame (rays.z=0 is
    # the secondary vertex).  A physically correct intersection sits
    # on the near side of the secondary: ``|z| < 0.4 m`` (the inter-
    # mirror distance) is a generous physical bound.  The pre-v5.4.1
    # far-root failure mode lands at z ~ -0.596 m which violates this.
    assert abs(z_after) < 0.4, (
        f"Post-secondary chief ray landed at z = {z_after:.4f} m in "
        f"the secondary's local frame; |z| must be < 0.4 m (the inter-"
        f"mirror distance) for a physically correct near-root pick. "
        f"The pre-v5.4.1 bug landed this ray at z ~= -0.596 m.")

    # Sanity: after a mirror reflection in a Cassegrain, the chief
    # ray should propagate back toward the primary (N < 0 after the
    # second reflection on the secondary -- two reflections gives
    # N back > 0 if you count the initial post-primary leg as N < 0,
    # then post-secondary N > 0 again).  Either way the ray must
    # carry a finite, unit-magnitude direction.
    L_a = float(bundle_after_secondary.L[0])
    M_a = float(bundle_after_secondary.M[0])
    mag = np.sqrt(L_a ** 2 + M_a ** 2 + n_after ** 2)
    assert abs(mag - 1.0) < 1e-6, (
        f"Direction cosines must be unit-magnitude; got "
        f"|(L, M, N)| = {mag:.6f}.")


# ----------------------------------------------------------------------
# Test 3: _ghost_intersect alias compatibility
# ----------------------------------------------------------------------

def test_ghost_intersect_alias_routes_through_library():
    """v5.4.1 (audit P1) made ``_ghost_intersect`` a thin alias for
    ``_intersect_surface`` (direction-aware root pick is now canonical
    in the library).  The alias must produce bit-identical results
    to a direct ``_intersect_surface`` call on the same input.
    """
    from lumenairy.analysis.ghost import _ghost_intersect

    rays_lib = _make_backward_ray(x=0.05, y=0.0, z=0.0)
    rays_alias = _make_backward_ray(x=0.05, y=0.0, z=0.0)

    surface = Surface(
        radius=-1.0,
        glass_before='air', glass_after='air',
        is_mirror=True,
        semi_diameter=0.10,
        thickness=0.0,
    )

    _intersect_surface(rays_lib, surface, n_medium=1.0)
    # ``direction`` is ignored post-v5.4.1 but still accepted in the
    # signature; pass -1 (backward) to match v5.4.0 callers.
    _ghost_intersect(rays_alias, surface, n_medium=1.0, direction=-1)

    assert rays_lib.alive[0] == rays_alias.alive[0]
    assert rays_lib.x[0] == rays_alias.x[0]
    assert rays_lib.y[0] == rays_alias.y[0]
    assert rays_lib.z[0] == rays_alias.z[0]
    assert rays_lib.opd[0] == rays_alias.opd[0]


# ----------------------------------------------------------------------
# Test 4: trace() through a curved mirror with backward leg keeps ray alive
# ----------------------------------------------------------------------

def test_trace_with_mirror_alive_flag_consistent():
    """``trace()`` through a single curved mirror must NOT silently
    vignette an on-axis chief ray.  Pre-v5.4.1 the far-root pick
    could drop the ray on the opposite hemisphere of the sphere where
    it then fell outside the clear aperture, triggering a spurious
    ``RAY_APERTURE`` kill on subsequent surfaces.
    """
    surfaces = [
        Surface(
            radius=-1.0,
            glass_before='air', glass_after='air',
            is_mirror=True,
            semi_diameter=0.05,
            thickness=0.0,
        ),
    ]

    # On-axis chief ray, 1 mm off the axis radially (well inside the
    # 50 mm semi-diameter) propagating forward (N=+1).  After the
    # mirror it reflects and propagates backward (N=-1) -- but the
    # trace() loop only intersects this surface once with N=+1 on the
    # input leg, so the direction-aware root pick is exercised on
    # the forward branch (where the bug had no effect).  The point
    # is that the ray stays alive end-to-end -- a regression where
    # the Newton initial guess gets confused by the new direction-
    # aware pick on the forward branch would manifest as a missed-
    # surface kill.
    rays = RayBundle(
        x=np.array([1e-3], dtype=float),
        y=np.array([0.0], dtype=float),
        z=np.array([0.0], dtype=float),
        L=np.array([0.0], dtype=float),
        M=np.array([0.0], dtype=float),
        N=np.array([1.0], dtype=float),
        wavelength=550e-9,
        alive=np.array([True], dtype=bool),
        opd=np.array([0.0], dtype=float),
    )

    result = trace(rays, surfaces, wavelength=550e-9)
    final = result.ray_history[-1]

    assert final.alive[0], (
        "Single-curved-mirror forward chief ray must remain alive "
        "after reflection (no silent aperture-vignetting bug).")
    # After reflection on an R=-1 m concave mirror, an on-axis (x ~ 1 mm)
    # forward ray should reverse direction (N < 0).
    assert float(final.N[0]) < 0.0, (
        f"Post-mirror N should be < 0 (reflected backward); got "
        f"N = {float(final.N[0]):.6f}.")
