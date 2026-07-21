"""
v5.1.0 split: ray-surface intersection + vector Snell helpers.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  Hosts the Newton-iteration / spherical
fast-path ray-surface intersection, vector Snell refraction, mirror
reflection, axial transfer, and the Zemax-style coordinate break
applier.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

import numpy as np

from ._conic_core import reflect_mirror, refract_snell
from .surface import (
    RAY_APERTURE,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_OK,
    RAY_TIR,
    _field_frame_active,
    _surface_normal,
    _surface_sag_derivatives_xy,
    _surface_sag_xy,
)

# ============================================================================
# Ray-surface intersection (Newton iteration)
# ============================================================================

def _intersect_surface(rays, surface, n_medium=1.0):
    """Find intersection of rays with a surface centred at z=0.

    Modifies ``rays`` in place: updates (x, y, z) to the intersection
    point and **accumulates OPD for the path travelled during the
    intersection** (in the medium the rays are currently in, which is
    the medium *before* this surface).  Rays that miss the clear
    aperture are marked dead.

    Parameters
    ----------
    rays : RayBundle
        Rays approaching the surface.  Their current z is the plane
        the previous transfer left them on (typically the vertex plane
        of this surface, z = 0 in this surface's frame).
    surface : Surface
        The surface to intersect.
    n_medium : float, default 1.0
        Refractive index of the medium the rays are travelling through
        on the way to this surface (i.e. the *glass_before* of this
        surface).  Used to accumulate the OPL of the small "vertex
        plane to actual sag" leg, which is critical for thick lenses
        with curved surfaces -- without it the trace under-counts (or
        over-counts, for concave) the true ray path between
        intersections.

    Notes
    -----
    v4.12.1 (Track C): pure-spherical surfaces (``conic == 0``, no
    aspheric / biconic / freeform / coord-break extensions, finite
    radius) take a "Newton-skip" fast path that uses the analytical
    ray-sphere quadratic root directly.  For a sphere this root is
    the exact intersection (modulo LSB rounding), so the legacy 10-
    iteration Newton refinement does at most one ~1e-17 step before
    converging -- on a 1k-ray doublet trace this represents the bulk
    of the per-surface cost.  The surface-normal pathway in
    :func:`_refract` / :func:`_reflect` still routes through
    :func:`_surface_sag_derivatives_xy` (numerical-radial-derivative
    based), so the normal rounding behaviour is bit-identical to
    pre-v4.12.1.  A v4.12.0 attempt that also switched the spherical
    normal to the analytic ``(x/R, y/R, (z-R)/R)`` form (matching
    :mod:`jax_trace`) compounded a 1.17e-3 cross-backend rel error in
    the Maslov asymptotic test -- this conservative variant avoids
    that drift.
    """
    R = surface.radius
    kc = surface.conic
    asph = surface.aspheric_coeffs
    radius_y = getattr(surface, 'radius_y', None)
    freeform = getattr(surface, 'freeform', None)
    # N10a: a FIELD-FRAME decenter / tilt / freeform sag_callable (the
    # displaced-pointwise convention) makes the surface z = sag(x-dx, y-dy) +
    # tilt-ramp, which neither the analytic ray-sphere quadratic nor the flat
    # fast path represents -- route those through Newton (which evaluates the
    # field-frame ``_surface_sag_xy`` / ``_surface_sag_derivatives_xy``).
    field_frame = _field_frame_active(surface)

    # v4.12.1: detect the pure-spherical fast path.  Requires
    # finite R, conic == 0, no aspherics, no biconic axis, no
    # freeform departure.  Coord-break surfaces never reach
    # :func:`_intersect_surface` -- the trace loop dispatches them
    # via :func:`_apply_coord_break` -- so no guard is needed here.
    is_pure_spherical = (
        (not np.isinf(R))
        and kc == 0.0
        and not asph
        and radius_y is None
        and freeform is None
        and not field_frame
    )

    if np.isinf(R) and not asph and not field_frame:
        # Flat surface: intersect at z = 0
        # t such that z + N*t = 0  =>  t = -z / N
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(rays.alive & (np.abs(rays.N) > 1e-30),
                         -rays.z / rays.N, 0.0)
    elif is_pure_spherical:
        # ---- v4.12.1 Track C: Newton-skip fast path -----------------
        # For a sphere ``x^2 + y^2 + (z - R)^2 = R^2`` the ray-surface
        # intersection is the smaller-magnitude root of a quadratic in
        # t (with |(L, M, N)| = 1 by construction so a = 1).  This is
        # exactly the formula the legacy Newton initial-guess block
        # used, so the resulting ``t`` is bit-identical to the legacy
        # path *before* its first Newton iteration -- the iteration
        # body would then make an ~1e-17 LSB-level correction and
        # converge.  Skipping the Newton loop yields a per-ray drift
        # at that LSB level (1.075e-17 measured on a 1k-ray doublet
        # trace) -- conservative enough that the cross-backend
        # Maslov asymptotic test still passes its 1e-3 tolerance.
        x0, y0, z0 = rays.x, rays.y, rays.z
        Ld, Md, Nd = rays.L, rays.M, rays.N

        dx, dy, dz = x0, y0, z0 - R
        b = 2.0 * (Ld * dx + Md * dy + Nd * dz)
        c = dx ** 2 + dy ** 2 + dz ** 2 - R ** 2
        disc = b ** 2 - 4.0 * c
        disc_safe = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc_safe)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        # Direction-aware root pick: choose the intersection whose
        # parametric distance is closer to zero (the NEAR root for
        # the ray's current direction).  v5.4.1 (audit P1): replaces
        # the prior direction-blind ``t = t1 if R > 0 else t2`` which
        # produced wrong-side-of-sphere results for any backward-
        # propagating ray (N=-1 after a reflection).  Audit reproducer
        # at docs/audits/AUDIT_V5_4_0_2026_05_25.md Part 5 P1: a
        # Cassegrain chief ray landed 20cm PAST the secondary vertex
        # on the wrong side of the R=-0.3m hyperbola.  See
        # analysis/ghost.py:_ghost_intersect for the original
        # workaround (now a thin alias).
        t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)

        # disc < 0: ray entirely misses the sphere.  v5.4.6 (audit P3-3):
        # disc == 0 is the tangent case -- a real single-point intersection
        # (e.g. a chief ray sitting exactly on the stop edge), so it is now
        # accepted (>= 0) rather than dropped as "missed".  Exact tangency
        # is measure-zero numerically, so this does not change vignetting
        # for ordinary rays.
        disc_pos = disc >= 0
        t = np.where(disc_pos, t, 0.0)
        missed_init = (~disc_pos) & rays.alive

        # Mark these missed rays as dead so they don't propagate
        # garbage through subsequent surfaces.  Convergence flag is
        # synthesised true for non-missed rays (Newton loop guarantee
        # parity).
        if missed_init.any():
            rays.alive = rays.alive & ~missed_init
            if rays.error_code is not None:
                first_failure = missed_init & (rays.error_code == RAY_OK)
                rays.error_code = np.where(
                    first_failure, RAY_MISSED_SURFACE, rays.error_code
                )
    else:
        # ---- Newton's method (conic / aspheric / biconic / freeform) -
        # The surface is z = sag(x, y).  We need to find t such that
        # z + N*t = sag(x + L*t, y + M*t).
        t = np.zeros(rays.n_rays)

        # Initial guess: paraxial approximation for a sphere
        if not np.isinf(R):
            # For a sphere: x^2 + y^2 + (z-R)^2 = R^2
            # Approximate t from the ray-sphere intersection
            x0, y0, z0 = rays.x, rays.y, rays.z
            Ld, Md, Nd = rays.L, rays.M, rays.N

            # Centre of curvature at (0, 0, R)
            dx, dy, dz = x0, y0, z0 - R
            a = 1.0  # L^2 + M^2 + N^2
            b = 2.0 * (Ld * dx + Md * dy + Nd * dz)
            c = dx ** 2 + dy ** 2 + dz ** 2 - R ** 2
            disc = b ** 2 - 4 * a * c
            disc_safe = np.maximum(disc, 0.0)
            sqrt_disc = np.sqrt(disc_safe)

            # Pick the smaller positive root (closer intersection)
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)
            # Direction-aware root pick: choose the intersection whose
            # parametric distance is closer to zero (the NEAR root for
            # the ray's current direction).  v5.4.1 (audit P1): replaces
            # the prior direction-blind ``t = t1 if R > 0 else t2`` which
            # produced wrong-side-of-sphere results for any backward-
            # propagating ray (N=-1 after a reflection) -- the Newton
            # loop below would then "converge" to that bogus initial
            # guess.  Audit reproducer at
            # docs/audits/AUDIT_V5_4_0_2026_05_25.md Part 5 P1.
            t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)

            # Track rays whose initial-guess sphere intersection has no
            # real root (disc < 0).  These never reach the surface; mark
            # them as missed.  Pre-4.10 the t = 0 silent fallback caused
            # such rays to land at z=0 with a residual sag, then masquerade
            # as converged once Newton found |dt|<1e-15 at a stuck point.
            missed_init = (disc < 0) & rays.alive
            # v5.4.6 (audit P3-3): keep the tangent case (disc == 0); only
            # disc < 0 (no real root) is a true miss.  Matches missed_init.
            t = np.where(disc >= 0, t, 0.0)
        else:
            missed_init = np.zeros(rays.n_rays, dtype=bool)
            # Flat surface with aspheric terms only: start at z=0
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(np.abs(rays.N) > 1e-30, -rays.z / rays.N, 0.0)

        # Newton iterations
        converged = np.zeros(rays.n_rays, dtype=bool)
        for _ in range(10):
            xi = rays.x + rays.L * t
            yi = rays.y + rays.M * t
            zi = rays.z + rays.N * t
            sag_i = _surface_sag_xy(xi, yi, surface)

            # Residual: F(t) = zi - sag(xi, yi) = 0
            F = zi - sag_i

            # Derivative of F with respect to t:
            # dF/dt = N - dz/dx * L - dz/dy * M
            dz_dx, dz_dy = _surface_sag_derivatives_xy(xi, yi, surface)
            dF_dt = rays.N - dz_dx * rays.L - dz_dy * rays.M

            # Newton step.  When |dF_dt| < eps the ray is grazing or
            # tangent to the surface; mark these as stuck rather than
            # silently zero-stepping (which the < 1e-15 convergence
            # check would then accept as "converged").
            stuck = np.abs(dF_dt) <= 1e-30
            dt = np.where(stuck, 0.0, F / np.where(stuck, 1.0, dF_dt))
            t = t - dt

            # Per-ray convergence: |dt| < 1e-15 AND not stuck-with-residual.
            converged = (np.abs(dt) < 1e-15) & (~stuck | (np.abs(F) < 1e-12))
            if converged.all():
                break

        # Rays that never converged (Newton stuck or disc<0) are missed.
        missed_final = (~converged | missed_init) & rays.alive
        if missed_final.any():
            rays.alive = rays.alive & ~missed_final
            if rays.error_code is not None:
                first_failure = missed_final & (rays.error_code == RAY_OK)
                rays.error_code = np.where(
                    first_failure, RAY_MISSED_SURFACE, rays.error_code
                )

    # Update ray positions
    t = np.where(rays.alive, t, 0.0)
    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = rays.z + rays.N * t

    # Accumulate OPL for the vertex-plane -> actual-sag-intersection
    # leg.  ``t`` is the parametric distance along the ray (with
    # |(L,M,N)| = 1 by construction), so |t| is the geometric path
    # length.  Use the SIGNED contribution: a negative t (which
    # happens when the surface is concave and the ray has already
    # passed it after the previous transfer) corresponds to back-
    # tracking, and we should subtract the over-counted OPL.
    rays.opd = rays.opd + n_medium * t

    # Vignette rays outside the clear aperture
    if np.isfinite(surface.semi_diameter):
        h_sq = rays.x ** 2 + rays.y ** 2
        clipped = (h_sq > surface.semi_diameter ** 2) & rays.alive
        if clipped.any():
            rays.alive = rays.alive & ~clipped
            if rays.error_code is not None:
                # First-failure-wins: only set RAY_APERTURE on rays
                # that were alive up to this surface.
                rays.error_code = np.where(clipped, RAY_APERTURE,
                                             rays.error_code)


# ============================================================================
# Vector Snell's law (refraction and reflection)
# ============================================================================

def _refract(rays, surface, n1, n2):
    """Apply vector Snell's law at the surface.

    Updates direction cosines (L, M, N) in place.  Rays that undergo
    total internal reflection are marked dead.

    Convention: n̂ points into the incident medium (against the
    incoming ray).  cos_i = -(d̂ · n̂) > 0.

    Refracted direction:
        d̂_t = mu * d̂_i + (mu * cos_i - cos_t) * n̂
    where mu = n1 / n2.
    """
    nx, ny, nz = _surface_normal(rays.x, rays.y, surface)

    # S3-10: vector Snell law via the backend-agnostic shared core
    # (raytrace._conic_core.refract_snell).  The core orients the normal
    # against the ray and applies Snell's law; ``eta_sq = mu ** 2``
    # preserves this site's scalar-power form (see the shared-core
    # docstring), and the default (no-op) TIR guard + a clamping ``sqrt``
    # reproduce the former ``cos_t = sqrt(max(1 - sin2_t, 0))`` exactly.
    # The TIR mask ``disc_r < 0`` is bit-identical to the former
    # ``sin2_t > 1`` (``disc_r = 1 - sin2_t`` is exact by Sterbenz near
    # the boundary).  This helper never mutates ``rays``; the in-place
    # alive / error-code / renormalise policy below is unchanged.
    mu = n1 / n2
    Lp, Mp, Np, nx, ny, nz, _cos_i, _disc_r, tir = refract_snell(
        rays.L, rays.M, rays.N, nx, ny, nz, mu, mu ** 2,
        sqrt=lambda a: np.sqrt(np.maximum(a, 0.0)), where=np.where)

    # Total internal reflection check
    newly_tir = tir & rays.alive
    rays.alive = rays.alive & ~tir
    if newly_tir.any() and rays.error_code is not None:
        # First-failure-wins: RAY_TIR overwrites only RAY_OK entries.
        rays.error_code = np.where(newly_tir, RAY_TIR, rays.error_code)

    # Refracted direction: d_t = mu * d_i + (mu * cos_i - cos_t) * n̂
    rays.L = np.where(rays.alive, Lp, rays.L)
    rays.M = np.where(rays.alive, Mp, rays.M)
    rays.N = np.where(rays.alive, Np, rays.N)

    # Renormalise.  If the direction vector magnitude collapsed to
    # zero (arithmetic fault: NaN-propagating refraction, degenerate
    # geometry, etc.), flag the ray dead with RAY_NAN instead of
    # silently promoting (0, 0, 0) to a bogus unit vector along
    # the small-epsilon direction.
    mag = np.sqrt(rays.L ** 2 + rays.M ** 2 + rays.N ** 2)
    _degenerate = (mag < 1e-30) | ~np.isfinite(mag)
    if np.any(_degenerate):
        new_fault = _degenerate & (rays.error_code == RAY_OK)
        rays.error_code = np.where(new_fault, RAY_NAN, rays.error_code)
        # 4.11.1: also flag the ray dead in ``alive`` -- pre-4.11.1
        # only the error_code was set, so downstream alive-based masks
        # treated NaN-direction rays as still active and propagated
        # garbage through subsequent surfaces.
        rays.alive = rays.alive & ~_degenerate
    mag = np.maximum(mag, 1e-30)
    rays.L /= mag
    rays.M /= mag
    rays.N /= mag

    # Accumulate OPD at this surface
    # OPD contribution from the refraction surface itself is zero
    # (OPD is accumulated during transfer between surfaces)


def _reflect(rays, surface):
    """Reflect rays at a mirror surface.

    Updates direction cosines in place.

    Convention: n̂ points into the incident medium (against the
    incoming ray).  cos_i = -(d̂ · n̂) > 0.

    Reflected direction:
        d̂_r = d̂_i + 2 * cos_i * n̂
    """
    nx, ny, nz = _surface_normal(rays.x, rays.y, surface)

    # S3-10: vector reflection via the backend-agnostic shared core
    # (raytrace._conic_core.reflect_mirror).  The core orients the
    # normal against the ray and applies ``d_r = d_i + 2 cos_i n``; it
    # never mutates ``rays``, so the in-place renormalise / degenerate
    # policy below is unchanged.
    Lp, Mp, Np, nx, ny, nz, _cos_i = reflect_mirror(
        rays.L, rays.M, rays.N, nx, ny, nz, where=np.where)
    rays.L = Lp
    rays.M = Mp
    rays.N = Np

    # Renormalise.  Flag degenerate rays as RAY_NAN rather than
    # silently promoting (0, 0, 0) to a unit vector.
    mag = np.sqrt(rays.L ** 2 + rays.M ** 2 + rays.N ** 2)
    _degenerate = (mag < 1e-30) | ~np.isfinite(mag)
    if np.any(_degenerate):
        new_fault = _degenerate & (rays.error_code == RAY_OK)
        rays.error_code = np.where(new_fault, RAY_NAN, rays.error_code)
        # 4.11.1: also flag dead -- pre-4.11.1 only error_code was set.
        rays.alive = rays.alive & ~_degenerate
    mag = np.maximum(mag, 1e-30)
    rays.L /= mag
    rays.M /= mag
    rays.N /= mag


def _transfer(rays, thickness, n_medium):
    """Transfer rays by the given axial thickness in a medium of index n.

    Translates ray positions so they arrive at the next surface vertex
    plane (z = 0) and accumulates OPD.
    """
    if thickness == 0:
        return

    # Transfer: advance each ray along its direction until it reaches
    # z = thickness (the next surface vertex plane).
    # t = (thickness - z) / N
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(rays.alive & (np.abs(rays.N) > 1e-30),
                     (thickness - rays.z) / rays.N, 0.0)

    # Accumulate OPD: geometric path * refractive index.
    # RT-1: use the SIGNED path (n*t), matching ``_intersect_surface``'s
    # vertex->sag leg convention -- a negative t means the ray already
    # crossed the next vertex plane (overlapping-sag geometry), and the
    # over-counted OPL must be SUBTRACTED, not added via ``abs``.  Byte-
    # identical for well-formed prescriptions (forward legs have t > 0;
    # post-mirror back-propagation uses negative thicknesses that still
    # yield t > 0), but the two primitives now implement the same
    # convention for the telescoping-OPL model to hold.
    rays.opd = rays.opd + n_medium * t

    rays.x = rays.x + rays.L * t
    rays.y = rays.y + rays.M * t
    rays.z = np.zeros_like(rays.z)  # reset to vertex of next surface


def _apply_coord_break(rays, surface):
    """3.7.0: Apply a Zemax-style coordinate break to the ray bundle.

    A coord-break shifts the local coordinate frame: subsequent
    surfaces are then expressed in the new frame.  To re-express the
    rays in the new frame we apply the *inverse* of the frame
    transformation:

    * Decenter (dx, dy) → subtract from (x, y).
    * Tilt about X / Y / Z by angles (tx, ty, tz) → rotate ray
      position and direction by the inverse of those rotations
      (i.e., by ``Rx(-tx) @ Ry(-ty) @ Rz(-tz)``).

    Order follows Zemax PARM 6:

    * 0 (default): decenter first, then tilts.
    * 1          : tilts first, then decenter.

    This is purely a frame transform -- no propagation, no OPL, no
    intersection.  The trace loop should call this for surfaces
    where ``surface.is_coordbrk`` is True and skip the usual
    intersect / refract / reflect path.
    """
    if not surface.is_coordbrk:
        return
    dx = float(surface.decenter_x_m)
    dy = float(surface.decenter_y_m)
    tx = np.radians(float(surface.tilt_x_deg))
    ty = np.radians(float(surface.tilt_y_deg))
    tz = np.radians(float(surface.tilt_z_deg))
    order = int(getattr(surface, 'coordbrk_order', 0) or 0)

    def _decenter():
        if dx:
            rays.x = rays.x - dx
        if dy:
            rays.y = rays.y - dy

    def _rot_x(theta):
        if theta == 0.0:
            return
        # 3.7.1: optical-convention frame rotation by +theta about X
        # (Zemax / Code-V convention) is the inverse of the math
        # right-hand-rule rotation.  To express a ray vector in the
        # new frame we apply Rx_math(+theta), i.e.
        #   y' = c*y - s*z;   z' = s*y + c*z.
        # The previous (3.7.0) implementation used Rx_math(-theta),
        # which made the 3D layout's post-fold orientation opposite
        # to the 2D layout's.
        c, s = np.cos(theta), np.sin(theta)
        y_n =  c * rays.y - s * rays.z
        z_n =  s * rays.y + c * rays.z
        rays.y, rays.z = y_n, z_n
        M_n =  c * rays.M - s * rays.N
        N_n =  s * rays.M + c * rays.N
        rays.M, rays.N = M_n, N_n

    def _rot_y(theta):
        if theta == 0.0:
            return
        c, s = np.cos(theta), np.sin(theta)
        # Optical convention: Ry_math(+theta) applied to the ray.
        x_n =  c * rays.x + s * rays.z
        z_n = -s * rays.x + c * rays.z
        rays.x, rays.z = x_n, z_n
        L_n =  c * rays.L + s * rays.N
        N_n = -s * rays.L + c * rays.N
        rays.L, rays.N = L_n, N_n

    def _rot_z(theta):
        if theta == 0.0:
            return
        c, s = np.cos(theta), np.sin(theta)
        # Optical convention: Rz_math(+theta) applied to the ray.
        x_n =  c * rays.x - s * rays.y
        y_n =  s * rays.x + c * rays.y
        rays.x, rays.y = x_n, y_n
        L_n =  c * rays.L - s * rays.M
        M_n =  s * rays.L + c * rays.M
        rays.L, rays.M = L_n, M_n

    def _tilts():
        # Intrinsic rotation order: X, then Y, then Z (Zemax default).
        _rot_x(tx)
        _rot_y(ty)
        _rot_z(tz)

    # Zemax PARM 6 semantics (see Surface.coordbrk_order docstring):
    #   0 -> decenter, then tilt (the default)
    #   1 -> tilt, then decenter
    # Pre-4.10 the two branches were swapped, so every imported folded
    # design with the Zemax default got a tilt-then-decenter frame
    # transform instead of decenter-then-tilt.
    if order == 1:
        _tilts()
        _decenter()
    else:
        _decenter()
        _tilts()


__all__ = [
    '_intersect_surface',
    '_refract',
    '_reflect',
    '_transfer',
    '_apply_coord_break',
]
