"""
lumenairy.raytrace.jax_trace -- JAX-traceable ray trace.

A focused JAX-aware ray trace for sequential refractive systems.
Walks a prescription's surfaces and interleaves
intersect / refract / aperture-clip / DOE-kick / transfer steps.
The entire trace is JIT-compilable and differentiable via
``jax.grad`` -- supporting end-to-end optical-system optimization.

Surface support
---------------

* **Flat / spherical / conic / aspheric.**  Aspheric surfaces use
  Newton iteration (``jax.lax.fori_loop`` with a fixed iteration
  count) to converge the sag intersection.  Spherical and flat
  surfaces use the closed-form quadratic root.
* **Apertures vignette rays.**  When a surface carries a finite
  ``semi_diameter``, rays whose intersection point falls outside
  it are marked dead in the ``alive`` mask (their state is still
  carried forward without further updates so downstream
  intersections don't NaN out).
* **Diffraction-order kicks.**  Pass a ``surface_diffraction``
  dict mapping surface index to ``(order_x, order_y, period_x,
  period_y)`` to apply a grating kick to the direction cosines
  and add the corresponding linear OPL at that surface.

For full-fidelity tracing including biconic / freeform / mirror
surfaces and per-ray diagnostic codes, use
:func:`lumenairy.raytrace.trace` on NumPy / CuPy arrays.

The output is a :class:`JaxRayState` named-tuple with the same
geometric fields as :class:`RayBundle`.  Convert via the
:func:`jax_state_to_raybundle` helper if you need to interoperate
with NumPy code downstream.

Author: Andrew Traverso
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, NamedTuple, Optional

import numpy as np

from ..backend import JAX_AVAILABLE
from ..glass import get_glass_index
from ._conic_core import conic_sag, conic_sag_derivs, refract_snell

# RT-6 (AUDIT_RAYTRACE_CORE_2026_07_08): the v4.16.1-4.16.3 high-NA
# "paraxial transfer" RuntimeWarning has been RETIRED.  It was built on a
# mischaracterisation: ``_transfer_jax``'s ``t ~= thickness`` step lands the
# ray at a point that is still EXACTLY on the ray line (only the parameter
# along the line differs from the vertex-plane value), and every downstream
# consumer -- the next surface intersect (a line/surface solve is invariant
# to which point on the line you start from) and the OPL accumulator (which
# telescopes: ``n*(thickness + t_int) == n*t_total`` with signed legs) -- is
# invariant to that choice.  A direct trace_jax-vs-NumPy-trace parity check
# at NA up to 0.64 (min|N|~0.77, well past the old 0.95 gate) agrees to
# ~2.4e-9 m, INVARIANT to the gap length across a 100x sweep (0.09 m -> 9 m);
# the residual is the surface-intersection solver tolerance, NOT a
# thickness-scaling paraxial error.  The warning therefore told users to
# distrust results that are correct to sub-ppm -- removed outright.


# Number of Newton iterations for aspheric ray-surface intersection.
# S3-13 (audit AUDIT_V5_24_2): ALIGNED to the NumPy reference's max
# iteration count.  The sag solver in ``intersection.py`` runs
# ``for _ in range(10)`` with an early-exit once every ray converges
# (``|dt| < 1e-15`` and residual ``|F| < 1e-12``); this JAX kernel runs
# a FIXED count (no data-dependent early exit -- required so jit/grad
# trace once).  Pre-fix it stopped at 8, so a marginal asphere whose
# Newton refinement needed the 9th/10th NumPy iteration could land at a
# slightly different ``t`` (and, in float32, alive-mask-diverge from the
# NumPy trace).  Matching the count (10) makes the two paths agree in the
# fully-converged regime; the quadratic tail costs ~2 extra evals but is
# a no-op once converged.  The post-loop residual kill
# (``_ASPHERIC_NEWTON_RESIDUAL_TOL`` / :func:`_newton_residual_tol`)
# mirrors the NumPy ``|F| < 1e-12`` acceptance for the genuinely
# non-convergent rays the fixed count can't early-exit on.
_ASPHERIC_NEWTON_ITERS = 10

# v5.17.1 (audit P3-59): post-Newton residual acceptance tolerance [m].
# The NumPy reference (intersection.py) tracks per-ray convergence and
# kills never-converged rays as RAY_MISSED_SURFACE, accepting a
# stuck-with-residual ray only when |F| < 1e-12.  The JAX kernels run a
# FIXED iteration count (no early exit -- required for jit/grad), so
# an unconverged finite t used to be accepted silently and the ray
# landed off-surface.  After the fixed iterations we evaluate the
# residual F(t) = z - sag(x, y) once and kill rays with |F| above this
# tolerance, mirroring the NumPy 1e-12 residual criterion.
_ASPHERIC_NEWTON_RESIDUAL_TOL = 1e-12


def _newton_residual_tol(dtype):
    """Dtype-aware residual tolerance for the post-Newton convergence
    check (audit P3-59).

    The 1e-12 [m] criterion is calibrated against the always-float64
    NumPy reference.  Under JAX's default float32 (``jax_enable_x64``
    off) a converged Newton residual bottoms out at ~eps(f32) * scale
    (~1e-9 m for mm-scale optics) -- far above 1e-12 -- so the raw
    tolerance would kill every ray.  Scale by the eps ratio ("same
    number of ulps above the rounding floor"): float64 keeps 1e-12,
    float32 gets ~5.4e-4, which still discriminates genuine
    non-convergence (the audit reproducers land 9e-4 .. 3e4 m
    off-surface).  ``dtype`` is static under jit, so this Python-level
    branch is trace-safe.
    """
    return _ASPHERIC_NEWTON_RESIDUAL_TOL * (
        float(np.finfo(dtype).eps) / float(np.finfo(np.float64).eps))


class JaxRayState(NamedTuple):
    """Functional / immutable counterpart to :class:`RayBundle` for
    JAX tracing.

    All fields are JAX arrays of shape (N,) (positions, directions,
    OPD, alive flag).  A NamedTuple is used so JAX can flatten and
    unflatten the state via ``jax.tree_util`` for ``lax.scan``.
    """
    x: object
    y: object
    z: object
    L: object
    M: object
    N: object
    opd: object
    alive: object


def make_jax_ray_state(x, y, z, L, M, N, opd=None, alive=None):
    """Build a :class:`JaxRayState` from arrays."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not installed; install with `pip install jax`")
    import jax.numpy as jnp
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)
    L = jnp.asarray(L)
    M = jnp.asarray(M)
    N = jnp.asarray(N)
    if opd is None:
        opd = jnp.zeros_like(x)
    else:
        opd = jnp.asarray(opd)
    if alive is None:
        alive = jnp.ones_like(x, dtype=bool)
    else:
        alive = jnp.asarray(alive)
    return JaxRayState(x, y, z, L, M, N, opd, alive)


# ----------------------------------------------------------------------
# Sag and sag-derivative for rotationally-symmetric conic + aspheric
# ----------------------------------------------------------------------

def _sag_value_jax(x, y, R_is_inf, R_safe, conic, asph_items):
    """Sag z = sag(x, y) for a rotationally symmetric conic + even-
    aspheric surface.

    Parameters
    ----------
    R_is_inf : bool
        Static flag (Python-time): True if the base surface is flat.
    R_safe : float
        Base radius (1.0 if flat -- caller masks out conic_sag).
    conic : float
        Conic constant.
    asph_items : tuple of (int, float)
        ``(power, coefficient)`` pairs for even aspheric terms.

    S3-10: delegates to the backend-agnostic
    :func:`raytrace._conic_core.conic_sag`.  The static ``R_is_inf`` /
    ``R_safe`` inputs are folded back into the true radius (``inf`` when
    flat) that the where-based shared core consumes -- byte-identical to
    the former inline form for finite, infinite, and huge ``R`` (see
    tests/unit/test_s3_10_conic_core_shared.py).
    """
    import jax.numpy as jnp
    R = float('inf') if R_is_inf else R_safe
    return conic_sag(x, y, R, conic, asph_items, xp=jnp)


def _sag_derivatives_jax(x, y, R_is_inf, R_safe, conic, asph_items):
    """Analytical derivatives dz/dx, dz/dy of the conic + aspheric sag.

    For rotationally symmetric z(h) with h^2 = x^2 + y^2 the NumPy
    reference (core.py:_surface_sag_derivative) uses
        dz/dh = h / (R * sqrt(1 - (1+k) h^2 / R^2))
    so that sign(dz/dh) follows sign(R) -- positive for convex (R>0),
    negative for concave (R<0).  Pre-4.10 this JAX twin used
    ``zx = x / sqrt(R^2 - (1+k) h^2)`` (always positive), so refracted
    rays at concave conic/aspheric surfaces got the wrong transverse
    direction sign.  Mirror the NumPy form here.

    S3-10: delegates to the backend-agnostic
    :func:`raytrace._conic_core.conic_sag_derivs` (byte-identical to the
    former inline form, verified in the shared-core test).
    """
    import jax.numpy as jnp
    R = float('inf') if R_is_inf else R_safe
    return conic_sag_derivs(x, y, R, conic, asph_items, xp=jnp)


# ----------------------------------------------------------------------
# Intersect: closed-form spherical/flat OR Newton-iterated aspheric
# ----------------------------------------------------------------------

def _intersect_jax(state, R, conic, asph_items, n_medium):
    """Intersect rays with a surface centred at z=0 and accumulate OPL.

    Static branches (resolved at trace-build time):
      - flat (R = inf, conic = 0, no aspherics) -> z' = 0
      - spherical (conic = 0, no aspherics)     -> closed-form quadratic
      - conic / aspheric                        -> Newton via fori_loop
    """
    import jax
    import jax.numpy as jnp

    R_is_inf = bool(np.isinf(R))
    has_aspheric = bool(asph_items) or (conic != 0.0)
    R_safe = 1.0 if R_is_inf else float(R)
    eps = 1e-30

    # 4.11.1: track "ray missed surface" so the alive-mask gets
    # propagated downstream instead of silently continuing with t=0
    # and the parent's direction (H-RT-5).  Initialise to all-alive
    # before each branch; tighten as we discover failures.
    miss = jnp.zeros_like(state.alive, dtype=jnp.bool_)

    if R_is_inf and not has_aspheric:
        # Pure flat surface: t = -z / N.
        N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
        t = -state.z / N_safe
        # Grazing ray (N -> 0) means the ray is parallel to the flat
        # surface and never intersects it.
        miss = miss | (jnp.abs(state.N) <= eps)
    else:
        # Spherical initial guess (also exact for conic == 0 / no asph).
        dx = state.x
        dy = state.y
        dz = state.z - R_safe
        b_q = 2.0 * (state.L * dx + state.M * dy + state.N * dz)
        c_q = dx ** 2 + dy ** 2 + dz ** 2 - R_safe ** 2
        disc = b_q ** 2 - 4.0 * c_q
        # 4.11.1 (H-RT-7): double-where on the sqrt-of-disc.  Single
        # ``sqrt(maximum(disc, 0))`` has gradient 1/(2 sqrt(0)) -> inf
        # at the tangent-ray (disc=0) boundary, which poisons
        # jax.grad through every ray that ever grazes a sphere.
        disc_pos = disc > 0
        disc_safe = jnp.where(disc_pos, disc, 1.0)
        sqrt_disc = jnp.where(disc_pos, jnp.sqrt(disc_safe), 0.0)
        # v5.17.1 (audit P3-58): acceptance is disc >= 0, matching the
        # NumPy path (intersection.py, v5.4.6 audit P3-3) which keeps
        # the tangent case disc == 0 as a real single-point
        # intersection (t = -b/2; sqrt_disc is 0 there via the
        # double-where above, so t1 == t2 already equals it).  The
        # sqrt guard stays on the STRICT disc > 0 so the disc = 0
        # gradient singularity keeps being masked (H-RT-7).
        disc_ok = disc >= 0
        t1 = (-b_q - sqrt_disc) / 2.0
        t2 = (-b_q + sqrt_disc) / 2.0
        if R_is_inf:
            # Flat-with-aspherics: start from z = 0 and let Newton run.
            N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
            t0 = -state.z / N_safe
        else:
            # v5.4.6 (audit P1-1): direction-AWARE root pick, mirroring the
            # v5.4.1 NumPy fix in intersection.py.  The old direction-blind
            # ``t1 if R_safe > 0 else t2`` picks the near root only on the
            # forward leg; a backward-propagating ray (N < 0 after a mirror
            # reflection) lands on the diametrically-opposite FAR root.  The
            # near root is min(|t1|, |t2|) regardless of curvature sign.
            t_pick = jnp.where(jnp.abs(t1) <= jnp.abs(t2), t1, t2)
            t0 = jnp.where(disc_ok, t_pick, 0.0)
            # disc < 0 means the ray missed the sphere entirely
            # (disc == 0 tangency is accepted -- audit P3-58).
            miss = miss | (~disc_ok)

        if has_aspheric:
            def body(_, t):
                xi = state.x + state.L * t
                yi = state.y + state.M * t
                zi = state.z + state.N * t
                sag_i = _sag_value_jax(xi, yi, R_is_inf, R_safe,
                                        conic, asph_items)
                F = zi - sag_i
                dz_dx, dz_dy = _sag_derivatives_jax(
                    xi, yi, R_is_inf, R_safe, conic, asph_items)
                dF_dt = state.N - dz_dx * state.L - dz_dy * state.M
                # Double-where pattern for grazing rays (dF_dt -> 0):
                # naive F/dF_dt produces NaN that poisons jax.grad.
                stuck = jnp.abs(dF_dt) <= eps
                dF_safe = jnp.where(stuck, 1.0, dF_dt)
                step = jnp.where(stuck, 0.0, F / dF_safe)
                return t - step

            t = jax.lax.fori_loop(0, _ASPHERIC_NEWTON_ITERS, body, t0)
            # v5.17.1 (audit P3-59): convergence check.  The fixed
            # iteration count has no early-exit/convergence tracking,
            # so a finite-but-unconverged t (steep asphere, grazing
            # incidence) used to be silently accepted where the NumPy
            # path kills the ray as RAY_MISSED_SURFACE.  Evaluate the
            # residual once and fold it into the miss mask.  The
            # residual only feeds a boolean comparison, so no gradient
            # flows through this extra sag evaluation (trace-safe);
            # the ~(<=) form also kills NaN residuals.
            xi = state.x + state.L * t
            yi = state.y + state.M * t
            zi = state.z + state.N * t
            resid = zi - _sag_value_jax(xi, yi, R_is_inf, R_safe,
                                         conic, asph_items)
            miss = miss | ~(jnp.abs(resid)
                            <= _newton_residual_tol(resid.dtype))
        else:
            t = t0

    # 4.11.1 (H-RT-5): any ray that produced a non-finite t in Newton
    # also counts as missed (filter for NaN/Inf rather than silently
    # carrying them through to the next surface).
    miss = miss | (~jnp.isfinite(t))
    t = jnp.where(state.alive & ~miss, t, 0.0)
    new_alive = state.alive & ~miss

    new_x = state.x + state.L * t
    new_y = state.y + state.M * t
    new_z = state.z + state.N * t
    new_opd = state.opd + n_medium * t

    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, new_alive)


# ----------------------------------------------------------------------
# Refract (vector Snell's law, normal from surface gradient)
# ----------------------------------------------------------------------

def _refract_jax(state, R, conic, asph_items, n1, n2):
    """Vector Snell's law at a conic + aspheric surface.

    Surface normal at the intersection point is built from the
    analytical derivatives of the sag.  For a flat surface with no
    aspheric terms this reduces to (0, 0, 1); for a sphere it is
    (x, y, z - R) / R.
    """
    import jax.numpy as jnp

    R_is_inf = bool(np.isinf(R))
    has_aspheric = bool(asph_items) or (conic != 0.0)
    R_safe = 1.0 if R_is_inf else float(R)

    if R_is_inf and not has_aspheric:
        nx = jnp.zeros_like(state.x)
        ny = jnp.zeros_like(state.x)
        nz = jnp.ones_like(state.x)
    elif not has_aspheric:
        # Pure spherical: outward unit normal is (x, y, z - R) / R.
        nx = state.x / R_safe
        ny = state.y / R_safe
        nz = (state.z - R_safe) / R_safe
    else:
        # General sag z = f(x, y).  Surface F = z - f(x, y) = 0;
        # outward normal is grad F = (-fx, -fy, 1), then normalised.
        zx, zy = _sag_derivatives_jax(state.x, state.y,
                                       R_is_inf, R_safe, conic, asph_items)
        nx = -zx
        ny = -zy
        nz = jnp.ones_like(state.x)
        nrm = jnp.sqrt(nx * nx + ny * ny + nz * nz)
        nrm = jnp.maximum(nrm, 1e-30)
        nx = nx / nrm
        ny = ny / nrm
        nz = nz / nrm

    # S3-10: orient the normal + apply vector Snell via the backend-
    # agnostic shared core (raytrace._conic_core.refract_snell).  The
    # injected ``tir_guard`` reproduces the gradient-safe double-where
    # (set the radicand to 1 for TIR rays BEFORE the sqrt so jax.grad
    # doesn't NaN-poison at the boundary), and ``eta_sq = eta ** 2``
    # keeps this site's scalar-power form.  Forward values AND jax.grad
    # are bit-identical to the former inline block (shared-core test).
    eta = n1 / n2
    Lt_r, Mt_r, Nt_r, nx, ny, nz, _cos_i, _disc_r, tir = refract_snell(
        state.L, state.M, state.N, nx, ny, nz, eta, eta ** 2,
        sqrt=lambda z: jnp.sqrt(jnp.maximum(z, 0.0)),
        where=jnp.where,
        tir_guard=lambda t, d: jnp.where(t, 1.0, d))

    Lt = jnp.where(tir, state.L, Lt_r)
    Mt = jnp.where(tir, state.M, Mt_r)
    Nt = jnp.where(tir, state.N, Nt_r)

    new_alive = state.alive & ~tir
    return JaxRayState(state.x, state.y, state.z, Lt, Mt, Nt,
                       state.opd, new_alive)


# ----------------------------------------------------------------------
# Aperture clipping
# ----------------------------------------------------------------------

def _apply_aperture_jax(state, semi_diameter):
    """Mark rays outside the clear aperture as dead.

    Uses the current (x, y) -- which is the intersection point on the
    surface -- against the surface's ``semi_diameter``.  Rays do not
    have their geometric state modified; only the alive mask is updated.
    """
    if not np.isfinite(semi_diameter):
        return state
    h_sq = state.x * state.x + state.y * state.y
    inside = h_sq <= semi_diameter * semi_diameter
    new_alive = state.alive & inside
    return state._replace(alive=new_alive)


# ----------------------------------------------------------------------
# Diffraction order kick (linear grating / DOE)
# ----------------------------------------------------------------------

def _apply_doe_kick_jax(state, order_x, order_y, period_x, period_y,
                         wavelength):
    """Apply a thin-grating diffraction-order kick at the current surface.

    Direction cosines are shifted by ``m * wavelength / period`` along
    each axis, and the corresponding linear OPL is added at the
    intersection point ``(x, y)``.  Use ``np.inf`` (or ``jnp.inf``) for
    ``period_y`` to disable the y-axis grating (1-D grating along x).

    Rays whose post-kick transverse direction cosines exceed unity
    (evanescent orders) are marked dead.

    Gradient support
    ----------------
    ``period_x`` / ``period_y`` may be either Python scalars (cheap
    path, no allocation) or JAX scalars / 0-D arrays (so users can
    ``jax.grad`` w.r.t. grating period).  Pre-v4.12 used
    ``float(period_*)`` which stripped the JAX trace -> silent zero
    gradient under ``jax.grad``; ``np.isfinite`` on a traced value
    further raised ``TracerArrayConversionError``.  v4.12 keeps the
    trace alive via ``jnp.where`` whenever the period argument is
    JAX-traced.
    """
    import jax.numpy as jnp

    def _is_traced(x):
        """True if ``x`` is a JAX tracer or JAX array (gradient flow
        required); False for Python int / float / NumPy 0-D scalar."""
        if isinstance(x, (int, float)):
            return False
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return False
        # JAX arrays / tracers / anything else with shape: keep the
        # trace alive (jnp.where instead of Python-level branch).
        return hasattr(x, 'shape')

    def _kick(order, period):
        """Compute ``order * wavelength / period`` keeping the JAX
        trace alive if ``period`` is traced.  Returns 0.0 when
        ``period`` is non-finite or zero (effectively no grating along
        that axis)."""
        if _is_traced(period):
            # JAX path: ``np.isfinite`` would raise on a tracer; use
            # ``jnp.where`` to keep the trace alive.  Guard the divide
            # so a non-finite or zero period yields the no-grating
            # branch (dL=0) without poisoning gradients with NaN.
            period_j = jnp.asarray(period)
            valid = jnp.isfinite(period_j) & (period_j != 0)
            safe = jnp.where(valid, period_j, 1.0)
            kick = float(order) * wavelength / safe
            return jnp.where(valid, kick, 0.0)
        # Concrete-period path: cheap, no JAX op.
        p = float(period)
        if np.isfinite(p) and p != 0:
            return float(order) * wavelength / p
        return 0.0

    dL = _kick(order_x, period_x)
    dM = _kick(order_y, period_y)

    L_new = state.L + dL
    M_new = state.M + dM
    sumsq = L_new * L_new + M_new * M_new
    # v5.17.1 (audit P3-58): evanescence is STRICTLY sumsq > 1.0,
    # matching the NumPy trace loop (trace.py ``_evan = _sumsq > 1.0``).
    # A grazing order with L^2 + M^2 exactly 1.0 propagates (N = 0)
    # on both backends instead of dying only on the JAX path.
    propagating = sumsq <= 1.0
    cos2 = jnp.maximum(1.0 - sumsq, 0.0)
    N_mag = jnp.sqrt(cos2)
    # Preserve the sign of the longitudinal cosine.
    N_new = jnp.where(state.N < 0, -N_mag, N_mag)

    # Linear OPL contribution from the grating phase gradient.
    new_opd = state.opd + dL * state.x + dM * state.y

    new_alive = state.alive & propagating
    return JaxRayState(state.x, state.y, state.z,
                       L_new, M_new, N_new,
                       new_opd, new_alive)


# ----------------------------------------------------------------------
# Free-space transfer between vertex planes
# ----------------------------------------------------------------------

def _transfer_jax(state, thickness, n_medium):
    """Free-space transfer through ``thickness`` in medium of index
    ``n_medium``.  All rays advance; OPL accumulates.

    Implementation note (RT-6, AUDIT_RAYTRACE_CORE_2026_07_08):
    ----------------------------------------------------------
    This advances every ray by the parameter ``t = thickness`` along its
    unit direction and shifts the frame origin by ``thickness``:

        new_x = x + L*thickness
        new_z = z + N*thickness - thickness   (= 0 only when N == 1)

    The NumPy ``_transfer`` instead solves ``t = (thickness - z) / N`` so
    the ray lands exactly on the next vertex plane (``new_z == 0``).  These
    look different, but they are numerically EQUIVALENT for the trace as a
    whole -- NOT a "paraxial approximation".  The frame-shifted point here
    is still a point on the SAME ray line (only its parameter along the
    line differs), and every downstream step is invariant to which point on
    the line the state carries:

    * the next surface intersect solves a line/surface problem (the sphere
      discriminant depends on the line, not the seed point);
    * refraction / aperture / DOE act at the intersection point;
    * the OPL telescopes exactly -- ``n*(thickness + t_int) == n*t_total``
      with signed legs (a paraxial overshoot yields a negative ``t_int``
      that subtracts the excess).

    Verified: a direct ``trace_jax`` vs NumPy ``trace`` parity check at NA
    up to 0.64 (min|N| ~ 0.77) agrees to ~2.4e-9 m, and that residual is
    INVARIANT to the gap length across a 100x sweep -- so it is the
    surface-solver tolerance, not a thickness-scaling transfer error.  The
    old v4.16.1-4.16.3 high-NA RuntimeWarning (which claimed a
    ``thickness*NA^2/2`` per-surface error and told NA>0.31 users to
    distrust the result) was therefore spurious and has been removed.

    The form here is also the one that keeps ``jax.grad`` clean through
    ``fit_canonical_polynomials_jax``'s downstream lstsq (the math-correct
    division form NaN-poisoned the gradient graph); since it is exact, no
    accuracy is traded for that differentiability.
    """
    # S3-12 (AUDIT_V5_24_2 robustness): freeze DEAD rays.  The NumPy
    # ``_transfer`` applies ``t = np.where(alive & ..., (thickness - z)/N, 0.0)``
    # -- a ray already marked ``alive == False`` keeps its (x, y, opd) exactly.
    # This path used to advance EVERY ray's position and OPL unconditionally,
    # which is harmless for in-tree consumers (they all mask by ``alive``) but
    # leaves an unmasked ``jax_state_to_raybundle`` reader seeing
    # backend-dependent drift on the dead rows.  Scale the transfer leg by the
    # alive mask so a dead ray matches the NumPy backend (position + OPL frozen).
    leg = thickness * state.alive.astype(state.x.dtype)
    new_x = state.x + state.L * leg
    new_y = state.y + state.M * leg
    new_z = state.z + state.N * leg - leg
    new_opd = state.opd + n_medium * leg
    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


# ----------------------------------------------------------------------
# Top-level trace
# ----------------------------------------------------------------------

# v4.12.1 perf cache for ``trace_jax``.
#
# The Python-level work in :func:`trace_jax` (parsing the prescription dict,
# resolving glass indices, building per-surface metadata) is repeated on
# every call.  Even though JAX's own dispatch cache short-circuits XLA
# compilation, the surrounding NumPy / Python work still costs ~2.5 ms
# per call (vs ~20 us for a cached jit'd kernel).  The cache below speeds
# up tight forward loops by ~100x in eager mode.
#
# History (failed v4.12.0 attempt):
#   v4.12.0 added a flat-tuple jit cache that stored ``jax.jit(_kernel)``
#   keyed on the static prescription signature and returned
#   ``kernel(initial_state)`` from every ``trace_jax`` call.  Forward calls
#   sped up 7470x but ``jax.grad(fit_canonical_polynomials_jax)`` returned
#   NaN.  Root cause (rediscovered while implementing v4.12.1): JAX's
#   backward pass through ``jax.jit`` + ``jnp.linalg.lstsq`` produces a
#   ``NaN`` in ``dot_general`` when the lstsq matrix is near rank-deficient
#   (the canonical-poly fit uses a 4-D Chebyshev basis where some
#   high-order cross-terms approach zero in the column norms).  The
#   pytree-keyed cache that this docstring originally promised does NOT
#   resolve the underlying JAX bug -- the NaN is in lstsq backward, not in
#   the prescription gradient flow.  So v4.12.1 takes a layered approach:
#
#   * Define :class:`JaxPrescription` as a pytree-registered wrapper (per
#     the v4.12.1 design): numeric per-surface values live as LEAVES so
#     ``jax.grad`` can flow through them when a user passes JAX arrays;
#     categorical / structural data lives in hashable AUX so we can key the
#     cache off it.
#   * Cache the compiled kernel on the AUX signature.
#   * **Skip the jit-cache layer when ``initial_state`` carries any JAX
#     tracer.**  Under ``jax.grad`` / ``jax.jit`` / ``jax.vmap`` the trace
#     happens inside the calling transform's own trace context, so an
#     extra ``jax.jit`` wrap is unnecessary AND triggers the lstsq-NaN bug
#     described above.  Eager Python calls still hit the cache.
#
# The cache key is the JaxPrescription's aux + a few extra immutable
# scalars (wavelength, diffraction spec).  Numeric leaves are passed
# through as positional JAX-array arguments to the jit'd kernel, so
# gradients flow through them if the caller supplies tracer leaves.


class JaxPrescription:
    """Pytree-registered prescription wrapper for :func:`trace_jax`.

    Splits a prescription dict into:

    * **leaves** -- numeric per-surface values that
      :func:`jax.tree_util.tree_flatten` exposes as the pytree leaves:

        - ``radii`` -- (n_surf,) JAX array of base radii in metres
          (``jnp.inf`` for flat).
        - ``conics`` -- (n_surf,) JAX array of conic constants.
        - ``thicks`` -- (n_surf - 1,) JAX array of inter-surface gaps.
        - ``asph_coeffs`` -- tuple of (n_coeffs_i,) JAX arrays, one per
          surface, aligned with the static ``asph_powers`` aux entry.

      Because these are pytree leaves, ``jax.grad`` flows through them
      naturally.  Eager / static use cases (the common case) supply them
      as concrete JAX arrays produced from the prescription dict's
      Python floats; differentiable use cases can substitute tracers.

    * **aux** -- hashable static structural data used as the cache key:

        - ``asph_powers`` -- tuple of int tuples, one per surface
          (the even-aspheric powers; aspheric COEFFICIENTS are leaves
          so they can be differentiated, but POWERS are integers and
          stay static).
        - ``semi_diameters`` -- tuple of floats; ``inf`` means no clip.
        - ``n_pre`` / ``n_post`` -- tuples of floats; pre-resolved
          per-surface glass indices at the trace wavelength.  Glass
          NAMES would be hashable categorical aux too, but resolving
          them once at build time is cheaper.
        - ``static_radii`` / ``static_conics`` / ``static_thicks`` --
          tuples of Python floats mirroring the leaf arrays.  These
          drive the Python-time static branches in :func:`_intersect_jax`
          (flat-vs-spherical-vs-aspheric) when the leaves are concrete.
          When the leaves are tracers (differentiable inputs), the
          kernel falls back to the always-Newton ``_intersect_jax_param``
          path which uses the JAX-array leaves directly.

    Construction is normally indirect via :func:`trace_jax` (which builds
    a ``JaxPrescription`` from a plain prescription dict).  Power users
    who want gradient flow through prescription parameters can construct
    one explicitly, passing tracer leaves; or via :func:`trace_jax_with_
    params` which has been the canonical entry point for that since
    v3.5.6.
    """

    __slots__ = ('radii', 'conics', 'thicks', 'asph_coeffs', 'aux')

    def __init__(self, radii, conics, thicks, asph_coeffs, aux):
        self.radii = radii
        self.conics = conics
        self.thicks = thicks
        self.asph_coeffs = asph_coeffs
        self.aux = aux

    def tree_flatten(self):
        children = (self.radii, self.conics, self.thicks, self.asph_coeffs)
        return children, self.aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        radii, conics, thicks, asph_coeffs = children
        return cls(radii, conics, thicks, asph_coeffs, aux)


# Register as a pytree exactly once on first import.  Guarded against
# re-registration so a hot reload during development doesn't error.
def _register_jaxprescription_pytree():
    if not JAX_AVAILABLE:
        return
    import jax
    try:
        jax.tree_util.register_pytree_node(
            JaxPrescription,
            JaxPrescription.tree_flatten,
            JaxPrescription.tree_unflatten,
        )
    except ValueError:
        # Already registered (this module was reloaded).  Fine.
        pass


_JAXPRESCRIPTION_REGISTERED = False


def _ensure_jaxprescription_registered():
    """Register the JaxPrescription pytree on FIRST jax use rather than at module
    import, so ``import lumenairy`` does not pull in jax (~4 s of the cold start;
    audit P2-D).  Called at every JAX trace entry point; idempotent (the flag
    short-circuits, and the underlying registration also guards re-registration).
    """
    global _JAXPRESCRIPTION_REGISTERED
    if not _JAXPRESCRIPTION_REGISTERED:
        _register_jaxprescription_pytree()
        _JAXPRESCRIPTION_REGISTERED = True


def _reject_unsupported_jax_surfaces(surfaces_raw, fn_name):
    """Fail loud on surface kinds the JAX trace cannot represent.

    4.10: refuse to silently pass mirrors / coord-breaks / biconic /
    freeform through as flat refractives.  These surface kinds are
    not yet implemented in the JAX trace; pre-4.10 the trace would
    happily pretend they were spherical refractors and return wrong
    answers.  Until proper implementations land, fail loud at build.

    4.13.2 (P1-NEW-D): also reject Welford-style mirrors signalled
    via ``glass_after='MIRROR'`` (case-insensitive).  The v4.13.1
    P1-A apply_real_lens fix added both guards together; the JAX
    prescription builder only got the ``is_mirror=True`` half, so
    hand-built mirrors with ``glass_after='MIRROR'`` slipped through
    and were silently traced as refractive air->air.

    v5.17.x (audit P2-34): hoisted to a module-level helper so
    :func:`trace_jax_with_params` (which bypasses
    :func:`_build_jax_prescription` for perf) applies the SAME guard --
    previously it silently traced mirrors / coord-breaks / biconics /
    freeforms as flat refractives, reproducing the pre-4.10 wrong-answer
    mode for differentiable prescriptions.

    Trace-safety: this guard inspects only STATIC Python fields of the
    prescription dict (booleans, glass strings, presence of biconic /
    freeform keys) -- never the differentiable ``radii`` / ``conics`` /
    ``aspheric_coeffs`` / ``thicknesses`` leaves -- so no JAX tracer is
    ever materialised or branched on and the guard is safe under
    ``jax.jit`` / ``jax.grad`` (cf. the tracer skip in
    ``lumenairy.elements.rcwa._core._reject_jax_offplane``, which must
    skip tracers because it inspects array VALUES; here there are no
    array values to inspect).
    """
    for i, s in enumerate(surfaces_raw):
        unsupported = []
        gl_after = s.get('glass_after')
        is_mirror_field = bool(s.get('is_mirror', False)) or (
            isinstance(gl_after, str)
            and gl_after.upper() == 'MIRROR'
        )
        if is_mirror_field:
            unsupported.append('is_mirror')
        if s.get('is_coordbrk'):
            unsupported.append('is_coordbrk')
        if s.get('radius_y') is not None:
            unsupported.append('radius_y (biconic)')
        if s.get('conic_y') is not None:
            unsupported.append('conic_y (biconic)')
        if s.get('aspheric_coeffs_y'):
            unsupported.append('aspheric_coeffs_y (biconic)')
        if s.get('freeform'):
            unsupported.append('freeform')
        if unsupported:
            raise NotImplementedError(
                f"{fn_name} does not yet support surface {i} with "
                f"{', '.join(unsupported)}.  Use the NumPy ``trace`` "
                f"backend, or open an issue if you need JAX-traceable "
                f"reflective / coord-broken / biconic / freeform support."
            )


def _resolve_semi_diameters(prescription):
    """Resolve the effective per-surface clear semi-aperture [m].

    Single source of the JAX backends' aperture semantics, mirroring
    the NumPy backend (``trace.surfaces_from_prescription``,
    trace.py:434-457) key for key:

    1. Default: ``aperture_diameter / 2`` (``inf`` when absent).
    2. A finite, positive per-surface ``'semi_diameter'`` REPLACES the
       default; ``None`` / non-finite / <= 0 falls back (audit P2-35).
    3. A ``prescription['elements']`` entry with
       ``element_type == 'surface'`` (matched by index within the
       refracting-surface entries, Zemax-loader layout) TIGHTENS the
       result via ``min()`` when its ``'semi_diameter'`` is finite and
       positive.

    v5.17.x (audit P2-35 residual): pre-fix the JAX builders read only
    the per-surface ``'semi_diameter'`` key and never consulted
    ``'elements'``, so a Zemax-loaded prescription (whose apertures
    live in ``'elements'``) vignetted under the NumPy trace but not
    under trace_jax / trace_jax_with_params.  Returns a list of Python
    floats (static -- never a JAX tracer), so it is jit/grad safe.
    """
    surfaces_raw = prescription.get('surfaces', [])
    aperture_d = prescription.get('aperture_diameter')
    default_semi = (aperture_d / 2.0
                    if aperture_d is not None else float('inf'))
    elements = prescription.get('elements', None)
    refr_elems = ([e for e in elements
                   if e.get('element_type') == 'surface']
                  if elements is not None else [])
    semi_ds = []
    for i, s in enumerate(surfaces_raw):
        sd = default_semi
        ps_sd = s.get('semi_diameter')
        if ps_sd is not None and np.isfinite(ps_sd) and ps_sd > 0:
            sd = float(ps_sd)
        if i < len(refr_elems):
            elem_sd = refr_elems[i].get('semi_diameter', np.inf)
            if elem_sd > 0 and np.isfinite(elem_sd):
                sd = min(sd, float(elem_sd))
        semi_ds.append(float(sd))
    return semi_ds


def _build_jax_prescription(prescription, wavelength,
                              surface_diffraction=None):
    """Build a :class:`JaxPrescription` from a plain prescription dict.

    Performs the surface-kind validation, glass-index lookups, and
    semi-diameter resolution that the legacy ``trace_jax`` used to do
    inline on every call.  The output is suitable both for direct kernel
    use AND for cache-key lookup (the ``aux`` field is hashable).
    """
    _ensure_jaxprescription_registered()    # lazy pytree reg (audit P2-D)
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    import jax.numpy as jnp

    surfaces_raw = prescription.get('surfaces', [])
    thicknesses = list(prescription.get('thicknesses', []))
    if len(thicknesses) < len(surfaces_raw) - 1:
        thicknesses = thicknesses + [0.0] * (
            len(surfaces_raw) - 1 - len(thicknesses))

    # Unsupported-surface fail-loud guard (4.10 / 4.13.2; hoisted to
    # _reject_unsupported_jax_surfaces for audit P2-34 -- see its
    # docstring for history).
    _reject_unsupported_jax_surfaces(surfaces_raw, 'trace_jax')

    n_surf = len(surfaces_raw)

    # Numeric per-surface values as Python floats first (for the static
    # branches in _intersect_jax) then mirrored as JAX arrays for the
    # pytree leaves.
    radii_py = tuple(
        float(s.get('radius', float('inf'))) for s in surfaces_raw)
    conics_py = tuple(
        float(s.get('conic', 0.0)) for s in surfaces_raw)
    thicks_py = tuple(
        float(thicknesses[i]) if i < len(thicknesses) else 0.0
        for i in range(max(0, n_surf - 1))
    )

    asph_powers = tuple(
        tuple(sorted(int(p) for p in (s.get('aspheric_coeffs') or {}).keys()))
        for s in surfaces_raw
    )
    asph_pairs = tuple(
        tuple(
            (int(p), float((s.get('aspheric_coeffs') or {})[p]))
            for p in pwr
        )
        for s, pwr in zip(surfaces_raw, asph_powers)
    )

    # Audit P2-35 (residual): shared resolver -- honours the per-surface
    # 'semi_diameter' key AND the 'elements' list exactly like the NumPy
    # backend (see _resolve_semi_diameters).
    semi_ds = tuple(_resolve_semi_diameters(prescription))
    n_pre = tuple(
        float(get_glass_index(s.get('glass_before', 'air'), wavelength))
        for s in surfaces_raw
    )
    n_post = tuple(
        float(get_glass_index(s.get('glass_after', 'air'), wavelength))
        for s in surfaces_raw
    )

    diff = dict(surface_diffraction) if surface_diffraction else {}
    diff_aux = tuple(sorted(
        (int(k), tuple(float(x) for x in v)) for k, v in diff.items()
    ))

    # Mirror the static Python tuples as JAX-array leaves so users who
    # want differentiable prescriptions can substitute tracer leaves.
    radii_arr = jnp.asarray(radii_py)
    conics_arr = jnp.asarray(conics_py)
    thicks_arr = jnp.asarray(thicks_py)
    asph_coeffs_leaves = tuple(
        jnp.asarray([c for _, c in pairs]) for pairs in asph_pairs
    )

    aux = (
        n_surf,
        asph_powers,
        semi_ds,
        n_pre,
        n_post,
        radii_py,
        conics_py,
        thicks_py,
        asph_pairs,
        diff_aux,
    )
    return JaxPrescription(
        radii_arr, conics_arr, thicks_arr, asph_coeffs_leaves, aux)


def _leaves_are_concrete(jp):
    """True if every leaf of ``jp`` is a concrete (non-traced) array.

    When False, the kernel must avoid the Python-float static-branch
    path (whose ``np.isinf(R)`` / ``conic != 0`` checks would raise on
    a tracer) and route through the always-Newton ``_intersect_jax_param``
    branch instead.
    """
    from jax.core import Tracer
    leaves = [jp.radii, jp.conics, jp.thicks]
    for c in jp.asph_coeffs:
        leaves.append(c)
    return not any(isinstance(l, Tracer) for l in leaves)


def _running_under_trace(initial_state, jp):
    """True if ANY tracer is reachable from the inputs.

    Used to decide whether to apply ``jax.jit`` -- under ``jax.grad`` /
    ``jax.jit`` / ``jax.vmap`` the calling transform already owns the
    trace context, and adding our own jit layer triggers JAX's
    ``dot_general`` NaN in the backward pass through ``jnp.linalg.lstsq``
    (the v4.12.0 failure mode).
    """
    import jax
    from jax.core import Tracer
    for leaf in jax.tree_util.tree_leaves(initial_state):
        if isinstance(leaf, Tracer):
            return True
    if not _leaves_are_concrete(jp):
        return True
    return False


def _trace_body_static(state, jp, wavelength, surface_diffraction):
    """Inline kernel using the static-branch ``_intersect_jax`` /
    ``_refract_jax`` (Python-time radius / conic / aspheric flags).

    Used when ALL leaves are concrete (no tracers in jp) -- the common
    case for cache hits.  All numeric values (radii / conics / asph
    coeffs / thicknesses) are pulled from ``jp.aux`` as Python floats
    so the static branches in :func:`_intersect_jax` / :func:`_refract_jax`
    work even when this body is called inside :func:`jax.jit` (where the
    LEAVES of ``jp`` would appear as tracers).  The leaves themselves are
    only used by the traced-leaf path (:func:`_trace_body_traced`).
    """
    (n_surf, asph_powers, semi_ds, n_pre, n_post,
     radii_py, conics_py, thicks_py, asph_pairs, diff_aux) = jp.aux
    diff_lookup = {idx: kick for idx, kick in diff_aux}

    for i in range(n_surf):
        R = radii_py[i]
        kc = conics_py[i]
        asph = asph_pairs[i]
        sd = semi_ds[i]
        n1 = n_pre[i]
        n2 = n_post[i]

        state = _intersect_jax(state, R, kc, asph, n_medium=n1)
        state = _refract_jax(state, R, kc, asph, n1, n2)
        state = _apply_aperture_jax(state, sd)

        if i in diff_lookup:
            ox, oy, px, py = diff_lookup[i]
            state = _apply_doe_kick_jax(
                state, ox, oy, px, py, float(wavelength))

        if i < n_surf - 1:
            t = thicks_py[i] if i < len(thicks_py) else 0.0
            state = _transfer_jax(state, t, n_medium=n2)
    return state


def _trace_body_traced(state, jp, wavelength, surface_diffraction):
    """Inline kernel using the always-Newton ``_intersect_jax_param`` /
    ``_refract_jax_param`` path (JAX-array radius / conic / aspheric
    coeffs).

    Used when ANY leaf is a tracer -- e.g., when the caller wants
    ``jax.grad`` to flow through a prescription parameter.  Cost is
    roughly 10% higher than the static path (one extra Newton iter on
    pure-spherical surfaces), and we lose the closed-form flat-surface
    fast path.  Gradient correctness is the reason we use this branch.
    """
    (n_surf, asph_powers, semi_ds, n_pre, n_post,
     radii_py, conics_py, thicks_py, asph_pairs, diff_aux) = jp.aux
    diff_lookup = {idx: kick for idx, kick in diff_aux}

    for i in range(n_surf):
        R = jp.radii[i]
        kc = jp.conics[i]
        coeffs = jp.asph_coeffs[i]
        powers = asph_powers[i]
        sd = semi_ds[i]
        n1 = n_pre[i]
        n2 = n_post[i]

        state = _intersect_jax_param(state, R, kc, powers, coeffs,
                                       n_medium=n1)
        state = _refract_jax_param(state, R, kc, powers, coeffs, n1, n2)
        state = _apply_aperture_jax(state, sd)

        if i in diff_lookup:
            ox, oy, px, py = diff_lookup[i]
            state = _apply_doe_kick_jax(
                state, ox, oy, px, py, float(wavelength))

        if i < n_surf - 1:
            t = jp.thicks[i]
            state = _transfer_jax(state, t, n_medium=n2)
    return state


# Cache of jit'd kernels keyed on the JaxPrescription aux + the
# wavelength scalar.  The kernel signature is ``(state, jp_leaves...)`` so
# leaf gradients flow through; under tracing we bypass this layer
# entirely (see :func:`_running_under_trace`).
#
# v4.12.2: converted to an LRU-bounded ``OrderedDict`` so long-running
# optimizers (e.g. design sweeps over hundreds of prescriptions) do not
# leak compiled XLA executables.  Accessed keys are moved to the end;
# when ``len > _TRACE_JAX_CACHE_MAXSIZE`` the oldest entry is evicted.
_TRACE_JAX_CACHE: 'OrderedDict[Any, Any]' = OrderedDict()
_TRACE_JAX_CACHE_MAXSIZE = 32  # tune; long-running optimizers may exceed
# v4.14.2 (P1-NEW-2 / Agent C): thread-safety lock for
# ``_TRACE_JAX_CACHE``.  Without this two threads racing through
# :func:`trace_jax` could see a torn OrderedDict (``get`` ->
# ``__setitem__`` -> ``popitem`` is a read-modify-write sequence).
# Follows the ``_ASM_CACHE_LOCK`` precedent in
# :mod:`propagators.propagation`.  Lock-scope discipline: the
# jit-compile step (``_make_jit_kernel``) is expensive (XLA compile)
# and runs OUTSIDE the lock so a concurrent cache hit on a different
# key isn't blocked.
_TRACE_JAX_CACHE_LOCK = threading.Lock()


def clear_trace_jax_cache() -> None:
    """Drop every cached jit'd ``trace_jax`` kernel (v4.12.2).

    Forces the next :func:`trace_jax` call to rebuild and re-cache its
    jit-compiled kernel from scratch.  Useful in unit tests that pin
    cache mechanics and in long-running notebooks / optimizers where
    the user wants to release the underlying XLA executables.
    """
    with _TRACE_JAX_CACHE_LOCK:
        _TRACE_JAX_CACHE.clear()


# v4.16.0 (ROADMAP #15): register the raytrace-JAX clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
# Late-binding closure preserves ``mock.patch.object`` test semantic.
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'trace_jax',
        lambda: getattr(_this_mod, 'clear_trace_jax_cache')(),
    )
except ImportError:
    pass


def _make_jit_kernel(jp_aux, wavelength_float, surface_diffraction):
    """Build a jit-compiled kernel keyed on ``jp_aux`` (static aux).

    The kernel takes ``(state, jp_concrete)`` where ``jp_concrete`` is a
    :class:`JaxPrescription` whose leaves are concrete JAX arrays.
    """
    import jax

    def _kernel(state, jp):
        # All leaves are concrete here (we only cache for eager calls).
        # The kernel itself uses the static-branch path; if a future
        # caller threads tracer leaves in we'd hit ``_running_under_trace``
        # and bypass this cached kernel entirely.
        return _trace_body_static(
            state, jp, wavelength_float, surface_diffraction)

    return jax.jit(_kernel)


def trace_jax(
    initial_state: 'JaxRayState',
    prescription: Any,
    wavelength: float,
    surface_diffraction: Optional[Dict[int, Any]] = None,
) -> 'JaxRayState':
    """JAX-traceable sequential ray trace.

    Walks the prescription's surfaces and interleaves
    intersect / refract / aperture-clip / DOE-kick / transfer steps.
    Compatible with ``jax.jit`` and ``jax.grad``.

    v4.12.1 perf cache: the prescription dict is converted to a
    pytree-registered :class:`JaxPrescription` and a jit-compiled inner
    kernel is cached on the prescription's hashable aux signature.
    Repeated EAGER calls with the same prescription reuse one compiled
    XLA graph (roughly 100x speedup vs the v4.11.2 baseline).  Under any
    of ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` the cache-layer is
    bypassed -- the calling transform already owns the trace context and
    nesting our jit under it triggers a known JAX bug where
    ``jnp.linalg.lstsq``'s backward pass produces NaN in ``dot_general``
    (the v4.12.0 failure mode).  Eager users get the full cache benefit;
    grad / jit / vmap users get correctness with the same per-call cost
    as v4.11.2.

    .. warning:: **Sweep / finite-difference re-JIT thrash (S3-15).**
       The eager cache key embeds EVERY numeric prescription value (radii,
       conics, aspheric coeffs, thicknesses -- all of ``jp.aux``).  A loop
       that perturbs any prescription number -- a parameter sweep, a
       finite-difference gradient, a tolerance study -- therefore produces
       a NEW cache key on every iteration, forcing a fresh XLA compile and
       evicting the 32-entry LRU (so even the unperturbed baseline kernel
       is lost).  The compile dominates the trace time and defeats the
       cache entirely.  For those loops use :func:`trace_jax_with_params`,
       which takes the swept radii / conics / aspheric coeffs / thicknesses
       as JAX-array leaves through ONE compiled kernel (and is
       ``jax.grad``-differentiable in them) -- so the geometry varies
       without re-JIT.  ``trace_jax``'s cache is meant for repeated calls
       at a FIXED prescription (e.g. re-tracing many ray bundles).

    Parameters
    ----------
    initial_state : JaxRayState
        Initial ray state (positions / directions / OPL / alive).
    prescription : dict or JaxPrescription
        Either a plain lumenairy prescription dict (read for
        ``radius``, ``conic``, ``aspheric_coeffs``, ``glass_before``,
        ``glass_after``, and ``semi_diameter`` per-surface; plus
        top-level ``aperture_diameter`` / ``thicknesses``) or a
        pre-built :class:`JaxPrescription`.  Dicts are converted on
        the fly; ``JaxPrescription``s are used as-is so their (possibly
        differentiable) leaves flow through unchanged.
    wavelength : float
        Vacuum wavelength.
    surface_diffraction : dict or None, optional
        Mapping ``{surface_index: (order_x, order_y, period_x,
        period_y)}`` describing grating / DOE kicks at specific
        surfaces.  Mirrors the signature of
        :func:`lumenairy.raytrace.trace`.

    Returns
    -------
    JaxRayState
        Final ray state at the image plane.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    _ensure_jaxprescription_registered()    # lazy pytree reg (audit P2-D)

    # Allow callers to pre-build a JaxPrescription.  This lets advanced
    # users substitute tracer leaves for radii / conics / asph coeffs /
    # thicknesses without re-walking the dict every call.
    if isinstance(prescription, JaxPrescription):
        # RT-8 (AUDIT_RAYTRACE_CORE): the DOE spec is baked into ``jp.aux``'s
        # ``diff_aux`` at build time; a ``surface_diffraction`` kwarg passed
        # alongside a prebuilt ``JaxPrescription`` was silently dropped (the
        # trace bodies read the DOE spec only from ``jp.aux``).  Refuse rather
        # than trace with no grating kick -- rebuild the prescription with the
        # DOE folded in via ``_build_jax_prescription(dict, wl, diff)``.
        if surface_diffraction is not None:
            raise ValueError(
                "trace_jax: surface_diffraction was passed together with a "
                "pre-built JaxPrescription, but the DOE spec is baked into the "
                "JaxPrescription at build time -- the kwarg would be silently "
                "ignored.  Either pass the plain prescription dict (so the "
                "kwarg is honoured) or rebuild the JaxPrescription with the "
                "diffraction spec folded in.")
        jp = prescription
    else:
        jp = _build_jax_prescription(
            prescription, wavelength, surface_diffraction)

    # RT-6: the v4.16.2/4.16.3 high-NA probe hoisted here has been removed
    # along with the spurious paraxial-transfer warning (see ``_transfer_jax``
    # -- the transfer is exact, so there was nothing to warn about).

    # Bypass the jit-cache layer if anything looks like a tracer.  The
    # cached jit'd kernel triggers a JAX bug in lstsq backward (see the
    # module-level cache docstring), and the calling transform already
    # owns the trace context so an extra jit wrap is unnecessary.
    if _running_under_trace(initial_state, jp):
        if not _leaves_are_concrete(jp):
            return _trace_body_traced(
                initial_state, jp, wavelength, surface_diffraction)
        return _trace_body_static(
            initial_state, jp, wavelength, surface_diffraction)

    # Eager path: look up or build a jit-compiled kernel keyed on the
    # static signature.  Wavelength + diffraction_spec roll into the
    # cache key so we don't accidentally share a kernel across distinct
    # ``wavelength`` calls (glass indices in aux already depend on it,
    # but DOE-kick wavelength would still be different).
    diff_aux = jp.aux[-1]
    # S3-15: jp.aux embeds every numeric prescription value, so a
    # sweep/FD loop over perturbed geometry re-keys (and re-JITs) here on
    # every iteration -- use trace_jax_with_params for that (see the
    # docstring warning).  This key is right for repeated FIXED-geometry
    # traces.
    cache_key = (jp.aux, float(wavelength), diff_aux)
    with _TRACE_JAX_CACHE_LOCK:
        kernel = _TRACE_JAX_CACHE.get(cache_key)
        if kernel is not None:
            # LRU touch: move this hit to the end so it survives eviction.
            _TRACE_JAX_CACHE.move_to_end(cache_key)
    if kernel is None:
        # Build OUTSIDE the lock -- jit tracing is the expensive step
        # (XLA compile) and holding the lock here would serialise
        # every concurrent trace_jax caller even when their cache
        # keys differ.  Two threads may double-build on a cold cache
        # for the same key -- benign waste; the second insert just
        # overwrites the first.
        kernel = _make_jit_kernel(
            jp.aux, float(wavelength), surface_diffraction)
        with _TRACE_JAX_CACHE_LOCK:
            _TRACE_JAX_CACHE[cache_key] = kernel
            while len(_TRACE_JAX_CACHE) > _TRACE_JAX_CACHE_MAXSIZE:
                _TRACE_JAX_CACHE.popitem(last=False)
    return kernel(initial_state, jp)


def jax_state_to_raybundle(state, wavelength=0.0):
    """Convert a :class:`JaxRayState` to a NumPy
    :class:`lumenairy.raytrace.RayBundle`.

    Useful when the JAX trace is the differentiable inner loop
    but downstream analysis (Strehl, Zernike, etc.) wants NumPy.
    The ``wavelength`` argument lets the caller restore that field
    (the JAX state does not carry it); ``error_code`` is auto-
    populated by :class:`RayBundle`'s post-init.
    """
    from .core import RayBundle
    return RayBundle(
        x=np.asarray(state.x),
        y=np.asarray(state.y),
        z=np.asarray(state.z),
        L=np.asarray(state.L),
        M=np.asarray(state.M),
        N=np.asarray(state.N),
        wavelength=float(wavelength),
        opd=np.asarray(state.opd),
        alive=np.asarray(state.alive),
    )


def raybundle_to_jax_state(rays):
    """Convert a NumPy :class:`RayBundle` to a :class:`JaxRayState`.

    Inverse of :func:`jax_state_to_raybundle`.  Drops ``wavelength``
    and ``error_code`` (which have no JAX-traceable analogue).  Useful
    when seeding ``trace_jax`` from a NumPy-built bundle.
    """
    return make_jax_ray_state(
        x=rays.x, y=rays.y, z=rays.z,
        L=rays.L, M=rays.M, N=rays.N,
        opd=rays.opd, alive=rays.alive,
    )


# ============================================================================
# JAX-array-aware variants: trace through prescriptions whose RADII /
# CONICS / ASPHERIC COEFFS are differentiable JAX arrays.
#
# The static versions above (_intersect_jax, _refract_jax, trace_jax) read
# every per-surface number as a Python float at trace-build time, so
# jax.grad cannot flow through them.  The variants below take the same
# numbers as JAX scalars / arrays and route through a single always-Newton
# intersection path with no Python branches on the radius value.  Cost on
# the non-aspheric path is ~10% (one extra Newton iter) and the spherical
# guess remains exact for R-finite surfaces.
#
# Aspheric POWERS remain compile-time-static (they're integers); only the
# COEFFICIENTS need to be differentiable.  The new functions accept
# coefficients as a JAX vector aligned with a static powers tuple.
#
# Audit perf #1 (3.5.6) -- closes the prescription-parameter
# differentiability gap.
# ============================================================================


def _sag_value_param(x, y, R, conic, asph_powers, asph_coeffs):
    """JAX-array-aware sag.  R, conic, and each aspheric coefficient
    may be a JAX scalar or array.  ``asph_powers`` is a static
    Python tuple of int powers.

    Handles ``R = inf`` cleanly via the small-norm branch in
    ``jnp.where``: when |R| is very large (or +inf), the conic
    contribution collapses to ~0.

    S3-10: delegates to the backend-agnostic
    :func:`raytrace._conic_core.conic_sag`.  The differentiable
    coefficient array is zipped with the static integer powers into the
    ``(power, coeff)`` pairs the shared core consumes, so ``jax.grad``
    still flows through the coefficients.  Byte-identical to the former
    inline form (shared-core test).
    """
    import jax.numpy as jnp
    asph_items = tuple(
        (int(power), asph_coeffs[i]) for i, power in enumerate(asph_powers))
    return conic_sag(x, y, R, conic, asph_items, xp=jnp)


def _sag_derivatives_param(x, y, R, conic, asph_powers, asph_coeffs):
    """JAX-array-aware sag derivatives dz/dx, dz/dy.

    S3-10: delegates to the backend-agnostic
    :func:`raytrace._conic_core.conic_sag_derivs` (byte-identical to the
    former inline form; the ``sign(R)`` C-RT-3 fix and the differentiable
    aspheric coefficients are preserved -- shared-core test).
    """
    import jax.numpy as jnp
    asph_items = tuple(
        (int(power), asph_coeffs[i]) for i, power in enumerate(asph_powers))
    return conic_sag_derivs(x, y, R, conic, asph_items, xp=jnp)


def _intersect_jax_param(state, R, conic, asph_powers, asph_coeffs,
                          n_medium):
    """Always-Newton intersect with JAX-array prescription params."""
    import jax
    import jax.numpy as jnp
    eps = 1e-30

    R_finite = jnp.where(jnp.isinf(R), 1e30, R)
    R_finite = jnp.where(jnp.abs(R_finite) < 1e-30, 1e-30, R_finite)
    is_flat = jnp.isinf(R) | (jnp.abs(R) > 1e15)

    # Spherical initial guess for the curved branch.
    dx = state.x
    dy = state.y
    dz = state.z - R_finite
    b_q = 2.0 * (state.L * dx + state.M * dy + state.N * dz)
    c_q = dx ** 2 + dy ** 2 + dz ** 2 - R_finite ** 2
    disc = b_q ** 2 - 4.0 * c_q
    # 4.11.1 (H-RT-7): double-where on sqrt(disc) so the disc=0
    # gradient singularity doesn't NaN-poison jax.grad on tangent rays.
    disc_pos = disc > 0
    disc_safe = jnp.where(disc_pos, disc, 1.0)
    sqrt_disc = jnp.where(disc_pos, jnp.sqrt(disc_safe), 0.0)
    # v5.17.1 (audit P3-58): acceptance is disc >= 0 (NumPy parity,
    # v5.4.6 audit P3-3 tangency semantics); the sqrt guard stays on
    # the strict disc > 0 for the H-RT-7 gradient mask.  At disc == 0
    # both roots equal -b/2, the tangent intersection.
    disc_ok = disc >= 0
    t1 = (-b_q - sqrt_disc) / 2.0
    t2 = (-b_q + sqrt_disc) / 2.0
    # v5.4.6 (audit P3-1): direction-aware near-root pick (min |t|), matching
    # the v5.4.1 NumPy fix and the static-branch JAX kernel (P1-1).  The old
    # ``R_finite > 0`` selector is direction-blind and lands a backward leg
    # (post-mirror N<0) on the far root, corrupting jax.grad of mirror/folded
    # prescriptions.  As a bonus this removes the dependence on the sign of a
    # (possibly traced) curvature, improving JAX traceability.
    t_sphere = jnp.where(jnp.abs(t1) <= jnp.abs(t2), t1, t2)
    t_sphere = jnp.where(disc_ok, t_sphere, 0.0)

    # Flat initial guess.
    N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
    t_flat = -state.z / N_safe

    t0 = jnp.where(is_flat, t_flat, t_sphere)
    # 4.11.1 (H-RT-5): rays that missed the sphere (disc < 0) AND aren't
    # on the flat branch are dead.  Flat rays parallel to the surface
    # (|N| -> 0) are also dead.  disc == 0 tangency is accepted
    # (v5.17.1, audit P3-58 -- NumPy parity).
    miss = (~is_flat & ~disc_ok) | (is_flat & (jnp.abs(state.N) <= eps))

    # Newton refinement.  For pure-spherical/flat (no asph, conic=0)
    # this is essentially a no-op (1 iter to verify convergence).
    # 4.11.1: double-where for the F / dF_dt step.  Single-where (the
    # old pattern below) still evaluates the division on the False
    # branch, so when dF_dt -> 0 the division produces NaN/Inf whose
    # gradient propagates through ``jnp.where`` and poisons jax.grad.
    # Mirrors the static `_intersect_jax` Newton body.
    def body(_, t):
        xi = state.x + state.L * t
        yi = state.y + state.M * t
        zi = state.z + state.N * t
        sag_i = _sag_value_param(xi, yi, R, conic, asph_powers,
                                   asph_coeffs)
        F = zi - sag_i
        dz_dx, dz_dy = _sag_derivatives_param(
            xi, yi, R, conic, asph_powers, asph_coeffs)
        dF_dt = state.N - dz_dx * state.L - dz_dy * state.M
        stuck = jnp.abs(dF_dt) <= eps
        dF_safe = jnp.where(stuck, 1.0, dF_dt)
        step = jnp.where(stuck, 0.0, F / dF_safe)
        return t - step

    t = jax.lax.fori_loop(0, _ASPHERIC_NEWTON_ITERS, body, t0)
    # v5.17.1 (audit P3-59): convergence check -- mirror the NumPy
    # residual kill (see _ASPHERIC_NEWTON_RESIDUAL_TOL and the static
    # `_intersect_jax` twin).  Residual only feeds a boolean, so this
    # extra sag evaluation is jit/grad trace-safe; the ~(<=) form
    # also kills NaN residuals.
    xi = state.x + state.L * t
    yi = state.y + state.M * t
    zi = state.z + state.N * t
    resid = zi - _sag_value_param(xi, yi, R, conic, asph_powers,
                                    asph_coeffs)
    miss = miss | ~(jnp.abs(resid) <= _newton_residual_tol(resid.dtype))
    # 4.11.1 (H-RT-5): also catch non-finite Newton output (NaN/Inf
    # from divergent surfaces) and propagate the kill into alive.
    miss = miss | (~jnp.isfinite(t))
    t = jnp.where(state.alive & ~miss, t, 0.0)
    new_alive = state.alive & ~miss

    new_x = state.x + state.L * t
    new_y = state.y + state.M * t
    new_z = state.z + state.N * t
    new_opd = state.opd + n_medium * t
    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, new_alive)


def _refract_jax_param(state, R, conic, asph_powers, asph_coeffs, n1, n2):
    """JAX-array-aware vector Snell's law with conic + aspheric surface
    normals."""
    import jax.numpy as jnp
    zx, zy = _sag_derivatives_param(state.x, state.y, R, conic,
                                      asph_powers, asph_coeffs)
    nx = -zx
    ny = -zy
    nz = jnp.ones_like(state.x)
    nrm = jnp.sqrt(nx * nx + ny * ny + nz * nz)
    nrm = jnp.maximum(nrm, 1e-30)
    nx = nx / nrm
    ny = ny / nrm
    nz = nz / nrm

    # S3-10: orient the normal + apply vector Snell via the backend-
    # agnostic shared core (see the _refract_jax twin above for the
    # tir_guard / eta_sq rationale).  Bit-identical forward + jax.grad.
    eta = n1 / n2
    Lt_r, Mt_r, Nt_r, nx, ny, nz, _cos_i, _disc_r, tir = refract_snell(
        state.L, state.M, state.N, nx, ny, nz, eta, eta ** 2,
        sqrt=lambda z: jnp.sqrt(jnp.maximum(z, 0.0)),
        where=jnp.where,
        tir_guard=lambda t, d: jnp.where(t, 1.0, d))

    Lt = jnp.where(tir, state.L, Lt_r)
    Mt = jnp.where(tir, state.M, Mt_r)
    Nt = jnp.where(tir, state.N, Nt_r)

    new_alive = state.alive & ~tir
    return JaxRayState(state.x, state.y, state.z, Lt, Mt, Nt,
                       state.opd, new_alive)


def trace_jax_with_params(initial_state, prescription, wavelength,
                          *,
                          radii=None,
                          conics=None,
                          aspheric_coeffs=None,
                          thicknesses=None,
                          surface_diffraction=None):
    """JAX-array-aware variant of :func:`trace_jax`.

    Differentiable in the per-surface ``radii`` / ``conics`` /
    ``aspheric_coeffs`` / ``thicknesses`` -- enables ``jax.grad`` of
    a downstream loss with respect to the prescription's geometric
    parameters.  Closes the audit-#1 gap that
    :func:`make_lg_aberration_merit_jax` partially addressed in 3.5.5.

    Parameters
    ----------
    initial_state : JaxRayState
    prescription : dict
        Static prescription.  Provides aspheric POWERS, glass strings,
        and aperture metadata.  The numeric values for radius / conic
        / asphere coeffs / thickness are OVERRIDDEN by the keyword
        arguments below when given (defaulting to the static values
        from the dict when None).
    wavelength : float
    radii : (n_surfaces,) JAX array, optional
        Per-surface radii in metres.  Use ``jnp.inf`` for flat.
        ``None`` -> use prescription's radii as static floats (no
        gradient flows through radii).
    conics : (n_surfaces,) JAX array, optional
        Per-surface conic constants.  ``None`` -> defaults to 0.0
        per surface.
    aspheric_coeffs : list of JAX arrays, optional
        ``aspheric_coeffs[i]`` is a 1-D JAX array of coefficients for
        surface ``i``, indexed by the static power tuple derived
        from the prescription's ``aspheric_coeffs[i]`` dict.  Pass
        an empty array for surfaces with no aspheric.
    thicknesses : (n_surfaces - 1,) JAX array, optional
        Inter-surface gaps in metres.  ``None`` -> static.
    surface_diffraction : same as :func:`trace_jax`

    Returns
    -------
    JaxRayState
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    _ensure_jaxprescription_registered()    # lazy pytree reg (audit P2-D)
    import jax.numpy as jnp

    surfaces_raw = prescription.get('surfaces', [])
    n_surf = len(surfaces_raw)

    # Audit P2-34: apply the same unsupported-surface fail-loud guard
    # as trace_jax's builder.  This entry point bypasses
    # _build_jax_prescription for perf, so pre-fix it silently traced
    # mirrors / coord-breaks / biconics / freeforms as flat
    # refractives.  Static-fields-only, hence jit/grad trace-safe (see
    # _reject_unsupported_jax_surfaces).
    _reject_unsupported_jax_surfaces(surfaces_raw, 'trace_jax_with_params')

    # Resolve per-surface params: prefer kwarg array, fall back to the
    # static prescription value.
    if radii is None:
        radii_list = [float(s.get('radius', float('inf')))
                      for s in surfaces_raw]
    else:
        radii_list = list(radii)
    if conics is None:
        conics_list = [float(s.get('conic', 0.0)) for s in surfaces_raw]
    else:
        conics_list = list(conics)

    # Aspheric powers stay static (they're integers).  Coefficients
    # are JAX arrays per surface.
    asph_powers_per_surface = []
    asph_coeffs_per_surface = []
    for i, s in enumerate(surfaces_raw):
        a = s.get('aspheric_coeffs') or {}
        powers = tuple(sorted(int(p) for p in a.keys()))
        asph_powers_per_surface.append(powers)
        if aspheric_coeffs is not None and i < len(aspheric_coeffs):
            asph_coeffs_per_surface.append(jnp.asarray(aspheric_coeffs[i]))
        else:
            asph_coeffs_per_surface.append(
                jnp.asarray([float(a[p]) for p in powers]))

    # Thicknesses
    thicknesses_static = list(prescription.get('thicknesses', []))
    if len(thicknesses_static) < n_surf - 1:
        thicknesses_static = thicknesses_static + [0.0] * (
            n_surf - 1 - len(thicknesses_static))
    if thicknesses is None:
        thicknesses_list = thicknesses_static
    else:
        thicknesses_list = list(thicknesses)

    # Apertures stay static.  Audit P2-35 (residual): same shared
    # resolver as _build_jax_prescription / the NumPy backend, so the
    # 'elements' list is honoured here too.
    semi_ds = _resolve_semi_diameters(prescription)
    n_pre = [
        float(get_glass_index(s.get('glass_before', 'air'), wavelength))
        for s in surfaces_raw
    ]
    n_post = [
        float(get_glass_index(s.get('glass_after', 'air'), wavelength))
        for s in surfaces_raw
    ]
    diff = dict(surface_diffraction) if surface_diffraction else {}

    state = initial_state
    for i in range(n_surf):
        R = radii_list[i]
        kc = conics_list[i]
        powers = asph_powers_per_surface[i]
        coeffs = asph_coeffs_per_surface[i]
        sd = semi_ds[i]
        n1 = n_pre[i]
        n2 = n_post[i]

        # Convert R, kc to JAX scalars if they aren't already (so the
        # always-Newton intersect produces traceable values).
        R_jax = R if hasattr(R, 'shape') else jnp.float64(R)
        kc_jax = kc if hasattr(kc, 'shape') else jnp.float64(kc)

        state = _intersect_jax_param(state, R_jax, kc_jax,
                                       powers, coeffs, n_medium=n1)
        state = _refract_jax_param(state, R_jax, kc_jax,
                                     powers, coeffs, n1, n2)
        state = _apply_aperture_jax(state, sd)
        if i in diff:
            ox, oy, px, py = diff[i]
            state = _apply_doe_kick_jax(
                state, ox, oy, px, py, float(wavelength))
        if i < n_surf - 1:
            t = thicknesses_list[i]
            state = _transfer_jax(state, t, n_medium=n2)

    return state


__all__ = [
    'JaxRayState',
    'JaxPrescription',
    'make_jax_ray_state',
    'trace_jax',
    'trace_jax_with_params',
    'jax_state_to_raybundle',
    'raybundle_to_jax_state',
]
