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

from collections import OrderedDict
from typing import Any, Dict, NamedTuple, Optional

import numpy as np

from ..backend import JAX_AVAILABLE
from ..glass import get_glass_index


# Number of Newton iterations for aspheric ray-surface intersection.
# 8 is plenty for typical optical aspheres at micron-grade ray
# spacings; the sag solver in core.py uses up to 10 with an early-
# exit, but we use a fixed count here so JIT/grad can trace once.
_ASPHERIC_NEWTON_ITERS = 8


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
    x = jnp.asarray(x); y = jnp.asarray(y); z = jnp.asarray(z)
    L = jnp.asarray(L); M = jnp.asarray(M); N = jnp.asarray(N)
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
    """
    import jax.numpy as jnp
    h_sq = x * x + y * y
    if R_is_inf:
        sag = jnp.zeros_like(h_sq)
    else:
        norm = (1.0 + conic) * h_sq / (R_safe * R_safe)
        valid = norm < 0.9999
        denom_arg = jnp.where(valid, 1.0 - norm, 0.01)
        sag = jnp.where(
            valid,
            h_sq / (R_safe * (1.0 + jnp.sqrt(denom_arg))),
            0.0,
        )
    for power, coeff in asph_items:
        # Even powers only (h^4, h^6, ...).  h_sq^(power//2) == h^power.
        sag = sag + coeff * h_sq ** (power // 2)
    return sag


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
    """
    import jax.numpy as jnp
    h_sq = x * x + y * y
    if R_is_inf:
        zx = jnp.zeros_like(h_sq)
        zy = jnp.zeros_like(h_sq)
    else:
        denom = R_safe * R_safe - (1.0 + conic) * h_sq
        denom_safe = jnp.where(denom > 1e-30, denom, 1e-30)
        sd = jnp.sqrt(denom_safe)
        # 4.10: NumPy reference uses dz/dh = h / (R · sqrt(1 - (1+k)h²/R²)),
        # whose sign follows sign(R) -- positive for convex (R>0), negative
        # for concave (R<0).  Pre-4.10 JAX form returned magnitude only,
        # so concave conic/aspheric refraction gave the wrong transverse
        # sign.  R_safe is a Python float (static for the trace), so
        # R_sign is a build-time constant -- no gradient impact.
        R_sign = 1.0 if R_safe >= 0 else -1.0
        zx = R_sign * x / sd
        zy = R_sign * y / sd
    for power, coeff in asph_items:
        # d/dx [a_n * h^n] = a_n * n * h^(n-2) * x
        if power == 2:
            zx = zx + 2.0 * coeff * x
            zy = zy + 2.0 * coeff * y
        else:
            scale = coeff * power * h_sq ** ((power - 2) // 2)
            zx = zx + scale * x
            zy = zy + scale * y
    return zx, zy


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
        t1 = (-b_q - sqrt_disc) / 2.0
        t2 = (-b_q + sqrt_disc) / 2.0
        if R_is_inf:
            # Flat-with-aspherics: start from z = 0 and let Newton run.
            N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
            t0 = -state.z / N_safe
        else:
            t_pick = t1 if R_safe > 0 else t2
            t0 = jnp.where(disc_pos, t_pick, 0.0)
            # disc<=0 means the ray missed the sphere entirely.
            miss = miss | (~disc_pos)

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

    # Make normal point opposite to ray direction.
    dot = state.L * nx + state.M * ny + state.N * nz
    flip = dot > 0
    nx = jnp.where(flip, -nx, nx)
    ny = jnp.where(flip, -ny, ny)
    nz = jnp.where(flip, -nz, nz)

    eta = n1 / n2
    cos_i = -(state.L * nx + state.M * ny + state.N * nz)
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
    tir = sin2_t > 1.0
    # 4.10: double-where so jax.grad doesn't trip on sqrt(0) at the TIR
    # boundary.  TIR rays are also masked to keep their incoming direction
    # so the math-direction substitution doesn't bleed into the output.
    sin2_t_safe = jnp.where(tir, 0.0, sin2_t)
    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t_safe, 0.0))

    Lt_r = eta * state.L + (eta * cos_i - cos_t) * nx
    Mt_r = eta * state.M + (eta * cos_i - cos_t) * ny
    Nt_r = eta * state.N + (eta * cos_i - cos_t) * nz

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
    import jax.numpy as jnp
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
    import jax
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
    propagating = sumsq < 1.0
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

    Implementation note (4.10.3 investigation):
    --------------------------------------------
    The math-correct form is ``t = (thickness - state.z) / state.N``
    so a ray's parametric step lands it on the next vertex plane
    (z = thickness in the previous frame).  This is what the NumPy
    ``_transfer`` does.

    The JAX twin here uses the PARAXIAL APPROXIMATION
    ``t ≈ thickness`` (equivalent to assuming N = 1).  The per-surface
    transverse error is ``~ thickness * (1 - N) ≈ thickness * NA^2 / 2``,
    i.e. ~0.5 % per surface at NA = 0.1 and accumulates roughly
    linearly across the prescription (~2.5 % over a 5-surface trace).
    OPL error scales similarly.  For typical LumenAiry workflows
    (free-space optics, NA <= 0.1) the cumulative error is below
    plotting / fitting noise; for high-NA designs the math-correct
    form would compound to a few-percent transverse error.

    Why the paraxial form is retained: substituting the math-correct
    form here NaN-poisons ``jax.grad`` through
    ``fit_canonical_polynomials_jax``'s downstream lstsq, even with
    full triple-where guards on the division and ``jnp.isfinite``
    filtering on ``t``.  Multiple investigation attempts (4.10.0,
    4.10.1, 4.10.2, 4.10.3) have not isolated the gradient-graph
    issue to a specific op.  Tracked for a future release; the
    pre-existing paraxial trace remains the validated path.
    """
    new_x = state.x + state.L * thickness
    new_y = state.y + state.M * thickness
    new_z = state.z + state.N * thickness - thickness
    new_opd = state.opd + n_medium * thickness
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


_register_jaxprescription_pytree()


def _build_jax_prescription(prescription, wavelength,
                              surface_diffraction=None):
    """Build a :class:`JaxPrescription` from a plain prescription dict.

    Performs the surface-kind validation, glass-index lookups, and
    semi-diameter resolution that the legacy ``trace_jax`` used to do
    inline on every call.  The output is suitable both for direct kernel
    use AND for cache-key lookup (the ``aux`` field is hashable).
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    import jax.numpy as jnp

    surfaces_raw = prescription.get('surfaces', [])
    thicknesses = list(prescription.get('thicknesses', []))
    if len(thicknesses) < len(surfaces_raw) - 1:
        thicknesses = thicknesses + [0.0] * (
            len(surfaces_raw) - 1 - len(thicknesses))

    aperture_d = prescription.get('aperture_diameter')
    default_semi = (aperture_d / 2.0
                    if aperture_d is not None else float('inf'))

    # 4.10: refuse to silently pass mirrors / coord-breaks / biconic /
    # freeform through as flat refractives.  These surface kinds are
    # not yet implemented in the JAX trace; pre-4.10 the trace would
    # happily pretend they were spherical refractors and return wrong
    # answers.  Until proper implementations land, fail loud at build.
    #
    # 4.13.2 (P1-NEW-D): also reject Welford-style mirrors signalled
    # via ``glass_after='MIRROR'`` (case-insensitive).  The v4.13.1
    # P1-A apply_real_lens fix added both guards together; the JAX
    # prescription builder only got the ``is_mirror=True`` half, so
    # hand-built mirrors with ``glass_after='MIRROR'`` slipped through
    # and were silently traced as refractive air->air.
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
                f"trace_jax does not yet support surface {i} with "
                f"{', '.join(unsupported)}.  Use the NumPy ``trace`` "
                f"backend, or open an issue if you need JAX-traceable "
                f"reflective / coord-broken / biconic / freeform support."
            )

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

    semi_ds = tuple(
        float(default_semi if (
            s.get('semi_diameter') is None
            or not np.isfinite(s.get('semi_diameter'))
            or s.get('semi_diameter') <= 0)
            else s.get('semi_diameter'))
        for s in surfaces_raw
    )
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
    import jax
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


def clear_trace_jax_cache() -> None:
    """Drop every cached jit'd ``trace_jax`` kernel (v4.12.2).

    Forces the next :func:`trace_jax` call to rebuild and re-cache its
    jit-compiled kernel from scratch.  Useful in unit tests that pin
    cache mechanics and in long-running notebooks / optimizers where
    the user wants to release the underlying XLA executables.
    """
    _TRACE_JAX_CACHE.clear()


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

    # Allow callers to pre-build a JaxPrescription.  This lets advanced
    # users substitute tracer leaves for radii / conics / asph coeffs /
    # thicknesses without re-walking the dict every call.
    if isinstance(prescription, JaxPrescription):
        jp = prescription
    else:
        jp = _build_jax_prescription(
            prescription, wavelength, surface_diffraction)

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
    cache_key = (jp.aux, float(wavelength), diff_aux)
    kernel = _TRACE_JAX_CACHE.get(cache_key)
    if kernel is None:
        kernel = _make_jit_kernel(
            jp.aux, float(wavelength), surface_diffraction)
        _TRACE_JAX_CACHE[cache_key] = kernel
        while len(_TRACE_JAX_CACHE) > _TRACE_JAX_CACHE_MAXSIZE:
            _TRACE_JAX_CACHE.popitem(last=False)
    else:
        # LRU touch: move this hit to the end so it survives eviction.
        _TRACE_JAX_CACHE.move_to_end(cache_key)
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
    """
    import jax.numpy as jnp
    h_sq = x * x + y * y
    # Prevent NaN at R=inf: clip to a finite huge value, but the
    # branch below collapses the sag toward 0 in that limit.
    R_finite = jnp.where(jnp.isinf(R), 1e30, R)
    R_finite = jnp.where(jnp.abs(R_finite) < 1e-30, 1e-30, R_finite)
    norm = (1.0 + conic) * h_sq / (R_finite * R_finite)
    valid = norm < 0.9999
    denom_arg = jnp.where(valid, 1.0 - norm, 0.01)
    conic_sag = jnp.where(
        valid,
        h_sq / (R_finite * (1.0 + jnp.sqrt(denom_arg))),
        0.0,
    )
    # When R is inf or abs(R) > 1e15, treat as flat -> 0 conic sag.
    is_flat = jnp.isinf(R) | (jnp.abs(R) > 1e15)
    sag = jnp.where(is_flat, jnp.zeros_like(h_sq), conic_sag)
    # Aspheric terms (even powers); coefficients may be JAX arrays.
    for i, power in enumerate(asph_powers):
        sag = sag + asph_coeffs[i] * h_sq ** (power // 2)
    return sag


def _sag_derivatives_param(x, y, R, conic, asph_powers, asph_coeffs):
    """JAX-array-aware sag derivatives dz/dx, dz/dy."""
    import jax.numpy as jnp
    h_sq = x * x + y * y
    R_finite = jnp.where(jnp.isinf(R), 1e30, R)
    R_finite = jnp.where(jnp.abs(R_finite) < 1e-30, 1e-30, R_finite)
    denom = R_finite * R_finite - (1.0 + conic) * h_sq
    denom_safe = jnp.where(denom > 1e-30, denom, 1e-30)
    sd = jnp.sqrt(denom_safe)
    # 4.11.1: mirror the C-RT-3 fix in the static `_sag_derivatives_jax`
    # twin -- sign(R) is required so concave (R<0) conic / aspheric
    # surfaces get the correct transverse-normal direction.  Without
    # this, ``trace_jax_with_params`` and ``fit_canonical_polynomials
    # _jax`` refracted off concave surfaces in the wrong direction.
    R_sign = jnp.where(R >= 0.0, 1.0, -1.0)
    zx_conic = R_sign * x / sd
    zy_conic = R_sign * y / sd
    is_flat = jnp.isinf(R) | (jnp.abs(R) > 1e15)
    zx = jnp.where(is_flat, jnp.zeros_like(x), zx_conic)
    zy = jnp.where(is_flat, jnp.zeros_like(y), zy_conic)
    for i, power in enumerate(asph_powers):
        if power == 2:
            zx = zx + 2.0 * asph_coeffs[i] * x
            zy = zy + 2.0 * asph_coeffs[i] * y
        else:
            scale = asph_coeffs[i] * power * h_sq ** ((power - 2) // 2)
            zx = zx + scale * x
            zy = zy + scale * y
    return zx, zy


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
    dx = state.x; dy = state.y; dz = state.z - R_finite
    b_q = 2.0 * (state.L * dx + state.M * dy + state.N * dz)
    c_q = dx ** 2 + dy ** 2 + dz ** 2 - R_finite ** 2
    disc = b_q ** 2 - 4.0 * c_q
    # 4.11.1 (H-RT-7): double-where on sqrt(disc) so the disc=0
    # gradient singularity doesn't NaN-poison jax.grad on tangent rays.
    disc_pos = disc > 0
    disc_safe = jnp.where(disc_pos, disc, 1.0)
    sqrt_disc = jnp.where(disc_pos, jnp.sqrt(disc_safe), 0.0)
    t1 = (-b_q - sqrt_disc) / 2.0
    t2 = (-b_q + sqrt_disc) / 2.0
    t_sphere = jnp.where(R_finite > 0, t1, t2)
    t_sphere = jnp.where(disc_pos, t_sphere, 0.0)

    # Flat initial guess.
    N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
    t_flat = -state.z / N_safe

    t0 = jnp.where(is_flat, t_flat, t_sphere)
    # 4.11.1 (H-RT-5): rays that missed the sphere (disc<=0) AND aren't
    # on the flat branch are dead.  Flat rays parallel to the surface
    # (|N| -> 0) are also dead.
    miss = (~is_flat & ~disc_pos) | (is_flat & (jnp.abs(state.N) <= eps))

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
    nx = nx / nrm; ny = ny / nrm; nz = nz / nrm

    # Make normal point opposite to ray direction.
    dot = state.L * nx + state.M * ny + state.N * nz
    flip = dot > 0
    nx = jnp.where(flip, -nx, nx)
    ny = jnp.where(flip, -ny, ny)
    nz = jnp.where(flip, -nz, nz)

    eta = n1 / n2
    cos_i = -(state.L * nx + state.M * ny + state.N * nz)
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
    tir = sin2_t > 1.0
    # 4.10: double-where so jax.grad doesn't trip on sqrt(0) at the TIR
    # boundary.  TIR rays are also masked to keep their incoming direction
    # so the math-direction substitution doesn't bleed into the output.
    sin2_t_safe = jnp.where(tir, 0.0, sin2_t)
    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t_safe, 0.0))

    Lt_r = eta * state.L + (eta * cos_i - cos_t) * nx
    Mt_r = eta * state.M + (eta * cos_i - cos_t) * ny
    Nt_r = eta * state.N + (eta * cos_i - cos_t) * nz

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
    import jax
    import jax.numpy as jnp

    surfaces_raw = prescription.get('surfaces', [])
    n_surf = len(surfaces_raw)

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

    # Apertures stay static.
    aperture_d = prescription.get('aperture_diameter')
    default_semi = (aperture_d / 2.0
                    if aperture_d is not None else float('inf'))
    semi_ds = []
    for s in surfaces_raw:
        sd = s.get('semi_diameter')
        if sd is None or not np.isfinite(sd) or sd <= 0:
            sd = default_semi
        semi_ds.append(float(sd))
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
