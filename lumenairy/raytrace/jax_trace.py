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

from typing import NamedTuple

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

    For rotationally symmetric z(h) with h^2 = x^2 + y^2:
      conic part:  dz/dx = x / sqrt(R^2 - (1+k) h^2)
                   dz/dy = y / sqrt(R^2 - (1+k) h^2)
      aspheric:    a_n * n * h^(n-2) * x   (and similarly for y)
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
        zx = x / sd
        zy = y / sd
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

    if R_is_inf and not has_aspheric:
        # Pure flat surface: t = -z / N.
        N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
        t = -state.z / N_safe
    else:
        # Spherical initial guess (also exact for conic == 0 / no asph).
        dx = state.x
        dy = state.y
        dz = state.z - R_safe
        b_q = 2.0 * (state.L * dx + state.M * dy + state.N * dz)
        c_q = dx ** 2 + dy ** 2 + dz ** 2 - R_safe ** 2
        disc = b_q ** 2 - 4.0 * c_q
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t1 = (-b_q - sqrt_disc) / 2.0
        t2 = (-b_q + sqrt_disc) / 2.0
        if R_is_inf:
            # Flat-with-aspherics: start from z = 0 and let Newton run.
            N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
            t0 = -state.z / N_safe
        else:
            t_pick = t1 if R_safe > 0 else t2
            t0 = jnp.where(disc > 0, t_pick, 0.0)

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
                step = jnp.where(jnp.abs(dF_dt) > eps, F / dF_dt, 0.0)
                return t - step

            t = jax.lax.fori_loop(0, _ASPHERIC_NEWTON_ITERS, body, t0)
        else:
            t = t0

    t = jnp.where(state.alive, t, 0.0)

    new_x = state.x + state.L * t
    new_y = state.y + state.M * t
    new_z = state.z + state.N * t
    new_opd = state.opd + n_medium * t

    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


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
    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))

    Lt = eta * state.L + (eta * cos_i - cos_t) * nx
    Mt = eta * state.M + (eta * cos_i - cos_t) * ny
    Nt = eta * state.N + (eta * cos_i - cos_t) * nz

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
    intersection point ``(x, y)``.  Use ``np.inf`` for ``period_y`` to
    disable the y-axis grating (1-D grating along x).

    Rays whose post-kick transverse direction cosines exceed unity
    (evanescent orders) are marked dead.
    """
    import jax.numpy as jnp

    dL = float(order_x) * wavelength / float(period_x) \
        if np.isfinite(period_x) and period_x != 0 else 0.0
    dM = float(order_y) * wavelength / float(period_y) \
        if np.isfinite(period_y) and period_y != 0 else 0.0

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
    ``n_medium``.  All rays advance; OPL accumulates."""
    new_x = state.x + state.L * thickness
    new_y = state.y + state.M * thickness
    new_z = state.z + state.N * thickness - thickness  # next-surface frame
    new_opd = state.opd + n_medium * thickness
    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


# ----------------------------------------------------------------------
# Top-level trace
# ----------------------------------------------------------------------

def trace_jax(initial_state, prescription, wavelength,
              surface_diffraction=None):
    """JAX-traceable sequential ray trace.

    Walks the prescription's surfaces and interleaves
    intersect / refract / aperture-clip / DOE-kick / transfer steps.
    Compatible with ``jax.jit`` and ``jax.grad``.

    Parameters
    ----------
    initial_state : JaxRayState
        Initial ray state (positions / directions / OPL / alive).
    prescription : dict
        lumenairy prescription.  Each entry of ``surfaces`` is read for
        ``radius``, ``conic``, ``aspheric_coeffs``, ``glass_before``,
        ``glass_after``, and ``semi_diameter`` (or ``aperture_diameter``
        on the top-level dict).  Aspheric surfaces are intersected via
        Newton iteration with a fixed iteration count.
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

    surfaces_raw = prescription.get('surfaces', [])
    thicknesses = list(prescription.get('thicknesses', []))
    if len(thicknesses) < len(surfaces_raw) - 1:
        thicknesses = thicknesses + [0.0] * (
            len(surfaces_raw) - 1 - len(thicknesses))

    # Top-level aperture (used as a default semi-diameter).
    aperture_d = prescription.get('aperture_diameter')
    default_semi = (aperture_d / 2.0
                    if aperture_d is not None else float('inf'))

    # Pre-resolve glass indices and aspheric / aperture metadata.
    radii = [float(s.get('radius', float('inf'))) for s in surfaces_raw]
    conics = [float(s.get('conic', 0.0)) for s in surfaces_raw]
    asph_lists = []
    for s in surfaces_raw:
        a = s.get('aspheric_coeffs') or {}
        asph_lists.append(tuple(sorted(
            (int(p), float(c)) for p, c in a.items())))
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
    for i, _surf in enumerate(surfaces_raw):
        R = radii[i]
        kc = conics[i]
        asph = asph_lists[i]
        sd = semi_ds[i]
        n1 = n_pre[i]
        n2 = n_post[i]

        # 1. Intersect (closed-form for spherical/flat, Newton for asph).
        state = _intersect_jax(state, R, kc, asph, n_medium=n1)

        # 2. Refract.
        state = _refract_jax(state, R, kc, asph, n1, n2)

        # 3. Aperture clip (vignette rays outside the clear aperture).
        state = _apply_aperture_jax(state, sd)

        # 4. DOE / grating order kick (if specified for this surface).
        if i in diff:
            ox, oy, px, py = diff[i]
            state = _apply_doe_kick_jax(
                state, ox, oy, px, py, float(wavelength))

        # 5. Transfer to next surface (or to image plane on last).
        if i < len(surfaces_raw) - 1:
            t = float(thicknesses[i]) if i < len(thicknesses) else 0.0
            state = _transfer_jax(state, t, n_medium=n2)

    return state


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
    zx_conic = x / sd
    zy_conic = y / sd
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
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b_q - sqrt_disc) / 2.0
    t2 = (-b_q + sqrt_disc) / 2.0
    t_sphere = jnp.where(R_finite > 0, t1, t2)
    t_sphere = jnp.where(disc > 0, t_sphere, 0.0)

    # Flat initial guess.
    N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
    t_flat = -state.z / N_safe

    t0 = jnp.where(is_flat, t_flat, t_sphere)

    # Newton refinement.  For pure-spherical/flat (no asph, conic=0)
    # this is essentially a no-op (1 iter to verify convergence).
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
        step = jnp.where(jnp.abs(dF_dt) > eps, F / dF_dt, 0.0)
        return t - step

    t = jax.lax.fori_loop(0, _ASPHERIC_NEWTON_ITERS, body, t0)
    t = jnp.where(state.alive, t, 0.0)

    new_x = state.x + state.L * t
    new_y = state.y + state.M * t
    new_z = state.z + state.N * t
    new_opd = state.opd + n_medium * t
    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


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
    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t, 0.0))

    Lt = eta * state.L + (eta * cos_i - cos_t) * nx
    Mt = eta * state.M + (eta * cos_i - cos_t) * ny
    Nt = eta * state.N + (eta * cos_i - cos_t) * nz

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
    'make_jax_ray_state',
    'trace_jax',
    'trace_jax_with_params',
    'jax_state_to_raybundle',
    'raybundle_to_jax_state',
]
