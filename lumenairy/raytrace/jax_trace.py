"""
lumenairy.raytrace.jax_trace -- JAX-traceable ray trace.

A focused JAX-aware ray trace for spherical and flat refractive
surfaces.  Walks a sequential list of surfaces via ``jax.lax.scan``
so the entire trace is JIT-compilable and differentiable via
``jax.grad`` -- supporting end-to-end optical-system optimization.

Compared to :func:`lumenairy.raytrace.trace`, this implementation
intentionally drops some features to stay within JAX's functional
semantics:

* **Spherical / flat surfaces only.**  Aspheric coefficients are
  ignored (using their paraxial / spherical projection).  The
  Newton-iteration sag solver in the standard trace doesn't
  translate cleanly without ``lax.while_loop``; aspheric support
  is a future extension.
* **Apertures do NOT kill rays** -- the trace produces the
  ``alive`` mask but doesn't terminate paths.  Apply aperture
  masking downstream if needed.
* **No diffraction-order kicks** (``surface_diffraction``).
* **No error_code diagnostic field** -- only a boolean ``alive``
  mask reflecting whether the ray crossed the aperture.

For full-fidelity tracing including aspherics, apertures, DOEs,
and per-ray diagnostic codes, use :func:`lumenairy.raytrace.trace`
on NumPy / CuPy arrays.

The output is a :class:`JaxRayState` named-tuple with the same
geometric fields as :class:`RayBundle`.  Convert via the
:func:`jax_state_to_raybundle` helper if you need to interoperate
with NumPy code downstream.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import NamedTuple, List

import numpy as np

from ..backend import JAX_AVAILABLE
from ..glass import get_glass_index


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


def _intersect_spherical_jax(state, R, n_medium):
    """Intersect rays with a spherical surface centred on the z-axis
    at the previous z=0 (the surface vertex plane).

    For an infinite radius (flat surface), reduces to z' = 0.
    Returns updated state with positions on the surface and OPL
    accumulated through the ``n_medium`` slab.
    """
    import jax.numpy as jnp

    is_flat = jnp.isinf(R)
    # Flat-surface path: t = -z / N
    eps = 1e-30
    N_safe = jnp.where(jnp.abs(state.N) > eps, state.N, eps)
    t_flat = -state.z / N_safe

    # Spherical path: ray-sphere intersection.  Centre of curvature
    # at (0, 0, R).  For an infinite R, R**2 explodes; guard with
    # is_flat selection.
    R_safe = jnp.where(is_flat, 1.0, R)
    dx = state.x; dy = state.y; dz = state.z - R_safe
    a = 1.0
    b = 2.0 * (state.L * dx + state.M * dy + state.N * dz)
    c = dx ** 2 + dy ** 2 + dz ** 2 - R_safe ** 2
    disc = b ** 2 - 4 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t_sph = jnp.where(R_safe > 0, t1, t2)
    t_sph = jnp.where(disc > 0, t_sph, 0.0)

    t = jnp.where(is_flat, t_flat, t_sph)
    t = jnp.where(state.alive, t, 0.0)

    new_x = state.x + state.L * t
    new_y = state.y + state.M * t
    new_z = state.z + state.N * t
    new_opd = state.opd + n_medium * t

    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


def _refract_jax(state, R, n1, n2):
    """Apply vector Snell's law at a spherical surface (centre of
    curvature at (0, 0, R)).

    For a flat surface (infinite R), the surface normal is (0, 0, 1).
    """
    import jax.numpy as jnp

    is_flat = jnp.isinf(R)
    R_safe = jnp.where(is_flat, 1.0, R)

    # Surface normal at the intersection point.  For a sphere centred
    # at (0, 0, R), the unit outward normal at point (x, y, z) is
    #   n_hat = ((x - 0), (y - 0), (z - R)) / R
    # For flat surface, n_hat = (0, 0, 1).
    nx_sph = state.x / R_safe
    ny_sph = state.y / R_safe
    nz_sph = (state.z - R_safe) / R_safe
    # For flat, override with (0, 0, 1).
    nx = jnp.where(is_flat, 0.0, nx_sph)
    ny = jnp.where(is_flat, 0.0, ny_sph)
    nz = jnp.where(is_flat, 1.0, nz_sph)
    # Make normal point opposite to ray direction (forward-facing
    # convention): if (L, M, N) . n > 0, flip n.
    dot = state.L * nx + state.M * ny + state.N * nz
    flip = dot > 0
    nx = jnp.where(flip, -nx, nx)
    ny = jnp.where(flip, -ny, ny)
    nz = jnp.where(flip, -nz, nz)

    # Snell's law in vector form:
    #   n1 (i) - n2 (t) = (n1 cos_i - n2 cos_t) n_hat
    # => t = (n1/n2) i + (n1/n2 cos_i - cos_t) n_hat
    eta = n1 / n2
    cos_i = -(state.L * nx + state.M * ny + state.N * nz)
    sin2_t = eta ** 2 * (1 - cos_i ** 2)
    # TIR mask (sin2_t > 1).  We don't kill rays here; we set their
    # direction unchanged and mark alive=False.
    tir = sin2_t > 1.0
    cos_t_sq = jnp.maximum(1 - sin2_t, 0.0)
    cos_t = jnp.sqrt(cos_t_sq)

    Lt = eta * state.L + (eta * cos_i - cos_t) * nx
    Mt = eta * state.M + (eta * cos_i - cos_t) * ny
    Nt = eta * state.N + (eta * cos_i - cos_t) * nz

    new_alive = state.alive & ~tir
    return JaxRayState(state.x, state.y, state.z, Lt, Mt, Nt,
                       state.opd, new_alive)


def _transfer_jax(state, thickness, n_medium):
    """Free-space transfer through ``thickness`` in medium of index
    ``n_medium``.  All rays advance; OPL accumulates."""
    new_x = state.x + state.L * thickness
    new_y = state.y + state.M * thickness
    new_z = state.z + state.N * thickness - thickness  # reset to next surface frame
    new_opd = state.opd + n_medium * thickness
    return JaxRayState(new_x, new_y, new_z,
                       state.L, state.M, state.N,
                       new_opd, state.alive)


def trace_jax(initial_state, prescription, wavelength):
    """JAX-traceable sequential ray trace.

    Walks the prescription's surfaces and interleaves
    intersect / refract / transfer steps.  Compatible with
    ``jax.jit`` and ``jax.grad``.

    Parameters
    ----------
    initial_state : JaxRayState
        Initial ray state (positions / directions / OPL / alive).
    prescription : dict
        lumenairy prescription.  Surfaces with aspheric coefficients
        are reduced to their spherical / flat base; aspherics are
        ignored.
    wavelength : float
        Vacuum wavelength.

    Returns
    -------
    JaxRayState
        Final ray state at the image plane.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")
    import jax.numpy as jnp

    surfaces_raw = prescription.get('surfaces', [])
    thicknesses = list(prescription.get('thicknesses', []))
    if len(thicknesses) < len(surfaces_raw) - 1:
        thicknesses = thicknesses + [0.0] * (len(surfaces_raw) - 1 - len(thicknesses))

    # Pre-resolve glass indices.
    radii = [float(s.get('radius', float('inf'))) for s in surfaces_raw]
    n_pre = [
        float(get_glass_index(s.get('glass_before', 'air'), wavelength))
        for s in surfaces_raw
    ]
    n_post = [
        float(get_glass_index(s.get('glass_after', 'air'), wavelength))
        for s in surfaces_raw
    ]

    state = initial_state
    for i, surf in enumerate(surfaces_raw):
        R = radii[i]
        n1 = n_pre[i]
        n2 = n_post[i]
        # 1. Intersect spherical/flat surface
        state = _intersect_spherical_jax(state, R, n_medium=n1)
        # 2. Refract
        state = _refract_jax(state, R, n1, n2)
        # 3. Transfer to next surface (or to image plane on last)
        if i < len(surfaces_raw) - 1:
            t = float(thicknesses[i]) if i < len(thicknesses) else 0.0
            state = _transfer_jax(state, t, n_medium=n2)

    return state


def jax_state_to_raybundle(state):
    """Convert a :class:`JaxRayState` to a NumPy
    :class:`lumenairy.raytrace.RayBundle`.

    Useful when the JAX trace is the differentiable inner loop
    but downstream analysis (Strehl, Zernike, etc.) wants NumPy.
    """
    import jax.numpy as jnp
    from .core import RayBundle
    return RayBundle(
        x=np.asarray(state.x),
        y=np.asarray(state.y),
        z=np.asarray(state.z),
        L=np.asarray(state.L),
        M=np.asarray(state.M),
        N=np.asarray(state.N),
        wavelength=0.0,
        opd=np.asarray(state.opd),
        alive=np.asarray(state.alive),
    )


__all__ = [
    'JaxRayState',
    'make_jax_ray_state',
    'trace_jax',
    'jax_state_to_raybundle',
]
