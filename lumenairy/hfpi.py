"""
lumenairy.hfpi -- Huygens-Fresnel Path Integration.

Monte Carlo ray-based diffraction propagator that combines the
Huygens-Fresnel principle (every wavefront point is a secondary
source) with geometric ray tracing between diffracting surfaces
and coherent accumulation at the output plane.

Strengths:

* Handles **cascaded diffraction** natively (multi-DOE, multi-stop)
  where exit-pupil approximation methods fail.
* Works at **any output plane**, including conjugate planes.
* Handles **arbitrary aperture geometries** (hard cutoffs).
* Embarrassingly parallel.

Trade-off: 1/sqrt(N_paths) Monte Carlo convergence.

See ``REFERENCES.txt`` Section C for the foundational publications.

Multi-backend
-------------

The full pipeline is written against
:func:`lumenairy._array.array_namespace`, accepting NumPy / CuPy /
JAX source fields and returning the same backend.  Random sampling
goes through :class:`lumenairy._random.RandomState`.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from ._array import (
    array_namespace, is_jax_array, to_numpy,
)
from ._random import RandomState


# ============================================================================
# Path bundle
# ============================================================================

@dataclass
class PathBundle:
    """Complex-weighted ray bundle for HFPI.

    Field naming aligns with :class:`lumenairy.raytrace.RayBundle`:
    ``positions`` / ``directions`` / ``opl`` / ``alive`` are shared.
    The HFPI-specific addition is ``weights`` (complex amplitude
    carried by each path).
    """

    positions: object       # (N, 3) array
    directions: object      # (N, 3) array
    weights: object         # (N,) complex array
    opl: object             # (N,) float array
    alive: object           # (N,) bool array

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except Exception:
            return 0

    @property
    def n_alive(self) -> int:
        if self.alive is None:
            return len(self)
        return int(np.sum(to_numpy(self.alive)))


# ============================================================================
# Source-plane sampling
# ============================================================================

def init_paths_from_field(
    E_in,
    dx: float,
    *,
    n_paths: int,
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    z_input_plane: float = 0.0,
):
    """Sample ``n_paths`` Huygens-Fresnel paths from a complex source
    field on a uniform 2-D grid of pitch ``dx``.

    Each path is initialised at a randomly chosen pixel of the
    source grid, with a direction drawn uniformly inside the
    forward hemisphere clipped to ``cone_half_angle``.  The complex
    weight is ``E_in[i, j] * cos(theta) * dx**2`` where ``cos(theta)``
    is the obliquity factor.
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    rs = RandomState(rng=rng if rng is not None else 0)

    iy = rs.integers((n_paths,), low=0, high=Ny)
    ix = rs.integers((n_paths,), low=0, high=Nx)

    iy_xp = xp.asarray(iy)
    ix_xp = xp.asarray(ix)
    x_s = (ix_xp - Nx / 2) * dx
    y_s = (iy_xp - Ny / 2) * dx
    z_s = xp.full((n_paths,), float(z_input_plane), dtype=x_s.dtype)
    positions = xp.stack([x_s, y_s, z_s], axis=-1)

    cos_max = float(np.cos(cone_half_angle))
    u = rs.uniform((n_paths,))
    phi = rs.uniform((n_paths,), low=0.0, high=2 * float(np.pi))
    cos_theta = 1.0 - xp.asarray(u) * (1.0 - cos_max)
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta ** 2, 0.0))
    L = sin_theta * xp.cos(xp.asarray(phi))
    M = sin_theta * xp.sin(xp.asarray(phi))
    N = cos_theta
    directions = xp.stack([L, M, N], axis=-1)

    sample = E_in[iy_xp, ix_xp]
    weights = sample * cos_theta * (dx * dx)

    opl = xp.zeros((n_paths,), dtype=xp.real(sample).dtype)
    alive = xp.ones((n_paths,), dtype=bool)

    return PathBundle(
        positions=positions,
        directions=directions,
        weights=weights,
        opl=opl,
        alive=alive,
    )


# ============================================================================
# Free-space propagation between planes
# ============================================================================

def propagate_to_plane(
    paths: PathBundle,
    z_target: float,
    wavelength: float,
    *,
    n_medium: float = 1.0,
) -> PathBundle:
    """Geometrically advance every alive path to ``z_target`` along
    its direction vector, accumulating OPL and phase."""
    xp = array_namespace(paths.positions)
    z_curr = paths.positions[..., 2]
    Nz = paths.directions[..., 2]
    eps = 1e-30
    t = (z_target - z_curr) / xp.where(xp.abs(Nz) > eps, Nz, eps)

    new_alive = paths.alive & (t >= 0) & (xp.abs(Nz) > eps)

    new_positions = paths.positions + t[..., None] * paths.directions
    delta_opl = n_medium * xp.abs(t)
    new_opl = paths.opl + delta_opl
    k = 2 * float(np.pi) / wavelength
    phase = xp.exp(1j * k * delta_opl).astype(paths.weights.dtype)
    new_weights = paths.weights * phase

    return PathBundle(
        positions=new_positions,
        directions=paths.directions,
        weights=new_weights,
        opl=new_opl,
        alive=new_alive,
    )


# ============================================================================
# Aperture / hard-cutoff diffraction
# ============================================================================

def apply_aperture_diffraction(
    paths: PathBundle,
    aperture_radius: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
    shape: str = 'circular',
    wavelength: float = 0.0,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
) -> PathBundle:
    """Apply a hard aperture at the current path-bundle plane.

    Paths landing outside the aperture are killed.  Surviving paths
    re-emit secondary HF sources at their current position with a
    fresh direction sample.  OPL is reset since the new secondary
    source's accumulator starts at zero.
    """
    xp = array_namespace(paths.positions)
    rs = RandomState(rng=rng if rng is not None else 0)

    cx, cy = centre
    x = paths.positions[..., 0] - cx
    y = paths.positions[..., 1] - cy

    if shape == 'circular':
        in_aperture = x * x + y * y <= aperture_radius * aperture_radius
    elif shape == 'square':
        in_aperture = (xp.abs(x) <= aperture_radius) & (xp.abs(y) <= aperture_radius)
    else:
        raise ValueError(
            f"apply_aperture_diffraction: shape must be 'circular' or "
            f"'square', got {shape!r}.")

    survives = paths.alive & in_aperture

    n = int(paths.positions.shape[0])
    cos_max = float(np.cos(cone_half_angle))
    u = rs.uniform((n,))
    phi = rs.uniform((n,), low=0.0, high=2 * float(np.pi))
    cos_theta = 1.0 - xp.asarray(u) * (1.0 - cos_max)
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta ** 2, 0.0))
    L = sin_theta * xp.cos(xp.asarray(phi))
    M = sin_theta * xp.sin(xp.asarray(phi))
    Nz = cos_theta
    new_directions = xp.stack([L, M, Nz], axis=-1)

    new_weights = paths.weights * cos_theta.astype(paths.weights.dtype)
    new_opl = xp.zeros_like(paths.opl)

    return PathBundle(
        positions=paths.positions,
        directions=new_directions,
        weights=new_weights,
        opl=new_opl,
        alive=survives,
    )


# ============================================================================
# Coherent accumulation at the output plane
# ============================================================================

def accumulate_to_grid(
    paths: PathBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    output_dtype=None,
) -> object:
    """Coherently bin a PathBundle into a 2-D output field.

    For each path, identify the destination pixel and add the
    path's complex weight to that pixel.  Paths that fall outside
    the grid are dropped.
    """
    xp = array_namespace(paths.positions)
    cx, cy = centre

    if output_dtype is None:
        output_dtype = paths.weights.dtype

    x = paths.positions[..., 0] - cx
    y = paths.positions[..., 1] - cy
    ix = xp.floor(x / dx + Nx / 2).astype(xp.int64)
    iy = xp.floor(y / dx + Ny / 2).astype(xp.int64)
    inside = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny) & paths.alive

    w_masked = xp.where(inside, paths.weights, 0)
    flat_idx = xp.where(inside, iy * Nx + ix, 0)

    if is_jax_array(paths.positions):
        import jax.numpy as jnp
        out = jnp.zeros(Ny * Nx, dtype=output_dtype)
        out = out.at[flat_idx].add(w_masked)
        return out.reshape(Ny, Nx)

    out = xp.zeros(Ny * Nx, dtype=output_dtype)
    if hasattr(xp, 'add') and hasattr(xp.add, 'at'):
        xp.add.at(out, flat_idx, w_masked)
    else:
        try:
            import cupyx
            cupyx.scatter_add(out, flat_idx, w_masked)
        except Exception:
            idx_h = to_numpy(flat_idx)
            w_h = to_numpy(w_masked)
            out_h = np.zeros(Ny * Nx, dtype=output_dtype)
            np.add.at(out_h, idx_h, w_h)
            out = xp.asarray(out_h)
    return out.reshape(Ny, Nx)


# ============================================================================
# End-to-end convenience
# ============================================================================

def propagate_hfpi_freespace_aperture(
    E_in,
    dx: float,
    *,
    z_to_aperture: float,
    aperture_radius: float,
    z_aperture_to_output: float,
    wavelength: float,
    n_paths: int,
    rng: Optional[Union[int, object]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    aperture_shape: str = 'circular',
    aperture_centre: Tuple[float, float] = (0.0, 0.0),
):
    """End-to-end three-leg HFPI: source plane -> free space ->
    aperture -> free space -> output plane.

    The canonical single-aperture-diffraction validation case.
    """
    paths = init_paths_from_field(
        E_in, dx,
        n_paths=n_paths,
        wavelength=wavelength,
        rng=rng,
        z_input_plane=0.0,
    )
    paths = propagate_to_plane(paths, z_target=z_to_aperture,
                                wavelength=wavelength)
    paths = apply_aperture_diffraction(
        paths,
        aperture_radius=aperture_radius,
        centre=aperture_centre,
        shape=aperture_shape,
        wavelength=wavelength,
        rng=rng,
    )
    paths = propagate_to_plane(paths,
                                z_target=z_to_aperture + z_aperture_to_output,
                                wavelength=wavelength)

    if output_grid is None:
        output_grid = E_in.shape[-2], E_in.shape[-1]
    if output_dx is None:
        output_dx = dx
    Ny, Nx = output_grid
    return accumulate_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=output_dx, centre=output_centre,
        output_dtype=E_in.dtype,
    )


__all__ = [
    'PathBundle',
    'init_paths_from_field',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi_freespace_aperture',
]
