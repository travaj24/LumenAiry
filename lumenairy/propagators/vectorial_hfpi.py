"""
lumenairy.propagators.vectorial_hfpi -- vectorial Huygens-Fresnel
Path Integration.

Extends the scalar HFPI (in :mod:`lumenairy.propagators.hfpi`) to
vector electromagnetic fields by associating a Jones polarization
vector (Ex, Ey) with every path and using the m-theory dipole
obliquity tensor for vector-correct secondary-source amplitudes.

Use cases that require vectorial HFPI:

* High-NA imaging (NA > ~0.3) where polarization rotates strongly
  across the focal plane
* Cascaded diffraction with polarizing elements (waveplates,
  polarizers) interleaved between apertures
* Birefringent elements in the system

For low-NA / unpolarized scalar fields, the scalar HFPI in
:mod:`lumenairy.propagators.hfpi` is faster and equivalent.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..backend import (
    RandomState,
    array_namespace,
    is_jax_array,
    to_numpy,
)
from .hfpi import _spawn_rng


@dataclass
class VectorPathBundle:
    """Vectorial counterpart to :class:`PathBundle` -- carries a Jones
    polarization vector ``(Ex, Ey)`` with every path.

    Attributes
    ----------
    positions : array (N, 3)
    directions : array (N, 3)
    Ex, Ey : array (N,) complex
        Jones polarization vector components in the lab frame.
    opl : array (N,)
    alive : array (N,) bool
    """

    positions: object
    directions: object
    Ex: object
    Ey: object
    opl: object
    alive: object

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except (AttributeError, TypeError, IndexError):
            return 0

    @property
    def n_alive(self) -> int:
        if self.alive is None:
            return len(self)
        return int(np.sum(to_numpy(self.alive)))


def init_vector_paths_from_field(
    Ex_in: np.ndarray,
    Ey_in: np.ndarray,
    dx: float,
    *,
    n_paths: int,
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    z_input_plane: float = 0.0,
) -> VectorPathBundle:
    """Sample paths from a 2-component (Ex, Ey) Jones source field.

    Each path is initialised at a randomly chosen pixel with a
    forward-cone direction (uniform on the spherical cap), and
    inherits the source-pixel's Jones vector ``(Ex[i,j], Ey[i,j])``
    scaled by the obliquity factor ``cos(theta)`` and the pixel
    area ``dx**2``.

    Parameters
    ----------
    Ex_in, Ey_in : array (Ny, Nx) complex
        Jones vector components of the source field.
    Same other parameters as :func:`init_paths_from_field`.

    Returns
    -------
    VectorPathBundle
    """
    xp = array_namespace(Ex_in, Ey_in)
    if Ex_in.shape != Ey_in.shape:
        raise ValueError("Ex_in and Ey_in must have matching shape.")
    Ny, Nx = Ex_in.shape[-2], Ex_in.shape[-1]
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

    sx = Ex_in[iy_xp, ix_xp]
    sy = Ey_in[iy_xp, ix_xp]
    # 4.11.2: full HF Kirchhoff weighting per Jones component (mirrors
    # the scalar :func:`lumenairy.propagators.hfpi.init_paths_from_field`
    # 4.10 fix).  Pre-4.11.2 the vector source weights carried only the
    # cosθ·dx² obliquity factor -- the ``1/(iλ)`` prefactor and the
    # Monte Carlo solid-angle weight ``2π(1-cosθ_max)/N_paths`` were
    # absent, so absolute Jones amplitudes were unphysical by ~10^6 at
    # visible wavelengths.  Relative polarization structure was
    # unaffected (factor is global).
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n_paths)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    kirchhoff = complex(inv_i_lambda) * solid_angle
    obl = cos_theta * (dx * dx)
    obl_complex = obl.astype(sx.dtype) if hasattr(obl, 'astype') else obl
    Ex_paths = sx * obl_complex * kirchhoff
    Ey_paths = sy * obl_complex * kirchhoff

    opl = xp.zeros((n_paths,), dtype=xp.real(sx).dtype)
    alive = xp.ones((n_paths,), dtype=bool)

    return VectorPathBundle(
        positions=positions,
        directions=directions,
        Ex=Ex_paths,
        Ey=Ey_paths,
        opl=opl,
        alive=alive,
    )


def propagate_vector_to_plane(
    paths: VectorPathBundle,
    z_target: float,
    wavelength: float,
    *,
    n_medium: float = 1.0,
) -> VectorPathBundle:
    """Free-space advance of every alive vector-path to ``z_target``.

    The Jones vector picks up a global phase ``exp(i k OPL)``;
    polarization-rotation-by-propagation effects (which the
    full m-theory dipole formalism captures for general directions)
    are neglected for paraxial advances.  At each diffracting
    surface, see :func:`apply_vector_aperture_diffraction`.
    """
    xp = array_namespace(paths.positions)
    z_curr = paths.positions[..., 2]
    Nz = paths.directions[..., 2]
    eps = 1e-30
    t = (z_target - z_curr) / xp.where(xp.abs(Nz) > eps, Nz, eps)
    new_alive = paths.alive & (t >= 0) & (xp.abs(Nz) > eps)
    # 4.13.2 (P1-NEW-H): zero the step for grazing / dead rays so their
    # position update is a no-op.  Mirrors the scalar hfpi fix; see
    # :func:`lumenairy.propagators.hfpi.propagate_to_plane` for details.
    t = xp.where(new_alive, t, 0.0)
    new_positions = paths.positions + t[..., None] * paths.directions
    delta_opl = n_medium * xp.abs(t)
    new_opl = paths.opl + delta_opl
    k = 2 * float(np.pi) / wavelength
    phase = xp.exp(1j * k * delta_opl).astype(paths.Ex.dtype)
    return VectorPathBundle(
        positions=new_positions,
        directions=paths.directions,
        Ex=paths.Ex * phase,
        Ey=paths.Ey * phase,
        opl=new_opl,
        alive=new_alive,
    )


def apply_vector_aperture_diffraction(
    paths: VectorPathBundle,
    aperture_radius: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
    shape: str = 'circular',
    wavelength: float = 0.0,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    vector_projection: bool = False,
) -> VectorPathBundle:
    """Vectorial counterpart of :func:`apply_aperture_diffraction`.

    Paths landing outside the aperture are killed.  Surviving paths
    re-emit secondary HF sources at their current position with
    fresh forward-cone directions; their Jones vector is multiplied
    by ``cos(theta_new)`` to account for the m-theory dipole
    obliquity, and the OPL accumulator is reset.

    Parameters
    ----------
    vector_projection : bool, default False
        v5.4.6 (audit P3-24): if ``True``, additionally project the input
        Jones vector onto the new direction's transverse plane,
        ``E_t = E_in - (E_in . rho_hat) rho_hat`` (with the unrepresentable
        longitudinal Ez' component dropped, since ``VectorPathBundle`` is
        2-component).  The default (``False``) keeps the historical scalar-
        magnitude obliquity weighting only.  For a fully rigorous high-NA
        vector focus, use
        :func:`lumenairy.propagators.vector_diffraction.richards_wolf_focus`,
        which carries the Ez component explicitly.
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
            f"shape must be 'circular' or 'square', got {shape!r}.")

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

    # 4.11.2: include the Kirchhoff ``1/(iλ)·dΩ`` factor per
    # re-emission, matching the scalar
    # :func:`lumenairy.propagators.hfpi.apply_aperture_diffraction`
    # 4.11.2 fix.  Pre-4.11.2 cascaded vector apertures dropped this
    # global factor, so multi-aperture vector HFPI underweighted by
    # ~10^6 per extra aperture at visible wavelengths.  Relative
    # polarization / phase structure unaffected.
    cos_theta_in = paths.directions[..., 2]
    obliquity = 0.5 * (cos_theta_in + cos_theta)
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    kirchhoff = complex(inv_i_lambda) * solid_angle
    obl = obliquity.astype(paths.Ex.dtype)
    # v5.4.6 (audit P3-24): optional transverse projection of the Jones
    # vector onto the new propagation direction's transverse plane.
    # E_in = (Ex, Ey, 0); rho_hat = (L, M, Nz); E.rho_hat = Ex*L + Ey*M.
    # The transverse parts of E - (E.rho_hat) rho_hat are kept; the
    # longitudinal Ez' = -(E.rho_hat) Nz is dropped (2-component bundle).
    if vector_projection:
        proj = (paths.Ex * xp.asarray(L).astype(paths.Ex.dtype)
                + paths.Ey * xp.asarray(M).astype(paths.Ey.dtype))
        Ex_t = paths.Ex - proj * xp.asarray(L).astype(paths.Ex.dtype)
        Ey_t = paths.Ey - proj * xp.asarray(M).astype(paths.Ey.dtype)
    else:
        Ex_t, Ey_t = paths.Ex, paths.Ey
    new_Ex = Ex_t * obl * kirchhoff
    new_Ey = Ey_t * obl * kirchhoff
    new_opl = xp.zeros_like(paths.opl)
    return VectorPathBundle(
        positions=paths.positions,
        directions=new_directions,
        Ex=new_Ex,
        Ey=new_Ey,
        opl=new_opl,
        alive=survives,
    )


def accumulate_vector_to_grid(
    paths: VectorPathBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    output_dtype: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coherently bin a VectorPathBundle into separate (Ex, Ey)
    output grids.

    Returns
    -------
    Ex_out, Ey_out : tuple of arrays (Ny, Nx) complex

    Notes
    -----
    v4.13.1 perf: shares pixel index computation (``ix``, ``iy``,
    ``inside``, ``flat_idx``) between the Ex and Ey scatter-adds.
    Pre-v4.13.1 routed through
    :func:`hfpi.accumulate_to_grid` twice, recomputing the same
    index arrays on each call.  The new path computes them once,
    then runs two scatter-adds.  Numerically bit-identical to the
    twice-routed version (same arithmetic, same scatter pattern).
    On JAX the underlying ``out.at[flat_idx].add(...)`` runs
    independently per component because JAX traces can't share
    work the same way; we keep the original path for JAX inputs to
    preserve tracing compatibility.
    """
    if output_dtype is None:
        output_dtype = paths.Ex.dtype

    xp = array_namespace(paths.positions)
    # JAX path: keep the original double-call form so jax.jit / vmap
    # over the pipeline can trace through it unchanged.
    if is_jax_array(paths.positions):
        from .hfpi import PathBundle
        from .hfpi import accumulate_to_grid as _scalar_acc
        ex_paths = PathBundle(
            positions=paths.positions, directions=paths.directions,
            weights=paths.Ex, opl=paths.opl, alive=paths.alive,
        )
        ey_paths = PathBundle(
            positions=paths.positions, directions=paths.directions,
            weights=paths.Ey, opl=paths.opl, alive=paths.alive,
        )
        Ex_out = _scalar_acc(
            ex_paths, Ny=Ny, Nx=Nx, dx=dx, centre=centre,
            output_dtype=output_dtype)
        Ey_out = _scalar_acc(
            ey_paths, Ny=Ny, Nx=Nx, dx=dx, centre=centre,
            output_dtype=output_dtype)
        return Ex_out, Ey_out

    # Shared-index NumPy / CuPy path.  Mirrors
    # :func:`hfpi.accumulate_to_grid` bit-for-bit on each component
    # but builds the (ix, iy, inside, flat_idx) tuple only once.
    cx, cy = centre
    x = paths.positions[..., 0] - cx
    y = paths.positions[..., 1] - cy
    ix = xp.floor(x / dx + Nx / 2).astype(xp.int64)
    iy = xp.floor(y / dx + Ny / 2).astype(xp.int64)
    inside = ((ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
              & paths.alive)
    flat_idx = xp.where(inside, iy * Nx + ix, 0)
    Ex_masked = xp.where(inside, paths.Ex, 0)
    Ey_masked = xp.where(inside, paths.Ey, 0)

    N_flat = Ny * Nx
    Ex_out_flat = xp.zeros(N_flat, dtype=output_dtype)
    Ey_out_flat = xp.zeros(N_flat, dtype=output_dtype)
    if hasattr(xp, 'add') and hasattr(xp.add, 'at'):
        xp.add.at(Ex_out_flat, flat_idx, Ex_masked)
        xp.add.at(Ey_out_flat, flat_idx, Ey_masked)
    else:
        # CuPy fallback: cupyx.scatter_add or NumPy round-trip.
        try:
            import cupyx
            cupyx.scatter_add(Ex_out_flat, flat_idx, Ex_masked)
            cupyx.scatter_add(Ey_out_flat, flat_idx, Ey_masked)
        except (ImportError, AttributeError, TypeError, ValueError):
            idx_h = to_numpy(flat_idx)
            ex_h = to_numpy(Ex_masked)
            ey_h = to_numpy(Ey_masked)
            ex_host = np.zeros(N_flat, dtype=output_dtype)
            ey_host = np.zeros(N_flat, dtype=output_dtype)
            np.add.at(ex_host, idx_h, ex_h)
            np.add.at(ey_host, idx_h, ey_h)
            Ex_out_flat = xp.asarray(ex_host)
            Ey_out_flat = xp.asarray(ey_host)
    return Ex_out_flat.reshape(Ny, Nx), Ey_out_flat.reshape(Ny, Nx)


def propagate_vector_hfpi_freespace_aperture(
    Ex_in: np.ndarray,
    Ey_in: np.ndarray,
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
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end vectorial HFPI: vector source -> free-space hop ->
    aperture -> free-space hop -> vector output.

    Returns
    -------
    Ex_out, Ey_out : tuple of arrays (Ny, Nx) complex
    """
    # 4.13.2 (P1-NEW-A): spawn a distinct child RNG for the aperture
    # re-emission so source-plane init and aperture re-sample are
    # statistically independent.  Mirrors the scalar
    # :func:`hfpi.propagate_hfpi_freespace_aperture` 4.11.2 fix.
    # Pre-4.13.2 the same int ``rng`` was reused at both sites;
    # ``RandomState(rng=int)`` rebuilds default_rng(int) so init and
    # re-emission drew identical samples (perfectly correlated).
    rng_source = _spawn_rng(rng, 0)
    rng_aperture = _spawn_rng(rng, 1)
    paths = init_vector_paths_from_field(
        Ex_in, Ey_in, dx,
        n_paths=n_paths,
        wavelength=wavelength, rng=rng_source,
        z_input_plane=0.0,
    )
    paths = propagate_vector_to_plane(paths, z_target=z_to_aperture,
                                       wavelength=wavelength)
    paths = apply_vector_aperture_diffraction(
        paths, aperture_radius=aperture_radius,
        centre=aperture_centre, shape=aperture_shape,
        wavelength=wavelength, rng=rng_aperture,
    )
    paths = propagate_vector_to_plane(
        paths, z_target=z_to_aperture + z_aperture_to_output,
        wavelength=wavelength,
    )
    Ny, Nx = (Ex_in.shape[-2], Ex_in.shape[-1]) if output_grid is None else output_grid
    if output_dx is None:
        output_dx = dx
    return accumulate_vector_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=output_dx, centre=output_centre,
        output_dtype=Ex_in.dtype,
    )


__all__ = [
    'VectorPathBundle',
    'init_vector_paths_from_field',
    'propagate_vector_to_plane',
    'apply_vector_aperture_diffraction',
    'accumulate_vector_to_grid',
    'propagate_vector_hfpi_freespace_aperture',
]
