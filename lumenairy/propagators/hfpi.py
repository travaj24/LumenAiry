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

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..backend import (
    RandomState,
    array_namespace,
    is_jax_array,
    to_numpy,
)


# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation; see :mod:`lumenairy.propagators.gbd` for the full
# rationale.  ``output_grid`` is renamed to ``output_shape`` here for
# the ``(Ny, Nx)`` shape-only meaning; the legacy ``output_grid``
# spelling keeps working but emits a ``DeprecationWarning``.
def _resolve_output_shape(
    output_shape: Optional[Tuple[int, int]],
    output_grid: Optional[Any],
    *,
    fn_name: str,
    default_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the (Ny, Nx) output shape from the v5.2 ``output_shape``
    kwarg and the deprecated ``output_grid`` legacy kwarg."""
    if output_shape is not None and output_grid is not None:
        raise ValueError(
            f"{fn_name}: both ``output_shape`` and ``output_grid`` were "
            f"provided.  Pass only ``output_shape=(Ny, Nx)`` (v5.2+) or "
            f"the dispatcher's ``output_grid=(N_out, dx_out)`` form via "
            f"``propagate(method=...)``.")
    if output_shape is not None:
        return (int(output_shape[0]), int(output_shape[1]))
    if output_grid is not None:
        warnings.warn(
            f"{fn_name}: the ``output_grid`` kwarg now (v5.2+) means "
            f"the dispatcher's ``(N_out, dx_out)`` grid spec; on "
            f"sub-propagators it has been renamed to ``output_shape`` "
            f"for the ``(Ny, Nx)`` shape-only meaning.  Pass "
            f"``output_shape=(Ny, Nx)`` to silence this warning, or "
            f"call via ``propagate(method='hfpi', output_grid=(N_out, "
            f"dx_out), ...)`` if you actually want grid resampling.",
            DeprecationWarning, stacklevel=3,
        )
        return (int(output_grid[0]), int(output_grid[1]))
    return default_shape


def _spawn_rng(rng, stream_index: int):
    """Derive a per-aperture child RNG so consecutive diffraction
    events draw independent samples.

    4.11.2: HFPI prescription walks previously called
    ``apply_aperture_diffraction(... rng=rng)`` with the same caller-
    supplied ``rng`` at every aperture.  Because
    :class:`lumenairy.backend.random.RandomState` rebuilds a fresh
    NumPy ``default_rng(rng)`` on every construction when ``rng`` is
    an integer (or None), passing the same int seed produced *the
    same uniform draws at every aperture* -- perfectly correlated
    diffraction events across the cascaded stack.  We derive a
    distinct child seed per stream index by hashing the parent seed
    with the stream counter.
    """
    if rng is None:
        # Caller did not pin a seed; let each aperture pull from
        # system entropy (RandomState(None) constructs a fresh
        # generator from a non-deterministic seed).
        return None
    if isinstance(rng, (int, np.integer)):
        # Mix the int seed with the stream index via a 64-bit hash.
        ss = np.random.SeedSequence(entropy=[int(rng), int(stream_index)])
        return int(ss.generate_state(1)[0])
    if isinstance(rng, np.random.Generator):
        # Use the generator's SeedSequence-spawn machinery if
        # available; otherwise pull a child seed.
        try:
            spawned = rng.spawn(stream_index + 1)[-1]
            return spawned
        except AttributeError:
            # NumPy < 1.25 -- fall back to drawing a 64-bit seed.
            child_seed = int(rng.integers(0, 2 ** 63 - 1))
            return np.random.default_rng(child_seed)
    # JAX PRNGKey: caller-side splitting is the canonical pattern,
    # but we can deterministically fold the stream index.
    try:
        import jax
        if hasattr(jax.random, 'fold_in'):
            return jax.random.fold_in(rng, stream_index)
    except Exception:
        pass
    # Unknown rng type: return as-is (preserves prior behaviour).
    return rng


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
        except (AttributeError, TypeError, IndexError):
            # positions may be None, a non-array sentinel, or 0-D --
            # all of which mean "no paths".
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
    E_in: np.ndarray,
    dx: float,
    *,
    n_paths: int,
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    z_input_plane: float = 0.0,
) -> PathBundle:
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
    # 4.10: HF Kirchhoff weighting.
    #   w_path = E_in(x_s) * cos(theta) * (1/(i*lambda)) * dOmega
    # with dOmega = 2*pi*(1-cos(theta_max)) / N_paths (uniform-cone MC).
    # Pre-4.10 omitted both 1/(i*lambda) and the solid-angle weight,
    # so absolute amplitudes were unphysical by ~10^6 per re-emission
    # at visible wavelengths.  Intensity ratios across paths were
    # unaffected (the missing factors are global), so existing relative-
    # contrast results still hold; absolute-photometry use is new.
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n_paths)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    weights = (sample * cos_theta * (dx * dx)
               * complex(inv_i_lambda) * solid_angle)

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

    # 4.13.2 (P1-NEW-H): zero the step for grazing / dead rays so their
    # position update is a no-op.  Pre-4.13.2 grazing rays with
    # ``|Nz| <= eps`` and ``z_target != z_curr`` produced ``t ~ 1e30``
    # which poisoned ``new_positions`` with +/-inf even though
    # ``new_alive`` correctly tagged them dead.  Downstream consumers
    # that read ``paths.positions`` without re-masking by ``alive``
    # (e.g. _hfpi_segment_trace) then carried inf/NaN into the trace.
    t = xp.where(new_alive, t, 0.0)
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

    # 4.10.2: Kirchhoff obliquity must use the angle between the
    # INCOMING ray direction and the surface normal (+z for a flat
    # aperture), AVERAGED with the outgoing-ray cos as per the
    # symmetric form (cos θ_in + cos θ_out)/2.  Pre-4.10.2 used only
    # the outgoing-ray cos relative to +z, which made the obliquity
    # weight anisotropic on tilted apertures or cascaded apertures
    # with strong oblique paths.  For a flat aperture with light
    # arriving on-axis cos θ_in ≈ 1 and this reduces to the original.
    cos_theta_in = paths.directions[..., 2]
    obliquity = 0.5 * (cos_theta_in + cos_theta)
    # 4.11.2: apply the Kirchhoff prefactor ``1/(iλ)·dΩ`` for each
    # re-emission, matching the convention applied in
    # :func:`init_paths_from_field`.  Pre-4.11.2 the per-aperture
    # re-emission left the weights with only the obliquity factor,
    # so each cascaded aperture under-weighted by ``2π(1-cosθ_max)
    # / (iλ·n_paths)`` -- absolute amplitudes were off by ~10^6 per
    # extra aperture at visible wavelengths.  The relative
    # phase/contrast structure is unaffected because the factor is
    # global per aperture.
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    kirchhoff = complex(inv_i_lambda) * solid_angle
    new_weights = (paths.weights
                   * obliquity.astype(paths.weights.dtype)
                   * kirchhoff)
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
    output_dtype: Optional[Any] = None,
) -> np.ndarray:
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

    # Scatter-add into the output grid.  ``np.add.at`` is the canonical
    # NumPy operation for unbuffered scatter-add; CuPy does not expose
    # the same ``xp.add.at`` interface, so we fall through to
    # ``cupyx.scatter_add`` or a NumPy host round-trip in that case.
    N_flat = Ny * Nx
    out = xp.zeros(N_flat, dtype=output_dtype)
    if hasattr(xp, 'add') and hasattr(xp.add, 'at'):
        xp.add.at(out, flat_idx, w_masked)
    else:
        try:
            import cupyx
            cupyx.scatter_add(out, flat_idx, w_masked)
        except (ImportError, AttributeError, TypeError, ValueError):
            # ImportError: cupyx unavailable in the active env.
            # AttributeError: older cupyx without scatter_add.
            # TypeError/ValueError: dtype or shape disagreement
            # between out / flat_idx / w_masked (e.g. on JAX
            # arrays routed here by mistake).  Fall back to a
            # NumPy scatter round-trip; correctness preserved
            # at the cost of a device->host hop.
            idx_h = to_numpy(flat_idx)
            w_h = to_numpy(w_masked)
            out_h = np.zeros(N_flat, dtype=output_dtype)
            np.add.at(out_h, idx_h, w_h)
            out = xp.asarray(out_h)
    return out.reshape(Ny, Nx)


# ============================================================================
# End-to-end convenience
# ============================================================================

def propagate_hfpi(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    *,
    aperture_radius: float,
    z_aperture_to_output: float,
    n_paths: int,
    **kwargs: Any,
) -> np.ndarray:
    """Canonical-order HFPI three-leg propagation.

    Argument order ``(E_in, z, wavelength, dx)`` matches
    :func:`angular_spectrum_propagate`.  ``z`` here is the source-plane
    -> aperture distance (i.e. the first leg).  Other HFPI-specific
    parameters remain keyword-only.

    Internally delegates to :func:`propagate_hfpi_freespace_aperture`
    (which retains its legacy
    ``(E_in, dx, *, z_to_aperture, ..., wavelength, ...)`` order).

    .. warning::
       **Phase-only stochastic propagator -- absolute amplitudes are
       unnormalized** (audit #3.1).  This implementation samples paths
       uniformly across the source grid and assigns each
       ``exp(j·k·Δs)`` propagation phase, then bins to the output grid.
       The full Fresnel-Kirchhoff integral

           E(P) = (1/jλ) ∫∫ E(Q) · (cos θ / r) · exp(jkr) dS

       carries three normalization factors that this code does **not**
       apply: the ``1/(jλ)`` Kirchhoff prefactor, the per-path
       ``1/r`` geometric-spreading attenuation, and the Monte Carlo
       solid-angle weight ``2π·(1 − cos θ_max) / N_paths``.  The
       returned field has the correct **phase structure** (fringe
       positions, interference contrast, relative-intensity ratios
       within a single experiment) but the absolute amplitude is
       arbitrary.  Use it as a phase-structure / interference
       diagnostic; do not interpret absolute coupling-efficiency
       numbers across experiments without re-normalising against a
       known-amplitude reference.
    """
    return propagate_hfpi_freespace_aperture(
        E_in, dx, z_to_aperture=z, wavelength=wavelength,
        aperture_radius=aperture_radius,
        z_aperture_to_output=z_aperture_to_output,
        n_paths=n_paths, **kwargs)


def propagate_hfpi_freespace_aperture(
    E_in: np.ndarray,
    dx: float,
    *,
    z_to_aperture: float,
    aperture_radius: float,
    z_aperture_to_output: float,
    wavelength: float,
    n_paths: int,
    rng: Optional[Union[int, object]] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    aperture_shape: str = 'circular',
    aperture_centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """End-to-end three-leg HFPI: source plane -> free space ->
    aperture -> free space -> output plane.

    The canonical single-aperture-diffraction validation case.

    .. note::
       This function uses a non-canonical argument order
       ``(E_in, dx, *, z_to_aperture, ..., wavelength, ...)``.  Prefer
       :func:`propagate_hfpi` for the canonical
       ``(E_in, z, wavelength, dx)`` order.
    """
    # 4.11.2: spawn a distinct child RNG for the aperture re-emission
    # so the source-plane init draws and the aperture re-sample draws
    # are statistically independent.  Pre-4.11.2 the same int ``rng``
    # was reused; ``RandomState(rng=int)`` rebuilds default_rng(int)
    # so both draws were identical (perfectly correlated init / re-
    # emission).
    rng_source = _spawn_rng(rng, 0)
    rng_aperture = _spawn_rng(rng, 1)
    paths = init_paths_from_field(
        E_in, dx,
        n_paths=n_paths,
        wavelength=wavelength,
        rng=rng_source,
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
        rng=rng_aperture,
    )
    paths = propagate_to_plane(paths,
                                z_target=z_to_aperture + z_aperture_to_output,
                                wavelength=wavelength)

    # v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): see module docstring;
    # ``output_grid`` is now the legacy spelling of ``output_shape``.
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_hfpi_freespace_aperture',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx
    return accumulate_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=output_dx, centre=output_centre,
        output_dtype=E_in.dtype,
    )


# ============================================================================
# Variance reduction: stratified sampling
# ============================================================================

def init_paths_stratified(
    E_in: np.ndarray,
    dx: float,
    *,
    n_paths: int,
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    z_input_plane: float = 0.0,
    n_strata_xy: Optional[Tuple[int, int]] = None,
    n_strata_dir: Optional[Tuple[int, int]] = None,
) -> PathBundle:
    """Stratified-sampling variant of :func:`init_paths_from_field`.

    Partitions the source-pixel index space and the forward-cone
    direction sphere into equal-area strata, then samples one path
    per stratum.  This reduces the variance of the resulting Monte
    Carlo HFPI estimate by ensuring uniform coverage of phase space
    -- the integrated complex weight has lower variance than naive
    uniform sampling for the same path count.

    Variance reduction factor depends on the integrand smoothness;
    typically 2-10x for smooth-amplitude / smooth-OPL systems.

    Parameters
    ----------
    n_strata_xy : (n_iy, n_ix), optional
        Per-axis number of strata for the source-pixel index.
        Default ``(sqrt(n_paths), sqrt(n_paths))``.
    n_strata_dir : (n_theta, n_phi), optional
        Per-axis number of strata for the direction sphere.
        Default same as n_strata_xy.

    All other parameters mirror :func:`init_paths_from_field`.
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    rs = RandomState(rng=rng if rng is not None else 0)

    # Default: square stratification.  4-D stratification has
    # n_iy * n_ix * n_th * n_ph cells, so scale per-axis to keep
    # the total close to n_paths (use the 4th root).
    if n_strata_xy is None and n_strata_dir is None:
        side = max(2, int(round(n_paths ** 0.25)))
        n_strata_xy = (side, side)
        n_strata_dir = (side, side)
    elif n_strata_xy is None:
        n_strata_xy = n_strata_dir
    elif n_strata_dir is None:
        n_strata_dir = n_strata_xy
    n_iy, n_ix = n_strata_xy
    n_th, n_ph = n_strata_dir

    n_total = n_iy * n_ix * n_th * n_ph
    # If user requested fewer than n_total, sample only n_paths
    # strata uniformly without replacement.  If they requested more,
    # sample multiple paths per stratum (jittered).
    n_per = max(1, n_paths // n_total)
    n_paths_actual = n_per * n_total

    # Build stratum index grid.  4.11.2: enumerate the full 4-D
    # cartesian product of stratum indices.  Pre-4.11.2 the
    # ``np.repeat`` pattern broadcast a single-axis vector instead of
    # iterating: ``np.repeat(np.arange(n_iy)[:, None, None, None],
    # n_ix*n_th*n_ph*n_per, axis=0)`` produces ``[0]*n_block ++
    # [1]*n_block ++ ...`` along axis 0 only, and the other strata
    # vectors followed the same pattern on their own axes.  Calling
    # ``.reshape(-1)`` then flattened to ``[0,0,...,1,1,...]`` for
    # every axis, so the paired index quadruple ``(iy[k], ix[k],
    # th[k], ph[k])`` only ever took values ``(0,0,0,0)`` then
    # ``(1,1,1,1)`` -- 2 distinct cells out of n_iy*n_ix*n_th*n_ph.
    # ``np.indices`` builds the true cartesian-product mesh.
    idx_grid = np.indices((n_iy, n_ix, n_th, n_ph)).reshape(4, -1)
    iy_strata = np.repeat(idx_grid[0], n_per)
    ix_strata = np.repeat(idx_grid[1], n_per)
    th_strata = np.repeat(idx_grid[2], n_per)
    ph_strata = np.repeat(idx_grid[3], n_per)

    # Truncate to n_paths_actual if needed.
    iy_strata = iy_strata[:n_paths_actual]
    ix_strata = ix_strata[:n_paths_actual]
    th_strata = th_strata[:n_paths_actual]
    ph_strata = ph_strata[:n_paths_actual]

    # Jitter within each stratum: uniform random offset on [0, 1)
    # times the stratum width.
    u_iy = rs.uniform((n_paths_actual,))
    u_ix = rs.uniform((n_paths_actual,))
    u_th = rs.uniform((n_paths_actual,))
    u_ph = rs.uniform((n_paths_actual,))

    iy_jit = (iy_strata + np.asarray(u_iy)) * (Ny / n_iy)
    ix_jit = (ix_strata + np.asarray(u_ix)) * (Nx / n_ix)
    iy_int = np.clip(iy_jit.astype(np.int64), 0, Ny - 1)
    ix_int = np.clip(ix_jit.astype(np.int64), 0, Nx - 1)

    # Direction stratification: uniform-on-cone in (cos_theta, phi).
    cos_max = float(np.cos(cone_half_angle))
    # cos_theta uniform on [cos_max, 1] partitioned into n_th strata.
    cth_strata_low = cos_max + (1.0 - cos_max) * (th_strata / n_th)
    cth_strata_width = (1.0 - cos_max) / n_th
    cos_theta_h = cth_strata_low + np.asarray(u_th) * cth_strata_width
    # phi uniform on [0, 2 pi].
    phi_h = 2 * float(np.pi) * (ph_strata + np.asarray(u_ph)) / n_ph

    iy_xp = xp.asarray(iy_int)
    ix_xp = xp.asarray(ix_int)
    x_s = (ix_xp - Nx / 2) * dx
    y_s = (iy_xp - Ny / 2) * dx
    z_s = xp.full((n_paths_actual,), float(z_input_plane), dtype=x_s.dtype)
    positions = xp.stack([x_s, y_s, z_s], axis=-1)

    cos_theta = xp.asarray(cos_theta_h)
    phi = xp.asarray(phi_h)
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta ** 2, 0.0))
    L = sin_theta * xp.cos(phi)
    M = sin_theta * xp.sin(phi)
    N = cos_theta
    directions = xp.stack([L, M, N], axis=-1)

    sample = E_in[iy_xp, ix_xp]
    # 4.10: HF Kirchhoff weighting (see init_paths_from_field).  Use
    # n_paths_actual for the solid-angle normalisation in the
    # stratified variant.
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n_paths_actual)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    weights = (sample * cos_theta * (dx * dx)
               * complex(inv_i_lambda) * solid_angle)
    opl = xp.zeros((n_paths_actual,), dtype=xp.real(sample).dtype)
    alive = xp.ones((n_paths_actual,), dtype=bool)

    return PathBundle(
        positions=positions,
        directions=directions,
        weights=weights,
        opl=opl,
        alive=alive,
    )


# ============================================================================
# Prescription-aware HFPI -- walks a sequential prescription
# ============================================================================

def propagate_hfpi_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    n_paths: int,
    rng: Optional[Union[int, object]] = None,
    diffracting_surfaces: Optional[List[int]] = None,
    surface_diffraction: Optional[Dict[int, Any]] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    sampling: str = 'stratified',
    cone_half_angle: float = np.pi / 2 - 1e-6,
) -> np.ndarray:
    """End-to-end HFPI through a sequential lumenairy prescription.

    Source plane -> first surface -> ... refraction / diffraction at each
    surface ... -> last surface -> output plane.  Uses the library's
    geometric :func:`lumenairy.raytrace.trace` for the per-surface
    refraction + OPL accumulation, and re-emits secondary HF sources at
    every surface listed in ``diffracting_surfaces``.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
        lumenairy prescription dict with surfaces, thicknesses, glasses.
    wavelength : float
        Vacuum wavelength (m).
    n_paths : int
        Number of independent paths.
    rng : int | Generator | PRNGKey | None
        Random source.
    diffracting_surfaces : list of int, optional
        Zero-based surface indices where HFPI re-samples directions
        (treated as hard-aperture diffractors).  Defaults to every
        surface that has a finite ``semi_diameter``, i.e. apertures
        and stops.
    surface_diffraction : dict, optional
        Per-surface DOE order kicks (forwarded to ``trace``).  Maps
        surface index to ``(order_x, order_y, period_x, period_y)``.
    output_grid : (Ny, Nx) | None
        Output grid shape.
    output_dx : float | None
        Output grid pitch.
    output_centre : (float, float)
        Output grid centre coordinates.
    sampling : str
        ``'uniform'`` or ``'stratified'`` (default).  Stratified
        partitions the source-direction cone into equal-solid-angle
        cells and forces one path per cell, reducing variance.
    cone_half_angle : float
        Half-angle of the forward emission cone for path
        initialisation and per-surface re-sampling.

    Returns
    -------
    array (Ny, Nx) complex
        Output-plane complex field.

    Notes
    -----
    Each surviving path's weight gains ``exp(i k OPL)`` along its
    journey through the prescription.  At each diffracting surface
    a fresh forward-cone direction is sampled and the OPL accumulator
    is reset (the new secondary source's accumulated phase is folded
    into the path's complex weight).
    """
    from ..raytrace import (
        surfaces_from_prescription,
    )

    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    # v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): see module helper;
    # ``output_grid`` is now the deprecated spelling of ``output_shape``.
    Ny_out, Nx_out = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_hfpi_through_prescription',
        default_shape=(Ny, Nx),
    )
    if output_dx is None:
        output_dx = dx

    # 2.  Resolve surface list and identify diffractors.
    surfaces = surfaces_from_prescription(prescription)
    object_distance = float(prescription.get('object_distance', 0.0))

    # 1.  Initialise paths at the source plane.  4.11.2: paths are
    # initialised AT z = -object_distance and travel forward through
    # the system.  Pre-4.11.2 initialised at z=0 and then called
    # ``propagate_to_plane(z_target=-object_distance)`` -- because the
    # paths have +z-going directions, the implied geometric step is
    # negative, ``t < 0``, and ``propagate_to_plane``'s ``t >= 0``
    # alive mask killed every path on entry.  Finite-conjugate HFPI
    # therefore returned an all-zero output.
    #
    # 4.11.2: spawn a per-aperture child RNG so cascaded diffractors
    # draw statistically independent samples.  Pre-4.11.2 the same
    # caller seed was passed to every ``apply_aperture_diffraction``
    # call; ``RandomState(rng=int_seed)`` rebuilds the same NumPy
    # generator each time, so every aperture in the cascade drew
    # identical uniform sequences (perfectly correlated diffraction
    # events).  We allocate stream 0 to the source init and the
    # remaining streams sequentially to each diffractor.
    rng_source = _spawn_rng(rng, 0)
    paths = init_paths_from_field(
        E_in, dx,
        n_paths=n_paths,
        wavelength=wavelength,
        rng=rng_source,
        cone_half_angle=cone_half_angle,
        z_input_plane=-object_distance,
    )

    if diffracting_surfaces is None:
        diffracting_surfaces = []
        for i, s in enumerate(surfaces):
            sd = getattr(s, 'semi_diameter', None)
            if sd is not None and sd > 0 and sd < float('inf'):
                diffracting_surfaces.append(i)
    diffracting_surfaces = set(diffracting_surfaces)

    # 3.  Walk forward through each surface.  Group surfaces between
    # diffractors into segments; trace each segment with the existing
    # trace() machinery, then re-sample at the diffractor.
    # Hand off to trace for the whole stack at once if there are no
    # interior diffractors; otherwise per-segment.
    surf_indices = list(range(len(surfaces)))
    if not diffracting_surfaces:
        # Single trace through the full stack.
        paths = _hfpi_segment_trace(
            paths, surfaces, wavelength,
            surface_diffraction=surface_diffraction,
        )
    else:
        # Per-segment trace, with HFPI resampling at each diffractor.
        sorted_diff = sorted(diffracting_surfaces)
        cursor = 0
        # Stream counter for spawn_rng:  index 0 was used for source
        # init, so per-aperture child seeds start at 1.
        rng_stream = 1
        for diff_idx in sorted_diff:
            if diff_idx > cursor:
                # Trace surfaces [cursor, diff_idx-1]
                segment = surfaces[cursor:diff_idx]
                paths = _hfpi_segment_trace(
                    paths, segment, wavelength,
                    surface_diffraction={
                        i - cursor: v
                        for i, v in (surface_diffraction or {}).items()
                        if cursor <= i < diff_idx
                    } if surface_diffraction else None,
                )
            # Trace through the diffractor surface itself (refract /
            # intersect), then re-sample directions HFPI-style.
            single = [surfaces[diff_idx]]
            paths = _hfpi_segment_trace(
                paths, single, wavelength,
                surface_diffraction={
                    0: surface_diffraction[diff_idx]
                } if surface_diffraction and diff_idx in surface_diffraction else None,
            )
            # Apply HFPI hard-aperture mask + resample directions.
            sd = getattr(surfaces[diff_idx], 'semi_diameter', None)
            if sd is not None and sd > 0 and sd < float('inf'):
                rng_aperture = _spawn_rng(rng, rng_stream)
                rng_stream += 1
                paths = apply_aperture_diffraction(
                    paths, aperture_radius=float(sd),
                    rng=rng_aperture, wavelength=wavelength,
                    cone_half_angle=cone_half_angle,
                )
            cursor = diff_idx + 1
        # Trace the trailing tail (if any).
        if cursor < len(surfaces):
            segment = surfaces[cursor:]
            paths = _hfpi_segment_trace(
                paths, segment, wavelength,
                surface_diffraction={
                    i - cursor: v
                    for i, v in (surface_diffraction or {}).items()
                    if i >= cursor
                } if surface_diffraction else None,
            )

    # 4.  Accumulate to output grid.
    return accumulate_to_grid(
        paths,
        Ny=Ny_out, Nx=Nx_out,
        dx=output_dx, centre=output_centre,
        output_dtype=E_in.dtype,
    )


def _hfpi_segment_trace(paths: PathBundle,
                        segment_surfaces,
                        wavelength: float,
                        *,
                        surface_diffraction=None) -> PathBundle:
    """Trace a PathBundle through a sub-list of surfaces using the
    existing :func:`lumenairy.raytrace.trace` for geometry + OPL,
    propagate the OPL into the complex weights, and return a new
    PathBundle.

    Internal helper for :func:`propagate_hfpi_through_prescription`.
    """
    if not segment_surfaces:
        return paths

    from ..raytrace import RayBundle, trace
    xp = array_namespace(paths.positions)

    # Build a RayBundle from the PathBundle's geometric state.
    n = int(paths.positions.shape[0])
    pos_h = np.asarray(paths.positions if not is_jax_array(paths.positions)
                       else to_numpy(paths.positions))
    dir_h = np.asarray(paths.directions if not is_jax_array(paths.directions)
                       else to_numpy(paths.directions))
    opl_in = np.asarray(paths.opl if not is_jax_array(paths.opl)
                        else to_numpy(paths.opl))
    alive_in = np.asarray(paths.alive if not is_jax_array(paths.alive)
                          else to_numpy(paths.alive))

    rb = RayBundle(
        x=pos_h[:, 0].copy(),
        y=pos_h[:, 1].copy(),
        z=pos_h[:, 2].copy(),
        L=dir_h[:, 0].copy(),
        M=dir_h[:, 1].copy(),
        N=dir_h[:, 2].copy(),
        wavelength=wavelength,
        opd=opl_in.copy(),
        alive=alive_in.copy(),
    )

    result = trace(rb, list(segment_surfaces), wavelength,
                    output_filter='last',
                    surface_diffraction=surface_diffraction)
    out_rb = result.image_rays

    # New geometric state and OPL delta.
    new_positions = xp.stack([
        xp.asarray(out_rb.x), xp.asarray(out_rb.y), xp.asarray(out_rb.z)
    ], axis=-1)
    new_directions = xp.stack([
        xp.asarray(out_rb.L), xp.asarray(out_rb.M), xp.asarray(out_rb.N)
    ], axis=-1)
    delta_opl = xp.asarray(out_rb.opd) - xp.asarray(opl_in)
    new_alive = xp.asarray(out_rb.alive) & paths.alive
    k = 2 * float(np.pi) / wavelength
    phase = xp.exp(1j * k * delta_opl).astype(paths.weights.dtype)
    new_weights = paths.weights * phase

    return PathBundle(
        positions=new_positions,
        directions=new_directions,
        weights=new_weights,
        opl=xp.asarray(out_rb.opd),
        alive=new_alive,
    )


__all__ = [
    'PathBundle',
    'init_paths_from_field',
    'init_paths_stratified',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi_freespace_aperture',
    'propagate_hfpi_through_prescription',
]
