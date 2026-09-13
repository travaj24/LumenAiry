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

The free-space pipeline is written against
:func:`lumenairy._array.array_namespace`, accepting NumPy / CuPy /
JAX source fields and returning the same backend.  Random sampling
goes through :class:`lumenairy._random.RandomState`.

The PRESCRIPTION walk (:func:`propagate_hfpi_through_prescription`) is
different: it calls the host ray tracer, so every segment round-trips
the bundle to NumPy via :func:`~lumenairy.backend.to_numpy` and back.
Results are correct on every backend, but the walk cannot be traced
under ``jax.jit`` / ``vmap`` / ``grad`` and a CuPy bundle loses device
residency each segment (K23, audit 2026-09-11).

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

    :class:`lumenairy.backend.random.RandomState` rebuilds a fresh NumPy
    ``default_rng(rng)`` on every construction when ``rng`` is an integer
    (or None), so handing the SAME int seed to every aperture would draw
    the same uniform samples at each of them -- perfectly correlated
    diffraction events across a cascaded stack.  A distinct child seed per
    stream index is derived by hashing the parent seed with the stream
    counter.
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
        # Derive the child from the parent's SeedSequence ENTROPY, exactly
        # as the ``int`` branch above does, so the mapping is a PURE
        # function of ``(parent, stream_index)``: it must not mutate the
        # caller's generator, and ``stream_index`` must be a stable key
        # whatever order the streams are drawn in (audit K24).
        try:
            parent_entropy = rng.bit_generator.seed_seq.entropy
        except AttributeError:
            parent_entropy = None
        if parent_entropy is not None:
            ent = (list(parent_entropy)
                   if isinstance(parent_entropy, (list, tuple))
                   else [int(parent_entropy)])
            return np.random.default_rng(
                np.random.SeedSequence(entropy=ent + [int(stream_index)]))
        # No readable seed sequence (a hand-built BitGenerator).  Fall
        # back to drawing a child seed; this DOES advance the parent, but
        # there is no entropy to key on.
        child_seed = int(rng.integers(0, 2 ** 63 - 1))
        return np.random.default_rng(child_seed)
    # JAX PRNGKey: caller-side splitting is the canonical pattern,
    # but we can deterministically fold the stream index.
    #
    # The except clause is deliberately NARROW (audit K15/K24): a blanket
    # ``except Exception: pass`` falls through to "return as-is" and hands
    # BOTH streams the IDENTICAL key -- the exact correlation this
    # function exists to prevent -- with no diagnostic whatever.  The
    # fall-through says what it did.
    try:
        import jax
        if hasattr(jax.random, 'fold_in'):
            return jax.random.fold_in(rng, stream_index)
    except (ImportError, AttributeError, TypeError, ValueError) as _exc:
        warnings.warn(
            f"_spawn_rng: could not fold stream index {stream_index} into "
            f"the supplied JAX key ({type(_exc).__name__}: {_exc}); the "
            f"key is returned UNCHANGED, so every stream derived from it "
            f"draws the SAME samples -- the correlation this function "
            f"exists to prevent.  Split the key caller-side "
            f"(jax.random.split) and pass one sub-key per aperture.",
            RuntimeWarning, stacklevel=2)
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
    # K13 (audit 2026-09-11): GEOMETRIC distance travelled since the last
    # emission / re-emission, in metres.  ``opl`` cannot serve: it is the
    # OPTICAL path (``n_medium * |t|``) on the free-space legs and the
    # ABSOLUTE accumulated ``opd`` from the ray tracer on the prescription
    # walk, whereas the Huygens-Fresnel ``1/r`` spreading needs the
    # geometric length of the CURRENT leg.  ``None`` means "not tracked"
    # and the binning Jacobian falls back to ``opl`` (the same number
    # whenever n_medium == 1 and the bundle came from a free-space entry
    # point).  Trailing default, so positional construction of the
    # pre-v5.46 five-field bundle keeps working.
    leg: object = None      # (N,) float array

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

    def leg_or_opl(self):
        """Current-leg geometric length, falling back to ``opl``."""
        return self.opl if self.leg is None else self.leg


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
    # ``rng=None`` -- the DEFAULT on every HFPI entry point -- must draw
    # fresh system entropy, which is exactly what ``RandomState(None)``
    # does (``np.random.default_rng(None)``).  HFPI is a 1/sqrt(N)
    # Monte-Carlo estimator: with a FIXED default seed, the canonical way
    # to see the estimator's own error -- re-run and compare -- returns
    # identically ZERO (audit K19).  Pass an int (or a Generator) for
    # reproducibility.
    rs = RandomState(rng=rng)

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
    # The SOURCE-AREA factor (audit K18).  The source pixel is
    # drawn uniformly over Ny*Nx pixels, so the unbiased estimate of
    # ``int dS int dOmega f`` is ``(Area * Omega / n_paths) * sum f`` with
    # ``Area = Ny*Nx*dx**2`` -- the whole illuminated area, not ONE
    # pixel's ``dx**2``.  Pre-fix the weight carried ``dx**2`` alone, so
    # every amplitude was low by the source pixel COUNT.  Measured
    # (single unit-amplitude source pixel, cone 0.20 rad, 200 000 paths,
    # against the exact ``int E cos(theta) dx^2/(i lambda) dOmega``):
    # sum(weights)/exact = 1.000004 (1x1), 0.062140 (4x4), 0.003945
    # (16x16), 0.000260 (64x64) -- tracking 1/N_pix to MC noise, i.e.
    # 4096x low at 64x64.
    n_src_px = float(Ny) * float(Nx)
    solid_angle = (2.0 * float(np.pi) * (1.0 - cos_max)
                   * n_src_px / float(n_paths))
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
        leg=xp.zeros_like(opl),
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
    # K13: the GEOMETRIC step is |t| (directions are unit vectors), which
    # is ``delta_opl / n_medium``; track it separately from the optical
    # path so the binning Jacobian has a true ``r``.
    base_leg = paths.leg if paths.leg is not None else xp.zeros_like(paths.opl)
    new_leg = base_leg + xp.abs(t)

    return PathBundle(
        positions=new_positions,
        directions=paths.directions,
        weights=new_weights,
        opl=new_opl,
        alive=new_alive,
        leg=new_leg,
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
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    normalisation: str = 'physical',
) -> PathBundle:
    """Apply a hard aperture at the current path-bundle plane.

    Paths landing outside the aperture are killed.  Surviving paths
    re-emit secondary HF sources at their current position with a
    fresh direction sample.  OPL is reset since the new secondary
    source's accumulator starts at zero.

    Parameters
    ----------
    wavelength : float, keyword-only, REQUIRED
        Vacuum wavelength [m], strictly positive.

        .. versionchanged:: 5.46
            No longer defaults to ``0.0`` (audit K12).  The
            ``1/(i*lambda)`` Kirchhoff prefactor is gated on
            ``wavelength > 0``, so a zero default silently DROPPED it --
            every path weight wrong by a factor ``1/lambda`` in magnitude
            and by -90 degrees in phase, with no warning at all.  A
            physically meaningless default must not silently mean "skip
            the physics".  Migration: pass ``wavelength=`` (the library's
            own entry points always did).
    normalisation : {'physical', 'legacy'}, default 'physical'
        Which re-emission measure to apply; see
        :func:`_reemission_measure`.  It MUST match the value passed to
        :func:`accumulate_to_grid` at the end of the chain -- the two
        halves are one estimator.  The library's own entry points thread
        a single value to both.

        .. versionchanged:: 5.46.1
            Added (audit K13 / verify V1).  ``'physical'`` applies the
            exact intermediate-leg measure
            ``(1/(i lambda)) * Omega_out * r_in * cos_out / cos_in``;
            ``'legacy'`` keeps the pre-v5.46 factor
            ``0.5(cos_in + cos_out) * (1/(i lambda)) * Omega_out /
            n_paths``, which made every cascaded amplitude low by
            ``n_paths * r_in``.
    """
    if normalisation not in ('physical', 'legacy'):
        raise ValueError(
            f"apply_aperture_diffraction: normalisation must be "
            f"'physical' or 'legacy'; got {normalisation!r}.")
    if not (wavelength > 0) or not np.isfinite(wavelength):
        raise ValueError(
            f"apply_aperture_diffraction: wavelength must be a positive "
            f"finite length in metres (got {wavelength!r}).  It scales the "
            f"1/(i*lambda) Kirchhoff prefactor applied to every re-emitted "
            f"path; without it the returned weights are wrong by 1/lambda "
            f"in magnitude and by -90 degrees in phase.")
    xp = array_namespace(paths.positions)
    # ``rng=None`` -- the DEFAULT on every HFPI entry point -- must draw
    # fresh system entropy, which is exactly what ``RandomState(None)``
    # does (``np.random.default_rng(None)``).  HFPI is a 1/sqrt(N)
    # Monte-Carlo estimator: with a FIXED default seed, the canonical way
    # to see the estimator's own error -- re-run and compare -- returns
    # identically ZERO (audit K19).  Pass an int (or a Generator) for
    # reproducibility.
    rs = RandomState(rng=rng)

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

    # Apply the Kirchhoff prefactor for each re-emission, matching the
    # convention applied in :func:`init_paths_from_field`.  The exact
    # intermediate-leg MEASURE -- and why it carries ``r`` rather than
    # ``1/r`` -- is derived in :func:`_reemission_measure`.
    new_weights = paths.weights * _reemission_measure(
        paths, cos_theta, cos_max, wavelength, normalisation,
        'apply_aperture_diffraction')
    new_opl = xp.zeros_like(paths.opl)

    return PathBundle(
        positions=paths.positions,
        directions=new_directions,
        weights=new_weights,
        opl=new_opl,
        alive=survives,
        # K13: a re-emission starts a new leg.
        leg=xp.zeros_like(paths.opl),
    )



def _reemission_measure(paths, cos_theta_out, cos_max, wavelength,
                        normalisation, fn_name):
    """Per-path factor for a Huygens re-emission (audit K13 / verify V1).

    A cascaded HFPI walk is an exact composition of Rayleigh-Sommerfeld-I
    integrals.  Write the intermediate surface integral in DIRECTION
    variables -- the variables the estimator actually samples.  For the
    leg from ``Q_m`` to ``Q_{m+1}``,

        dOmega_m = dS_{m+1} cos(theta_m) / r_m^2
        =>  dS_{m+1} (cos(theta_m)/r_m) e^{ik r_m}
              = dOmega_m * r_m * e^{ik r_m} ,

    so in direction variables the Kirchhoff kernel of an INTERMEDIATE leg
    is ``(1/(i lambda)) e^{ik r} * r * dOmega`` -- a factor ``r``, not
    ``1/r``, and NO obliquity.  The estimator therefore has to carry,
    after the ``m``-th leg,

        W_m = E(Q_1) (1/(i lambda))^m (A_src Omega_1 / n_paths)
              Omega_2 ... Omega_m  r_1 ... r_{m-1}  e^{ik sum r}
              * cos(theta_m)

    -- the trailing ``cos(theta_m)`` being exactly what
    :func:`_binning_jacobian`'s ``r/(dx_out^2 cos(theta_out))`` cancels
    when that leg turns out to be the last one.  Propagating that
    invariant from leg ``m`` to leg ``m+1`` gives the factor this
    function returns:

        F = (1/(i lambda)) * Omega_out * r_in * cos(theta_out)
                                              / cos(theta_in)

    where ``r_in`` is the GEOMETRIC length of the leg that just ended
    (``paths.leg``), ``cos(theta_in)`` the incoming direction's
    z-component and ``cos(theta_out)`` the freshly drawn one.  The
    ``1/cos(theta_in)`` is bookkeeping, not physics: it removes the
    ``cos(theta)`` :func:`init_paths_from_field` applied on the
    then-unknown assumption that the first leg would be the final one.
    The ``cos(theta_out)`` is the genuine RS-I obliquity of the new
    surface.  The formula composes, so it is correct for any number of
    apertures.

    ``normalisation='legacy'`` returns the pre-v5.46 factor
    ``0.5*(cos_in + cos_out) * (1/(i lambda)) * Omega_out / n_paths``
    instead.  That factor is wrong twice for a chain.  (a) The
    ``/ n_paths`` divides by the sample count a SECOND time: a path is
    ONE sample of the joint (source pixel, direction_1, ...,
    direction_m) integral, so the ``1/n_paths`` belongs once and already
    lives in ``init_paths_from_field`` alongside ``A_src * Omega_1``.
    (b) The intermediate leg carries no ``r``.  Together those make a
    cascaded amplitude low by exactly ``n_paths * r_in``.  The symmetric
    Kirchhoff obliquity ``0.5(cos_in + cos_out)`` is a heuristic; the
    exact RS-I composition wants the outgoing cosine alone, which is what
    this returns.  The measurement that establishes the composition is in
    ``docs/history/lumenairy.propagators.hfpi.md``.

    Parameters
    ----------
    paths : PathBundle or VectorPathBundle
        The bundle arriving at the surface; ``leg`` and ``directions``
        are read.
    cos_theta_out : array
        z-component of the freshly drawn emission directions.
    cos_max : float
        ``cos(cone_half_angle)`` of the emission cone.
    wavelength : float
        Vacuum wavelength [m], strictly positive (checked by the caller).
    normalisation : {'physical', 'legacy'}
        ``'legacy'`` returns the pre-v5.46 factor unchanged.
    fn_name : str
        For the error message (CONVENTIONS section 2).

    Returns
    -------
    factor : array
        Complex per-path multiplier.
    """
    xp = array_namespace(paths.positions)
    n = int(paths.positions.shape[0])
    cos_theta_in = paths.directions[..., 2]
    omega = 2.0 * float(np.pi) * (1.0 - float(cos_max))
    inv_i_lambda = complex(1.0 / (1j * float(wavelength)))

    if normalisation == 'legacy':
        # The pre-v5.46 factor: symmetric Kirchhoff obliquity, and the
        # cone solid angle divided by the sample count.
        obliquity = 0.5 * (cos_theta_in + cos_theta_out)
        return obliquity.astype(_paths_weight_dtype(paths)) * (
            inv_i_lambda * omega / float(n))

    r_in = getattr(paths, 'leg', None)
    if r_in is None:
        r_in = paths.opl
    ok = paths.alive & (r_in > 0) & (cos_theta_in > 1e-12)
    if not is_jax_array(paths.positions):
        if (int(np.count_nonzero(to_numpy(paths.alive))) > 0
                and int(np.count_nonzero(to_numpy(ok))) == 0):
            raise ValueError(
                f"{fn_name}: normalisation='physical' needs the geometric "
                f"length of the leg that ended at this surface (the "
                f"intermediate-leg Jacobian of the Huygens-Fresnel "
                f"composition), and every surviving path has travelled "
                f"zero distance -- the aperture sits on the plane the "
                f"paths were emitted from.  An aperture there is a mask "
                f"on the source field, not a Huygens re-emission: apply "
                f"it to the field before calling "
                f"init_paths_from_field, propagate to a non-zero "
                f"distance first, or pass normalisation='legacy' for the "
                f"raw (non-photometric) path sum.")
    safe_cos_in = xp.where(ok, cos_theta_in, 1.0)
    factor = xp.where(ok, r_in * cos_theta_out / safe_cos_in, 0.0)
    return factor.astype(_paths_weight_dtype(paths)) * (inv_i_lambda * omega)


def _paths_weight_dtype(paths):
    """Complex dtype a bundle's amplitudes are carried in."""
    w = getattr(paths, 'weights', None)
    if w is None:
        w = paths.Ex
    return w.dtype


# ============================================================================
# Coherent accumulation at the output plane
# ============================================================================

# v5.31 (audit W9-14): a Monte-Carlo estimator that never puts a path in a
# pixel has not estimated that pixel.  Below ONE landed path per output pixel
# the returned array is not a field at all -- it is the sampling envelope, and
# it is seed-noise on top.  MEASURED (thin air plate, 2 mm, 128^2 output, the
# DEFAULT ~90-degree cone): occupancy -- the fraction of output pixels that ever
# receive a path -- was 0.6% / 2.1% / 7.9% at n_paths = 20k / 80k / 320k, and the
# seed-to-seed intensity-shape fidelity was 0.005 / 0.005 / 0.021, i.e. two runs
# of the same physics agreed with each other not at all.
#
# One path per pixel is a NECESSARY, nowhere near sufficient, condition: at ~12
# landed paths per pixel the same probe still only reached seed-to-seed fidelity
# 0.44.  The bar is set where the failure is UNAMBIGUOUS rather than where the
# result becomes trustworthy, so the guard cannot cry wolf.
_MIN_LANDED_PATHS_PER_OUTPUT_PIXEL = 1.0


def _check_landed(inside, Ny, Nx, policy, fn_name, positions=None):
    """The v5.31 sampling-adequacy guard, shared by BOTH accumulators.

    K23 (audit 2026-09-11): v4.13.1 re-implemented the vector
    accumulator inline "bit-identically to the twice-routed version",
    and the guard v5.31 added to the scalar one therefore never ran on
    the vector path -- identical geometry, scalar warns, vector silent
    (measured: 20 000 paths onto a 64x64 grid gave 2 non-zero pixels,
    0.05 %, with zero warnings).  One helper, called from both.

    Parameters
    ----------
    inside : bool array
        Per-path "landed on the grid AND alive" mask.
    Ny, Nx : int
        Output grid shape.
    policy : {'warn', 'silent', 'error'}
    fn_name : str
        Caller name for the message prefix (CONVENTIONS section 2).
    positions : array, optional
        Used only to skip the check under JAX tracing, where the path
        count is not readable.  Defaults to ``inside``.
    """
    if policy not in ('warn', 'silent', 'error'):
        raise ValueError(
            f"{fn_name}: on_undersampled must be 'warn', 'silent' or "
            f"'error' (got {policy!r})")
    if policy == 'silent':
        return
    probe = inside if positions is None else positions
    if is_jax_array(probe):
        return
    _n_total = int(np.asarray(inside).size)
    _n_landed = int(np.count_nonzero(to_numpy(inside)))
    _per_px = _n_landed / float(max(Ny * Nx, 1))
    if _per_px >= _MIN_LANDED_PATHS_PER_OUTPUT_PIXEL:
        return
    _msg = (
        f"HFPI is UNDER-SAMPLED for this output grid: of {_n_total} "
        f"paths only {_n_landed} landed on the {Ny}x{Nx} grid "
        f"({100.0 * _n_landed / max(_n_total, 1):.2f}%), i.e. "
        f"{_per_px:.3g} landed paths per output pixel.  Below "
        f"{_MIN_LANDED_PATHS_PER_OUTPUT_PIXEL:g} per pixel most pixels "
        f"are EXACTLY ZERO and the returned array is the Monte-Carlo "
        f"sampling envelope plus shot noise, not a propagated field -- "
        f"measured on a 128^2 probe at the default cone, two seeds of "
        f"the same physics agreed to an intensity-shape fidelity of "
        f"0.005.  The docstring's guarantee that 'fringe positions and "
        f"interference contrast are correct' does NOT hold here.  Two "
        f"levers: raise n_paths, and -- usually far more effective -- "
        f"narrow ``cone_half_angle`` from its ~90-degree default "
        f"(a full forward hemisphere) toward the angle the output grid "
        f"actually subtends, so paths are not spent where they cannot "
        f"land.  NOTE one path per pixel is necessary, not sufficient: "
        f"the same probe still only reached seed-to-seed fidelity 0.44 "
        f"at ~12 paths per pixel.  Pass on_undersampled='silent' to "
        f"acknowledge."
    )
    if policy == 'error':
        raise ValueError(f'{fn_name}: ' + _msg)
    warnings.warn(f'{fn_name}: ' + _msg, RuntimeWarning, stacklevel=3)


def _bin_paths(positions, alive, Ny, Nx, dx, centre):
    """Map path landing points to flat output-pixel indices.

    Returns ``(flat_idx, inside)``.  Cell-centred binning
    ``floor(x/dx + N/2 + 0.5)`` -- the exact inverse of the library's
    pixel-centred grid ``x_i = (i - N/2)*dx``, so pixel *i* collects
    ``[x_i - dx/2, x_i + dx/2)``.  Shared by both accumulators (K24).
    """
    xp = array_namespace(positions)
    cx, cy = centre
    x = positions[..., 0] - cx
    y = positions[..., 1] - cy
    ix = xp.floor(x / dx + Nx / 2 + 0.5).astype(xp.int64)
    iy = xp.floor(y / dx + Ny / 2 + 0.5).astype(xp.int64)
    inside = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny) & alive
    flat_idx = xp.where(inside, iy * Nx + ix, 0)
    return flat_idx, inside


def _binning_jacobian(paths, dx_out, inside):
    """Per-path correction turning the raw HFPI sum into the
    Huygens-Fresnel integral (audit K13).

    The estimator's exact bias law, derived and confirmed to MC noise
    (24 M paths, occupancy 1.000, Gaussian w0 = 12 um on a 64x64 grid at
    dx = 2 um):

        E[HFPI(P)] / E_true(P) = dx_out**2 * cos(theta_out)
                                 / (N_src_px * r)

    measured/predicted = 1.0068 at z = 2 mm and 0.9950 at z = 4 mm, with
    the 2 mm / 4 mm ratio 2.024 -- the 1/r signature (1.00 would mean no
    bias).  ``N_src_px`` is now carried by the source-area weight (K18),
    so what remains here is ``r / (dx_out**2 * cos(theta_out))``.

    ``r`` is ``paths.leg`` -- the GEOMETRIC distance since the last
    emission or re-emission, tracked by :func:`propagate_to_plane` and
    :func:`_hfpi_segment_trace` and reset by
    :func:`apply_aperture_diffraction` -- and ``cos(theta_out)`` is
    ``paths.directions[..., 2]``.  A hand-built bundle with ``leg=None``
    falls back to ``opl``, which is the same number whenever the medium
    index is 1 and the bundle has not been through the ray tracer.

    Paths outside the grid get 0 -- they are masked out anyway.
    """
    xp = array_namespace(paths.positions)
    r = getattr(paths, 'leg', None)
    if r is None:
        r = paths.opl
    cos_out = paths.directions[..., 2]
    ok = inside & (r > 0) & (cos_out > 1e-12)
    if not is_jax_array(paths.positions):
        # A bundle whose landed paths have NOT travelled since their last
        # emission has no 1/r spreading to undo, and multiplying by r = 0
        # would silently return an all-zero field.  Say so instead.
        if (int(np.count_nonzero(to_numpy(inside))) > 0
                and int(np.count_nonzero(to_numpy(ok))) == 0):
            raise ValueError(
                "accumulate_to_grid: normalisation='physical' needs each "
                "path's geometric distance since its last emission (the "
                "1/r Huygens-Fresnel spreading), and every path that "
                "landed on this grid has travelled zero distance -- the "
                "bundle is being binned at the plane it was emitted on.  "
                "Propagate the bundle to the output plane first "
                "(propagate_to_plane), or pass normalisation='legacy' to "
                "get the raw (non-photometric) path sum.")
    denom = xp.where(ok, cos_out, 1.0) * (float(dx_out) ** 2)
    return xp.where(ok, r / denom, 0.0)


def accumulate_to_grid(
    paths: PathBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    output_dtype: Optional[Any] = None,
    on_undersampled: str = 'warn',
    normalisation: str = 'physical',
) -> np.ndarray:
    """Coherently bin a PathBundle into a 2-D output field.

    For each path, identify the destination pixel and add the
    path's complex weight to that pixel.  Paths that fall outside
    the grid are dropped.

    Parameters
    ----------
    on_undersampled : {'warn', 'silent', 'error'}, default 'warn'
        Policy for the v5.31 sampling-adequacy guard (audit W9-14).  Paths are
        sampled into a cone of half-angle ``cone_half_angle`` (default ~90
        degrees, a full forward hemisphere) while a realistic output grid
        subtends a few degrees, so the great majority of paths can miss the grid
        entirely and the survivors are too few to interfere into a field.  This
        counts the paths that actually LANDED and warns when there is less than
        :data:`_MIN_LANDED_PATHS_PER_OUTPUT_PIXEL` of them per output pixel --
        the point below which most pixels are exactly zero and the array is the
        sampling envelope plus shot noise, not a propagated field.  ``'error'``
        raises instead; ``'silent'`` suppresses (use it when you are pinning
        plumbing rather than physics).  Skipped for JAX arrays, whose path count
        is not readable under tracing.
    normalisation : {'physical', 'legacy'}, default 'physical'
        Whether to apply the output-binning Jacobian that turns the raw
        path sum into the Huygens-Fresnel integral (audit K13).

        * ``'physical'`` (default since v5.46) -- multiply each landed
          path by ``r / (dx**2 * cos(theta_out))``; together with the
          source-area weight (K18) this removes the estimator's exact
          bias law ``dx_out**2 * cos(theta) / (N_src_px * r)``.  Without
          it the returned amplitude depends on the OUTPUT PIXEL AREA and
          the SOURCE PIXEL COUNT and does not converge with path count:
          measured ``|E|max`` moved 14x between 2 M and 8 M paths, and
          simply rebinning the grids changed the answer with no physics
          change.  See :func:`_binning_jacobian` for the derivation and
          the measured confirmation.
        * ``'legacy'`` -- the pre-v5.46 raw sum.  Use it only to
          reproduce historical numbers; its amplitudes are not the HF
          integral's, so anything photometric must be re-normalised
          against a known-amplitude reference.

        .. warning::
           This CHANGES returned amplitudes by orders of magnitude
           relative to v5.45 and earlier.  Phase structure (fringe
           positions, interference contrast) is unaffected by the
           source-area factor and only weakly by the per-path Jacobian.
    """
    if normalisation not in ('physical', 'legacy'):
        raise ValueError(
            f"accumulate_to_grid: normalisation must be 'physical' "
            f"(default: the Huygens-Fresnel integral) or 'legacy' (the "
            f"pre-v5.46 raw path sum); got {normalisation!r}.")
    xp = array_namespace(paths.positions)

    if output_dtype is None:
        output_dtype = paths.weights.dtype

    # HFPI-1: CELL-CENTRED binning (+0.5) so pixel i collects
    # [x_i - dx/2, x_i + dx/2), matching every other grid consumer -- the
    # prior floor(x/dx + N/2) used [x_i, x_i + dx), a systematic half-pixel
    # image shift vs ASM/Fresnel on the same geometry.
    flat_idx, inside = _bin_paths(paths.positions, paths.alive,
                                  Ny, Nx, dx, centre)

    # v5.31 (audit W9-14): the sampling-adequacy guard, now shared with
    # the vector accumulator (K23).
    _check_landed(inside, Ny, Nx, on_undersampled, 'accumulate_to_grid',
                  positions=paths.positions)

    weights = paths.weights
    if normalisation == 'physical':
        # K13: the output-binning Jacobian.
        weights = weights * _binning_jacobian(
            paths, dx, inside).astype(weights.dtype)

    w_masked = xp.where(inside, weights, 0)

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


def _complex_output_dtype(dtype):
    """Promote a real dtype to its matching complex dtype
    (float64 -> complex128, float32 -> complex64); pass complex through.

    Path weights are intrinsically complex -- the ``1/(j*lambda)``
    prefactor and every ``exp(j*k*ds)`` leg -- so a REAL output buffer
    makes ``np.add.at`` discard their imaginary part behind nothing
    louder than a suppressible ComplexWarning.  Measured on a flat real
    source: ~40 % of the total intensity silently lost.  ``hf.py`` guards
    the same bug class the same way.
    """
    dt = np.dtype(dtype)
    if dt.kind == 'c':
        return dt
    if dt == np.float32:
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


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

    Normalisation (v5.46, audit K13 / K18)
    -------------------------------------
    The full Fresnel-Kirchhoff integral

        E(P) = (1/jλ) ∫∫ E(Q) · (cos θ / r) · exp(jkr) dS

    is sampled by Monte Carlo paths, and with the default
    ``normalisation='physical'`` every factor of it is now applied:

    * the ``1/(jλ)`` Kirchhoff prefactor -- at the source init (v4.10)
      and at every aperture re-emission (v4.11.2);
    * the Monte Carlo solid-angle weight ``2π·(1 − cos θ_max)/N_paths``
      -- same sites;
    * the obliquity factor (``cos θ`` at init; the symmetric
      ``(cos θ_in + cos θ_out)/2`` at re-emissions, v4.10.2);
    * the SOURCE AREA ``N_src_px·dx²`` (v5.46, K18 -- see
      :func:`init_paths_from_field`; pre-v5.46 only one pixel's ``dx²``
      was applied, so every amplitude was low by the source pixel
      count, measured 4096x at a 64x64 source);
    * the per-path ``r/(dx_out²·cos θ_out)`` output-binning Jacobian
      (v5.46, K13 -- see :func:`_binning_jacobian`), which supplies the
      missing ``1/r`` geometric spreading and the pixel-solid-angle
      conversion together.

    .. warning::
       What this warning said before v5.46 was itself wrong, which is
       why it is restated here.  It listed the Monte-Carlo solid angle
       and "the source pixel area ``dx²``" under *does apply* and
       attributed the non-quantitative amplitude solely to the missing
       ``1/r`` and binning Jacobian -- so a reader who corrected for
       the listed omissions still landed ``N_src_px`` low.  Both gaps
       are closed; the exact bias law
       ``E[HFPI]/E_true = dx_out²·cos θ/(N_src_px·r)`` was derived and
       confirmed to MC noise (meas/pred 1.0068 and 0.9950 at
       z = 2 / 4 mm with 24 M paths at occupancy 1.000).

       **This changes returned amplitudes by orders of magnitude versus
       v5.45 and earlier.**  Pass ``normalisation='legacy'`` to restore
       the raw path sum.

       HFPI remains a Monte-Carlo estimator: it converges as
       ``1/sqrt(N_paths)``, and below about one LANDED path per output
       pixel the returned array is the sampling envelope plus shot
       noise, not a field -- see :func:`accumulate_to_grid`'s
       ``on_undersampled`` guard and narrow ``cone_half_angle``
       (reachable from here since v5.46) before raising ``n_paths``.
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
    on_undersampled: str = 'warn',
    cone_half_angle: float = np.pi / 2 - 1e-6,
    normalisation: str = 'physical',
) -> np.ndarray:
    """End-to-end three-leg HFPI: source plane -> free space ->
    aperture -> free space -> output plane.

    The canonical single-aperture-diffraction validation case.

    Parameters
    ----------
    cone_half_angle : float, default ~pi/2
        Half-angle of the forward cone the source and the aperture
        re-emit into [rad], threaded to :func:`init_paths_from_field`
        and :func:`apply_aperture_diffraction`.

        .. versionadded:: 5.46
            Audit K14/K23.  The v5.31 under-sampling guard's own message
            recommends narrowing this "from its ~90-degree default (a
            full forward hemisphere) toward the angle the output grid
            actually subtends" -- and passing it raised ``TypeError:
            propagate_hfpi_freespace_aperture() got an unexpected
            keyword argument 'cone_half_angle'``, on this function, on
            :func:`propagate_hfpi` and through
            ``propagate(method='hfpi')``.  Only the prescription walk
            exposed it, so the only free-space HFPI entry point in the
            library was the one that could not take the lever it
            recommended.
    normalisation : {'physical', 'legacy'}, default 'physical'
        Forwarded to :func:`accumulate_to_grid`; see K13 there.

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
        cone_half_angle=cone_half_angle,
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
        cone_half_angle=cone_half_angle,
        # V1: the re-emission measure and the binning Jacobian are two
        # halves of ONE estimator; they must agree.
        normalisation=normalisation,
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
        # v5.17.x (P2-32): promote real input dtypes to complex so the
        # scatter-add keeps the imaginary half of the path weights.
        output_dtype=_complex_output_dtype(E_in.dtype),
        # v5.31 (audit W9-14): sampling-adequacy guard.
        on_undersampled=on_undersampled,
        # K13: the output-binning Jacobian.
        normalisation=normalisation,
    )


# ============================================================================
# Variance reduction: stratified sampling
# ============================================================================

def _walk_legs_are_free_space(surfaces, surface_diffraction, wavelength):
    """Would every leg of a prescription walk be a straight free-space
    hop?  (audit K13, the condition the binning Jacobian needs.)

    :func:`_binning_jacobian` and :func:`_reemission_measure` convert
    between emitted SOLID ANGLE and landed AREA with the free-space
    ray-tube Jacobian ``dS = r^2 dOmega / cos(theta)``.  That identity is
    a statement about a straight line in a uniform medium.  Put anything
    with power between the emission and the landing point -- a curved
    surface, an index step, a tilt, a grating order -- and the system's
    own Jacobian replaces it, while the per-path factor does not know.

    The size of the mistake is not subtle.  Walking a 19.41 mm thin
    singlet (N-BK7, R = +-20 mm, d = 10 um, object 60 mm, 2 M paths) with
    ``normalisation='physical'`` and comparing total power against
    ASM + thin-lens phase + ASM on the same geometry: the walk reads
    **4879x** the reference at the image plane and **13.9x** at half that
    distance, while the spot metrics stay in the right ballpark (r50
    15.9 um against 17.9 um, r84 25.3 um against 31.9 um at the image
    plane) -- an amplitude error, not a shape error, and one no caller
    could infer from the output.

    A prescription this returns True for has no such element: every
    surface is a plane, with equal index on both sides, unsteered and
    unkicked, so the only thing the walk does between emissions is
    travel.  That is the regime the composition is exact in (measured
    least-squares scale 1.13 / 0.89 against ASM x mask x ASM at
    0.5 M / 2 M paths).

    Curved-but-index-matched and tilted-but-flat surfaces are both
    reported False: the first is a plane only by accident of the current
    prescription and the second steers the ray tube.
    """
    if surface_diffraction:
        return False
    from ..glass import get_glass_index
    for s in surfaces:
        if getattr(s, 'is_mirror', False) or getattr(s, 'is_coordbrk', False):
            return False
        if getattr(s, 'freeform', None) is not None:
            return False
        if getattr(s, 'field_sag_callable', None) is not None:
            return False
        if getattr(s, 'field_decenter', None) or getattr(s, 'field_tilt', None):
            return False
        for _key in ('radius', 'radius_y'):
            _r = getattr(s, _key, None)
            if _r is not None and np.isfinite(_r):
                return False
        for _key in ('aspheric_coeffs', 'aspheric_coeffs_y'):
            _c = getattr(s, _key, None)
            if _c is not None and np.any(np.asarray(_c, dtype=float) != 0.0):
                return False
        for _key in ('tilt_x_deg', 'tilt_y_deg', 'tilt_z_deg',
                     'decenter_x_m', 'decenter_y_m'):
            if getattr(s, _key, 0.0):
                return False
        _n1 = float(get_glass_index(getattr(s, 'glass_before', None),
                                    wavelength))
        _n2 = float(get_glass_index(getattr(s, 'glass_after', None),
                                    wavelength))
        if abs(_n2 - _n1) > 1e-12 * max(1.0, abs(_n1)):
            return False
    return True


def _sobol_cube_draw(rs, n_paths, Ny, Nx, cos_max):
    """Place ``n_paths`` scrambled Sobol points in the 4-D
    ``(pixel_x, pixel_y, cos theta, phi)`` cube -- the ``sampler='sobol'``
    half of :func:`init_paths_stratified` (audit K22).

    Returns the same five values :func:`_jittered_cube_draw` does, with
    ``n_paths_actual == n_paths`` exactly: a low-discrepancy sequence has
    no stratification grid to round the count onto.

    The cube is the one the Huygens-Fresnel source term integrates over,
    and the mapping is the measure-preserving one the jittered sampler
    uses: the two pixel axes scale by ``Ny`` / ``Nx``, ``cos theta`` maps
    affinely onto ``[cos_max, 1]`` (equal solid angle per unit of the
    coordinate) and ``phi`` onto ``[0, 2 pi)``.  Owen scrambling is ON,
    which is what keeps the estimator UNBIASED -- an unscrambled Sobol
    sequence is a fixed point set, so its error is a deterministic bias
    with no way to measure it and no convergence in the sense the caller
    wants.  The scramble seed is drawn from ``rs``, so the bundle stays a
    pure function of the caller's ``rng`` on every backend.
    """
    from scipy.stats import qmc

    n = int(n_paths)
    if n < 1:
        raise ValueError(
            f"init_paths_stratified: sampler='sobol' needs n_paths >= 1 "
            f"(got {n_paths!r}).")
    if n & (n - 1):
        warnings.warn(
            f"init_paths_stratified: sampler='sobol' with n_paths={n}, "
            f"which is not a power of two.  A Sobol sequence is balanced "
            f"only on its 2**m prefixes -- a partial block leaves the "
            f"projections uneven, which costs exactly the low-discrepancy "
            f"property the sampler is chosen for.  Round n_paths to "
            f"{1 << (n.bit_length() - 1)} or {1 << n.bit_length()}, or "
            f"pass sampler='jittered' (whose count is a cap, not a power "
            f"of two).",
            UserWarning, stacklevel=3)
    seed = int(np.asarray(to_numpy(
        rs.integers((1,), low=0, high=2 ** 31 - 1))).reshape(-1)[0])
    with warnings.catch_warnings():
        # scipy warns about the same non-power-of-two balance property,
        # naming its own class rather than the function the caller
        # invoked; the message above says it with the remedy.
        warnings.simplefilter('ignore', UserWarning)
        u = np.asarray(qmc.Sobol(d=4, scramble=True, seed=seed).random(n),
                       dtype=np.float64)

    ix_int = np.clip((u[:, 0] * Nx).astype(np.int64), 0, Nx - 1)
    iy_int = np.clip((u[:, 1] * Ny).astype(np.int64), 0, Ny - 1)
    cos_theta_h = cos_max + (1.0 - cos_max) * u[:, 2]
    phi_h = 2.0 * float(np.pi) * u[:, 3]
    return n, iy_int, ix_int, cos_theta_h, phi_h


def _jittered_cube_draw(rs, n_paths, Ny, Nx, cos_max,
                        n_strata_xy, n_strata_dir):
    """Place ``n_paths`` jittered points in the 4-D
    ``(pixel_x, pixel_y, cos theta, phi)`` cube by equal-measure
    stratification -- the ``sampler='jittered'`` half of
    :func:`init_paths_stratified`.

    Returns ``(n_paths_actual, iy_int, ix_int, cos_theta_h, phi_h)`` as
    host arrays; the caller turns them into positions, directions and
    weights.  ``n_paths_actual`` can differ from ``n_paths``: the
    stratification grid is what the points are placed on, so the count
    lands on a multiple of the strata used (``n_paths`` is a cap, never
    a floor).
    """
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
    # If the user requested fewer than n_total, sample only n_paths
    # strata uniformly without replacement.  If they requested more,
    # sample multiple paths per stratum (jittered).
    #
    # K23 (audit 2026-09-11): the first sentence describes behaviour the
    # code never implemented -- ``n_per = max(1, n_paths // n_total)``
    # clamps to 1 and ``n_paths_actual = n_per * n_total >= n_total``
    # REGARDLESS of n_paths, so an explicit stratification allocated far
    # MORE paths than requested and the ``[:n_paths_actual]`` truncation
    # below was always a no-op.  Measured on a 16x16 source:
    # ``n_paths=100, n_strata_xy=(16,16), n_strata_dir=(16,16)`` ->
    # 65 536 paths (655x, 4.8 MB); ``(32,32)/(32,32)`` -> 1 048 576
    # (10 486x, 76.6 MB); ``(64,64)/(64,64)`` -> 16.8 M (~1.2 GB) from an
    # n_paths=100 call.  ``n_paths`` is now an honest CAP: when
    # ``n_total > n_paths`` a uniform random subset of n_paths strata is
    # drawn WITHOUT replacement, which is what the sentence above always
    # promised.  The default path (the 4th-root rule keeps
    # ``n_total ~ n_paths``) is unchanged.
    n_per = max(1, n_paths // n_total)
    n_strata_used = min(n_total, int(n_paths)) if n_total > n_paths else n_total
    n_paths_actual = n_per * n_strata_used

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
    if n_strata_used < n_total:
        # K23: the documented sub-sampling.  A uniform subset of the
        # strata, without replacement -- still an unbiased estimator of
        # the same integral (each stratum is equally likely), just with
        # the variance of n_strata_used samples rather than n_total.
        # Drawn from the SAME RandomState as the jitter so the whole
        # bundle stays a pure function of ``rng``.
        _sub = np.sort(np.asarray(
            rs.choice(n_total, (n_strata_used,), replace=False)))
        idx_grid = idx_grid[:, _sub]
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
    # cos_theta uniform on [cos_max, 1] partitioned into n_th strata.
    cth_strata_low = cos_max + (1.0 - cos_max) * (th_strata / n_th)
    cth_strata_width = (1.0 - cos_max) / n_th
    cos_theta_h = cth_strata_low + np.asarray(u_th) * cth_strata_width
    # phi uniform on [0, 2 pi].
    phi_h = 2 * float(np.pi) * (ph_strata + np.asarray(u_ph)) / n_ph
    return n_paths_actual, iy_int, ix_int, cos_theta_h, phi_h


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
    sampler: str = 'jittered',
) -> PathBundle:
    """Low-discrepancy variant of :func:`init_paths_from_field`.

    Both samplers cover the same 4-D unit cube
    ``(pixel_x, pixel_y, cos theta, phi)`` that the Huygens-Fresnel
    source term integrates over; they differ in how the ``n_paths``
    points are placed in it.  The default partitions the cube into
    equal-measure strata and jitters one path inside each, which reduces
    the variance of the Monte Carlo estimate by forcing uniform coverage
    of phase space -- the integrated complex weight has lower variance
    than naive uniform sampling at the same path count (typically 2-10x
    for smooth-amplitude / smooth-OPL systems).

    Parameters
    ----------
    sampler : {'jittered', 'sobol'}, default 'jittered'
        ``'jittered'`` -- the stratified sampler described above, and the
        one ``n_strata_xy`` / ``n_strata_dir`` configure.

        ``'sobol'`` -- a scrambled 4-D Sobol sequence
        (``scipy.stats.qmc.Sobol(d=4).random(n_paths)``) over the same
        cube (audit K22).  ``n_paths`` is honoured EXACTLY, with no
        stratification grid to round it to; the Owen scrambling is seeded
        from ``rng`` so the bundle stays a pure function of it, and keeps
        the estimator unbiased so both samplers converge to the same
        field.

        The QMC textbook rate (``O(N^-1)`` against Monte Carlo's
        ``O(N^-1/2)``) assumes a smooth integrand and this one has hard
        edges -- the aperture, the cone cut and the output-pixel bin all
        put discontinuities inside the cube -- so the gain here is
        MEASURED, not assumed; see the figures in the changelog.  Raises
        if ``n_strata_xy`` / ``n_strata_dir`` are also given, which would
        be a contradictory request.  Warns when ``n_paths`` is not a
        power of two, where the sequence's balance property does not
        hold.
    n_strata_xy : (n_iy, n_ix), optional
        Per-axis number of strata for the source-pixel index.
        Default ``(sqrt(n_paths), sqrt(n_paths))``.
    n_strata_dir : (n_theta, n_phi), optional
        Per-axis number of strata for the direction sphere.
        Default same as n_strata_xy.

    All other parameters mirror :func:`init_paths_from_field`.
    """
    if sampler not in ('jittered', 'sobol'):
        raise ValueError(
            f"init_paths_stratified: sampler must be 'jittered' (default: "
            f"one jittered path per equal-measure stratum) or 'sobol' (a "
            f"scrambled 4-D Sobol sequence over the same cube, with "
            f"n_paths honoured exactly); got {sampler!r}.")
    if sampler == 'sobol' and (n_strata_xy is not None
                               or n_strata_dir is not None):
        raise ValueError(
            f"init_paths_stratified: sampler='sobol' places points by the "
            f"Sobol sequence, not on a stratification grid, so "
            f"n_strata_xy={n_strata_xy!r} / n_strata_dir={n_strata_dir!r} "
            f"have nothing to configure.  Drop them for the Sobol sampler "
            f"(n_paths is already an exact count there), or pass "
            f"sampler='jittered' to use them.")
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    # ``rng=None`` -- the DEFAULT on every HFPI entry point -- must draw
    # fresh system entropy, which is exactly what ``RandomState(None)``
    # does (``np.random.default_rng(None)``).  HFPI is a 1/sqrt(N)
    # Monte-Carlo estimator: with a FIXED default seed, the canonical way
    # to see the estimator's own error -- re-run and compare -- returns
    # identically ZERO (audit K19).  Pass an int (or a Generator) for
    # reproducibility.
    rs = RandomState(rng=rng)

    cos_max = float(np.cos(cone_half_angle))
    if sampler == 'sobol':
        (n_paths_actual, iy_int, ix_int, cos_theta_h,
         phi_h) = _sobol_cube_draw(rs, n_paths, Ny, Nx, cos_max)
    else:
        (n_paths_actual, iy_int, ix_int, cos_theta_h,
         phi_h) = _jittered_cube_draw(rs, n_paths, Ny, Nx, cos_max,
                                      n_strata_xy, n_strata_dir)

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
    # K18: the source-AREA factor -- see :func:`init_paths_from_field`.
    solid_angle = (2.0 * float(np.pi) * (1.0 - cos_max)
                   * (float(Ny) * float(Nx)) / float(n_paths_actual))
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
        leg=xp.zeros_like(opl),
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
    z_output: Optional[float] = None,
    sampling: str = 'stratified',
    sampler: str = 'jittered',
    cone_half_angle: float = np.pi / 2 - 1e-6,
    on_undersampled: str = 'warn',
    normalisation: str = 'auto',
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
    z_output : float | None, optional
        Axial position of the OUTPUT PLANE [m], in the same world
        coordinate the prescription's surfaces sit at (paths start at
        ``-object_distance``).  ``None`` (default) bins the bundle where
        the walk leaves it -- at the last surface.

        Give it a value and the walk closes with a
        :func:`propagate_to_plane` hop to that plane, which is what makes
        ``normalisation='physical'`` meaningful here: the per-path
        Huygens-Fresnel binning Jacobian ``r/(dx_out^2 cos theta_out)``
        needs the geometric length of the leg that ended at the output
        plane, and a bundle binned at a diffracting last surface has just
        been re-emitted, so that length is zero (audit K13).  The closing
        leg is taken in the medium the prescription puts after its last
        surface (``glass_after``), so an immersed image space is handled.
        A REFLECTIVE last surface keeps its medium -- the ``'MIRROR'``
        marker resolves to the surface's own ``glass_before``, because
        reflection does not change the surrounding medium -- but it folds
        the propagation direction, so ``z_output`` then has to lie on the
        side the reflected paths travel toward.

        The hop is a necessary and NOT a sufficient condition for
        photometric amplitudes: that Jacobian is the free-space ray-tube
        relation, so it is the right one only when the walk's legs are
        free space.  ``normalisation`` (below) resolves both questions.

        Paths whose direction cannot reach the plane (already past it, or
        travelling parallel to it) are killed by the hop, exactly as on
        the free-space entry points; the sampling-adequacy guard then
        reports what landed.
    sampling : str
        ``'uniform'`` or ``'stratified'`` (default).  Stratified
        partitions the source-direction cone into equal-solid-angle
        cells and forces one path per cell, reducing variance.
    sampler : {'jittered', 'sobol'}, default 'jittered'
        Point placement inside the stratified sampler's 4-D cube; see
        :func:`init_paths_stratified` (audit K22).  Read only when
        ``sampling='stratified'``, and refused otherwise rather than
        silently ignored.
    cone_half_angle : float
        Half-angle of the forward emission cone for path
        initialisation and per-surface re-sampling.
    on_undersampled : {'warn', 'silent', 'error'}, default 'warn'
        Policy for the v5.31 sampling-adequacy guard (audit W9-14): HFPI warns
        when fewer than one sampled path per output pixel actually LANDS on the
        grid, the point below which the return is the Monte-Carlo sampling
        envelope rather than a field.  The usual cause is ``cone_half_angle``
        (a full forward hemisphere by default) versus an output grid that
        subtends a few degrees.  See :func:`accumulate_to_grid`.
    normalisation : {'auto', 'physical', 'legacy'}, default 'auto'
        Which estimator to run; the same value is threaded to every
        aperture re-emission and to the final binning, because those are
        two halves of ONE estimator.

        ``'auto'`` resolves to ``'physical'`` when BOTH conditions the
        estimator needs hold -- ``z_output`` gives the walk a plane to
        close on, and every leg of the walk is free space
        (:func:`_walk_legs_are_free_space`) -- and to ``'legacy'``
        otherwise.  So a call that passes no ``z_output`` returns exactly
        what it did before the keyword existed, a flat-optics walk that
        names its output plane gets photometric amplitudes with no second
        keyword to remember, and a walk through an element WITH POWER is
        not silently handed an amplitude the free-space Jacobian cannot
        produce (measured 4879x at a thin lens's image plane).

        ``'physical'`` and ``'legacy'`` force the choice.  Forcing
        ``'physical'`` through a powered prescription is allowed and
        warns with that measurement, because the shape is still usable
        and re-normalising against a reference is a legitimate workflow.
        See :func:`accumulate_to_grid` for what each does per path.

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
    # Validate the sampling selector up front, before the expensive
    # prescription parse / trace, so a typo cannot silently fall through
    # to uniform sampling (HFPI-2).
    if sampling not in ('uniform', 'stratified'):
        raise ValueError(
            f"sampling must be 'uniform' or 'stratified'; got {sampling!r}.")
    if sampler not in ('jittered', 'sobol'):
        raise ValueError(
            f"propagate_hfpi_through_prescription: sampler must be "
            f"'jittered' or 'sobol'; got {sampler!r}.  See "
            f"init_paths_stratified for what each places where.")
    if sampler != 'jittered' and sampling != 'stratified':
        raise ValueError(
            f"propagate_hfpi_through_prescription: sampler={sampler!r} "
            f"places points inside the STRATIFIED sampler's 4-D cube, and "
            f"this call passes sampling={sampling!r}, which does not use "
            f"that cube -- the request has no effect and would be silently "
            f"dropped.  Pass sampling='stratified' (the default) with the "
            f"sampler you want, or drop the sampler keyword.")
    if normalisation not in ('auto', 'physical', 'legacy'):
        raise ValueError(
            f"propagate_hfpi_through_prescription: normalisation must be "
            f"'auto' (default: 'physical' when z_output gives the walk a "
            f"plane to close on AND every leg of the walk is free space, "
            f"'legacy' otherwise), 'physical' (forced, and warned about "
            f"when the legs are not free space) or 'legacy' (the raw path "
            f"sum); got {normalisation!r}.")
    if z_output is not None:
        z_output = float(z_output)
        if not np.isfinite(z_output):
            raise ValueError(
                f"propagate_hfpi_through_prescription: z_output must be a "
                f"finite axial position in metres, in the same world "
                f"coordinate the prescription's surfaces sit at; got "
                f"{z_output!r}.  Pass None to bin the bundle where the walk "
                f"leaves it (at the last surface).")
    from ..raytrace import (
        surfaces_from_prescription,
    )

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

    # Resolve the estimator UP FRONT, because the aperture re-emissions
    # and the final binning are two halves of ONE estimator and the walk
    # threads a single value to both.
    #
    # K13: the per-path Jacobian both halves apply is the FREE-SPACE
    # ray-tube relation, so 'physical' means what it says only when the
    # walk's legs are free space AND a closing hop gives the last one a
    # length.  'auto' therefore asks both questions; see
    # :func:`_walk_legs_are_free_space` for the measurement that settles
    # the first one.
    _free_legs = _walk_legs_are_free_space(surfaces, surface_diffraction,
                                           wavelength)
    norm_used = normalisation
    if norm_used == 'auto':
        norm_used = ('physical' if (z_output is not None and _free_legs)
                     else 'legacy')
    elif norm_used == 'physical' and not _free_legs:
        warnings.warn(
            "propagate_hfpi_through_prescription: normalisation='physical' "
            "converts emitted solid angle to landed area with the "
            "FREE-SPACE ray-tube Jacobian r/(dx_out^2 cos theta_out), and "
            "this prescription puts an element with power (a curved "
            "surface, an index step, a tilt or a grating order) between "
            "the emissions and the output plane, where the system's own "
            "Jacobian applies instead.  The returned amplitudes are "
            "therefore not photometric: measured against ASM + thin-lens "
            "phase + ASM through a 19.41 mm singlet, the walk's total "
            "power reads 4879x the reference at the image plane and 13.9x "
            "at half that distance, while the spot metrics stay close "
            "(r50 15.9 um against 17.9 um).  Shape is usable, absolute "
            "scale is not -- re-normalise against a known-amplitude "
            "reference, or pass normalisation='legacy' to say so "
            "explicitly.",
            RuntimeWarning, stacklevel=2)

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
    # Honour the ``sampling`` selector (validated up front): 'stratified'
    # routes to :func:`init_paths_stratified`, the variance-reduction
    # path (HFPI-2).
    if sampling == 'stratified':
        paths = init_paths_stratified(
            E_in, dx,
            n_paths=n_paths,
            wavelength=wavelength,
            rng=rng_source,
            cone_half_angle=cone_half_angle,
            z_input_plane=-object_distance,
            sampler=sampler,
        )
    else:
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
                    # V1: must match the accumulator's choice below.
                    normalisation=norm_used,
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

    # 4.  Close on the output plane, if the caller named one.
    #
    # K13: the binning Jacobian r/(dx_out^2 cos theta_out) needs the
    # GEOMETRIC length of the leg that ended at the plane being binned.
    # Without this hop the walk bins where it stops -- at the last
    # surface -- and a diffracting last surface has just re-emitted every
    # path, so that length is zero and there is no photometric answer to
    # give.  The hop supplies the leg, in the medium the prescription
    # puts after its last surface.
    if z_output is not None:
        # The index of the closing leg.  ``surfaces_from_prescription``
        # normalises a reflective surface's ``'MIRROR'`` marker to its
        # own ``glass_before`` (reflection does not change the medium),
        # so this reads correctly for a folded stack too -- what a
        # reflective last surface does change is the DIRECTION, so
        # ``z_output`` then has to sit on the side the folded paths
        # travel toward or the hop kills them.
        from ..glass import get_glass_index
        n_out_medium = (float(get_glass_index(
            getattr(surfaces[-1], 'glass_after', None), wavelength))
            if surfaces else 1.0)
        paths = propagate_to_plane(paths, z_target=z_output,
                                   wavelength=wavelength,
                                   n_medium=n_out_medium)

    # 5.  Accumulate to output grid.
    if norm_used == 'legacy':
        warnings.warn(
            "propagate_hfpi_through_prescription: the returned amplitudes "
            "are NOT photometric.  This walk bins the bundle at the last "
            "surface rather than propagating it to a separate output "
            "plane, so the per-path r/(dx_out^2 cos theta_out) "
            "Huygens-Fresnel binning Jacobian (audit K13, applied by the "
            "free-space entry points) cannot be evaluated -- the last leg "
            "has zero length.  Fringe positions and interference contrast "
            "are unaffected; re-normalise against a known-amplitude "
            "reference (e.g. ASM on the same geometry) for anything "
            "photometric.  Pass z_output=<the plane you want the field on> "
            "to give the walk a final leg -- on a prescription whose legs "
            "are all free space (flat, index-matched, unsteered surfaces) "
            "that makes the amplitudes photometric and retires this "
            "warning.  Through an element WITH POWER it cannot: the "
            "per-path Jacobian is the free-space ray-tube relation and the "
            "system's own applies instead, so 'auto' stays here and "
            "normalisation='physical' has to be asked for explicitly.",
            RuntimeWarning, stacklevel=2)
    return accumulate_to_grid(
        paths,
        Ny=Ny_out, Nx=Nx_out,
        dx=output_dx, centre=output_centre,
        normalisation=norm_used,
        # v5.17.x (P2-32): promote real input dtypes to complex so the
        # scatter-add keeps the imaginary half of the path weights.
        output_dtype=_complex_output_dtype(E_in.dtype),
        # v5.31 (audit W9-14): sampling-adequacy guard.
        on_undersampled=on_undersampled,
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
    #
    # K23 (audit 2026-09-11): route EVERY backend through ``to_numpy``,
    # not just JAX.  ``np.asarray`` on a CuPy device array raises
    # ``TypeError: Implicit conversion to a NumPy array is not allowed``
    # in modern CuPy, so a CuPy bundle crashed here -- on the only path
    # the module docstring's "accepting NumPy / CuPy / JAX source fields"
    # claim could be exercised.  (Desk-check: CuPy is not installed in
    # this environment.)  ``to_numpy`` is the identity for NumPy, so the
    # NumPy path is unchanged.
    #
    # The host round-trip itself is unavoidable here -- ``raytrace.trace``
    # is a host solver -- so the prescription walk is NOT traceable under
    # ``jit`` / ``vmap`` / ``grad``, as the module docstring now states.
    pos_h = np.asarray(to_numpy(paths.positions))
    dir_h = np.asarray(to_numpy(paths.directions))
    opl_in = np.asarray(to_numpy(paths.opl))
    alive_in = np.asarray(to_numpy(paths.alive))

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

    # K13: accumulate the GEOMETRIC distance the segment moved each path.
    _base_leg = (paths.leg if paths.leg is not None
                 else xp.zeros_like(paths.opl))
    _step = xp.sqrt(xp.sum((new_positions - paths.positions) ** 2, axis=-1))
    return PathBundle(
        positions=new_positions,
        directions=new_directions,
        weights=new_weights,
        opl=xp.asarray(out_rb.opd),
        alive=new_alive,
        leg=_base_leg + _step,
    )


__all__ = [
    'PathBundle',
    # K24 (audit 2026-09-11): ``propagate_hfpi`` -- "Canonical-order HFPI
    # three-leg propagation" -- was absent from ``__all__`` (reachable as
    # ``lumenairy.propagate_hfpi`` throughout, so an integrity gap rather
    # than a breakage).
    'propagate_hfpi',
    'init_paths_from_field',
    'init_paths_stratified',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi_freespace_aperture',
    'propagate_hfpi_through_prescription',
]
