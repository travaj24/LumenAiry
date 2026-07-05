"""
lumenairy.gbd -- Gaussian Beamlet Decomposition propagator.

Decompose an arbitrary complex source field into a finite set of
Gaussian beamlets, propagate each beamlet's base ray + complex
beam parameter through the optical system using ABCD matrices,
then coherently recombine at the output plane.

The deterministic counterpart to Monte Carlo HFPI.  Strengths:

* **Deterministic** -- no Monte Carlo noise.
* **Fast** -- typically 100x faster than HFPI for comparable
  image-plane accuracy on smooth refractive systems.
* **Composes with raytrace** -- each beamlet's base ray is just a
  geometric ray, so the existing ``trace`` infrastructure
  propagates everything.
* **Aberration-aware (per-surface form)** --
  ``propagate_gbd_through_prescription(..., per_surface=True)`` evolves each
  beamlet's complex parameter surface-by-surface via the real per-ray
  differential ray transfer (``raytrace.ray_transfer_jacobian``), promoting
  ``Q`` to a ``(N, 2, 2)`` tensor that captures off-axis **astigmatism** and
  higher-order aberration (the paraxial whole-system-ABCD form cannot).
* **Differentiable** -- backend-dispatched (``array_namespace``), so the
  free-space / thin-lens paths run under ``jax.numpy`` and are ``jax.grad`` /
  ``jax.jit`` friendly.

Feature helpers: Husimi (direction-sampling) decomposition for tilted /
diverging sources (``direction_sampling=True``), aperture vignetting
(``apply_aperture_to_beamlets``), polychromatic (``propagate_gbd_freespace_
spectral``), vector / Jones (``propagate_gbd_freespace_vector``), and
auto-sampling (``recommend_gbd_sampling``).

Limitations:

* **Smooth aperture handling** -- a Gaussian beamlet has continuous
  edges; HFPI handles hard cutoffs better.
* **Caustic-region accuracy** -- like all paraxial complex-ray
  methods, GBD's accuracy degrades near a caustic.
* **Polarization** -- ``propagate_gbd_freespace_vector`` propagates the Jones
  components independently (exact for non-polarizing / free-space systems);
  ``propagate_gbd_vector_through_prescription`` carries the vector field through
  a real prescription applying per-surface Fresnel s/p transmission
  (polarization ray tracing).  Reflection / thin-film coatings build on the same
  base-ray trace and remain a future extension.

See ``REFERENCES.txt`` Section C for the foundational publications.

Multi-backend
-------------

Backend dispatched via :func:`lumenairy._array.array_namespace`.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation.  Pre-v5.2 the sub-propagators below interpreted
# ``output_grid`` as ``(Ny, Nx)`` (shape only) while the dispatcher's
# ``propagate(output_grid=...)`` contract advertises ``(N_out, dx_out)``.
# The mismatch silently produced wrong-shape output arrays when the
# dispatcher forwarded the kwarg.  v5.2 keeps the dispatcher contract
# as canonical and renames the sub-propagator kwarg to ``output_shape``
# for the shape-only meaning.  The legacy ``output_grid`` kwarg on
# the sub-propagators is preserved with a ``DeprecationWarning``.
def _resolve_output_shape(
    output_shape: Optional[Tuple[int, int]],
    output_grid: Optional[Any],
    *,
    fn_name: str,
    default_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the (Ny, Nx) output shape from the v5.2 ``output_shape``
    kwarg and the deprecated ``output_grid`` legacy kwarg.

    Both kwargs may be ``None`` (use the default).  Passing both raises
    ``ValueError``.  Passing ``output_grid`` emits a
    ``DeprecationWarning`` directing the caller to either
    ``output_shape=(Ny, Nx)`` (the new name for this sub-propagator's
    shape-only kwarg) or the dispatcher's ``output_grid=(N_out, dx_out)``
    convention if they actually want grid resampling.
    """
    if output_shape is not None and output_grid is not None:
        raise ValueError(
            f"{fn_name}: both ``output_shape`` and ``output_grid`` were "
            f"provided.  Pass only ``output_shape=(Ny, Nx)`` for the "
            f"shape-only kwarg (v5.2+) or the dispatcher's "
            f"``output_grid=(N_out, dx_out)`` form via "
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
            f"call via ``propagate(method='gbd', output_grid=(N_out, "
            f"dx_out), ...)`` if you actually want grid resampling.",
            DeprecationWarning, stacklevel=3,
        )
        return (int(output_grid[0]), int(output_grid[1]))
    return default_shape


# ============================================================================
# Beamlet bundle
# ============================================================================

@dataclass
class BeamletBundle:
    """Coherent set of Gaussian beamlets.

    Field naming aligns with :class:`lumenairy.raytrace.RayBundle`
    and :class:`lumenairy.hfpi.PathBundle`: ``positions`` and
    ``directions`` are the per-beamlet base ray's central position
    and direction.  GBD-specific additions are ``Q`` (complex beam
    parameter), ``amplitude`` (complex on-axis scale), and
    ``waist0`` (initial waist used to evaluate the transverse
    profile).
    """

    positions: object       # (N, 3) -- base ray position per beamlet
    directions: object      # (N, 3) -- base ray direction per beamlet
    # v5.4.7 (audit AUDIT_V5_4_6 #9): Q is the ENGINEERING 1/q parameter
    # (q_code = conj(q_physics)); at the waist Q = -i/z_R.  The physics
    # (exp(-i omega t) / exp(+ikz)) field is rendered in
    # reconstruct_field_from_beamlets via exp(+0.5j k conj(Q) rho^2) =
    # exp(+i k rho^2 / (2 q_physics)) (the v5.4.6 F-1 fix).  The Q-evolution
    # (Q/(1+tQ), Q-1/f) is left in this engineering convention.
    Q: object               # (N,) complex (engineering 1/q; see note above)
    amplitude: object       # (N,) complex on-axis amplitude
    waist0: object          # (N,) initial waist (for profile)

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except (AttributeError, TypeError, IndexError):
            # positions may be None, a non-array sentinel, or 0-D.
            return 0


def _q_is_tensor(Q: Any) -> bool:
    """True if ``Q`` is a (N, 2, 2) tensor beam parameter (general astigmatic
    Gaussian), False for the (N,) scalar (rotationally-symmetric) parameter.
    The single dispatch predicate that keeps the scalar paths untouched."""
    return getattr(Q, 'ndim', 1) == 3


# ============================================================================
# Source-plane decomposition
# ============================================================================

def decompose_field_to_beamlets(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    dy: Optional[float] = None,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    z_input_plane: float = 0.0,
    direction_sampling: bool = False,
) -> BeamletBundle:
    """Decomposition of a 2-D complex source field into a regular grid
    of Gaussian beamlets.

    Each grid pixel becomes a beamlet centred at its physical
    coordinate, with on-axis amplitude ``E_in[i, j]`` and Gaussian
    waist ``w0 = waist_factor * dx``.

    Parameters
    ----------
    direction_sampling : bool, default ``False``
        Selects how each beamlet's launch **direction** is chosen.

        ``False`` (default, position-only) -- every beamlet is launched
        along ``(0, 0, +1)`` and the field's local tilt is carried in
        the beamlet AMPLITUDE.  Backward-compatible and byte-identical to
        the historical behaviour.  Correct for near-collimated / paraxial
        sources and for any beam that *acquires* its angles from later
        optics (the direction is updated at each lens / ABCD element), but
        a source that is *already* tilted or diverging at the input plane
        is not walked off correctly (its intensity envelope stays put).

        ``True`` ("Husimi" / Gabor decomposition) -- each beamlet is
        launched along the field's **local wavevector**, obtained from the
        transverse phase gradient
        ``k_x = d(arg E)/dx = Im((dE/dx) / E)`` (and likewise ``k_y``);
        the direction cosines are ``(k_x/k, k_y/k, sqrt(1 - ...))``.  The
        tilt then lives in the *direction*, so a tilted or diverging
        source walks off correctly on propagation.  Falls back to axial
        where ``|E|`` is negligible or the local angle exceeds the light
        cone (``k_x^2 + k_y^2 >= k^2``, i.e. under-sampled / evanescent),
        so it never produces NaNs.  The local gradient must be resolved by
        the grid (phase change ``< pi`` per pixel); beyond that use HFPI.
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    # v5.21: anamorphic sampling.  dy == dx (the default) keeps the scalar-Q
    # circular-beamlet path byte-identical; dy != dx makes each beamlet an
    # ELLIPTICAL (axis-aligned) Gaussian -> a DIAGONAL tensor Q.
    if dy is None:
        dy = dx
    _aniso = abs(float(dy) - float(dx)) > 1e-12 * max(abs(float(dx)), 1.0)

    iy = xp.arange(0, Ny, sample_step)
    ix = xp.arange(0, Nx, sample_step)
    Iy, Ix = xp.meshgrid(iy, ix, indexing='ij')
    Iy = Iy.reshape(-1)
    Ix = Ix.reshape(-1)
    n = Iy.shape[0]

    # v4.12.1 (B1-10): switch from cell-centred `(arange(N) - N/2 + 0.5)*dx`
    # to pixel-centred `(arange(N) - N/2)*dx`, matching the library-wide
    # convention (ASM, Fresnel, RS, sources, ``apply_fresnel_curvature``).
    # ``reconstruct_field_from_beamlets`` (line ~264) already uses the
    # pixel-centred grid, so prior to this fix a coherent self-roundtrip
    # walked the beamlet centres half a pixel relative to the
    # reconstruction grid -- producing a `k_0 * dx / 2 * off-axis` phase
    # error that grew with NA and field angle.
    x_b = (Ix - Nx / 2) * dx
    y_b = (Iy - Ny / 2) * dy
    z_b = xp.full((n,), float(z_input_plane), dtype=x_b.dtype)
    positions = xp.stack([x_b, y_b, z_b], axis=-1)

    if direction_sampling:
        # Husimi / Gabor: launch each beamlet along the field's LOCAL
        # wavevector k_local = grad(arg E) = Im((grad E) / E).  This puts
        # the input tilt into the beamlet DIRECTION (so it walks off on
        # propagation) instead of only its amplitude.
        kmag = 2.0 * float(np.pi) / wavelength
        gx = xp.gradient(E_in, dx, axis=-1)      # dE/dx
        gy = xp.gradient(E_in, dy, axis=-2)      # dE/dy
        E_c = E_in[Iy, Ix]
        mag_c = xp.abs(E_c)
        thr = 1e-6 * xp.abs(E_in).max()
        E_safe = xp.where(mag_c > thr, E_c, xp.full_like(E_c, thr))
        kx = xp.imag(gx[Iy, Ix] / E_safe) / kmag
        ky = xp.imag(gy[Iy, Ix] / E_safe) / kmag
        # Fall back to axial where |E| is negligible or the local angle
        # leaves the light cone (aliased / evanescent) -- keeps N real.
        weak = (mag_c <= thr) | (kx * kx + ky * ky >= 1.0)
        zero = xp.zeros_like(kx)
        L = xp.where(weak, zero, kx).astype(x_b.dtype)
        M = xp.where(weak, zero, ky).astype(x_b.dtype)
        N = xp.sqrt(xp.maximum(1.0 - L * L - M * M, xp.zeros_like(L)))
    else:
        L = xp.zeros((n,), dtype=x_b.dtype)
        M = xp.zeros((n,), dtype=x_b.dtype)
        N = xp.ones((n,), dtype=x_b.dtype)
    directions = xp.stack([L, M, N], axis=-1)

    _cdt = xp.complex128 if hasattr(xp, 'complex128') else 'complex128'
    wx = waist_factor * dx
    wy = waist_factor * dy
    sample = E_in[Iy, Ix]
    if _aniso:
        # Elliptical beamlet -> diagonal tensor Q = diag(-i/zRx, -i/zRy).
        zRx = float(np.pi) * (wx ** 2) / wavelength
        zRy = float(np.pi) * (wy ** 2) / wavelength
        diag = xp.stack([xp.full((n,), -1j / zRx, dtype=_cdt),
                         xp.full((n,), -1j / zRy, dtype=_cdt)], axis=-1)
        Q = diag[:, :, None] * xp.eye(2, dtype=_cdt)[None, :, :]  # (n,2,2)
        waist0 = xp.full((n,), float(np.sqrt(wx * wy)), dtype=x_b.dtype)
        pixel_area = (sample_step ** 2) * dx * dy
        amplitude = sample * pixel_area / (float(np.pi) * wx * wy)
    else:
        w0 = wx
        z_R = float(np.pi) * (w0 ** 2) / wavelength
        Q = xp.full((n,), -1j / z_R, dtype=_cdt)
        waist0 = xp.full((n,), float(w0), dtype=x_b.dtype)
        pixel_area = (sample_step * dx) ** 2
        amplitude = sample * pixel_area / (float(np.pi) * w0 * w0)

    return BeamletBundle(
        positions=positions,
        directions=directions,
        Q=Q,
        amplitude=amplitude.astype(Q.dtype),
        waist0=waist0,
    )


def recommend_gbd_sampling(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    target_overlap: float = 1.5,
    oversample: float = 4.0,
) -> Dict[str, Any]:
    """Recommend ``sample_step`` and ``waist_factor`` for a source field.

    Picks the beamlet grid automatically instead of leaving ``sample_step``
    and ``waist_factor`` as blind manual knobs.  The spacing must resolve BOTH
    the field's amplitude structure and its local wavevector (phase gradient),
    so the beamlet grid is chosen as the finer of:

    * an **amplitude** feature scale ``||E|| / || grad E ||`` (L2), and
    * a **phase / angular** Nyquist from the max local tilt
      ``k_local = grad(arg E)`` (a beamlet spacing that resolves the tilt
      variation) --

    downsampled by ``oversample`` beamlets per feature, clamped to
    ``[1, N/4]``.  ``waist_factor`` is then set to ``target_overlap *
    sample_step`` so neighbouring beamlets overlap (``w0 = target_overlap *
    spacing``); ``target_overlap ~ 1.5`` is a good smooth-field default.

    Returns
    -------
    dict
        ``{'sample_step': int, 'waist_factor': float, 'n_beamlets': int}``
        -- splat into ``decompose_field_to_beamlets`` /
        ``propagate_gbd_*`` (``**recommend_gbd_sampling(...)`` minus
        ``n_beamlets``).
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    gx = xp.gradient(E_in, dx, axis=-1)
    gy = xp.gradient(E_in, dx, axis=-2)
    amp = xp.abs(E_in)
    norm_E = float(xp.sqrt(xp.sum(amp ** 2)))
    grad_mag = xp.sqrt(xp.abs(gx) ** 2 + xp.abs(gy) ** 2)
    norm_grad = float(xp.sqrt(xp.sum(grad_mag ** 2)))
    # Amplitude feature scale (m); guard the smooth/flat field (grad ~ 0).
    if norm_grad > 0.0:
        feat_amp = norm_E / norm_grad
    else:
        feat_amp = float(Nx * dx)
    # Phase / angular feature: max local transverse spatial frequency of the
    # tilt.  k_local = Im(grad E / E) [rad/m]; the beamlet spacing must be
    # small vs 1/max|k_local| (in cycles) so the launch direction is locally
    # constant.
    thr = 1e-6 * float(xp.max(amp)) if amp.size else 0.0
    safe = xp.where(amp > thr, E_in, xp.full_like(E_in, thr + 1e-30))
    kx = xp.imag(gx / safe)
    ky = xp.imag(gy / safe)
    kloc_max = float(xp.max(xp.sqrt(kx ** 2 + ky ** 2)))
    feat_phase = (2.0 * float(np.pi) / kloc_max) if kloc_max > 0.0 else feat_amp
    feature = min(feat_amp, feat_phase)
    s_px = feature / (oversample * dx)
    sample_step = int(min(max(1, round(s_px)), max(1, min(Ny, Nx) // 4)))
    waist_factor = float(target_overlap * sample_step)
    n_beamlets = ((Ny + sample_step - 1) // sample_step) * \
                 ((Nx + sample_step - 1) // sample_step)
    return {
        'sample_step': sample_step,
        'waist_factor': waist_factor,
        'n_beamlets': int(n_beamlets),
    }


# ============================================================================
# ABCD evolution
# ============================================================================

def propagate_beamlets_freespace(
    beamlets: BeamletBundle,
    z_distance: float,
    wavelength: float,
    *,
    n_medium: float = 1.0,
) -> BeamletBundle:
    """Advance every beamlet by free-space distance ``z_distance``."""
    xp = array_namespace(beamlets.positions)

    Nz = beamlets.directions[..., 2]
    eps = 1e-30
    t = z_distance / xp.where(xp.abs(Nz) > eps, Nz, eps)

    new_positions = beamlets.positions + t[..., None] * beamlets.directions

    Q_old = beamlets.Q
    k = 2 * float(np.pi) / wavelength * n_medium
    # 4.9 fix (audit #2.2): use raw ``t`` (signed) instead of
    # ``abs(t)``.  Under the exp(-iωt) time convention forward
    # propagation by distance z imparts exp(+i·k·z) -- correct for
    # both signs of z.  Pre-4.9 ``abs(t)`` accidentally took the
    # complex conjugate of the axial phase on back-propagation,
    # giving wrong sign on the propagated wavefront.  Forward
    # propagation was unaffected (because abs(positive) == positive).
    axial_phase = xp.exp(1j * k * t)
    if _q_is_tensor(Q_old):
        # v5.21: tensor (astigmatic / anamorphic) Q free-space -- matrix
        # Mobius Q_new = Q (I + t Q)^{-1} with a per-eigenvalue branch-safe
        # amplitude prod_i 1/sqrt(1 + t lambda_i) (eigenvalues have Im<0 for a
        # physical beam, so each principal sqrt is continuous through a focus).
        I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        tt = t[:, None, None].astype(Q_old.dtype)
        Q_new = Q_old @ xp.linalg.inv(I2 + tt * Q_old)
        Q_new = 0.5 * (Q_new + xp.transpose(Q_new, (0, 2, 1)))
        lam = xp.linalg.eigvals(Q_old)
        qratio = xp.prod(
            1.0 / xp.sqrt(1.0 + t[:, None].astype(Q_old.dtype) * lam), axis=1)
    else:
        Q_new = Q_old / (1 + t.astype(Q_old.dtype) * Q_old)
        qratio = Q_new / Q_old
    new_amplitude = beamlets.amplitude * qratio * axial_phase.astype(Q_old.dtype)

    return BeamletBundle(
        positions=new_positions,
        directions=beamlets.directions,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def apply_thin_lens_to_beamlets(
    beamlets: BeamletBundle,
    focal_length: float,
    wavelength: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> BeamletBundle:
    """Apply an ideal thin lens to every beamlet."""
    xp = array_namespace(beamlets.positions)
    cx, cy = centre

    if _q_is_tensor(beamlets.Q):
        # v5.21: tensor Q -- an ideal (rotationally-symmetric) lens subtracts
        # 1/f from BOTH principal curvatures, i.e. (1/f) I2.
        I2 = xp.eye(2, dtype=beamlets.Q.dtype)[None, :, :]
        Q_new = beamlets.Q - (1.0 / focal_length) * I2
    else:
        Q_new = beamlets.Q - (1.0 / focal_length)

    x_off = beamlets.positions[..., 0] - cx
    y_off = beamlets.positions[..., 1] - cy
    L_old = beamlets.directions[..., 0]
    M_old = beamlets.directions[..., 1]
    N_old = beamlets.directions[..., 2]
    # 4.10.2: the thin-lens kick acts on PARAXIAL SLOPES u = L/N, not on
    # direction cosines.  Pre-4.10.2 subtracted x/f directly from L,
    # which is only correct in the small-angle limit (N -> 1).  For
    # moderately non-paraxial bundles (N ~ 0.95-0.99) this introduces
    # a few-percent error per surface; for wide-angle fans the error
    # compounds.  Convert to slope, apply the kick, re-normalise.
    N_safe = xp.where(xp.abs(N_old) > 1e-30, N_old, 1e-30)
    u_x_old = L_old / N_safe
    u_y_old = M_old / N_safe
    u_x_new = u_x_old - x_off / focal_length
    u_y_new = u_y_old - y_off / focal_length

    norm = xp.sqrt(u_x_new ** 2 + u_y_new ** 2 + 1.0)
    L_new = u_x_new / norm
    M_new = u_y_new / norm
    N_new = 1.0 / norm
    new_direction = xp.stack([L_new, M_new, N_new], axis=-1)

    k = 2 * float(np.pi) / wavelength
    lens_phase = xp.exp(-1j * k * (x_off * x_off + y_off * y_off) / (2 * focal_length))
    new_amplitude = beamlets.amplitude * lens_phase.astype(beamlets.amplitude.dtype)

    return BeamletBundle(
        positions=beamlets.positions,
        directions=new_direction,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def apply_aperture_to_beamlets(
    beamlets: BeamletBundle,
    semi_diameter: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
    shape: str = 'circular',
) -> BeamletBundle:
    """Vignette a beamlet bundle at an aperture stop of half-size
    ``semi_diameter``.

    Beamlets whose base-ray transverse position falls outside the aperture
    have their amplitude zeroed (geometric / chief-ray vignetting).  This is
    the *approximate* GBD aperture model: because each beamlet is a smooth
    Gaussian, a hard edge is resolved only to the beamlet-spacing scale (a
    beamlet straddling the rim is kept or dropped whole).  For hard-edged
    apertures where the diffraction ripple matters at finer than the beamlet
    pitch, use finer ``sample_step`` (more, narrower beamlets) or the HFPI
    propagator.  ``shape='circular'`` clips on ``sqrt(x^2+y^2)``;
    ``shape='rectangular'`` clips on ``max(|x|,|y|)`` (``semi_diameter`` is the
    half-width).

    Parameters
    ----------
    beamlets : BeamletBundle
    semi_diameter : float
        Aperture half-size (radius for circular, half-width for rectangular).
    centre : (float, float), optional
        Aperture centre in the transverse plane (m).
    shape : {'circular', 'rectangular'}
    """
    xp = array_namespace(beamlets.positions)
    cx, cy = centre
    x = beamlets.positions[..., 0] - cx
    y = beamlets.positions[..., 1] - cy
    if shape == 'rectangular':
        inside = (xp.abs(x) <= semi_diameter) & (xp.abs(y) <= semi_diameter)
    elif shape == 'circular':
        inside = (x * x + y * y) <= (semi_diameter * semi_diameter)
    else:
        raise ValueError(
            f"apply_aperture_to_beamlets: shape must be 'circular' or "
            f"'rectangular', got {shape!r}")
    mask = inside.astype(beamlets.amplitude.dtype)
    new_amplitude = beamlets.amplitude * mask

    return BeamletBundle(
        positions=beamlets.positions,
        directions=beamlets.directions,
        Q=beamlets.Q,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


# ============================================================================
# Reconstruction
# ============================================================================

def reconstruct_field_from_beamlets(
    beamlets: BeamletBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    wavelength: float,
    dy: Optional[float] = None,
    chunk_beamlets: int = 4096,
) -> np.ndarray:
    """Coherently sum every beamlet's transverse profile on a 2-D
    output grid.  ``dy`` (default ``dx``) sets the y-axis output pitch for an
    anamorphic grid."""
    xp = array_namespace(beamlets.positions)
    cx, cy = centre
    if dy is None:
        dy = dx

    ix = xp.arange(Nx, dtype=beamlets.positions.dtype)
    iy = xp.arange(Ny, dtype=beamlets.positions.dtype)
    Xg, Yg = xp.meshgrid((ix - Nx / 2) * dx + cx,
                         (iy - Ny / 2) * dy + cy,
                         indexing='xy')

    k = 2 * float(np.pi) / wavelength
    out = xp.zeros((Ny, Nx), dtype=beamlets.amplitude.dtype)

    n = int(beamlets.positions.shape[0])
    # Per-beamlet direction cosines (paraxial tilt).  These produce a
    # linear phase ramp `exp(i k (L dx + M dy))` from each beamlet's
    # centroid -- needed for non-paraxial bundles to interfere
    # correctly off-chief-ray.  Pre-4.10 omitted this ramp; the
    # focal spot still focused correctly (the chief-ray phase is the
    # same), but off-chief-ray interference patterns and PSF wings
    # were degraded.  When the beamlets bundle was assembled with
    # ``directions = (0, 0)`` (the default for axial-input decompositions)
    # the ramp is zero so this fix is a no-op for that path.
    has_dirs = (hasattr(beamlets, 'directions')
                and beamlets.directions is not None)
    # v5.21 (per-surface GBD): a beamlet bundle may carry a (N, 2, 2) complex-
    # symmetric TENSOR Q (general astigmatic Gaussian beam) instead of the (N,)
    # scalar Q.  Detected once here; the scalar branch below is preserved
    # verbatim (byte-identical) and only the tensor case takes the new quadratic
    # form.  ``_q_is_tensor`` is the single dispatch predicate.
    is_tensor = _q_is_tensor(beamlets.Q)
    # v4.13.1 perf: fuse the two ``xp.exp`` calls into one (only on the
    # has_dirs branch, where pre-v4.13.1 evaluated ``exp(-i*k*Q*rho2/2)``
    # and ``exp(i*k*tilt)`` separately and multiplied them).  ``exp(A) *
    # exp(B) == exp(A + B)`` analytically; in complex128 the round-off
    # difference is ulp-level (<1e-15 relative) -- well within the
    # propagator accuracy budget.  This roughly halves the per-chunk
    # transcendental cost (exp dominates the inner-loop runtime on
    # moderate grids).  Also switches the per-chunk reduction from
    # ``out + sum(a_b * phase, axis=-1)`` to ``out += einsum('mnk,k->mn',
    # phase, a_b)`` -- the ``a_b * phase`` intermediate is the
    # largest 3-D buffer the loop allocates (chunk * Ny * Nx complex),
    # so dropping it shrinks the working set noticeably for the
    # default chunk_beamlets=4096 and saves one big allocation per
    # chunk on numpy.
    for start in range(0, n, chunk_beamlets):
        end = min(start + chunk_beamlets, n)
        x_b = beamlets.positions[start:end, 0]
        y_b = beamlets.positions[start:end, 1]
        Q_b = beamlets.Q[start:end]
        a_b = beamlets.amplitude[start:end]

        dX = Xg[..., None] - x_b[None, None, :]
        dY = Yg[..., None] - y_b[None, None, :]
        rho2 = dX * dX + dY * dY
        if is_tensor:
            # General astigmatic Gaussian: transverse curvature is the
            # quadratic form 0.5 * conj(Q) : (rho rho^T) with the F-1 conj
            # (gbd.py physics convention) applied ELEMENTWISE to every Q
            # component, incl. the off-diagonal Qxy -- dropping conj on Qxy
            # would silently conjugate only the skew-astigmatism phase (an
            # intensity-invisible bug).  Q is symmetric so Qxy == Qyx.
            Qxx = Q_b[:, 0, 0]
            Qyy = Q_b[:, 1, 1]
            Qxy = Q_b[:, 0, 1]
            L_b = beamlets.directions[start:end, 0]
            M_b = beamlets.directions[start:end, 1]
            arg = (0.5 * (xp.conj(Qxx)[None, None, :] * (dX * dX)
                          + xp.conj(Qyy)[None, None, :] * (dY * dY)
                          + 2.0 * xp.conj(Qxy)[None, None, :] * (dX * dY))
                   + L_b[None, None, :] * dX + M_b[None, None, :] * dY)
            phase = xp.exp(1j * k * arg)
        elif has_dirs:
            L_b = beamlets.directions[start:end, 0]
            M_b = beamlets.directions[start:end, 1]
            # Fused phase argument.  v5.4.6 (audit F-1): the stored Q uses
            # the engineering 1/q parameterisation (q_code = conj(q_physics));
            # the reconstructed FIELD must be expressed in the library's
            # exp(-i omega t) / forward exp(+ikz) convention, i.e. the
            # transverse curvature is exp(+i k rho^2 / (2 q_physics)) =
            # exp(+0.5j k conj(Q) rho^2).  The pre-fix ``-0.5*Q`` rendered the
            # complex CONJUGATE of the propagated wavefront curvature (the
            # |E| envelope and waist are sign-blind, so intensity/focus tests
            # never caught it).  conj(Q) has the same Im part, so the Gaussian
            # decay and the on-axis-waist (Re(Q)=0) reconstruction are
            # unchanged; only the off-waist phase sign is corrected.
            arg = (0.5 * xp.conj(Q_b[None, None, :]) * rho2
                   + L_b[None, None, :] * dX + M_b[None, None, :] * dY)
            phase = xp.exp(1j * k * arg)
        else:
            phase = xp.exp(0.5j * k * xp.conj(Q_b[None, None, :]) * rho2)
        # ``out += einsum('mnk,k->mn', phase, a_b)`` if numpy; the
        # operator is equivalent to ``sum(a_b * phase, axis=-1)``
        # but avoids the (Ny, Nx, chunk) ``a_b * phase`` intermediate.
        # JAX / CuPy also have einsum; fall back to the original
        # pattern when einsum is unavailable.  In-place ``+=`` on
        # numpy / cupy avoids the per-chunk (Ny, Nx) allocation that
        # ``out = out + ...`` would create; JAX arrays are immutable
        # so we keep the rebind form for that backend.
        einsum = getattr(xp, 'einsum', None)
        if einsum is not None:
            contrib_sum = einsum('mnk,k->mn', phase, a_b)
        else:
            contrib = a_b[None, None, :] * phase
            contrib_sum = xp.sum(contrib, axis=-1)
        if is_jax_array(out):
            out = out + contrib_sum
        else:
            out += contrib_sum

    return out


# ============================================================================
# End-to-end convenience
# ============================================================================

def propagate_gbd(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    **kwargs: Any,
) -> np.ndarray:
    """Canonical-order GBD free-space propagation.

    Argument order ``(E_in, z, wavelength, dx)`` matches
    :func:`angular_spectrum_propagate` and
    :func:`propagate_huygens_fresnel`.  This is the recommended entry
    point for new code.  Internally delegates to
    :func:`propagate_gbd_freespace` (which retains its legacy
    ``(E_in, dx, *, z, wavelength, ...)`` order for backwards
    compatibility).
    """
    return propagate_gbd_freespace(
        E_in, dx, z=z, wavelength=wavelength, **kwargs)


def propagate_gbd_freespace(
    E_in: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    dy: Optional[float] = None,
    output_dy: Optional[float] = None,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
    direction_sampling: bool = False,
) -> np.ndarray:
    """End-to-end free-space GBD: source -> z -> output.

    ``direction_sampling=True`` selects the Husimi (position + direction)
    decomposition -- needed for a source that is already tilted / diverging
    at the input plane; see :func:`decompose_field_to_beamlets`.

    ``dy`` (source y-pitch, default ``dx``) and ``output_dy`` (output y-pitch,
    default ``output_dx``) support **anamorphic** grids: ``dy != dx`` makes each
    beamlet an axis-aligned ellipse (a diagonal tensor ``Q``).

    .. note::
       This function uses a non-canonical argument order
       ``(E_in, dx, *, z, wavelength, ...)``.  Prefer
       :func:`propagate_gbd` for the canonical
       ``(E_in, z, wavelength, dx)`` order shared with
       :func:`angular_spectrum_propagate` and
       :func:`propagate_huygens_fresnel`.

    .. versionchanged:: 5.2
        ``output_grid`` was renamed to ``output_shape`` for the
        ``(Ny, Nx)`` shape-only meaning -- the dispatcher's
        ``propagate(output_grid=...)`` is canonical for the
        ``(N_out, dx_out)`` grid spec (AUDIT_V4_13_1 Part 2 P1-A).
        The legacy ``output_grid`` form still works but emits a
        ``DeprecationWarning``.
    """
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_freespace',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx
    if output_dy is None:
        output_dy = dy if dy is not None else output_dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength, dy=dy,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z,
                                          wavelength=wavelength)
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx, dy=output_dy,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets,
    )


def propagate_gbd_freespace_spectral(
    E_in: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelengths: Any,
    weights: Optional[Any] = None,
    combine: str = 'stack',
    **freespace_kwargs: Any,
) -> np.ndarray:
    """Polychromatic free-space GBD: propagate ``E_in`` at each wavelength.

    Because the beamlet complex parameter ``Q = -i/z_R``, its Rayleigh range
    ``z_R = pi w0^2 / lambda`` and the reconstruction phase ``k`` are all
    wavelength-dependent, each wavelength is decomposed and propagated
    independently and the results are combined.

    Parameters
    ----------
    wavelengths : array-like of float
        The wavelengths (m) to propagate.
    weights : array-like, optional
        Per-wavelength spectral weights (e.g. source SED); defaults to equal
        weights.  Used only by ``combine='intensity'``.
    combine : {'stack', 'intensity'}
        ``'stack'`` returns the per-wavelength complex fields stacked as
        ``(n_lambda, Ny, Nx)``.  ``'intensity'`` returns the weighted
        **incoherent** sum ``sum_l w_l |E_l|^2`` (real ``(Ny, Nx)``) -- the
        broadband PSF / image-plane intensity.
    **freespace_kwargs
        Forwarded to :func:`propagate_gbd_freespace` (``output_shape``,
        ``output_dx``, ``waist_factor``, ``sample_step``, ``chunk_beamlets``,
        ``direction_sampling``, ...).

    Returns
    -------
    array
        ``(n_lambda, Ny, Nx)`` complex for ``combine='stack'`` or
        ``(Ny, Nx)`` real for ``combine='intensity'``.
    """
    xp = array_namespace(E_in)
    lams = [float(_l) for _l in wavelengths]
    if not lams:
        raise ValueError(
            "propagate_gbd_freespace_spectral: ``wavelengths`` is empty.")
    fields = [
        propagate_gbd_freespace(E_in, dx, z=z, wavelength=_l,
                                **freespace_kwargs)
        for _l in lams
    ]
    if combine == 'stack':
        return xp.stack(fields, axis=0)
    if combine == 'intensity':
        if weights is None:
            w = [1.0] * len(lams)
        else:
            w = [float(_w) for _w in weights]
            if len(w) != len(lams):
                raise ValueError(
                    "propagate_gbd_freespace_spectral: ``weights`` length "
                    f"{len(w)} != ``wavelengths`` length {len(lams)}.")
        acc = xp.zeros(fields[0].shape, dtype=xp.abs(fields[0]).dtype)
        for _w, _F in zip(w, fields):
            acc = acc + _w * xp.abs(_F) ** 2
        return acc
    raise ValueError(
        f"propagate_gbd_freespace_spectral: combine must be 'stack' or "
        f"'intensity', got {combine!r}")


def propagate_gbd_freespace_vector(
    E_vec: np.ndarray,
    dx: float,
    *,
    z: float,
    wavelength: float,
    **freespace_kwargs: Any,
) -> np.ndarray:
    """Vector (polarized) free-space GBD.

    Propagates a two-component Jones field.  In free space -- and through
    ideal, **polarization-preserving** optics -- the two transverse
    polarization components ``E_x`` and ``E_y`` propagate independently (the
    paraxial beamlet geometry is polarization-agnostic), so each is decomposed
    and reconstructed on its own and the vector field is reassembled.

    .. note::
       This free-space helper models only polarization-**preserving**
       propagation (independent components).  For **polarization-changing**
       transmission through a real prescription -- per-surface Fresnel s/p
       diattenuation along each base ray -- use
       :func:`propagate_gbd_vector_through_prescription`.  Reflection at
       surfaces, waveplates, and thin-film coatings build on the same base-ray
       trace and remain future extensions.

    Parameters
    ----------
    E_vec : array ``(2, Ny, Nx)`` complex
        The ``(E_x, E_y)`` Jones components of the source field.
    dx, z, wavelength : see :func:`propagate_gbd_freespace`.
    **freespace_kwargs
        Forwarded to :func:`propagate_gbd_freespace` for each component.

    Returns
    -------
    array ``(2, Ny, Nx)`` complex
        The propagated ``(E_x, E_y)`` Jones field.
    """
    xp = array_namespace(E_vec)
    E_vec = xp.asarray(E_vec)
    if E_vec.shape[0] != 2 or E_vec.ndim != 3:
        raise ValueError(
            "propagate_gbd_freespace_vector: E_vec must be (2, Ny, Nx) "
            f"(the E_x, E_y Jones components); got shape {E_vec.shape}.")
    comps = [
        propagate_gbd_freespace(E_vec[c], dx, z=z, wavelength=wavelength,
                                **freespace_kwargs)
        for c in range(2)
    ]
    return xp.stack(comps, axis=0)


def _fresnel_jones_matrix_per_beamlet(x, y, ux, uy, prescription, wavelength):
    """Per-beamlet 2x2 transverse Jones matrix ``(E_x, E_y)_in -> _out`` from
    per-surface Fresnel s/p transmission along each base ray (polarization ray
    tracing).  Traces surface-by-surface, and at each refracting surface splits
    the field into s (perp to plane of incidence) and p components, scaling by
    the Fresnel amplitude transmission ``t_s``, ``t_p`` and rotating the p axis
    from the incident to the refracted ray.  Transmission only (no reflection /
    thin-film coatings).

    Convention: the returned matrix is the **transverse** ``(E_x, E_y)`` field,
    so the s channel (always transverse) is Fresnel-exact at every angle, while
    the p channel carries the honest ``cos(theta_out)`` projection of the tilted
    output ray's p-vector (its small longitudinal component is not part of the
    transverse field).  Input rays should be launched **axially** (``ux = uy =
    0``), as they are in :func:`propagate_gbd_vector_through_prescription`, so
    that ``(x, y)`` *is* the input transverse frame; the incidence angles then
    come from the surface curvature/tilt, not from a pre-tilted input ray.

    Returns ``(P, alive)``: ``P`` is ``(N, 2, 2)`` complex; ``alive`` is
    ``(N,)`` bool (False where the base ray vignetted / TIR'd).  ``P`` is zeroed
    for dead rays so ``P @ E`` never propagates NaNs.
    """
    from ..glass import get_glass_index
    from ..raytrace.intersection import _intersect_surface, _refract, _surface_normal, _transfer
    from ..raytrace.trace import _make_bundle, surfaces_from_prescription

    surfs = surfaces_from_prescription(prescription)
    inv = 1.0 / np.sqrt(1.0 + ux ** 2 + uy ** 2)
    L0, M0 = ux * inv, uy * inv
    n = x.shape[0]

    def _trace_jones(J0):
        rb = _make_bundle(x.copy(), y.copy(), L0.copy(), M0.copy(), wavelength)
        J = J0.astype(np.complex128).copy()
        for s in surfs:
            _intersect_surface(rb, s)
            nx, ny, nz = _surface_normal(rb.x, rb.y, s)
            d_in = np.stack([rb.L, rb.M, rb.N], axis=-1)
            nrm = np.stack([nx, ny, nz], axis=-1)
            cosd = np.sum(d_in * nrm, axis=-1)
            nrm = np.where(cosd[:, None] > 0, -nrm, nrm)
            cos_i = np.abs(cosd)
            n1 = float(get_glass_index(
                getattr(s, 'glass_before', None) or 'air', wavelength))
            n2 = float(get_glass_index(
                getattr(s, 'glass_after', None) or 'air', wavelength))
            _refract(rb, s, n1, n2)
            d_out = np.stack([rb.L, rb.M, rb.N], axis=-1)
            mu = n1 / n2
            cos_t = np.sqrt(np.maximum(1.0 - mu ** 2 * (1.0 - cos_i ** 2), 0.0))
            ts = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
            tp = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
            s_vec = np.cross(d_in, nrm)
            s_norm = np.linalg.norm(s_vec, axis=-1, keepdims=True)
            at_normal = s_norm[:, 0] < 1e-9
            s_hat = np.where(s_norm > 1e-30, s_vec / np.where(s_norm > 0, s_norm, 1.0),
                             np.array([1.0, 0.0, 0.0]))
            p_in = np.cross(d_in, s_hat)
            p_out = np.cross(d_out, s_hat)
            Es = np.sum(J * s_hat, axis=-1)
            Ep = np.sum(J * p_in, axis=-1)
            J_sp = (Es * ts)[:, None] * s_hat + (Ep * tp)[:, None] * p_out
            # at normal incidence s/p is degenerate but ts == tp -> just scale
            J = np.where(at_normal[:, None], ts[:, None] * J, J_sp)
            _transfer(rb, float(getattr(s, 'thickness', 0.0) or 0.0), n2)
        # project onto the output transverse (x, y) plane (paraxial output ray)
        return J[:, 0], J[:, 1], rb.alive

    ex_x, ex_y, alive = _trace_jones(
        np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)))
    ey_x, ey_y, _ = _trace_jones(np.tile(np.array([0.0, 1.0, 0.0]), (n, 1)))
    P = np.empty((n, 2, 2), dtype=np.complex128)
    P[:, 0, 0] = ex_x
    P[:, 1, 0] = ex_y
    P[:, 0, 1] = ey_x
    P[:, 1, 1] = ey_y
    # zero out dead / non-finite rays so ``P @ E`` never propagates NaNs into
    # live neighbouring beamlets (a vignetted base ray can leave NaN in P).
    alive = np.asarray(alive, bool) & np.isfinite(P).all(axis=(1, 2))
    P[~alive] = 0.0
    return P, alive


def propagate_gbd_vector_through_prescription(
    E_vec: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    per_surface: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Vector (Jones) GBD through a prescription with **polarization ray
    tracing** -- per-surface Fresnel s/p diattenuation along each beamlet's
    base ray, mixing the ``(E_x, E_y)`` components.

    Combines the scalar/tensor envelope propagation
    (:func:`propagate_gbd_through_prescription`, ``per_surface`` selecting the
    aberration-aware form) with the per-beamlet Fresnel Jones matrix from
    :func:`_fresnel_jones_matrix_per_beamlet`: the diattenuation modifies the
    beamlet amplitudes, then each transformed component is reconstructed.

    Transmission only (no reflection / thin-film coatings -- those build on the
    same base-ray trace and are a documented extension).

    Parameters
    ----------
    E_vec : array ``(2, Ny, Nx)`` complex -- the ``(E_x, E_y)`` Jones field.
    dx, prescription, wavelength : as elsewhere.
    per_surface : bool -- passed to the envelope propagator.
    **kwargs : forwarded to :func:`propagate_gbd_through_prescription`.

    Returns
    -------
    array ``(2, Ny, Nx)`` complex -- the ``(E_x, E_y)`` output Jones field.
    """
    E_vec = np.asarray(E_vec)
    if E_vec.shape[0] != 2 or E_vec.ndim != 3:
        raise ValueError(
            "propagate_gbd_vector_through_prescription: E_vec must be "
            f"(2, Ny, Nx); got {E_vec.shape}.")
    # per-beamlet Fresnel Jones matrix from the base-ray grid (axial launch)
    Ny, Nx = E_vec.shape[-2], E_vec.shape[-1]
    sample_step = kwargs.get('sample_step', 1)
    iy = np.arange(0, Ny, sample_step)
    ix = np.arange(0, Nx, sample_step)
    Iy, Ix = np.meshgrid(iy, ix, indexing='ij')
    xb = (Ix.ravel() - Nx / 2) * dx
    yb = (Iy.ravel() - Ny / 2) * dx
    zc = np.zeros_like(xb)
    P, _alive = _fresnel_jones_matrix_per_beamlet(
        xb, yb, zc, zc, prescription, wavelength)
    # apply P to the (E_x, E_y) samples, then envelope-propagate each mixed
    # component and reconstruct.
    ExS = E_vec[0][Iy, Ix].ravel()
    EyS = E_vec[1][Iy, Ix].ravel()
    ExM = (P[:, 0, 0] * ExS + P[:, 0, 1] * EyS).reshape(Iy.shape)
    EyM = (P[:, 1, 0] * ExS + P[:, 1, 1] * EyS).reshape(Iy.shape)
    # scatter the mixed samples back onto full grids (per component) so the
    # existing decompose sub-samples them consistently.
    ExF = np.zeros((Ny, Nx), dtype=np.complex128)
    EyF = np.zeros((Ny, Nx), dtype=np.complex128)
    ExF[Iy, Ix] = ExM
    EyF[Iy, Ix] = EyM
    kw = dict(kwargs)
    kw['sample_step'] = sample_step
    out_x = propagate_gbd_through_prescription(
        ExF, dx, prescription, wavelength=wavelength, per_surface=per_surface,
        **kw)
    out_y = propagate_gbd_through_prescription(
        EyF, dx, prescription, wavelength=wavelength, per_surface=per_surface,
        **kw)
    return np.stack([out_x, out_y], axis=0)


def propagate_gbd_thin_lens(
    E_in: np.ndarray,
    dx: float,
    *,
    z_to_lens: float,
    focal_length: float,
    z_lens_to_output: float,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    lens_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
    direction_sampling: bool = False,
    aperture_semi_diameter: Optional[float] = None,
    aperture_shape: str = 'circular',
) -> np.ndarray:
    """End-to-end three-leg GBD: source -> free space -> thin lens
    -> free space -> output (the canonical GBD validation case).

    ``direction_sampling=True`` selects the Husimi (position + direction)
    decomposition so a source already tilted / diverging at the input plane
    walks off correctly through the system; see
    :func:`decompose_field_to_beamlets`.

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A); see :func:`propagate_gbd_freespace`.
    """
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_thin_lens',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z_to_lens,
                                          wavelength=wavelength)
    if aperture_semi_diameter is not None:
        bundle = apply_aperture_to_beamlets(
            bundle, aperture_semi_diameter, centre=lens_centre,
            shape=aperture_shape)
    bundle = apply_thin_lens_to_beamlets(bundle, focal_length=focal_length,
                                         wavelength=wavelength,
                                         centre=lens_centre)
    bundle = propagate_beamlets_freespace(bundle, z_distance=z_lens_to_output,
                                          wavelength=wavelength)
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets,
    )


# ============================================================================
# Prescription-aware GBD via system ABCD
# ============================================================================

def apply_abcd_to_beamlets(
    beamlets: BeamletBundle,
    A: float,
    B: float,
    C: float,
    D: float,
    wavelength: float,
    axial_opl: Optional[float] = None,
) -> BeamletBundle:
    """Apply a paraxial 2x2 ABCD matrix to every beamlet.

    Each beamlet's complex Q-parameter (= 1/q) transforms as

        Q_out = (C + D Q_in) / (A + B Q_in)

    and its base ray's transverse offset / slope transform as

        x_out = A x_in + B u_in
        u_out = C x_in + D u_in

    where ``u`` is the ray slope (paraxial direction cosine).  This
    is a paraxial approximation suitable for propagation through a
    sequential refractive system characterised by its system ABCD;
    see :func:`lumenairy.raytrace.system_abcd_prescription`.

    Parameters
    ----------
    beamlets : BeamletBundle
    A, B, C, D : float
        ABCD matrix elements.
    wavelength : float
        Vacuum wavelength.
    """
    xp = array_namespace(beamlets.positions)

    # Q evolution.
    Q_old = beamlets.Q
    if _q_is_tensor(Q_old):
        # v5.21: tensor Q -- generalized (matrix) Collins with scalar blocks
        # A,B,C,D -> A I2 etc.
        _I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        Q_new = (C * _I2 + D * Q_old) @ xp.linalg.inv(A * _I2 + B * Q_old)
        Q_new = 0.5 * (Q_new + xp.transpose(Q_new, (0, 2, 1)))
    else:
        Q_new = (C + D * Q_old) / (A + B * Q_old)

    # Base-ray paraxial transform: ray height x and slope u.
    x_in = beamlets.positions[..., 0]
    y_in = beamlets.positions[..., 1]
    L_in = beamlets.directions[..., 0]
    M_in = beamlets.directions[..., 1]
    N_in = beamlets.directions[..., 2]
    u_x = L_in / xp.where(xp.abs(N_in) > 1e-30, N_in, 1e-30)
    u_y = M_in / xp.where(xp.abs(N_in) > 1e-30, N_in, 1e-30)

    x_out = A * x_in + B * u_x
    y_out = A * y_in + B * u_y
    u_x_out = C * x_in + D * u_x
    u_y_out = C * y_in + D * u_y

    # Re-normalise direction (paraxial slope -> direction cosines).
    norm = xp.sqrt(u_x_out ** 2 + u_y_out ** 2 + 1.0)
    L_out = u_x_out / norm
    M_out = u_y_out / norm
    N_out = 1.0 / norm
    new_directions = xp.stack([L_out, M_out, N_out], axis=-1)
    z_out = beamlets.positions[..., 2]
    new_positions = xp.stack([x_out, y_out, z_out], axis=-1)

    # Amplitude correction: Collins/Siegman on-axis Gaussian-beam factor
    #   u_out(0) = u_in(0) / (A + B/q_in) = u_in(0) / (A + B*Q_in)
    # (Siegman ch. 20 / Collins integral; both transverse dimensions of
    # the rotationally-symmetric beamlet give 1/sqrt each).
    #
    # v5.17.x (P2-30) BEHAVIOUR CHANGE: pre-fix this applied
    # ``Q_new / Q_old`` = ``q_in / q_out`` = (C*q_in + D)/(A + B*Q_in),
    # i.e. the Collins factor times a spurious ``(C*q_in + D)``.  For any
    # focusing system (C != 0) that factor has non-unit modulus, so the
    # single-ABCD path disagreed with the sequential per-leg path
    # (``propagate_beamlets_freespace`` + ``apply_thin_lens_to_beamlets``
    # compose exactly to exp(ikL)/(A + B*Q_in), verified to 1e-15) in
    # both amplitude and piston phase -- e.g. a t1=20mm -> f=50mm ->
    # t2=30mm system came out |C*q_in+D| = 0.60x low in field amplitude
    # (0.36x in intensity) with a wrong piston, defeating the v4.11.1
    # axial_opl coherent-superposition fix.  Free-space-only ABCD (C=0,
    # D=1) is unchanged.  ``Q`` here is the module's engineering 1/q
    # parameterisation (Q = 1/q_code, q_code = conj(q_physics)); ABCD
    # elements are real so the Collins factor commutes with that
    # convention.
    if _q_is_tensor(Q_old):
        _I2 = xp.eye(2, dtype=Q_old.dtype)[None, :, :]
        qratio = 1.0 / xp.sqrt(xp.linalg.det(A * _I2 + B * Q_old))
    else:
        qratio = 1.0 / (A + B * Q_old)
    # 4.10.2: include the chief-ray axial OPL phase exp(+i*k*L_chief)
    # when supplied.  The three-leg helpers (propagate_gbd_freespace,
    # propagate_gbd_thin_lens) accumulate this leg-wise; the single-
    # ABCD path doesn't see L_chief unless the caller passes it
    # explicitly.  Missing this factor is a constant piston that
    # only matters when interfering the GBD output with another
    # (separately-propagated) reference arm.
    if axial_opl is not None:
        k0 = 2.0 * np.pi / wavelength
        axial_phase = xp.exp(1j * k0 * float(axial_opl))
        qratio = qratio * axial_phase
    new_amplitude = beamlets.amplitude * qratio

    return BeamletBundle(
        positions=new_positions,
        directions=new_directions,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def apply_prescription_persurface_to_beamlets(
    beamlets: BeamletBundle,
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    z_image: Optional[float] = None,
) -> BeamletBundle:
    """Per-surface tensor-Q evolution of a beamlet bundle through a prescription.

    The per-element form of :func:`apply_abcd_to_beamlets` (which applies a
    single whole-system paraxial ABCD).  Each beamlet's complex parameter is
    evolved **surface by surface** via the real per-ray differential ray
    transfer (:func:`lumenairy.raytrace.ray_transfer_jacobian`), so the
    scalar ``Q`` promotes to a ``(N, 2, 2)`` complex-symmetric TENSOR that
    captures off-axis **astigmatism** and higher-order aberration the paraxial
    system ABCD cannot.  The beamlets are carried through the refracting
    surfaces (generalized Collins ``Q_out = (C + D Q)(A + B Q)^{-1}``, amplitude
    ``1/sqrt(det(A + B Q))`` accumulated per surface so the sqrt branch stays
    unambiguous, plus the base-ray OPL piston), then a branch-safe tensor
    free-space to the image plane.

    NumPy backend (the trace + finite-difference Jacobian are numpy); the
    returned bundle's tensor ``Q`` reconstructs via the tensor branch of
    :func:`reconstruct_field_from_beamlets`.  Dead beamlets (vignette / TIR /
    miss) are dropped.

    Parameters
    ----------
    beamlets : BeamletBundle
        Source-plane bundle (scalar or tensor Q).
    prescription : dict
    wavelength : float
    z_image : float, optional
        Axial distance from the last surface vertex to the output/image plane
        [m].  Defaults to the system back focal length
        (``system_abcd_prescription`` BFL).
    """
    import copy as _copy

    from ..raytrace import surfaces_from_prescription, system_abcd_prescription
    from ..raytrace.differential import ray_transfer_jacobian

    surfs = list(surfaces_from_prescription(prescription))
    if z_image is None:
        _res = system_abcd_prescription(prescription, wavelength)
        z_image = float(_res[2])
    # Trace to the LAST surface vertex (zero its transfer); the image-side
    # free-space is done branch-safe below (its single leg crosses the focus).
    surfs[-1] = _copy.copy(surfs[-1])
    try:
        surfs[-1].thickness = 0.0
    except (AttributeError, TypeError):
        pass

    pos = np.asarray(beamlets.positions)
    dr = np.asarray(beamlets.directions)
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    Nz = dr[:, 2]
    ux = (dr[:, 0] / Nz).astype(np.float64)
    uy = (dr[:, 1] / Nz).astype(np.float64)

    dt = ray_transfer_jacobian(x, y, ux, uy, surfs, wavelength,
                               per_surface=True)
    n = x.shape[0]
    I2 = np.eye(2)[None, :, :]
    Qsrc = np.asarray(beamlets.Q)
    Q = (Qsrc.astype(np.complex128) if _q_is_tensor(Qsrc)
         else Qsrc[:, None, None] * I2)
    amp = np.ones(n, dtype=np.complex128)
    k0 = 2.0 * np.pi / wavelength

    for k in range(dt.jacobian.shape[0]):
        J = dt.jacobian[k]
        A = J[:, 0:2, 0:2]
        B = J[:, 0:2, 2:4]
        C = J[:, 2:4, 0:2]
        D = J[:, 2:4, 2:4]
        ABQ = A + B @ Q
        amp = amp / np.sqrt(np.linalg.det(ABQ))
        Q = (C + D @ Q) @ np.linalg.inv(ABQ)
        Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    # base-ray OPL piston (input plane -> last vertex): where the wavefront
    # aberration enters (different beamlets accumulate different OPL).
    amp = amp * np.exp(1j * k0 * dt.opd)

    # branch-safe tensor free-space to the image plane (per-eigenvalue sqrt so
    # the focus-crossing Gouy phase is unambiguous; eigenvalues have Im<0).
    inv = 1.0 / np.sqrt(1.0 + dt.ux ** 2 + dt.uy ** 2)
    Lx = dt.ux * inv
    My = dt.uy * inv
    Nz2 = inv
    t = z_image / Nz2
    lam = np.linalg.eigvals(Q)
    amp = amp * np.prod(1.0 / np.sqrt(1.0 + t[:, None] * lam), axis=1)
    amp = amp * np.exp(1j * k0 * t)
    Q = Q @ np.linalg.inv(I2 + t[:, None, None] * Q)
    Q = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
    new_pos = np.stack([dt.x + t * Lx, dt.y + t * My,
                        np.zeros_like(dt.x)], axis=-1)
    new_dir = np.stack([Lx, My, Nz2], axis=-1)
    new_amp = np.asarray(beamlets.amplitude) * amp
    alive = np.asarray(dt.alive, dtype=bool)
    w0 = np.asarray(beamlets.waist0)
    return BeamletBundle(
        positions=new_pos[alive], directions=new_dir[alive], Q=Q[alive],
        amplitude=new_amp[alive], waist0=w0[alive])


def propagate_gbd_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
    direction_sampling: bool = False,
    per_surface: bool = False,
    z_image: Optional[float] = None,
) -> np.ndarray:
    """End-to-end GBD through a sequential lumenairy prescription
    via system ABCD evolution.

    Decomposes the source field into a regular grid of Gaussian
    beamlets, transforms each beamlet's complex Q-parameter and
    base ray by the prescription's paraxial system ABCD matrix,
    and coherently reconstructs the output field.

    Two forms, selected by ``per_surface``:

    * ``per_surface=False`` (default) -- the **paraxial** form: a single
      whole-system ABCD applied beamlet-by-beamlet (the Collins integral).
      Exact for well-corrected / paraxial systems; carries no aberration.
    * ``per_surface=True`` -- the **per-element** form: each beamlet's
      complex parameter is evolved surface-by-surface via the real per-ray
      differential ray transfer, promoting ``Q`` to a ``(N, 2, 2)`` tensor
      that captures off-axis **astigmatism** (tangential/sagittal focal
      separation growing ~field^2) and higher-order aberration.  NumPy
      backend; see :func:`apply_prescription_persurface_to_beamlets` and
      :func:`lumenairy.raytrace.ray_transfer_jacobian`.  ``z_image`` sets the
      last-vertex-to-output distance (defaults to the system BFL).

    ``direction_sampling=True`` selects the Husimi (position + direction)
    decomposition so a source already tilted / diverging at the input
    plane walks off correctly; see :func:`decompose_field_to_beamlets`.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
    wavelength : float
    output_grid, output_dx, output_centre : grid geometry
    waist_factor, sample_step, chunk_beamlets : decomposition tuning

    Returns
    -------
    array (Ny, Nx) complex
        Output-plane reconstructed field.

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A); see :func:`propagate_gbd_freespace`.
    """
    from ..raytrace import system_abcd_prescription

    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_gbd_through_prescription',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
        direction_sampling=direction_sampling,
    )

    if per_surface:
        # Per-element form: evolve each beamlet's tensor Q surface-by-surface
        # via the real per-ray differential ray transfer (captures off-axis
        # astigmatism / aberration the single whole-system ABCD cannot).
        bundle = apply_prescription_persurface_to_beamlets(
            bundle, prescription, wavelength, z_image=z_image)
        return reconstruct_field_from_beamlets(
            bundle, Ny=Ny, Nx=Nx, dx=output_dx,
            centre=output_centre, wavelength=wavelength,
            chunk_beamlets=chunk_beamlets,
        )

    # Get the system's paraxial ABCD matrix.  ``system_abcd_prescription``
    # returns ``(matrix, efl, bfl)``.
    abcd_result = system_abcd_prescription(prescription, wavelength)
    if isinstance(abcd_result, tuple):
        M = abcd_result[0]
    else:
        M = abcd_result
    A = float(M[0, 0])
    B = float(M[0, 1])
    C = float(M[1, 0])
    D = float(M[1, 1])

    # 4.11.1 (H-AS-1): compute the axial OPL = sum_k n_k * t_k across
    # every glass/air segment plus the BFL gap to the image plane.
    # Pre-4.11.1 ``axial_opl=`` was never populated so the per-beamlet
    # complex envelope lacked the system's axial phase reference and
    # multi-prescription reconstructions had the wrong piston relative
    # to ASM / Fresnel cross-checks.
    #
    # 4.11.2: the v4.11.1 implementation here was dead-on-arrival:
    # ``surfaces_from_prescription`` returns ``List[Surface]`` (Surface
    # is a @dataclass, not a dict), so ``_s.get('thickness', 0.0)``
    # raised AttributeError on the first iteration -- silently swallowed
    # by the surrounding bare ``except Exception`` and ``axial_opl``
    # always defaulted to None.  Switched to attribute access on the
    # Surface dataclass.  Caught by AUDIT_ROUND3_2026_05_16.md (CRIT-8).
    try:
        from ..glass import get_glass_index
        from ..raytrace import surfaces_from_prescription
        _surfs = surfaces_from_prescription(prescription)
        axial_opl = 0.0
        for _s in _surfs:
            _t = float(getattr(_s, 'thickness', 0.0) or 0.0)
            _glass = (getattr(_s, 'glass_after', None)
                      or getattr(_s, 'glass_before', None)
                      or 'air')
            try:
                _n = float(get_glass_index(_glass, wavelength))
            except (KeyError, ValueError, TypeError):
                # get_glass_index can raise: KeyError on unknown
                # catalogue / glass name, ValueError on
                # outside-Sellmeier-range wavelength, TypeError on
                # a malformed (non-string) glass identifier.  Fall
                # back to n=1.0 (air); this matches the v4.11.2
                # behaviour but no longer hides AttributeError /
                # ImportError from the broader try.
                _n = 1.0
            axial_opl += _n * _t
    except (ImportError, AttributeError, TypeError, ValueError,
            KeyError) as _exc:
        # Surface the failure rather than silently fall through to a
        # missing axial-phase reference; reconstruction still proceeds
        # without the piston.  Errors we expect from the inner block:
        # ImportError (raytrace / glass modules missing),
        # AttributeError (Surface dataclass missing expected field),
        # TypeError/ValueError (thickness/glass coercion failures),
        # KeyError (catalogue lookup re-raised).
        import warnings as _w
        _w.warn(
            f"propagate_gbd_through_prescription: axial-OPL "
            f"computation failed ({type(_exc).__name__}: {_exc}); "
            f"reconstructed field will lack the absolute axial-phase "
            f"reference and may not coherently superpose with other "
            f"propagator outputs.", RuntimeWarning, stacklevel=2)
        axial_opl = None

    # Apply ABCD to every beamlet.
    bundle = apply_abcd_to_beamlets(bundle, A, B, C, D,
                                     wavelength=wavelength,
                                     axial_opl=axial_opl)

    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets,
    )


__all__ = [
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'recommend_gbd_sampling',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'apply_aperture_to_beamlets',
    'apply_abcd_to_beamlets',
    'apply_prescription_persurface_to_beamlets',
    'reconstruct_field_from_beamlets',
    'propagate_gbd_freespace',
    'propagate_gbd_freespace_spectral',
    'propagate_gbd_freespace_vector',
    'propagate_gbd_vector_through_prescription',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',
]
