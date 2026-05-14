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

Limitations:

* **Smooth aperture handling** -- a Gaussian beamlet has continuous
  edges; HFPI handles hard cutoffs better.
* **Caustic-region accuracy** -- like all paraxial complex-ray
  methods, GBD's accuracy degrades near a caustic.

See ``REFERENCES.txt`` Section C for the foundational publications.

Multi-backend
-------------

Backend dispatched via :func:`lumenairy._array.array_namespace`.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


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
    Q: object               # (N,) complex (= 1/q)
    amplitude: object       # (N,) complex on-axis amplitude
    waist0: object          # (N,) initial waist (for profile)

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except Exception:
            return 0


# ============================================================================
# Source-plane decomposition
# ============================================================================

def decompose_field_to_beamlets(
    E_in: np.ndarray,
    dx: float,
    *,
    wavelength: float,
    waist_factor: float = 1.0,
    sample_step: int = 1,
    z_input_plane: float = 0.0,
) -> BeamletBundle:
    """Position-decomposition of a 2-D complex source field into a
    regular grid of Gaussian beamlets.

    Each grid pixel becomes a beamlet centred at its physical
    coordinate, with on-axis amplitude ``E_in[i, j]``, propagating
    along ``+z``, and Gaussian waist ``w0 = waist_factor * dx``.
    """
    xp = array_namespace(E_in)
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]

    iy = xp.arange(0, Ny, sample_step)
    ix = xp.arange(0, Nx, sample_step)
    Iy, Ix = xp.meshgrid(iy, ix, indexing='ij')
    Iy = Iy.reshape(-1)
    Ix = Ix.reshape(-1)
    n = Iy.shape[0]

    x_b = (Ix - Nx / 2 + 0.5) * dx
    y_b = (Iy - Ny / 2 + 0.5) * dx
    z_b = xp.full((n,), float(z_input_plane), dtype=x_b.dtype)
    positions = xp.stack([x_b, y_b, z_b], axis=-1)

    L = xp.zeros((n,), dtype=x_b.dtype)
    M = xp.zeros((n,), dtype=x_b.dtype)
    N = xp.ones((n,), dtype=x_b.dtype)
    directions = xp.stack([L, M, N], axis=-1)

    w0 = waist_factor * dx
    z_R = float(np.pi) * (w0 ** 2) / wavelength
    Q = xp.full((n,), -1j / z_R,
                dtype=xp.complex128 if hasattr(xp, 'complex128') else 'complex128')
    waist0 = xp.full((n,), float(w0), dtype=x_b.dtype)

    sample = E_in[Iy, Ix]
    pixel_area = (sample_step * dx) ** 2
    amplitude = sample * pixel_area / (float(np.pi) * w0 * w0)

    return BeamletBundle(
        positions=positions,
        directions=directions,
        Q=Q,
        amplitude=amplitude.astype(Q.dtype),
        waist0=waist0,
    )


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
    Q_new = Q_old / (1 + t.astype(Q_old.dtype) * Q_old)

    k = 2 * float(np.pi) / wavelength * n_medium
    axial_phase = xp.exp(1j * k * xp.abs(t))
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

    Q_new = beamlets.Q - (1.0 / focal_length)

    x_off = beamlets.positions[..., 0] - cx
    y_off = beamlets.positions[..., 1] - cy
    L_old = beamlets.directions[..., 0]
    M_old = beamlets.directions[..., 1]
    N_old = beamlets.directions[..., 2]
    L_new = L_old - x_off / focal_length
    M_new = M_old - y_off / focal_length

    norm = xp.sqrt(L_new ** 2 + M_new ** 2 + N_old ** 2)
    L_new = L_new / norm
    M_new = M_new / norm
    N_new = N_old / norm
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
    chunk_beamlets: int = 4096,
) -> np.ndarray:
    """Coherently sum every beamlet's transverse profile on a 2-D
    output grid."""
    xp = array_namespace(beamlets.positions)
    cx, cy = centre

    ix = xp.arange(Nx, dtype=beamlets.positions.dtype)
    iy = xp.arange(Ny, dtype=beamlets.positions.dtype)
    Xg, Yg = xp.meshgrid((ix - Nx / 2) * dx + cx,
                         (iy - Ny / 2) * dx + cy,
                         indexing='xy')

    k = 2 * float(np.pi) / wavelength
    out = xp.zeros((Ny, Nx), dtype=beamlets.amplitude.dtype)

    n = int(beamlets.positions.shape[0])
    for start in range(0, n, chunk_beamlets):
        end = min(start + chunk_beamlets, n)
        x_b = beamlets.positions[start:end, 0]
        y_b = beamlets.positions[start:end, 1]
        Q_b = beamlets.Q[start:end]
        a_b = beamlets.amplitude[start:end]

        rho2 = ((Xg[..., None] - x_b[None, None, :]) ** 2
                + (Yg[..., None] - y_b[None, None, :]) ** 2)
        phase = xp.exp(-1j * k * Q_b[None, None, :] * rho2 / 2)
        contrib = a_b[None, None, :] * phase
        out = out + xp.sum(contrib, axis=-1)

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
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
) -> np.ndarray:
    """End-to-end free-space GBD: source -> z -> output.

    .. note::
       This function uses a non-canonical argument order
       ``(E_in, dx, *, z, wavelength, ...)``.  Prefer
       :func:`propagate_gbd` for the canonical
       ``(E_in, z, wavelength, dx)`` order shared with
       :func:`angular_spectrum_propagate` and
       :func:`propagate_huygens_fresnel`.
    """
    Ny, Nx = (E_in.shape[-2], E_in.shape[-1]) if output_grid is None else output_grid
    if output_dx is None:
        output_dx = dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z,
                                          wavelength=wavelength)
    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets,
    )


def propagate_gbd_thin_lens(
    E_in: np.ndarray,
    dx: float,
    *,
    z_to_lens: float,
    focal_length: float,
    z_lens_to_output: float,
    wavelength: float,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    lens_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
) -> np.ndarray:
    """End-to-end three-leg GBD: source -> free space -> thin lens
    -> free space -> output (the canonical GBD validation case)."""
    Ny, Nx = (E_in.shape[-2], E_in.shape[-1]) if output_grid is None else output_grid
    if output_dx is None:
        output_dx = dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
    )
    bundle = propagate_beamlets_freespace(bundle, z_distance=z_to_lens,
                                          wavelength=wavelength)
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

    # Amplitude correction: q_in / q_out factor for Gaussian-beam
    # amplitude conservation (the Q-parameter formulation absorbs
    # this as Q_out / Q_in).
    qratio = Q_new / Q_old
    new_amplitude = beamlets.amplitude * qratio

    return BeamletBundle(
        positions=new_positions,
        directions=new_directions,
        Q=Q_new,
        amplitude=new_amplitude,
        waist0=beamlets.waist0,
    )


def propagate_gbd_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    waist_factor: float = 1.0,
    sample_step: int = 1,
    chunk_beamlets: int = 4096,
) -> np.ndarray:
    """End-to-end GBD through a sequential lumenairy prescription
    via system ABCD evolution.

    Decomposes the source field into a regular grid of Gaussian
    beamlets, transforms each beamlet's complex Q-parameter and
    base ray by the prescription's paraxial system ABCD matrix,
    and coherently reconstructs the output field.

    This is the **paraxial** GBD form: it reduces to the
    Collins integral applied beamlet-by-beamlet.  For wide-field
    or strongly-aberrated systems the per-surface evolution form
    (not yet implemented; tracked as a future extension) gives
    higher accuracy.

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
    """
    from ..raytrace import system_abcd_prescription

    Ny, Nx = (E_in.shape[-2], E_in.shape[-1]) if output_grid is None else output_grid
    if output_dx is None:
        output_dx = dx

    bundle = decompose_field_to_beamlets(
        E_in, dx, wavelength=wavelength,
        waist_factor=waist_factor,
        sample_step=sample_step,
    )

    # Get the system's paraxial ABCD matrix.  ``system_abcd_prescription``
    # returns ``(matrix, efl, bfl)``.
    abcd_result = system_abcd_prescription(prescription, wavelength)
    if isinstance(abcd_result, tuple):
        M = abcd_result[0]
    else:
        M = abcd_result
    A = float(M[0, 0]); B = float(M[0, 1])
    C = float(M[1, 0]); D = float(M[1, 1])

    # Apply ABCD to every beamlet.
    bundle = apply_abcd_to_beamlets(bundle, A, B, C, D,
                                     wavelength=wavelength)

    return reconstruct_field_from_beamlets(
        bundle, Ny=Ny, Nx=Nx, dx=output_dx,
        centre=output_centre, wavelength=wavelength,
        chunk_beamlets=chunk_beamlets,
    )


__all__ = [
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'apply_abcd_to_beamlets',
    'reconstruct_field_from_beamlets',
    'propagate_gbd_freespace',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',
]
