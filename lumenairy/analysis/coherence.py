"""
Partial coherence and extended-source imaging.

For fully-coherent simulations a single plane wave suffices.  For
partially-coherent or extended-source illumination the image is the
incoherent superposition of coherent sub-images, each corresponding
to one source point (or one tilt angle in Koehler illumination).

This module provides:

* :func:`koehler_image` — Koehler (condenser) illumination model:
  integrates the coherent PSF over a source distribution to build a
  partially-coherent image.
* :func:`extended_source_image` — generic extended-source model with
  arbitrary source intensity vs angle.
* :func:`mutual_coherence` — compute the mutual coherence function
  Gamma(r1, r2) from a field ensemble.

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..elements.lenses import apply_real_lens
from ..propagators.propagation import angular_spectrum_propagate

__all__ = [
    'koehler_image',
    'extended_source_image',
    'mutual_coherence',
]


def koehler_image(
    object_field: np.ndarray,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    condenser_NA: float = 0.1,
    n_source_points: int = 9,
    focal_length: Optional[float] = None,
) -> np.ndarray:
    """Simulate Koehler-illumination imaging through a lens.

    Koehler illumination models a spatially-incoherent extended
    source by summing coherent images from a set of tilted plane
    waves covering the condenser NA.

    Parameters
    ----------
    object_field : ndarray, complex, shape (N, N)
        Complex transmission of the object (amplitude * phase).
        Illuminated by each source direction in turn.
    prescription : dict
        Lens prescription (for ``apply_real_lens``).
    wavelength : float
    dx : float
    condenser_NA : float
        Numerical aperture of the condenser (determines the cone of
        illumination angles).
    n_source_points : int
        Number of source directions to sample across the condenser
        pupil.  Total evaluations = ``n_source_points^2`` (grid) or
        ``~pi/4 * n_source_points^2`` (circle).  More = better
        integration but slower.
    focal_length : float, optional
        Back focal length for propagation to the image plane.
        If omitted the image is returned at the lens exit vertex.

    Returns
    -------
    I_image : ndarray, real, shape (N, N)
        Partially-coherent image intensity.
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble object failed at ``object_field.shape[0]`` indexing
    # (3-D returned a meaningful but wrong N) or attribute access
    # (MCF).  Routes both to the canonical v4.16 message via the V6
    # walker.  Input kind: 'field' (the object transmission is a
    # 2-D scalar complex field on the object plane).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(object_field, 'koehler_image')
    N = object_field.shape[0]
    k0 = 2 * np.pi / wavelength
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    theta_max = np.arcsin(min(condenser_NA, 0.999))
    angles = np.linspace(-theta_max, theta_max, n_source_points)
    I_total = np.zeros((N, N), dtype=np.float64)
    count = 0

    for ax in angles:
        for ay in angles:
            if np.sqrt(ax ** 2 + ay ** 2) > theta_max:
                continue
            # Tilted illumination
            E_illum = object_field * np.exp(
                1j * k0 * (np.sin(ax) * X + np.sin(ay) * Y))
            E_exit = apply_real_lens(E_illum, prescription=prescription, wavelength=wavelength, dx=dx)
            if focal_length is not None:
                E_exit = angular_spectrum_propagate(
                    E_exit, focal_length, wavelength, dx)
            I_total = I_total + np.abs(E_exit) ** 2
            count += 1

    return I_total / max(count, 1)


def extended_source_image(
    object_field: np.ndarray,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    source_angles: Sequence[Tuple[float, float]],
    source_weights: Optional[Sequence[float]] = None,
    focal_length: Optional[float] = None,
) -> np.ndarray:
    """Simulate imaging with an arbitrary extended source.

    Each source direction contributes a coherent sub-image; the
    weighted incoherent sum gives the partially-coherent result.

    Parameters
    ----------
    object_field : ndarray, complex, shape (N, N)
        Object transmission.
    prescription : dict
        Lens prescription.
    wavelength, dx : float
    source_angles : sequence of (angle_x, angle_y) [rad]
        Illumination directions.
    source_weights : sequence of float, optional
        Relative intensities per direction (default: uniform).
    focal_length : float, optional
        BFL for image-plane propagation.

    Returns
    -------
    I_image : ndarray, shape (N, N)
    """
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper (sibling of ``koehler_image``).
    # Input kind: 'field' (object transmission).
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(object_field, 'extended_source_image')
    N = object_field.shape[0]
    k0 = 2 * np.pi / wavelength
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)

    if source_weights is None:
        source_weights = np.ones(len(source_angles))
    source_weights = np.asarray(source_weights, dtype=np.float64)
    source_weights = source_weights / source_weights.sum()

    I_total = np.zeros((N, N), dtype=np.float64)
    for (ax, ay), w in zip(source_angles, source_weights):
        E_illum = object_field * np.exp(
            1j * k0 * (np.sin(ax) * X + np.sin(ay) * Y))
        E_exit = apply_real_lens(E_illum, prescription=prescription, wavelength=wavelength, dx=dx)
        if focal_length is not None:
            E_exit = angular_spectrum_propagate(
                E_exit, focal_length, wavelength, dx)
        I_total = I_total + w * np.abs(E_exit) ** 2

    return I_total


def mutual_coherence(
    fields: Sequence[np.ndarray],
    dx: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mutual coherence function from an ensemble of fields.

    Given N realisations of a partially coherent field (e.g. from
    different source points or turbulence realisations), computes

        Gamma(x1, x2) = <E(x1) E*(x2)>

    along the central row (y = 0) for all pairs (x1, x2).

    Parameters
    ----------
    fields : sequence of ndarray, each (Ny, Nx) complex
        Ensemble of field realisations.
    dx : float
        Grid spacing [m].

    Returns
    -------
    Gamma : ndarray, shape (Nx, Nx)
        Mutual coherence matrix along the central row.
    x : ndarray, shape (Nx,)
        Coordinate array [m].
    """
    Ny, Nx = fields[0].shape
    cy = Ny // 2
    rows = np.array([f[cy, :] for f in fields])  # (N_ensemble, Nx)
    # Gamma[i, j] = < E(x_i) conj(E(x_j)) > over the ensemble.  4.10:
    # pre-4.10 used rows.T.conj() @ rows which produces
    # < conj(E(x_i)) E(x_j) > -- the complex conjugate of the
    # documented quantity.  Hermiticity is still preserved (Gamma is
    # always Hermitian), so the bug was silent, but any phase-sensitive
    # consumer (degree of coherence, Wolf-Mandel imaging) saw the
    # off-diagonals with flipped sign.
    Gamma = rows.T @ rows.conj() / len(fields)
    x = (np.arange(Nx) - Nx / 2) * dx
    return Gamma, x
