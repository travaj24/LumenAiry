"""
lumenairy._subaperture -- patch / subaperture decomposition utilities.

Wide-field PSF computation with the deterministic asymptotic
propagator breaks down when a single global Chebyshev polynomial
fit cannot accurately represent the system OPL across the entire
source / pupil region.  The remedy is to split the source plane
(and / or the pupil) into smaller patches, fit a local polynomial
per patch, and recombine the per-patch propagated fields at the
output.

This module provides the patch-decomposition primitives that the
existing :mod:`lumenairy.asymptotic` machinery can use to support
that subaperture mode.  It is a separate module so the patch logic
stays decoupled from the polynomial fit -- callers can use these
utilities for HFPI / GBD subaperture modes too.

Multi-backend
-------------

All functions use :func:`lumenairy._array.array_namespace` so they
run on NumPy / CuPy / JAX inputs uniformly.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


@dataclass
class PatchGrid:
    """Regular tiling of a 2-D box into overlapping patches."""

    centres: object             # (N_patch, 2)
    half_widths: object         # (N_patch, 2)
    overlap: float
    box_size: Tuple[float, float]

    def __len__(self) -> int:
        try:
            return int(self.centres.shape[0])
        except (AttributeError, TypeError, IndexError):
            return 0


def patches_for_box(
    box_size: Tuple[float, float],
    patch_size: Tuple[float, float],
    *,
    overlap: float = 0.25,
    centred: bool = True,
) -> PatchGrid:
    """Build a regular patch tiling over a rectangular box."""
    W_x, W_y = float(box_size[0]), float(box_size[1])
    w_x, w_y = float(patch_size[0]), float(patch_size[1])

    step_x = w_x * (1 - overlap)
    step_y = w_y * (1 - overlap)

    n_x = max(1, int(np.ceil(W_x / step_x)))
    n_y = max(1, int(np.ceil(W_y / step_y)))

    x0 = -W_x / 2 + w_x / 2
    y0 = -W_y / 2 + w_y / 2

    cx = np.array([x0 + i * step_x for i in range(n_x)])
    cy = np.array([y0 + j * step_y for j in range(n_y)])
    CX, CY = np.meshgrid(cx, cy, indexing='xy')
    centres = np.stack([CX.reshape(-1), CY.reshape(-1)], axis=-1)
    half_widths = np.full_like(centres, [w_x / 2, w_y / 2])

    return PatchGrid(
        centres=centres,
        half_widths=half_widths,
        overlap=float(overlap),
        box_size=(W_x, W_y),
    )


def patch_window(
    x: np.ndarray,
    y: np.ndarray,
    centre: Tuple[float, float],
    half_widths: Tuple[float, float],
    *,
    edge_smoothness: float = 0.1,
) -> np.ndarray:
    """Smooth window for a single patch.  Returns a value in
    ``[0, 1]`` for each ``(x, y)`` position, equal to 1 inside
    ``|x - cx| < half_w_x * (1 - edge_smoothness)`` and tapered
    smoothly to 0 at ``|x - cx| = half_w_x``."""
    xp = array_namespace(x, y)
    cx, cy = centre
    hwx, hwy = half_widths

    inner_x = hwx * (1 - edge_smoothness)
    inner_y = hwy * (1 - edge_smoothness)

    dx = xp.abs(x - cx)
    dy = xp.abs(y - cy)

    def axis_window(d, inner, outer):
        t = xp.clip((d - inner) / xp.maximum(outer - inner, 1e-30), 0.0, 1.0)
        return 0.5 * (1 + xp.cos(float(np.pi) * t))

    return axis_window(dx, inner_x, hwx) * axis_window(dy, inner_y, hwy)


def combine_patch_fields(
    patch_fields: List[np.ndarray],
    patch_grid: PatchGrid,
    *,
    output_grid_x: np.ndarray,
    output_grid_y: np.ndarray,
    edge_smoothness: float = 0.1,
) -> np.ndarray:
    """Coherent recombination of per-patch output fields into a
    global output field via partition-of-unity weights."""
    if len(patch_fields) != len(patch_grid):
        raise ValueError(
            f"combine_patch_fields: got {len(patch_fields)} patch_fields "
            f"but patch_grid has {len(patch_grid)} patches.")

    if len(patch_fields) == 0:
        raise ValueError("combine_patch_fields: empty patch list.")

    xp = array_namespace(patch_fields[0])
    X, Y = xp.meshgrid(output_grid_x, output_grid_y, indexing='xy')

    out = xp.zeros_like(patch_fields[0])
    weight_total = xp.zeros(out.shape, dtype=xp.real(patch_fields[0]).dtype)

    for i, F in enumerate(patch_fields):
        centre = (float(patch_grid.centres[i, 0]),
                  float(patch_grid.centres[i, 1]))
        hw = (float(patch_grid.half_widths[i, 0]),
              float(patch_grid.half_widths[i, 1]))
        w = patch_window(X, Y, centre, hw,
                         edge_smoothness=edge_smoothness)
        out = out + F * w.astype(F.dtype)
        weight_total = weight_total + w

    weight_total = xp.where(weight_total > 1e-12, weight_total, 1.0)
    return out / weight_total.astype(out.dtype)


# ============================================================================
# Subaperture asymptotic propagator -- per-patch fit_canonical_polynomials
# ============================================================================

def propagate_subaperture_asymptotic(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    n_patches: Tuple[int, int] = (3, 3),
    patch_overlap: float = 0.25,
    edge_smoothness: float = 0.15,
) -> np.ndarray:
    """Wide-field deterministic asymptotic propagator using
    subaperture (patch) decomposition.

    For optical systems whose OPL surface cannot be accurately
    represented by a single global Chebyshev polynomial across the
    entire source box (e.g. wide-field imagers, high-NA systems),
    the standard
    :func:`lumenairy.propagators.asymptotic.propagate_modal_asymptotic`
    over a single global fit produces accuracy that degrades at
    the box edges.

    This function decomposes the source plane into
    ``n_patches[0] x n_patches[1]`` overlapping patches, fits a
    local polynomial per patch, propagates each patch through the
    asymptotic propagator, and recombines the per-patch output
    fields with a partition-of-unity weighting.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
    wavelength : float
    output_grid, output_dx, output_centre : grid geometry
    source_box_half : float
        Per-patch source half-width (m).  Total source box covered
        is ``n_patches * source_box_half * (1 - patch_overlap) * 2``.
    pupil_box_half, n_field, n_pupil, poly_order : forwarded to
        :func:`fit_canonical_polynomials`.
    n_patches : (int, int)
        Number of patches per axis (n_y, n_x).
    patch_overlap : float
        Fractional overlap between adjacent patches.  Default 0.25.
    edge_smoothness : float
        Window-taper width as fraction of patch half-width.

    Returns
    -------
    array (Ny, Nx) complex
        Coherently recombined output field.
    """
    from .asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )
    from ..analysis.core import beam_d4sigma
    import numpy as _np

    Ny, Nx = (E_in.shape[-2], E_in.shape[-1]) if output_grid is None else output_grid
    if output_dx is None:
        output_dx = dx

    # Build the patch grid.  Total source box = n_patches *
    # patch_size * (1 - overlap) + patch_size.
    n_y, n_x = n_patches
    patch_size = (2 * source_box_half, 2 * source_box_half)
    total_box_x = n_x * patch_size[1] * (1 - patch_overlap)
    total_box_y = n_y * patch_size[0] * (1 - patch_overlap)
    pg = patches_for_box(
        box_size=(total_box_x, total_box_y),
        patch_size=patch_size,
        overlap=patch_overlap,
    )

    # Output grid (used both for evaluation and for window
    # construction).
    cx, cy = output_centre
    # v4.12.1 (B1-10): pixel-centred `(arange(N) - N/2)*dx`, matches the
    # library-wide convention (ASM, Fresnel, RS, sources,
    # ``apply_fresnel_curvature``).  The sub-aperture output is intended
    # to be stacked / overlaid with ASM/HF outputs, so the grid must
    # align pixel-for-pixel.
    out_x = (_np.arange(Nx) - Nx / 2) * output_dx + cx
    out_y = (_np.arange(Ny) - Ny / 2) * output_dx + cy

    # Estimate source waist for the asymptotic propagator.
    try:
        d4 = beam_d4sigma(E_in, dx=dx)
        w_s = float(d4) / 4.0
    except (TypeError, ValueError, RuntimeError, ZeroDivisionError):
        # beam_d4sigma rejects non-array / empty E_in or diverging
        # moments; fall back to a half-box estimate.
        w_s = source_box_half / 2

    # Per-patch evaluation: build a local fit centred on each patch
    # centre, run asymptotic propagator, capture field on the global
    # output grid.
    OX, OY = _np.meshgrid(out_x, out_y, indexing='xy')

    patch_fields = []
    for i in range(len(pg)):
        cx_i, cy_i = float(pg.centres[i, 0]), float(pg.centres[i, 1])
        # Fit centred on this patch's source point.
        # 4.11.2: pass ``source_centre=(cx_i, cy_i)`` so the local
        # polynomial fit samples the prescription on a Chebyshev grid
        # *centred on this patch's object-plane footprint*.  Pre-
        # 4.11.2 the per-patch fit had source_centre fixed at the
        # origin -- every patch built the same on-axis fit, so the
        # patch decomposition was effectively the on-axis fit alone
        # and off-axis patches contributed zero field at any
        # evaluation pixel outside the on-axis fit's training box.
        fit = fit_canonical_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field,
            n_pupil=n_pupil,
            poly_order=poly_order,
            source_centre=(cx_i, cy_i),
        )
        # Propagate from this patch's source point.  4.10: the actual
        # `propagate_modal_asymptotic` signature uses
        # `source_amplitudes` / `pupil_amplitudes` (not
        # `source_lg_amps` / `pupil_lg_amps`) and `s2_grid_x` /
        # `s2_grid_y` (not `output_grid`).  Pre-4.10 calls raised
        # TypeError on first invocation -- the subaperture path was
        # dead on import.  4.11.1: feed the (Ny, Nx) meshgrids
        # directly; the 4.10 patch built a 3-D ``np.stack(...,axis=-1)``
        # array and then tried to unpack it 2-ways, which always raised
        # ``ValueError: too many values to unpack`` for any Ny != 2.
        sgx, sgy = OX, OY
        F_i = propagate_modal_asymptotic(
            fit,
            source_amplitudes={(0, 0): 1.0 + 0.0j},
            pupil_amplitudes={(0, 0): 1.0 + 0.0j},
            source_point=(cx_i, cy_i),
            w_s=w_s,
            w_p=pupil_box_half,
            v2_centre=(0.0, 0.0),
            s2_grid_x=sgx,
            s2_grid_y=sgy,
        )
        patch_fields.append(F_i)

    # Recombine via partition-of-unity windows centred on patch
    # centres in the *output* plane.  This treats the patch
    # decomposition as defining "which patch the output point sees
    # most clearly" -- adjacent patches' contributions blend smoothly.
    return combine_patch_fields(
        patch_fields, pg,
        output_grid_x=out_x, output_grid_y=out_y,
        edge_smoothness=edge_smoothness,
    )


__all__ = [
    'PatchGrid',
    'patches_for_box',
    'patch_window',
    'combine_patch_fields',
    'propagate_subaperture_asymptotic',
]
