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
from typing import List, Tuple

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
        except Exception:
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
    x,
    y,
    centre: Tuple[float, float],
    half_widths: Tuple[float, float],
    *,
    edge_smoothness: float = 0.1,
):
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
    patch_fields: List,
    patch_grid: PatchGrid,
    *,
    output_grid_x,
    output_grid_y,
    edge_smoothness: float = 0.1,
):
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


__all__ = [
    'PatchGrid',
    'patches_for_box',
    'patch_window',
    'combine_patch_fields',
]
