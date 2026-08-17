"""Shared geometry builders for the 2-D slant validation scripts.

HISTORY.  Through Phase A this module carried a monkeypatch prototype that
injected the slant into ``twod_jones._layer_eigenmodes_tensor`` without touching
the library.  Phase B shipped the slant as a first-class ``slant=(t_x, t_y)``
parameter of ``pmm_jones_2d`` and ``PMM2DStackHybrid.add_layer``, so the
prototype is RETIRED -- keeping it would have shadowed the very code under test
(and its old signature actively breaks the native path).

What remains here is only the exact-wall cell builders the scripts share.
"""
from __future__ import annotations

import numpy as np

import lumenairy

assert "lum_sl" in lumenairy.__file__, lumenairy.__file__

_C = np.complex128


def binary_cell(nx, duty, eps_r, eps_g, shift_px=0, ny=1):
    """Binary x-grating (uniform in y) as an (nx, ny, 3, 3) tensor cell.

    The ridge is an INTEGER number of pixels wide and rolled by an INTEGER
    number of pixels, so every wall lands exactly on a pixel boundary and the
    cell is represented EXACTLY by ``_cell_to_walls_tile`` (no pixelation).
    """
    nr = int(round(duty * nx))
    line = np.full(nx, eps_g, dtype=_C)
    line[:nr] = eps_r
    line = np.roll(line, int(shift_px))
    cell = np.zeros((nx, ny, 3, 3), dtype=_C)
    for i in range(3):
        cell[:, :, i, i] = line[:, None]
    return cell


def pillar_cell(nx, eps_r, eps_g, sx=0, sy=0,
                fx=(0.25, 0.60), fy=(0.30, 0.65)):
    """A genuinely 2-D rectangular pillar, rolled by INTEGER pixel offsets."""
    g = np.full((nx, nx), eps_g, dtype=_C)
    g[int(fx[0] * nx):int(fx[1] * nx), int(fy[0] * nx):int(fy[1] * nx)] = eps_r
    g = np.roll(np.roll(g, int(sx), axis=0), int(sy), axis=1)
    cell = np.zeros((nx, nx, 3, 3), dtype=_C)
    for i in range(3):
        cell[:, :, i, i] = g
    return cell
