"""Subaperture utilities tests."""
from __future__ import annotations

import sys

import numpy as np

from _harness import Harness

from lumenairy.propagators.subaperture import (
    PatchGrid, patches_for_box, patch_window, combine_patch_fields,
)


H = Harness('subaperture')


def t_patches_for_box_count():
    pg = patches_for_box(box_size=(1e-3, 1e-3),
                         patch_size=(0.3e-3, 0.3e-3),
                         overlap=0.25)
    return len(pg) == 25, f'len={len(pg)}'


def t_patch_window_centre():
    w = patch_window(np.array([0.0]), np.array([0.0]),
                     centre=(0, 0), half_widths=(1e-3, 1e-3))
    return abs(float(w[0]) - 1.0) < 1e-12, f'w_centre={float(w[0])}'


def t_patch_window_outside():
    w = patch_window(np.array([2e-3]), np.array([0.0]),
                     centre=(0, 0), half_widths=(1e-3, 1e-3))
    return abs(float(w[0])) < 1e-12, f'w_out={float(w[0])}'


def t_combine_partition_of_unity():
    pg = patches_for_box(box_size=(1e-3, 1e-3),
                         patch_size=(0.3e-3, 0.3e-3),
                         overlap=0.25)
    N = 32
    dx = 0.5e-3 / N
    gx = (np.arange(N) - N/2 + 0.5) * dx
    gy = (np.arange(N) - N/2 + 0.5) * dx
    fields = [np.ones((N, N), dtype=np.complex128) for _ in range(len(pg))]
    combined = combine_patch_fields(fields, pg, output_grid_x=gx, output_grid_y=gy)
    centre = combined[N//4:3*N//4, N//4:3*N//4]
    return float(np.max(np.abs(centre - 1.0))) < 1e-6, 'centre err'


def main():
    H.section('Patch grid')
    H.run('patches_for_box count', t_patches_for_box_count)

    H.section('Patch window')
    H.run('window 1 at centre', t_patch_window_centre)
    H.run('window 0 outside', t_patch_window_outside)

    H.section('Recombination')
    H.run('partition of unity reproduces uniform', t_combine_partition_of_unity)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
