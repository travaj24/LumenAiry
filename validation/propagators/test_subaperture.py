"""Subaperture utilities tests."""
from __future__ import annotations

import sys

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
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


def t_subaperture_asymptotic_runs():
    """Subaperture asymptotic propagator returns finite output."""
    from lumenairy.propagators.subaperture import propagate_subaperture_asymptotic
    import lumenairy as la
    presc = la.make_singlet(R1=5.16e-3, R2=float('inf'), d=2e-3,
                             glass='N-BK7', aperture=4e-3)
    presc['object_distance'] = 30e-3
    N = 32; dx = 5e-6; lam = 633e-9
    x = (np.arange(N) - N/2 + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X*X + Y*Y) / (30e-6)**2).astype(np.complex128)
    # 4.11.2 (audit round-3 test-quality #6): the previous ``except``
    # caught every exception and returned ``True``, hiding the same
    # class of N5-style unpack regressions this validation file is
    # supposed to detect.  Let the exception propagate so any actual
    # failure surfaces as a fail with a traceback.
    out = propagate_subaperture_asymptotic(
        E, dx, presc, wavelength=lam,
        n_patches=(2, 2),
        source_box_half=40e-6, pupil_box_half=2e-3,
        n_field=6, n_pupil=6, poly_order=4,
    )
    return out.shape == E.shape and bool(np.all(np.isfinite(np.abs(out)))), 'ok'


def t_subaperture_window_sum_in_overlap_region_smooth():
    """Sum of patch windows over a 0.5-overlap tiling produces a
    smooth, finite, all-positive map across the source box -- no
    division-by-zero or holes between patches.  (A strict
    partition-of-unity guarantee depends on edge_smoothness; here
    we just verify the sum is well-conditioned.)
    """
    pg = patches_for_box(box_size=(1e-3, 1e-3),
                         patch_size=(0.4e-3, 0.4e-3),
                         overlap=0.5)
    N = 64
    x = np.linspace(-0.4e-3, 0.4e-3, N)
    X, Y = np.meshgrid(x, x, indexing='xy')
    total = np.zeros_like(X)
    for i in range(len(pg)):
        c = (float(pg.centres[i, 0]), float(pg.centres[i, 1]))
        hw = (float(pg.half_widths[i, 0]), float(pg.half_widths[i, 1]))
        total += patch_window(X, Y, c, hw)
    inside = (np.abs(X) < 0.2e-3) & (np.abs(Y) < 0.2e-3)
    inside_min = float(np.min(total[inside]))
    inside_max = float(np.max(total[inside]))
    return (inside_min > 0 and bool(np.all(np.isfinite(total)))), (
        f'inside-box sum range: [{inside_min:.3f}, {inside_max:.3f}]')


def main():
    import lumenairy as la

    H.section('Patch grid')
    H.run('patches_for_box count', t_patches_for_box_count)

    H.section('Patch window')
    H.run('window 1 at centre', t_patch_window_centre)
    H.run('window 0 outside', t_patch_window_outside)

    H.section('Recombination')
    H.run('partition of unity reproduces uniform', t_combine_partition_of_unity)
    H.run('window sum over overlap tiling is positive + finite',
          t_subaperture_window_sum_in_overlap_region_smooth)

    H.section('Subaperture asymptotic')
    H.run('subaperture asymptotic through singlet',
          t_subaperture_asymptotic_runs)

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
