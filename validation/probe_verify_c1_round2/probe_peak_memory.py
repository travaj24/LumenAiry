"""VERIFY-WP-C1 ROUND 2 -- the peak-memory claim, measured with tracemalloc.

``apply_aperture``'s grey branch carries the comment

    Accumulated one sub-mask at a time, so the peak stays at the cost of a
    single sub-mask evaluation (measured 6.0 float64 grids at N = 2048,
    against 5.0 for the hard edge) instead of growing with n_sub**2 --
    measured identical at n_sub = 2, 4 and 8.

VERIFY-C1 recorded this as "not re-measured; it needs an allocator trace", and
round 2 repeated that.  This probe runs the trace.  NumPy registers its
allocations with ``tracemalloc`` under ``numpy.lib.tracemalloc_domain``, and
``tracemalloc.get_traced_memory()`` sums every domain, so the peak below is the
array peak and not merely the Python-object peak.  The probe checks that numpy
is in fact traced (by allocating a known-size array and reading the delta)
before it reports anything.

One "float64 grid" is ``N*N*8`` bytes; the input field is complex128 and
therefore TWO such grids.  Peaks are reported both as an absolute figure and
with the input field subtracted, because the source comment does not say which
it means.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_peak_memory.py <out.json>
"""
import gc
import json
import sys
import tracemalloc

import numpy as np

import lumenairy  # noqa: F401
from lumenairy.elements.elements import apply_aperture

N = 2048
DX = 1e-6
GRID = N * N * 8.0          # one float64 grid, bytes


def _peak(fn):
    gc.collect()
    tracemalloc.start()
    base_cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    out = fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
    return peak - base_cur, cur - base_cur


def main(out_path):
    # -- is numpy traced at all?  Allocate a known 1.0-grid array. ---------
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    probe_arr = np.zeros((N, N), dtype=np.float64)
    _, peak_known = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del probe_arr
    gc.collect()
    numpy_traced = peak_known > 0.5 * GRID

    E = np.ones((N, N), dtype=np.complex128)   # 2.0 float64 grids
    # WARM-UP.  Measured 2026-09-20: the FIRST apply_aperture call in a
    # process carries ~1.37 (Windows) / ~1.05 (WSL) extra float64 grids of
    # one-off allocation, so whichever arm is measured first reads high.
    # Every arm below is measured in steady state, and the reading is taken
    # twice to show it has settled.
    for kw in ({'edge': 'hard'}, {'edge': 'gray'}):
        apply_aperture(E, DX, 'circular', {'diameter': 0.5 * N * DX}, **kw)
    gc.collect()
    res = {}
    arms = [('hard', {'edge': 'hard'}),
            ('gray_n2', {'edge': 'gray', 'edge_samples': 2}),
            ('gray_n4', {'edge': 'gray', 'edge_samples': 4}),
            ('gray_n8', {'edge': 'gray', 'edge_samples': 8}),
            ('gray_n16', {'edge': 'gray', 'edge_samples': 16}),
            ('default', {})]
    for label, kw in arms:
        peak, held = _peak(lambda kw=kw: apply_aperture(
            E, DX, 'circular', {'diameter': 0.5 * N * DX}, **kw))
        peak2, _ = _peak(lambda kw=kw: apply_aperture(
            E, DX, 'circular', {'diameter': 0.5 * N * DX}, **kw))
        res[label] = {
            'peak_bytes': int(peak),
            'peak_grids': peak / GRID,
            'peak_grids_repeat': peak2 / GRID,
            'settled': abs(peak - peak2) < 0.01 * GRID,
            'peak_grids_plus_input': peak / GRID + 2.0,
            'held_bytes': int(held),
        }

    grey_peaks = [res[k]['peak_grids'] for k in
                  ('gray_n2', 'gray_n4', 'gray_n8', 'gray_n16')]
    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'N': N,
        'one_float64_grid_bytes': GRID,
        'numpy_allocations_are_traced': bool(numpy_traced),
        'known_1grid_peak_bytes': int(peak_known),
        'arms': res,
        'grey_peak_independent_of_n_sub': (max(grey_peaks) - min(grey_peaks))
                                          < 0.05,
        'grey_minus_hard_grids': res['gray_n4']['peak_grids']
                                 - res['hard']['peak_grids'],
        'default_equals_gray_n4': (res['default']['peak_bytes']
                                   == res['gray_n4']['peak_bytes']),
        'all_settled': all(r['settled'] for r in res.values()),
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    print('numpy allocations traced:', numpy_traced,
          '(known 1-grid peak = {0:.3f} grids)'.format(peak_known / GRID))
    print('N =', N, ' one float64 grid =', int(GRID), 'bytes')
    for label, r in res.items():
        print("  {0:10s} peak={1:11d} B = {2:.4f} float64 grids "
              "(repeat {3:.4f}, settled={4}) (+input = {5:.4f})".format(
                  label, r['peak_bytes'], r['peak_grids'],
                  r['peak_grids_repeat'], r['settled'],
                  r['peak_grids_plus_input']))
    print("grey peak independent of n_sub:",
          out['grey_peak_independent_of_n_sub'],
          " grey - hard =", '{0:.3f}'.format(out['grey_minus_hard_grids']),
          "grids")


if __name__ == '__main__':
    main(sys.argv[1])
