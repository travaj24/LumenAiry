"""V16 -- is the complex64 phasor INDEPENDENT of ``_phasor_rows``'s band size?

``_PHASOR_BAND_BYTES = 32e6`` is a magic constant with no test on it.  The
band loop calls ``np.exp`` on row slices, and numpy's transcendental kernels
take different SIMD paths for different lengths / alignments, so a banded
``exp`` is not guaranteed to be bit-identical to a whole-grid one.  If the
c64 phasor depended on the band size, the D14/D15 determinism contract would
have a new free parameter in it.

This hashes each helper's complex64 output at band sizes spanning one row to
the whole grid, and also compares the c64 result against
``np.exp(float64 argument).astype(complex64)`` computed WHOLE-GRID.

Usage:  python v16_phasor_band_invariance.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

WL = _fix.WL
K = 2 * np.pi / WL


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    N, dx, R = 1024, 1.9e-6, 5.0e-3
    L, M = 0.0515, -0.02
    shape = (N, N)
    helpers = {
        '_radial_carrier_phase': lambda dt: C._radial_carrier_phase(
            shape, dx, dx, WL, R, +1, dtype=dt),
        '_tilt_ramp': lambda dt: C._tilt_ramp(
            shape, dx, WL, L, M, 1.3e-3, -0.4e-3, -1, dtype=dt),
        '_tilt_exactness_phase': lambda dt: C._tilt_exactness_phase(
            shape, dx, dx, WL, R, L, M, +1, centre=(1.3e-3, -0.4e-3),
            dtype=dt),
        '_sphere_parab_conversion': lambda dt: C._sphere_parab_conversion(
            shape, dx, WL, R, +1, centre=(1.3e-3, -0.4e-3), dtype=dt),
    }
    orig = C._PHASOR_BAND_BYTES
    # band size = max(16, min(N, bytes // (16 * N))) rows
    sizes = {'1row_floor16': 16 * 16 * N, 'shipped_32MB': 32e6,
             'tiny_1MB': 1e6, 'huge_1GB': 1e9, 'whole_grid': 16.0 * N * N * 4}
    out = {'version': la.__version__, 'N': N,
           'shipped_PHASOR_BAND_BYTES': float(orig), 'helpers': {}}
    try:
        for nm, fn in helpers.items():
            ref128 = fn(np.complex128)
            if ref128 is None:
                continue
            whole_narrow = ref128.astype(np.complex64)
            hashes = {}
            for tag, val in sizes.items():
                C._PHASOR_BAND_BYTES = val
                g = fn(np.complex64)
                rows = int(max(16, min(N, val // (16 * N))))
                hashes[tag] = {'rows_per_band': rows, 'hash': _fix.h(g),
                               'equals_whole_narrow': bool(
                                   np.array_equal(g, whole_narrow))}
            eq = len({v['hash'] for v in hashes.values()}) == 1
            out['helpers'][nm] = {'band_size_invariant': eq,
                                  'equals_whole_grid_exp_narrowed':
                                      all(v['equals_whole_narrow']
                                          for v in hashes.values()),
                                  'by_band_size': hashes}
            print('%-26s band-size invariant=%s  == whole-grid exp narrowed=%s'
                  % (nm, eq, out['helpers'][nm]
                     ['equals_whole_grid_exp_narrowed']), flush=True)
            for tag, v in hashes.items():
                print('    %-14s rows=%-5d %s eq_whole=%s'
                      % (tag, v['rows_per_band'], v['hash'],
                         v['equals_whole_narrow']), flush=True)
    finally:
        C._PHASOR_BAND_BYTES = orig
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
