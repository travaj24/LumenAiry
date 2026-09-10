"""Q5c (D2) -- is the sub-crossover complex64 PENALTY new, or inherited?

Q5b measures the ``carrier_referenced_envelope`` complex64 peak RISING by a
flat 14.3 % against v5.44.0 for every N below the band/grid crossover.  The
mechanism is ``_phasor_rows``: when one band IS the grid it holds the
complex64 output AND the whole-grid complex128 ``exp`` transient at the same
time, where the shipped whole-grid build freed the complex128 array as it
narrowed.

That route was already shipped in 5.44.0 for the OTHER FOUR reference-phase
helpers.  This measures those four directly, ``dtype=complex64`` against
``dtype=complex128``, at N below and above the crossover, on both arms -- so
the penalty can be attributed to the pre-existing ``_phasor_rows`` design
rather than to the D2 change, or not.

Usage: python q5c_phasor_rows_penalty.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

MIB = 2 ** 20


def helpers(C, N, dx, R):
    shape = (N, N)
    L, M = 0.041, -0.017
    x0, y0 = 1.1e-3, -0.5e-3
    return {
        '_radial_carrier_phase': lambda dt: C._radial_carrier_phase(
            shape, dx, dx, _vf.WL, R, +1, dtype=dt),
        '_tilt_ramp': lambda dt: C._tilt_ramp(
            shape, dx, _vf.WL, L, M, x0, y0, -1, dtype=dt),
        '_tilt_exactness_phase': lambda dt: C._tilt_exactness_phase(
            shape, dx, dx, _vf.WL, R, L, M, +1, centre=(x0, y0), dtype=dt),
        '_sphere_parab_conversion': lambda dt: C._sphere_parab_conversion(
            shape, dx, _vf.WL, R, +1, centre=(x0, y0), dtype=dt),
    }


def main():
    args = _vf.argp(__doc__).parse_args()
    _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    out = {}
    for N in (1024, 2048):
        dx = 6.0e-6 * (1024.0 / N)
        for name, fn in helpers(C, N, dx, 62e-3).items():
            row = {}
            for dt in (np.complex128, np.complex64):
                fn(dt)                                      # warm
                tracemalloc.start()
                tracemalloc.reset_peak()
                a = fn(dt)
                _, pk = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                row[str(np.dtype(dt))] = {
                    'peak_MiB': round(pk / MIB, 3),
                    'peak_c64_grids': round(pk / (8.0 * N * N), 4),
                    'hash': _vf.h(np.asarray(a))}
                del a
            row['c64_over_c128'] = round(
                row['complex64']['peak_MiB']
                / row['complex128']['peak_MiB'], 4)
            out[f'N={N}|{name}'] = row
            print(f"  N={N:<5d} {name:26s} c128="
                  f"{row['complex128']['peak_MiB']:8.3f}  c64="
                  f"{row['complex64']['peak_MiB']:8.3f} MiB   ratio="
                  f"{row['c64_over_c128']:.3f}", flush=True)
    _vf.dump(args, {'cases': out, 'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
