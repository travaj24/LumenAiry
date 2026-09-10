"""Q5b (D2) -- the transient across the band/grid CROSSOVER.

The FIX doc states "below N ~ 1414 the band IS the grid and there is no
transient saving at all (only the narrower output)".  ``_phasor_rows`` /
``_narrow_rows`` band at ``br = max(16, min(ny, 32e6 // (16 nx)))``, so the
band equals the grid while ``16 N^2 <= 32e6``, i.e. N <= 1414.

This sweeps the complex64 peak of ONE ``carrier_referenced_envelope`` call
straight through that crossover -- N = 512, 1024, 1414, 1415, 1448, 2048 --
on the scalar and the astigmatic branch, with the complex128 arm alongside as
the reference the audit's "requesting complex64 saved 0.0 GB" is stated
against.  Run on both arms and diff: "no saving" and "a saving of the wrong
sign" are different claims.

Usage: python q5b_band_crossover.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

MIB = 2 ** 20
NS = (512, 1024, 1414, 1415, 1448, 2048)


def _field(N, dx, dtype, seed=17):
    rng = np.random.default_rng(seed)
    x = (np.arange(N) - N / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    a = np.exp(-r2 / (0.3 * N * dx) ** 2)
    a = a * (1.0 + 0.05 * rng.standard_normal((N, N)))
    return a.astype(dtype)


def main():
    args = _vf.argp(__doc__).parse_args()
    _vf.banner(args.tree)
    from lumenairy.propagators import carrier as C
    out = {}
    for N in NS:
        dx = 6.0e-6 * (1024.0 / N)
        br = int(max(16, min(N, C._PHASOR_BAND_BYTES // (16 * N))))
        for tag, R in (('scalar', 71e-3), ('astig', (71e-3, -59e-3))):
            row = {'band_rows': br, 'bands': int(np.ceil(N / br)),
                   'band_is_grid': br >= N}
            for dt in (np.complex128, np.complex64):
                E = _field(N, dx, dt)
                r = C.carrier_referenced_envelope(E, R, _vf.WL, dx)  # warm
                hh = _vf.h(np.asarray(r))
                del r
                tracemalloc.start()
                tracemalloc.reset_peak()
                r = C.carrier_referenced_envelope(E, R, _vf.WL, dx)
                _, pk = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert _vf.h(np.asarray(r)) == hh
                row[str(np.dtype(dt))] = {
                    'peak_bytes': int(pk), 'peak_MiB': round(pk / MIB, 3),
                    'peak_c64_grids': round(pk / (8.0 * N * N), 4),
                    'hash': hh}
                del E, r
            p128 = row['complex128']['peak_bytes']
            p64 = row['complex64']['peak_bytes']
            row['c64_minus_c128_MiB'] = round((p64 - p128) / MIB, 3)
            row['c64_over_c128'] = round(p64 / p128, 4)
            out[f'N={N}|{tag}'] = row
            print(f"  N={N:<5d} {tag:7s} band={br:<5d} bands="
                  f"{row['bands']:<3d} c128={row['complex128']['peak_MiB']:9.3f}"
                  f"  c64={row['complex64']['peak_MiB']:9.3f} MiB   "
                  f"c64/c128={row['c64_over_c128']:.3f}", flush=True)
    _vf.dump(args, {'cases': out, 'band_bytes': 32e6,
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
