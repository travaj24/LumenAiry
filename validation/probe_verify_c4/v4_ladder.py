"""VERIFY-WP-C4 claim 1 -- the boundary, re-measured on MY ladder.

35 shapes, ``N_in`` up to 2048, ``N_out`` down to 4, chosen to answer three
questions the branch's square ladder cannot:

1. is ``_MFT_DIRECT_MAX_RATIO = 1/32`` safe at every shape the rule CAPTURES
   (both ratios at or under 1/32)?  A captured shape that is slower than the
   faster of the two chirp-Z fallbacks on either build is a defect in the
   constant, and this ladder includes ANISOTROPIC captured shapes
   (``2048x512 -> 64x16`` and friends), which the branch's ladder has none of;
2. is the MAX conjunction the right one?  Mixed-ratio shapes -- one axis under
   the boundary, one over -- are timed on both sides, including the brief's
   ``(1/64, 1/8)``.  MAX refuses those; MIN would take them.  Whether MAX
   leaves time on the table is a measurement, not an opinion;
3. does the boundary hold at non-dyadic sizes (1000, 768, 1536), where
   ``next_fast_len`` pads the chirp-Z route differently?

Instrument: the repeat loop is OUTERMOST and the route order ROTATES with the
repeat index, so a load excursion is spread over all three routes instead of
being charged to whichever one happened to be running.  Best of ``--reps``
(default 7).  The reference loop is read before and after every shape and its
drift is carried into the row.  Each route's ANSWER is digested in the same
pass, so a fast row cannot be a row where the route did not run.

    PYTHONPATH=<tree> python v4_ladder.py <tree> [--what time|mem|rss]
                                                 [--reps N] [--tag NAME]
                                                 [--only SUBSTR]
"""
from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, Ny_in, Nx_in, My_out, Mx_out)
SHAPES = [
    # --- square, captured by the rule (both ratios <= 1/32) ---------------
    ('sq_2048_16',    2048, 2048,   16,   16),   # 1/128
    ('sq_2048_32',    2048, 2048,   32,   32),   # 1/64
    ('sq_2048_64',    2048, 2048,   64,   64),   # 1/32  BOUNDARY
    ('sq_1024_16',    1024, 1024,   16,   16),   # 1/64
    ('sq_1024_32',    1024, 1024,   32,   32),   # 1/32  BOUNDARY (thin)
    ('sq_512_16',      512,  512,   16,   16),   # 1/32  BOUNDARY
    ('sq_512_8',       512,  512,    8,    8),   # 1/64
    ('sq_256_8',       256,  256,    8,    8),   # 1/32  BOUNDARY
    ('sq_128_4',       128,  128,    4,    4),   # 1/32  BOUNDARY
    ('sq_64_2',         64,   64,    2,    2),   # 1/32  BOUNDARY
    # --- square, just OUTSIDE the boundary ---------------------------------
    ('sq_1024_33',    1024, 1024,   33,   33),   # 33/1024, just over
    ('sq_512_17',      512,  512,   17,   17),   # 17/512, just over
    ('sq_2048_128',   2048, 2048,  128,  128),   # 1/16  refused
    ('sq_1024_64',    1024, 1024,   64,   64),   # 1/16  refused (WSL S1)
    ('sq_512_32',      512,  512,   32,   32),   # 1/16  refused
    ('sq_2048_256',   2048, 2048,  256,  256),   # 1/8   refused
    ('sq_1024_128',   1024, 1024,  128,  128),   # 1/8   refused
    ('sq_512_64',      512,  512,   64,   64),   # 1/8   refused
    ('sq_1024_256',   1024, 1024,  256,  256),   # 1/4   refused
    ('sq_1024_512',   1024, 1024,  512,  512),   # 1/2   refused
    # --- ANISOTROPIC input, BOTH ratios at the boundary (captured) ---------
    ('an_2048x512',   2048,  512,   64,   16),   # (1/32, 1/32)
    ('an_512x2048',    512, 2048,   16,   64),   # (1/32, 1/32)
    ('an_2048x256',   2048,  256,   64,    8),   # (1/32, 1/32)
    ('an_1024x128',   1024,  128,   32,    4),   # (1/32, 1/32)
    ('an_2048x64',    2048,   64,   64,    2),   # (1/32, 1/32)
    # --- captured, ratios UNEQUAL but both at or under 1/32 ----------------
    ('mx_in_1024_8x32',  1024, 1024,   8,  32),  # (1/128, 1/32) -> DENSE
    ('mx_in_2048_16x64', 2048, 2048,  16,  64),  # (1/128, 1/32) -> DENSE
    ('mx_in_an',         2048, 1024,  64,  16),  # (1/32,  1/64) -> DENSE
    # --- MIXED ratio: one under, one over.  MAX refuses; MIN would take ----
    ('mx_out_1024_16x128', 1024, 1024,  16, 128),  # (1/64, 1/8)  THE SHAPE
    ('mx_out_1024_128x16', 1024, 1024, 128,  16),  # (1/8,  1/64)
    ('mx_out_2048_32x256', 2048, 2048,  32, 256),  # (1/64, 1/8)
    ('mx_out_1024_8x256',  1024, 1024,   8, 256),  # (1/128, 1/4)
    ('mx_out_2048_64x128', 2048, 2048,  64, 128),  # (1/32, 1/16)
    # --- non-dyadic sizes --------------------------------------------------
    ('nd_1000_25',    1000, 1000,   25,   25),   # 1/40  -> DENSE
    ('nd_768_24',      768,  768,   24,   24),   # 1/32  BOUNDARY -> DENSE
    ('nd_1536_48',    1536, 1536,   48,   48),   # 1/32  BOUNDARY -> DENSE
]

ROUTES = ('chirpz2d', 'separable', 'dense')
KW = {'chirpz2d': dict(separable=False, method='bluestein'),
      'separable': dict(separable=True, method='separable'),
      'dense': dict(method='direct')}


def alpha_for(ny, nx, my, mx):
    """Hold the phase budget at 1e3 -- six decades under ``_PHASE_BUDGET_MAX``
    -- so no route pays the warning and the clock times the arithmetic."""
    return 1.0e3 / float(max(ny, nx, my, mx)) ** 2


def run_once(E, a, my, mx, name):
    import numpy as np
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    return _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                         ifft2=_ifft2, **KW[name])


def main(tree, what, reps, tag, only):
    import numpy as np
    from lumenairy.propagators._bluestein import (
        _auto_selects_direct, _MFT_DIRECT_MAX_RATIO)
    v4lib.anchor(tree)
    build = v4lib.build_tag()
    out = {'build': build, 'what': what, 'reps': reps,
           'python': sys.version.split()[0],
           'numpy': np.__version__,
           'constant': float(_MFT_DIRECT_MAX_RATIO),
           'instrument': 'interleaved best-of-N, route order rotates per rep',
           'load_before': v4lib.load_census(), 'rows': []}
    rng = np.random.default_rng(20260920)
    for (name, ny, nx, my, mx) in SHAPES:
        if only and only not in name:
            continue
        E = (rng.standard_normal((ny, nx))
             + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
        a = alpha_for(ny, nx, my, mx)
        ry, rx = my / ny, mx / nx
        row = {'name': name, 'Ny_in': ny, 'Nx_in': nx,
               'My': my, 'Mx': mx, 'ratio_y': ry, 'ratio_x': rx,
               'ratio_max': max(ry, rx), 'ratio_min': min(ry, rx),
               'alpha': a, 'budget': a * float(max(ny, nx, my, mx)) ** 2,
               'rule_says_direct': bool(_auto_selects_direct(ny, nx, my, mx)),
               'min_rule_would_say_direct':
                   bool(min(ry, rx) <= float(_MFT_DIRECT_MAX_RATIO))}
        row['ref_before'] = v4lib.ref_loop()
        if what == 'time':
            best = {r: float('inf') for r in ROUTES}
            digests = {}
            for rep in range(reps):
                order = ROUTES[rep % 3:] + ROUTES[:rep % 3]
                for r in order:
                    v4lib.cold()
                    t0 = time.perf_counter()
                    F = run_once(E, a, my, mx, r)
                    dt = time.perf_counter() - t0
                    best[r] = min(best[r], dt)
                    if r not in digests:
                        digests[r] = v4lib.digest_array(F)
                    del F
            fb = min(best['chirpz2d'], best['separable'])
            row['best_s'] = best
            row['fallback_s'] = fb
            row['dense_over_fallback'] = best['dense'] / fb
            row['dense_never_slower'] = bool(best['dense'] <= fb)
            row['winner'] = min(best, key=best.get)
            row['digests'] = digests
            row['all_three_differ'] = bool(
                len(set(digests.values())) == 3)
            print(f"TIME {name:22s} {ny}x{nx}->{my}x{mx} "
                  f"rmax={max(ry, rx):9.6f} rule={'D' if row['rule_says_direct'] else 'c'}"
                  f"  dense={best['dense']:.4f} sep={best['separable']:.4f} "
                  f"chirp={best['chirpz2d']:.4f}  d/fb="
                  f"{row['dense_over_fallback']:7.3f}"
                  f"{'  <-- SLOWER' if not row['dense_never_slower'] else ''}",
                  flush=True)
        elif what == 'mem':
            peaks = {}
            for r in ROUTES:
                v4lib.cold()
                tracemalloc.start()
                F = run_once(E, a, my, mx, r)
                _, pk = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peaks[r] = int(pk)
                del F
                gc.collect()
            row['peak_bytes'] = peaks
            row['dense_cheapest'] = bool(
                peaks['dense'] < peaks['separable']
                and peaks['dense'] < peaks['chirpz2d'])
            row['dense_smaller_by'] = min(peaks['separable'],
                                          peaks['chirpz2d']) / peaks['dense']
            row['ordering_dense_sep_chirp'] = bool(
                peaks['dense'] < peaks['separable'] < peaks['chirpz2d'])
            # --- the DERIVED byte counts, from the code -------------------
            import scipy.fft as sfft
            Ly = int(sfft.next_fast_len(ny + my - 1))
            Lx = int(sfft.next_fast_len(nx + mx - 1))
            row['L'] = [Ly, Lx]
            row['derived_bytes'] = {
                # two kernels (built complex128 via a float64 t, then cast),
                # the larger intermediate, and the output
                'dense_kernels': 16 * (my * ny + mx * nx),
                'dense_kernel_build_peak': 16 * max(my * ny, mx * nx)
                                           + 8 * max(my * ny, mx * nx),
                'dense_intermediate': 16 * my * nx,
                'dense_out': 16 * my * mx,
                'chirpz2d_one_L2': 16 * Ly * Lx,
                'separable_one_NxL': 16 * max(ny * Lx, my * Ly),
            }
            print(f"MEM  {name:22s} {ny}x{nx}->{my}x{mx} "
                  f"dense={peaks['dense']/1e6:9.3f} "
                  f"sep={peaks['separable']/1e6:9.3f} "
                  f"chirp={peaks['chirpz2d']/1e6:9.3f} MB  "
                  f"smaller_by={row['dense_smaller_by']:7.1f}x "
                  f"{'OK' if row['dense_cheapest'] else 'NOT-CHEAPEST'}",
                  flush=True)
        else:                                          # rss
            r = os.environ.get('V4_ROUTE', 'dense')
            base = v4lib.peak_rss_bytes()
            v4lib.cold()
            F = run_once(E, a, my, mx, r)
            pk = v4lib.peak_rss_bytes()
            row['route'] = r
            row['rss_before_bytes'] = base
            row['rss_peak_bytes'] = pk
            row['rss_delta_bytes'] = pk - base
            row['digest'] = v4lib.digest_array(F)
            del F
            print(f"RSS  {name:22s} route={r:9s} "
                  f"peak={pk/1e6:9.1f} MB  delta={(pk-base)/1e6:9.1f} MB",
                  flush=True)
        row['ref_after'] = v4lib.ref_loop()
        row['ref_drift'] = row['ref_after'] / row['ref_before']
        out['rows'].append(row)
        del E
        gc.collect()
    out['load_after'] = v4lib.load_census()
    if what == 'time':
        cap = [r for r in out['rows'] if r['rule_says_direct']]
        out['summary'] = {
            'captured_shapes': len(cap),
            'captured_dense_never_slower': sum(
                1 for r in cap if r['dense_never_slower']),
            'captured_worst_ratio': max([r['dense_over_fallback']
                                         for r in cap], default=None),
            'captured_worst_shape': max(
                cap, key=lambda r: r['dense_over_fallback'])['name']
                if cap else None,
            'refused_shapes_where_dense_would_win': [
                r['name'] for r in out['rows']
                if not r['rule_says_direct'] and r['dense_over_fallback'] < 1.0],
        }
        print("SUMMARY", out['summary'], flush=True)
    suffix = f"_{tag}" if tag else ""
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_ladder_{what}{suffix}_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    main(sys.argv[1], _arg('--what', 'time'), int(_arg('--reps', 7)),
         _arg('--tag', ''), _arg('--only', ''))
