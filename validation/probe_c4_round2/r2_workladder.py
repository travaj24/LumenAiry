"""WP-C4 round 2, V-C4-D1 -- the ladder the SECOND condition's constant is
derived from.

WHAT THIS MEASURES AND WHY.  The shipped rule reads two RATIOS.  VERIFY-WP-C4
showed that two shapes with the same ratio pair can differ by more than two
decades in the quantity that actually decides the race between the dense route
and the chirp-Z ones: the dense route builds ``My*Ny + Mx*Nx`` transcendental
kernel entries and then spends
``min(My*Ny*Nx + My*Nx*Mx, Ny*Nx*Mx + My*Ny*Mx)`` multiply-adds using them, so
the BUILD dominates whenever the multiply-adds PER kernel entry is small.
``2048x2048 -> 64x64`` reads 1056 of them and ``2048x64 -> 64x2`` reads 4.0,
and both sit at ratio exactly ``(1/32, 1/32)``.

This ladder spans that quantity from 1.25 to 64 over 34 CAPTURED anisotropic
shapes -- both orientations, and several absolute sizes at each decade, because
the screen is one-sided and not a crossover -- and carries the 17 square,
non-dyadic and mildly anisotropic shapes the shipped constant was derived from
as the control that must not move.

INSTRUMENT.  Best-of-``reps`` per route, routes INTERLEAVED with the order
rotating by repeat index, every registered cache dropped and ``gc.collect()``
before each repeat, all three routes' answers digested in the same pass (so a
row cannot be fast because a route did not run), the reference loop read before
and after every shape, and ``fft_infra.SCIPY_FFT_WORKERS`` forced to 1 so the
comparison is SYMMETRIC: ``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`` pins the dense route's two BLAS products to one thread and
does NOT constrain scipy's pocketfft (VERIFY-WP-C4 N1).  ``rounds`` independent
rounds, verdict on the WORST round.

    PYTHONPATH=<tree> python r2_workladder.py <tree> [--rounds 2] [--reps 9]
                                              [--family thin|square|all]
                                              [--workers 1]
"""
from __future__ import annotations

import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'probe_verify_c4'))
import v4lib  # noqa: E402

ROUTES = ('chirpz2d', 'separable', 'dense')
KW = {'chirpz2d': dict(separable=False, method='bluestein'),
      'separable': dict(separable=True, method='separable'),
      'dense': dict(method='direct')}


def work_per_kernel_entry(ny, nx, my, mx) -> float:
    """The quantity the second condition reads, from the code in
    :func:`lumenairy.propagators._bluestein._direct_matrix_2d`."""
    entries = my * ny + mx * nx
    flops = min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
    return flops / entries


#: 34 CAPTURED anisotropic shapes (both ratios exactly 1/32 unless noted),
#: spanning work/entry 1.25 .. 64.  Both orientations appear, and at several
#: work/entry values two different ABSOLUTE sizes appear, because the screen is
#: one-sided: at small absolute sizes the chirp-Z route's fixed costs
#: (planning, padding to ``next_fast_len``) dominate whatever the asymptotic
#: count says, so a low work/entry shape can still be safe.
THIN = [
    ('w0125_4096x32',   4096,   32,  128,   1),
    ('w0150_2048x32',   2048,   32,   64,   1),
    ('w0150_32x2048',     32, 2048,    1,  64),
    ('w0200_1024x32',   1024,   32,   32,   1),
    ('w0200_4096x64',   4096,   64,   64,   1),
    ('w0299_512x32',     512,   32,   16,   1),
    ('w0300_2048x64',   2048,   64,   32,   1),
    ('w0300_4096x64b',  4096,   64,  128,   2),
    ('w0400_2048x64b',  2048,   64,   64,   2),
    ('w0400_64x2048',     64, 2048,    2,  64),
    ('w0492_256x32',     256,   32,    8,   1),
    ('w0498_1024x64',   1024,   64,   16,   1),
    ('w0500_4096x128',  4096,  128,   32,   1),
    ('w0598_1024x64b',  1024,   64,   32,   2),
    ('w0799_4096x128b', 4096,  128,  128,   4),
    ('w0886_512x64',     512,   64,    8,   1),
    ('w0896_2048x128',  2048,  128,   16,   1),
    ('w0985_512x64b',    512,   64,   16,   2),
    ('w0996_2048x128b', 2048,  128,   32,   2),
    ('w1195_2048x128c', 2048,  128,   64,   4),
    ('w1195_128x2048',   128, 2048,    4,  64),
    ('w1600_256x64',     256,   64,    4,   1),
    ('w1674_1024x128',  1024,  128,    8,   1),
    ('w1772_1024x128b', 1024,  128,   16,   2),
    ('w1793_4096x256',  4096,  256,   32,   2),
    ('w1969_1024x128c', 1024,  128,   32,   4),
    ('w1992_4096x256b', 4096,  256,   64,   4),
    ('w2391_4096x256c', 4096,  256,  128,   8),
    ('w3200_512x128',    512,  128,    8,   2),
    ('w3348_256x2048',   256, 2048,    2,  16),
    ('w3545_2048x256',  2048,  256,   32,   4),
    ('w3938_2048x256b', 2048,  256,   64,   8),
    ('w5280_256x128',    256,  128,    4,   2),
    ('w6400_1024x256',  1024,  256,   16,   4),
]

#: The control: the square / near-square / mildly anisotropic captured shapes
#: the shipped constant was derived from.  None of these may be refused by the
#: second condition, or the condition is not a screen for the thin regime but a
#: retune of the boundary.
SQUARE = [
    ('sq_2048_16',  2048, 2048,   16,  16),
    ('sq_2048_32',  2048, 2048,   32,  32),
    ('sq_2048_64',  2048, 2048,   64,  64),
    ('sq_1024_16',  1024, 1024,   16,  16),
    ('sq_1024_32',  1024, 1024,   32,  32),
    ('sq_512_16',    512,  512,   16,  16),
    ('sq_512_8',     512,  512,    8,   8),
    ('sq_256_8',     256,  256,    8,   8),
    ('sq_128_4',     128,  128,    4,   4),
    ('sq_64_2',       64,   64,    2,   2),
    ('nd_1000_25',  1000, 1000,   25,  25),
    ('nd_768_24',    768,  768,   24,  24),
    ('nd_1536_48',  1536, 1536,   48,  48),
    ('an_2048x512', 2048,  512,   64,  16),
    ('an_512x2048',  512, 2048,   16,  64),
    ('an_2048x256', 2048,  256,   64,   8),
    ('mx_in_2048',  2048, 2048,   16,  64),
]


def alpha_for(ny, nx, my, mx):
    """Hold the phase budget at 1e3 -- six decades under
    ``_PHASE_BUDGET_MAX``, so no route pays the warning and the clock times the
    arithmetic and nothing else."""
    return 1.0e3 / float(max(ny, nx, my, mx)) ** 2


def run_once(E, a, my, mx, name):
    import numpy as np
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    return _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np, fft2=_fft2,
                         ifft2=_ifft2, **KW[name])


def main(tree, rounds, reps, family, workers):
    import numpy as np
    from lumenairy.propagators import fft_infra
    from lumenairy.propagators._bluestein import (
        _MFT_DIRECT_MAX_RATIO, _auto_selects_direct)
    v4lib.anchor(tree)
    fft_infra.SCIPY_FFT_WORKERS = int(workers)
    shapes = (THIN if family == 'thin' else
              SQUARE if family == 'square' else THIN + SQUARE)
    out = {'build': v4lib.build_tag(), 'rounds': rounds, 'reps': reps,
           'family': family,
           'constant_max_ratio': float(_MFT_DIRECT_MAX_RATIO),
           'scipy_fft_workers': int(fft_infra.SCIPY_FFT_WORKERS),
           'instrument': 'interleaved best-of-N, route order rotates per rep,'
                         ' cold before every repeat, workers=1 both sides',
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': [], 'round_load': []}
    rng = np.random.default_rng(20260921)
    per = {}
    for rnd in range(rounds):
        out['round_load'].append(v4lib.load_census())
        for (name, ny, nx, my, mx) in shapes:
            E = (rng.standard_normal((ny, nx))
                 + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
            a = alpha_for(ny, nx, my, mx)
            ref0 = v4lib.ref_loop()
            best = {r: float('inf') for r in ROUTES}
            dig = {}
            for rep in range(reps):
                order = ROUTES[rep % 3:] + ROUTES[:rep % 3]
                for r in order:
                    v4lib.cold()
                    t0 = time.perf_counter()
                    F = run_once(E, a, my, mx, r)
                    best[r] = min(best[r], time.perf_counter() - t0)
                    if r not in dig:
                        dig[r] = v4lib.digest_array(F)
                    del F
            ref1 = v4lib.ref_loop()
            fb = min(best['chirpz2d'], best['separable'])
            rec = per.setdefault(name, {
                'name': name, 'Ny_in': ny, 'Nx_in': nx, 'My': my, 'Mx': mx,
                'ratio_max': max(my / ny, mx / nx),
                'work_per_kernel_entry': work_per_kernel_entry(ny, nx, my, mx),
                'rule_says_direct': bool(
                    _auto_selects_direct(ny, nx, my, mx)),
                'distinct_route_digests': len(set(dig.values())),
                'rounds': []})
            rec['rounds'].append({
                'round': rnd, 'best_s': best, 'fallback_s': fb,
                'dense_over_fallback': best['dense'] / fb,
                'ref_loop_drift': ref1 / ref0 if ref0 else -1.0,
                'faster_fallback': ('separable'
                                    if best['separable'] <= best['chirpz2d']
                                    else 'chirpz2d')})
            print(f"R{rnd} {name:16s} {ny}x{nx}->{my}x{mx} "
                  f"w/e={rec['work_per_kernel_entry']:8.2f} "
                  f"rule={'D' if rec['rule_says_direct'] else 'c'} "
                  f"dense={best['dense']:.4f} sep={best['separable']:.4f} "
                  f"chirp={best['chirpz2d']:.4f} "
                  f"d/fb={best['dense'] / fb:7.3f}", flush=True)
            del E
            gc.collect()
    for name, rec in per.items():
        rec['worst_dense_over_fallback'] = max(
            r['dense_over_fallback'] for r in rec['rounds'])
        rec['best_dense_over_fallback'] = min(
            r['dense_over_fallback'] for r in rec['rounds'])
        rec['safe_worst_round'] = bool(rec['worst_dense_over_fallback'] <= 1.0)
        out['rows'].append(rec)
    out['load_after'] = v4lib.load_census()
    slower = [(r['work_per_kernel_entry'], r['name'],
               r['worst_dense_over_fallback'])
              for r in out['rows'] if not r['safe_worst_round']]
    safe = [(r['work_per_kernel_entry'], r['name'],
             r['worst_dense_over_fallback'])
            for r in out['rows'] if r['safe_worst_round']]
    slower.sort()
    safe.sort()
    out['SLOWER_sorted_by_work_per_entry'] = slower
    out['SAFE_sorted_by_work_per_entry'] = safe
    out['largest_work_per_entry_measured_slower'] = (
        max(w for w, _n, _r in slower) if slower else None)
    above = [w for w, _n, _r in safe
             if slower and w > max(x for x, _n2, _r2 in slower)]
    out['smallest_work_per_entry_safe_above_it'] = (
        min(above) if above else None)
    print("LARGEST work/entry measured SLOWER:",
          out['largest_work_per_entry_measured_slower'], flush=True)
    print("SMALLEST work/entry SAFE above it :",
          out['smallest_work_per_entry_safe_above_it'], flush=True)
    print("SLOWER:", [(f"{w:.2f}", n, f"{r:.3f}") for w, n, r in slower],
          flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"r2_workladder_{family}_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    main(sys.argv[1], int(_arg('--rounds', 2)), int(_arg('--reps', 9)),
         _arg('--family', 'all'), int(_arg('--workers', 1)))
