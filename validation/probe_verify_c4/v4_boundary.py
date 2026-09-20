"""VERIFY-WP-C4 claim 1b -- the deciding shapes, re-measured on their own.

Two families:

* ``REPRO`` -- the shapes ``probe_c4_mft_direct/c4_boundary.py`` read the
  constant off, so its table can be reproduced or refuted with a second
  instrument;
* ``THIN`` -- ANISOTROPIC inputs whose SHORT axis is small while both
  output/input ratios sit at or under 1/32, i.e. shapes the shipped rule
  CAPTURES and the branch's square-only ladder never timed.  The dense route
  pays ``O(My*Ny + Mx*Nx)`` complex ``exp`` calls to build its two kernels and
  only ``O(My*Ny*Nx)`` multiply-adds to use them, so the build dominates once
  the OTHER axis is short -- which is a regime a ratio cannot see.

THREE independent rounds of best-of-nine each, the load snapshotted per round,
the verdict taken on the WORST round -- the same protocol the branch used, so
the two tables are comparable line for line.  The route order rotates within
each repeat (see ``v4lib``), which the branch's instrument does not do.

    PYTHONPATH=<tree> python v4_boundary.py <tree> [--rounds 3] [--reps 9]
                                                   [--family repro|thin|all] [--workers 1]
                                        [--instrument interleaved|blocked]
"""
from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402
from v4_ladder import KW, alpha_for, run_once  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ('chirpz2d', 'separable', 'dense')

REPRO = [
    ('r_1024_16',  1024, 1024,  16,  16),   # 1/64
    ('r_2048_32',  2048, 2048,  32,  32),   # 1/64
    ('r_512_16',    512,  512,  16,  16),   # 1/32
    ('r_1024_32',  1024, 1024,  32,  32),   # 1/32  branch's thin margin
    ('r_2048_64',  2048, 2048,  64,  64),   # 1/32
    ('r_512_32',    512,  512,  32,  32),   # 1/16
    ('r_1024_64',  1024, 1024,  64,  64),   # 1/16  branch: WSL 1.450 SLOWER
    ('r_2048_128', 2048, 2048, 128, 128),   # 1/16
    ('r_512_64',    512,  512,  64,  64),   # 1/8
    ('r_1024_128', 1024, 1024, 128, 128),   # 1/8
    ('r_2048_256', 2048, 2048, 256, 256),   # 1/8
    ('r_mixed_1_64_1_8', 1024, 1024, 16, 128),   # (1/64, 1/8): MAX refuses
]

THIN = [
    # every one of these is CAPTURED by the shipped rule (both ratios 1/32)
    ('t_2048x256', 2048,  256,  64,   8),
    ('t_2048x128', 2048,  128,  64,   4),
    ('t_2048x64',  2048,   64,  64,   2),
    ('t_2048x32',  2048,   32,  64,   1),
    ('t_1024x128', 1024,  128,  32,   4),
    ('t_1024x64',  1024,   64,  32,   2),
    ('t_1024x32',  1024,   32,  32,   1),
    ('t_512x64',    512,   64,  16,   2),
    ('t_512x32',    512,   32,  16,   1),
    ('t_64x2048',    64, 2048,   2,  64),   # the transpose of t_2048x64
    ('t_32x2048',    32, 2048,   1,  64),
    ('t_4096x64',  4096,   64, 128,   2),
]


#: The same anisotropy at HALF the boundary ratio (1/64 on both axes).  If
#: tightening the RATIO were the fix for the thin-input failures, these would
#: be safe; they are the control that says it is not.
THIN64 = [
    ('h_4096x64',  4096,   64,  64,   1),
    ('h_2048x64',  2048,   64,  32,   1),
    ('h_2048x128', 2048,  128,  32,   2),
    ('h_64x4096',    64, 4096,   1,  64),
    ('h_4096x128', 4096,  128,  64,   2),
    ('h_1024x64',  1024,   64,  16,   1),
]


#: Shapes a MIN conjunction would take and MAX refuses -- one axis far under
#: the boundary, the other at or above 1.  If MIN were safe these would all be
#: faster on the dense route; they are the two-sided arm for the conjunction.
MINARM = [
    ('m_1024_8x1024',  1024, 1024,    8, 1024),
    ('m_512_8x512',     512,  512,    8,  512),
    ('m_1024x64_16x1024', 1024, 64,  16, 1024),
    ('m_2048_16x2048', 2048, 2048,   16, 2048),
    ('m_1024_1024x8',  1024, 1024, 1024,    8),
]


def main(tree, rounds, reps, family, workers, instrument):
    import numpy as np
    from lumenairy.propagators import fft_infra
    from lumenairy.propagators._bluestein import (
        _auto_selects_direct, _MFT_DIRECT_MAX_RATIO)
    v4lib.anchor(tree)
    # THE THREAD ASYMMETRY, made explicit.  ``OMP_NUM_THREADS=1
    # OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`` pins the DENSE route (two
    # BLAS products) to one thread and does NOT constrain scipy's pocketfft,
    # which ``fft_infra.SCIPY_FFT_WORKERS = -1`` gives every core.  So the
    # shipped comparison races a one-thread dense route against an
    # all-core chirp-Z route, and its outcome moves with how many cores the
    # box has spare.  ``--workers 1`` runs the SAME ladder with both sides
    # single-threaded; the two arms are reported separately.
    if workers:
        fft_infra.SCIPY_FFT_WORKERS = int(workers)
    shapes = (REPRO if family == 'repro' else
              THIN if family == 'thin' else
              THIN64 if family == 'thin64' else
              MINARM if family == 'minarm' else REPRO + THIN)
    out = {'build': v4lib.build_tag(), 'rounds': rounds, 'reps': reps,
           'family': family, 'constant': float(_MFT_DIRECT_MAX_RATIO),
           'scipy_fft_workers': int(fft_infra.SCIPY_FFT_WORKERS),
           'instrument': instrument,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': [], 'round_load': []}
    rng = np.random.default_rng(4242)
    per = {}
    for rnd in range(rounds):
        out['round_load'].append(v4lib.load_census())
        for (name, ny, nx, my, mx) in shapes:
            E = (rng.standard_normal((ny, nx))
                 + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
            a = alpha_for(ny, nx, my, mx)
            best = {r: float('inf') for r in ROUTES}
            if instrument == 'blocked':
                # the BRANCH's arrangement: all repeats of one route, then
                # the next.  Kept as an arm because it is the instrument the
                # shipped constant was read with, and because it is the one
                # that gives the FFT routes the warmest plan cache and the
                # most-awake pocketfft worker pool.
                for r in ROUTES:
                    for _rep in range(reps):
                        v4lib.cold()
                        t0 = time.perf_counter()
                        F = run_once(E, a, my, mx, r)
                        best[r] = min(best[r], time.perf_counter() - t0)
                        del F
            else:
                for rep in range(reps):
                    order = ROUTES[rep % 3:] + ROUTES[:rep % 3]
                    for r in order:
                        v4lib.cold()
                        t0 = time.perf_counter()
                        F = run_once(E, a, my, mx, r)
                        best[r] = min(best[r], time.perf_counter() - t0)
                        del F
            fb = min(best['chirpz2d'], best['separable'])
            rec = per.setdefault(name, {
                'name': name, 'Ny_in': ny, 'Nx_in': nx, 'My': my, 'Mx': mx,
                'ratio_y': my / ny, 'ratio_x': mx / nx,
                'ratio_max': max(my / ny, mx / nx),
                'rule_says_direct': bool(
                    _auto_selects_direct(ny, nx, my, mx)),
                'rounds': []})
            rec['rounds'].append({'round': rnd, 'best_s': best,
                                  'fallback_s': fb,
                                  'dense_over_fallback': best['dense'] / fb,
                                  'faster_fallback': ('separable'
                                                      if best['separable']
                                                      <= best['chirpz2d']
                                                      else 'chirpz2d')})
            print(f"R{rnd} {name:14s} {ny}x{nx}->{my}x{mx} "
                  f"rule={'D' if rec['rule_says_direct'] else 'c'} "
                  f"dense={best['dense']:.4f} sep={best['separable']:.4f} "
                  f"chirp={best['chirpz2d']:.4f} d/fb="
                  f"{best['dense']/fb:7.3f}", flush=True)
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
    bad = [r['name'] for r in out['rows']
           if r['rule_says_direct'] and not r['safe_worst_round']]
    out['CAPTURED_AND_SLOWER'] = bad
    print("CAPTURED AND SLOWER (worst round):", bad, flush=True)
    wtag = (f"_w{workers}" if workers else "") + (
        "_blocked" if instrument == 'blocked' else "")
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_boundary_{family}{wtag}_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    main(sys.argv[1], int(_arg('--rounds', 3)), int(_arg('--reps', 9)),
         _arg('--family', 'all'), _arg('--workers', ''),
         _arg('--instrument', 'interleaved'))
