"""VERIFY-WP-C4 ROUND 2, item 1b -- an INDEPENDENT thin ladder across the
work/entry band the second condition's constant sits in.

WHAT IS DIFFERENT FROM THE BRANCH'S OWN LADDER.  ``r2_workladder.py`` spans
work/entry 1.25 .. 64 with 34 thin shapes, but inside the band that actually
decides the constant -- 6 to 30 -- it carries 14 shapes, of which 13 are the
SAME orientation (short x-axis).  A one-sided screen derived from one
orientation cannot see an asymmetry between the two matrix-product association
orders (the thin-x shapes all take ``x_first``, the thin-y shapes all take
``y_first``).  This ladder carries 26 shapes with work/entry in [6, 30], in
MATCHED PAIRS of both orientations, at three absolute sizes, plus four anchors
below the band and three square controls above it.

INSTRUMENT: two independent rounds of best-of-nine, routes interleaved with the
order rotating by repeat, every registered cache dropped and ``gc.collect()``
before every single timed call, all three routes' answers digested in the same
pass, the reference loop read before and after every shape,
``fft_infra.SCIPY_FFT_WORKERS = 1`` so both sides are single-threaded.  Verdict
on the WORST round.  Against ``min(chirp-Z 2-D, separable)`` -- the faster of
the two routes ``'auto'`` could otherwise have taken.

    PYTHONPATH=<tree> python vc4b_ladder.py <tree> [--rounds 2] [--reps 9]

Author:  Andrew Traverso
"""
from __future__ import annotations

import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

ROUTES = ('chirpz2d', 'separable', 'dense')
KW = {'chirpz2d': dict(separable=False, method='bluestein'),
      'separable': dict(separable=True, method='separable'),
      'dense': dict(method='direct')}

#: (name, Ny_in, Nx_in, My, Mx, group).  ``T`` = thin in x (tall input),
#: ``W`` = thin in y (wide input) -- the matched orientation of the row above.
#: ``sm`` / ``md`` / ``lg`` name the absolute size decade.
LADDER = [
    # ---- below the band: the region the screen must refuse ---------------
    ('a0125_T_lg',  4096,   32,  128,   1, 'below'),
    ('a0125_W_lg',    32, 4096,    1, 128, 'below'),
    ('a0299_T_sm',   512,   32,   16,   1, 'below'),
    ('a0300_W_md',    64, 2048,    1,  32, 'below'),
    # ---- the band [6, 30], matched orientations, three absolute sizes ----
    ('b0799_T_lg',  4096,  128,  128,   4, 'band'),
    ('b0799_W_lg',   128, 4096,    4, 128, 'band'),
    ('b0873_T_sm',   256,   64,    8,   1, 'band'),
    ('b0873_W_sm',    64,  256,    1,   8, 'band'),
    ('b0896_T_md',  2048,  128,   16,   1, 'band'),
    ('b0896_W_md',   128, 2048,    1,  16, 'band'),
    ('b0985_T_sm',   512,   64,   16,   2, 'band'),
    ('b0985_W_sm',    64,  512,    2,  16, 'band'),
    ('b0992_T_md',  1024,   64,   16,   2, 'band'),
    ('b0992_W_md',    64, 1024,    2,  16, 'band'),
    ('b1195_T_md',  2048,  128,   64,   4, 'band'),
    ('b1195_W_md',   128, 2048,    4,  64, 'band'),
    ('b1198_T_lg',  4096,  256,  128,   4, 'band'),
    ('b1198_W_lg',   256, 4096,    4, 128, 'band'),
    ('b1511_T_sm',   128,   64,    4,   1, 'band'),
    ('b1511_W_sm',    64,  128,    1,   4, 'band'),
    ('b1600_T_sm',   256,   64,    4,   1, 'band'),
    ('b1600_W_sm',    64,  256,    1,   4, 'band'),
    ('b1687_T_md',  2048,  256,   16,   1, 'band'),
    ('b1687_W_md',   256, 2048,    1,  16, 'band'),
    ('b1793_T_lg',  4096, 1024,  128,   2, 'band'),
    ('b1793_W_lg',  1024, 4096,    2, 128, 'band'),
    ('b1969_T_md',  1024,  128,   32,   4, 'band'),
    ('b1969_W_md',   128, 1024,    4,  32, 'band'),
    ('b2391_T_lg',  4096,  256,  128,   8, 'band'),
    ('b2391_W_lg',   256, 4096,    8, 128, 'band'),
    ('b2933_T_sm',   256,  128,    4,   1, 'band'),
    ('b2933_W_sm',   128,  256,    1,   4, 'band'),
    # ---- above the band: the shapes the shipped suite actually drives ----
    ('c0264_sq_md',  512,  512,   16,  16, 'control'),
    ('c0528_sq_md', 1024, 1024,   32,  32, 'control'),
    ('c1056_sq_lg', 2048, 2048,   64,  64, 'control'),
]


def work_per_entry(ny, nx, my, mx):
    """The quantity the second condition reads, re-derived from the two cost
    expressions ``_direct_matrix_2d`` itself compares (verified against the
    live call by ``vc4b_formula.py``)."""
    return (min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)
            / (my * ny + mx * nx))


def alpha_for(ny, nx, my, mx):
    """Hold the phase budget at 1e3 -- six decades under
    ``_PHASE_BUDGET_MAX``, so no route pays the warning and the clock times
    the arithmetic and nothing else."""
    return 1.0e3 / float(max(ny, nx, my, mx)) ** 2


def main(tree, rounds, reps):
    import numpy as np
    L.anchor(tree)
    workers = L.single_thread_ffts()
    from lumenairy.propagators._bluestein import (
        _MFT_DIRECT_MAX_RATIO, _MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY,
        _auto_selects_direct, _bluestein_2d)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    out = {'build': L.build(), 'python': sys.version.split()[0],
           'numpy': np.__version__, 'rounds': rounds, 'reps': reps,
           'scipy_fft_workers': workers,
           'constant_max_ratio': float(_MFT_DIRECT_MAX_RATIO),
           'constant_min_work': float(_MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY),
           'instrument': 'third instrument; interleaved best-of-N, route '
                         'order rotates per rep, cold before every timed '
                         'call, workers=1 on both sides, worst round',
           'rows': [], 'round_load': []}
    rng = np.random.default_rng(20260920)
    per = {}
    for rnd in range(rounds):
        out['round_load'].append(L.load())
        for (name, ny, nx, my, mx, grp) in LADDER:
            E = (rng.standard_normal((ny, nx))
                 + 1j * rng.standard_normal((ny, nx))).astype(np.complex128)
            a = alpha_for(ny, nx, my, mx)
            r0 = L.refloop()
            best = {r: float('inf') for r in ROUTES}
            dig = {}
            for rep in range(reps):
                order = ROUTES[rep % 3:] + ROUTES[:rep % 3]
                for r in order:
                    L.cold()
                    t0 = time.perf_counter()
                    F = _bluestein_2d(E, a, a, my, mx, sign=-1, xp=np,
                                      fft2=_fft2, ifft2=_ifft2, **KW[r])
                    best[r] = min(best[r], time.perf_counter() - t0)
                    if r not in dig:
                        dig[r] = L.digest(F)
                    del F
            r1 = L.refloop()
            fb = min(best['chirpz2d'], best['separable'])
            rec = per.setdefault(name, {
                'name': name, 'group': grp,
                'Ny_in': ny, 'Nx_in': nx, 'My': my, 'Mx': mx,
                'in_elements': ny * nx,
                'ratio_max': max(my / ny, mx / nx),
                'work_per_entry': work_per_entry(ny, nx, my, mx),
                'rule_says_direct': bool(_auto_selects_direct(ny, nx, my, mx)),
                'distinct_route_digests': len(set(dig.values())),
                'rounds': []})
            rec['rounds'].append({
                'round': rnd, 'best_s': dict(best), 'fallback_s': fb,
                'dense_over_fallback': best['dense'] / fb,
                'ref_loop_drift': r1 / r0 if r0 else -1.0,
                'faster_fallback': ('separable'
                                    if best['separable'] <= best['chirpz2d']
                                    else 'chirpz2d')})
            print("R%d %-14s %dx%d->%dx%-4d w/e=%8.2f rule=%s dense=%.5f "
                  "sep=%.5f chirp=%.5f d/fb=%8.3f dig=%d"
                  % (rnd, name, ny, nx, my, mx, rec['work_per_entry'],
                     'D' if rec['rule_says_direct'] else 'c',
                     best['dense'], best['separable'], best['chirpz2d'],
                     best['dense'] / fb, rec['distinct_route_digests']),
                  flush=True)
            del E
            gc.collect()

    for name, rec in per.items():
        rec['worst_dense_over_fallback'] = max(
            r['dense_over_fallback'] for r in rec['rounds'])
        rec['best_dense_over_fallback'] = min(
            r['dense_over_fallback'] for r in rec['rounds'])
        rec['safe_worst_round'] = bool(rec['worst_dense_over_fallback'] <= 1.0)
        out['rows'].append(rec)
    out['load_after'] = L.load()

    slower = sorted((r['work_per_entry'], r['name'],
                     r['worst_dense_over_fallback'])
                    for r in out['rows'] if not r['safe_worst_round'])
    safe = sorted((r['work_per_entry'], r['name'],
                   r['worst_dense_over_fallback'])
                  for r in out['rows'] if r['safe_worst_round'])
    out['SLOWER'] = slower
    out['SAFE'] = safe
    out['largest_work_per_entry_measured_slower'] = (
        max(w for w, _n, _r in slower) if slower else None)
    above = [w for w, _n, _r in safe
             if slower and w > max(x for x, _n2, _r2 in slower)]
    out['smallest_work_per_entry_safe_above_it'] = min(above) if above else None
    cap = [r for r in out['rows'] if r['rule_says_direct']]
    out['captured_shapes'] = len(cap)
    out['captured_worst_dense_over_fallback'] = (
        max(r['worst_dense_over_fallback'] for r in cap) if cap else None)
    out['captured_best_dense_over_fallback'] = (
        min(r['worst_dense_over_fallback'] for r in cap) if cap else None)
    out['CAPTURED_AND_SLOWER'] = [
        (r['name'], r['work_per_entry'], r['worst_dense_over_fallback'])
        for r in cap if not r['safe_worst_round']]
    out['SAFE_BUT_REFUSED'] = [
        (r['name'], r['work_per_entry'], r['worst_dense_over_fallback'])
        for r in out['rows']
        if r['safe_worst_round'] and not r['rule_says_direct']]
    print("largest w/e measured SLOWER :",
          out['largest_work_per_entry_measured_slower'])
    print("smallest w/e SAFE above it  :",
          out['smallest_work_per_entry_safe_above_it'])
    print("captured shapes             :", out['captured_shapes'],
          "worst d/fb", out['captured_worst_dense_over_fallback'])
    print("CAPTURED AND SLOWER         :", out['CAPTURED_AND_SLOWER'])
    print("SAFE BUT REFUSED            :", len(out['SAFE_BUT_REFUSED']))
    L.write(out, os.path.join(HERE, "vc4b_ladder_%s.json" % L.tag()))


if __name__ == '__main__':
    main(sys.argv[1], int(L.arg('--rounds', 2)), int(L.arg('--reps', 9)))
