"""WP-C4 task 1 -- the TIME ladder and the MEMORY ladder the boundary is read
from.

``N`` in {64,128,256,512,1024,2048} x ``M`` in {16,32,64,128,256,512,1024} --
42 shapes -- for each of the three routes through the SAME sum, best of five,
COLD (every registered library cache dropped and ``gc.collect()`` before each
repeat, which is the production regime ``_bluestein.py``'s own byte-cap comment
records with ``hits = 0`` across a full production order).

TIMING AND ``tracemalloc`` ARE SEPARATE PASSES, and separate INVOCATIONS of
this file (``--what time`` / ``--what mem``).  tracemalloc charges per
allocation and the chirp-Z route allocates far more objects, so timing inside
it would bias the very comparison the ladder is for.

THE LOAD IS RECORDED IN THE JSON, before and after, as a process census and a
measured single-thread reference loop.  The box carries other work; the
readings here are BOUNDS (a contended box can only make a route look slower),
and the quantity the boundary is read from is a RATIO taken under the same
conditions for all three routes, which is the robust part.

WHAT "the previous route" MEANS HERE.  ``method='auto'`` before this work
dispatched to the separable two-pass route when ``separable=True`` and
``xp is numpy``, and to the 2-D chirp-Z convolution otherwise.  Either can be
the route a given caller would have had, so the rule's premise -- "the dense
route is never slower" -- is taken against ``min(chirpz2d, separable)``, the
FASTER of the two fallbacks, at every shape.  That is the strictest reading and
the only one that makes the claim true whichever fallback the caller was on.

    PYTHONPATH=<tree> python c4_ladder.py <tree> --what time|mem
"""
from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

N_LADDER = (64, 128, 256, 512, 1024, 2048)
M_LADDER = (16, 32, 64, 128, 256, 512, 1024)
SHAPES = [(N, M) for N in N_LADDER for M in M_LADDER]

ROUTES = (('chirpz2d', dict(separable=False, method='bluestein')),
          ('separable', dict(separable=True, method='separable')),
          ('dense', dict(method='direct')))

# alpha is held at a value whose phase budget stays far under the guard at
# every shape on the ladder, so no route pays a warning and the timing is of
# the arithmetic and not of the warnings machinery.  budget = alpha*N_max^2,
# so alpha = 1/N_max^2 * 1e3 keeps it at 1e3 everywhere.


def _alpha(N, M):
    return 1.0e3 / float(max(N, M)) ** 2


def _time_one(E, N, M, kw, repeats=5):
    import numpy as np
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    a = _alpha(N, M)
    best = float('inf')
    for _ in range(repeats):
        c4lib.cold()
        t0 = time.perf_counter()
        _bluestein_2d(E, a, a, M, M, sign=-1, xp=np, fft2=_fft2,
                      ifft2=_ifft2, **kw)
        best = min(best, time.perf_counter() - t0)
    return best


def _peak_one(E, N, M, kw):
    import numpy as np
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    a = _alpha(N, M)
    c4lib.cold()
    tracemalloc.start()
    _bluestein_2d(E, a, a, M, M, sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2,
                  **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def main(tree, what):
    import numpy as np
    c4lib.anchor(tree)
    build = c4lib.build_tag()
    out = {'build': build, 'what': what, 'python': sys.version.split()[0],
           'load_before': c4lib.load_snapshot(),
           'n_ladder': list(N_LADDER), 'm_ladder': list(M_LADDER),
           'rows': []}
    rng = np.random.default_rng(20260920)
    for (N, M) in SHAPES:
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        row = {'N': N, 'M': M, 'ratio': M / N, 'alpha': _alpha(N, M),
               'budget': _alpha(N, M) * float(max(N, M)) ** 2}
        if what == 'time':
            ts = {name: _time_one(E, N, M, kw) for name, kw in ROUTES}
            fallback = min(ts['chirpz2d'], ts['separable'])
            row['best_of_5_s'] = ts
            row['fallback_s'] = fallback
            row['dense_over_fallback'] = ts['dense'] / fallback
            row['dense_never_slower'] = bool(ts['dense'] <= fallback)
            row['winner'] = min(ts, key=ts.get)
            print(f"TIME N={N:5d} M={M:5d} M/N={M/N:8.5f}  "
                  f"dense={ts['dense']:.4f} sep={ts['separable']:.4f} "
                  f"chirp={ts['chirpz2d']:.4f}  d/fb="
                  f"{row['dense_over_fallback']:6.3f} -> {row['winner']}",
                  flush=True)
        else:
            peaks = {name: _peak_one(E, N, M, kw) for name, kw in ROUTES}
            row['peak_bytes'] = peaks
            row['dense_cheapest'] = bool(
                peaks['dense'] < peaks['separable']
                and peaks['dense'] < peaks['chirpz2d'])
            row['ordering_holds'] = bool(
                peaks['dense'] < peaks['separable'] < peaks['chirpz2d'])
            print(f"MEM  N={N:5d} M={M:5d} M/N={M/N:8.5f}  "
                  f"dense={peaks['dense']/1e6:9.2f} "
                  f"sep={peaks['separable']/1e6:9.2f} "
                  f"chirp={peaks['chirpz2d']/1e6:9.2f}  "
                  f"{'dense-cheapest' if row['dense_cheapest'] else 'NO'}",
                  flush=True)
        out['rows'].append(row)
        del E
        gc.collect()
    out['load_after'] = c4lib.load_snapshot()
    tag = build.split('-')[0].lower()
    c4lib.write_json(out, os.path.join(HERE, f"c4_ladder_{what}_{tag}.json"))


if __name__ == '__main__':
    tree = sys.argv[1]
    what = sys.argv[sys.argv.index('--what') + 1] \
        if '--what' in sys.argv else 'time'
    main(tree, what)
