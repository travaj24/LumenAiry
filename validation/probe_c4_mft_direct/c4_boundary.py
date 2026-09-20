"""WP-C4 task 1 -- the boundary shapes, re-measured on their own.

The 42-shape ladder (``c4_ladder.py``) is one run per shape.  The BOUNDARY is
decided by a handful of shapes, and one of them -- WSL, ``N = 1024``,
``M = 64``, i.e. ``M/N = 1/16`` -- is where the two EARLIER campaigns already
disagreed with each other:

* ``WAVE5_HYGIENE2_REPORT.md``'s WSL table reads separable 0.0100 against
  dense 0.0118 there (dense LOSES), while
* ``VERIFY_WAVE5_HYGIENE2.md``'s re-measurement reads dense 0.0185 against
  separable 0.0204 at the same shape (dense WINS).

A constant derived from a shape whose two prior measurements point opposite
ways is exactly what ``docs/TESTING_STANDARDS.md`` calls an S1 pin.  So the
shapes that decide the boundary are measured again here, THREE independent
rounds of best-of-nine each, with the load snapshotted per round, and the
decision is taken on the WORST round -- a bound, not an average.

The comparison is against ``min(chirp-Z 2-D, separable)``: either is a route
``method='auto'`` could otherwise have taken, so "never slower" has to hold
against the faster of them.

    PYTHONPATH=<tree> python c4_boundary.py <tree>
"""
from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: ``(ratio_label, N, M)`` -- two octaves either side of the candidate.
CASES = [
    ('1/64', 1024, 16), ('1/64', 2048, 32),
    ('1/32', 512, 16), ('1/32', 1024, 32), ('1/32', 2048, 64),
    ('1/16', 512, 32), ('1/16', 1024, 64), ('1/16', 2048, 128),
    ('1/8', 512, 64), ('1/8', 1024, 128), ('1/8', 2048, 256),
]

ROUTES = (('chirpz2d', dict(separable=False, method='bluestein')),
          ('separable', dict(separable=True, method='separable')),
          ('dense', dict(method='direct')))

ROUNDS = 3
REPEATS = 9


def _alpha(N, M):
    """Phase budget pinned at 1e3 -- five decades under the guard at every
    shape, so no route pays a warning and the timing is of the arithmetic."""
    return 1.0e3 / float(max(N, M)) ** 2


def _time_one(E, N, M, kw, repeats):
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


def main(tree):
    import numpy as np
    c4lib.anchor(tree)
    build = c4lib.build_tag()
    out = {'build': build, 'python': sys.version.split()[0],
           'rounds': ROUNDS, 'repeats': REPEATS,
           'load_before': c4lib.load_snapshot(), 'loads': [], 'cases': []}
    rng = np.random.default_rng(496)
    per = {}
    for rnd in range(ROUNDS):
        out['loads'].append(c4lib.load_snapshot())
        for (lab, N, M) in CASES:
            E = (rng.standard_normal((N, N))
                 + 1j * rng.standard_normal((N, N))).astype(np.complex128)
            ts = {name: _time_one(E, N, M, kw, REPEATS)
                  for name, kw in ROUTES}
            fb = min(ts['chirpz2d'], ts['separable'])
            r = ts['dense'] / fb
            per.setdefault((lab, N, M), []).append(
                {'round': rnd, 'best_of_n_s': ts, 'fallback_s': fb,
                 'dense_over_fallback': r,
                 'dense_never_slower': bool(ts['dense'] <= fb)})
            print(f"round {rnd}  {lab:>5}  N={N:5d} M={M:5d}  "
                  f"dense={ts['dense']:.4f} sep={ts['separable']:.4f} "
                  f"chirp={ts['chirpz2d']:.4f}  d/fb={r:6.3f}  "
                  f"{'ok' if ts['dense'] <= fb else 'SLOWER'}", flush=True)
            del E
            gc.collect()
    for (lab, N, M), rows in per.items():
        worst = max(r['dense_over_fallback'] for r in rows)
        out['cases'].append({
            'ratio_label': lab, 'N': N, 'M': M, 'ratio': M / N,
            'rounds': rows, 'worst_dense_over_fallback': worst,
            'never_slower_every_round': all(r['dense_never_slower']
                                            for r in rows)})
    out['load_after'] = c4lib.load_snapshot()
    # the boundary, read off the worst round at every shape
    by_ratio = {}
    for c in out['cases']:
        by_ratio.setdefault(c['ratio_label'], []).append(c)
    verdict = {lab: {'worst': max(c['worst_dense_over_fallback'] for c in cs),
                     'safe': all(c['never_slower_every_round'] for c in cs)}
               for lab, cs in by_ratio.items()}
    out['verdict_by_ratio'] = verdict
    print("\nVERDICT (worst of every round, this build):")
    for lab in ('1/64', '1/32', '1/16', '1/8'):
        v = verdict.get(lab)
        if v:
            print(f"  M/N = {lab:>5}  worst d/fb = {v['worst']:6.3f}  "
                  f"{'SAFE' if v['safe'] else 'NOT SAFE'}")
    tag = build.split('-')[0].lower()
    c4lib.write_json(out, os.path.join(HERE, f"c4_boundary_{tag}.json"))


if __name__ == '__main__':
    main(sys.argv[1])
