"""VERIFY-WP-B7c round 3 -- the FALLBACK sweep (claim 8).

The ladder's tail already reaches past both edges of the completion band; this
adds the rest of the propagation axis, from well short of the first focus to
well past the last, so the fallback population is not selected by how close it
sits to the fold ring.  The z list is DERIVED from the same geometry the
ladders are: the paraxial and marginal foci, and the interior-turning-point
window between them.

Usage:  python v3fbsweep.py <optic> <out.json> [--n 18]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import v3fixtures as FX
import v3geom as G
import v3scan as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=18)
    ap.add_argument('--refine', type=int, default=3)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    fx = FX.FIXTURES[a.fixture]
    presc, wl = fx['prescription'], fx['wavelength']
    _na, fp, fm, _h, _f = G.na_and_foci(presc, wl)
    zs_geom = np.linspace(0.55 * min(fp, fm), 1.45 * max(fp, fm), 120)
    zt = [z for z, rc, _ in G.fold_window(presc, wl, zs_geom) if rc]
    lo, hi = (min(zt), max(zt)) if zt else (0.9 * fm, 1.05 * fp)
    zs = np.concatenate([
        np.linspace(0.40 * min(fp, fm), lo * 0.995, a.n // 2),
        np.linspace(hi * 1.005, 1.60 * max(fp, fm), a.n - a.n // 2)])
    print('--- fallback sweep %s: %d planes, %.1f .. %.1f um'
          % (fx['name'], zs.size, zs.min() * 1e6, zs.max() * 1e6), flush=True)
    rows = S.scan(fx, zs, oracle='asm', refine=a.refine)
    with open(a.out, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__,
                       python=sys.version.split()[0], numpy=np.__version__,
                       fixture=fx['name'], bars=S.bars(), rows=rows),
                  f, indent=1)
    print('wrote', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
