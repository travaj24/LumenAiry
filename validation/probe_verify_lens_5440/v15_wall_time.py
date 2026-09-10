"""V15 -- WALL TIME of the banded traced call, both builds, both models.

The CHANGELOG prices the change as "neutral on the coarse route and ~10 %
higher on the evaluator route (the 7/4 evaluation)".  That prices the
RAY-DENSITY route against itself.  It does NOT price the route the
production runner actually drives: a banded SCREEN call at the shipped
default, which on v5.43.0 ran the coarse-Newton incumbent and on 5.44.0 runs
the evaluator.  This measures both, on both builds, at the AUTO band height,
with no ``tracemalloc`` attached.

Usage:  python v15_wall_time.py <out.json> [N] [sub] [dx_um] [reps]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402


def main():
    la = _fix.banner()
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    sub = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    dx = (float(sys.argv[4]) if len(sys.argv) > 4 else 1.5) * 1e-6
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    auto_rows = max(256, N // 16)
    from lumenairy.elements import _lens_imap as IM
    E = _fix.gauss(N, dx, 1.0e-3)
    out = {'version': la.__version__, 'N': N, 'sub': sub, 'dx': dx,
           'auto_rows': auto_rows, 'cases': {}}
    for model in ('screen', 'ray_density'):
        kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
                  wavelength=_fix.WL, dx=dx, ray_subsample=sub, n_workers=1,
                  on_undersample='silent', on_noncollimated='off',
                  on_aperture_beam='silent', parallel_amp=False,
                  amplitude_model=model)
        for tag, rows in (('whole', 0), ('auto_band', auto_rows)):
            rec = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                IM.inverse_map_cache_clear()
                la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
                ts = []
                for _ in range(reps):
                    IM.inverse_map_cache_clear()
                    t = time.perf_counter()
                    la.apply_real_lens_traced(E, sag_chunk_rows=rows,
                                              _imap_out=rec, **kw)
                    ts.append(time.perf_counter() - t)
                IM.inverse_map_cache_clear()
            key = '%s.%s' % (model, tag)
            out['cases'][key] = {'rows': rows, 'best': min(ts), 'all': ts,
                                 'engaged': bool(rec.get('engaged', False))}
            print('%-24s rows=%-5s best %7.3f s  eng=%s  (%s)'
                  % (key, rows, min(ts), out['cases'][key]['engaged'],
                     ', '.join('%.2f' % v for v in ts)), flush=True)
        b = out['cases']['%s.auto_band' % model]['best']
        w = out['cases']['%s.whole' % model]['best']
        out['cases']['%s.band_over_whole' % model] = b / w
        print('  %s band/whole = %.3f' % (model, b / w), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
