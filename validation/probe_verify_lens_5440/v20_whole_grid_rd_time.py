"""V20 -- is the WHOLE-GRID ray-density call itself slower in 5.44.0?

V15 measured, inside one process at N=4096: on v5.43.0 the whole-grid
ray-density call ran 21.1 s against 20.2 s for whole-grid screen (+4.5 %);
on 50824e9 it ran 28.5 s against 20.3 s (+40 %).  The whole-grid path is
byte-identical across the release (V1, V8), so this isolates the timing in a
fresh process with nothing else run first.

Usage:  python v20_whole_grid_rd_time.py <out.json> [N] [sub] [dx_um] [reps]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402


def main():
    la = _fix.banner()
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    sub = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    dx = (float(sys.argv[4]) if len(sys.argv) > 4 else 1.5) * 1e-6
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    from lumenairy.elements import _lens_imap as IM
    E = _fix.gauss(N, dx, 1.0e-3)
    out = {'version': la.__version__, 'N': N, 'sub': sub, 'cases': {}}
    for model in ('ray_density', 'screen'):
        kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
                  wavelength=_fix.WL, dx=dx, ray_subsample=sub, n_workers=1,
                  on_undersample='silent', on_noncollimated='off',
                  on_aperture_beam='silent', parallel_amp=False,
                  amplitude_model=model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(E, sag_chunk_rows=0, **kw)     # warm
            ts = []
            for _ in range(reps):
                IM.inverse_map_cache_clear()
                t = time.perf_counter()
                la.apply_real_lens_traced(E, sag_chunk_rows=0, **kw)
                ts.append(time.perf_counter() - t)
            IM.inverse_map_cache_clear()
        out['cases'][model] = {'best': min(ts), 'all': ts}
        print('whole-grid %-12s best %7.3f s  (%s)'
              % (model, min(ts), ', '.join('%.2f' % v for v in ts)),
              flush=True)
    out['rd_over_screen'] = (out['cases']['ray_density']['best']
                             / out['cases']['screen']['best'])
    print('rd / screen = %.3f' % out['rd_over_screen'], flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
