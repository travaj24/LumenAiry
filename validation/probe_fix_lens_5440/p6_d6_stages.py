"""P6 (D6) -- the per-stage split of the evaluator route, one level finer.

P3 showed that at N=4096 only 0.88 s of the banded SCREEN call's 18.5 s is
inside ``InverseCharacteristic.eval_into``, so "the 7/4 evaluation" cannot be
what the route costs.  This adds the two stages that were folded into
``rest``:

  * ``InverseCharacteristic.domain_mask`` -- the screened hull test, which the
    evaluator route runs over EVERY exit pixel and the coarse-Newton incumbent
    does not run at all;
  * ``build_inverse_map`` -- the model fit and its guards, once per call
    (the cache is cleared before every measured call, as V15 did).

and prints total = eval_into + domain_mask + build + map_coordinates + rest.

Usage:  python p6_d6_stages.py <out.json> [N] [sub] [dx_um] [reps]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE),
                                'probe_verify_lens_5440'))
import _fix  # noqa: E402
import _fixp  # noqa: E402


class _Stages:
    def __init__(self, IM):
        self.IM = IM
        self.d = {'eval_calls': 0, 'eval_channels': 0, 'eval_s': 0.0,
                  'dm_calls': 0, 'dm_pixels': 0, 'dm_s': 0.0,
                  'build_calls': 0, 'build_s': 0.0,
                  'mc_calls': 0, 'mc_s': 0.0}
        self._saved = []

    def __enter__(self):
        import scipy.ndimage as ND
        d = self.d
        IM = self.IM
        cls = IM.InverseCharacteristic

        real_eval = cls.eval_into

        def eval_into(slf, Xg, Yg, out, channels=None, **kw):
            t = time.perf_counter()
            r = real_eval(slf, Xg, Yg, out, channels=channels, **kw)
            d['eval_s'] += time.perf_counter() - t
            d['eval_calls'] += 1
            n = int(np.asarray(Xg).size)
            d['eval_channels'] += n * (len(channels) if channels is not None
                                       else len(out))
            return r
        cls.eval_into = eval_into
        self._saved.append((cls, 'eval_into', real_eval))

        real_dm = cls.domain_mask

        def domain_mask(slf, Xg, Yg, *a, **kw):
            t = time.perf_counter()
            r = real_dm(slf, Xg, Yg, *a, **kw)
            d['dm_s'] += time.perf_counter() - t
            d['dm_calls'] += 1
            d['dm_pixels'] += int(np.asarray(Xg).size)
            return r
        cls.domain_mask = domain_mask
        self._saved.append((cls, 'domain_mask', real_dm))

        real_build = IM.build_inverse_map

        def build_inverse_map(*a, **kw):
            t = time.perf_counter()
            r = real_build(*a, **kw)
            d['build_s'] += time.perf_counter() - t
            d['build_calls'] += 1
            return r
        IM.build_inverse_map = build_inverse_map
        self._saved.append((IM, 'build_inverse_map', real_build))

        real_mc = ND.map_coordinates

        def map_coordinates(*a, **kw):
            t = time.perf_counter()
            r = real_mc(*a, **kw)
            d['mc_s'] += time.perf_counter() - t
            d['mc_calls'] += 1
            return r
        ND.map_coordinates = map_coordinates
        self._saved.append((ND, 'map_coordinates', real_mc))
        return self

    def __exit__(self, *exc):
        for obj, nm, real in self._saved:
            setattr(obj, nm, real)
        return False


def main():
    la = _fixp.banner(expect=None)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    sub = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    dx = (float(sys.argv[4]) if len(sys.argv) > 4 else 1.5) * 1e-6
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    auto = max(256, N // 16)
    from lumenairy.elements import _lens_imap as IM
    E = _fix.gauss(N, dx, 1.0e-3)
    print('# free RAM %.1f GB  N=%d sub=%d dx=%.2f um auto=%d reps=%d'
          % (_fixp.free_gb(), N, sub, dx * 1e6, auto, reps), flush=True)
    out = {'version': la.__version__, 'file': la.__file__, 'N': N, 'sub': sub,
           'dx': dx, 'auto_rows': auto, 'cases': {}}
    cases = []
    for model in ('screen', 'ray_density'):
        cases.append((model, 'whole', 0, None))
        cases.append((model, 'auto_band', auto, None))
        cases.append((model, 'auto_band_incumbent', auto, False))
    for model, tag, rows, imap in cases:
        kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
                  wavelength=_fix.WL, dx=dx, ray_subsample=sub, n_workers=1,
                  on_undersample='silent', on_noncollimated='off',
                  on_aperture_beam='silent', parallel_amp=False,
                  amplitude_model=model)
        if imap is not None:
            kw['inverse_map'] = imap
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)     # warm
            for _ in range(reps):
                IM.inverse_map_cache_clear()
                with _Stages(IM) as st:
                    t = time.perf_counter()
                    f = la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
                    tot = time.perf_counter() - t
                if best is None or tot < best[0]:
                    best = (tot, dict(st.d), _fixp.h(np.asarray(f)))
                del f
            IM.inverse_map_cache_clear()
    # ---- record
        tot, d, hsh = best
        d['total_s'] = tot
        d['rest_s'] = tot - d['eval_s'] - d['dm_s'] - d['build_s'] - d['mc_s']
        d['hash'] = hsh
        d['rows'] = rows
        d['inverse_map_kw'] = imap
        d['dm_grids'] = d['dm_pixels'] / float(N * N)
        d['eval_ch_per_px'] = d['eval_channels'] / float(N * N)
        out['cases']['%s.%s' % (model, tag)] = d
        print('%-32s tot %7.3f = eval %5.2f (%d calls, %.2f ch/px) + dmask '
              '%5.2f (%d calls, %.2f grids) + build %5.2f + mc %5.2f + rest '
              '%6.2f   %s'
              % ('%s.%s' % (model, tag), tot, d['eval_s'], d['eval_calls'],
                 d['eval_ch_per_px'], d['dm_s'], d['dm_calls'], d['dm_grids'],
                 d['build_s'], d['mc_s'], d['rest_s'], hsh), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
