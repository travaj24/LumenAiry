"""P3 (D6) -- WHERE the banded SCREEN route's +108 % went, per stage.

VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D6: at the shipped default the banded
SCREEN call went 10.529 s -> 21.890 s across the release (N=4096, sub=32,
dx=1.5 um, AUTO band height), whole-grid controls unchanged.  The verification
ATTRIBUTES that to the route change -- the banded screen call now runs the
inverse-characteristic evaluator where v5.43.0 silently ran the coarse-Newton
incumbent.  This CONFIRMS the attribution by measurement instead:

  * ``InverseCharacteristic.eval_into`` call and channel-evaluation counts,
    and the seconds inside it, per route;
  * ``scipy.ndimage.map_coordinates`` calls and seconds (the incumbent's
    upsample stage, and the band loop's);
  * ``rest`` = total - the two above;
  * the CONTROL that isolates the route from the banding: 5.44.0's banded
    screen call with ``inverse_map=False``, i.e. the SAME incumbent v5.43.0
    ran, on the SAME build.  If the +108 % is the route, that control lands
    on v5.43.0's banded number and not on 5.44.0's.

Run on this tree and on a v5.43.0 worktree; ``--counts`` adds the
instrumentation (which perturbs wall time, so the timing rows are taken with
it OFF and one instrumented rep is taken afterwards).

Usage:  python p3_d6_time.py <out.json> [N] [sub] [dx_um] [reps]
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


def _kw(model, sub, dx, imap):
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=dx, ray_subsample=sub, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False, amplitude_model=model)
    if imap is not None:
        kw['inverse_map'] = imap
    return kw


class _Stages:
    """Accumulate per-stage calls / seconds by wrapping the two stages that
    can carry a route difference."""

    def __init__(self, IM):
        self.IM = IM
        self.d = {'eval_calls': 0, 'eval_pixels': 0, 'eval_channels': 0,
                  'eval_s': 0.0, 'mc_calls': 0, 'mc_s': 0.0}
        self._saved = []

    def __enter__(self):
        import scipy.ndimage as ND
        d = self.d
        cls = self.IM.InverseCharacteristic
        real_eval = cls.eval_into

        def eval_into(slf, Xg, Yg, out, channels=None, **kw):
            t = time.perf_counter()
            r = real_eval(slf, Xg, Yg, out, channels=channels, **kw)
            d['eval_s'] += time.perf_counter() - t
            d['eval_calls'] += 1
            n = int(np.asarray(Xg).size)
            d['eval_pixels'] += n
            d['eval_channels'] += n * (len(channels) if channels is not None
                                       else len(out))
            return r
        cls.eval_into = eval_into
        self._saved.append((cls, 'eval_into', real_eval))
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
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    auto = max(256, N // 16)
    from lumenairy.elements import _lens_imap as IM
    E = _fix.gauss(N, dx, 1.0e-3)
    print('# free RAM %.1f GB   N=%d sub=%d dx=%.2f um auto_rows=%d reps=%d'
          % (_fixp.free_gb(), N, sub, dx * 1e6, auto, reps), flush=True)
    out = {'version': la.__version__, 'file': la.__file__, 'N': N, 'sub': sub,
           'dx': dx, 'auto_rows': auto, 'reps': reps, 'cases': {}}
    cases = []
    for model in ('screen', 'ray_density'):
        cases.append((model, 'whole', 0, None))
        cases.append((model, 'auto_band', auto, None))
        cases.append((model, 'auto_band_incumbent', auto, False))
    for model, tag, rows, imap in cases:
        kw = _kw(model, sub, dx, imap)
        rec = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)      # warm
            ts = []
            for _ in range(reps):
                IM.inverse_map_cache_clear()
                t = time.perf_counter()
                f = la.apply_real_lens_traced(E, sag_chunk_rows=rows,
                                              _imap_out=rec, **kw)
                ts.append(time.perf_counter() - t)
            hsh = _fixp.h(np.asarray(f))
            del f
            # one INSTRUMENTED rep for the per-stage split
            IM.inverse_map_cache_clear()
            with _Stages(IM) as st:
                t = time.perf_counter()
                la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
                inst_total = time.perf_counter() - t
            IM.inverse_map_cache_clear()
        key = '%s.%s' % (model, tag)
        d = dict(st.d)
        d.update({'rows': rows, 'inverse_map_kw': imap, 'best': min(ts),
                  'all': ts, 'hash': hsh,
                  'engaged': bool(rec.get('engaged', False)),
                  'gate_open': bool(rec.get('gate_open', False)),
                  'inst_total_s': inst_total,
                  'inst_rest_s': inst_total - d['eval_s'] - d['mc_s'],
                  'eval_channels_per_pixel':
                      (d['eval_channels'] / (N * N)) if N else 0.0})
        out['cases'][key] = d
        print('%-32s rows=%-5s best %7.3f s  eng=%-5s  eval calls %4d '
              '(%.3f ch/px, %6.2f s)  mc %3d (%5.2f s)  rest %6.2f s  %s'
              % (key, rows, min(ts), d['engaged'], d['eval_calls'],
                 d['eval_channels_per_pixel'], d['eval_s'], d['mc_calls'],
                 d['mc_s'], d['inst_rest_s'], hsh), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
