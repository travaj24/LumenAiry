"""V11 -- the three side effects the byte-identity pins do not cover.

(a) WARNING ATTRIBUTION.  Moving the three ray-density self-checks into the
    shared ``_ray_density_self_checks`` closure put them ONE FRAME deeper
    while their ``stacklevel=2`` stayed as it was.  A warning's reported
    file/line is what ``warnings.filterwarnings(module=...)`` and every
    user-side log filter key on, so this records ``w.filename`` /
    ``w.lineno`` for every warning on both builds.

(b) THE niche-D9 ORIGIN FRACTION.  The whole-grid path sums two full-grid
    reductions; the band path accumulates them band by band.  Different
    summation ORDER, so the fraction is not bit-identical by construction --
    this reads the fraction the two paths print (6 decimals of a percent)
    and compares.

(c) WALL TIME of the banded evaluator route against the whole-grid route
    (the CHANGELOG says "~10 % higher on the evaluator route"), measured
    WITHOUT tracemalloc.

Usage:  python v11_side_effects.py <out.json> [N] [sub] [dx_um]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402

N, DX, W, SUB = 384, 13e-6, 1.0e-3, 4


def _base(dx=DX, sub=SUB, **over):
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=dx, ray_subsample=sub, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def _catch(la, E, kw, rows):
    from lumenairy.elements import _lens_imap as IM
    IM.inverse_map_cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f = la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
    IM.inverse_map_cache_clear()
    return np.asarray(f), [{'msg': str(w.message)[:110],
                            'file': os.path.basename(str(w.filename)),
                            'line': int(w.lineno)} for w in caught]


def main():
    la = _fix.banner()
    from lumenairy.elements import _lens_traced as LT
    Nn = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    subn = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    dxn = (float(sys.argv[4]) if len(sys.argv) > 4 else 3.0) * 1e-6
    out = {'version': la.__version__}

    # ---- (a) + (b) --------------------------------------------------------
    LT._RD_ENERGY_GAIN_TOL = -1.0
    LT._RD_ENERGY_DEFICIT_BASE = -1.0
    LT._RD_ENERGY_DEFICIT_PER_SUB = 0.0
    LT._RD_HALO_AMAX_TOL = 0.0
    LT._SUPPORT_BAND_PEAK_RATIO_TOL = 0.0
    LT._ORIGIN_AMP_SUPPORT_TOL = -1.0
    LT.ORIGIN_AMP_SUPPORT_CHECK = 'warn'
    E = _fix.gauss(N, DX, W, x0=0.30e-3, y0=-0.22e-3)
    kw = _base(amplitude_model='ray_density', preserve_input_phase='remap',
               remap_sampling='full', origin=(0.30e-3, -0.22e-3))
    attr = {}
    for rows in (0, 32, 7):
        f, ws = _catch(la, E, kw, rows)
        attr[str(rows)] = {'hash': _fix.h(f), 'warnings': ws}
        print('rows=%s  hash=%s' % (rows, _fix.h(f)), flush=True)
        for w in ws:
            print('    %s:%d  %s' % (w['file'], w['line'], w['msg'][:95]),
                  flush=True)
    out['warning_attribution'] = attr
    # the D9 fraction, as each path prints it
    def _frac(ws):
        for w in ws:
            if 'decentres the WAVE GRID' in w['msg'] or '% of the ray' in w['msg']:
                return w['msg']
        return None
    out['d9_fraction_text'] = {r: _frac(attr[r]['warnings']) for r in attr}

    # ---- (c) wall time ----------------------------------------------------
    LT._RD_ENERGY_GAIN_TOL = 0.050
    Et = _fix.gauss(Nn, dxn, 1.0e-3)
    kwt = _base(dx=dxn, sub=subn, amplitude_model='ray_density')
    timing = {}
    for tag, rows in (('whole', 0), ('band32', 32),
                      ('band_auto', max(256, Nn // 16))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from lumenairy.elements import _lens_imap as IM
            IM.inverse_map_cache_clear()
            la.apply_real_lens_traced(Et, sag_chunk_rows=rows, **kwt)  # warm
            ts = []
            for _ in range(3):
                IM.inverse_map_cache_clear()
                t = time.perf_counter()
                la.apply_real_lens_traced(Et, sag_chunk_rows=rows, **kwt)
                ts.append(time.perf_counter() - t)
            IM.inverse_map_cache_clear()
        timing[tag] = {'rows': rows, 'best': min(ts), 'all': ts}
        print('time %-10s rows=%-5s best %.3f s  (%s)'
              % (tag, rows, min(ts), ', '.join('%.3f' % v for v in ts)),
              flush=True)
    timing['band32_over_whole'] = (timing['band32']['best']
                                   / timing['whole']['best'])
    timing['auto_over_whole'] = (timing['band_auto']['best']
                                 / timing['whole']['best'])
    print('band/whole = %.3f (rows=32), %.3f (AUTO rows=%d)'
          % (timing['band32_over_whole'], timing['auto_over_whole'],
             timing['band_auto']['rows']), flush=True)
    out['timing'] = {'N': Nn, 'sub': subn, 'dx': dxn, **timing}

    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
