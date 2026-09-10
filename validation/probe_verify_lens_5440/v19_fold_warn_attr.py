"""V19 -- attribution of the FOLD-CAUSTIC warning after it moved into the
shared ``_warn_ray_density_fold`` closure (stacklevel 2 -> 3).

Companion to V11, which covers the three self-check warnings.  Run on both
builds with the census thresholds forced so the warning fires.

Usage:  python v19_fold_warn_attr.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix  # noqa: E402

N, DX, W, SUB = 384, 13e-6, 1.0e-3, 4


def main():
    la = _fix.banner()
    from lumenairy.elements import _lens_imap as IM
    from lumenairy.elements import _lens_traced as LT
    LT._RAY_DENSITY_CAUSTIC_FLOOR_REL = 0.999
    LT._RAY_DENSITY_CAUSTIC_MAXMIN = 1.0000001
    E = _fix.gauss(N, DX, W)
    out = {'version': la.__version__, 'cases': {}}
    for inv in (None, False):
        for rows in (0, 32):
            kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
                      wavelength=_fix.WL, dx=DX, ray_subsample=SUB,
                      n_workers=1, on_undersample='silent',
                      on_noncollimated='off', on_aperture_beam='silent',
                      parallel_amp=False, amplitude_model='ray_density')
            if inv is not None:
                kw['inverse_map'] = inv
            IM.inverse_map_cache_clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                f = np.asarray(la.apply_real_lens_traced(
                    E, sag_chunk_rows=rows, **kw))
            IM.inverse_map_cache_clear()
            ws = [{'file': os.path.basename(str(w.filename)),
                   'line': int(w.lineno), 'msg': str(w.message)[:60]}
                  for w in caught if 'fold caustic' in str(w.message)]
            key = 'inv=%s.rows=%s' % (inv, rows)
            out['cases'][key] = {'hash': _fix.h(f), 'fold_warnings': ws}
            print('%-18s %s  fold warnings: %s'
                  % (key, _fix.h(f),
                     [(w['file'], w['line']) for w in ws]), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
