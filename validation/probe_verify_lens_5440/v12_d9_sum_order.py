"""V12 -- the ONE quantity the band path does NOT compute bit-identically.

The niche-D9 origin support measurement is two full-grid reductions on the
whole-grid path and an accumulation of per-band partial sums on the band
path.  numpy's ``sum`` is pairwise, so the two orders are not bit-equal in
general.  The CHANGELOG names it ("a decision against a tolerance, not a
field value"); this measures how big the difference actually is.

(1) The percentage the two paths PRINT, read out of the warning text.
(2) A synthetic bound: full-grid ``sum`` vs the sum of per-band sums on
    arrays of the same shape and magnitude distribution, at every band
    height the shipped tests use.

Usage:  python v12_d9_sum_order.py <out.json>
"""
from __future__ import annotations

import json
import os
import re
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
    LT._ORIGIN_AMP_SUPPORT_TOL = -1.0
    LT.ORIGIN_AMP_SUPPORT_CHECK = 'warn'
    out = {'version': la.__version__, 'printed': {}}
    E = _fix.gauss(N, DX, W, x0=0.30e-3, y0=-0.22e-3)
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3),
              wavelength=_fix.WL, dx=DX, ray_subsample=SUB, n_workers=1,
              on_undersample='silent', on_noncollimated='off',
              on_aperture_beam='silent', parallel_amp=False,
              amplitude_model='ray_density', preserve_input_phase='remap',
              remap_sampling='full', origin=(0.30e-3, -0.22e-3))
    pat = re.compile(r'deleting\s+([0-9.]+)\s+% of the ray-density exit power')
    for rows in (0, 1, 7, 32, 128, N):
        IM.inverse_map_cache_clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f = np.asarray(la.apply_real_lens_traced(
                E, sag_chunk_rows=rows, **kw))
        IM.inverse_map_cache_clear()
        pct = None
        for w in caught:
            m = pat.search(str(w.message))
            if m:
                pct = m.group(1)
        out['printed'][str(rows)] = {'pct': pct, 'hash': _fix.h(f)}
        print('rows=%-5s D9 pct = %s   field %s' % (rows, pct, _fix.h(f)),
              flush=True)
    vals = {v['pct'] for v in out['printed'].values()}
    out['printed_all_equal'] = (len(vals) == 1)
    print('printed percentages all equal:', out['printed_all_equal'],
          flush=True)

    # ---- (2) the synthetic summation-order bound -------------------------
    rng = np.random.default_rng(3)
    x = (np.arange(N) - N / 2) * DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    a = (np.exp(-r2 / (0.25 * N * DX) ** 2)
         * (1 + 0.3 * rng.random((N, N)))) ** 2          # ~ |a_rd|^2
    full = float(a.sum())
    bounds = {}
    for cr in (1, 7, 32, 128, N):
        acc = 0.0
        for r0 in range(0, N, cr):
            acc += float(a[r0:min(N, r0 + cr)].sum())
        bounds[str(cr)] = {'banded': acc, 'full': full,
                           'abs_diff': abs(acc - full),
                           'rel_diff': abs(acc - full) / abs(full)}
        print('band %-5s sum rel diff vs whole-grid sum: %.3e'
              % (cr, bounds[str(cr)]['rel_diff']), flush=True)
    out['synthetic_sum_order'] = bounds
    out['worst_rel_sum_diff'] = max(v['rel_diff'] for v in bounds.values())
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
