"""V18 -- the niche-C15 private diagnostic probe on a BANDED call.

``_imap_out['probe_rc'] = (rows, cols)`` asks for the FINALISED ``opl_map``
sampled at those pixels.  The probe block sits AFTER the row-band assembly's
``return``, so a banded call never fills it.  That was inert before v5.44 (a
banded call was always the coarse-Newton incumbent and the probe compares
inversions); it is worth recording now that a banded call at the shipped
default IS the evaluator.

Usage:  python v18_c15_probe_on_band.py <out.json>
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
    E = _fix.gauss(N, DX, W)
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    rc = (np.array([100, 150, 200]), np.array([100, 150, 200]))
    out = {'version': la.__version__, 'cases': {}}
    for rows in (0, 32):
        IM.inverse_map_cache_clear()
        rec = {'probe_rc': rc}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, sag_chunk_rows=rows, _imap_out=rec,
                                      **kw)
        IM.inverse_map_cache_clear()
        out['cases'][str(rows)] = {
            'engaged': bool(rec.get('engaged', False)),
            'has_probe_opl': 'probe_opl' in rec,
            'probe_opl': (None if 'probe_opl' not in rec
                          else [float(v) for v in rec['probe_opl']]),
            'n_out_of_domain': rec.get('n_out_of_domain')}
        print('rows=%-4s engaged=%s  probe_opl present=%s  n_ood=%s'
              % (rows, out['cases'][str(rows)]['engaged'],
                 out['cases'][str(rows)]['has_probe_opl'],
                 out['cases'][str(rows)]['n_out_of_domain']), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
