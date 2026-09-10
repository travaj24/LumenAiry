"""P2 (D1) -- warning ATTRIBUTION of the three ray-density self-checks.

VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D1: the three ``warn(...)`` calls
inside ``_ray_density_self_checks`` kept ``stacklevel=2`` when the block moved
into a nested closure, so on BOTH the banded and the whole-grid path they are
now attributed to ``_lens_traced.py`` instead of to the caller (v5.43.0
attributed all three to the caller).  This forces all five ray-density
notices over their thresholds and records ``w.filename`` / ``w.lineno``.

Uses the VERIFICATION's own fixture module so the arms are comparable.

Usage:  python p2_d1_attr.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE),
                                'probe_verify_lens_5440'))
import _fix  # noqa: E402
import _fixp  # noqa: E402

N, DX, W, SUB = 384, 13e-6, 1.0e-3, 4
THIS = os.path.basename(__file__)


def _base(**over):
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def _call(la, E, kw, rows):
    from lumenairy.elements import _lens_imap as IM
    IM.inverse_map_cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f = la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
    IM.inverse_map_cache_clear()
    return np.asarray(f), [{'msg': str(w.message)[:100],
                            'file': os.path.basename(str(w.filename)),
                            'line': int(w.lineno)} for w in caught]


def main():
    la = _fixp.banner()
    from lumenairy.elements import _lens_traced as LT
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
    out = {'version': la.__version__, 'file': la.__file__, 'this': THIS,
           'arms': {}}
    bad = 0
    for rows in (0, 32, 7):
        f, ws = _call(la, E, kw, rows)
        out['arms'][str(rows)] = {'hash': _fixp.h(f), 'warnings': ws,
                                  'n_from_caller':
                                      sum(w['file'] == THIS for w in ws),
                                  'n_total': len(ws)}
        print('rows=%-3s hash=%s  %d/%d warnings attributed to the CALLER'
              % (rows, _fixp.h(f), out['arms'][str(rows)]['n_from_caller'],
                 len(ws)), flush=True)
        for w in ws:
            flag = ' ' if w['file'] == THIS else '  <-- LIBRARY'
            print('    %-22s:%-6d %s%s'
                  % (w['file'], w['line'], w['msg'][:70], flag), flush=True)
            bad += (w['file'] != THIS)
    out['n_attributed_to_library'] = bad
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('TOTAL attributed to the library:', bad)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
