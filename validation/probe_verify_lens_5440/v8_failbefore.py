"""V8 -- FAIL-BEFORE on change 1, and the cross-build identity of the answer.

On v5.43.0 a BANDED traced call at the shipped default silently selected the
incumbent coarse-Newton inversion; on 50824e9 it evaluates the model.  This
runs the SAME fixtures on both builds and reports, per fixture:

  whole_hash   sag_chunk_rows=0   (the evaluator on both builds)
  band_hash    sag_chunk_rows=64  (coarse-Newton on 5.43.0, evaluator on 5.44)
  rel          ||band - whole|| / ||whole||          (the route change size)
  engaged      whether the evaluator engaged on the banded arm

The cross-build check is done by the companion comparison: 5.44.0's BANDED
hash must equal v5.43.0's WHOLE-GRID hash -- the evaluator's answer, not a
third one.

``--auto`` runs ONE N=4096 fixture at the TRUE shipped default
(``sag_chunk_rows=None`` -> AUTO banding at N >= 4096).

Usage:  python v8_failbefore.py <out.json> [--auto]
"""
from __future__ import annotations

import json
import os
import sys
import time

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


def _sph(n, dx, w, R):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = X ** 2 + Y ** 2
    k = 2 * np.pi / _fix.WL
    return (np.exp(-r2 / w ** 2) * np.exp(1j * k * r2 / (2.0 * R))
            ).astype(np.complex128)


def main():
    la = _fix.banner()
    auto = '--auto' in sys.argv
    out = {'version': la.__version__, 'auto': auto, 'cases': {}}
    if auto:
        NA, DXA, SUBA = 4096, 1.5e-6, 32
        cases = [('auto_screen_N4096', _fix.gauss(NA, DXA, 1.0e-3),
                  _base(dx=DXA, sub=SUBA), [0, None]),
                 ('auto_rd_N4096', _fix.gauss(NA, DXA, 1.0e-3),
                  _base(dx=DXA, sub=SUBA, amplitude_model='ray_density'),
                  [0, None])]
    else:
        cases = [
            ('screen', _fix.gauss(N, DX, W), _base(), [0, 64]),
            ('screen_carrier', _sph(N, DX, W, 0.055),
             _base(carrier=0.055), [0, 64]),
            ('rd', _fix.gauss(N, DX, W),
             _base(amplitude_model='ray_density'), [0, 64]),
            ('rd_remap_full', _fix.gauss(N, DX, W),
             _base(amplitude_model='ray_density',
                   preserve_input_phase='remap', remap_sampling='full'),
             [0, 64]),
        ]
    for name, E, kw, rows_list in cases:
        rec = {}
        fields = {}
        for rows in rows_list:
            t = time.time()
            f, r, msgs = _fix.run(la, E, kw, rows)
            tag = 'whole' if rows == 0 else 'band'
            fields[tag] = f
            rec[tag] = {'rows': rows, 'hash': _fix.h(f),
                        'engaged': bool(r.get('engaged', False)),
                        'gate_open': bool(r.get('gate_open', False)),
                        'n_out_of_domain': r.get('n_out_of_domain'),
                        'nwarn': len(msgs), 'secs': round(time.time() - t, 2)}
        a, b = fields['whole'], fields['band']
        rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        rec['rel_band_vs_whole'] = rel
        rec['bit_equal'] = bool(np.array_equal(a, b))
        out['cases'][name] = rec
        print('%-22s whole=%s(eng=%s)  band=%s(eng=%s)  rel=%.4e  bit_eq=%s'
              % (name, rec['whole']['hash'], rec['whole']['engaged'],
                 rec['band']['hash'], rec['band']['engaged'], rel,
                 rec['bit_equal']), flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
