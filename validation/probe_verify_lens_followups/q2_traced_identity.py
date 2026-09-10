"""Q2 -- BIT IDENTITY of every traced call, whole-grid AND banded.

Task 1, second half.  Band heights {None (AUTO), 0 (whole-grid), 7, 32, 128}
on 11 of my own fixtures covering both amplitude models, both inversion routes
(evaluator and the coarse-Newton incumbent), preserve / remap x {lattice,
full}, a decentred origin, a caustic-bearing fold, sub=1 and a tilt-aware
call.  Field hash + sum|E|^2 + max|E| + the diagnostic record + the full
warning list (messages AND attribution) are recorded for each.

The D1 change MOVES the attribution of three self-check warnings, so warnings
are recorded twice: ``w_msgs`` (message text only -- MUST be identical) and
``w_attr`` (filename:lineno -- the D1 movement).  Nothing else may move.

Usage: python q2_traced_identity.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX = 288, 11e-6
W = 0.85e-3
SUB = 4
ROWS = [None, 0, 7, 32, 128]


def _base(**over):
    kw = dict(prescription=_vf.presc_meniscus(ap=4.4e-3), wavelength=_vf.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def fixtures():
    E = lambda: _vf.gauss(N, DX, W)                              # noqa: E731
    Esp = lambda: _vf.speckled(N, DX, W, seed=5)                 # noqa: E731
    Edec = lambda: _vf.gauss(N, DX, W, x0=0.30e-3, y0=-0.22e-3)  # noqa: E731
    Ecv = lambda: _vf.sph(N, DX, W, 0.055)                       # noqa: E731
    return [
        ('g01_screen_preserve', E, _base()),
        ('g02_screen_carrier', Ecv, _base(carrier=0.055)),
        ('g03_screen_speckled', Esp, _base()),
        ('g04_rd_carrier', Ecv, _base(carrier=0.055,
                                      amplitude_model='ray_density')),
        ('g05_rd_remap_lattice', Ecv,
         _base(carrier=0.055, amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='lattice')),
        ('g06_rd_remap_full', Ecv,
         _base(carrier=0.055, amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='full')),
        ('g07_rd_origin', Edec,
         _base(amplitude_model='ray_density', preserve_input_phase='remap',
               remap_sampling='full', origin=(0.30e-3, -0.22e-3))),
        ('g08_screen_noinv', Ecv, _base(carrier=0.055, inverse_map=False)),
        ('g09_rd_noinv', Ecv, _base(carrier=0.055,
                                    amplitude_model='ray_density',
                                    inverse_map=False)),
        ('g10_rd_caustic', lambda: _vf.gauss(N, DX, 1.4e-3),
         _base(prescription=_vf.presc_fast(ap=4.4e-3),
               amplitude_model='ray_density')),
        ('g11_rd_tilt_nosub', Edec,
         _base(amplitude_model='ray_density', ray_subsample=1,
               tilt_aware_rays=True, preserve_input_phase='remap',
               remap_sampling='lattice',
               beam_centre=(0.30e-3, -0.22e-3))),
    ]


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    out = {}
    t00 = time.perf_counter()
    for name, mk, kw in fixtures():
        E = mk()
        for rows in ROWS:
            t0 = time.perf_counter()
            F, rec, wl = _vf.run_traced(la, E, kw, rows)
            key = f'{name}|rows={rows}'
            out[key] = {
                'field': _vf.field_record(F),
                'rec': _vf.rec_record(rec),
                'w_msgs': sorted(w[0] for w in wl),
                'w_attr': sorted(f'{w[1]}:{w[2]}' for w in wl),
                'secs': round(time.perf_counter() - t0, 3),
            }
            print(f"  {key:44s} {out[key]['field']['hash']} "
                  f"nw={len(wl)} ({out[key]['secs']} s)", flush=True)
    print(f"# total {time.perf_counter() - t00:.1f} s", flush=True)
    _vf.dump(args, {'cases': out, 'N': N, 'DX': DX, 'SUB': SUB,
                    'rows': [str(r) for r in ROWS],
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
