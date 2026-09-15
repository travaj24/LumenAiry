"""P8 -- is the serial fallback really bit-identical to the pooled answer?

The broken-pool fallback's whole claim is that it costs wall time and moves no
number, so the claim is checked on a SPREAD of fields rather than on one:
different apertures, radii, subsamples, worker counts and both fit backends,
each run twice -- once with ``n_workers=1`` (the serial Newton path) and once
with the pool engaged -- and compared with ``np.array_equal`` on the raw
bytes.

Run with ``--rows`` to widen the sweep.  Output is one JSON object per row
plus a summary; ``--json-out`` writes the whole table.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

T0 = 0.0


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def singlet(ap, r):
    return {'name': 'singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def field(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def run(la, N, ap, r, rs, fit, nw, wl):
    dx = 2.2 * ap / N
    E0 = field(N, dx, 0.4 * ap)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=singlet(ap, r), wavelength=wl, dx=dx,
            ray_subsample=rs, newton_fit=fit, n_workers=nw,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


ROWS = [
    # N,   ap,     r,      ray_subsample, newton_fit,    n_workers
    (256, 3.0e-3, 9.0e-3, 2, 'spline', 2),
    (256, 3.0e-3, 9.0e-3, 2, 'spline', 4),
    (256, 3.0e-3, 9.0e-3, 2, 'polynomial', 4),
    (256, 3.0e-3, 12.0e-3, 2, 'spline', 3),
    (256, 4.0e-3, 9.0e-3, 2, 'polynomial', 2),
    (320, 3.0e-3, 9.0e-3, 2, 'spline', 4),
    (320, 3.0e-3, 15.0e-3, 2, 'polynomial', 4),
    (384, 3.0e-3, 9.0e-3, 3, 'spline', 5),
    (384, 5.0e-3, 12.0e-3, 3, 'polynomial', 3),
    (512, 3.0e-3, 9.0e-3, 4, 'spline', 4),
    (512, 3.0e-3, 9.0e-3, 4, 'polynomial', 6),
    (512, 6.0e-3, 18.0e-3, 4, 'spline', 2),
]


def main():
    global T0
    T0 = time.monotonic()
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--rows', type=int, default=len(ROWS))
    ap_.add_argument('--wavelength', type=float, default=1.31e-6)
    ap_.add_argument('--dump-after', type=float, default=1800.0)
    ap_.add_argument('--json-out', default=None)
    args = ap_.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, exit=True)

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, version=la.__version__,
         python=sys.version.split()[0], pid=os.getpid())
    # The size bars keep a one-shot call off a pool it cannot amortise; they
    # are not what this probe is about, so the dispatch is forced at every
    # row instead of only at the big ones.
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    table = []
    for N, apr, r, rs, fit, nw in ROWS[:args.rows]:
        LT.close_worker_pool()
        t = time.monotonic()
        ser = run(la, N, apr, r, rs, fit, 1, args.wavelength)
        t_ser = time.monotonic() - t
        t = time.monotonic()
        par = run(la, N, apr, r, rs, fit, nw, args.wavelength)
        t_par = time.monotonic() - t
        row = {'N': N, 'aperture_mm': round(apr * 1e3, 3),
               'radius_mm': round(r * 1e3, 3), 'ray_subsample': rs,
               'newton_fit': fit, 'n_workers': nw,
               'newton_points': (N // rs) ** 2,
               'serial_seconds': round(t_ser, 3),
               'pooled_seconds': round(t_par, 3),
               'serial_sha': sha(ser), 'pooled_sha': sha(par),
               'identical': bool(np.array_equal(ser, par)),
               'max_abs_delta': float(np.abs(ser - par).max())}
        table.append(row)
        emit('row', **row)

    n_ok = sum(1 for r in table if r['identical'])
    summary = {'rows': len(table), 'identical': n_ok,
               'all_identical': n_ok == len(table),
               'worst_delta': max((r['max_abs_delta'] for r in table),
                                  default=0.0),
               'python': sys.version.split()[0],
               'lumenairy': la.__file__}
    emit('summary', **summary)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump({'table': table, 'summary': summary}, fh, indent=1)
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    return 0 if summary['all_identical'] else 5


if __name__ == '__main__':
    sys.exit(main())
