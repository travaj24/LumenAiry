"""V7 -- does requesting complex64 actually SAVE memory now?

The audit's finding (AUDIT_TRACED_MEMORY_2026_08_09 sec 3 / row 12) was that
requesting complex64 saved 0.0 GB.  The 5.44.0 claim is that the six helpers
now follow the field's dtype, so it does.  This measures the whole-call
``tracemalloc`` peak of the SAME two-group chain at complex128 and at
complex64, on both builds, and records WHICH call site still returns a
complex128 phasor on the complex64 arm (by caller frame).

Usage:  python v7_c64_memory.py <out.json> [N] [dx_um]
"""
from __future__ import annotations

import json
import os
import sys
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

WL = _fix.WL


def _singlet(R1, R2, d, glass, ap, name):
    def s(r, gb, ga):
        return {'radius': r, 'glass_before': gb, 'glass_after': ga,
                'conic': 0.0, 'radius_y': None, 'conic_y': None,
                'aspheric_coeffs': None, 'aspheric_coeffs_y': None}
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [s(R1, 'air', glass), s(R2, glass, 'air')]}


def chain(la, env, dx):
    p1 = _singlet(55.0e-3, -55.0e-3, 3.0e-3, 'N-BK7', 13.0e-3, 'g1')
    p2 = _singlet(40.0e-3, -1e12, 2.5e-3, 'N-SF11', 13.0e-3, 'g2')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return la.propagate_traced_carrier_chain(
            env, [{'prescription': p1, 'gap_before': 0.0},
                  {'prescription': p2, 'gap_before': 12.0e-3}],
            WL, dx, r_in=55e-3, ray_subsample=8, n_workers=1,
            traced_kwargs=dict(on_undersample='silent',
                               on_noncollimated='silent'))


def trace_callers(C, log):
    """Log every ``_radial_carrier_phase`` / ``_sphere_parab_conversion`` /
    ``_tilt_ramp`` / ``_tilt_exactness_phase`` return whose dtype is
    complex128, WITH the caller's function name and line."""
    saved = {}
    for nm in ('_radial_carrier_phase', '_tilt_ramp', '_tilt_exactness_phase',
               '_sphere_parab_conversion'):
        fn = getattr(C, nm)
        saved[nm] = fn

        def mk(nm, fn):
            def wrapper(*a, **kw):
                out = fn(*a, **kw)
                fr = sys._getframe(1)
                log.append({
                    'fn': nm,
                    'caller': fr.f_code.co_name,
                    'lineno': fr.f_lineno,
                    'dtype_kw': (None if kw.get('dtype') is None
                                 else str(kw['dtype'])),
                    'out_dtype': (None if out is None else str(out.dtype)),
                    'nbytes': (0 if out is None else int(out.nbytes))})
                return out
            return wrapper
        setattr(C, nm, mk(nm, fn))
    return saved


def main():
    la = _fix.banner()
    from lumenairy.propagators import carrier as C
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    dx = (float(sys.argv[3]) if len(sys.argv) > 3 else 13.0) * 1e-6
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    w = 0.28 * N * dx
    env = np.exp(-r2 / w ** 2).astype(np.complex128)
    grid128 = 16.0 * N * N
    out = {'version': la.__version__, 'N': N, 'dx': dx,
           'grid_c128_bytes': grid128}
    try:
        import psutil
        print('# free RAM %.1f GB'
              % (psutil.virtual_memory().available / 2 ** 30), flush=True)
    except ImportError:
        pass
    for tag, e in (('c128', env), ('c64', env.astype(np.complex64))):
        chain(la, e, dx)                                   # warm
        tracemalloc.start()
        tracemalloc.reset_peak()
        r = chain(la, e, dx)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        log = []
        saved = trace_callers(C, log)
        try:
            r2_ = chain(la, e, dx)
        finally:
            for nm, fn in saved.items():
                setattr(C, nm, fn)
        f = np.asarray(r.field)
        wide = [d for d in log if d['out_dtype'] == 'complex128']
        out[tag] = {'peak_bytes': pk, 'peak_grids_c128': pk / grid128,
                    'field_dtype': str(f.dtype),
                    'result_dtype2': str(np.asarray(r2_.field).dtype),
                    'helper_calls': len(log),
                    'complex128_returns': wide,
                    'complex128_bytes_returned':
                        int(sum(d['nbytes'] for d in wide))}
        print('%s: peak %8.1f MiB = %5.2f c128-grids   field %s   '
              'c128 phasor returns %d (%.1f MiB)'
              % (tag, pk / 2 ** 20, pk / grid128, f.dtype, len(wide),
                 out[tag]['complex128_bytes_returned'] / 2 ** 20), flush=True)
        for d in wide:
            print('    ', d, flush=True)
    if 'c128' in out and 'c64' in out:
        out['saving_bytes'] = out['c128']['peak_bytes'] - out['c64']['peak_bytes']
        out['saving_frac'] = out['saving_bytes'] / out['c128']['peak_bytes']
        print('complex64 saves %.1f MiB = %.1f %% of the complex128 peak'
              % (out['saving_bytes'] / 2 ** 20, 100 * out['saving_frac']),
              flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
