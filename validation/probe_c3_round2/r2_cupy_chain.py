"""WP-C3 round 2 -- D10(a): where does a DEVICE array die on the public chain?

VERIFY-WP-C3 D10(a) measured the public chain dying with exactly the
implicit-conversion ``TypeError`` the package's own section 5.3 says must not
happen, at ``_chain_entry_congruence_stats`` -- a HOST DEMOTION in the chain,
before any transform, on the default ``on_multi_congruence='warn'`` and on
both transports.  The sibling probe ``v_cupy_chain.py`` drove an INVALID
group spec, so after the demotion is fixed it stops at the group validator
instead of reaching the transform; this one uses a valid chain.

    python r2_cupy_chain.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import traceback

_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)
OUT = sys.argv[2]

import numpy as np                                            # noqa: E402
import lumenairy                                              # noqa: E402
import lumenairy.propagators.carrier as CA                     # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(_TREE), (
    lumenairy.__file__, _TREE)
print('[anchor]', lumenairy.__file__)

WL = 1.064e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _singlet():
    return {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 60e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -60e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def _classify(exc):
    msg = f'{type(exc).__name__}: {exc}'
    tb = traceback.extract_tb(exc.__traceback__)
    return {
        'outcome': 'raised', 'error_type': type(exc).__name__,
        'error': str(exc)[:300],
        'frames': [f'{os.path.basename(f.filename)}:{f.lineno} {f.name}'
                   for f in tb][-5:],
        'is_implicit_conversion_TypeError': (
            type(exc).__name__ == 'TypeError'
            and 'implicit conversion' in msg.lower()),
        'names_cufft': 'cufft' in msg.lower(),
    }


def main():
    out = {'tree': _TREE, 'lumenairy': lumenairy.__file__, 'rows': []}
    try:
        import cupy as cp
        out['cupy'] = cp.__version__
    except Exception as exc:                                  # noqa: BLE001
        out['cupy'] = None
        out['cupy_import_error'] = str(exc)[:200]
        print('[premise] no CuPy on this build -- nothing to measure')
        with open(OUT, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, default=str)
        print('WROTE', OUT)
        return
    n, dx, w = 128, 30e-6, 1.0e-3
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    e_host = np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)
    e_dev = cp.asarray(e_host)
    groups = [{'prescription': _singlet(), 'gap_before': 20e-3}]
    cases = [
        ('single-step leg', lambda env: CA.propagate_carrier_referenced(
            env, -40e-3, 5e-3, WL, dx, transport='collins',
            on_collins_sampling='ignore')),
        ('chain, defaults', lambda env: CA.propagate_traced_carrier_chain(
            env, groups, WL, dx, r_in=np.inf, ray_subsample=16, n_workers=1,
            final_distance=5e-3, on_collins_sampling='ignore')),
        ('chain, guards silent',
         lambda env: CA.propagate_traced_carrier_chain(
             env, groups, WL, dx, r_in=np.inf, ray_subsample=16, n_workers=1,
             traced_kwargs=TKW, on_multi_congruence='ignore',
             final_distance=5e-3, on_collins_sampling='ignore')),
    ]
    for label, fn in cases:
        rec = {'label': label}
        try:
            r = fn(e_dev)
            rec['outcome'] = 'ran'
            rec['namespace'] = type(
                getattr(r, 'field', getattr(r, 'env', r))).__module__.split(
                    '.')[0]
        except BaseException as exc:                          # noqa: BLE001
            rec.update(_classify(exc))
        out['rows'].append(rec)
        print('[%-20s] %-7s %-12s implicit=%s cufft=%s' % (
            label, rec['outcome'], rec.get('error_type', ''),
            rec.get('is_implicit_conversion_TypeError'),
            rec.get('names_cufft')))
        for f in rec.get('frames', [])[-3:]:
            print('      ', f)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
