"""VERIFY-WP-C3 CLAIM 8c follow-up -- WHERE a device array dies on the public
CHAIN entry point, guard by guard.

    python v_cupy_chain.py <tree> <out.json>

The report's decision is stated for ``propagate_carrier_referenced``.  The
default this package flipped is also the CHAIN's, and the chain is what the
design-121 gate drives.  This walks the chain with a CuPy field and, each time
it dies on a host demotion, silences the guard that caused it and runs again --
so the output is a MAP of every implicit-conversion site on the public chain
path, not just the first.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)

WL = 1.064e-6


def main():
    out_path = sys.argv[2]
    import cupy as cp
    n, dx = 64, 8e-6
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2)).astype(np.complex128)
    Ed = cp.asarray(E)

    attempts = [
        ('defaults', {}),
        ('on_multi_congruence=ignore', {'on_multi_congruence': 'ignore'}),
        ('+ on_undersample/noncollimated silent',
         {'on_multi_congruence': 'ignore',
          'traced_kwargs': {'on_undersample': 'silent',
                            'on_noncollimated': 'silent'}}),
    ]
    rows = []
    for label, kw in attempts:
        rec = {'label': label, 'kwargs': {k: str(v) for k, v in kw.items()}}
        try:
            r = CA.propagate_traced_carrier_chain(
                Ed, [{'gap_before': 5e-3}], WL, dx, r_in=-0.05,
                transport='collins', on_collins_sampling='ignore', **kw)
            rec['outcome'] = 'ran'
            rec['namespace'] = type(r.field).__module__.split('.')[0]
        except Exception as exc:                      # noqa: BLE001
            msg = '%s: %s' % (type(exc).__name__, exc)
            tb = traceback.extract_tb(exc.__traceback__)
            rec['outcome'] = 'raised'
            rec['error_type'] = type(exc).__name__
            rec['error'] = str(exc)[:300]
            rec['frames'] = ['%s:%d %s' % (os.path.basename(f.filename),
                                           f.lineno, f.name) for f in tb]
            rec['is_implicit_conversion_TypeError'] = (
                type(exc).__name__ == 'TypeError'
                and 'implicit conversion' in msg.lower())
            rec['names_cufft'] = 'cufft' in msg.lower()
        rows.append(rec)
        print('[chain] %-42s %s %s %s' % (
            label, rec['outcome'], rec.get('error_type', ''),
            'IMPLICIT-CONVERSION' if rec.get(
                'is_implicit_conversion_TypeError')
            else ('cufft' if rec.get('names_cufft') else '')))
        if rec.get('frames'):
            for f in rec['frames'][-4:]:
                print('        ', f)
    vlib.write_json({'build': vlib.build_tag(), 'tree': _TREE,
                     'carrier_file': CA.__file__, 'rows': rows}, out_path)


if __name__ == '__main__':
    main()
