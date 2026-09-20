"""VERIFY-WP-C5 -- shared digest / IO helpers, written independently of
``validation/probe_c5_three_defaults/_digest.py`` so a defect in that module
cannot hide in both.

A digest is the SHA-256 of the array's exact bytes plus its shape and dtype,
so "identical" means identical to the last bit and a dtype change is visible.
"""
import hashlib
import json
import os
import platform
import sys

import numpy as np


def dig(a):
    """SHA-256 over the C-contiguous raw bytes, with shape and dtype."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def stamp():
    import lumenairy
    return dict(
        lumenairy_file=lumenairy.__file__,
        lumenairy_version=getattr(lumenairy, '__version__', '?'),
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
        omp=os.environ.get('OMP_NUM_THREADS'),
        openblas=os.environ.get('OPENBLAS_NUM_THREADS'),
        mkl=os.environ.get('MKL_NUM_THREADS'),
        mem_budget_mb_env=os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
    )


def write(path, payload):
    payload = dict(payload)
    payload['_stamp'] = stamp()
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
    print('WROTE', path)
    print('lumenairy.__file__ =', payload['_stamp']['lumenairy_file'])


def compare(a_path, b_path, out_path, key='digests'):
    """Two digest maps -> identical / moved / only-in-one."""
    with open(a_path, encoding='cp1252') as fh:
        A = json.load(fh)
    with open(b_path, encoding='cp1252') as fh:
        B = json.load(fh)
    da, db = A[key], B[key]
    same = sorted(k for k in da if k in db and da[k] == db[k])
    moved = sorted(k for k in da if k in db and da[k] != db[k])
    only_a = sorted(k for k in da if k not in db)
    only_b = sorted(k for k in db if k not in da)
    res = dict(a=a_path, b=b_path, n_identical=len(same), n_moved=len(moved),
               identical=same, moved=moved, only_in_a=only_a,
               only_in_b=only_b,
               a_stamp=A.get('_stamp', {}).get('lumenairy_file'),
               b_stamp=B.get('_stamp', {}).get('lumenairy_file'))
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print('WROTE', out_path)
    print(f'identical {len(same)}  moved {len(moved)}  '
          f'only_a {len(only_a)}  only_b {len(only_b)}')
    for k in moved:
        print('  MOVED', k)
    return res


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2], sys.argv[3],
            key=(sys.argv[4] if len(sys.argv) > 4 else 'digests'))
