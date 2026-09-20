"""Shared digesting for the WP-C5 probes -- arrays by RAW BYTES.

Every probe in this directory writes one JSON of ``key -> sha256``.  The two
trees (a ``git archive`` of the parent commit and a ``git archive`` of the
branch tip) are digested by SEPARATE PROCESSES with different ``PYTHONPATH``
pins, so no run ever sees two trees at once and a mis-pinned run is visible in
the output (``lumenairy_file`` is written into every JSON).
"""
import hashlib
import json
import numbers
import os
import platform
import sys

import numpy as np


def h(*parts):
    m = hashlib.sha256()
    for p in parts:
        m.update(p if isinstance(p, bytes) else str(p).encode('utf-8'))
        m.update(b'\x00')
    return m.hexdigest()


def dig_arr(a):
    a = np.asarray(a)
    return h('arr', a.dtype.str, a.shape, np.ascontiguousarray(a).tobytes())


def dig(o):
    if isinstance(o, BaseException):
        return h('exc', type(o).__name__, str(o))
    if hasattr(o, 'env') and hasattr(o, 'R') and hasattr(o, 'dx'):
        return h('crf', dig_arr(o.env), repr(o.R), repr(o.dx))
    if hasattr(o, 'field') and hasattr(o, 'stages'):
        return h('chain', dig_arr(o.field))
    if isinstance(o, dict):
        return h('dict', *[f"{k}={dig(v)}" for k, v in sorted(o.items())])
    if isinstance(o, (tuple, list)):
        return h('seq', *[dig(v) for v in o])
    if isinstance(o, (numbers.Number, str, bool, type(None))):
        return h('sca', repr(o))
    return dig_arr(o)


def env_block():
    import lumenairy
    import numpy
    return {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'platform': platform.platform(),
        'threads': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS', 'LUMENAIRY_MEM_BUDGET_MB')},
    }


def write(path, payload):
    payload = dict(payload)
    payload['_env'] = env_block()
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print('wrote', path, 'keys', len(payload.get('digests', {})))
