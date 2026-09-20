"""VERIFY-WP-C3 -- my own harness.  Nothing is imported from the package's
own probe directory: the anchor, the build tag and the JSON writer are
re-implemented here so a defect in their harness cannot hide in my readings."""
from __future__ import annotations

import json
import os
import platform
import sys


def anchor(tree):
    """Refuse to measure a tree other than the one named."""
    import lumenairy
    got = os.path.abspath(lumenairy.__file__)
    want = os.path.abspath(tree)
    print(f'[anchor] lumenairy.__file__ = {got}')
    if not got.lower().startswith(want.lower() + os.sep):
        raise SystemExit(f'ANCHOR FAILED: {got} is not under {want}')
    return got


def build_tag():
    import numpy as np
    if sys.platform.startswith('win'):
        tag = 'WIN'
    else:
        tag = 'WSL' if 'microsoft' in platform.uname().release.lower() else 'LIN'
    return f'{tag}-py{sys.version_info[0]}.{sys.version_info[1]}-np{np.__version__}'


def env_tag():
    keys = ('OPENBLAS_CORETYPE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')
    return {k: os.environ.get(k) for k in keys}


def write_json(rec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    print(f'[wrote] {path}')
