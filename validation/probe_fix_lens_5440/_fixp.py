"""Shared helpers for the 5.44.0 FOLLOW-UP fix probes (D1-D7).

Companion to ``validation/probe_verify_lens_5440/`` (the verification that
recorded the seven follow-ups).  Everything here asserts which tree it is
importing before it measures anything.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

WL = 1.064e-6


def banner(expect='lum_lensfix'):
    import lumenairy as la
    f = os.path.abspath(la.__file__)
    print(f"# lumenairy.__file__ = {f}", flush=True)
    print(f"# lumenairy.__version__ = {la.__version__}", flush=True)
    print(f"# numpy {np.__version__}  python {sys.version.split()[0]}",
          flush=True)
    print(f"# OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    if expect is not None and expect not in f.replace(os.sep, '/'):
        raise SystemExit(f"WRONG TREE: expected {expect!r} in {f}")
    return la


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except ImportError:
        return float('nan')
