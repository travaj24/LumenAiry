"""Shared helpers for the ``e_`` engine-census probes (MEASUREMENT ONLY).

Nothing here imports from, or writes to, the library.  Every probe pins the
imported tree before it measures anything.
"""
from __future__ import annotations

import inspect
import json
import os
import platform
import sys

import numpy as np


def pin_tree():
    """HARD tree pin: refuse to measure any checkout but ``lum_borscope``."""
    import lumenairy
    p = os.path.abspath(lumenairy.__file__)
    low = p.replace("\\", "/").lower()
    if "lum_borscope" not in low:
        raise SystemExit("REFUSED: lumenairy imported from %s" % p)
    root = os.path.abspath(os.path.dirname(os.path.dirname(lumenairy.__file__)))
    cwd = os.path.abspath(os.getcwd())
    if os.path.normcase(root) != os.path.normcase(cwd):
        raise SystemExit("REFUSED: lumenairy root %s != cwd %s" % (root, cwd))
    print("lumenairy:", p, lumenairy.__version__, flush=True)
    return p


def stamp():
    import lumenairy
    return dict(
        lumenairy_file=os.path.abspath(lumenairy.__file__),
        version=lumenairy.__version__,
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.system(),
        node=platform.node(),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
    )


def src(fn):
    try:
        return inspect.getsource(fn)
    except Exception:
        return ""


def _default(o):
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def dump(path, payload):
    payload = dict(payload)
    payload["_stamp"] = stamp()
    with open(path, "w", encoding="cp1252", errors="replace") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_default)
    print("wrote %s" % path, flush=True)


def rt_per_pol(res):
    """(R+T) per incident polarization for a 2-row (2, N) return, else the
    scalar sum."""
    R, T = np.asarray(res[1]), np.asarray(res[2])
    if R.ndim == 2:
        return (R.sum(axis=1) + T.sum(axis=1)).astype(float)
    return np.array([float(R.sum() + T.sum())])
