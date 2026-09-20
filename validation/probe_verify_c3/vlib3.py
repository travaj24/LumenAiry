"""VERIFY-WP-C3 harness -- written for this verification, sharing no code with
``validation/probe_c3_collins_default/clib.py`` or ``probe_wave5_hyg2/hlib.py``.

Three things live here and nothing else:

* :func:`bind` -- import ``lumenairy`` and REFUSE to continue unless its
  ``__file__`` resolves under the tree named on the command line; print what
  bound, on stdout AND stderr, so a mis-bound child is loud.
* :func:`fold` / :func:`sha` -- the record digest.  Arrays go in as RAW BYTES
  of a C-contiguous copy plus dtype and shape (so NaN, -0.0 and a dtype change
  are all moved bytes); floats go in as their ``float64`` bytes, never as
  ``repr``; containers are walked with an explicit tag per type so
  ``(1,)`` and ``[1]`` cannot collide.
* :class:`Keys` -- the ``{key: sha256}`` accumulator, which refuses a
  duplicate key and which CALLS the entry point inside
  ``warnings.catch_warnings(record=True)`` with ``simplefilter('always')``, so
  every warning is folded in EMISSION order (category name + message text).

No float ``==`` and no tolerance appears anywhere in this file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------
def bind(tree):
    """Import lumenairy, refuse anything outside ``tree``, print what bound."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    msg = ("[bind] lumenairy.__file__ = %s\n[bind] lumenairy.__version__ = %s\n"
           "[bind] tree = %s\n" % (got, getattr(lumenairy, '__version__', '?'),
                                   want))
    sys.stdout.write(msg)
    sys.stderr.write(msg)
    sys.stdout.flush()
    if os.path.normcase(got).startswith(os.path.normcase(want) + os.sep):
        return lumenairy
    raise SystemExit("WRONG TREE: %r is not under %r" % (got, want))


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------
_MAXDEPTH = 32


def fold(m, x, depth=0):
    if depth > _MAXDEPTH:
        m.update(b"TOODEEP")
        return
    # a jax / cupy array is not an np.ndarray: normalise by DLPack-free
    # __array__ only when the object advertises a shape and a dtype.
    if not isinstance(x, np.ndarray) and not isinstance(x, (str, bytes)) \
            and hasattr(x, 'shape') and hasattr(x, 'dtype') \
            and not isinstance(x, np.generic) and not isinstance(x, type):
        try:
            x = np.asarray(x)
        except Exception:                                   # noqa: BLE001
            pass
    if isinstance(x, np.ndarray):
        m.update(b"A|" + str(x.dtype).encode() + b"|"
                 + repr(tuple(x.shape)).encode() + b"|")
        m.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, np.generic):
        m.update(b"G|" + str(x.dtype).encode() + b"|")
        m.update(np.asarray(x).tobytes())
    elif isinstance(x, bool):
        m.update(b"B|" + (b"T" if x else b"F"))
    elif isinstance(x, int):
        m.update(b"I|" + str(x).encode())
    elif isinstance(x, float):
        m.update(b"F|")
        m.update(np.float64(x).tobytes())
    elif isinstance(x, complex):
        m.update(b"C|")
        m.update(np.complex128(x).tobytes())
    elif isinstance(x, (bytes, bytearray)):
        m.update(b"Y|" + bytes(x))
    elif isinstance(x, str):
        m.update(b"S|" + x.encode("utf-8", "backslashreplace"))
    elif x is None:
        m.update(b"N|")
    elif isinstance(x, tuple):
        m.update(b"T|" + str(len(x)).encode())
        for v in x:
            m.update(b"\x1e")
            fold(m, v, depth + 1)
    elif isinstance(x, list):
        m.update(b"L|" + str(len(x)).encode())
        for v in x:
            m.update(b"\x1e")
            fold(m, v, depth + 1)
    elif isinstance(x, (set, frozenset)):
        m.update(b"E|" + str(len(x)).encode())
        for v in sorted(x, key=repr):
            m.update(b"\x1e")
            fold(m, v, depth + 1)
    elif isinstance(x, dict):
        m.update(b"D|" + str(len(x)).encode())
        for k in sorted(x, key=repr):
            m.update(b"\x1d" + repr(k).encode("utf-8", "backslashreplace")
                     + b"\x1c")
            fold(m, x[k], depth + 1)
    elif isinstance(x, BaseException):
        m.update(b"X|" + type(x).__name__.encode() + b"|"
                 + str(x).encode("utf-8", "backslashreplace"))
    elif hasattr(x, "__dict__") and not callable(x):
        m.update(b"O|" + type(x).__name__.encode() + b"|")
        fold(m, {k: v for k, v in sorted(vars(x).items())
                 if not k.startswith("__")}, depth + 1)
    else:
        m.update(b"R|" + type(x).__name__.encode() + b"|"
                 + repr(x).encode("utf-8", "backslashreplace"))


def sha(x):
    m = hashlib.sha256()
    fold(m, x)
    return m.hexdigest()


# ---------------------------------------------------------------------------
# the key accumulator
# ---------------------------------------------------------------------------
class Keys:
    def __init__(self):
        self.digests = {}
        self.notes = {}

    def _record(self, fn, a, kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                val = fn(*a, **kw)
                head = ("ok", type(val).__name__, val)
            except BaseException as exc:                     # noqa: BLE001
                head = ("raised", type(exc).__name__, str(exc))
        tail = [(w.category.__name__, str(w.message)) for w in caught]
        return (head, tail)

    def call(self, key, fn, *a, **kw):
        if key in self.digests:
            raise SystemExit("duplicate key %r" % key)
        rec = self._record(fn, a, kw)
        self.digests[key] = sha(rec)
        head, tail = rec
        self.notes[key] = {
            "outcome": head[0], "type": head[1],
            "detail": (head[2][:200] if head[0] == "raised" else ""),
            "warnings": [(c, t[:160]) for c, t in tail],
        }
        if head[0] == "raised":
            sys.stderr.write("[raised] %s -> %s: %s\n"
                             % (key, head[1], head[2][:160]))
        return rec

    def write(self, path, meta):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "digests": self.digests,
                       "notes": self.notes}, fh, indent=1, sort_keys=True,
                      default=str)
        sys.stderr.write("[keys] %d keys -> %s\n" % (len(self.digests), path))


def build_tag():
    plat = "WIN" if sys.platform.startswith("win") else "WSL"
    return "%s-py%d.%d" % (plat, sys.version_info.major, sys.version_info.minor)
