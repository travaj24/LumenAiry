"""Shared harness for the WP-B11c bit-identity probes.

Every probe is a pure function of the library: it is executed in a CHILD
process whose ``cwd`` and ``PYTHONPATH`` are set to ONE tree (a ``git archive``
of the parent commit, or the working tree), asserts ``lumenairy.__file__``
lives under that tree BEFORE it computes anything, and writes a JSON map of
SHA-256 digests over the exact IEEE-754 bytes (plus dtype and shape) of every
answer.  Two trees' JSON maps are then compared key by key.

Never through pytest: pytest puts the repository root ahead of ``PYTHONPATH``,
so a probe run under it would import the working tree from both arms.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys


def bind(expected_root: str):
    """Import lumenairy and REFUSE unless it resolves under ``expected_root``.

    Called first in every probe, before any numeric work, so a probe can never
    silently measure the pip -e install (which points at another checkout on a
    different branch).
    """
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(expected_root)
    if not got.startswith(want):
        raise SystemExit(
            f"probe bound the WRONG tree: lumenairy.__file__ = {got!r}, "
            f"expected it under {want!r}")
    return lumenairy


def h(obj) -> str:
    """SHA-256 over the exact bytes of an answer, tagged by its structure.

    Arrays hash dtype + shape + raw C-contiguous bytes (so a float that moved
    one ULP, a dtype promotion and a reshape are all visible).  Everything else
    hashes its ``repr`` after a canonicalising walk, which makes warning
    messages, exception texts, tuples of ints and dict orderings comparable.
    """
    import numpy as np

    m = hashlib.sha256()

    def walk(x):
        if isinstance(x, np.ndarray):
            m.update(b"A|")
            m.update(str(x.dtype).encode())
            m.update(b"|")
            m.update(str(x.shape).encode())
            m.update(b"|")
            m.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, (np.generic,)):
            m.update(b"S|")
            m.update(str(x.dtype).encode())
            m.update(b"|")
            m.update(np.asarray(x).tobytes())
        elif isinstance(x, float):
            m.update(b"f|")
            m.update(np.asarray(x, dtype=np.float64).tobytes())
        elif isinstance(x, complex):
            m.update(b"c|")
            m.update(np.asarray(x, dtype=np.complex128).tobytes())
        elif isinstance(x, (list, tuple)):
            m.update(b"L|" if isinstance(x, list) else b"T|")
            m.update(str(len(x)).encode())
            for v in x:
                m.update(b";")
                walk(v)
        elif isinstance(x, dict):
            m.update(b"D|")
            for k in sorted(x, key=repr):
                m.update(repr(k).encode())
                m.update(b"=")
                walk(x[k])
                m.update(b";")
        elif isinstance(x, BaseException):
            m.update(b"E|")
            m.update(type(x).__name__.encode())
            m.update(b"|")
            m.update(str(x).encode())
        else:
            m.update(b"R|")
            m.update(repr(x).encode())

    walk(obj)
    return m.hexdigest()


def caught(fn, *a, **kw):
    """Run ``fn`` and return a hashable record of EVERYTHING it produced:
    its value (or the exception type and message) and every warning it emitted
    (category name + text), so a refactor that moves a message or a warning's
    category is caught as loudly as one that moves a float."""
    import warnings

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            val = fn(*a, **kw)
            kind = "value"
        except BaseException as e:               # noqa: BLE001 - recorded
            val = (type(e).__name__, str(e))
            kind = "raised"
    return (kind, val,
            tuple(sorted((w.category.__name__, str(w.message)) for w in wl)))


def emit(results: dict) -> None:
    """Write the probe's {key: digest} map to stdout as JSON."""
    json.dump({k: (v if isinstance(v, str) else h(v))
               for k, v in results.items()}, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
