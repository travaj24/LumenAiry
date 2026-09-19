"""Wave-5 hygiene-2 probe harness.

Every probe module here runs in a CHILD process whose ``cwd`` and
``PYTHONPATH`` name ONE tree -- either the live worktree or a read-only
``git archive`` extraction.  :func:`anchor` refuses to continue unless
``lumenairy.__file__`` resolves under that tree and PRINTS what it bound, so a
probe can never silently measure a pip install or the other arm's checkout.

Two kinds of output are produced by the probes that use this module:

* a ``{key: sha256}`` digest map, for archive-to-archive BIT-IDENTITY
  (:class:`Probe`, compared key by key by ``run_bitid.py``); and
* a plain measurement JSON (timings, peaks, residuals), which is NOT compared
  bit for bit -- it is the evidence the report's tables are read from.

The digest path folds the returned object's type name, every warning in
EMISSION order, and NaN / signed zero as BYTES.  No float ``==`` appears
anywhere in it.

Shares no code with ``validation/probe_verify_b11c/vlib.py``; the digest
encoding is deliberately the same shape so the two campaigns' keys are
comparable by eye.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings


def anchor(tree: str):
    """Import lumenairy, refuse anything outside ``tree``, print what bound."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    print(f"[anchor] lumenairy.__file__ = {got}", file=sys.stderr)
    print(f"[anchor] lumenairy.__version__ = {lumenairy.__version__}",
          file=sys.stderr)
    if os.path.commonpath([got, want]) != want:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def _feed(m, x, depth=0):
    import numpy as np
    if depth > 24:
        m.update(b"<deep>")
        return
    if isinstance(x, np.ndarray):
        m.update(b"nd\x00" + str(x.dtype).encode() + b"\x00"
                 + repr(x.shape).encode() + b"\x00")
        m.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, np.generic):
        m.update(b"sc\x00" + str(x.dtype).encode() + b"\x00")
        m.update(np.asarray(x).tobytes())
    elif isinstance(x, bool):
        m.update(b"bo\x00" + (b"1" if x else b"0"))
    elif isinstance(x, int):
        m.update(b"in\x00" + str(x).encode())
    elif isinstance(x, float):
        m.update(b"fl\x00")
        m.update(np.float64(x).tobytes())
    elif isinstance(x, complex):
        m.update(b"cx\x00")
        m.update(np.complex128(x).tobytes())
    elif isinstance(x, (bytes, bytearray)):
        m.update(b"by\x00" + bytes(x))
    elif isinstance(x, str):
        m.update(b"st\x00" + x.encode("utf-8", "backslashreplace"))
    elif x is None:
        m.update(b"no\x00")
    elif isinstance(x, (list, tuple)):
        m.update(b"sq\x00" + type(x).__name__.encode() + b"\x00"
                 + str(len(x)).encode())
        for v in x:
            m.update(b"\x01")
            _feed(m, v, depth + 1)
    elif isinstance(x, (set, frozenset)):
        m.update(b"se\x00" + str(len(x)).encode())
        for v in sorted(x, key=repr):
            m.update(b"\x01")
            _feed(m, v, depth + 1)
    elif isinstance(x, dict):
        m.update(b"di\x00" + str(len(x)).encode())
        for k in sorted(x, key=repr):
            m.update(b"\x02" + repr(k).encode() + b"\x03")
            _feed(m, x[k], depth + 1)
    elif isinstance(x, BaseException):
        m.update(b"ex\x00" + type(x).__name__.encode() + b"\x00"
                 + str(x).encode("utf-8", "backslashreplace"))
    elif hasattr(x, "__dict__") and not callable(x):
        m.update(b"ob\x00" + type(x).__name__.encode() + b"\x00")
        _feed(m, {k: v for k, v in sorted(vars(x).items())
                  if not k.startswith("__")}, depth + 1)
    else:
        m.update(b"rp\x00" + type(x).__name__.encode() + b"\x00"
                 + repr(x).encode("utf-8", "backslashreplace"))


def digest(x) -> str:
    m = hashlib.sha256()
    _feed(m, x)
    return m.hexdigest()


def record(fn, *a, **kw):
    """Call ``fn`` and fold EVERYTHING observable into one digestible record."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            val = fn(*a, **kw)
            out = ("ok", type(val).__name__, val)
        except BaseException as exc:            # noqa: BLE001 -- recorded
            out = ("raised", type(exc).__name__, str(exc))
    return (out, [(w.category.__name__, str(w.message)) for w in caught])


class Probe:
    """Accumulates ``{key: digest}`` and refuses a duplicate key."""

    def __init__(self):
        self.out = {}

    def add(self, key, value):
        if key in self.out:
            raise SystemExit(f"duplicate probe key {key!r}")
        self.out[key] = digest(value)

    def call(self, key, fn, *a, **kw):
        rec = record(fn, *a, **kw)
        if rec[0][0] == "raised":
            print(f"[raised] {key}: {rec[0][1]}: {rec[0][2][:140]}",
                  file=sys.stderr)
        self.add(key, rec)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.out, fh, indent=1, sort_keys=True)
        print(f"[probe] {len(self.out)} keys -> {path}", file=sys.stderr)


def build_tag() -> str:
    """``WIN-py3.14`` / ``WSL-py3.12`` -- which of the two builds this is."""
    plat = "WIN" if sys.platform.startswith("win") else "WSL"
    return f"{plat}-py{sys.version_info.major}.{sys.version_info.minor}"


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    print(f"[probe] measurement -> {path}", file=sys.stderr)
