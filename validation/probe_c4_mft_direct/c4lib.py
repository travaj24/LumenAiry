"""WP-C4 probe harness -- the direct-matrix MFT route as the automatic default.

Every probe here runs in a CHILD process whose ``cwd`` and ``PYTHONPATH`` name
ONE tree -- the live worktree, or a read-only ``git archive`` extraction of the
base commit.  :func:`anchor` refuses to continue unless ``lumenairy.__file__``
resolves under that tree and PRINTS what it bound, because this box carries an
installed lumenairy that predates the ``method=`` keyword and binds silently
otherwise (MEASURED during hygiene-2 round 3, recorded in
``validation/probe_wave5_hyg2_round3/r3_budget_exact.py``'s docstring).

Two kinds of output:

* ``{key: sha256}`` digest maps for archive-to-archive BIT IDENTITY, compared
  key by key by :mod:`c4_compare`; the digest folds the returned object's type
  name, every warning in EMISSION order, and NaN / signed zero as BYTES, so no
  float ``==`` appears anywhere in it; and
* plain measurement JSON (timings, peaks, residuals), which is NOT compared bit
  for bit -- it is the evidence the report's tables are read from.

Shares no code with ``probe_wave5_hyg2/hlib.py`` or
``probe_verify_wave5_hyg2/vlib.py``; the digest encoding is deliberately the
same shape so keys are comparable by eye across the three campaigns.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings


def anchor(tree: str):
    """Import lumenairy, refuse anything outside ``tree``, print what bound."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    print(f"[anchor] lumenairy.__file__ = {got}", file=sys.stderr)
    print(f"[anchor] lumenairy.__version__ = {lumenairy.__version__}",
          file=sys.stderr)
    try:
        same = os.path.commonpath([got, want]) == want
    except ValueError:                    # different drives on Windows
        same = False
    if not same:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def build_tag() -> str:
    """``WIN-py3.14`` / ``WSL-py3.12`` -- the two builds, named the same way
    every Wave-5 document names them."""
    v = ".".join(sys.version.split()[0].split(".")[:2])
    return ("WIN" if sys.platform.startswith("win") else "WSL") + f"-py{v}"


def load_snapshot() -> dict:
    """How busy is the box?  A process census AND a measured single-thread
    reference loop, so a contended reading is visible in the JSON rather than
    argued about in the report."""
    import numpy as np
    try:
        if sys.platform.startswith('win'):
            out = subprocess.run(['tasklist'], capture_output=True, text=True,
                                 timeout=120).stdout
            npy = sum(1 for ln in out.splitlines()
                      if ln.lower().startswith('python'))
            nproc = len(out.splitlines())
        else:
            out = subprocess.run(['ps', '-e'], capture_output=True, text=True,
                                 timeout=120).stdout
            npy = sum(1 for ln in out.splitlines() if 'python' in ln)
            nproc = len(out.splitlines())
    except Exception as exc:                        # noqa: BLE001 -- recorded
        npy, nproc = -1, f"{type(exc).__name__}"
    a = np.ones(1 << 16)
    t0 = time.perf_counter()
    for _ in range(200):
        a = a * 1.0000001
    ref = time.perf_counter() - t0
    return {'python_processes': npy, 'total_processes': nproc,
            'reference_loop_s': ref}


def cold():
    """The production regime: every registered library cache dropped."""
    try:
        from lumenairy._cache_registry import (
            clear_all_registered_caches)
        clear_all_registered_caches()
    except Exception:                               # noqa: BLE001
        try:
            from lumenairy.propagators._bluestein import _clear_h_fft_cache
            _clear_h_fft_cache()
        except Exception:                           # noqa: BLE001
            pass
    gc.collect()


def digest(obj, caught=()) -> str:
    """sha256 over the RAW BYTES of an array plus its dtype, shape and the
    warnings emitted, in emission order.

    NaN and signed zero fold as bytes, so ``-0.0`` and ``+0.0`` are different
    keys and two NaNs with different payloads are different keys.  This is the
    point: a bit-identity claim that uses float ``==`` is not one.
    """
    import numpy as np
    h = hashlib.sha256()
    h.update(type(obj).__name__.encode('utf-8'))
    a = np.asarray(obj)
    h.update(str(a.dtype).encode('utf-8'))
    h.update(str(a.shape).encode('utf-8'))
    h.update(np.ascontiguousarray(a).tobytes())
    for w in caught:
        h.update(b'|W|')
        h.update(getattr(w.category, '__name__', str(w.category))
                 .encode('utf-8'))
        h.update(str(w.message).encode('utf-8', 'replace'))
    return h.hexdigest()


class Recorder:
    """Collects ``{key: sha256}`` while capturing warnings per call."""

    def __init__(self):
        self.keys = {}
        self.notes = {}

    def record(self, key, fn, *args, **kwargs):
        cold()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                out = fn(*args, **kwargs)
            except Exception as exc:                # noqa: BLE001 -- recorded
                self.keys[key] = ('RAISE:' + type(exc).__name__ + ':'
                                  + str(exc)[:200])
                return None
        self.keys[key] = digest(out, caught)
        if caught:
            self.notes[key] = [getattr(c.category, '__name__', '?')
                               for c in caught]
        return out


def write_json(obj, path):
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    print(f"-> {path}", file=sys.stderr)
