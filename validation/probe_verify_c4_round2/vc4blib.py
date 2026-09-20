"""VERIFY-WP-C4 ROUND 2 probe harness -- a THIRD instrument.

Independent of both ``validation/probe_c4_mft_direct/c4lib.py`` (the WP-C4
branch's own) and ``validation/probe_verify_c4/v4lib.py`` (round 1's
verification).  Nothing is imported from either: the point of a re-verification
is that the numbers come out of an instrument that shares no code with the
one that produced them.

What this harness insists on:

* **Tree binding.**  Every probe takes the tree it is pinned to as ``argv[1]``
  and refuses to run unless ``lumenairy.__file__`` resolves under it.  This box
  carries an installed lumenairy and ``''`` precedes ``PYTHONPATH`` on
  ``sys.path``, so a script run from inside a worktree binds the worktree
  whatever ``PYTHONPATH`` says.
* **Single-threaded on BOTH sides of every comparison.**  The three
  ``*_NUM_THREADS`` variables pin BLAS; they do NOT constrain scipy's
  pocketfft, which the separable route's 1-D passes go through, so
  ``fft_infra.SCIPY_FFT_WORKERS`` is forced to 1 in-process as well.
* **Interleaving with a rotating route order**, a cold cache before every
  single timed call, and a reference loop read before and after each shape so a
  row measured across a load excursion is identifiable after the fact.
* **Every route's answer digested in the same pass**, so a timing row cannot be
  fast because a route did not run.

Author:  Andrew Traverso
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time


def anchor(tree):
    """Import lumenairy, refuse any tree but ``tree``, and say what bound."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    sys.stderr.write(f"[anchor] lumenairy.__file__    = {got}\n")
    sys.stderr.write(f"[anchor] lumenairy.__version__ = "
                     f"{lumenairy.__version__}\n")
    try:
        ok = os.path.commonpath([got, want]) == want
    except ValueError:                        # different drives on Windows
        ok = False
    if not ok:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def single_thread_ffts():
    """Force scipy's pocketfft worker pool to one worker, in process.

    ``OMP_NUM_THREADS`` and friends do not reach it (VERIFY-WP-C4 N1), and a
    dense-vs-chirp comparison in which only one side is single-threaded reads
    the thread count, not the arithmetic.
    """
    from lumenairy.propagators import fft_infra
    fft_infra.SCIPY_FFT_WORKERS = 1
    return int(fft_infra.SCIPY_FFT_WORKERS)


def tag():
    return "win" if sys.platform.startswith("win") else "wsl"


def build():
    v = ".".join(sys.version.split()[0].split(".")[:2])
    return ("WIN" if sys.platform.startswith("win") else "WSL") + f"-py{v}"


def refloop():
    """A fixed single-thread workload in seconds.  Its DRIFT between two
    readings is the box's load moving.  Recorded, never corrected for."""
    import numpy as np
    a = np.linspace(0.0, 1.0, 1 << 16)
    best = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        s = 0.0
        for _k in range(6):
            s += float(np.dot(a, a))
        best = min(best, time.perf_counter() - t0)
    return best


def load():
    try:
        if sys.platform.startswith('win'):
            out = subprocess.run(['tasklist'], capture_output=True, text=True,
                                 timeout=180).stdout
            py = sum(1 for ln in out.splitlines()
                     if ln.lower().startswith('python'))
            tot = max(0, len(out.splitlines()) - 3)
        else:
            out = subprocess.run(['ps', '-e'], capture_output=True, text=True,
                                 timeout=180).stdout
            py = sum(1 for ln in out.splitlines() if 'python' in ln)
            tot = max(0, len(out.splitlines()) - 1)
    except Exception as exc:                          # noqa: BLE001
        py, tot = -1, f"{type(exc).__name__}: {exc}"
    return {'python_processes': py, 'total_processes': tot,
            'ref_loop_s': refloop(), 'wall': time.time()}


def cold():
    """Drop every registered library cache and collect."""
    try:
        from lumenairy._cache_registry import clear_all_registered_caches
        clear_all_registered_caches()
    except Exception:                                 # noqa: BLE001
        try:
            from lumenairy.propagators._bluestein import _clear_h_fft_cache
            _clear_h_fft_cache()
        except Exception:                             # noqa: BLE001
            pass
    gc.collect()


def digest(a, warns=()):
    """SHA-256 over dtype, shape and RAW BYTES, plus warnings in order."""
    import numpy as np
    h = hashlib.sha256()
    arr = np.asarray(a)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    for w in warns:
        h.update(b'\x00W\x00')
        h.update(str(w).encode('utf-8', 'replace'))
    return h.hexdigest()


def write(obj, path):
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    sys.stderr.write(f"[wrote] {path}\n")


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
