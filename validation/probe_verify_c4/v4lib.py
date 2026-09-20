"""VERIFY-WP-C4 probe harness -- independent of ``probe_c4_mft_direct/c4lib.py``.

Written for the adversarial re-measurement of WP-C4's ``method='auto'`` shape
rule.  Shares no code with the branch's own probes on purpose: the point of a
verification is that the numbers come out of a SECOND instrument.

Three things this harness does that the branch's harness does not:

* **The timing loop INTERLEAVES the routes.**  ``probe_c4_mft_direct/
  c4_ladder.py`` times all repeats of route A, then all repeats of route B.
  On a box that carries other work, a load excursion that lands inside one
  route's block biases exactly the ratio the boundary is read from.  Here the
  repeat loop is outermost and the route order ROTATES by repeat index, so any
  drift is spread across all three routes rather than charged to one.
* **The reference loop is measured per SHAPE**, before and after that shape's
  whole interleaved block, and its drift is carried into the JSON per row --
  so a row measured across a load excursion is identifiable after the fact
  rather than argued about.
* **Every route's answer is digested in the same pass**, so a timing row that
  is fast because the route did not run is impossible.

Tree binding: every probe takes the tree it is pinned to as ``argv[1]``,
imports lumenairy, and REFUSES to continue unless ``lumenairy.__file__``
resolves under it -- this box carries an installed lumenairy and ``''``
precedes ``PYTHONPATH`` on ``sys.path``, so a script run from inside a
worktree binds the worktree whatever ``PYTHONPATH`` says.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time


def anchor(tree: str):
    """Import lumenairy; refuse any tree but ``tree``; print what bound."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    print(f"[anchor] lumenairy.__file__   = {got}", file=sys.stderr)
    print(f"[anchor] lumenairy.__version__= {lumenairy.__version__}",
          file=sys.stderr)
    try:
        ok = os.path.commonpath([got, want]) == want
    except ValueError:                       # different drives on Windows
        ok = False
    if not ok:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def build_tag() -> str:
    v = ".".join(sys.version.split()[0].split(".")[:2])
    return ("WIN" if sys.platform.startswith("win") else "WSL") + f"-py{v}"


def short_tag() -> str:
    return "win" if sys.platform.startswith("win") else "wsl"


def ref_loop() -> float:
    """A fixed single-thread workload, in seconds.  Its DRIFT between two
    readings is the box's load moving; it is recorded, never corrected for."""
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


def load_census() -> dict:
    """Process census + reference loop, so a contended reading is visible."""
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
            'ref_loop_s': ref_loop(), 'wall': time.time()}


def cold() -> None:
    """Drop every registered library cache and collect, so each repeat is the
    COLD regime the production order runs in."""
    try:
        from lumenairy._cache_registry import (
            clear_all_registered_caches)
        clear_all_registered_caches()
    except Exception:                                 # noqa: BLE001
        try:
            from lumenairy.propagators._bluestein import _clear_h_fft_cache
            _clear_h_fft_cache()
        except Exception:                             # noqa: BLE001
            pass
    gc.collect()


def digest_array(a, warns=()) -> str:
    """SHA-256 over dtype, shape and RAW BYTES, plus every warning in emission
    order.  No float ``==`` anywhere -- NaN and signed zero are bytes here."""
    import numpy as np
    h = hashlib.sha256()
    arr = np.asarray(a)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    for w in warns:
        h.update(b'\x00WARN\x00')
        h.update(str(w).encode('utf-8', 'replace'))
    return h.hexdigest()


def write_json(obj, path) -> None:
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    print(f"[wrote] {path}", file=sys.stderr)


def peak_rss_bytes() -> int:
    """Peak resident set of THIS process, in bytes, from the OS."""
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return int(getattr(mi, 'peak_wset', getattr(mi, 'rss', 0)))
    except Exception:                                 # noqa: BLE001
        try:
            import resource
            return int(resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:                             # noqa: BLE001
            return -1
