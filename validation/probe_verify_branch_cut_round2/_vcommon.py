"""Shared helpers for the round-2 branch-cut VERIFICATION probes.

Every probe in this directory calls :func:`pin_tree` FIRST.  ``sys.path[0]``
is the SCRIPT's directory, not the working directory, and this box carries an
editable install of ``lumenairy`` pointing at a DIFFERENT tree -- so a probe
launched without ``PYTHONPATH=.`` silently measures the wrong library.  The pin
derives the expected tree from ``__file__`` (three parents up from this module)
and REFUSES to produce a number if ``lumenairy.__file__`` is anywhere else.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]


def pin_tree():
    """Import lumenairy and assert it came from THIS worktree.  Returns the
    module."""
    if str(TREE) not in sys.path:
        sys.path.insert(0, str(TREE))
    import lumenairy
    got = Path(lumenairy.__file__).resolve()
    want = TREE / "lumenairy" / "__init__.py"
    if got != want:
        raise SystemExit(
            f"REFUSING to measure: lumenairy imported from {got}, "
            f"expected {want}.  Run with PYTHONPATH=. from {TREE}.")
    return lumenairy


def arm() -> str:
    """Detect which ARM of the round-2 change this tree carries, from the LIVE
    source rather than from a git revision: PRE = the five private
    ``_sqrt_decay`` copies still exist under ``pmm/``; POST = exactly one
    definition, in ``rcwa/_core.py``."""
    import re
    n = 0
    where = []
    for p in (TREE / "lumenairy" / "elements").rglob("*.py"):
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^\s*def\s+_sqrt_decay\s*\(", src, re.M):
            n += 1
            where.append(str(p.relative_to(TREE)).replace("\\", "/"))
    return ("post" if n == 1 else "pre" if n > 1 else "unknown"), n, sorted(where)


def stamp() -> dict:
    import numpy as np
    lm = pin_tree()
    a, n, where = arm()
    return {
        "lumenairy_file": lm.__file__,
        "tree": str(TREE),
        "arm": a,
        "n_sqrt_decay_definitions": n,
        "sqrt_decay_sites": where,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def dump(path, payload):
    payload = dict(payload)
    payload["_stamp"] = stamp()
    Path(path).write_text(json.dumps(payload, indent=1, default=str),
                          encoding="utf-8")
    print(f"[wrote] {path}")


# --------------------------------------------------------------------------
# fixture builders -- MINE, not the fix's
# --------------------------------------------------------------------------

def weak_cell(npx: int, npy: int, eps_bg: float, rel: float,
              bx: int = 2, by: int = 2):
    """A weakly modulated square cell: ``eps_bg`` everywhere with a
    ``bx x by`` block at ``eps_bg * (1 + rel)``.  Piecewise constant on the
    pixel walls, so an RCWA solve of the same device is EXACT under Laurent."""
    import numpy as np
    c = np.full((npx, npy), float(eps_bg), dtype=complex)
    c[:bx, :by] = eps_bg * (1.0 + rel)
    return c
