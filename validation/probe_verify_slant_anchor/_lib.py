"""Shared plumbing for the INDEPENDENT verification of V1 / V2 / O2.

Every probe in this directory imports :func:`arm`, which decides WHICH TREE
answered from ``lumenairy.__file__`` alone and refuses any tree that is not one
of the three this verification is allowed to read:

===========  =====================================================
``post``     ``C:/tmp/lum_vslant``      -- wave2 tip 2ec4359, HOLDS the fix
``pre``      ``C:/tmp/lum_vslant_pre``  -- 4a987e3, the fix's branch point
``rev``      ``C:/tmp/lum_vslant_rev``  -- HEAD with ONLY c8cb563 / 006c031 /
                                          b678747 reverted (the surgical
                                          isolation: everything else that
                                          landed on wave2 after 4a987e3 is
                                          still present)
===========  =====================================================

``pre`` and ``rev`` are BOTH carried because wave2/pmm2d moved between the
fix's branch point and the tip for reasons unrelated to the fix (the mortar
round-2 and sliver round-2 merges both edited ``pmm/stack.py``), so ``pre``
alone cannot attribute a moved byte to this fix.

Select the arm by exporting ``LUM_ARM_TREE`` before running a probe; the
header block every probe carries puts that directory ahead of the script's own
directory on ``sys.path``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

_TREES = {
    "post": ("c/tmp/lum_vslant", ),
    "pre": ("c/tmp/lum_vslant_pre", ),
    "rev": ("c/tmp/lum_vslant_rev", ),
}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _norm(p):
    p = os.path.abspath(p).replace("\\", "/").lower()
    if p.startswith("/mnt/"):
        p = p[5:]
    elif len(p) > 1 and p[1] == ":":
        p = p[0] + p[2:]
    return p.lstrip("/")


def arm():
    """``('post'|'pre'|'rev', tree_path)`` decided from ``lumenairy.__file__``."""
    import lumenairy
    root = _norm(os.path.dirname(os.path.dirname(lumenairy.__file__)))
    for name, keys in _TREES.items():
        for k in keys:
            if root == k:
                return name, os.path.dirname(os.path.dirname(
                    lumenairy.__file__))
    raise SystemExit(
        f"_lib.arm: lumenairy resolved to {lumenairy.__file__!r} (normalized "
        f"{root!r}), which is NOT one of the three trees this verification is "
        f"allowed to read {sorted(_TREES)}.  Export LUM_ARM_TREE.")


def build():
    """``'win'`` or ``'wsl'`` -- the interpreter/BLAS arm, not the tree."""
    return "win" if platform.system() == "Windows" else "wsl"


def stamp():
    import lumenairy
    a, tree = arm()
    try:
        import scipy
        sv = scipy.__version__
    except Exception:                                       # noqa: BLE001
        sv = None
    try:
        import jax
        jv = jax.__version__
        jx64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:                                       # noqa: BLE001
        jv, jx64 = None, None
    return dict(
        arm=a, build=build(), tree=tree,
        lumenairy=lumenairy.__version__,
        lumenairy_file=lumenairy.__file__,
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, scipy=sv, jax=jv, jax_x64=jx64,
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def save(name, payload):
    a, _ = arm()
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(stamp=stamp(), **payload)
    path = os.path.join(RESULTS, f"{name}.{a}.{build()}.json")
    with open(path, "w", encoding="cp1252", errors="backslashreplace") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=_default)
    print(f"[{a}/{build()}] wrote {path}")
    return path


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def sha(*arrays):
    """sha256 over the raw C-contiguous bytes of every array, in order."""
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"<None>")
            continue
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def rel(a, b):
    """Relative Frobenius residual ``||a - b|| / ||b||`` (0.0 when both 0)."""
    a = np.asarray(a)
    b = np.asarray(b)
    nb = float(np.linalg.norm(np.ravel(b)))
    d = float(np.linalg.norm(np.ravel(a) - np.ravel(b)))
    if nb == 0.0:
        return d
    return d / nb


def jsons(name):
    """Load every arm/build JSON written under ``name``."""
    out = {}
    for fn in sorted(os.listdir(RESULTS)):
        if fn.startswith(name + ".") and fn.endswith(".json"):
            key = fn[len(name) + 1:-5]
            with open(os.path.join(RESULTS, fn), encoding="cp1252") as fh:
                out[key] = json.load(fh)
    return out
