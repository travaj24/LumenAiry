"""Shared plumbing for the FIX of D1 -- the hybrid 2-D PMM's FRAME-referenced
transmitted amplitudes on a SLANTED PATTERNED layer
(``docs/audits/FIX_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md``).

Every arm asserts which ``lumenairy`` it imported and stamps the resolved path,
interpreter, numpy, scipy and thread caps into its JSON, because the arms run
against different checkouts AND different builds:

  * ``fix``  -- ``C:/tmp/lum_hyb`` (fix/hybrid-slant-transmission-anchor);
  * ``base`` -- ``D:/Metacept/.../Lumenairy``, the READ-ONLY main clone, i.e.
    the reference for every bit-identity gate.

Both are reachable from Windows (``C:\...``) or WSL (``/mnt/c/...``), which is
why the roots below are listed in every spelling.  Set ``FIXTAG=wsl`` on the
WSL arm so the two builds' JSONs sit side by side.
"""
import json
import os
import pathlib
import sys

import numpy as np

import lumenairy

_HERE = pathlib.Path(__file__).resolve().parent
RESULTS = _HERE / "results"
RESULTS.mkdir(exist_ok=True)

FIX_ROOTS = (str(pathlib.PureWindowsPath("C:/tmp/lum_hyb")),
             "C:/tmp/lum_hyb", "/mnt/c/tmp/lum_hyb")
_BASE = "D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
BASE_ROOTS = (str(pathlib.PureWindowsPath(_BASE)), _BASE,
              "/mnt/d/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics"
              "/Lumenairy")


def arm():
    """``'fix'`` or ``'base'`` -- decided by where ``lumenairy`` was imported
    FROM, never by a flag, so a mis-set PYTHONPATH cannot mislabel a run."""
    lib = str(pathlib.Path(lumenairy.__file__).resolve())
    if any(lib.startswith(r) for r in FIX_ROOTS):
        return "fix"
    if any(lib.startswith(r) for r in BASE_ROOTS):
        return "base"
    raise SystemExit(
        f"probe_fix_hybrid_slant_anchor: lumenairy imported from an "
        f"unexpected tree: {lib}\n  expected under {FIX_ROOTS} (fix) or "
        f"{BASE_ROOTS} (base).")


def env():
    import scipy
    return dict(
        lumenairy=str(pathlib.Path(lumenairy.__file__).resolve()),
        version=getattr(lumenairy, "__version__", "?"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        arm=arm(),
        stage=stage(),
        tag=os.environ.get("FIXTAG", "win"),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
    )


def stage():
    """``'pre'`` or ``'post'`` -- which side of the library fix this run is on.
    Set ``FIXSTAGE=pre`` before the fix commit, ``post`` after."""
    return os.environ.get("FIXSTAGE", "post")


def dump(name, payload):
    tag = os.environ.get("FIXTAG", "win")
    out = RESULTS / f"{name}_{stage()}_{arm()}_{tag}.json"
    payload = dict(payload)
    payload["_env"] = env()
    out.write_text(json.dumps(payload, indent=1, default=_jdefault),
                   encoding="cp1252")
    print(f"[wrote] {out}")
    return out


def _jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def sha(*arrs):
    import hashlib
    m = hashlib.sha256()
    for a in arrs:
        m.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return m.hexdigest()


def mx(a, b):
    """max abs difference, shape-safe."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)))


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    d = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b)) / d)


def align(o1, o2):
    """Index pairs of the orders common to two 2-D order lists."""
    k1 = {tuple(int(v) for v in o1[i]): i for i in range(len(o1))}
    k2 = {tuple(int(v) for v in o2[i]): i for i in range(len(o2))}
    common = sorted(set(k1) & set(k2))
    return [k1[c] for c in common], [k2[c] for c in common], common
