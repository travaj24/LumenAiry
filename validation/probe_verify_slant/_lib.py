"""Shared plumbing for the INDEPENDENT verification of the pure staggered 2-D
PMM's native constant-shear slant (``docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_
2026_09_10.md``).

Every probe here is written from the SPEC and the geometry, not from the build's
probes: fixtures, oracles and reference constructions are this verification's
own.  The one thing borrowed is the discipline -- each script asserts which
``lumenairy`` it imported and stamps that into its JSON, because the arms of
this verification run against different checkouts AND different builds:

  * ``tip``  -- ``C:/tmp/lum_vslant`` (verify/slant @ 8b9af801), slant present;
  * ``base`` -- ``D:/Metacept/.../Lumenairy`` (2efc7a2), the same wave-2 tree
    with the slant REMOVED, i.e. the "without" arm of the bit-identity gate;

and each of those is reachable from Windows (``C:\\...``) or from WSL
(``/mnt/c/...``), which is why the roots below are listed in every spelling.
Set ``VSLANT_TAG=wsl`` on the WSL arm so the two builds' JSONs sit side by side.
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

TIP_ROOTS = (str(pathlib.PureWindowsPath("C:/tmp/lum_vslant")),
             "C:/tmp/lum_vslant", "/mnt/c/tmp/lum_vslant")
_BASE = "D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
BASE_ROOTS = (str(pathlib.PureWindowsPath(_BASE)), _BASE,
              "/mnt/d/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics"
              "/Lumenairy")


def arm():
    """``'tip'`` or ``'base'`` -- decided by where ``lumenairy`` was imported
    FROM, never by a flag, so a mis-set PYTHONPATH cannot mislabel a run."""
    lib = str(pathlib.Path(lumenairy.__file__).resolve())
    if any(lib.startswith(r) for r in TIP_ROOTS):
        return "tip"
    if any(lib.startswith(r) for r in BASE_ROOTS):
        return "base"
    raise SystemExit(
        f"probe_verify_slant: lumenairy imported from an unexpected tree: "
        f"{lib}\n  expected under {TIP_ROOTS} (tip) or {BASE_ROOTS} (base).")


def env():
    import scipy
    return dict(
        lumenairy=str(pathlib.Path(lumenairy.__file__).resolve()),
        version=getattr(lumenairy, "__version__", "?"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        arm=arm(),
        tag=os.environ.get("VSLANT_TAG", "win"),
        threads={k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS")},
    )


def dump(name, payload):
    tag = os.environ.get("VSLANT_TAG", "win")
    out = RESULTS / f"{name}_{arm()}_{tag}.json"
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
