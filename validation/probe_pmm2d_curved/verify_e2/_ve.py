"""Shared helpers of the Phase E2 VERIFIER probes (independent of build_e2/).

Every probe asserts that ``lumenairy`` is imported from the tree it is meant
to measure: this worktree, or the tree named by ``LUM_TREE`` (the PRE tree =
``git archive eae470d9`` extracted to ``C:/tmp/vcurved_e2_pre``).  Run with
BLAS pinned on the command line:

  cd /c/tmp/lum_vcurved_e2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_e2 \
    python validation/probe_pmm2d_curved/verify_e2/<probe>.py ...

JSON files are suffixed ``_win`` (Windows build) or ``_wsl`` (WSL Ubuntu
second build, ``~/lumvenv``).
"""
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE") or os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, ROOT)
import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
TREE = "pre" if os.environ.get("LUM_TREE") else "post"
BUILD = "wsl" if sys.platform.startswith("linux") else "win"

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402,F401


def env_record():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "tree": TREE, "build": BUILD, "machine": platform.node(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}


def _jsonable(v):
    if isinstance(v, complex) or (isinstance(v, np.generic)
                                  and np.iscomplexobj(v)):
        return [float(np.real(v)), float(np.imag(v))]
    if np.isscalar(v):
        return float(v)
    a = np.asarray(v)
    if np.iscomplexobj(a):
        return {"re": a.real.tolist(), "im": a.imag.tolist()}
    return a.tolist()


def dump(name, obj):
    """Write ``obj`` (+ env) to ``<name>_<build>.json`` next to this file."""
    obj = dict(obj)
    obj["env"] = env_record()
    fn = f"{name}_{BUILD}.json"
    with open(os.path.join(HERE, fn), "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    print("wrote", fn)
    return fn


def rt_diff(a, b):
    """max |R_a - R_b|, |T_a - T_b| over orders and both inputs."""
    return float(max(np.abs(np.asarray(a[1]) - np.asarray(b[1])).max(),
                     np.abs(np.asarray(a[2]) - np.asarray(b[2])).max()))


def closure(res):
    R, T = np.asarray(res[1]), np.asarray(res[2])
    return np.abs(R.sum(1) + T.sum(1) - 1.0)


def solve(st, wl=1.0, theta=0.0, phi=0.0, retain=False):
    st.set_source(wl, theta=theta, phi=phi)
    o, R, T, J = st.solve(retain_internal=retain)
    return np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)
