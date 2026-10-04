"""Shared harness of the VERIFIER's probes (2026-10-04).

LUM_TREE selects the tree (default: this worktree); the import path is
asserted.  BLAS threads must be set on the command line (<= 2).  jax x64.
"""
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(os.environ.get(
    "LUM_TREE", os.path.join(HERE, "..", ".."))))
if ROOT not in [os.path.normcase(os.path.abspath(p)) for p in sys.path]:
    sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(_k) in ("1", "2"), f"{_k} must be 1 or 2"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402,F401

BUILD = "wsl" if sys.platform.startswith("linux") else "win"
warnings.simplefilter("ignore")


def env():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "jax": jax.__version__, "lumenairy": lumenairy.__file__,
            "build": BUILD, "tag": os.environ.get("VTAG", "")}


def dump(name, obj):
    obj = dict(obj)
    obj["env"] = env()
    tag = os.environ.get("VTAG")
    root = name + (f"_{tag}" if tag else "")
    path = os.path.join(HERE, f"{root}_{BUILD}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: (
            float(o) if np.ndim(o) == 0 and np.isrealobj(o) else str(o)))
    print("wrote", path)
    return path


def fd_halving(fun, x0, h0=2e-2):
    """Central differences at h0, h0/2, h0/4; Richardson of the last two;
    PREMISE = ratio of successive rung changes (4 for a clean h^2 law):
    returned as (norm ratio, min, max over the entries whose first change
    is >= 1e-2 of the largest)."""
    rows = []
    for h in (h0, h0 / 2, h0 / 4):
        rows.append((np.asarray(fun(x0 + h), dtype=float)
                     - np.asarray(fun(x0 - h), dtype=float)) / (2 * h))
    rows = np.asarray(rows)
    c1 = np.abs(rows[0] - rows[1])
    c2 = np.abs(rows[1] - rows[2])
    big = c1 >= 1e-2 * np.max(c1)
    ratio = c1[big] / np.maximum(c2[big], 1e-300)
    rich = (4.0 * rows[2] - rows[1]) / 3.0
    rich1 = (4.0 * rows[1] - rows[0]) / 3.0
    norm = float(np.max(c1) / max(np.max(c2), 1e-300))
    # FD resolution: the change of the Richardson value between the two
    # rung pairs, relative to the largest entry (an O(h^4) estimate)
    res = float(np.max(np.abs(rich - rich1)) / np.max(np.abs(rich)))
    return rich, (norm, float(ratio.min()), float(ratio.max()), res)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def tic():
    return time.perf_counter()
