"""Shared harness of the JAX symmetric-point gradient probes (2026-10-03).

Pins lumenairy to LUM_TREE (default this worktree), BLAS threads = 1 on the
command line, jax x64.  Outputs carry the build tag (win / wsl) and an
optional SG_TAG (e.g. ``pre`` / ``post``) so both builds and both trees dump
side by side.
"""
import json
import os
import sys
import time
import warnings

ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE", "C:/tmp/lum_symgrad")))
import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(_k) == "1", f"{_k} must be 1 on the command line"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = "wsl" if sys.platform.startswith("linux") else "win"
P, WL = 1.2, 1.0
warnings.simplefilter("ignore")


def env():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "jax": jax.__version__, "lumenairy": lumenairy.__file__,
            "platform": sys.platform, "build": BUILD}


def dump(name, obj):
    obj = dict(obj)
    obj["env"] = env()
    root, ext = os.path.splitext(name)
    tag = os.environ.get("SG_TAG")
    if tag:
        root = f"{root}_{tag}"
    path = os.path.join(HERE, f"{root}_{BUILD}{ext}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: (
            float(o) if np.ndim(o) == 0 and np.isrealobj(o) else str(o)))
    return path


def tic():
    return time.perf_counter()


def ladder(fun, x0, steps=(1e-3, 3e-4, 1e-4), scale=1.0):
    """Central-difference ladder: the rows, the Richardson value of the last
    two rungs, and the h^2 PREMISE (ratio of successive rung changes, which
    must approach (h_k / h_k+1)^2)."""
    rows = []
    for hs in steps:
        h = hs * scale
        rows.append((np.asarray(fun(x0 + h), dtype=float)
                     - np.asarray(fun(x0 - h), dtype=float)) / (2 * h))
    rows = np.asarray(rows)
    r = steps[-2] / steps[-1]
    rich = (r * r * rows[-1] - rows[-2]) / (r * r - 1.0)
    ch = np.abs(np.diff(rows, axis=0))
    ratios = ch[:-1] / np.maximum(ch[1:], 1e-300)
    return rows, rich, ratios


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def min_rel_gap(lam):
    lam = np.asarray(lam)
    s = np.max(np.abs(lam))
    d = np.abs(lam[:, None] - lam[None, :]) + np.eye(lam.size) * 1e300
    return float(np.min(d) / s)


def n_pairs_below(lam, gap):
    lam = np.asarray(lam)
    s = np.max(np.abs(lam))
    d = np.abs(lam[:, None] - lam[None, :]) <= gap * s
    np.fill_diagonal(d, False)
    return int(np.sum(np.any(d, axis=1)))
