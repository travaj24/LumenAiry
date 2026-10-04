"""Shared helpers of the round-2 verification probes (verifier, 2026-10-04).
LUMROOT selects the tree (HEAD worktree by default; the base 7c0bc8bd
worktree for PRE readings).  The import path is asserted."""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("LUMROOT") or os.path.abspath(
    os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(
    os.path.normcase(os.path.abspath(ROOT))), (lumenairy.__file__, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402,F401

BUILD = "wsl" if sys.platform.startswith("linux") else "win"


def fd_rich(fn, x0, hs=(1e-3, 3e-4, 1e-4), scale=1.0, floor=1e-3):
    """Richardson FD on (h, h/sqrt(10)... ) rungs; returns (fd, ratios) with
    ratios = |c1|/|c2| of successive rung changes on the big components
    (h^2 premise: (h1^2-h2^2)/(h2^2-h3^2))."""
    rows = [(np.asarray(fn(x0 + h * scale)) - np.asarray(fn(x0 - h * scale)))
            / (2 * h * scale) for h in hs]
    c1, c2 = np.abs(rows[0] - rows[1]), np.abs(rows[1] - rows[2])
    h1, h2, h3 = hs
    expect = (h1 ** 2 - h2 ** 2) / (h2 ** 2 - h3 ** 2)
    fd = rows[2] + (rows[2] - rows[1]) * (h3 ** 2 / (h2 ** 2 - h3 ** 2))
    big = np.abs(fd) >= floor * np.max(np.abs(fd))
    ratio = c1[big] / np.maximum(c2[big], 1e-300)
    return fd, ratio, expect


def premise_ok(ratio, expect, tol=0.12):
    return bool(np.all(np.abs(ratio / expect - 1.0) < tol))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
