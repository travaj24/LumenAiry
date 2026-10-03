"""Shared harness of the Phase E3 VERIFIER probes (independent of build_e3/).

Pins: lumenairy from LUM_TREE (default the verifier worktree), BLAS threads
= 1 on the command line, jax x64.  Outputs are suffixed with the build tag
(win / wsl) so both builds dump side by side.
"""
import json
import os
import sys
import time
import warnings

ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE", "C:/tmp/lum_vcurved_e3")))
import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(_k) == "1", f"{_k} must be 1 on the command line"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import (
    PMM2DStackPure,  # noqa: E402,F401
    twod_staggered as TS,  # noqa: E402,F401
)

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
    tag = os.environ.get("VE3_TAG")
    if tag:
        root = f"{root}_{tag}"
        obj["tree"] = ROOT
    path = os.path.join(HERE, f"{root}_{BUILD}{ext}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: (
            float(o) if np.ndim(o) == 0 and np.isrealobj(o) else str(o)))
    return path


def tic():
    return time.perf_counter()


def stack(M, layers, *, n_orders=2, theta=0.0, phi=0.0, backend="numpy",
          n_sup=1.0, n_sub=1.45, **kw):
    st = PMM2DStackPure(P, P, n_superstrate=n_sup, n_substrate=n_sub,
                        n_modes=M, n_orders=n_orders, backend=backend, **kw)
    for L in layers:
        st.add_layer(**L)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def ladder(fun, x0, steps, scale=1.0):
    """Central-difference ladder; returns rows, the Richardson value of the
    last two rungs (h ratio r, h^2 law), the rung changes and the ratios of
    successive changes (the h^2 PREMISE: ratio -> r^2)."""
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
    return rows, rich, ch, ratios


def amax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def lc(phi, no=1.5, ne=1.8):
    """In-plane uniaxial tensor, director at angle phi from x (traceable)."""
    xp = jnp if isinstance(phi, jax.Array) or hasattr(phi, "aval") else np
    d = xp.stack([xp.cos(phi), xp.sin(phi), 0.0 * phi])
    return (no ** 2 * xp.eye(3) + (ne ** 2 - no ** 2) * xp.outer(d, d)
            ).astype(complex)
