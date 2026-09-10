"""Shared fixtures + the worktree assert for the Phase-D SLANT build probes.

Every script here MEASURES the SHIPPED library (``lumenairy.elements.pmm``),
never a prototype: the build transplanted the prototype's formulation, so the
numbers a test bar is derived from must come out of the code that ships.

Build doc: ``docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md``.
Formulation + the GO decision: ``docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md``.
"""
from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lumenairy  # noqa: E402

if not os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()):
    raise RuntimeError(
        f"probe_pmm2d_staggered_slant_build: lumenairy resolved to "
        f"{lumenairy.__file__!r}, outside the worktree {_ROOT!r}.")

RESULTS = os.path.join(_HERE, "results")


def build_pin():
    import platform

    import scipy
    return {
        "lumenairy": os.path.abspath(lumenairy.__file__),
        "version": getattr(lumenairy, "__version__", "?"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "tag": os.environ.get("SLANT_BUILD_TAG", "win"),
    }


def write(name, payload):
    payload["build"] = build_pin()
    os.makedirs(RESULTS, exist_ok=True)
    tag = payload["build"]["tag"]
    path = os.path.join(RESULTS, f"{name}_{tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=float)
    print(f"WROTE {os.path.relpath(path, _ROOT)}")


def uniaxial(no, ne, tilt, azim):
    from lumenairy.elements.rcwa._core import uniaxial_tensor
    return uniaxial_tensor(no, ne, tilt, phi=azim)


def tile(t33, n):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t33
    return c
