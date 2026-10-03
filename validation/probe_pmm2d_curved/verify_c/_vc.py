"""Shared helpers of the Phase C VERIFIER probes (independent of build_c/).

Every probe asserts that ``lumenairy`` is imported from the tree it is meant
to measure: this worktree, or the tree named by ``LUM_TREE`` (the PRE tree =
``git archive 91d00288`` extracted to ``C:/tmp/vcc_pre``).  Run with BLAS
pinned on the command line:

  cd /c/tmp/lum_vcurved_c && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_vcurved_c \
    python validation/probe_pmm2d_curved/verify_c/<probe>.py ...
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


def env_record():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "tree": TREE, "build": BUILD, "machine": platform.node(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}


def dump(name, obj):
    obj = dict(obj)
    obj["env"] = env_record()
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: (
            float(v) if np.isscalar(v) else np.asarray(v).tolist()))
    print("wrote", name)
