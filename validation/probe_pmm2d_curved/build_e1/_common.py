"""Shared fixtures of the Phase-E1 build probes (out-of-plane tensors and
slanted walls inside curved cells; a copy of build_d/_common.py).

Every probe drives the LIBRARY and asserts that ``lumenairy`` is imported from
the tree it is meant to measure: the worktree that holds this file, or -- for
the BEFORE arm of a before/after measurement -- the tree named by the
environment variable ``LUM_TREE`` (the PRE tree is ``git archive eae470d9``
extracted to ``C:/tmp/curved_pre_e1``).  Run every probe with BLAS pinned on
the command line:

  cd /c/tmp/lum_curved_e1 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e1 \
    python validation/probe_pmm2d_curved/build_e1/<probe>.py ...

Fixture (the planning probes' P3 / P4 fixture, as Phases B and C): lambda = 1,
square period 1.2, depth 0.5, air above, n = 1.45 below, eps 4 (n = 2)
features; the circle has r = 0.36; the fillet pillar side 0.6.  Row 0 of
R / T is the input E along x, row 1 E along y.
"""
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normcase(os.path.abspath(
    os.environ.get("LUM_TREE") or os.path.join(HERE, "..", "..", "..")))
if ROOT not in [os.path.normcase(os.path.abspath(p)) for p in sys.path]:
    sys.path.insert(0, ROOT)
import lumenairy  # noqa: E402

assert os.path.normcase(os.path.abspath(lumenairy.__file__)).startswith(ROOT), (
    f"lumenairy imported from {lumenairy.__file__}, not {ROOT}")

from lumenairy.elements.pmm import (  # noqa: E402,F401 -- re-exported
    PMM2DStackPure,
    _curvemap as CM,
    stack2d_pure as SP,
    twod_staggered as TS,
)

P = 1.2
WL = 1.0
DEPTH = 0.5
N_SUP, N_SUB = 1.0, 1.45
R_CIRC = 0.36
EPS_P = 4.0
ORD9 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1),
        (-1, -1)]
TREE = "pre" if os.environ.get("LUM_TREE") else "post"


def circle3(center=None):
    cm, _w = CM._circle_map_3x3(P, R_CIRC, center=center)
    eps = np.ones((3, 3), complex)
    eps[1, 1] = EPS_P
    return cm, eps


def solve(cmap, eps_cell, M, n_orders=3, theta=0.0, phi=0.0, spacer=None):
    """Single patterned layer through PMM2DStackPure under ``cmap``;
    ``spacer`` = thickness of a uniform VACUUM layer on top (a physical
    no-op).  Returns (orders, R, T, J)."""
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    if spacer:
        st.add_layer(float(spacer), eps=1.0)
    st.add_layer(DEPTH, eps_cell=eps_cell)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve()
    return np.asarray(o), np.asarray(R), np.asarray(T), J


def idx(o, orders=ORD9):
    return [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
            for m, n in orders]


def vec(o, R, T):
    i = idx(o)
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def env_record():
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "lumenairy": lumenairy.__file__,
            "tree": TREE, "machine": platform.node(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS")}}


def _jsonable(x):
    if isinstance(x, (complex, np.complexfloating)):
        return [float(x.real), float(x.imag)]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if not np.iscomplexobj(x) else {
            "re": x.real.tolist(), "im": x.imag.tolist()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return float(x)


def dump(name, obj):
    """Write ``obj`` plus the build record to ``name`` (a ``.json`` file next
    to this module)."""
    obj = dict(obj)
    obj.setdefault("env", env_record())
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)

