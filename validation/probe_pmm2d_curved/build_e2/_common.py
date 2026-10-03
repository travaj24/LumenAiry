"""Shared fixtures of the Phase-E2 build probes (a different curved map in
every layer, joined by the non-separable curved mortar).

Every probe drives the LIBRARY and asserts that ``lumenairy`` is imported from
the tree it is meant to measure: the worktree that holds this file, or -- for
the BEFORE arm of a before/after measurement -- the tree named by the
environment variable ``LUM_TREE`` (the PRE tree is ``git archive eae470d9``
extracted to ``C:/tmp/curved_pre_e2``).  Run every probe with BLAS pinned on
the command line:

  cd /c/tmp/lum_curved_e2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e2 \
    python validation/probe_pmm2d_curved/build_e2/<probe>.py ...

Fixture (the planning probes' fixture, as Phases B-D): lambda = 1, square
period 1.2, air above, n = 1.45 below, eps 4 (n = 2) circle of r = 0.36 in
layer 1 (depth 0.3); layer 2 (depth 0.25) a sinusoidal interface
x = x0 + A sin(2 pi y / p) between eps 2.25 (right) and air (left).  The
OVERLAPPING arm has x0 = 0.6, A = 0.12 (the wall runs through the disk, so
the two outlines CROSS in plan view and Phase C's stack-wide merge refuses);
the NON-overlapping arm has x0 = 0.12, A = 0.05 (the wall stays left of the
disk).  Row 0 of R / T is the input E along x, row 1 E along y.
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
N_SUP, N_SUB = 1.0, 1.45
R_CIRC = 0.36
EPS_P = 4.0
EPS_W = 2.25
D1, D2 = 0.3, 0.25
TREE = "pre" if os.environ.get("LUM_TREE") else "post"


def env():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, tree=ROOT,
                lumenairy=os.path.dirname(lumenairy.__file__),
                platform=platform.platform())


def dump(name, payload):
    payload = dict(payload)
    payload.setdefault("env", env())
    path = os.path.join(HERE, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else
            complex(o).__repr__() if isinstance(o, complex) else str(o)))
    return path


def shapes_stack(layers, M, *, per_layer=True, n_orders=3, sub=N_SUB,
                 sup=N_SUP):
    """A stack of ``layers`` = [(thickness, shapes or None, background or
    eps), ...] (``None`` shapes -> a uniform layer of eps ``background``)."""
    kw = dict(n_superstrate=sup, n_substrate=sub, n_modes=M,
              n_orders=n_orders)
    if per_layer:
        kw["layer_grids"] = "per-layer"
    st = PMM2DStackPure(P, P, **kw)
    for t, shp, bg in layers:
        if shp is None:
            st.add_layer(t, eps=bg)
        else:
            st.add_layer(t, shapes=shp, background_eps=bg)
    return st


def solve(st, theta=0.0, phi=0.0, retain=False):
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(retain_internal=retain)
    return np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)


def jones_block(st, k):
    """Power-normalised 2 x 2 reflection Jones block of order index ``k``
    from the stack's retained per-order amplitudes (the Phase C reciprocity
    instrument), or None when the order does not propagate."""
    r = st._modal
    kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
    A = np.array([[r["rx"][c][k] for c in (0, 1)],
                  [r["ry"][c][k] for c in (0, 1)]])
    kxo, kyo = r["kx"][k], r["ky"][k]
    kzo = complex(r["kz_ref"][k])
    if abs(kzo.imag) > 1e-12 or kzo.real <= 0:
        return None
    kzo = kzo.real
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        d = w ** (-0.5 if inv else 0.5)
        return (V * d) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def reverse_angles(theta, phi, m, n):
    """(theta, phi) in radians of the reversed channel of reflection order
    (m, n) at incidence (theta, phi) in air."""
    st_ = np.sin(theta)
    kx = st_ * np.cos(phi) + m * WL / P
    ky = st_ * np.sin(phi) + n * WL / P
    kx, ky = -kx, -ky
    return float(np.arcsin(np.hypot(kx, ky))), float(np.arctan2(ky, kx))


def order_index(o, mn):
    return int(np.nonzero((o[:, 0] == mn[0]) & (o[:, 1] == mn[1]))[0][0])
