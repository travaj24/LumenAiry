"""Gate 2b -- the SAME entry point (PMM2DStackPure) with a scalar cell vs its
e*I promotion, ON a layer cut-off.  argv[1] = required lumenairy root."""
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
_HERE = os.path.abspath(lumenairy.__file__).replace("\\", "/").lower()
assert _HERE.startswith(_ROOT), (_HERE, _ROOT)

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402


def promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def run(cell, *, px, wl, theta=0.0, phi=0.0, uniform=None):
    s = PMM2DStackPure(px, px, n_superstrate=1.0, n_substrate=1.5,
                       n_modes=5, n_orders=3)
    if uniform is not None:
        s.add_layer(0.15e-6, eps=uniform)
    s.add_layer(0.28e-6, eps_cell=cell)
    s.set_source(wl, theta=theta, phi=phi)
    o, R, T, J = s.solve(jones=True)
    return R, T


CASES = {
    "patterned {4,1} ON eps=4 layer cutoff":
        dict(cell=np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex),
             px=0.5e-6, wl=1.0e-6),
    "patterned {4,1} 1e-9 off it":
        dict(cell=np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex),
             px=0.5e-6, wl=1.0e-6 * (1 - 1e-9)),
    "patterned {4,1} far off, oblique conical":
        dict(cell=np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex),
             px=0.7e-6, wl=0.55e-6, theta=0.2, phi=0.5),
    "UNIFORM scalar layer eps=2.25 on ITS cutoff (kt=1.5)":
        dict(cell=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex),
             px=1.0e-6, wl=1.5e-6, uniform=2.25),
}
for name, kw in CASES.items():
    u = kw.pop("uniform", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Rs, Ts = run(uniform=u, **kw)
        Rt, Tt = run(uniform=(promote(np.array([[u]]))[0, 0]
                              if u is not None else None),
                     cell=promote(kw.pop("cell")), **kw)
    print(f"{name}:\n    max|dR| = {np.max(np.abs(Rs - Rt)):.3e}   "
          f"max|dT| = {np.max(np.abs(Ts - Tt)):.3e}   bit-identical = "
          f"{np.array_equal(Rs, Rt) and np.array_equal(Ts, Tt)}")
