"""H4 -- conditioning of the anisotropic 1-D layer eigenproblem vs the
closure defect.  argv[1] = required lumenairy root."""
import os
import sys
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
assert os.path.abspath(lumenairy.__file__).replace("\\", "/").lower().startswith(_ROOT)

from lumenairy.elements.rcwa._core import (  # noqa: E402
    _layer_eigenmodes_tensor,
    _tensor_convolutions,
)
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments  # noqa: E402

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


ER = _rot(np.deg2rad(35.0), 1.5, 2.3)
EG = np.diag([2.25] * 3).astype(complex)
SEG = [(0.5, ER), (0.5, EG)]
EDGES = (0.0, 0.5, 1.0)
COMP = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1), "zz": (2, 2)}


def internals(n):
    # PUBLIC -> INTERNAL conjugation, as rcwa_jones_1d_segments does
    ts = [np.conj(ER), np.conj(EG)]
    profiles = {k: np.stack([t[i, j] for t in ts]) for k, (i, j) in COMP.items()}
    Cxx, Cxy, Cyx, Cyy, EZZ = _tensor_convolutions(profiles, n, "li", EDGES)
    orders = np.arange(-n, n + 1)
    kx = orders * (WL / PX)
    Kx = np.diag(kx.astype(complex))
    Ky = np.zeros_like(Kx)
    C = np.block([[Cxx, Cxy], [Cyx, Cyy]])
    herm = float(np.max(np.abs(C - C.conj().T))) / float(np.max(np.abs(C)))
    W, V, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    Mmode = np.block([[W, W], [V, -V]])
    lam = np.asarray(lam).ravel()
    order = np.argsort(np.abs(lam))
    ls = lam[order]
    gaps = np.abs(np.diff(ls))
    return (herm, np.linalg.cond(W), np.linalg.cond(Mmode),
            float(np.min(gaps) / max(np.max(np.abs(lam)), 1e-300)))


print(f"numpy {np.__version__} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"{'n':>4} {'defect':>11} {'C non-herm':>11} {'cond(W)':>11} "
      f"{'cond[W;V]':>11} {'min rel eig gap':>16} {'cond*eps':>10}")
for n in (5, 7, 9, 11, 13, 19, 21, 23, 31, 41, 45, 53, 61):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1, _J = rcwa_jones_1d_segments(
            PX, SEG, 1.5, 1.0, DEPTH, WL, theta=0.0, n_orders=n)
        d = float(np.sum(R1) + np.sum(T1) - 2.0)
        herm, cW, cM, gap = internals(n)
    print(f"{n:>4} {d:>11.2e} {herm:>11.2e} {cW:>11.3e} {cM:>11.3e} "
          f"{gap:>16.3e} {cM * 2.2e-16:>10.2e}")
