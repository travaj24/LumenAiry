"""ROUND 3 item 2: what the S1-2 exact-index-coincidence fixture DOES now."""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import threadpoolctl
from lumenairy.elements.rcwa import rcwa_jones_1d_segments
from lumenairy.elements.rcwa._core import _EnergyWarning

arch = threadpoolctl.threadpool_info()[0].get("architecture", "?")
print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
      f"BLAS arch {arch}  CORETYPE={os.environ.get('OPENBLAS_CORETYPE','-')}")

th = np.deg2rad(35.0)
c, s = np.cos(th), np.sin(th)
rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
er = rot @ np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex) @ rot.T
eg = np.diag([1.5 ** 2] * 3).astype(complex)      # == no^2 == n_substrate^2
eg_det = np.diag([(1.5 * (1 + 1e-6)) ** 2] * 3).astype(complex)   # DETUNED

def closure(segs, n):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = rcwa_jones_1d_segments(0.7e-6, segs, 1.5, 1.0, 0.5e-6, 1.0e-6,
                                     angle=0.0, n_orders=n)
    warned = any(isinstance(w.message, _EnergyWarning) for w in rec)
    R, T = out[1], out[2]
    return float(np.sum(R) + np.sum(T)), warned

print(f"{'n_orders':>9} {'|R+T-2| COINCIDENT':>20} {'warn':>5} "
      f"{'|R+T-2| DETUNED':>18} {'warn':>5}")
worst_c = worst_d = 0.0
nwarn = 0
for n in range(11, 42, 2):
    a, wa = closure([(0.5, er), (0.5, eg)], n)
    b, wb = closure([(0.5, er), (0.5, eg_det)], n)
    ea, eb = abs(a - 2.0), abs(b - 2.0)
    worst_c = max(worst_c, ea); worst_d = max(worst_d, eb)
    nwarn += int(wa)
    print(f"{n:>9} {ea:>20.6e} {str(wa):>5} {eb:>18.6e} {str(wb):>5}")
print(f"WORST coincident {worst_c:.6e}   WORST detuned {worst_d:.6e}   "
      f"warnings fired {nwarn}/16")
