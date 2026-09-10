"""ROUND 3 item 2: the S1-2 exact-index-coincidence fixture, two-sided.

POST arm  = the shipped tree.
PRE  arm  = the pre-round-1 branch body reinstated (the EXACT ``Re(r) == 0``
            pin, which an ``eig`` output never satisfies).
CONTROL   = the same stack with the groove permittivity walked off the
            substrate's by a relative 1e-3, so the coincidence is gone.

Oracle: the structure is provably lossless, so ``sum R + sum T == 2`` exactly
at any truncation under the Laurent rule (2, not 1: two incident
polarizations).  No reference solve, no prior reading.
"""
import os, sys, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import threadpoolctl
import importlib
from lumenairy.elements.rcwa import rcwa_jones_1d_segments
import lumenairy.elements.rcwa._core as _rc
from lumenairy.elements.rcwa._core import _EnergyWarning

_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")


def _pre_body(x, xp=None, band=1e-8):
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


class _pre_arm:
    def __enter__(self):
        self._saved = []
        for n in _BOUND:
            m = importlib.import_module(n)
            if hasattr(m, "_sqrt_decay"):
                self._saved.append((m, m._sqrt_decay))
                m._sqrt_decay = _pre_body
        return self

    def __exit__(self, *a):
        for m, f in self._saved:
            m._sqrt_decay = f
        return False


th = np.deg2rad(35.0)
c, s = np.cos(th), np.sin(th)
rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
_ER = rot @ np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex) @ rot.T
_EG = np.diag([1.5 ** 2] * 3).astype(complex)          # == no^2 == n_sub^2
_EG_OFF = np.diag([(1.5 * 1.001) ** 2] * 3).astype(complex)   # detuned 1e-3


def closure(eg, n):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = rcwa_jones_1d_segments(0.7e-6, [(0.5, _ER), (0.5, eg)], 1.5, 1.0,
                                     0.5e-6, 1.0e-6, angle=0.0, n_orders=n)
    warned = any(isinstance(w.message, _EnergyWarning) for w in rec)
    return abs(float(np.sum(out[1]) + np.sum(out[2])) - 2.0), warned


LADDER = list(range(11, 42, 2))
arch = threadpoolctl.threadpool_info()[0].get("architecture", "?")
print(f"# py{sys.version.split()[0]} np{np.__version__} arch={arch} "
      f"CORETYPE={os.environ.get('OPENBLAS_CORETYPE', '-')}")

post_c = [closure(_EG, n) for n in LADDER]
post_d = [closure(_EG_OFF, n) for n in LADDER]
with _pre_arm():
    pre_c = [closure(_EG, n) for n in LADDER]
    pre_d = [closure(_EG_OFF, n) for n in LADDER]

for nm, rows in (("POST coincident", post_c), ("POST detuned", post_d),
                 ("PRE  coincident", pre_c), ("PRE  detuned", pre_d)):
    v = [r[0] for r in rows]
    w = sum(r[1] for r in rows)
    print(f"{nm:16s} worst {max(v):.6e}  median {sorted(v)[len(v)//2]:.6e}  "
          f"warned {w}/{len(v)}")
print(f"n_orders ladder {LADDER[0]}..{LADDER[-1]} step 2")
print("PRE coincident per-rung: "
      + " ".join(f"{n}:{r[0]:.2e}" for n, r in zip(LADDER, pre_c)))
