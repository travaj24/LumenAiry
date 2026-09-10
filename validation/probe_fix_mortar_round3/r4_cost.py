"""ROUND 3 -- what the generalized site's RESIDUAL screen costs.

Round 2's ``rcond`` screen was priced at 1.00x because ``gecon`` rides the
factors that already exist.  The round-3 residual screen must be priced the
same way or the claim does not carry.

Three arms, on operands of the site's own sizes (``n`` = 324 .. 1764):

  ``bare``   ``lu_factor`` + ``lu_solve``                       O(n^3)
  ``gecon``  + the round-2 condition estimate                   O(n^2)
  ``probe``  + the SHIPPED residual screen: three matvecs       O(n^2)
  ``full``   + the EXACT Frobenius residual ``A X - B``         O(n^3)

The box this runs on is SHARED, so every timing is the MINIMUM of many
interleaved repetitions -- a minimum is the only load-robust statistic here --
and the FLOP count is the primary evidence: ``lu_factor`` + ``lu_solve`` with
``n`` right-hand sides is ``(2/3 + 2) n^3`` complex multiply-adds, the probe is
``3 n^2``, so the screen is ``~1.7 / n`` of the call it rides -- 0.5 % at
``n`` = 324 and 0.1 % at ``n`` = 1764.

``python r4_cost.py [win|wsl]``
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(_ROOT.lower()), (
    lumenairy.__file__)
TAG = (sys.argv[1] if len(sys.argv) > 1 else "win")
REPS = int(os.environ.get("R4_REPS", "9"))
T0 = time.time()
print(f"[arm {TAG}] lumenairy = {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


OUT = {}
for n in (324, 576, 900, 1296, 1764):
    rg = np.random.default_rng(7)
    A = np.ascontiguousarray(
        rg.standard_normal((n, n)) + 1j * rg.standard_normal((n, n)))
    B = np.ascontiguousarray(
        rg.standard_normal((n, n)) + 1j * rg.standard_normal((n, n)))

    def bare():
        lu, piv = sla.lu_factor(A)
        return sla.lu_solve((lu, piv), B)

    def gecon():
        lu, piv = sla.lu_factor(A)
        an = float(np.max(np.sum(np.abs(A), axis=0)))
        sla.get_lapack_funcs("gecon", (A,))(lu, an)
        return sla.lu_solve((lu, piv), B)

    def probe():
        X = gecon()
        _pc._mortar_residual(A, X, B, probe=True)
        return X

    def full():
        X = gecon()
        _pc._mortar_residual(A, X, B, probe=False)
        return X

    arms = {"bare": bare, "gecon": gecon, "probe": probe, "full": full}
    best = {k: float("inf") for k in arms}
    for _ in range(REPS):                     # INTERLEAVED, so drift is shared
        for k, fn in arms.items():
            t = time.perf_counter()
            fn()
            best[k] = min(best[k], time.perf_counter() - t)
    OUT[n] = dict(best, x_gecon=best["gecon"] / best["bare"],
                  x_probe=best["probe"] / best["bare"],
                  x_full=best["full"] / best["bare"],
                  flops_ratio_probe=3.0 * n * n / ((2.0 / 3 + 2) * n ** 3))
    _log(f"n={n:5d}  bare {best['bare'] * 1e3:8.1f} ms   +gecon "
         f"{OUT[n]['x_gecon']:5.3f}x   +probe {OUT[n]['x_probe']:5.3f}x   "
         f"+full-GEMM {OUT[n]['x_full']:5.3f}x   (probe flop share "
         f"{OUT[n]['flops_ratio_probe']:.2e})")

p = os.path.join(_HERE, f"r4_cost_{TAG}.json")
with open(p, "w") as fh:
    json.dump({"tag": TAG, "lumenairy": lumenairy.__file__, "reps": REPS,
               "rows": {str(k): v for k, v in OUT.items()}}, fh, indent=1)
_log(f"wrote {p}")
