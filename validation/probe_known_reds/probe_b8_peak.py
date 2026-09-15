"""B8 apply_jones_matrix peak-array probe, and the two ELISION premises the
a11/b8 peak readings depend on.

NumPy's ``temp_elide.c`` rewrites a binary op whose operand is an unreferenced
temporary into that temporary's own buffer.  It is compiled in only where
``HAVE_BACKTRACE`` holds and it verifies, by walking the C stack, that the
temporary was created by the interpreter.  Both conditions are BUILD
properties, so the "how many full grids does this expression peak at" reading
is a build quantity and has to be measured, not assumed.

Two distinct patterns are probed because CI shows them disagreeing:
  * ``Ex * conj(Ey)``      -- a11 z3's pre-fix Stokes cross term,
  * ``a*X + b*Y``          -- b8's pre-fix Jones product (BOTH operands of the
                              ``+`` are temporaries).

Usage:  python probe_b8_peak.py <out.json> [N] [reps]
"""
import gc
import json
import os
import sys
import tracemalloc

import numpy as np

import lumenairy
from lumenairy.elements.polarization import JonesField, apply_jones_matrix

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
WL = 633e-9


def _peak(fn, reps=3):
    peaks = []
    for _ in range(reps):
        gc.collect()
        tracemalloc.start()
        out = fn()
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        del out
    return peaks


def _pre_fix_jones(J, Ex, Ey):
    return (J[0, 0] * Ex + J[0, 1] * Ey,
            J[1, 0] * Ex + J[1, 1] * Ey)


grid = N * N * 16
rng = np.random.default_rng(1)
Ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
Ey = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
J = np.array([[0.3 + 0.4j, -0.5 + 0.1j], [0.2 - 0.7j, 0.9 + 0.05j]],
             dtype=complex)

out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "platform": sys.platform, "N": N, "grid_bytes": grid,
       "lumenairy_file": lumenairy.__file__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS")}}

old_runs, new_runs = [], []
for _ in range(REPS):
    old_runs += [p / grid for p in _peak(lambda: _pre_fix_jones(J, Ex, Ey), 3)]
    new_runs += [p / grid for p in
                 _peak(lambda: apply_jones_matrix(
                     JonesField(Ex, Ey, 1e-6, WL), J), 3)]
out["old_grids"] = dict(all=sorted(set(old_runs)), min=min(old_runs),
                        max=max(old_runs),
                        median=float(np.median(old_runs)))
out["new_grids"] = dict(all=sorted(set(new_runs)), min=min(new_runs),
                        max=max(new_runs),
                        median=float(np.median(new_runs)))

# ---- premise A: the a11 pattern, Ex * conj(Ey) -----------------------------
a = np.median(_peak(lambda: Ex * np.conj(Ey), 3)) / grid


def _bound_conj():
    c = np.conj(Ey)                       # named -> refcount 2 -> no elision
    return Ex * c


b = np.median(_peak(_bound_conj, 3)) / grid

# ---- premise B: the b8 pattern, a*X + b*Y (both operands temporaries) ------
c_ = np.median(_peak(lambda: J[0, 0] * Ex + J[0, 1] * Ey, 3)) / grid


def _bound_add():
    t1 = J[0, 0] * Ex
    t2 = J[0, 1] * Ey
    return t1 + t2                        # both named -> no elision


d_ = np.median(_peak(_bound_add, 3)) / grid

out["elision"] = {
    "conj_pattern_grids": float(a), "conj_pattern_named_grids": float(b),
    "conj_elided": bool(a < b - 0.5),
    "add_pattern_grids": float(c_), "add_pattern_named_grids": float(d_),
    "add_elided": bool(c_ < d_ - 0.5),
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: out[k] for k in ("python", "numpy", "platform",
                                      "old_grids", "new_grids", "elision")},
                 indent=1))
