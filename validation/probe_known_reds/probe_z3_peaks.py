"""Z3 peak-array probe.

Measures, per arm:
  * the four tracemalloc peaks the test reads, REPEATED (default 5x), in raw
    BYTES as well as in units of a full-grid real array, so the spread and the
    fixed bookkeeping offset are both visible;
  * whether NumPy's TEMPORARY ELISION is active on this build -- the
    ``temp_elide.c`` optimisation that rewrites ``Ex * np.conj(Ey)`` into the
    ``conj`` temporary's own buffer.  It is compiled in only where
    ``HAVE_BACKTRACE`` holds (glibc), so it is ON on Linux and OFF on Windows.
    Measured directly: the peak of ``Ex * np.conj(Ey)`` is 4 full-grid REAL
    arrays without elision (conj temp + product) and 2 with it.

Usage:  python probe_z3_peaks.py <out.json> [N] [repeats]
"""
import gc
import json
import os
import sys
import tracemalloc

import numpy as np

import lumenairy
from lumenairy.elements.polarization import (JonesField,
                                             degree_of_polarization,
                                             stokes_parameters)

assert "lum_reds" in lumenairy.__file__, lumenairy.__file__

N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def _old_stokes(field):
    Ex, Ey = field.Ex, field.Ey
    return {'S0': np.abs(Ex) ** 2 + np.abs(Ey) ** 2,
            'S1': np.abs(Ex) ** 2 - np.abs(Ey) ** 2,
            'S2': 2 * np.real(Ex * np.conj(Ey)),
            'S3': -2 * np.imag(Ex * np.conj(Ey))}


def _old_dop(field):
    S = _old_stokes(field)
    S0 = S['S0']
    finite = np.isfinite(S0)
    s0_max = float(S0[finite].max()) if finite.any() else 0.0
    eps = float(np.finfo(S0.dtype).eps
                if np.issubdtype(S0.dtype, np.floating) else np.finfo(float).eps)
    live = S0 > s0_max * eps * eps
    safe = np.where(live, S0, 1.0)
    with np.errstate(invalid='ignore'):
        dop = np.sqrt((S['S1'] / safe) ** 2 + (S['S2'] / safe) ** 2
                      + (S['S3'] / safe) ** 2)
    dop = np.clip(np.where(live, dop, 0.0), 0.0, 1.0)
    return np.where(np.isnan(S0) | np.isnan(dop), np.nan, dop)


def _pathological_field(N, dtype):
    rng = np.random.default_rng(7)
    Ex = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dtype)
    Ey = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dtype)
    Ex[0, 0] = Ey[0, 0] = 0.0
    Ex[0, 1] = np.nan
    Ex[1, 0] = np.inf
    Ex[2, 0] = Ey[2, 0] = 1e-160
    return JonesField(Ex, Ey, 1e-6, 1e-6)


def peak_of(fn, *a):
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = fn(*a)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del res
    gc.collect()
    return peak


unit = N * N * 8
f = _pathological_field(N, np.complex128)
f = JonesField(np.nan_to_num(f.Ex, posinf=3.0),
               np.nan_to_num(f.Ey, posinf=3.0), 1e-6, 1e-6)

out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "platform": sys.platform, "N": N, "unit_bytes": unit,
       "lumenairy_file": lumenairy.__file__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS")},
       "runs": []}

fns = (('stokes_new', stokes_parameters), ('stokes_old', _old_stokes),
       ('dop_new', degree_of_polarization), ('dop_old', _old_dop))
for _ in range(REP):
    with np.errstate(invalid='ignore'):
        out["runs"].append({tag: peak_of(fn, f) for tag, fn in fns})

out["bytes"] = {tag: sorted({r[tag] for r in out["runs"]})
                for tag, _ in fns}
out["units"] = {tag: sorted({r[tag] / unit for r in out["runs"]})
                for tag, _ in fns}
out["overhead_bytes"] = {tag: sorted({r[tag] - unit * round(r[tag] / unit * 4) / 4
                                      for r in out["runs"]}) for tag, _ in fns}

# ---- the premise: is NumPy's temporary elision active on this build? ----
Ex, Ey = f.Ex, f.Ey


def _cross_expr():
    return Ex * np.conj(Ey)          # the elidable pattern


def _cross_expr_noelide():
    c = np.conj(Ey)                  # bound name -> refcount 2 -> no elision
    return Ex * c


p_elidable = peak_of(_cross_expr)
p_forced = peak_of(_cross_expr_noelide)
out["elision"] = {
    "peak_units_elidable_pattern": p_elidable / unit,
    "peak_units_name_bound_pattern": p_forced / unit,
    "active": bool(p_elidable < p_forced - 0.5 * unit),
}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: out[k] for k in
                  ("python", "numpy", "platform", "units", "overhead_bytes",
                   "elision")}, indent=1))
