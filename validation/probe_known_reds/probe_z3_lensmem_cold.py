"""Z3 lens-memory COLD vs STEADY-STATE probe.

One (N, dtype) per PROCESS, so the first measured call is genuinely the first
``apply_real_lens`` of that size in the interpreter and its peak carries the
one-time lazy initialisation.  The second and third calls are the steady state.

Usage:  python probe_z3_lensmem_cold.py <out.json> <N> <c128|c64>
"""
import gc
import json
import os
import sys
import tracemalloc
import warnings

import numpy as np

import lumenairy as la

N = int(sys.argv[2])
dt = np.complex128 if sys.argv[3] == "c128" else np.complex64

wl, dx = 633e-9, 30e-3 / N
rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                     aperture=25e-3)
E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
E = np.ascontiguousarray(E)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    la.apply_real_lens(E[:64, :64].copy(), prescription=rx,
                       wavelength=wl, dx=dx)     # what the test warms with


def one():
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del res
    gc.collect()
    return peak


peaks = [one() for _ in range(3)]
est = float(la.estimate_lens_memory(N, dt, lens_model='real'))
out = dict(python=sys.version.split()[0], numpy=np.__version__,
           platform=sys.platform, lumenairy_file=la.__file__,
           N=N, dtype=np.dtype(dt).name,
           env={k: os.environ.get(k) for k in
                ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS")},
           est_bytes=est, peaks_bytes=[float(p) for p in peaks],
           peak_cold=float(peaks[0]),
           peak_steady=float(np.median(peaks[1:])),
           ratio_est_over_cold=est / peaks[0],
           ratio_est_over_steady=est / float(np.median(peaks[1:])),
           cold_over_steady=peaks[0] / float(np.median(peaks[1:])))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{out['platform']:6s} N={N:5d} {out['dtype']:10s} "
      f"est={est / 1e6:7.1f} cold={peaks[0] / 1e6:7.1f} "
      f"steady={out['peak_steady'] / 1e6:7.1f} MB | "
      f"est/cold={out['ratio_est_over_cold']:.3f} "
      f"est/steady={out['ratio_est_over_steady']:.3f} "
      f"cold/steady={out['cold_over_steady']:.3f}")
