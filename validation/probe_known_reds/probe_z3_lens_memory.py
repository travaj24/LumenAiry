"""Z3 lens-memory probe (NOT one of the four briefed items -- this red was
found by running the a11 file on the kernel ladder).

Replicates the body of
``test_audit2609_a11_polar_sources_infra.py::test_z3_estimate_lens_memory_real_bounds_apply_real_lens``
outside pytest so it can be pointed at a DIFFERENT lumenairy tree via
PYTHONPATH (``git archive HEAD lumenairy`` into a scratch dir), which is what
separates "another agent's in-flight library edit moved the peak" from "this
bar never held on this platform".

Usage:  python probe_z3_lens_memory.py <out.json> [N] [reps]
"""
import gc
import json
import os
import sys
import tracemalloc
import warnings

import numpy as np

import lumenairy as la

N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

out = {"python": sys.version.split()[0], "numpy": np.__version__,
       "platform": sys.platform, "lumenairy_file": la.__file__,
       "env": {k: os.environ.get(k) for k in
               ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS")},
       "rows": []}

for dt in (np.complex128, np.complex64):
    for n in (N, 2 * N):
        wl, dx = 633e-9, 30e-3 / n
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                             aperture=25e-3)
        E, _, _ = la.create_gaussian_beam(n, dx, wl, w0=5e-3, dtype=dt)
        E = np.ascontiguousarray(E)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens(E[:64, :64].copy(), prescription=rx,
                               wavelength=wl, dx=dx)          # warm the caches
        peaks = []
        for _ in range(REPS):
            gc.collect()
            tracemalloc.start()
            tracemalloc.reset_peak()
            res = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del res
            gc.collect()
            peaks.append(peak)
        est = la.estimate_lens_memory(n, dt, lens_model='real')
        unit = n * n * 8
        out["rows"].append(dict(
            N=n, dtype=np.dtype(dt).name, est_bytes=float(est),
            peaks_bytes=[float(p) for p in peaks],
            peak_median=float(np.median(peaks)),
            ratio_est_over_peak=float(est / np.median(peaks)),
            peak_in_f64_grids=float(np.median(peaks) / unit),
            est_in_f64_grids=float(est / unit)))

json.dump(out, open(sys.argv[1], "w"), indent=1)
print(la.__file__)
for r in out["rows"]:
    print(f"N={r['N']:5d} {r['dtype']:10s} est={r['est_bytes']/1e6:7.1f} MB "
          f"peak={r['peak_median']/1e6:7.1f} MB ratio={r['ratio_est_over_peak']:.3f} "
          f"peak_grids(f64)={r['peak_in_f64_grids']:.2f} "
          f"est_grids(f64)={r['est_in_f64_grids']:.2f}")
