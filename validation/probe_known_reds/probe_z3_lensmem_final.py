"""Z3 lens-memory: the clean first-call reading after a warm-up big enough to
pay the deferred imports.  One (N, dtype) per PROCESS.
Usage: probe_z3_lensmem_final.py <out.json> <N> <c128|c64> <warmN>"""
import gc, json, os, sys, tracemalloc, warnings
import numpy as np
import lumenairy as la
N = int(sys.argv[2]); dt = np.complex128 if sys.argv[3] == "c128" else np.complex64
warmN = int(sys.argv[4])
wl, dx = 633e-9, 30e-3 / N
rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7', aperture=25e-3)
E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
E = np.ascontiguousarray(E)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    la.apply_real_lens(E[:warmN, :warmN].copy(), prescription=rx, wavelength=wl, dx=dx)
rows = []
for _ in range(3):
    gc.collect(); tracemalloc.start(); tracemalloc.reset_peak()
    res = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    del res; gc.collect(); rows.append((peak, cur))
est = float(la.estimate_lens_memory(N, dt, lens_model='real'))
cgrid = N * N * (16 if dt is np.complex128 else 8)
out = dict(platform=sys.platform, python=sys.version.split()[0],
           numpy=np.__version__, N=N, warmN=warmN, dtype=np.dtype(dt).name,
           env={k: os.environ.get(k) for k in
                ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS")},
           est_bytes=est, complex_grid_bytes=cgrid,
           peaks=[float(p) for p, _ in rows], retained=[float(c) for _, c in rows],
           ratio_est_over_first=est / rows[0][0],
           retained_first_in_complex_grids=rows[0][1] / cgrid,
           first_over_steady=rows[0][0] / float(np.median([p for p, _ in rows[1:]])))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{sys.platform:6s} N={N:5d} {out['dtype']:10s} warm={warmN} est={est/1e6:7.1f} "
      f"peak1={rows[0][0]/1e6:7.1f} retained1={rows[0][1]/1e6:6.1f} "
      f"({out['retained_first_in_complex_grids']:.2f} cgrids) "
      f"est/first={out['ratio_est_over_first']:.3f} steady={rows[1][0]/1e6:6.1f}")
