"""Does a SMALLER full-size warm-up pay the deferred imports without warming
the N-sized caches?  Usage: probe_z3_lensmem_warmup.py <out.json> <N> <c128|c64> <warmN>"""
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
def one():
    gc.collect(); tracemalloc.start(); tracemalloc.reset_peak()
    res = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    del res; gc.collect(); return peak
peaks = [one() for _ in range(3)]
est = float(la.estimate_lens_memory(N, dt, lens_model='real'))
out = dict(platform=sys.platform, python=sys.version.split()[0], N=N, warmN=warmN,
           dtype=np.dtype(dt).name, est_bytes=est,
           peaks_bytes=[float(p) for p in peaks],
           ratio_est_over_first=est / peaks[0],
           ratio_est_over_steady=est / float(np.median(peaks[1:])))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"{sys.platform:6s} N={N} warmN={warmN} {out['dtype']:10s} est={est/1e6:7.1f} "
      f"peaks={[round(p/1e6,1) for p in peaks]} est/first={out['ratio_est_over_first']:.3f} "
      f"est/steady={out['ratio_est_over_steady']:.3f}")
