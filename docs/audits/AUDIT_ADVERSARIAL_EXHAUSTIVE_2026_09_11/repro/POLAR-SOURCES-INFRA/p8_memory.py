import sys, tracemalloc, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la

wl = 633e-9
rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7', aperture=25e-3)
print("prescription keys:", sorted(rx)[:12])

for N in (1024, 2048):
    dx = 30e-3 / N
    E, x, y = la.create_gaussian_beam(N, dx, wl, w0=5e-3)
    E = np.ascontiguousarray(E)
    # warm up caches so the measured peak is the call's own working set
    la.apply_real_lens(E[:64, :64].copy(), prescription=rx, wavelength=wl, dx=dx)
    est = la.estimate_lens_memory(N, 'complex128', lens_model='real',
                                  parallel_amp=False, itemized=True)
    est_par = la.estimate_lens_memory(N, 'complex128', lens_model='real',
                                      parallel_amp=True)
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    base = N * N * 16
    print("N=%d  field=%.1f MB" % (N, base / 1e6))
    print("   tracemalloc peak            = %10.1f MB" % (peak / 1e6))
    print("   estimate_lens_memory(real, parallel_amp=False) = %10.1f MB  ratio est/meas = %.2f"
          % (est['total'] / 1e6, est['total'] / peak))
    print("   estimate_lens_memory(real, parallel_amp=True ) = %10.1f MB  ratio est/meas = %.2f"
          % (est_par / 1e6, est_par / peak))
    print("   itemized:", {k: round(v / 1e6, 1) for k, v in est['items'].items()})
    est_tr = la.estimate_lens_memory(N, 'complex128', lens_model='traced',
                                     parallel_amp=True)
    print("   estimate_lens_memory(traced default) = %10.1f MB  ratio vs measured(real) = %.2f"
          % (est_tr / 1e6, est_tr / peak))
    del out, E

print("")
print("=== estimate_asm_memory vs measured ASM peak ===")
for N in (1024, 2048):
    dx = 30e-3 / N
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3)
    la.angular_spectrum_propagate(E, 1e-3, wl, dx, dx)   # warm
    la.clear_asm_caches()
    tracemalloc.start(); tracemalloc.reset_peak()
    la.angular_spectrum_propagate(E, 2e-3, wl, dx, dx)
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    est = la.estimate_asm_memory(N, 'complex128')
    print("  N=%d measured peak %.1f MB, estimate %.1f MB, ratio %.2f"
          % (N, peak / 1e6, est / 1e6, est / peak))
