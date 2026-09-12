"""Probe 10: profile + memory of the carrier step and the carrier phase build."""
import sys, time, tracemalloc, cProfile, pstats, io, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.carrier as C
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    _radial_carrier_phase)

wl = 1.31e-6; k = 2*np.pi/wl
for N in (2048,):
    dx = 2e-6
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy')
    env = np.exp(-(X**2+Y**2)/(600e-6)**2).astype(np.complex128)
    R = 0.05; z = 5e-3
    # warm up FFT plans
    propagate_carrier_referenced(env, R, z, wl, dx)
    t0 = time.perf_counter()
    for _ in range(5):
        propagate_carrier_referenced(env, R, z, wl, dx)
    t1 = time.perf_counter()
    print(f"N={N}: propagate_carrier_referenced (gap_kernel=auto) {1e3*(t1-t0)/5:.2f} ms/call")
    t0 = time.perf_counter()
    for _ in range(5):
        propagate_carrier_referenced(env, R, z, wl, dx, gap_kernel='fresnel')
    t1 = time.perf_counter()
    print(f"        gap_kernel='fresnel' {1e3*(t1-t0)/5:.2f} ms/call")
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    out = propagate_carrier_referenced(env, R, z, wl, dx)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    g = 16.0*N*N
    print(f"        tracemalloc peak {peak/1e6:.1f} MB = {(peak-base)/g:.2f} complex128 full grids "
          f"(one grid = {g/1e6:.1f} MB)")

    pr = cProfile.Profile(); pr.enable()
    for _ in range(3):
        propagate_carrier_referenced(env, R, z, wl, dx)
    pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(12)
    print('\n'.join(s.getvalue().splitlines()[4:22]))

print("\n=== carrier phase build: whole-grid vs separable outer product ===")
for N in (2048, 4096):
    dx = 2e-6; R = 0.05
    t0 = time.perf_counter()
    for _ in range(3):
        ph = _radial_carrier_phase((N, N), dx, dx, wl, R, +1)
    t1 = time.perf_counter()
    xx = (np.arange(N, dtype=np.float64) - N/2)*dx
    a = k/(2.0*R)
    t2 = time.perf_counter()
    for _ in range(3):
        px = np.exp(1j*a*xx*xx); py = np.exp(1j*a*xx*xx)
        ph2 = px[None, :]*py[:, None]
    t3 = time.perf_counter()
    err = np.abs(ph-ph2).max()
    print(f"N={N}: whole-grid {1e3*(t1-t0)/3:8.2f} ms   separable {1e3*(t3-t2)/3:8.2f} ms "
          f"speedup {(t1-t0)/(t3-t2):.2f}x   max|diff|={err:.3e}")
    # memory
    tracemalloc.start(); b = tracemalloc.get_traced_memory()[0]
    ph = _radial_carrier_phase((N, N), dx, dx, wl, R, +1)
    c, p1 = tracemalloc.get_traced_memory(); tracemalloc.stop()
    tracemalloc.start(); b2 = tracemalloc.get_traced_memory()[0]
    px = np.exp(1j*a*xx*xx); ph2 = px[None, :]*px[:, None]
    c2, p2 = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"        peak: whole-grid {(p1-b)/1e6:8.1f} MB ({(p1-b)/(16.0*N*N):.2f} grids)  "
          f"separable {(p2-b2)/1e6:8.1f} MB ({(p2-b2)/(16.0*N*N):.2f} grids)")
