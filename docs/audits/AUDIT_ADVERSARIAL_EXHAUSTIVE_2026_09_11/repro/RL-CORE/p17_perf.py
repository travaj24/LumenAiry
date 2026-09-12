"""Probe 17: performance + memory of the default path."""
import sys, time, tracemalloc, cProfile, pstats, io
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens

lam = 632.8e-9
rx = dict(surfaces=[
    dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-SF11'),
    dict(radius=-200e-3, glass_before='N-SF11', glass_after='AIR')],
    thicknesses=[3e-3, 2e-3], aperture_diameter=4e-3)


def field(N, dx, dt=np.complex128):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(dt)


print("=== wall clock (best of 3), 3-surface element ===")
for N in (512, 1024, 2048):
    dx = 4e-3 / (0.8 * N)
    for cr, tag in ((0, 'whole-grid'), (None, 'auto-band'), (256, 'band=256')):
        E = field(N, dx)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                            sag_chunk_rows=cr)
            ts.append(time.perf_counter() - t0)
        print(f"  N={N:<6} {tag:<11} {min(ts)*1e3:8.2f} ms")

print()
print("=== tracemalloc peak (whole-grid vs banded), grids of 8*N*N bytes ===")
for N in (1024, 2048):
    dx = 4e-3 / (0.8 * N)
    for cr, tag in ((0, 'whole-grid'), (256, 'band=256')):
        E = field(N, dx)
        E.copy()                         # warm
        tracemalloc.start()
        apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                        sag_chunk_rows=cr)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        g = 8.0 * N * N
        print(f"  N={N:<6} {tag:<11} peak {peak/1e6:9.2f} MB = "
              f"{peak/g:6.2f} float64 grids  ({peak/(2*g):5.2f} complex128)")

print()
print("=== validator + setup overhead: empty-ish call at tiny N ===")
Etiny = field(64, 1e-5)
t0 = time.perf_counter()
for _ in range(200):
    apply_real_lens(Etiny.copy(), prescription=rx, wavelength=lam, dx=1e-5)
print(f"  200 calls at N=64: {(time.perf_counter()-t0)*1e3:.1f} ms total "
      f"({(time.perf_counter()-t0)*5:.3f} ms/call)")

print()
print("=== cProfile at N=2048 (whole-grid) ===")
N = 2048
dx = 4e-3 / (0.8 * N)
E = field(N, dx)
pr = cProfile.Profile()
pr.enable()
apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                sag_chunk_rows=0)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(16)
print(s.getvalue()[:3500])

print()
print("=== is the ASM H cache HIT across surfaces with equal gaps? ===")
rx_eq = dict(surfaces=[
    dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-BK7'),
    dict(radius=-200e-3, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[3e-3, 3e-3], aperture_diameter=4e-3)
rx_ne = dict(surfaces=[
    dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-BK7'),
    dict(radius=-200e-3, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[3e-3, 3.0001e-3], aperture_diameter=4e-3)
N, dx = 2048, 4e-3 / (0.8 * 2048)
E = field(N, dx)
for nm, r in (('equal gaps (cache hit)', rx_eq),
              ('unequal gaps (2 kernels)', rx_ne)):
    apply_real_lens(E.copy(), prescription=r, wavelength=lam, dx=dx)  # warm
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        apply_real_lens(E.copy(), prescription=r, wavelength=lam, dx=dx)
        ts.append(time.perf_counter() - t0)
    print(f"  {nm:<26} {min(ts)*1e3:8.2f} ms")
