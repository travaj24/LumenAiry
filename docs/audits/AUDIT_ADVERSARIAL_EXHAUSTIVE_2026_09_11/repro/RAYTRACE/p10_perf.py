"""RAYTRACE probe 13: performance / memory of the NumPy trace."""
import sys, time, tracemalloc, cProfile, pstats, io
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import trace, Surface, make_rings
from lumenairy.raytrace.trace import _make_bundle

WL = 1.31e-6


def system6():
    return [
        Surface(radius=50e-3, semi_diameter=15e-3, glass_before='air',
                glass_after='N-BK7', thickness=6e-3),
        Surface(radius=-80e-3, semi_diameter=15e-3, glass_before='N-BK7',
                glass_after='air', thickness=3e-3),
        Surface(radius=90e-3, semi_diameter=15e-3, glass_before='air',
                glass_after='N-SF2', thickness=4e-3),
        Surface(radius=-120e-3, semi_diameter=15e-3, glass_before='N-SF2',
                glass_after='air', thickness=20e-3),
        Surface(radius=200e-3, conic=-0.5, semi_diameter=15e-3,
                glass_before='air', glass_after='N-BK7', thickness=5e-3),
        Surface(radius=np.inf, semi_diameter=15e-3, glass_before='N-BK7',
                glass_after='air', thickness=80e-3),
        Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                glass_after='air'),
    ]


def bundle(n):
    r = 12e-3 * np.sqrt(np.random.default_rng(1).random(n))
    th = 2 * np.pi * np.random.default_rng(2).random(n)
    return _make_bundle(r * np.cos(th), r * np.sin(th),
                        np.zeros(n), np.zeros(n), WL)


surfs = system6()
for n in (1000, 100_000, 1_000_000):
    rb = bundle(n)
    for of in ('last', 'all'):
        t0 = time.perf_counter()
        res = trace(rb, surfs, WL, output_filter=of)
        t1 = time.perf_counter()
        print(f'  N={n:>9,d} output_filter={of:5s}: {t1-t0:8.4f} s '
              f'({(t1-t0)/n*1e9:8.1f} ns/ray)  alive='
              f'{int(res.image_rays.alive.sum())}')

print()
print('peak memory (tracemalloc), N=1e6, 7 surfaces:')
rb = bundle(1_000_000)
for of in ('last', 'all'):
    tracemalloc.start()
    res = trace(rb, surfs, WL, output_filter=of)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'  output_filter={of:5s}: peak {peak/2**20:8.1f} MiB  '
          f'(a single float64 (N=1e6) array is {8e6/2**20:.1f} MiB; '
          f'a RayBundle.copy() is ~{9*8e6/2**20:.1f} MiB)')
    del res

print()
print('profile, N=200k, output_filter=last:')
rb = bundle(200_000)
pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    trace(rb, surfs, WL, output_filter='last')
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(18)
print('\n'.join(s.getvalue().split('\n')[4:30]))
