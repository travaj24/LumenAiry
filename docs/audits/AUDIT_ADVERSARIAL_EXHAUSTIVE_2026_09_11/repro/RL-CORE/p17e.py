import sys, time, tracemalloc
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements.lenses import surface_sag_general as sg
N = 2048
x = (np.arange(N) - N/2)*2e-6
X, Y = np.meshgrid(x, x)
h2 = X*X + Y*Y
g = 8.0*N*N
def t(f, n=3):
    f(); ts=[]
    for _ in range(n):
        a=time.perf_counter(); f(); ts.append(time.perf_counter()-a)
    return min(ts)*1e3
def shipped(): return sg(h2, 50e-3, -0.5, None)
def lean():
    c = 1.0/50e-3; k = -0.5
    nrm = np.empty_like(h2); np.multiply(h2, (1+k)*c*c, out=nrm)
    np.subtract(1.0, nrm, out=nrm); np.maximum(nrm, 0.0, out=nrm)
    np.sqrt(nrm, out=nrm); np.add(nrm, 1.0, out=nrm)
    out = np.empty_like(h2); np.multiply(h2, c, out=out); np.divide(out, nrm, out=out)
    return out
for nm, f in (('surface_sag_general', shipped), ('in-place equivalent', lean)):
    tracemalloc.start(); f()
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  {nm:<24} {t(f):8.2f} ms   tracemalloc peak {peak/1e6:8.2f} MB "
          f"= {peak/g:.2f} float64 grids")
print()
k0 = 2*np.pi/632.8e-9
opd = np.abs(np.random.default_rng(0).random((N, N)))*1e-5
E0 = (np.random.default_rng(1).random((N, N)) +
      1j*np.random.default_rng(2).random((N, N)))
def out_of_place():
    E = E0.copy(); ph = np.exp(-1j*k0*opd); E = E*ph; return E
def in_place():
    E = E0.copy(); ph = np.exp(-1j*k0*opd); E *= ph; return E
for nm, f in (('E = E * exp(...)  (shipped)', out_of_place),
              ('E *= exp(...)      (in-place)', in_place)):
    tracemalloc.start(); f()
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  {nm:<32} {t(f):8.2f} ms  peak {peak/1e6:8.2f} MB "
          f"= {peak/(2*g):.2f} complex128 grids")
