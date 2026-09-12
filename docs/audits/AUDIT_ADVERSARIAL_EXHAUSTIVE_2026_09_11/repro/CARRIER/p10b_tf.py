"""Probe 10b: cost of the exact TF build; cheaper equivalents."""
import sys, time, tracemalloc, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
wl = 1.31e-6; k = 2*np.pi/wl
N = 2048; dx = 2e-6; z = 5e-3

def shipped(L=0.0, M=0.0):
    kx = 2.0*np.pi*np.fft.fftfreq(N, d=dx); ky = kx
    s2 = L*L+M*M; Nz = np.sqrt(1.0-s2)
    KX = kx[None, :]; KY = ky[:, None]
    ax = k*L + KX; ay = k*M + KY
    rad = k*k - (ax*ax + ay*ay)
    np.maximum(rad, 0.0, out=rad)
    root = np.sqrt(rad)
    root0 = float(np.sqrt(max(k*k*(1.0-s2), 0.0)))
    lin = (L*KX + M*KY)/Nz
    phase = (k*z) + z*(root - root0 + lin)
    return np.exp(1j*phase)

def lean_untilted():
    kx2 = (2.0*np.pi*np.fft.fftfreq(N, d=dx))**2
    rad = (k*k - kx2[None, :]) - kx2[:, None]
    np.maximum(rad, 0.0, out=rad)
    np.sqrt(rad, out=rad)
    rad -= k
    rad *= z
    rad += k*z
    H = np.empty((N, N), dtype=np.complex128)
    np.cos(rad, out=H.real); np.sin(rad, out=H.imag)
    return H

def lean_cossin_only():
    kx = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
    KX = kx[None, :]; KY = kx[:, None]
    rad = k*k - (KX*KX + KY*KY)
    np.maximum(rad, 0.0, out=rad)
    root = np.sqrt(rad)
    phase = (k*z) + z*(root - k)
    H = np.empty((N, N), dtype=np.complex128)
    np.cos(phase, out=H.real); np.sin(phase, out=H.imag)
    return H

for name, fn in (('shipped (tilt 0)', shipped),
                 ('cos/sin instead of exp', lean_cossin_only),
                 ('lean+inplace+cos/sin', lean_untilted)):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(3): H = fn()
    t1 = time.perf_counter()
    tracemalloc.start(); b = tracemalloc.get_traced_memory()[0]
    H = fn(); c, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"{name:26s} {1e3*(t1-t0)/3:8.1f} ms  peak {(p-b)/1e6:7.1f} MB = {(p-b)/(16.0*N*N):.2f} grids")
Hs = shipped(); Hl = lean_untilted(); Hc = lean_cossin_only()
print("max|shipped-lean| =", np.abs(Hs-Hl).max(), " max|shipped-cossin| =", np.abs(Hs-Hc).max())

print("\n=== tilt ramp: whole-grid exp vs separable outer product ===")
from lumenairy.propagators.carrier import _tilt_ramp
L, M = 0.03, -0.02
_tilt_ramp((N,N), dx, wl, L, M, 0.0, 0.0, +1)
t0 = time.perf_counter()
for _ in range(3): rp = _tilt_ramp((N,N), dx, wl, L, M, 0.0, 0.0, +1)
t1 = time.perf_counter()
x = (np.arange(N, dtype=np.float64) - N/2)*dx
t2 = time.perf_counter()
for _ in range(3):
    rp2 = np.exp(1j*k*L*x)[None, :]*np.exp(1j*k*M*x)[:, None]
t3 = time.perf_counter()
print(f"  whole-grid {1e3*(t1-t0)/3:7.1f} ms   separable {1e3*(t3-t2)/3:7.1f} ms  "
      f"speedup {(t1-t0)/(t3-t2):.1f}x  max|diff|={np.abs(rp-rp2).max():.3e}")
