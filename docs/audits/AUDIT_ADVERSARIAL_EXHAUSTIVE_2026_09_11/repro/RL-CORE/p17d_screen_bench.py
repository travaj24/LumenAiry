"""Micro-benchmark: the per-surface phase-screen application."""
import numpy as np, time, tracemalloc
N = 2048
k0 = 2*np.pi/632.8e-9
rng = np.random.default_rng(0)
opd = (rng.random((N, N)) * 1e-5)
E0 = (rng.random((N, N)) + 1j*rng.random((N, N))).astype(np.complex128)

def t(f, n=3):
    f()  # warm
    ts = []
    for _ in range(n):
        a = time.perf_counter(); f(); ts.append(time.perf_counter()-a)
    return min(ts)*1e3

def shipped():
    E = E0.copy()
    ph = np.exp(-1j*k0*opd)
    return E*ph

def cos_sin_out():
    E = E0.copy()
    phi = np.empty_like(opd); np.multiply(opd, -k0, out=phi)
    c = np.empty_like(phi); np.cos(phi, out=c)
    s = phi; np.sin(phi, out=s)          # reuse phi
    er = E.real.copy()
    ei = E.imag
    tmp = np.empty_like(c)
    np.multiply(er, c, out=tmp); tmp += ei*s        # new real
    np.multiply(ei, c, out=ei); ei -= er*s          # new imag (in place)
    E.real[...] = tmp
    return E

def cos_sin_complex():
    E = E0.copy()
    phi = opd * (-k0)
    ph = np.empty(opd.shape, dtype=np.complex128)
    pv = ph.view(np.float64).reshape(N, N, 2)
    np.cos(phi, out=pv[..., 0]); np.sin(phi, out=pv[..., 1])
    E *= ph
    return E

a = shipped(); b = cos_sin_out(); c = cos_sin_complex()
print("agreement: shipped vs cos/sin-out      max|d| =", np.abs(a-b).max())
print("agreement: shipped vs cos/sin-complex  max|d| =", np.abs(a-c).max())
print()
print(f"shipped   E*np.exp(-1j*k0*opd)      : {t(shipped):8.2f} ms")
print(f"cos/sin into a complex view + E*=ph : {t(cos_sin_complex):8.2f} ms")
print(f"cos/sin, fully in-place on E        : {t(cos_sin_out):8.2f} ms")
print()
for nm, f in (('shipped', shipped), ('cos_sin_complex', cos_sin_complex),
              ('cos_sin_out', cos_sin_out)):
    tracemalloc.start(); f()
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  {nm:<18} tracemalloc peak {peak/1e6:8.2f} MB "
          f"({peak/(16.0*N*N):.2f} complex128 grids)")
