import numpy as np, sys
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.asm import angular_spectrum_propagate as asm, _build_asm_H_square
from lumenairy.propagators import fft_infra as fi

lam=1e-6; dx=0.5e-6
def mkfield(N, dt):
    x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    return (np.exp(-(X**2+Y**2)/(2*(10e-6)**2))*np.exp(3j*X/1e-5)).astype(dt)

print("=== (g) stream_transfer_function byte identity ===")
for N in (128, 512):
    for dt in (np.complex128, np.complex64):
        E = mkfield(N, dt)
        fi.clear_asm_caches()
        A = asm(E.copy(), 1e-4, lam, dx)
        fi.clear_asm_caches()
        B = asm(E.copy(), 1e-4, lam, dx, stream_transfer_function=True)
        same = (A.dtype==B.dtype) and np.array_equal(A.view(np.float64 if A.dtype==np.complex128 else np.float32),
                                                     B.view(np.float64 if B.dtype==np.complex128 else np.float32))
        rel = np.linalg.norm(A-B)/np.linalg.norm(A)
        print(f"  N={N} {np.dtype(dt).name}: plain.dtype={A.dtype} stream.dtype={B.dtype} "
              f"byte-identical={same} relL2={rel:.3e}")

print()
print("=== dtype of _fft2 on complex64 input ===")
E64 = mkfield(512, np.complex64)
s = fi._fft2(np.fft.ifftshift(E64))
print(f"  _fft2(complex64).dtype = {s.dtype}")

print()
print("=== (f) H cache key staleness probes ===")
E = mkfield(256, np.complex128)
fi.clear_asm_caches()
A1 = asm(E, 1e-4, lam, dx, bandlimit=True)
A2 = asm(E, 1e-4, lam, dx, bandlimit=False)
print(f"  bandlimit toggle differs: {not np.allclose(A1,A2)}  rel={np.linalg.norm(A1-A2)/np.linalg.norm(A1):.3e}")
Ap = asm(E, +1e-4, lam, dx); Am = asm(E, -1e-4, lam, dx)
print(f"  z sign differs: {not np.allclose(Ap,Am)}")
# dtype switch mid-run
fi.clear_asm_caches()
a = asm(mkfield(256,np.complex128), 1e-4, lam, dx)
fi.set_default_complex_dtype(np.complex64)
b = asm(mkfield(256,np.complex128), 1e-4, lam, dx)
fi.set_default_complex_dtype(np.complex128)
print(f"  set_default_complex_dtype mid-run on complex128 input: identical={np.array_equal(a,b)} (expected True: dtype follows input)")
# real input + dtype switch
fi.clear_asm_caches()
Er = np.real(mkfield(256,np.complex128)).copy()
c = asm(Er, 1e-4, lam, dx); print(f"  real input default c128 -> out dtype {c.dtype}")
fi.set_default_complex_dtype(np.complex64)
d = asm(Er, 1e-4, lam, dx); print(f"  real input after set c64 -> out dtype {d.dtype}")
fi.set_default_complex_dtype(np.complex128)

print()
print("=== dy!=dx anamorphic bandlimit uses dx,dy separately ===")
N=256
E2 = np.zeros((N,N), complex); E2[N//2,N//2]=1.0
Ea = asm(E2, 5e-4, lam, dx, dy=2*dx)
print(f"  anamorphic ran, finite={np.isfinite(Ea).all()}")
# check the H directly
_,H = asm(E2, 5e-4, lam, dx, dy=2*dx, return_transfer_function=True)
fx = (np.arange(N)-N//2)/(N*dx); fy=(np.arange(N)-N//2)/(N*(2*dx))
fxmax = N*dx/(2*lam*5e-4); fymax = N*2*dx/(2*lam*5e-4)
mask = (np.abs(fx)[None,:]<fxmax)&(np.abs(fy)[:,None]<fymax)
nz = (H!=0)
print(f"  H nonzero pattern == expected anamorphic mask & propagating: "
      f"{np.array_equal(nz, mask & ((2*np.pi/lam)**2 - (2*np.pi*fx)[None,:]**2 - (2*np.pi*fy)[:,None]**2 > 0))}")

print()
print("=== (e) complex64 phase error at z=1 m, lam=1um ===")
N=64; dxb=10e-6
Hc128 = _build_asm_H_square(N, dxb, 1.0, lam, dtype=np.complex128, bandlimit=False)
Hc64  = _build_asm_H_square(N, dxb, 1.0, lam, dtype=np.complex64,  bandlimit=False)
m = Hc128!=0
ph = np.angle(Hc64[m].astype(np.complex128)/Hc128[m])
print(f"  k*z = {2*np.pi/lam*1.0:.3e} rad;  max |phase err| c64 vs c128 = {np.abs(ph).max():.3e} rad")
print(f"  (naive astype(c64) of c128 exp would give: ", end='')
Hnaive = Hc128.astype(np.complex64)
ph2 = np.angle(Hnaive[m].astype(np.complex128)/Hc128[m])
print(f"{np.abs(ph2).max():.3e} rad)")
