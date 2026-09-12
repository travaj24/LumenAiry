import numpy as np, sys, gc, tracemalloc, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.asm import (angular_spectrum_propagate as asm,
                                       angular_spectrum_propagate_tilted as asmt)
from lumenairy.propagators import fft_infra as fi
lam=0.633e-6

print("=== tilted ASM vs plain ASM of the SAME tilted field ===")
N=512; dx=0.5e-6; w0=8e-6
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
z=2e-5
for th_deg in (0.5, 5.0, 15.0):
    th=np.radians(th_deg); fx0=np.sin(th)/lam
    E=(np.exp(-(X**2+Y**2)/w0**2)*np.exp(2j*np.pi*fx0*X)).astype(complex)
    # oracle: plain ASM (no bandlimit) on the SAME grid, which is exact for the
    # carrier as long as fx0 is inside Nyquist
    Eo=asm(E,z,lam,dx,bandlimit=False)
    Et=asmt(E,z,lam,dx,tilt_x=th,bandlimit=False)
    Et_bl=asmt(E,z,lam,dx,tilt_x=th,bandlimit=True)
    nyq=1/(2*dx)
    print(f"  theta={th_deg:5.1f}deg fx0={fx0:.3e} (Nyq={nyq:.3e}, {fx0/nyq*100:.0f}% of Nyq): "
          f"tilted-vs-plain relL2(bl=F)={np.linalg.norm(Et-Eo)/np.linalg.norm(Eo):.3e} "
          f"(bl=T)={np.linalg.norm(Et_bl-Eo)/np.linalg.norm(Eo):.3e}")

print()
print("=== tilted ASM: no-tilt shortcut vs tiny tilt discontinuity ===")
E=(np.exp(-(X**2+Y**2)/w0**2)).astype(complex)
A=asmt(E,z,lam,dx,tilt_x=0.0)
B=asmt(E,z,lam,dx,tilt_x=1e-16)   # below the 1e-15 fx0 cutoff? fx0=sin(1e-16)/lam=1.6e-10 -> above 1e-15
C=asmt(E,z,lam,dx,tilt_x=1e-22)
print(f"  tilt=0 vs tilt=1e-16: rel={np.linalg.norm(A-B)/np.linalg.norm(A):.3e}; "
      f"tilt=0 vs 1e-22 (shortcut): rel={np.linalg.norm(A-C)/np.linalg.norm(A):.3e}")

print()
print("=== _get_asm_H_natural chunk: peak transient vs band cap (bit-identity check) ===")
import lumenairy.memory as mem
from lumenairy.propagators.asm import _get_asm_H_natural
M=2048; dxv=0.5e-6
fi.clear_asm_caches()
gc.collect(); tracemalloc.start()
H_full=_get_asm_H_natural(M,M,dxv,dxv,lam,1e-4,True,np.dtype(np.complex128),np)
_,peak_full=tracemalloc.get_traced_memory(); tracemalloc.stop()
orig=mem.get_ram_budget
import lumenairy.propagators.asm as A
# force a small chunk by shrinking the ram budget the builder reads
mem.get_ram_budget=lambda: 3*M*16*512/0.1   # -> chunk = 512 rows
fi.clear_asm_caches(); gc.collect(); tracemalloc.start()
H_512=_get_asm_H_natural(M,M,dxv,dxv,lam,1e-4,True,np.dtype(np.complex128),np)
_,peak_512=tracemalloc.get_traced_memory(); tracemalloc.stop()
mem.get_ram_budget=lambda: 3*M*16*128/0.1   # -> chunk = 128 rows
fi.clear_asm_caches(); gc.collect(); tracemalloc.start()
H_128=_get_asm_H_natural(M,M,dxv,dxv,lam,1e-4,True,np.dtype(np.complex128),np)
_,peak_128=tracemalloc.get_traced_memory(); tracemalloc.stop()
mem.get_ram_budget=orig
one=M*M*16/1e6
print(f"  one grid = {one:.1f} MB")
print(f"  chunk=whole grid : peak {peak_full/1e6:7.1f} MB = {peak_full/1e6/one:.2f} grids")
print(f"  chunk=512 rows   : peak {peak_512/1e6:7.1f} MB = {peak_512/1e6/one:.2f} grids  byte-identical={np.array_equal(H_full.view(np.float64),H_512.view(np.float64))}")
print(f"  chunk=128 rows   : peak {peak_128/1e6:7.1f} MB = {peak_128/1e6/one:.2f} grids  byte-identical={np.array_equal(H_full.view(np.float64),H_128.view(np.float64))}")

print()
print("=== redundant where in kz: sqrt(maximum(x,0)) == where(x>0, sqrt(maximum(x,0)), 0)? ===")
rng=np.random.default_rng(0); v=rng.standard_normal(100000)*1e6
p=v>0
a=np.where(p,np.sqrt(np.maximum(v,0)),0); b=np.sqrt(np.maximum(v,0))
print(f"  identical over 1e5 random values incl. negatives: {np.array_equal(a,b)}")
