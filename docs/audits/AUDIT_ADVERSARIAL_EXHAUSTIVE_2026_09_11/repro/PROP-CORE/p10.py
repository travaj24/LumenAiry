import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
import lumenairy, lumenairy.backend as B
print("=== backend.scipy.jv export surface ===")
print("  lumenairy.backend exports jv:", hasattr(B,'jv'))
import lumenairy.backend.scipy as bs
print("  module attrs:", [a for a in dir(bs) if not a.startswith('_')][:20])

print()
print("=== fresnel_propagate chirp-aliasing guard? (needs z >= N dx^2 / lambda) ===")
from lumenairy.propagators.fresnel import fresnel_propagate
from lumenairy.propagators.mft import fresnel_propagate_mft
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
lam=0.633e-6; N=256; dx=2e-6
z_crit = N*dx*dx/lam
print(f"  N={N} dx={dx*1e6}um lam={lam*1e9}nm -> z_crit = N dx^2/lam = {z_crit*1e3:.4f} mm")
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E=(np.sqrt(X**2+Y**2)<60e-6).astype(complex)
for f in (0.05, 0.2, 0.5, 1.0, 2.0):
    z=f*z_crit
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Ef,dxo,_=fresnel_propagate(E,z,lam,dx)
    Em = fresnel_propagate_mft(E,z,lam,dx,dxo,N)   # same sum, no chirp-aliasing (direct)
    rel=np.linalg.norm(Ef-Em)/np.linalg.norm(Em)
    print(f"   z={f:4.2f}*z_crit: warnings={len(w)}  relL2(single-FFT vs MFT)={rel:.3e}  dx_out={dxo*1e6:.3f}um")

print()
print("=== does the H cache distinguish bandlimit on a broadband field? ===")
from lumenairy.propagators import fft_infra as fi
N=256; dx=0.5e-6; z=2e-4
rng=np.random.default_rng(7)
Eb=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N))).astype(complex)
fi.clear_asm_caches()
A=asm(Eb,z,lam,dx,bandlimit=True); Bq=asm(Eb,z,lam,dx,bandlimit=False)
print(f"  bandlimit T vs F relL2 = {np.linalg.norm(A-Bq)/np.linalg.norm(Bq):.3e} (must be >0: keys differ)")
fi.clear_asm_caches()
Bq2=asm(Eb,z,lam,dx,bandlimit=False); A2=asm(Eb,z,lam,dx,bandlimit=True)
print(f"  reversed order: identical to first order? A:{np.array_equal(A,A2)} B:{np.array_equal(Bq,Bq2)}")

print()
print("=== reset_fft_backend / _PYFFTW_BAD_SHAPES key granularity ===")
fi._PYFFTW_BAD_SHAPES.clear()
fi._handle_pyfftw_failure(np.zeros((512,512),np.complex128),'fft2',MemoryError('x'))
print("  after a complex128 (512,512) failure, blacklist =", fi._PYFFTW_BAD_SHAPES)
print("  -> a complex64 (512,512) fft now also skips pyFFTW:",
      tuple((512,512)) in fi._PYFFTW_BAD_SHAPES)
fi.reset_fft_backend(); print("  reset clears it:", fi._PYFFTW_BAD_SHAPES)

print()
print("=== propagate() has no dy parameter ===")
import inspect
from lumenairy.propagators.dispatch import propagate
print("  signature:", list(inspect.signature(propagate).parameters))
