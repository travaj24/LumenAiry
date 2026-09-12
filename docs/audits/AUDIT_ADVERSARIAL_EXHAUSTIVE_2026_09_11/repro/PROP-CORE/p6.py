import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.system import propagate_through_system
from lumenairy.propagators.asm import angular_spectrum_propagate as asm, _build_asm_H_square, _get_asm_H_natural
from lumenairy.propagators.mft import resample_field
from lumenairy.propagators import fft_infra as fi
from lumenairy.propagators.dispatch import propagate
lam=0.633e-6

print("=== ADVERSE case: system fresnel leg on a grid-filling field ===")
N=512; dx=2e-6; z=5e-3
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
R=np.sqrt(X**2+Y**2)
E0=(R < 0.42*N*dx).astype(complex)     # broad top-hat filling most of the grid
Ea=asm(E0,z,lam,dx,bandlimit=False)
for meth in ('asm','fresnel','sas'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=propagate_through_system(E0,[{'type':'propagate','z':z}],wavelength=lam,dx=dx,method=meth)
    Ef=r[0] if isinstance(r,tuple) else r
    print(f"  {meth:8s}: P_out/P_in={np.sum(abs(Ef)**2)/np.sum(abs(E0)**2):.6f}  relL2 vs ASM={np.linalg.norm(Ef-Ea)/np.linalg.norm(Ea):.3e}")

print()
print("=== resample_field on a REAL Fresnel output (the system path) ===")
from lumenairy.propagators.fresnel import fresnel_propagate
Efr,dxn,_=fresnel_propagate(E0,z,lam,dx)
P_fr=np.sum(abs(Efr)**2)*dxn*dxn; P_in=np.sum(abs(E0)**2)*dx*dx
Er,_=resample_field(Efr,dxn,dx,N_out=N)
P_rs=np.sum(abs(Er)**2)*dx*dx
print(f"  dx_out/dx={dxn/dx:.4f}; P(fresnel out)/P_in={P_fr/P_in:.6f}; after resample P/P_in={P_rs/P_in:.6f}")
# how much is crop vs interpolation?
keep = (np.abs((np.arange(N)-N//2)*dxn)[None,:] <= N/2*dx) & (np.abs((np.arange(N)-N//2)*dxn)[:,None] <= N/2*dx)
print(f"  power inside the retained window before resample: {np.sum(abs(Efr)**2*keep)*dxn*dxn/P_in:.6f} (crop loss)")

print()
print("=== _build_asm_H_square bit-identity vs _get_asm_H_natural (docstring claim) ===")
for N_,dx_,z_ in ((64,1e-6,1e-4),(256,0.5e-6,3e-4),(255,0.5e-6,3e-4)):
    for bl in (True,False):
        Hs=_build_asm_H_square(N_,dx_,z_,lam,dtype=np.complex128,bandlimit=bl)
        fi.clear_asm_caches()
        Hn=_get_asm_H_natural(N_,N_,dx_,dx_,lam,z_,bl,np.dtype(np.complex128),np)
        Hn_c=np.fft.fftshift(Hn)
        eq=np.array_equal(Hs.view(np.float64),Hn_c.view(np.float64))
        mx=np.abs(Hs-Hn_c).max()
        print(f"  N={N_} bl={bl}: byte-identical={eq}  max|diff|={mx:.3e}")

print()
print("=== H cache: is the entry handed out by reference (mutation hazard)? ===")
fi.clear_asm_caches()
E=np.ones((256,256),complex)
_,H1=asm(E,1e-4,lam,1e-6,return_transfer_function=True)
Hc=fi._h_cache_lookup((256,256,1e-6,1e-6,lam,1e-4,True,np.dtype(np.complex128).str,'ASM'))
print(f"  cached H is not None: {Hc is not None}; returned H is a copy: {Hc is not None and not np.shares_memory(H1,Hc)}")
# but internal callers get the raw object:
H2=_get_asm_H_natural(256,256,1e-6,1e-6,lam,1e-4,True,np.dtype(np.complex128),np)
H3=_get_asm_H_natural(256,256,1e-6,1e-6,lam,1e-4,True,np.dtype(np.complex128),np)
print(f"  two internal fetches share memory (aliased cache entry): {np.shares_memory(H2,H3)}")

print()
print("=== _PYFFTW_BAD_SHAPES keyed on bare shape only? ===")
print("  _PYFFTW_BAD_SHAPES type:", type(fi._PYFFTW_BAD_SHAPES).__name__,
      "; _handle_pyfftw_failure adds tuple(x.shape) only ->",
      "shape-only key (dtype/direction agnostic)")

print()
print("=== snapshot/restore_fft_state completeness ===")
g=vars(fi)
keys=set(fi._FFT_STATE_KEYS)
cands=[k for k,v in g.items() if (k.isupper() or k.startswith('_PYFFTW') or k.startswith('_H_CACHE') or k.startswith('_FREQ') or k.startswith('_BANDLIMIT') or k.startswith('_FFTW'))
       and isinstance(v,(int,float,bool,str,tuple,type(np.dtype('c16'))))
       and not k.endswith('_AVAILABLE') and not k.endswith('_LOCK') and not k.endswith('CACHE')]
print("  captured:", sorted(keys))
print("  NOT captured but setter/tunable-looking:", sorted(set(cands)-keys))
