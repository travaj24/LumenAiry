import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate as rs
from lumenairy.propagators.sas import scalable_angular_spectrum_propagate as sas
from lumenairy.propagators.mft import fresnel_propagate_mft, resample_field

print("=== SAS vs exact Fresnel-MFT on the SAS output grid (paraxial Gaussian) ===")
lam=0.633e-6
for N,dx,zfac in ((512,1e-6,4.0),(512,1e-6,8.0),(1024,1e-6,4.0)):
    z = zfac*N*dx**2/lam
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    E=np.exp(-(X**2+Y**2)/(20e-6)**2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Es, dxo, _ = sas(E, z, lam, dx)
    Ef = fresnel_propagate_mft(E, z, lam, dx, dxo, N)
    sc = np.vdot(Ef,Es)/np.vdot(Ef,Ef)
    print(f"  N={N} z={z*1e3:.3f}mm dx_out={dxo*1e6:.3f}um: "
          f"scale={sc.real:+.6f}{sc.imag:+.6f}j |scale|={abs(sc):.6f} "
          f"relL2(raw)={np.linalg.norm(Es-Ef)/np.linalg.norm(Ef):.3e} "
          f"relL2(scaled)={np.linalg.norm(Es-sc*Ef)/np.linalg.norm(Ef):.3e}")
    # power check
    print(f"      P_in={np.sum(abs(E)**2)*dx*dx:.6e}  P_sas={np.sum(abs(Es)**2)*dxo*dxo:.6e}  "
          f"P_fresnelmft={np.sum(abs(Ef)**2)*dxo*dxo:.6e}")

print()
print("=== SAS pad dependence (physical result must not depend on pad) ===")
N=512; dx=1e-6; z=4*N*dx**2/lam
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E=np.exp(-(X**2+Y**2)/(20e-6)**2).astype(np.complex128)
for pad in (1,2,3,4):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Es, dxo, _ = sas(E, z, lam, dx, pad=pad)
    Ef = fresnel_propagate_mft(E, z, lam, dx, dxo, N)
    sc = np.vdot(Ef,Es)/np.vdot(Ef,Ef)
    print(f"  pad={pad}: dx_out={dxo*1e6:.4f}um  scale vs exact Fresnel = {abs(sc):.6f}  (1/pad^2={1/pad**2:.6f})")

print()
print("=== SAS complex64 cancellation in (h_AS - h_Fr) ===")
N=512; dx=1e-6; z=4*N*dx**2/lam
for dt in (np.complex128, np.complex64):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Es, dxo, _ = sas(E.astype(dt), z, lam, dx)
    print(f"  dtype={np.dtype(dt).name}: max|E|={abs(Es).max():.6e}")
# direct kernel comparison
Nn=1024
f=np.fft.fftfreq(Nn, d=dx)
for fdt in (np.float64, np.float32):
    fx=f.astype(fdt); cx=lam*fx[None,:]; cy=lam*fx[:,None]
    hAS=np.sqrt((1.0+0j)-cx**2-cy**2); hFr=1.0-0.5*(cx**2+cy**2)
    d=(hAS-hFr)
    print(f"  {np.dtype(fdt).name}: (h_AS-h_Fr) dtype={d.dtype}")
    if fdt is np.float64: ref=d.astype(np.complex128)
    else:
        err=np.abs(d.astype(np.complex128)-ref)
        k=2*np.pi/lam
        print(f"     max|delta(h_AS-h_Fr)| = {err.max():.3e} -> phase err k*z*that at z={z*1e3:.2f}mm: "
              f"{(k*z*err).max():.3e} rad ; at z=1m: {(2*np.pi/lam*1.0*err).max():.3e} rad")

print()
print("=== resample_field accuracy vs carrier frequency (order=3, complex) ===")
N=256; dx=1e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
env=np.exp(-(X**2+Y**2)/(60e-6)**2)
for cyc_per_px in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
    E=(env*np.exp(2j*np.pi*cyc_per_px*X/dx)).astype(np.complex128)
    for fac in (0.5, 1.5):
        Er,dxo=resample_field(E, dx, dx*fac, N_out=int(N/fac))
        P0=np.sum(abs(E)**2)*dx*dx; P1=np.sum(abs(Er)**2)*dxo*dxo
        print(f"  carrier {cyc_per_px:.2f} cyc/px  scale {fac}: P_out/P_in={P1/P0:.6f}")

print()
print("=== resample_field identity (dx_out == dx_in) ===")
E=(env*np.exp(2j*np.pi*0.3*X/dx)).astype(np.complex128)
Er,_=resample_field(E,dx,dx,N_out=N)
print(f"  relL2 vs input = {np.linalg.norm(Er-E)/np.linalg.norm(E):.3e}")

print()
print("=== RS circular-aperture convergence vs dx (on-axis) ===")
lam=0.633e-6; a=20e-6; z=100e-6
k=2*np.pi/lam
I_an = 4*np.sin(k/2*(np.sqrt(z**2+a**2)-z))**2
for N,dx in ((512,0.5e-6),(1024,0.25e-6),(2048,0.125e-6),(4096,0.0625e-6)):
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    E=(np.sqrt(X**2+Y**2)<=a).astype(np.complex128)
    Er=rs(E,z,lam,dx); Ea=asm(E,z,lam,dx,bandlimit=False)
    print(f"  N={N:5d} dx={dx*1e9:6.1f}nm : I_RS={abs(Er[N//2,N//2])**2:.5f}  "
          f"I_ASM={abs(Ea[N//2,N//2])**2:.5f}  analytic={I_an:.5f}  "
          f"RS-ASM relL2={np.linalg.norm(Er-Ea)/np.linalg.norm(Ea):.2e}")
