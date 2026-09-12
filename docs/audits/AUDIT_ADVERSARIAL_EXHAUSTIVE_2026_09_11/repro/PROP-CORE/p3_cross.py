import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
from lumenairy.propagators.fresnel import (fresnel_propagate, fraunhofer_propagate,
                                           fresnel_tf_propagate)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate as rs
from lumenairy.propagators.sas import scalable_angular_spectrum_propagate as sas
from lumenairy.propagators.mft import (angular_spectrum_propagate_mft,
                                       fresnel_propagate_mft, resample_field)
from lumenairy.propagators._bluestein import _bluestein_2d, _bluestein_centred_2d
from lumenairy.propagators.fft_infra import _fft2, _ifft2

# ---------- 6: Bluestein vs direct DFT ----------
print("=== Bluestein vs direct DFT (N=64) ===")
rng = np.random.default_rng(1)
Nin=64; Nout=48
E = rng.standard_normal((Nin,Nin))+1j*rng.standard_normal((Nin,Nin))
for sign in (+1,-1):
    ax, ay = 0.013, 0.021
    F = _bluestein_2d(E, ax, ay, Nout, Nout, sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2)
    n = np.arange(Nin); kk = np.arange(Nout)
    Mx = np.exp(sign*2j*np.pi*ax*np.outer(kk,n))
    My = np.exp(sign*2j*np.pi*ay*np.outer(kk,n))
    Fd = My @ E @ Mx.T
    print(f"  sign={sign:+d} 2d : rel = {np.linalg.norm(F-Fd)/np.linalg.norm(Fd):.3e}")
    Fs = _bluestein_2d(E, ax, ay, Nout, Nout, sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2, separable=True)
    print(f"  sign={sign:+d} sep: rel = {np.linalg.norm(Fs-Fd)/np.linalg.norm(Fd):.3e}")
    Fc = _bluestein_centred_2d(E, ax, ay, Nout, Nout, sign=sign, xp=np, fft2=_fft2, ifft2=_ifft2)
    nn = n-Nin/2; kkc = kk-Nout/2
    Mxc = np.exp(sign*2j*np.pi*ax*np.outer(kkc,nn)); Myc=np.exp(sign*2j*np.pi*ay*np.outer(kkc,nn))
    Fcd = Myc @ E @ Mxc.T
    print(f"  sign={sign:+d} ctr: rel = {np.linalg.norm(Fc-Fcd)/np.linalg.norm(Fcd):.3e}")

# ---------- 3: fresnel vs fresnel_mft & vs ASM paraxial ----------
print()
print("=== Fresnel single-FFT vs Fresnel-MFT (same output grid) ===")
lam=0.633e-6
for N,dx,z in ((256,4e-6,0.05),(257,4e-6,0.05),(512,2e-6,0.02)):
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    E=( (np.sqrt(X**2+Y**2)<N*dx/8) ).astype(np.complex128)
    Ef, dxo, dyo = fresnel_propagate(E, z, lam, dx)
    Em = fresnel_propagate_mft(E, z, lam, dx, dxo, N)
    print(f"  N={N} z={z}: dx_out={dxo*1e6:.4f}um relL2={np.linalg.norm(Ef-Em)/np.linalg.norm(Em):.3e}")

print()
print("=== Fresnel-TF vs ASM (paraxial regime) and its own paraxial oracle ===")
N=512; dx=2e-6; lam=0.633e-6; z=2e-3
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
w0=30e-6
E=np.exp(-(X**2+Y**2)/w0**2).astype(np.complex128)
Etf = fresnel_tf_propagate(E,z,lam,dx)
Ea  = asm(E,z,lam,dx,bandlimit=False)
print(f"  fresnel_tf vs asm relL2 = {np.linalg.norm(Etf-Ea)/np.linalg.norm(Ea):.3e}")
# own oracle: paraxial H
f=np.fft.fftfreq(N,dx); FX,FY=np.meshgrid(f,f,indexing='xy')
k=2*np.pi/lam
Horacle=np.exp(1j*k*z)*np.exp(-1j*np.pi*lam*z*(FX**2+FY**2))
Eo=np.fft.ifft2(np.fft.fft2(E)*Horacle)
print(f"  fresnel_tf vs analytic-paraxial-H relL2 = {np.linalg.norm(Etf-Eo)/np.linalg.norm(Eo):.3e}")

# ---------- 4: RS vs ASM and vs analytic circ aperture on-axis ----------
print()
print("=== RS: circular aperture on-axis analytic ===")
N=1024; dx=0.25e-6; lam=0.633e-6; a=20e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E=(np.sqrt(X**2+Y**2)<=a).astype(np.complex128)
k=2*np.pi/lam
for z in (30e-6, 100e-6, 300e-6):
    Er = rs(E,z,lam,dx)
    Ea = asm(E,z,lam,dx,bandlimit=False)
    I_an = 4*np.sin(k/2*(np.sqrt(z**2+a**2)-z))**2
    print(f"  z={z*1e6:6.1f}um  I_RS(0)={abs(Er[N//2,N//2])**2:.5f}  "
          f"I_ASM(0)={abs(Ea[N//2,N//2])**2:.5f}  I_analytic={I_an:.5f}  "
          f"RS-vs-ASM relL2={np.linalg.norm(Er-Ea)/np.linalg.norm(Ea):.3e}")

# ---------- 5: SAS vs ASM ----------
print()
print("=== SAS vs ASM (output pitch 2x input) ===")
N=512; dx=1e-6; lam=0.633e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
w0=20e-6
E=np.exp(-(X**2+Y**2)/w0**2).astype(np.complex128)
# pick z such that dx_out = lam z /(2 N dx) = 2 dx  ->  z = 4 N dx^2/lam
z = 4*N*dx**2/lam
print(f"  target z={z*1e3:.4f} mm")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Es, dxo, _ = sas(E, z, lam, dx)
print(f"  dx_out={dxo*1e6:.4f} um  (target {2*dx*1e6} um)")
# oracle: ASM on a fine grid then sample at dx_out
Ea = asm(E, z, lam, dx, bandlimit=False)
xo=(np.arange(N)-N/2)*dxo
# compare via ASM-MFT at the SAS output grid
Em = angular_spectrum_propagate_mft(E, z, lam, dx, dxo, N, bandlimit=False)
sc = np.vdot(Em, Es)/np.vdot(Em,Em)
print(f"  SAS vs ASM-MFT on SAS grid: complex scale={sc:.6f}  "
      f"relL2(after scale)={np.linalg.norm(Es-sc*Em)/np.linalg.norm(Em):.3e}  "
      f"raw relL2={np.linalg.norm(Es-Em)/np.linalg.norm(Em):.3e}")

# ---------- asm_mft vs asm on the same grid ----------
print()
print("=== ASM-MFT vs ASM on identical grid ===")
for N in (256, 257):
    dx=1e-6; z=5e-4
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    E=np.exp(-(X**2+Y**2)/(15e-6)**2).astype(np.complex128)
    A=asm(E,z,lam,dx,bandlimit=False)
    M=angular_spectrum_propagate_mft(E,z,lam,dx,dx,N,bandlimit=False)
    print(f"  N={N}: relL2 = {np.linalg.norm(A-M)/np.linalg.norm(A):.3e}")

# ---------- resample_field ----------
print()
print("=== resample_field energy / phase ===")
N=256; dx=1e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E=(np.exp(-(X**2+Y**2)/(20e-6)**2)*np.exp(20j*X/1e-5)).astype(np.complex128)
for fac in (0.5, 2.0):
    Er, dxo = resample_field(E, dx, dx*fac, N_out=N)
    P0=np.sum(abs(E)**2)*dx*dx; P1=np.sum(abs(Er)**2)*dxo*dxo
    print(f"  scale {fac}x: P_out/P_in = {P1/P0:.6f}  max|E|={abs(Er).max():.4f} (in {abs(E).max():.4f})")
