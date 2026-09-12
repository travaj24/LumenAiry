import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.dispatch import propagate, _auto_select_method
from lumenairy.propagators.system import propagate_through_system
from lumenairy.propagators.asm import angular_spectrum_propagate as asm

lam=0.633e-6
print("=== auto-selection map (N=512, dx=2um, lam=633nm) ===")
N=512; dx=2e-6
E=np.ones((N,N),complex)
a=0.5*dx*N
print(f"  a={a*1e6}um  z(Q=1)={N*dx*dx/lam*1e3:.4f} mm  z(N_F=0.1)={a*a/(lam*0.1)*1e3:.3f} mm")
for z in (1e-5,1e-4,1.04e-3,1.05e-3,1e-2,1e-1,1.0,10.0,32.0,33.0,-1e-3):
    m=_auto_select_method(E,z=z,wavelength=lam,dx=dx,prescription=None)
    Q=lam*abs(z)/(N*dx*dx); NF=a*a/(lam*abs(z))
    print(f"   z={z:+.3e}  Q={Q:9.4f}  N_F={NF:11.4f}  -> {m}")

print()
print("=== does auto ever pick fraunhofer where the dropped input chirp is large? ===")
# The Fraunhofer criterion drops exp(i k r1^2/(2z)); max dropped phase = k a^2/(2z) = pi*N_F
for NF in (0.1, 0.09):
    print(f"   N_F={NF}: max dropped input-plane phase = pi*N_F = {np.pi*NF:.4f} rad = lambda/{1/(NF/2):.1f}")

print()
print("=== ASM aliasing at the Q<=1 boundary: compare auto-asm vs an oracle ===")
# Gaussian, compare asm at Q just below 1 against a 4x-padded ASM (alias-free oracle)
N=256; dx=2e-6; w0=20e-6
for Qt in (0.5, 0.95, 1.0):
    z = Qt*N*dx*dx/lam
    x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
    E=np.exp(-(X**2+Y**2)/w0**2).astype(complex)
    Ea=asm(E,z,lam,dx)                       # default bandlimit=True
    Ea_nb=asm(E,z,lam,dx,bandlimit=False)
    P=4*N
    Ep=np.zeros((P,P),complex); o=(P-N)//2; Ep[o:o+N,o:o+N]=E
    Eo=asm(Ep,z,lam,dx,bandlimit=False)[o:o+N,o:o+N]
    print(f"   Q={Qt}: asm(bl=T) vs padded-oracle relL2={np.linalg.norm(Ea-Eo)/np.linalg.norm(Eo):.3e}  "
          f"asm(bl=F) relL2={np.linalg.norm(Ea_nb-Eo)/np.linalg.norm(Eo):.3e}")

print()
print("=== 4f relay via propagate_through_system (ASM), unit magnification ===")
N=512; dx=2e-6; f=10e-3
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
obj = (np.exp(-((X-20e-6)**2+Y**2)/(8e-6)**2)).astype(complex)
els=[{'type':'propagate','z':f},{'type':'lens','f':f},
     {'type':'propagate','z':2*f},{'type':'lens','f':f},
     {'type':'propagate','z':f}]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r=propagate_through_system(obj,els,wavelength=lam,dx=dx)
Eo = r.field if hasattr(r,'field') else r
# expect inverted image
ref = obj[::-1,::-1]
# find centroid
def centroid(I):
    I=np.abs(I)**2; tot=I.sum()
    return ((np.arange(N)[None,:]*I).sum()/tot - N//2, (np.arange(N)[:,None]*I).sum()/tot - N//2)
print(f"  input centroid  (x,y) px = {centroid(obj)}")
print(f"  output centroid (x,y) px = {centroid(Eo)}")
print(f"  power ratio = {np.sum(abs(Eo)**2)/np.sum(abs(obj)**2):.6f}")
ph = np.angle(Eo/ (ref+1e-30))
m = np.abs(ref)>0.2*np.abs(ref).max()
print(f"  |E_out| vs |E_in flipped| relL2 = {np.linalg.norm(abs(Eo)-abs(ref))/np.linalg.norm(abs(ref)):.3e}")
print(f"  phase flatness over the spot: std(angle(Eout/Eref)) = {np.std(ph[m]):.4f} rad  ptv={np.ptp(ph[m]):.4f}")

print()
print("=== system 'fresnel' leg: resample_field crop/energy loss ===")
N=512; dx=2e-6; z=5e-3
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E0=np.exp(-(X**2+Y**2)/(20e-6)**2).astype(complex)
dxn = lam*z/(N*dx)
print(f"  dx_in={dx*1e6}um  Fresnel dx_out={dxn*1e6:.4f}um  (ratio {dxn/dx:.3f})")
for meth in ('asm','fresnel','sas'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=propagate_through_system(E0,[{'type':'propagate','z':z}],wavelength=lam,dx=dx,method=meth)
    Ef=r.field if hasattr(r,'field') else r
    print(f"   method={meth:8s}: P_out/P_in={np.sum(abs(Ef)**2)/np.sum(abs(E0)**2):.6f}  max|E|={abs(Ef).max():.5f}  dx_out={getattr(r,'dx',dx)*1e6:.4f}um")
# oracle
Ea=asm(E0,z,lam,dx,bandlimit=False)
print(f"   ASM direct   : P_out/P_in={np.sum(abs(Ea)**2)/np.sum(abs(E0)**2):.6f}  max|E|={abs(Ea).max():.5f}")
