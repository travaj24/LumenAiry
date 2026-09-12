import numpy as np, sys, warnings
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.system import propagate_through_system
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
from lumenairy.propagators.dispatch import propagate
lam=0.633e-6
def unwrap(r):
    if hasattr(r,'field'): return r.field, getattr(r,'dx',None)
    if isinstance(r,tuple): 
        return r[0], (r[1] if len(r)>1 else None)
    return r, None

print("=== 4f relay via propagate_through_system (ASM) ===")
N=512; dx=2e-6; f=10e-3
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
obj = (np.exp(-((X-20e-6)**2+Y**2)/(8e-6)**2)).astype(complex)
els=[{'type':'propagate','z':f},{'type':'lens','f':f},
     {'type':'propagate','z':2*f},{'type':'lens','f':f},
     {'type':'propagate','z':f}]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r=propagate_through_system(obj,els,wavelength=lam,dx=dx)
Eo,_=unwrap(r)
print("  returned type:", type(r).__name__, "shape", np.shape(Eo))
def centroid(E):
    I=np.abs(E)**2; tot=I.sum()
    return (float((np.arange(N)[None,:]*I).sum()/tot - N//2), float((np.arange(N)[:,None]*I).sum()/tot - N//2))
print(f"  input centroid  (x,y) px = {centroid(obj)}")
print(f"  output centroid (x,y) px = {centroid(Eo)}")
print(f"  power ratio = {np.sum(abs(Eo)**2)/np.sum(abs(obj)**2):.6f}")
ref = obj[::-1,::-1]
print(f"  |E_out| vs |E_in flipped| relL2 = {np.linalg.norm(abs(Eo)-abs(ref))/np.linalg.norm(abs(ref)):.3e}")
m=np.abs(ref)>0.2*np.abs(ref).max()
ph=np.angle(Eo/(ref+1e-300)); ph=ph[m]; ph=np.unwrap(ph-np.median(ph))
print(f"  phase over spot: std={np.std(ph):.4f} rad ptv={np.ptp(ph):.4f} rad")

print()
print("=== two-lens telescope vs ABCD (Gaussian) ===")
# afocal: f1 then d=f1+f2 then f2.  w_out/w_in = f2/f1
N=1024; dx=2e-6; f1=10e-3; f2=20e-3; w0=60e-6
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E=np.exp(-(X**2+Y**2)/w0**2).astype(complex)
els=[{'type':'lens','f':f1},{'type':'propagate','z':f1+f2},{'type':'lens','f':f2}]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    r=propagate_through_system(E,els,wavelength=lam,dx=dx)
Eo,_=unwrap(r)
def waist(E):
    I=np.abs(E)**2; tot=I.sum()
    xx=(np.arange(N)-N//2)*dx
    cx=(xx[None,:]*I).sum()/tot
    var=(((xx[None,:]-cx)**2)*I).sum()/tot
    return 2*np.sqrt(var)   # 1/e^2 radius for a gaussian: w = 2*sigma_I^... use 2*sqrt(var)
print(f"  w_in (2*sigma)={waist(E)*1e6:.4f} um   w_out={waist(Eo)*1e6:.4f} um   ratio={waist(Eo)/waist(E):.5f}  expect f2/f1={f2/f1}")
print(f"  power ratio = {np.sum(abs(Eo)**2)/np.sum(abs(E)**2):.6f}")

print()
print("=== system 'fresnel'/'sas' leg: resample energy loss ===")
N=512; dx=2e-6; z=5e-3
x=(np.arange(N)-N//2)*dx; X,Y=np.meshgrid(x,x,indexing='xy')
E0=np.exp(-(X**2+Y**2)/(20e-6)**2).astype(complex)
dxn=lam*z/(N*dx)
print(f"  dx_in={dx*1e6}um Fresnel dx_out={dxn*1e6:.4f}um ratio={dxn/dx:.3f}")
Ea=asm(E0,z,lam,dx,bandlimit=False)
print(f"  ASM direct : P_out/P_in={np.sum(abs(Ea)**2)/np.sum(abs(E0)**2):.6f} max|E|={abs(Ea).max():.5f}")
for meth in ('asm','fresnel','sas'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=propagate_through_system(E0,[{'type':'propagate','z':z}],wavelength=lam,dx=dx,method=meth)
    Ef,_=unwrap(r)
    print(f"  system {meth:8s}: P_out/P_in={np.sum(abs(Ef)**2)/np.sum(abs(E0)**2):.6f} max|E|={abs(Ef).max():.5f} "
          f"relL2 vs ASM={np.linalg.norm(Ef-Ea)/np.linalg.norm(Ea):.3e}")

print()
print("=== auto-selected SAS vs its own z_limit ===")
from lumenairy.propagators.sas import scalable_angular_spectrum_propagate as sas
N=512; dx=2e-6
L=N*dx
s=L**2/(8*L**2+N**2*lam**2)
den=lam*(-1+2*np.sqrt(2)*np.sqrt(s))
zlim=-4*L*np.sqrt(8*L**2/N**2+lam**2)*np.sqrt(s)/den
print(f"  N={N} dx={dx*1e6}um: SAS z_limit = {zlim*1e3:.3f} mm ; auto picks sas for z in "
      f"({N*dx*dx/lam*1e3:.3f} mm, {(0.5*dx*N)**2/(lam*0.1)*1e3:.1f} mm)")
E=np.exp(-(X**2+Y**2)/(20e-6)**2).astype(complex)
for z in (5e-3, 1e-2, 1e-1, 1.0):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out=propagate(E, z=z, wavelength=lam, dx=dx, method='auto')
    msgs=[str(x.message)[:60] for x in w]
    Ef,dxo=unwrap(out)
    print(f"  z={z*1e3:8.2f} mm -> warnings: {len(w)}  {msgs[:1]}")
