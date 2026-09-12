import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
import lumenairy as la
from lumenairy.elements import apply_aperture, apply_spherical_lens, apply_real_lens
from lumenairy.elements.doe import create_periodic_phase_mask, create_fresnel_zone_plate

print("=== apply_aperture: hard-edge area error vs analytic (no grey-pixel option) ===", flush=True)
for N,dx,D in [(128,1e-5,5e-4),(512,2.5e-6,5e-4),(2048,6.25e-7,5e-4)]:
    E=np.ones((N,N),complex)
    A=apply_aperture(E,dx,'circular',{'diameter':D})
    area=np.sum(np.abs(A)>0)*dx*dx
    print(f"  N={N:5d} D/dx={D/dx:7.1f}px  pixel area={area:.6e}  analytic={np.pi*(D/2)**2:.6e}"
          f"  rel err={100*(area/(np.pi*(D/2)**2)-1):+7.3f}%", flush=True)
print("  -> no anti-aliased / grey-pixel edge option exists in apply_aperture", flush=True)

print("\n=== create_periodic_phase_mask with a NON-SQUARE cell ===", flush=True)
for shape in [(8,8),(4,8),(8,4)]:
    cell=np.random.default_rng(0).random(shape)*2*np.pi
    try:
        M=create_periodic_phase_mask(64, 2e-6, cell, 2e-6)
        # check whether all cell columns are actually used
        coord=(np.arange(64)-32)*2e-6
        idx=np.round(np.mod(coord,shape[0]*2e-6)/2e-6).astype(int)%shape[0]
        print(f"  cell{shape}: OK, used axis-1 indices = {sorted(set(idx.tolist()))} of 0..{shape[1]-1}"
              f"  -> {'ALL COLUMNS USED' if set(idx.tolist())==set(range(shape[1])) else 'SOME CELL COLUMNS NEVER SAMPLED'}", flush=True)
    except Exception as e:
        print(f"  cell{shape}: {type(e).__name__}: {str(e)[:80]}", flush=True)

print("\n=== FZP binary amplitude 1st-order efficiency (~1/pi^2) ===", flush=True)
N,dx,f,wl=2048,1e-6,20e-3,1e-6
T=create_fresnel_zone_plate(N,dx,f,wl,binary=True)
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); R=np.hypot(X,Y)
pup=(R<=0.9e-3).astype(complex)
Pin=np.sum(np.abs(pup)**2)
Ez=la.angular_spectrum_propagate(pup*T,f,wl,dx); I=np.abs(Ez)**2
rr=3*1.22*wl*f/(2*0.9e-3)
print(f"  binary amplitude: core power/P_in = {I[R<rr].sum()/Pin:.4f}  (theory 1/pi^2 = {1/np.pi**2:.4f})", flush=True)
Tp=create_fresnel_zone_plate(N,dx,f,wl,binary=False)
Ez=la.angular_spectrum_propagate(pup*Tp,f,wl,dx); I=np.abs(Ez)**2
print(f"  binary phase    : core power/P_in = {I[R<rr].sum()/Pin:.4f}  (theory 4/pi^2 = {4/np.pi**2:.4f})", flush=True)

print("\n=== apply_spherical_lens vs apply_real_lens on the same 2-surface prescription ===", flush=True)
wl=1.0e-6; N=1024; dx=2.0e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
E0=np.exp(-(X**2+Y**2)/(0.6e-3)**2).astype(complex)
n=1.5168; R1,R2,d=20e-3,-20e-3,3e-3
Es=apply_spherical_lens(E0,R1=R1,R2=R2,d=d,n_lens=n,wavelength=wl,dx=dx)
presc={'surfaces':[{'radius':R1,'thickness':d,'material':n,'semi_diameter':2.0e-3},
                   {'radius':R2,'thickness':0.0,'material':1.0,'semi_diameter':2.0e-3}]}
try:
    Er=apply_real_lens(E0,prescription=presc,wavelength=wl,dx=dx)
    m=(np.abs(E0)>1e-3*np.abs(E0).max())
    dphi=np.angle(Es[m]/Er[m]); dphi-=np.mean(dphi)
    print(f"  RMS phase difference over the illuminated pupil = {np.std(dphi):.4f} rad"
          f" = {np.std(dphi)/(2*np.pi):.4f} waves ; PV = {np.ptp(dphi):.4f} rad", flush=True)
except Exception as e:
    print("  apply_real_lens:", type(e).__name__, str(e)[:250], flush=True)
