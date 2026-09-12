import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
import lumenairy as la
from lumenairy.elements import apply_spherical_lens, apply_real_lens
from lumenairy.glass import GLASS_REGISTRY
GLASS_REGISTRY['TESTN'] = lambda wl: 1.5168

wl=1.0e-6; N=1024; dx=2.0e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
E0=np.exp(-(X**2+Y**2)/(0.6e-3)**2).astype(complex)
R1,R2,d,n=20e-3,-20e-3,3e-3,1.5168
presc={'surfaces':[{'radius':R1,'glass_before':'air','glass_after':'TESTN'},
                   {'radius':R2,'glass_before':'TESTN','glass_after':'air'}],
       'thicknesses':[d],
       'aperture_diameter':3.0e-3}
Es=apply_spherical_lens(E0,R1=R1,R2=R2,d=d,n_lens=n,wavelength=wl,dx=dx)
try:
    Er=apply_real_lens(E0,prescription=presc,wavelength=wl,dx=dx)
except Exception as e:
    print("apply_real_lens:",type(e).__name__,str(e)[:300]); raise SystemExit
m=(np.abs(E0)>1e-2*np.abs(E0).max())
dphi=np.angle(Es[m]/Er[m]); dphi=np.unwrap(dphi); dphi-=np.mean(dphi)
print(f"apply_spherical_lens vs apply_real_lens, same 2-surface singlet (f/6.7, d=3mm):")
print(f"  RMS phase diff over illuminated pupil = {np.std(dphi):.4f} rad = {np.std(dphi)/(2*np.pi):.4f} waves")
print(f"  PV                                    = {np.ptp(dphi):.4f} rad = {np.ptp(dphi)/(2*np.pi):.4f} waves")
print(f"  amplitude ratio spread |Es|/|Er|: {np.min(np.abs(Es[m])/np.abs(Er[m])):.4f} .. {np.max(np.abs(Es[m])/np.abs(Er[m])):.4f}")
# focus comparison
def peakz(E, zs):
    return np.array([np.max(np.abs(la.angular_spectrum_propagate(E,z,wl,dx))**2) for z in zs])
f_thin=1/((n-1)*(1/R1-1/R2)); f_thick=1/((n-1)*(1/R1-1/R2+(n-1)*d/(n*R1*R2)))
BFL=f_thick*(1-(n-1)*d/(n*R1))
zs=np.linspace(0.94*f_thin,1.06*f_thin,61)
zs_s=zs[np.argmax(peakz(Es,zs))]; zs_r=zs[np.argmax(peakz(Er,zs))]
print(f"  focus(screen) = {zs_s*1e3:.4f} mm ; focus(real_lens) = {zs_r*1e3:.4f} mm")
print(f"  thin f={f_thin*1e3:.4f} thick EFL={f_thick*1e3:.4f} BFL={BFL*1e3:.4f} mm")
