import warnings, numpy as np, sys, inspect
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.analysis.through_focus import diffraction_limited_peak
from lumenairy.analysis.opd import wave_opd_2d
zernike = None
from lumenairy.propagators.propagation import angular_spectrum_propagate
print("diffraction_limited_peak signature:", inspect.signature(diffraction_limited_peak))
# (1) perfect spherical wave, f/2.5: D=400um, f=1.0mm, lam=600nm, N=1024, dx=0.5um
wl=600e-9; k0=2*np.pi/wl; N=1024; dx=0.5e-6; D=400e-6; f=1.0e-3
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r2=X**2+Y**2
pup=(r2<=(D/2)**2).astype(float)
E=pup*np.exp(-1j*k0*(np.sqrt(r2+f*f)-f))          # exact converging sphere (aberration-free)
try:
    ref=diffraction_limited_peak(E, wl, f, dx)
except TypeError:
    ref=diffraction_limited_peak(E, dx=dx, wavelength=wl, f=f)
zs=np.linspace(0.97*f,1.03*f,31); pk=max(np.max(np.abs(angular_spectrum_propagate(E,z,wl,dx))**2) for z in zs)
W040=(D/wl)/(128*(f/D)**3)
print(f"(1) perfect sphere f/{f/D:.1f}: actual peak / 'diffraction-limited' reference = {pk/ref:.3f}  (should be 1.000; predicted W040 of the quadratic ref = {W040:.2f} waves)")
# (2) wave_opd_2d on a flat pupil with pure coma (OSA j=8) at 0.5 waves rms
N2=512; dx2=1e-6; ap=400e-6; x2=(np.arange(N2)-N2/2)*dx2; X2,Y2=np.meshgrid(x2,x2); r2b=X2**2+Y2**2
rho=np.sqrt(r2b)/(ap/2); th=np.arctan2(Y2,X2)
coma = None
if coma is None:
    coma = (3*rho**3-2*rho)*np.cos(th)*np.sqrt(8)   # OSA-normalised vertical/horizontal coma
W = 0.5*wl*coma*(r2b<=(ap/2)**2)
E2=(r2b<=(ap/2)**2)*np.exp(1j*k0*W)
try:
    out=wave_opd_2d(E2, dx2, wl, aperture=ap)
except TypeError:
    out=wave_opd_2d(E2, dx=dx2, wavelength=wl, aperture=ap)
opd = out[2]
m=(r2b<=(0.95*ap/2)**2)&np.isfinite(opd)
err=(opd[m]-W[m]); err-=np.median(err)
print(f"(2) wave_opd_2d, 0.5 waves rms coma: max |OPD error| = {np.max(np.abs(err))/wl:.3f} waves ; fraction of pupil wrong by >0.4 waves = {np.mean(np.abs(err)>0.4*wl)*100:.1f}%  (max dphi/sample = {np.max(np.abs(np.diff(k0*W,axis=1))[m[:,1:]]):.2f} rad)")
