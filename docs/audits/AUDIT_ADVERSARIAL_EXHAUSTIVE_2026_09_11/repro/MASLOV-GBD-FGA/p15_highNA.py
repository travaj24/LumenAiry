"""Probe 15: at higher NA the missing sqrt(J) re-weights the ANGULAR SPECTRUM
inside the integral -> a shape error that normalize_output='power' cannot fix.
Free-space chart:  J = z^2/N^4,  sqrt(J) = z/N^2  (N = sqrt(1-|v2|^2))."""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
warnings.simplefilter("ignore")
lam = 1.0e-6
def gap(z, ap):
    return {'name':'gap','aperture_diameter':ap,
            'surfaces':[{'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'},
                        {'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'}],
            'thicknesses':[z]}
for N, dx, w0, z in ((128, 0.5e-6, 1.2e-6, 6.0e-6),      # NA ~ 0.27
                     (128, 0.5e-6, 2.5e-6, 2.0e-5),      # NA ~ 0.13
                     (128, 1.0e-6, 8.0e-6, 1.5e-4)):     # NA ~ 0.04
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
    E0=np.exp(-(X**2+Y**2)/w0**2).astype(complex)
    na_g = lam/(np.pi*w0)
    ref=angular_spectrum_propagate(E0,z,lam,dx)
    Em=np.asarray(la.apply_real_lens_maslov(E0.copy(),prescription=gap(z,N*dx*0.95),
        wavelength=lam,dx=dx,normalize_output='power',integration_method='quadrature',
        n_v2=96,poly_order=6,ray_field_samples=12,ray_pupil_samples=14,input_na=3.5*na_g))
    refn=ref*np.sqrt(np.sum(np.abs(E0)**2)/np.sum(np.abs(ref)**2))
    m=np.abs(refn)>0.05*np.abs(refn).max()
    r=np.abs(Em[m])/np.abs(refn[m])
    rel=np.linalg.norm(np.abs(Em[m])-np.abs(refn[m]))/np.linalg.norm(np.abs(refn[m]))
    print(f"waist={w0*1e6:.2f}um (NA~{na_g:.3f})  z={z*1e6:.1f}um: "
          f"|E| ratio mean={r.mean():.4f} min={r.min():.4f} max={r.max():.4f} "
          f"spread={r.std()/r.mean():.4f}  rel |E| err={rel:.4f}")
