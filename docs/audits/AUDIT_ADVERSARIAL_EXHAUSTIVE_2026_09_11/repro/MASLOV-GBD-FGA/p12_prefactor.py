import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
warnings.simplefilter("ignore")
def gap(z, ap):
    return {'name':'gap','aperture_diameter':ap,
            'surfaces':[{'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'},
                        {'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'}],
            'thicknesses':[z]}
print("If the Maslov integrand amplitude is |det ds1/dv2| (no sqrt) and there is no")
print("1/(i*lambda) prefactor, then for free space  E_maslov / E_exact = i*lambda*z.")
print("If it were sqrt|det| (correct), the ratio would be i*lambda (z-independent).\n")
for N,dx in ((64,4.0e-6),):
  x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
  for lam in (1.0e-6,2.0e-6):
    for z in (0.5e-3,1.0e-3,2.0e-3):
      w0=30e-6
      E0=np.exp(-(X**2+Y**2)/w0**2).astype(complex)
      Em=np.asarray(la.apply_real_lens_maslov(E0.copy(),prescription=gap(z,N*dx*0.9),
          wavelength=lam,dx=dx,normalize_output='none',integration_method='quadrature',
          n_v2=64,input_na=0.06,poly_order=4,ray_field_samples=12,ray_pupil_samples=12))
      ref=angular_spectrum_propagate(E0,z,lam,dx)
      m=np.abs(ref)>0.2*np.abs(ref).max()
      r=Em[m]/ref[m]; a=float(np.mean(np.abs(r)))
      print(f"  lam={lam*1e6:.1f}um z={z*1e3:.2f}mm N={N} dx={dx*1e6:.1f}um: "
            f"|ratio|={a:.5e}  spread={float(np.std(np.abs(r))/a):.1e}  "
            f"arg={float(np.angle(np.mean(r))):+.4f}  ratio/(lam*z)={a/(lam*z):.5f}  ratio/lam={a/lam:.4e}")
