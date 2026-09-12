"""Probe 13: does the missing sqrt(J) show up as a SPATIAL amplitude-profile
error that normalize_output='power' cannot remove?"""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
warnings.simplefilter("ignore")

lam, N, dx = 1.0e-6, 64, 4.0e-6
x = (np.arange(N)-N/2)*dx; X,Y = np.meshgrid(x,x)
def gap(z, ap):
    return {'name':'gap','aperture_diameter':ap,
            'surfaces':[{'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'},
                        {'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'}],
            'thicknesses':[z]}

# A: converging input through a free-space gap -> the ray-tube Jacobian |ds1/dv2|
#    becomes strongly position-dependent when the beam converges.
w0 = 40e-6; k = 2*np.pi/lam
for Rc, z in ((-2.0e-3, 1.0e-3), (-1.2e-3, 1.0e-3)):
    E0 = (np.exp(-(X**2+Y**2)/w0**2)*np.exp(1j*k*(X**2+Y**2)/(2*Rc))).astype(complex)
    ref = angular_spectrum_propagate(E0, z, lam, dx)
    Em = np.asarray(la.apply_real_lens_maslov(E0.copy(), prescription=gap(z, N*dx*0.9),
            wavelength=lam, dx=dx, normalize_output='power',
            integration_method='quadrature', n_v2=96, poly_order=5,
            ray_field_samples=12, ray_pupil_samples=12))
    # power-normalise the reference the same way
    ref_n = ref*np.sqrt(np.sum(np.abs(E0)**2)/np.sum(np.abs(ref)**2))
    m = np.abs(ref_n) > 0.1*np.abs(ref_n).max()
    r = np.abs(Em[m])/np.abs(ref_n[m])
    print(f"R_in={Rc*1e3:+.1f}mm z={z*1e3:.1f}mm: |E|maslov/|E|exact  mean={r.mean():.4f}"
          f"  min={r.min():.4f} max={r.max():.4f}  rel-spread={r.std()/r.mean():.4f}")
