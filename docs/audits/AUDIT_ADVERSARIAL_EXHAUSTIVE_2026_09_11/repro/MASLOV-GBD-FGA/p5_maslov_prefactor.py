"""Probe 5: Maslov amplitude prefactor / Jacobian power, on an EXACT oracle
(free-space propagation expressed as a flat air 'lens')."""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate

def freespace_presc(z, ap):
    return {'name': 'air gap', 'aperture_diameter': ap,
            'surfaces': [
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'air', 'glass_after': 'air'}],
            'thicknesses': [z]}

N, dx = 256, 2.0e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)

def run(z, lam, w0=40e-6, na=0.06, method='quadrature', n_v2=192):
    E0 = np.exp(-(X**2+Y**2)/w0**2).astype(complex)
    presc = freespace_presc(z, N*dx*0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Em = np.asarray(la.apply_real_lens_maslov(
            E0.copy(), prescription=presc, wavelength=lam, dx=dx,
            normalize_output='none', integration_method=method,
            n_v2=n_v2, input_na=na, poly_order=4,
            ray_field_samples=14, ray_pupil_samples=14))
    ref = angular_spectrum_propagate(E0, z, lam, dx)
    m = np.abs(ref) > 0.1*np.abs(ref).max()
    r = Em[m]/ref[m]
    return float(np.mean(np.abs(r))), float(np.std(np.abs(r))/np.mean(np.abs(r))), \
           float(np.angle(np.mean(r))), Em, ref

print("prediction if amplitude uses |J| (not sqrt|J|) and no 1/(i*lambda) prefactor:")
print("   |E_maslov| / |E_exact|  =  lambda * z     (and arg = -pi/2)")
print()
for lam in (1.0e-6, 2.0e-6):
    for z in (0.5e-3, 1.0e-3, 2.0e-3):
        a, sd, ph, _, _ = run(z, lam)
        print(f"lam={lam*1e6:.1f}um z={z*1e3:.2f}mm: |ratio|={a:.6e}  rel-spread={sd:.2e}"
              f"  arg={ph:+.4f} rad   lambda*z={lam*z:.6e}   ratio/(lam*z)={a/(lam*z):.6f}"
              f"   ratio/lambda={a/lam:.6e}")
