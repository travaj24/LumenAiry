"""Probe 8: (a) Maslov 'none' prefactor vs exact free space; (b) local_quadrature
vs quadrature on a REAL focusing chart."""
import numpy as np, sys, warnings, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
warnings.simplefilter("ignore")

def gap(z, ap):
    return {'name':'gap','aperture_diameter':ap,
            'surfaces':[{'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'},
                        {'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'}],
            'thicknesses':[z]}

print("### (a) prefactor: Maslov normalize_output='none' vs exact ASM, free-space chart")
N, dx = 64, 4.0e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
for lam in (1.0e-6, 2.0e-6):
  for z in (0.5e-3, 1.0e-3, 2.0e-3):
    w0 = 30e-6
    E0 = np.exp(-(X**2+Y**2)/w0**2).astype(complex)
    Em = np.asarray(la.apply_real_lens_maslov(E0.copy(), prescription=gap(z, N*dx*0.9),
            wavelength=lam, dx=dx, normalize_output='none',
            integration_method='quadrature', n_v2=48, input_na=0.05,
            poly_order=4, ray_field_samples=12, ray_pupil_samples=12))
    ref = angular_spectrum_propagate(E0, z, lam, dx)
    m = np.abs(ref) > 0.2*np.abs(ref).max()
    r = Em[m]/ref[m]
    a = float(np.mean(np.abs(r))); sd = float(np.std(np.abs(r))/a)
    print(f"  lam={lam*1e6:.1f}um z={z*1e3:.2f}mm  |ratio|={a:.5e} spread={sd:.1e} "
          f"arg={float(np.angle(np.mean(r))):+.4f}  ratio/(lam*z)={a/(lam*z):.5f}  ratio/lam={a/lam:.4e}")

print("\n### (b) local_quadrature vs quadrature on a focusing singlet chart")
N, dx, lam = 96, 3.0e-6, 1.0e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
E0 = np.exp(-(X**2+Y**2)/(80e-6)**2).astype(complex)
presc = la.make_singlet(2.0e-3, -2.0e-3, 0.6e-3, 'N-BK7', aperture=0.22e-3)
kw = dict(prescription=presc, wavelength=lam, dx=dx, normalize_output='none',
          collimated_input=True, poly_order=5, ray_field_samples=12, ray_pupil_samples=12)
for opd in (0.0, 1.0e-3, 1.6e-3):
    res = {}
    for meth, extra in (('quadrature', dict(n_v2=256)),
                        ('local_quadrature', dict()),
                        ('local_quadrature(n=48,ws=8)', dict(local_n_samples=48, local_window_sigma=8.0)),
                        ('stationary_phase', dict())):
        mm = meth.split('(')[0]
        t = time.perf_counter()
        E = np.asarray(la.apply_real_lens_maslov(E0.copy(), integration_method=mm,
                                                 output_plane_distance=opd, **kw, **extra))
        res[meth] = (E, time.perf_counter()-t)
    ref = res['quadrature'][0]
    m = np.abs(ref) > 0.05*np.abs(ref).max()
    print(f"  output_plane_distance={opd*1e3:.2f} mm  (inbox pixels={m.sum()})")
    for meth,(E,t) in res.items():
        if meth == 'quadrature':
            print(f"     {meth:28s}: reference, {t:.1f}s, peak|E|={np.abs(E).max():.4e}")
            continue
        rel = np.linalg.norm(E[m]-ref[m])/np.linalg.norm(ref[m])
        amp = np.mean(np.abs(E[m]))/np.mean(np.abs(ref[m]))
        print(f"     {meth:28s}: relL2={rel:.3e}  mean|E| ratio={amp:.4f}  {t:.1f}s")
