"""Fast singlet: traced(ray_density)+ASM(BFL) vs analytic+ASM(BFL)."""
import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy import angular_spectrum_propagate as asm
from lumenairy import raytrace as rt

ap, R, t = 1.2e-3, 5.168e-3, 1.0e-3
p = fixt.small_singlet(ap=ap, t=t, R=R)
N, dx = 1024, 1.4e-6
E = fixt.gauss(N, dx, 0.40e-3)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R2 = X**2+Y**2
surf = rt.surfaces_from_prescription(p)
def bfl(h):
    b = rt._make_bundle(x=np.array([h]), y=np.array([0.0]), L=np.array([0.0]),
                        M=np.array([0.0]), wavelength=fixt.WL)
    f = rt.trace(b, surf, fixt.WL, output_filter='last').image_rays
    t0 = -f.z[0]/f.N[0]; xv = f.x[0]+f.L[0]*t0
    return -xv/(f.L[0]/f.N[0])
BFLp, BFLm = bfl(1e-5), bfl(0.55e-3)
print('paraxial BFL %.5f mm ; marginal(h=0.55mm) BFL %.5f mm  (SA = %.1f um)'
      % (BFLp*1e3, BFLm*1e3, (BFLp-BFLm)*1e6))
def met(F, lbl, z):
    I = np.abs(F)**2; tot = I.sum(); i = np.unravel_index(np.argmax(I), I.shape)
    ee = I[R2 <= (10e-6)**2].sum()/tot
    print('  %-28s z=%.4fmm P=%.4e peak=%.4e @(%.2f,%.2f)um EE10=%.4f'
          % (lbl, z*1e3, tot, I[i], (i[1]-N/2)*dx*1e6, (i[0]-N/2)*dx*1e6, ee))
    return I[i], ee
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    A = apply_real_lens(E, prescription=p, wavelength=fixt.WL, dx=dx)
    T = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                               amplitude_model='ray_density', preserve_input_phase=True,
                               fit_radius_beam_factor=2.0,
                               on_undersample='silent', on_pool_memory='silent',
                               on_aperture_beam='silent')
    print('exit-plane: |T-A|/|A| = %.4e   P_T/P_A = %.6f'
          % (float(np.linalg.norm(T-A)/np.linalg.norm(A)),
             float((np.abs(T)**2).sum()/(np.abs(A)**2).sum())))
    for z in (BFLm, 0.5*(BFLp+BFLm), BFLp):
        met(asm(A, wavelength=fixt.WL, dx=dx, z=z), 'analytic+ASM', z)
        met(asm(T, wavelength=fixt.WL, dx=dx, z=z), 'traced_rd+ASM', z)
