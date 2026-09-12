import fixt, numpy as np, warnings, time
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy import angular_spectrum_propagate as asm
from lumenairy import raytrace as rt

p = fixt.small_singlet()
N, dx = 512, 4e-6
w0 = 0.30e-3
E = fixt.gauss(N, dx, w0)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R2 = X**2 + Y**2

surf = rt.surfaces_from_prescription(p)
h0 = 1e-5
bun = rt._make_bundle(x=np.array([h0]), y=np.array([0.0]),
                      L=np.array([0.0]), M=np.array([0.0]), wavelength=fixt.WL)
fin = rt.trace(bun, surf, fixt.WL, output_filter='last').image_rays
# propagate to the vertex plane first (exit-vertex correction), then to focus
t0 = -fin.z[0]/fin.N[0]
xv = fin.x[0] + fin.L[0]*t0
BFL = -xv / (fin.L[0] / fin.N[0])
print('paraxial marginal: x_vertex=%.6e  L=%.6e  BFL = %.6f mm' % (xv, fin.L[0], BFL*1e3))

def metrics(F, lbl):
    I = np.abs(F)**2
    tot = I.sum()
    i = np.unravel_index(np.argmax(I), I.shape)
    pk = I[i]
    # encircled energy in 25 um
    ee = I[R2 <= (25e-6)**2].sum()/tot if tot > 0 else 0.0
    r2m = np.sqrt((I*R2).sum()/tot) if tot>0 else 0.0
    print('  %-34s P=%.4e  peak=%.4e @ (%d,%d)=(%.1f,%.1f)um  EE25=%.4f  rms_r=%.2f um'
          % (lbl, tot, pk, i[0], i[1], (i[1]-N/2)*dx*1e6, (i[0]-N/2)*dx*1e6, ee, r2m*1e6))
    return tot, pk, ee

print('P_in = %.4e' % (np.abs(E)**2).sum())
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    Aex = apply_real_lens(E, prescription=p, wavelength=fixt.WL, dx=dx)
    Awave = asm(Aex, wavelength=fixt.WL, dx=dx, z=BFL)
    Tex = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                 amplitude_model='ray_density',
                                 preserve_input_phase=True,
                                 on_undersample='silent', on_pool_memory='silent',
                                 on_aperture_beam='silent')
    Twave = asm(Tex, wavelength=fixt.WL, dx=dx, z=BFL)
    t=time.perf_counter()
    Tmb = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                 amplitude_model='ray_density', caustic='multibranch',
                                 output_plane_distance=BFL, caustic_ray_subsample=4,
                                 on_undersample='silent')
    tmb = time.perf_counter()-t
    Tmb0 = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                  amplitude_model='ray_density', caustic='multibranch',
                                  output_plane_distance=0.0, caustic_ray_subsample=4,
                                  on_undersample='silent')
    Tmbp = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                  amplitude_model='ray_density', caustic='multibranch',
                                  output_plane_distance=BFL, caustic_ray_subsample=4,
                                  caustic_band='plain', on_undersample='silent')
print('multibranch wall %.2f s' % tmb)
metrics(Aex, 'analytic @ exit vertex')
metrics(Awave, 'analytic + ASM(BFL)')
metrics(Tex, 'traced ray_density @ vertex')
metrics(Twave, 'traced rd + ASM(BFL)')
metrics(Tmb0, 'multibranch @ vertex (opd=0)')
metrics(Tmb,  'multibranch @ BFL (ludwig)')
metrics(Tmbp, 'multibranch @ BFL (plain)')
seen=set()
for w in W:
    s = w.category.__name__+': '+str(w.message)[:150]
    if s not in seen: seen.add(s); print('  WARN', s)
