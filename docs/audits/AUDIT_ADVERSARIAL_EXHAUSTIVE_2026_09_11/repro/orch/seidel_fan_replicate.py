"""Replicate the seidel_correction fan block of _apply_real_lens_impl exactly and
measure the 'correction' with and without the exit-vertex transfer."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace
from lumenairy.elements.lenses import surface_sag_general
from lumenairy.glass import get_glass_index
wl = 632.8e-9
R1, R2, R3 = 62.75e-3, -45.71e-3, -128.23e-3
rx = {'surfaces': [
        {'radius': R1, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
        {'radius': R2, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'N-SF11'},
        {'radius': R3, 'conic': 0.0, 'glass_before': 'N-SF11', 'glass_after': 'AIR'}],
      'thicknesses': [4.0e-3, 2.5e-3], 'aperture_diameter': 10e-3}
r_pupil = 5e-3; n_fan = 41
h = np.linspace(-0.9*r_pupil, 0.9*r_pupil, n_fan); z0 = np.zeros_like(h)
fan = _make_bundle(x=h, y=z0, L=z0, M=z0, wavelength=wl)
res = trace(fan, surfaces_from_prescription(rx), wl); final = res.image_rays
alive = final.alive
print("final.z of fan rays (m): min=%.3e max=%.3e  -> nonzero means rays sit on the last SAG, not the vertex plane" % (final.z[alive].min(), final.z[alive].max()))
opl_ray = final.opd[alive]; ha = h[alive]
opl_an = np.zeros_like(ha)
for s in rx['surfaces']:
    n1 = get_glass_index(s['glass_before'], wl); n2 = get_glass_index(s['glass_after'], wl)
    opl_an += (n2-n1)*surface_sag_general(ha*ha, s['radius'], s.get('conic',0.0), None)
iax = int(np.argmin(np.abs(ha)))
def corr_from(opl):
    d = opl - opl[iax]; wave_rel = -(opl_an - opl_an[iax]); return d - wave_rel
rho = ha/r_pupil; A = np.column_stack([rho**p for p in (2,4,6)])
c_bug = corr_from(opl_ray)
n_exit = get_glass_index('AIR', wl)
t_v = -final.z[alive]/final.N[alive]
c_fix = corr_from(opl_ray + n_exit*t_v)
for name, c in (("AS SHIPPED (no vertex transfer)", c_bug), ("WITH signed exit-vertex transfer", c_fix)):
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    print(f"{name}: corr RMS = {np.sqrt(np.mean(c**2))*1e9:.1f} nm ; even-poly coeffs [rho^2, rho^4, rho^6] (m) = {coef}")
print("gate is 5 nm RMS; sag3 at h=4.5mm = %.3e m" % surface_sag_general(np.array([(0.9*r_pupil)**2]), R3, 0.0, None)[0])
