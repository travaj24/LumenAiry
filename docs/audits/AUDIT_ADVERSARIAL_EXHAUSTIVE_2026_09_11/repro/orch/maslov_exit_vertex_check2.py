"""Unwrap-safe version: 1-D radial cut of arg(E_maslov * conj(E_traced)) for the curved-exit biconvex."""
import warnings, time, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements import apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
from lumenairy.elements.lenses import surface_sag_general
wl = 1.0e-6; k0 = 2*np.pi/wl; N = 512; dx = 14e-6; ap = 5e-3
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r2 = X**2+Y**2
E0 = np.exp(-r2/(1.6e-3)**2).astype(complex)
rx = {'surfaces': [
    {'radius': 100e-3, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
    {'radius': -100e-3, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
    'thicknesses': [3e-3], 'aperture_diameter': ap}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Et = apply_real_lens_traced(E0, prescription=rx, wavelength=wl, dx=dx, ray_subsample=4)
    Em = apply_real_lens_maslov(E0, prescription=rx, wavelength=wl, dx=dx)
    Ea = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx)
np.save(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/Em.npy", Em)
np.save(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/Et.npy", Et)
row = N//2
sel = np.abs(x) <= 0.44*ap
h = x[sel]; rho = h/(ap/2)
for lab, E in (('maslov-traced', Em[row]*np.conj(Et[row])), ('thin-traced', Ea[row]*np.conj(Et[row]))):
    ph = np.unwrap(np.angle(E[sel]))
    A = np.column_stack([np.ones_like(rho), rho, rho**2, rho**4])
    c, *_ = np.linalg.lstsq(A, ph/k0, rcond=None)
    res = ph/k0 - A@c
    print(f"{lab:14s}: tilt={c[1]*1e6:+.3f} um  rho^2={c[2]*1e6:+.3f} um  rho^4={c[3]*1e6:+.3f} um  residual RMS={np.std(res)*1e9:.1f} nm ; max |dphi/pixel| = {np.max(np.abs(np.diff(ph))):.2f} rad")
sag2 = surface_sag_general(np.array([(ap/2)**2]), -100e-3, 0.0, None)[0]
print(f"exit sag at rho=1: {sag2*1e6:+.3f} um ; predicted maslov-traced rho^2 term if fitted at the sag: {sag2*1e6:+.3f} um (n_exit=1)")
# also: amplitude comparison in the pupil (power normalisation hides scale; compare shapes)
m = r2 <= (0.44*ap)**2
am = np.abs(Em[m]); at = np.abs(Et[m])
print(f"amplitude shape correlation maslov vs traced in pupil: {np.corrcoef(am, at)[0,1]:.6f}; power ratio {np.sum(np.abs(Em)**2)/np.sum(np.abs(Et)**2):.4f}")
