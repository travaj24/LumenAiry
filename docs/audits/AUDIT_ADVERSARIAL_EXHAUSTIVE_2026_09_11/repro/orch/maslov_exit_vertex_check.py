"""Does apply_real_lens_maslov (output at the exit vertex) carry an n*sag_last(h) defocus error
because the canonical map is fitted at the last-surface SAG?  Compare exit-plane phase vs traced
for a curved-exit biconvex singlet and a flat-exit plano-convex singlet of the same power."""
import warnings, time, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements import apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
from lumenairy.elements.lenses import surface_sag_general
wl = 1.0e-6; k0 = 2*np.pi/wl; N = 512; dx = 14e-6; ap = 5e-3
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r2 = X**2+Y**2
E0 = np.exp(-r2/(1.6e-3)**2).astype(complex)
def rx_of(R1, R2):
    return {'surfaces': [
        {'radius': R1, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
        {'radius': R2, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
        'thicknesses': [3e-3], 'aperture_diameter': ap}
cases = {'biconvex R=+100/-100 (curved exit)': rx_of(100e-3, -100e-3),
         'plano-convex R=+50/inf (flat exit)': rx_of(50e-3, np.inf)}
m = (r2 <= (0.42*ap)**2)
rho2 = (r2[m]/(ap/2)**2)
def defocus_fit(dphi):
    # remove piston, fit rho^2 and rho^4 to the wrapped-safe phase difference (small enough here?)
    A = np.column_stack([np.ones_like(rho2), rho2, rho2**2])
    c, *_ = np.linalg.lstsq(A, dphi, rcond=None); return c[1]/k0, c[2]/k0, np.std(dphi - A@c)/k0
for name, rx in cases.items():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0=time.time(); Et = apply_real_lens_traced(E0, prescription=rx, wavelength=wl, dx=dx, ray_subsample=4); tt=time.time()-t0
        t0=time.time(); Em = apply_real_lens_maslov(E0, prescription=rx, wavelength=wl, dx=dx); tm=time.time()-t0
        Ea = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx)
    for lab, E in (('maslov - traced', Em*np.conj(Et)), ('thin - traced', Ea*np.conj(Et))):
        ph = np.angle(E[m]); ph = np.unwrap(ph) if False else ph
        c2, c4, res = defocus_fit(ph)
        print(f"{name:40s} {lab:16s}: rho^2 coeff = {c2*1e6:+9.3f} um, rho^4 = {c4*1e6:+8.3f} um, residual RMS = {res*1e9:8.1f} nm")
    sag_exit = surface_sag_general(np.array([(ap/2)**2]), rx['surfaces'][1]['radius'], 0.0, None)[0]
    print(f"{'':40s} exit-surface sag at pupil edge = {sag_exit*1e6:+.3f} um   (traced {tt:.1f}s, maslov {tm:.1f}s)")
