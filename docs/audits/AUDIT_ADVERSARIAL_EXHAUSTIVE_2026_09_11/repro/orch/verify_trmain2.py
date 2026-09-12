import warnings, numpy as np, sys, traceback
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
wl = 1.31e-6; N = 256; dx = 8e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r2 = X**2+Y**2
rx = {'surfaces': [
    {'radius': 26e-3, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
    {'radius': -26e-3, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
    'thicknesses': [2e-3], 'aperture_diameter': 1.6e-3}
E_real = np.exp(-r2/(0.5e-3)**2)            # float64 field
print("1) REAL-dtype E_in:")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        out = apply_real_lens_traced(E_real, prescription=rx, wavelength=wl, dx=dx, ray_subsample=4)
        print("   traced OK ->", out.dtype)
    except Exception as e:
        print("   traced RAISED", type(e).__name__, ":", str(e)[:100])
    out = apply_real_lens(E_real, prescription=rx, wavelength=wl, dx=dx); print("   analytic OK ->", out.dtype)
print("2) newton_fit='spline' with a vignetting semi_diameter on surface 2:")
rx2 = {**rx, 'surfaces': [dict(rx['surfaces'][0]), dict(rx['surfaces'][1], semi_diameter=0.6e-3)]}
E0 = E_real.astype(complex)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for fit in ('polynomial', 'spline'):
        try:
            o = apply_real_lens_traced(E0, prescription=rx2, wavelength=wl, dx=dx, ray_subsample=4, newton_fit=fit, on_undersample='silent')
            print(f"   {fit:10s}: P_out/P_in = {np.sum(np.abs(o)**2)/np.sum(np.abs(E0)**2):.4f}, nonzero px = {np.count_nonzero(o)}")
        except Exception as e:
            print(f"   {fit:10s}: RAISED {type(e).__name__}: {str(e)[:120]}")
print("3) caustic='multibranch' at the paraxial focus:")
from lumenairy.raytrace import surfaces_from_prescription, system_abcd
Mabcd, efl, bfl, ffl = system_abcd(surfaces_from_prescription(rx), wl)
print(f"   BFL = {bfl*1e3:.4f} mm")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    try:
        o = apply_real_lens_traced(E0, prescription=rx, wavelength=wl, dx=dx, ray_subsample=4,
                                   amplitude_model='ray_density', caustic='multibranch', output_plane_distance=bfl)
        print(f"   P_out/P_in = {np.sum(np.abs(o)**2)/np.sum(np.abs(E0)**2):.3e}, nonzero px = {np.count_nonzero(o)}, warnings = {len(w)}")
    except Exception as e:
        print("   RAISED", type(e).__name__, ":", str(e)[:200])
