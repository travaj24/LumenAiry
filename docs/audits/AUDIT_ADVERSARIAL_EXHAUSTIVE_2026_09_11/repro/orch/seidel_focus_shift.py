"""Physical consequence: where does the beam focus with/without seidel_correction?"""
import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements import apply_real_lens
from lumenairy.propagators.propagation import angular_spectrum_propagate
from lumenairy.raytrace import surfaces_from_prescription, system_abcd
wl = 632.8e-9; N = 1024; dx = 20e-6
R1, R2, R3 = 62.75e-3, -45.71e-3, -128.23e-3
rx = {'surfaces': [
        {'radius': R1, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
        {'radius': R2, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'N-SF11'},
        {'radius': R3, 'conic': 0.0, 'glass_before': 'N-SF11', 'glass_after': 'AIR'}],
      'thicknesses': [4.0e-3, 2.5e-3], 'aperture_diameter': 10e-3}
E0 = np.ones((N, N), complex)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ea = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx)
    Eb = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx, seidel_correction=True)
M, efl, bfl, ffl = system_abcd(surfaces_from_prescription(rx), wl)
print(f"paraxial EFL={efl*1e3:.3f} mm  BFL={bfl*1e3:.3f} mm")
zs = np.linspace(0.030, 0.130, 51)
def peak_curve(E):
    return np.array([np.max(np.abs(angular_spectrum_propagate(E, z, wl, dx))**2) for z in zs])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    pa = peak_curve(Ea); pb = peak_curve(Eb)
print(f"seidel_correction=False : peak intensity at z = {zs[np.argmax(pa)]*1e3:.1f} mm")
print(f"seidel_correction=True  : peak intensity at z = {zs[np.argmax(pb)]*1e3:.1f} mm")
