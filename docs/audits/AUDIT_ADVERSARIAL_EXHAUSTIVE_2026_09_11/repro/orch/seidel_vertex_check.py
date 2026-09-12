"""Does seidel_correction=True inject a spurious defocus because the 1-D fan's
ray OPL is read at the last-surface SAG instead of the exit VERTEX plane?"""
import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.elements import apply_real_lens

wl = 632.8e-9; k0 = 2*np.pi/wl
N = 1024; dx = 20e-6
# AC254-100-like cemented doublet (stand-in glasses)
R1, R2, R3 = 62.75e-3, -45.71e-3, -128.23e-3
t1, t2 = 4.0e-3, 2.5e-3
ap = 10e-3
rx = {'surfaces': [
        {'radius': R1, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7'},
        {'radius': R2, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'N-SF11'},
        {'radius': R3, 'conic': 0.0, 'glass_before': 'N-SF11', 'glass_after': 'AIR'}],
      'thicknesses': [t1, t2], 'aperture_diameter': ap}
x = (np.arange(N) - N/2)*dx; X, Y = np.meshgrid(x, x); r2 = X**2 + Y**2
E0 = np.ones((N, N), complex)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ea = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx, seidel_correction=False)
    Eb = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx, seidel_correction=True)
m = (r2 <= (0.45*ap)**2) & (np.abs(Ea) > 1e-3)
dphi = np.angle(Eb[m]*np.conj(Ea[m]))          # correction phase, wrapped
# unwrap radially via a fit on the wrapped data is unsafe; instead fit exp form:
# fit dOPD = c2*rho^2 + c4*rho^4 + c6*rho^6 by least squares on the UNWRAPPED 1-D cut
row = Eb[N//2]*np.conj(Ea[N//2]); mrow = m[N//2]
ph = np.unwrap(np.angle(row[mrow])); hh = x[mrow]
rho = hh/(ap/2)
A = np.column_stack([rho**2, rho**4, rho**6])
c, *_ = np.linalg.lstsq(A, ph/k0 - (ph/k0)[np.argmin(np.abs(hh))], rcond=None)
print("Seidel correction OPD polynomial (metres at pupil edge rho=1): c2=%.3e  c4=%.3e  c6=%.3e" % tuple(c))
print("Predicted missing exit-vertex term at h=5mm: -sag3 = h^2/(2|R3|) = %.3e m" % ((0.5*ap)**2/(2*abs(R3))))
# focal-power impact: a c2*rho^2 OPD term is a defocus; equivalent focal length f_eq = -(ap/2)^2/(2*c2)
print("Equivalent added lens power from c2: 1/f = %.4f 1/m  (f_eq = %.3f m)" % (-2*c[0]/(0.5*ap)**2, -(0.5*ap)**2/(2*c[0]) if c[0] else np.inf))
# Also find the paraxial BFL from the library's ABCD for context
from lumenairy.raytrace import surfaces_from_prescription, system_abcd
out = system_abcd(surfaces_from_prescription(rx), wl)
print("system_abcd ->", out if not hasattr(out, '__len__') else [float(v) if np.isscalar(v) else v for v in out][:4])
