"""A tilted FLAT refracting face must deviate the beam by ~(n-1)*theta (thin prism).
Compare field-frame (default) vs surface_frame=True."""
import warnings, numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements import apply_real_lens
from lumenairy.glass import get_glass_index
wl = 1.0e-6; N = 512; dx = 10e-6; k0 = 2*np.pi/wl
n = get_glass_index('N-BK7', wl)
theta = 5e-3
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
E0 = np.exp(-(X**2+Y**2)/(0.8e-3)**2).astype(complex)
fx = np.fft.fftfreq(N, dx)
def mean_angles(E):
    S = np.abs(np.fft.fft2(E))**2
    FX, FY = np.meshgrid(fx, fx)
    return (np.sum(S*FX)/S.sum()*wl, np.sum(S*FY)/S.sum()*wl)   # sin(theta_x), sin(theta_y)
for tilt in [(theta, 0.0), (0.0, theta)]:
    rx = {'surfaces': [
        {'radius': np.inf, 'conic': 0.0, 'glass_before': 'AIR', 'glass_after': 'N-BK7', 'tilt': tilt},
        {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7', 'glass_after': 'AIR'}],
        'thicknesses': [2e-3]}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ef = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx, surface_frame=False)
        Es = apply_real_lens(E0, prescription=rx, wavelength=wl, dx=dx, surface_frame=True)
    print(f"tilt={tilt}: field-frame deviation (sx, sy) = ({mean_angles(Ef)[0]*1e3:+.3f}, {mean_angles(Ef)[1]*1e3:+.3f}) mrad ;"
          f"  surface_frame=True: ({mean_angles(Es)[0]*1e3:+.3f}, {mean_angles(Es)[1]*1e3:+.3f}) mrad ;"
          f"  expected |dev| ~ (n-1)*theta = {(n-1)*theta*1e3:.3f} mrad")
