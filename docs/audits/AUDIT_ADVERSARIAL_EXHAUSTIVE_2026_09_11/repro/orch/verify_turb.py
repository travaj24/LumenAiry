import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.elements import generate_turbulence_screen
N, dx, r0 = 256, 5e-3, 0.1
# expected variance of the code's OWN lattice without any sqrt(2): sum_k PSD(f_k) df^2, PSD = 0.023 r0^-5/3 f^-11/3 (cycles/m), DC excluded
fx = np.fft.fftfreq(N, dx); FX, FY = np.meshgrid(fx, fx); f = np.hypot(FX, FY); f[0,0] = np.inf
psd = 0.023 * r0**(-5/3) * f**(-11/3); df = 1/(N*dx)
var_lattice = np.sum(psd) * df**2
vs = [np.var(generate_turbulence_screen(N, dx, r0, seed=s)) for s in range(40)]
print(f"mean screen variance = {np.mean(vs):.3f} +- {np.std(vs)/np.sqrt(40):.3f} ; lattice sum PSD*df^2 = {var_lattice:.3f} ; ratio = {np.mean(vs)/var_lattice:.3f} (1.0 expected; 2.0 = spurious sqrt(2))")
