"""Probe 4b: is the 'gradient' estimator bias really only sin(h)/h?"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import carrier_referenced_fit_radius
wl = 1.31e-6; k = 2*np.pi/wl
w = 100e-6
print(" dx/w      h        sin(h)/h   1/(sin(h)/h)-1   measured bias   (dx/w)^2")
for R in (0.5,):
    for N, dx in ((2048, 0.5e-6), (1024, 1e-6), (512, 2e-6), (256, 4e-6), (128, 8e-6)):
        x = (np.arange(N)-N/2)*dx
        X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2+Y**2
        E = np.exp(-r2/w**2)*np.exp(1j*k*r2/(2*R))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Rf = carrier_referenced_fit_radius(E, wl, dx)
            Ri = carrier_referenced_fit_radius(E, wl, dx, estimator='increment')
        h = k*dx*w/abs(R)
        print(f" {dx/w:6.3f}  {h:9.2e}  {np.sin(h)/h:.8f}   {1/(np.sin(h)/h)-1:.3e}    "
              f"{Rf/R-1:.4e}  (incr {Ri/R-1:+.2e})   {(dx/w)**2:.3e}")
