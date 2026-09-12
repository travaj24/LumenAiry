"""Probe 8b: exact field-frame height of a RIGID-BODY rotated sphere vs what
surface_frame=True evaluates.  Pure geometry -- no ray trace needed."""
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements.lenses import surface_sag_general as sg
lam = 632.8e-9; n = 1.5150891983370924
R = 50e-3
x = np.linspace(-2e-3, 2e-3, 9)
for th in (1e-3, 5e-3, 2e-2):
    # exact: sphere centre moves from (0,0,R) to (R sin th, 0, R cos th)
    z_exact = R*np.cos(th) - np.sqrt(R**2 - (x - R*np.sin(th))**2)
    # what the code evaluates: sag(cos(th)*x, y=0)  [tilt = (0, th)]
    z_code = sg((np.cos(th)*x)**2, R, 0.0, None)
    # first-order truth = sag(x) - th*x
    z_fo = sg(x*x, R, 0.0, None) - th*x
    d = z_exact - z_code
    print(f"th={th*1e3:5.1f} mrad:  max|z_exact - z_code| = {np.abs(d).max()*1e6:9.3f} um"
          f"  -> OPD (n-1)*d = {np.abs(d).max()*(n-1)*1e6:8.3f} um = "
          f"{np.abs(d).max()*(n-1)/lam:8.2f} waves")
    print(f"               max|z_exact - (sag - th*x)| = "
          f"{np.abs(z_exact-z_fo).max()*1e9:9.4f} nm   "
          f"(the MISSING term is exactly the -th*x ramp)")
