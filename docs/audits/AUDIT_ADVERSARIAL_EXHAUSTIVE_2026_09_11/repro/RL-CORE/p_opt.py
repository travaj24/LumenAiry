import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
lam = 632.8e-9
bad = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                     dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-SF11'),
                     dict(radius=-200e-3, glass_before='N-SF11', glass_after='AIR')],
           thicknesses=[3e-3])
try:
    E = apply_real_lens(np.ones((64,64), complex), prescription=bad, wavelength=lam, dx=1e-5)
    print("3 surfaces / 1 thickness ACCEPTED; finite:", bool(np.isfinite(E).all()),
          "mean|E| =", float(np.abs(E).mean()))
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:80])
bad2 = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                      dict(radius=float('inf'), glass_before='N-BK7', glass_after='AIR')],
            thicknesses=[3e-3, 9e-3])
try:
    E = apply_real_lens(np.ones((64,64), complex), prescription=bad2, wavelength=lam, dx=1e-5)
    print("2 surfaces / 2 thicknesses ACCEPTED; finite:", bool(np.isfinite(E).all()))
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:80])
