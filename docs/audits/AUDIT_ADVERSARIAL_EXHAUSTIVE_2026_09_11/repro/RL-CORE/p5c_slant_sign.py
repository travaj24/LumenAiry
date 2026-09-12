"""Probe 5c: compare the three screen coefficients against the EXACT
vertex-plane eikonal, as a function of the same transverse coordinate."""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional, sag_of, dsag_dh
from lumenairy.glass import get_glass_index
lam = 632.8e-9
n1 = 1.0; n2 = float(get_glass_index('N-BK7', lam))
rx_of = lambda R: dict(surfaces=[dict(radius=R, glass_before='AIR',
                                      glass_after='N-BK7')], thicknesses=[])
print(f"n2 = {n2:.6f};  predicted slant/paraxial error ratio n2/(n2-1) = "
      f"{n2/(n2-1):.4f}")
print(f"{'R[mm]':>7}{'h_max[mm]':>10}{'theta':>9}"
      f"{'paraxial[nm]':>14}{'slant[nm]':>12}{'eq3[nm]':>12}{'sl/par':>8}")
for R, hmax in ((100e-3, 4e-3), (50e-3, 4e-3), (30e-3, 3e-3), (20e-3, 3e-3)):
    h = np.linspace(1e-6, hmax, 6001)
    hf = np.linspace(-hmax*1.6, hmax*1.6, 24001)
    r = trace_meridional(rx_of(R), lam, hf)
    o = np.argsort(r['x'])
    W_exact = np.interp(h, r['x'][o], r['opl'][o])
    W_exact = W_exact - W_exact[0]
    sag = sag_of(h*h, R)
    g = dsag_dh(h, R); th_i = np.arctan(g)
    th_t = np.arcsin(np.sin(th_i)*n1/n2)
    out = []
    for c in ((n2-n1)*np.ones_like(h),
              n2*np.cos(th_t) - n1*np.cos(th_i),
              n2*np.cos(th_i-th_t) - n1):
        W = -(c*sag); W = W - W[0]      # screen exp(-i k0 c sag) -> OPL = -c sag
        e = W_exact - W
        out.append(np.sqrt(np.mean((e-e.mean())**2))*1e9)
    print(f"{R*1e3:7.0f}{hmax*1e3:10.2f}{th_i[-1]:9.4f}"
          f"{out[0]:14.4f}{out[1]:12.4f}{out[2]:12.4f}{out[1]/out[0]:8.4f}")
