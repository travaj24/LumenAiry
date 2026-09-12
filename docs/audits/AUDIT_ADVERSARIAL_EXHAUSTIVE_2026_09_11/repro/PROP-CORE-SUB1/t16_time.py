import sys, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import propagate_huygens_fresnel_with_opl_callable as F
lam, z, dx = 633e-9, 200e-6, 2e-6
def opl(a,b,c,d): return np.sqrt((a-c)**2+(b-d)**2+z*z)/lam
# warm-up (pays one-time backend init)
w = np.ones((8,8), complex); g = np.zeros(2)
t0=time.perf_counter(); F(w, opl_fn=opl, output_grid_x=g, output_grid_y=g, input_grid_dx=dx); print(f"warm-up (8x8->2x2) incl. one-time init: {time.perf_counter()-t0:.3f}s")
print(f"{'N_in':>6}{'N_out':>7}{'t (s)':>10}{'s / (Nin^2*Nout^2)':>22}")
for (Ni,No) in ((32,8),(48,8),(64,8),(48,16),(64,16)):
    E = np.ones((Ni,Ni), complex); go=(np.arange(No)-No/2)*dx
    t0=time.perf_counter(); F(E, opl_fn=opl, output_grid_x=go, output_grid_y=go, input_grid_dx=dx); t=time.perf_counter()-t0
    print(f"{Ni:>6}{No:>7}{t:>10.3f}{t/(Ni**2*No**2):>22.3e}")
t128 = 0.0
