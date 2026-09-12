import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
np.set_printoptions(precision=6, suppress=False, linewidth=200)
import lumenairy as lm
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d, rcwa_jones_1d

wl = 1.0e-6
Px = Py = 0.5e-6
dep = 0.3e-6
nsup, nsub = 1.0, 1.5

def iso(e):
    return e*np.eye(3, dtype=complex)

# ---------- (a) UNPATTERNED layer vs coatings TMM -------------------
print("=== (a) uniform layer limit vs TMM ===")
S = 4
cell = np.empty((S, S, 3, 3), dtype=complex)
cell[...] = iso(4.0)
for th in (0.0, np.deg2rad(30.0)):
    o, R, T, J = pmm_jones_2d(Px, Py, cell, nsub, nsup, dep, wl,
                              theta=th, phi=0.0, degree=5, n_orders=3)
    from lumenairy.coatings import stack_rt  # may not exist; try
    print("theta", np.rad2deg(th), "J=\n", J)
