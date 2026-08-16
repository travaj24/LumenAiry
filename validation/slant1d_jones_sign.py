import numpy as np, lumenairy
from lumenairy.elements.pmm import pmm_jones_1d_slanted, pmm_jones_1d
np.set_printoptions(precision=6, linewidth=220)
P, WL, D = 1.0e-6, 633e-9, 300e-9
ER, EG = 4.0+0j, 1.0+0j
def run(phi, theta, fac='convection'):
    return pmm_jones_1d_slanted(P, ER*np.eye(3), EG*np.eye(3), 1.5, 1.0, D, 0.45, WL,
                                phi, angle=theta, degree=12, far_field_orders=11,
                                factorization=fac)
for theta in (0.0, 0.25):
    print(f"===== REAL GRATING theta={theta} =====")
    J0 = np.asarray(run(0.0, theta)[3])
    Jv = np.asarray(pmm_jones_1d(P, ER*np.eye(3), EG*np.eye(3), 1.5, 1.0, D, 0.45, WL,
                                 angle=theta, degree=12, far_field_orders=11)[3])
    print("slant-solver phi=0 J00,J11 =", J0[0,0], J0[1,1])
    print("pmm_jones_1d       J00,J11 =", Jv[0,0], Jv[1,1])
    for pd in (1e-6, 1e-3, 1.0, 5.0, 20.0):
        J = np.asarray(run(np.radians(pd), theta)[3])
        print(f"  phi={pd:<8g} J00/J0_00={J[0,0]/J0[0,0]:+.6f}  J11/J0_11={J[1,1]/J0[1,1]:+.6f}")
