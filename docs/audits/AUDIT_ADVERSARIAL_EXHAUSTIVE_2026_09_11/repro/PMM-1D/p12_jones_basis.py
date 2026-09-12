"""PROBE 12: the 1-D Jones BASIS.  CONVENTIONS.md 7.1 says "the 1-D solvers
return te/tm (s/p)"; pmm_jones_1d's own docstring says lab (E_x, E_y).  The
two differ by the sign of the p unit vector.  Which is it, and do the siblings
(rcwa_jones_1d, berreman_jones_1d) agree with PMM?
"""
import sys, os, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import tmm_slab
from lumenairy.elements.pmm import pmm_jones_1d
from lumenairy.elements.rcwa import rcwa_jones_1d
from lumenairy.elements.berreman import berreman_jones_1d

wl, per, depth = 0.55e-6, 0.4e-6, 0.32e-6
nl, nsub, nsup = 2.1, 1.5, 1.0
eps = nl ** 2 * np.eye(3)

print("=== UNIFORM slab: every 1-D Jones entry vs the analytic Fresnel ===")
print(f"{'ang':>5} {'solver':>14} {'J[0,0] (x/tm)':>30} {'J[1,1] (y/te)':>30}")
for angd in (0.0, 30.0, 60.0):
    ang = np.deg2rad(angd)
    rs, _, _, _ = tmm_slab(nsup, nl, nsub, depth, wl, ang, 's')
    rp, _, _, _ = tmm_slab(nsup, nl, nsub, depth, wl, ang, 'p')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, _R, _T, Jp = pmm_jones_1d(per, eps, eps, nsub, nsup, depth, 0.5,
                                      wl, angle=ang, degree=12,
                                      far_field_orders=11, stabilize=False)
        _o, _R, _T, Jr = rcwa_jones_1d(per, eps, eps, nsub, nsup, depth, 0.5,
                                       wl, angle=ang, n_orders=11)
    print(f"{angd:5.0f} {'analytic r_p':>14} {rp:30.12f} {'(r_s)':>30}")
    print(f"{angd:5.0f} {'analytic r_s':>14} {'':>30} {rs:30.12f}")
    print(f"{angd:5.0f} {'pmm_jones_1d':>14} {Jp[0,0]:30.12f} {Jp[1,1]:30.12f}")
    print(f"{angd:5.0f} {'rcwa_jones_1d':>14} {Jr[0,0]:30.12f} {Jr[1,1]:30.12f}")
    try:
        Jb = berreman_jones_1d([(nl, depth)], wl, angle=ang,
                               n_substrate=nsub, n_superstrate=nsup)
        Jb = np.asarray(Jb)
        if Jb.ndim == 3:
            Jb = Jb[0]
        print(f"{angd:5.0f} {'berreman':>14} {Jb[0,0]:30.12f} "
              f"{Jb[1,1]:30.12f}")
    except Exception as e:
        print(f"{angd:5.0f} {'berreman':>14}  N/A: {type(e).__name__} "
              f"{str(e)[:70]}")
    print(f"      ratio pmm/analytic: xx {Jp[0,0]/rp:+.6f}  "
          f"yy {Jp[1,1]/rs:+.6f}   |  rcwa/pmm: xx {Jr[0,0]/Jp[0,0]:+.6f} "
          f"yy {Jr[1,1]/Jp[1,1]:+.6f}")
    print()
