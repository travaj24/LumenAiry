"""PROBE 5: energy conservation + reciprocity for a lossless Si/SiO2 lamellar
grating.  Also TE/TM labelling cross-check against the library's own RCWA.
"""
import sys, os
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
np.set_printoptions(precision=12, suppress=False, linewidth=200)
from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

wl = 1.55e-6
period = 1.0e-6
depth = 0.45e-6
duty = 0.5
n_si, n_ox = 3.48, 1.444
n_sup, n_sub = 1.0, 1.444

print("=== 5a: energy  sum(R)+sum(T)-1  for lossless Si/SiO2 lamellar ===")
print(f"{'ang':>5} {'pol':>4} {'deg':>4} {'sumR':>16} {'sumT':>16} {'tot-1':>12}")
for ang_deg in (0.0, 17.0, 45.0):
    for pol in ("te", "tm"):
        for deg in (16, 24, 32):
            o, R, T = pmm_efficiency_1d(
                period, n_si, n_ox, n_sub, n_sup, depth, duty, wl,
                angle=np.deg2rad(ang_deg), polarization=pol, degree=deg,
                far_field_orders=15, stabilize=False)
            print(f"{ang_deg:5.0f} {pol:>4} {deg:4d} {R.sum():16.12f} "
                  f"{T.sum():16.12f} {R.sum()+T.sum()-1:12.3e}")

print()
print("=== 5b: PMM vs RCWA (library's own, high truncation) per-order ===")
for ang_deg in (0.0, 17.0, 45.0):
    for pol in ("te", "tm"):
        o, R, T = pmm_efficiency_1d(
            period, n_si, n_ox, n_sub, n_sup, depth, duty, wl,
            angle=np.deg2rad(ang_deg), polarization=pol, degree=28,
            far_field_orders=11, stabilize=False)
        o2, R2, T2 = rcwa_efficiency_1d(
            period, n_si, n_ox, n_sub, n_sup, depth, duty, wl,
            angle=np.deg2rad(ang_deg), polarization=pol, n_orders=201)
        # align orders
        idx = {m: i for i, m in enumerate(o2)}
        dR = max(abs(R[i] - R2[idx[m]]) for i, m in enumerate(o) if m in idx)
        dT = max(abs(T[i] - T2[idx[m]]) for i, m in enumerate(o) if m in idx)
        print(f"ang={ang_deg:5.1f} {pol}: maxdR={dR:.3e} maxdT={dT:.3e} "
              f"(rcwa tot-1 = {R2.sum()+T2.sum()-1:.2e})")

print()
print("=== 5c: RECIPROCITY:  eff(sup->sub, order m) vs reversed ===")
# Lorentz reciprocity for a grating: T_{0->m}(theta_i) with kx0 = n_sup
# sin(theta_i) equals T_{-m->0} of the reversed structure, i.e. illuminate
# from the substrate at the angle of the outgoing order m.
for pol in ("te", "tm"):
    th_i = np.deg2rad(17.0)
    o, R, T = pmm_efficiency_1d(
        period, n_si, n_ox, n_sub, n_sup, depth, duty, wl,
        angle=th_i, polarization=pol, degree=28, far_field_orders=11,
        stabilize=False)
    kx0 = n_sup * np.sin(th_i)
    for m in (-1, 0, 1):
        i = int(np.where(o == m)[0][0])
        kxm = kx0 + m * wl / period
        arg = kxm / n_sub
        if abs(arg) >= 1:
            continue
        th_out = np.arcsin(arg)          # exit angle in substrate
        # reversed: illuminate from the substrate side at -th_out, look at
        # order -m into the superstrate
        o_r, R_r, T_r = pmm_efficiency_1d(
            period, n_si, n_ox, n_sup, n_sub, depth, duty, wl,
            angle=-th_out, polarization=pol, degree=28, far_field_orders=11,
            stabilize=False)
        j = int(np.where(o_r == m)[0][0])
        print(f"{pol}: m={m:+d}  T_fwd={T[i]:.12f}  T_rev={T_r[j]:.12f}  "
              f"d={abs(T[i]-T_r[j]):.3e}")
