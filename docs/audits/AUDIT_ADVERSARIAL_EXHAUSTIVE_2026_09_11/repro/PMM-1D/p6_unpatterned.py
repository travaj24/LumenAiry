"""PROBE 6: unpatterned limit -- PMM vs analytic slab TMM, s and p, 0/30/60 deg.

Also the AMPLITUDE/PHASE check via pmm_jones_1d.
"""
import sys, os
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle import tmm_slab
from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d

wl = 0.55e-6
period = 0.4e-6      # sub-wavelength so only order 0 propagates
depth = 0.32e-6
n_sup, n_sub = 1.0, 1.5
n_layer = 2.1 + 0.0j

print("=== PROBE 6a: uniform layer (ridge == groove), scalar PMM vs TMM ===")
print(f"{'ang':>5} {'pol':>4} {'R_pmm':>14} {'R_tmm':>14} {'dR':>10} "
      f"{'T_pmm':>14} {'T_tmm':>14} {'dT':>10}")
worst = 0.0
for ang_deg in (0.0, 30.0, 60.0):
    ang = np.deg2rad(ang_deg)
    for pol, spol in (("te", "s"), ("tm", "p")):
        orders, R, T = pmm_efficiency_1d(
            period, n_layer, n_layer, n_sub, n_sup, depth, 0.5, wl,
            angle=ang, polarization=pol, degree=12, far_field_orders=11,
            stabilize=False)
        m0 = int(np.where(orders == 0)[0][0])
        r, t, Rt, Tt = tmm_slab(n_sup, n_layer, n_sub, depth, wl, ang, spol)
        dR, dT = abs(R[m0] - Rt), abs(T[m0] - Tt)
        worst = max(worst, dR, dT)
        print(f"{ang_deg:5.0f} {pol:>4} {R[m0]:14.10f} {Rt:14.10f} {dR:10.2e} "
              f"{T[m0]:14.10f} {Tt:14.10f} {dT:10.2e}")
        # also check no power leaks into nonzero orders
        leak = float(np.sum(R) + np.sum(T) - R[m0] - T[m0])
        if abs(leak) > 1e-14:
            print(f"        !! leak into m!=0: {leak:.3e}")
print(f"worst |dR|,|dT| = {worst:.3e}")

print()
print("=== PROBE 6b: AMPLITUDE + PHASE via pmm_jones_1d (uniform layer) ===")
print(f"{'ang':>5} {'ch':>3} {'|r|_pmm':>12} {'|r|_tmm':>12} {'d|r|':>10} "
      f"{'arg_pmm':>12} {'arg_tmm':>12} {'dphase':>10}")
eps_l = n_layer ** 2 * np.eye(3)
for ang_deg in (0.0, 30.0, 60.0):
    ang = np.deg2rad(ang_deg)
    orders, R, T, J = pmm_jones_1d(
        period, eps_l, eps_l, n_sub, n_sup, depth, 0.5, wl,
        angle=ang, degree=12, far_field_orders=11, stabilize=False)
    # J is the order-0 REFLECTION Jones in the (Ex, Ey) lab basis.
    # Ey = s (TE, along grooves); Ex = p-ish (in-plane).
    rs, ts, _, _ = tmm_slab(n_sup, n_layer, n_sub, depth, wl, ang, 's')
    rp, tp, _, _ = tmm_slab(n_sup, n_layer, n_sub, depth, wl, ang, 'p')
    # tmm 'p' r is for the tangential-H (Hy) amplitude ratio; the Ex ratio
    # differs by the sign convention of Ex vs Hy on reflection.
    for lbl, jval, ref in (("yy(s)", J[1, 1], rs), ("xx(p)", J[0, 0], rp)):
        d1 = abs(abs(jval) - abs(ref))
        dph = np.angle(jval / ref) if abs(ref) > 0 else np.nan
        # allow a pi flip for the p convention
        dph2 = np.angle(jval / (-ref)) if abs(ref) > 0 else np.nan
        print(f"{ang_deg:5.0f} {lbl:>5} {abs(jval):12.9f} {abs(ref):12.9f} "
              f"{d1:10.2e} {np.angle(jval):12.8f} {np.angle(ref):12.8f} "
              f"dphi={dph:+.3e} dphi(pi)={dph2:+.3e}")
    print(f"      offdiag |Jxy|={abs(J[0,1]):.3e} |Jyx|={abs(J[1,0]):.3e}")

print()
print("=== PROBE 6c: LOSSY uniform layer (metal-ish), scalar PMM vs TMM ===")
for nl in (2.0 + 0.1j, 0.2 + 3.5j):
    for ang_deg in (0.0, 40.0):
        ang = np.deg2rad(ang_deg)
        for pol, spol in (("te", "s"), ("tm", "p")):
            orders, R, T = pmm_efficiency_1d(
                period, nl, nl, n_sub, n_sup, 0.05e-6, 0.5, wl,
                angle=ang, polarization=pol, degree=14, far_field_orders=11,
                stabilize=False)
            m0 = int(np.where(orders == 0)[0][0])
            r, t, Rt, Tt = tmm_slab(n_sup, nl, n_sub, 0.05e-6, wl, ang, spol)
            print(f"n={nl} ang={ang_deg:4.0f} {pol}: R {R[m0]:.12f} vs "
                  f"{Rt:.12f} (d={abs(R[m0]-Rt):.2e})  T {T[m0]:.12f} vs "
                  f"{Tt:.12f} (d={abs(T[m0]-Tt):.2e})")
