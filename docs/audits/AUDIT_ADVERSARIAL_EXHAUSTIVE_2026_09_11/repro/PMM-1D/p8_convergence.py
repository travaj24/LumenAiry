"""PROBE 8: convergence vs degree for a METALLIC (Au) TM grating + stabilize
behaviour; and the lossy ABSORPTION balance (energy cannot self-check there).
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

wl = 0.633e-6
per = 0.6e-6
depth = 0.1e-6
duty = 0.5
n_au = 0.18 + 3.43j            # Au @ 633 nm
n_sup, n_sub = 1.0, 1.0

print("=== 8a: Au / air lamellar, TM, degree ladder (stabilize=False) ===")
o_r, R_r, T_r = rcwa_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth, duty,
                                   wl, angle=np.deg2rad(10.0),
                                   polarization="tm", n_orders=401)
idx_r = {int(m): i for i, m in enumerate(o_r)}
print(f"RCWA n=401 oracle: R0={R_r[idx_r[0]]:.10f} R-1={R_r[idx_r[-1]]:.10f} "
      f"sumR={R_r.sum():.10f} sumT={T_r.sum():.10f} A={1-R_r.sum()-T_r.sum():.10f}")
print(f"{'deg':>5} {'R0':>14} {'R-1':>14} {'sumR':>12} {'sumT':>12} "
      f"{'A=1-R-T':>12} {'|dR0|oracle':>12}")
prev = None
for deg in range(6, 45, 2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth, duty,
                                    wl, angle=np.deg2rad(10.0),
                                    polarization="tm", degree=deg,
                                    far_field_orders=15, stabilize=False)
    i0 = int(np.where(o == 0)[0][0])
    im = int(np.where(o == -1)[0][0])
    A = 1 - R.sum() - T.sum()
    print(f"{deg:5d} {R[i0]:14.10f} {R[im]:14.10f} {R.sum():12.8f} "
          f"{T.sum():12.8f} {A:12.8f} {abs(R[i0]-R_r[idx_r[0]]):12.3e}")

print()
print("=== 8b: same with elements_per_region=3, grade=True (hp) ===")
for deg in (8, 12, 16, 20, 24):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth, duty,
                                    wl, angle=np.deg2rad(10.0),
                                    polarization="tm", degree=deg,
                                    elements_per_region=3, grade=True,
                                    far_field_orders=15, stabilize=False)
    i0 = int(np.where(o == 0)[0][0])
    print(f"  deg={deg:3d}: R0={R[i0]:.10f} |d|={abs(R[i0]-R_r[idx_r[0]]):.3e} "
          f"A={1-R.sum()-T.sum():.8f}")

print()
print("=== 8c: TE for reference (should converge fast) ===")
o_r2, R_r2, T_r2 = rcwa_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth,
                                      duty, wl, angle=np.deg2rad(10.0),
                                      polarization="te", n_orders=401)
i0r = {int(m): i for i, m in enumerate(o_r2)}[0]
for deg in (8, 12, 16, 20, 28, 36):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth, duty,
                                    wl, angle=np.deg2rad(10.0),
                                    polarization="te", degree=deg,
                                    far_field_orders=15, stabilize=False)
    i0 = int(np.where(o == 0)[0][0])
    print(f"  deg={deg:3d}: R0={R[i0]:.10f} |d|={abs(R[i0]-R_r2[i0r]):.3e}")

print()
print("=== 8d: what does stabilize=True return on the Au TM case? ===")
for deg in (10, 16, 24, 32):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        try:
            o, R, T = pmm_efficiency_1d(per, n_au, 1.0, n_sub, n_sup, depth,
                                        duty, wl, angle=np.deg2rad(10.0),
                                        polarization="tm", degree=deg,
                                        far_field_orders=15, stabilize=True)
            i0 = int(np.where(o == 0)[0][0])
            msg = "; ".join(str(x.message)[:70] for x in W)
            print(f"  deg={deg:3d}: R0={R[i0]:.10f} |d|="
                  f"{abs(R[i0]-R_r[idx_r[0]]):.3e}  {msg}")
        except Exception as e:
            print(f"  deg={deg:3d}: RAISED {type(e).__name__}: {str(e)[:100]}")
