"""PROBE 8b: is the Au-TM convergence ALGEBRAIC or SPECTRAL, and does
elements_per_region>1 + grade=True 'recover the rate' as the docstring says?

Independent oracle: Richardson-extrapolate BOTH the RCWA n_orders sequence and
the PMM degree sequence, and cross-check they agree.
"""
import sys, warnings, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

wl, per, depth, duty = 0.633e-6, 0.6e-6, 0.1e-6, 0.5
n_au = 0.18 + 3.43j
ang = np.deg2rad(10.0)


def rcwa(n):
    o, R, T = rcwa_efficiency_1d(per, n_au, 1.0, 1.0, 1.0, depth, duty, wl,
                                 angle=ang, polarization="tm", n_orders=n)
    return R[int(np.where(o == 0)[0][0])]


def pmm(deg, nel=1, grade=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(per, n_au, 1.0, 1.0, 1.0, depth, duty, wl,
                                    angle=ang, polarization="tm", degree=deg,
                                    elements_per_region=nel, grade=grade,
                                    far_field_orders=15, stabilize=False)
    return R[int(np.where(o == 0)[0][0])], 2 * nel * deg


print("=== RCWA n_orders sequence (the oracle's own convergence) ===")
rv = {}
for n in (101, 201, 301, 401, 601, 801, 1201):
    t = time.perf_counter()
    rv[n] = rcwa(n)
    print(f"  n={n:5d}: R0={rv[n]:.12f}  ({time.perf_counter()-t:.1f}s)")
ns = sorted(rv)
print("  successive diffs:", [f"{rv[ns[i+1]]-rv[ns[i]]:+.3e}"
                              for i in range(len(ns) - 1)])
# Richardson in 1/n (FMM TM metal converges ~1/n)
a, b = ns[-2], ns[-1]
rich = rv[b] + (rv[b] - rv[a]) / (a / b - 1.0) * 0  # placeholder
# fit R0(n) = R_inf + c/n
A = np.array([[1.0, 1.0 / n] for n in ns[-4:]])
y = np.array([rv[n] for n in ns[-4:]])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
print(f"  fit R0 = {coef[0]:.12f} + {coef[1]:.4g}/n   -> R_inf(rcwa) = "
      f"{coef[0]:.12f}")
R_inf = coef[0]

print()
print("=== PMM degree sequence, elements_per_region = 1 ===")
prev = None
for deg in (6, 8, 12, 16, 24, 32, 44, 60):
    t = time.perf_counter()
    v, dof = pmm(deg)
    e = abs(v - R_inf)
    rate = "" if prev is None else f" rate={np.log(prev[1]/e)/np.log(deg/prev[0]):.2f}"
    print(f"  deg={deg:3d} dof={dof:4d}: R0={v:.12f} err={e:.3e}{rate} "
          f"({time.perf_counter()-t:.2f}s)")
    prev = (deg, e)

print()
print("=== PMM, elements_per_region = 3, grade=True (the docstring's cure) ===")
for deg in (6, 8, 12, 16, 24, 32):
    v, dof = pmm(deg, 3, True)
    print(f"  deg={deg:3d} dof={dof:4d}: R0={v:.12f} err={abs(v-R_inf):.3e}")

print()
print("=== PMM, elements_per_region = 6, grade=True ===")
for deg in (6, 8, 12, 16, 24):
    v, dof = pmm(deg, 6, True)
    print(f"  deg={deg:3d} dof={dof:4d}: R0={v:.12f} err={abs(v-R_inf):.3e}")

print()
print("=== PMM, elements_per_region = 3, grade=FALSE (control) ===")
for deg in (8, 16, 24, 32):
    v, dof = pmm(deg, 3, False)
    print(f"  deg={deg:3d} dof={dof:4d}: R0={v:.12f} err={abs(v-R_inf):.3e}")

print()
print("=== same, but a LOSSLESS high-contrast TM grating (no metal corner) ===")
def pmm2(deg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(per, 3.48, 1.0, 1.0, 1.0, depth, duty, wl,
                                    angle=ang, polarization="tm", degree=deg,
                                    far_field_orders=15, stabilize=False)
    return R[int(np.where(o == 0)[0][0])]
o, R, T = rcwa_efficiency_1d(per, 3.48, 1.0, 1.0, 1.0, depth, duty, wl,
                             angle=ang, polarization="tm", n_orders=801)
ref = R[int(np.where(o == 0)[0][0])]
for deg in (8, 12, 16, 24, 32, 44):
    print(f"  deg={deg:3d}: R0={pmm2(deg):.12f} err_vs_rcwa801="
          f"{abs(pmm2(deg)-ref):.3e}")
