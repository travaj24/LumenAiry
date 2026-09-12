import os, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
TMP = r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/IO-OPTIMIZE"
import numpy as np, lumenairy as la
from lumenairy.raytrace.seidel import system_abcd_prescription
from lumenairy.glass import get_glass_index

def analytic_efl(radii, thick, ns):
    """Exact paraxial EFL by surface-by-surface ray transfer (air-to-air)."""
    y, u = 1.0, 0.0
    n_prev = 1.0
    for i, R in enumerate(radii):
        n_next = ns[i]
        u = (n_prev*u - y*(n_next-n_prev)/R) / n_next if np.isfinite(R) else n_prev*u/n_next
        n_prev = n_next
        if i < len(thick):
            y = y + thick[i]*u
    return -1.0/u

for wl, label in ((587.6e-9, 'd-line'), (1.31e-6, '1310nm')):
    nb = get_glass_index('N-BAF10', wl); ns = get_glass_index('N-SF6HT', wl)
    print(f"{label}: n(N-BAF10)={nb:.6f} n(N-SF6HT)={ns:.6f} n(N-BK7)={get_glass_index('N-BK7', wl):.6f}")

# prompt's AC254-100-A numbers
R = [62.75e-3, -45.71e-3, -128.23e-3]; t = [4.0e-3, 2.5e-3]
wl = 587.6e-9
ns = [get_glass_index('N-BAF10', wl), get_glass_index('N-SF6HT', wl), 1.0]
print("analytic EFL (prompt AC254-100-A numbers) =", analytic_efl(R, t, ns)*1e3, "mm")

# repo fixture AC254_100_C
rx = la.load_zemax_zmx('validation/real_lens_opd/zemax_prescriptions/AC254_100_C.zmx')
abcd, efl, bfl, ffl = system_abcd_prescription(rx, 1.31e-6)
Rr = [s['radius'] for s in rx['surfaces']]
nsr = [get_glass_index('N-BAF10', 1.31e-6), get_glass_index('N-SF6HT', 1.31e-6), 1.0]
print(f"AC254_100_C.zmx: radii(mm)={[f'{x*1e3:.3f}' for x in Rr]} t={[f'{x*1e3:.3f}' for x in rx['thicknesses']]}")
print(f"   system_abcd EFL={efl*1e3:.4f} mm BFL={bfl*1e3:.4f}  analytic EFL={analytic_efl(Rr, rx['thicknesses'], nsr)*1e3:.4f} mm (file DISZ after S3 = 84.143 mm)")

# repo fixture LA1509_C (plano-convex, exact f = R/(n-1))
rx2 = la.load_zemax_zmx('validation/real_lens_opd/zemax_prescriptions/LA1509_C.zmx')
abcd, efl2, bfl2, _ = system_abcd_prescription(rx2, 1.31e-6)
R1 = rx2['surfaces'][0]['radius']; n = get_glass_index('N-BK7', 1.31e-6)
print(f"LA1509_C.zmx: R1={R1*1e3:.4f} mm  system_abcd EFL={efl2*1e3:.4f}  analytic R/(n-1)={R1/(n-1)*1e3:.4f} mm")
print(f"   -> Thorlabs LA1509 datasheet: R=51.5 mm, f=100 mm.  repo fixture/catalog encode R=103.29 mm (f=200 mm).")

# THORLABS_CATALOG cross-check
for part in ('LA1050-C', 'LA1509-C', 'LA1301-C'):
    p = la.thorlabs_lens(part)
    _, e, _, _ = system_abcd_prescription(p, 1.31e-6)
    print(f"   thorlabs_lens({part}): R1={p['surfaces'][0]['radius']*1e3:.3f} mm  EFL@1310={e*1e3:.2f} mm")
