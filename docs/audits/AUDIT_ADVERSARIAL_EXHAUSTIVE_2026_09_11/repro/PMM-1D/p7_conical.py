"""PROBE 7: conical PMM -- cross-pol vanishing at phi=0, phi->classical limit,
phi=90 vs the 1-D solver with swapped s/p, and energy.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import (pmm_jones_1d_conical, pmm_jones_1d,
                                    pmm_efficiency_1d)

per = 1.0e-6
wl = 1.55e-6
depth = 0.4e-6
duty = 0.5
er, eg = 3.48 ** 2, 1.444 ** 2
nsub, nsup = 1.444, 1.0
th = np.deg2rad(20.0)

print("=== 7a: cross-pol at phi -> 0 (Jones offdiag must vanish) ===")
for phid in (0.0, 1e-9, 1e-6, 1e-3, 1.0, 10.0, 45.0, 89.0, 90.0):
    o, R, T, J = pmm_jones_1d_conical(per, er, eg, nsub, nsup, depth, duty,
                                      wl, theta=th, phi=np.deg2rad(phid),
                                      degree=16, n_orders=9)
    tot = R.sum(axis=1) + T.sum(axis=1)
    print(f"phi={phid:8.5g} deg: |Jxy|={abs(J[0,1]):.3e} |Jyx|={abs(J[1,0]):.3e} "
          f"|Jxx|={abs(J[0,0]):.9f} |Jyy|={abs(J[1,1]):.9f}  tot={tot}")

print()
print("=== 7b: phi = 0 conical vs the CLASSICAL 1-D solvers ===")
o, R, T, J = pmm_jones_1d_conical(per, er, eg, nsub, nsup, depth, duty, wl,
                                  theta=th, phi=0.0, degree=20, n_orders=9)
oc, Rc, Tc, Jc = pmm_jones_1d(per, er * np.eye(3), eg * np.eye(3), nsub, nsup,
                              depth, duty, wl, angle=th, degree=20,
                              far_field_orders=19, stabilize=False)
ote, Rte, Tte = pmm_efficiency_1d(per, 3.48, 1.444, nsub, nsup, depth, duty,
                                  wl, angle=th, polarization="te", degree=20,
                                  far_field_orders=19, stabilize=False)
otm, Rtm, Ttm = pmm_efficiency_1d(per, 3.48, 1.444, nsub, nsup, depth, duty,
                                  wl, angle=th, polarization="tm", degree=20,
                                  far_field_orders=19, stabilize=False)
mm = {int(m): i for i, m in enumerate(ote)}
for m in (-1, 0, 1):
    i = int(np.where(o[:, 0] == m)[0][0])
    j = mm[m]
    ic = int(np.where(oc == m)[0][0])
    print(f"m={m:+d}: conical R[Ey]={R[1,i]:.10f} scalar-TE={Rte[j]:.10f} "
          f"jones1d[Ey]={Rc[1,ic]:.10f} | conical R[Ex]={R[0,i]:.10f} "
          f"scalar-TM={Rtm[j]:.10f} jones1d[Ex]={Rc[0,ic]:.10f}")
print("jones conical J =", J.ravel())
print("jones classical =", Jc.ravel())
print("max|dJ| =", np.max(np.abs(J - Jc)))

print()
print("=== 7c: phi = 90 deg -- grooves along the incidence plane ===")
# At phi = 90 the plane of incidence is the y-z plane, so the wavevector has
# no component along the grating periodicity: kx0 = 0.  The structure is then
# a set of uniform-in-y slabs, and the problem separates into TE/TM with an
# effective index shift.  The conical solver must give kx0 = 0 -> the order
# set is the NORMAL-incidence one, but with a ky0 offset in kz.
o, R, T, J = pmm_jones_1d_conical(per, er, eg, nsub, nsup, depth, duty, wl,
                                  theta=th, phi=np.pi / 2, degree=20,
                                  n_orders=9)
print("tot =", R.sum(axis=1) + T.sum(axis=1))
print("J =", J.ravel())
# Independent check: phi=90 with theta is equivalent to a 1-D grating solved
# at NORMAL incidence but with the wavelength replaced by wl/cos(theta) in the
# z-direction... instead cross-check against PMM2D via PMMStack conical.
from lumenairy.elements.pmm import PMMStack
st = PMMStack(per, n_substrate=nsub, n_superstrate=nsup, degree=20,
              far_field_orders=19)
st.add_layer(depth, segments=[(duty, er), (1 - duty, eg)])
st.set_source(wl, angle=th, phi=np.pi / 2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o2, R2, T2, J2 = st.solve()
print("stack conical J =", np.asarray(J2).ravel())
print("max|dJ| single-vs-stack =", np.max(np.abs(np.asarray(J2) - J)))

print()
print("=== 7d: energy at conical, degree ladder ===")
for phid in (0.0, 30.0, 60.0, 90.0):
    row = []
    for deg in (10, 14, 18, 22):
        o, R, T, J = pmm_jones_1d_conical(per, er, eg, nsub, nsup, depth, duty,
                                          wl, theta=th, phi=np.deg2rad(phid),
                                          degree=deg, n_orders=9)
        tot = R.sum(axis=1) + T.sum(axis=1)
        row.append(f"{tot[0]-1:+.2e}/{tot[1]-1:+.2e}")
    print(f"phi={phid:5.1f}: " + "  ".join(row))
