"""ANALYSIS probe 11: is trace(...).image_rays.opd a true OPL?  Independent
geometric recomputation from the ray history."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.raytrace import surfaces_from_prescription, system_abcd, trace
from lumenairy.raytrace.core import _make_bundle
from lumenairy.glass import get_glass_index

lam = 587.6e-9
sing = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
surfs = surfaces_from_prescription(sing)
_, efl, bfl, _ = system_abcd(surfs, lam)
print("surfaces:")
for i, s in enumerate(surfs):
    print(f"  {i}: R={s.radius}, t={s.thickness}, gb={s.glass_before} ga={s.glass_after}")
a = 5e-3
rho = np.array([0.0, 0.25, 0.5, 0.75, 0.95])
n_f = rho.size
obj_d = 1e6
# replicate eval_image_plane_wfe's launch exactly (ep_z = 0, ep_r = a)
aim_x = np.zeros(n_f); aim_y = rho*a
aim_z = obj_d
nrm = np.sqrt(aim_x**2+aim_y**2+aim_z**2)
b = _make_bundle(x=np.zeros(n_f), y=np.zeros(n_f), L=aim_x/nrm, M=aim_y/nrm,
                 wavelength=lam)
b.z = np.full(n_f, -obj_d)
res = trace(b, surfs, lam, output_filter='all')
f = res.image_rays
opl = np.asarray(f.opd)
print()
print("library opd (from trace) minus chief, in waves:")
print("  ", np.array2string((opl-opl[0])/lam, precision=4))

# --- independent geometric OPL -----------------------------------------
hist = res.ray_history
print(f"  ray_history has {len(hist)} entries")
n_air = 1.0
n_g = get_glass_index('N-BK7', lam)
# segment 0: launch plane (z=-obj_d) -> surface 0
P = np.stack([np.zeros(n_f), np.zeros(n_f), np.full(n_f, -obj_d)], axis=1)
opl_manual = np.zeros(n_f)
media = [n_air, n_g]
for k, hb in enumerate(hist):
    Q = np.stack([np.asarray(hb.x), np.asarray(hb.y), np.asarray(hb.z)], axis=1)
    # surface k local frame: z measured from its own vertex.  Convert to global.
    z_vert = 0.0 if k == 0 else float(sum(s.thickness for s in surfs[:k]))
    Qg = Q.copy(); Qg[:, 2] = Q[:, 2] + z_vert
    seg = np.linalg.norm(Qg - P, axis=1)
    opl_manual += media[k]*seg
    P = Qg
print("manual geometric OPL minus chief, in waves:")
print("  ", np.array2string((opl_manual-opl_manual[0])/lam, precision=4))

# --- analytic expectation for the OPL difference ------------------------
# For a perfect focusing system the exit OPL difference is -(L(h)-R).
R = float(bfl)
hh = np.asarray(hist[-1].y)
ss = np.asarray(hist[-1].z)
L = np.sqrt(hh**2 + (R-ss)**2)
print(f"exit heights h = {np.array2string(hh*1e3, precision=4)} mm")
print(f"exit sag    s = {np.array2string(ss*1e6, precision=3)} um")
print(f"-(L-R)/lam    = {np.array2string(-(L-R)/lam, precision=4)} waves "
      f"(perfect-system expectation for opd_a_w)")

# --- t from the ray-sphere quadratic ------------------------------------
Ld = np.asarray(f.L); Md = np.asarray(f.M); Nd = np.asarray(f.N)
s2x = np.asarray(f.x); s2y = np.asarray(f.y); s2z = np.asarray(f.z)
cz = s2z[0] + R
bq = 2.0*((s2x-0.0)*Ld + (s2y-0.0)*Md + (s2z-cz)*Nd)
cq = (s2x)**2 + (s2y)**2 + (s2z-cz)**2 - R**2
disc = bq**2-4*cq
sq = np.sqrt(np.maximum(disc, 0))
t1 = (-bq-sq)/2; t2 = (-bq+sq)/2
t = np.where(np.abs(t1) < np.abs(t2), t1, t2)
print(f"t (ray->sphere)/lam = {np.array2string(t/lam, precision=4)} waves")
print()
print("SUM opd_a_w + t/lam (should be ~ -W, a few waves):")
print("  library opd :", np.array2string((opl-opl[0])/lam + t/lam, precision=4))
print("  manual  opd :", np.array2string((opl_manual-opl_manual[0])/lam + t/lam,
                                         precision=4))
