"""ANALYSIS probe 10: dissect eval_image_plane_wfe against an independent
transverse-ray-aberration oracle."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
from lumenairy.raytrace import surfaces_from_prescription, system_abcd, trace
from lumenairy.raytrace.core import _make_bundle

lam = 587.6e-9
sing = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
surfs = surfaces_from_prescription(sing)
_, efl, bfl, _ = system_abcd(surfs, lam)
a = 5e-3
pres = dict(sing); pres['object_distance'] = 1e6

# ---- Oracle: W(rho) by integrating the transverse ray aberration -------
# eps(rho) = -(R/a) dW/drho   ->   W(rho) = -(a/R) * int_0^rho eps drho'
n_f = 201
rho = np.linspace(0.0, 1.0, n_f)
rays = _make_bundle(x=np.zeros(n_f), y=rho*a, L=np.zeros(n_f), M=np.zeros(n_f),
                    wavelength=lam)
rays.z = np.full(n_f, 0.0)
res = trace(rays, surfs, lam, output_filter='last')
f = res.image_rays
y2 = np.asarray(f.y); z2 = np.asarray(f.z)
M2 = np.asarray(f.M); N2 = np.asarray(f.N)
R_img = float(bfl)
# transverse aberration at the paraxial image plane (z = R_img from vertex)
t_adv = (R_img - z2)/N2
eps = y2 + M2*t_adv                     # transverse height at the image plane
eps = eps - eps[0]                      # relative to the paraxial ray
W_oracle = -(a/R_img)*np.concatenate(([0.0], np.cumsum(
    0.5*(eps[1:]+eps[:-1])*np.diff(rho))))
W_oracle_w = W_oracle/lam
print(f"EFL={efl*1e3:.4f} BFL={bfl*1e3:.4f} mm   a={a*1e3:.2f} mm  f/#={bfl/(2*a):.2f}")
print(f"ORACLE (transverse-ray-aberration integral):")
print(f"  eps(rho=1) = {eps[-1]*1e6:+.3f} um")
print(f"  W(rho=1)   = {W_oracle_w[-1]:+.4f} waves   PV = "
      f"{W_oracle_w.max()-W_oracle_w.min():.4f} waves")
# fit W040 rho^4
c4 = np.polyfit(rho**4, W_oracle_w, 1)[0]
print(f"  fitted W040 = {c4:+.4f} waves")

# ---- Library ----------------------------------------------------------
px = np.zeros(n_f); py = rho
w = eval_image_plane_wfe(pres, lam, pupil_grid=(px, py), img_d_m=R_img)
print()
print(f"LIBRARY eval_image_plane_wfe (paraxial, vertex-tangent):")
print(f"  opd_w(rho=1) = {w.opd_w[-1]:+.4f} waves   PV = {w.pv_waves:.4f} waves"
      f"   RMS = {w.rms_waves:.4f}")
print(f"  r_sphere_m = {w.r_sphere_m*1e3:.4f} mm, img_d_m = {w.img_d_m*1e3:.4f} mm")
print(f"  ratio library/oracle at rho=1 = "
      f"{w.opd_w[-1]/W_oracle_w[-1] if W_oracle_w[-1] else float('nan'):.4f}")
print()
print("  rho     oracle[waves]   library[waves]   difference")
for i in range(0, n_f, 20):
    print(f"  {rho[i]:4.2f}   {W_oracle_w[i]:+12.4f}   {w.opd_w[i]:+12.4f}   "
          f"{w.opd_w[i]-W_oracle_w[i]:+12.4f}")
# is the difference a pure quadratic in rho?
d = w.opd_w - W_oracle_w
p = np.polyfit(rho**2, d, 1)
print(f"  difference fitted as A + B*rho^2 : A={p[1]:+.4f}  B={p[0]:+.4f} waves; "
      f"residual max = {np.abs(d - (p[1]+p[0]*rho**2)).max():.4f} waves")

print()
print("  exit_pupil tangent variant:")
w2 = eval_image_plane_wfe(pres, lam, pupil_grid=(px, py), img_d_m=R_img,
                          sphere_tangent='exit_pupil')
print(f"    opd_w(rho=1) = {w2.opd_w[-1]:+.4f} waves  PV={w2.pv_waves:.4f} "
      f"r_sphere={w2.r_sphere_m*1e3:.4f} mm")
d2 = w2.opd_w - W_oracle_w
p2 = np.polyfit(rho**2, d2, 1)
print(f"    difference A={p2[1]:+.4f} B={p2[0]:+.4f}; residual max="
      f"{np.abs(d2-(p2[1]+p2[0]*rho**2)).max():.4f} waves")

print()
print("  --- where does the quadratic come from? raw pieces at rho=1 ---")
# replicate the internals
from lumenairy.raytrace import first_order_data
fod = first_order_data(surfs, lam)
print(f"    fod.xp_z = {getattr(fod,'xp_z',None)}, ep_z={getattr(fod,'ep_z',None)}, "
      f"ep_radius={getattr(fod,'ep_radius',None)}")
obj_d = 1e6
ep_z = float(getattr(fod, 'ep_z', 0.0)); ep_r = float(getattr(fod, 'ep_radius', a))
print(f"    ep_r used for launch = {ep_r*1e3:.4f} mm  (semi-aperture {a*1e3:.4f} mm)")
