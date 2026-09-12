"""ANALYSIS probe 12: eval_image_plane_wfe object-distance precision cliff,
plus validation against a transverse-ray oracle in the sane regime."""
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
print("=== A. z-intersection error at surface 0 vs object_distance ===")
print("   obj_d [m]   y0 [mm]     z0 [um]   analytic sag [um]   error [um]")
for obj in (0.5, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7):
    aim_y = np.array([0.0, 0.95*a]); nrm = np.sqrt(aim_y**2+obj**2)
    b = _make_bundle(x=np.zeros(2), y=np.zeros(2), L=np.zeros(2),
                     M=aim_y/nrm, wavelength=lam)
    b.z = np.full(2, -obj)
    res = trace(b, surfs, lam, output_filter='all')
    h0 = res.ray_history[0]
    y0 = float(np.asarray(h0.y)[1]); z0 = float(np.asarray(h0.z)[1])
    z0c = float(np.asarray(h0.z)[0])
    sag = y0**2/(2*0.05)
    print(f"  {obj:9.1e}  {y0*1e3:8.5f}  {z0*1e6:10.4f}   {sag*1e6:10.4f}   "
          f"{(z0-sag)*1e6:+10.4f}   (chief z0 = {z0c*1e6:+.4f} um, should be 0)")

print()
print("=== B. reported WFE vs object_distance (same physical system) ===")
print("   obj_d [m]   img_d [mm]    PV [waves]   RMS [waves]")
for obj in (0.5, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
    pres = dict(sing); pres['object_distance'] = float(obj)
    try:
        w = eval_image_plane_wfe(pres, lam, n_pupil=31)
        print(f"  {obj:9.1e}  {w.img_d_m*1e3:10.4f}  {w.pv_waves:11.4f}  "
              f"{w.rms_waves:11.4f}")
    except Exception as e:
        print(f"  {obj:9.1e}  FAILED: {e}")

print()
print("=== C. validate against transverse-ray oracle at obj_d = 1.0 m ===")
obj = 1.0
pres = dict(sing); pres['object_distance'] = obj
n_f = 201
rho = np.linspace(0.0, 1.0, n_f)
w = eval_image_plane_wfe(pres, lam, pupil_grid=(np.zeros(n_f), rho))
R_img = w.img_d_m
# oracle: trace the same rays, propagate to the image plane, integrate eps
aim_y = rho*a; nrm = np.sqrt(aim_y**2+obj**2)
b = _make_bundle(x=np.zeros(n_f), y=np.zeros(n_f), L=np.zeros(n_f),
                 M=aim_y/nrm, wavelength=lam)
b.z = np.full(n_f, -obj)
res = trace(b, surfs, lam, output_filter='last')
f = res.image_rays
y2 = np.asarray(f.y); z2 = np.asarray(f.z); M2 = np.asarray(f.M); N2 = np.asarray(f.N)
eps = y2 + M2*(R_img - z2)/N2
eps = eps - eps[0]
a_exit = float(np.nanmax(np.abs(y2)))
W_or = -(a_exit/R_img)*np.concatenate(([0.0], np.cumsum(
    0.5*(eps[1:]+eps[:-1])*np.diff(rho))))/lam
good = np.isfinite(w.opd_w) & np.isfinite(W_or)
print(f"  img_d_m = {R_img*1e3:.5f} mm, a_exit = {a_exit*1e3:.4f} mm")
print(f"  oracle  PV = {W_or[good].max()-W_or[good].min():.4f} waves")
print(f"  library PV = {w.pv_waves:.4f} waves  RMS = {w.rms_waves:.4f}")
print("   rho    oracle      library")
for i in range(0, n_f, 25):
    print(f"  {rho[i]:4.2f}  {W_or[i]:+10.4f}  {w.opd_w[i]:+10.4f}")

print()
print("=== D. what does the module's own doctest / test suite use? ===")
import subprocess
