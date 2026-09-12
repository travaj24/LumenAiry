"""ANALYSIS probe 9: eval_image_plane_wfe -- sign convention and best_rms closed form."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.analysis.image_plane_wfe import eval_image_plane_wfe
from lumenairy.raytrace import surfaces_from_prescription, system_abcd

lam = 587.6e-9
sing = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
surfs = surfaces_from_prescription(sing)
_, efl, bfl, _ = system_abcd(surfs, lam)
print(f"EFL = {efl*1e3:.4f} mm, BFL = {bfl*1e3:.4f} mm")

pres = dict(sing); pres['object_distance'] = 1e6
w0 = eval_image_plane_wfe(pres, lam, n_pupil=31)
print(f"derived img_d_m (obj at 1e6 m) = {w0.img_d_m*1e3:.4f} mm   (BFL={bfl*1e3:.4f} mm)")
print(f"  PV={w0.pv_waves:.3f} RMS={w0.rms_waves:.3f}")

print()
print("RMS(img_d) scan around BFL, image_plane='paraxial' with explicit img_d_m:")
best = (None, 1e99)
for d in np.linspace(bfl-2e-3, bfl+2e-3, 21):
    w = eval_image_plane_wfe(pres, lam, n_pupil=31, img_d_m=float(d))
    if w.rms_waves < best[1]:
        best = (d, w.rms_waves)
    print(f"  img_d = {d*1e3:8.4f} mm   PV={w.pv_waves:9.4f}  RMS={w.rms_waves:9.4f} waves")
print(f"  -> grid minimum RMS = {best[1]:.4f} waves at img_d = {best[0]*1e3:.4f} mm")

print()
print("best_rms closed form, started from several img_d_m values:")
for d0 in (bfl, bfl-1e-3, bfl+1e-3):
    wb = eval_image_plane_wfe(pres, lam, n_pupil=31, img_d_m=float(d0),
                              image_plane='best_rms')
    print(f"  start {d0*1e3:8.4f} mm -> img_d_m {wb.img_d_m*1e3:10.4f} mm  "
          f"RMS={wb.rms_waves:9.4f} (grid min {best[1]:.4f} at {best[0]*1e3:.4f} mm)")

print()
print("best_pv:")
wp = eval_image_plane_wfe(pres, lam, n_pupil=31, img_d_m=float(bfl),
                          image_plane='best_pv')
print(f"  img_d_m -> {wp.img_d_m*1e3:.4f} mm  PV={wp.pv_waves:.4f} RMS={wp.rms_waves:.4f}")

print()
print("Sign convention at the rim (undercorrected singlet, doc says POSITIVE):")
r = np.sqrt(w0.px**2+w0.py**2)
for lo, hi in ((0.0, 0.2), (0.45, 0.55), (0.95, 1.0)):
    m = (r >= lo) & (r <= hi)
    print(f"  rho in [{lo:.2f},{hi:.2f}]: mean OPD = "
          f"{np.nanmean(w0.opd_w[m]):+10.4f} waves")

print()
print("Cross-check against an independent oracle: transverse ray aberration.")
print("For an undercorrected (positive-SA) lens the marginal ray crosses the axis")
print("BEFORE the paraxial focus; the wave aberration W(rho) then has W040 > 0 in")
print("the convention where W is the OPD of the ray path relative to the reference")
print("sphere and dW/drho = -(a/R) * transverse_aberration.")
from lumenairy.raytrace import make_rings, trace
from lumenairy.raytrace.core import _make_bundle
n_f = 21
t = np.linspace(0, 1, n_f)
rays = _make_bundle(x=np.zeros(n_f), y=t*5e-3, L=np.zeros(n_f), M=np.zeros(n_f),
                    wavelength=lam)
res = trace(rays, surfs, lam, output_filter='last')
f = res.image_rays
# axial crossing of each ray
zc = -np.asarray(f.y)/np.asarray(f.M)*np.asarray(f.N)
print(f"  marginal-ray axial crossing (rho=1) = {zc[-1]*1e3:.4f} mm; "
      f"paraxial (rho->0) = {zc[1]*1e3:.4f} mm")
print(f"  => longitudinal SA = {(zc[-1]-zc[1])*1e3:+.4f} mm "
      f"(negative = undercorrected)")
