"""ANALYSIS probe 8: ghost retrace on a plane-parallel plate vs analytic;
   eval_image_plane_wfe sanity on a known singlet."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.analysis.ghost import (ghost_analysis, enumerate_ghost_paths,
                                      _path_from_pair, retrace_ghost_path)
from lumenairy.analysis.image_plane_wfe import (eval_image_plane_wfe,
                                                remove_low_order_aberrations)

lam = 587.6e-9
print("=== A. Plane-parallel plate (two flat surfaces, n=1.5) ghost ===")
# flat-flat plate, thickness t, index n.  Double-bounce (0,1): the ghost
# beam is collimated (flat surfaces), displaced 0 laterally, and its
# power fraction is R^2 (1-R)^2 relative to the incident beam.
n_g = 1.5168   # N-BK7 @ 587.6
R = ((n_g-1)/(n_g+1))**2
plate = {
    'surfaces': [
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
    ],
    'thicknesses': [5e-3],
    'aperture_diameter': 10e-3,
}
g = ghost_analysis(plate, lam, verbose=False)
print(f"  analytic normal-incidence R = {R:.6f} ; R^2 = {R*R:.6e}")
print(f"  ghost_analysis: {g}")
path = _path_from_pair(2, 0, 1)
print(f"  path (0,1) = {path}")
res = retrace_ghost_path(plate, path, lam, semi_aperture=4e-3, n_rays=64,
                         image_plane_z=0.0)
print(f"  retrace total_transmittance = {res['total_transmittance']:.6e}")
print(f"  analytic T*R*R*T            = {(1-R)*R*R*(1-R):.6e}")
print(f"  rms spot = {res['rms_radius_mm']:.6e} mm (collimated flat-plate ghost "
      f"should stay collimated -> tiny)")
print(f"  peak_xy_mm = {res['peak_xy_mm']}")

print()
print("=== B. Singlet ghost sanity (2 curved surfaces) ===")
sing = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3, glass='N-BK7', aperture=10e-3)
gs = ghost_analysis(sing, lam, verbose=False)
for e in gs:
    print(f"  path {e['path']}: R_i={e['R_i']:.5f} R_j={e['R_j']:.5f} "
          f"I={e['intensity']:.4e} focus_z_est={e['focus_z_estimate']:.4e}")

print()
print("=== C. eval_image_plane_wfe on a singlet: Seidel oracle ===")
# thin-lens spherical aberration for a plano-convex at infinite conjugate is
# large; here just check self-consistency + sign + best-focus behaviour.
pres = dict(sing)
pres['object_distance'] = 1e6      # effectively infinity
try:
    wfe = eval_image_plane_wfe(pres, lam, n_pupil=41)
    print(f"  PV = {wfe.pv_waves:.4f} waves, RMS = {wfe.rms_waves:.4f} waves, "
          f"Strehl = {wfe.strehl:.4f}")
    print(f"  img_d_m = {wfe.img_d_m*1e3:.4f} mm (paraxial {wfe.img_d_m_paraxial*1e3:.4f})")
    # sign convention: positive at the marginal edge for an undercorrected singlet
    r = np.sqrt(wfe.px**2+wfe.py**2)
    edge = r > 0.95
    print(f"  mean OPD at rim = {np.nanmean(wfe.opd_w[edge]):+.4f} waves "
          f"(doc says POSITIVE at the marginal edge for an undercorrected singlet)")
    wfe_b = eval_image_plane_wfe(pres, lam, n_pupil=41, image_plane='best_rms')
    print(f"  best_rms: RMS = {wfe_b.rms_waves:.4f} waves (was {wfe.rms_waves:.4f}) "
          f"shift = {(wfe_b.img_d_m-wfe.img_d_m)*1e6:+.2f} um")
    # low-order removal
    res_opd, coef = remove_low_order_aberrations(wfe.px, wfe.py, wfe.opd_w)
    print(f"  after piston+tilt+defocus removal: RMS = "
          f"{np.sqrt(np.nanmean((res_opd-np.nanmean(res_opd))**2)):.4f} waves; coeffs={coef}")
except Exception as e:
    import traceback; traceback.print_exc()
