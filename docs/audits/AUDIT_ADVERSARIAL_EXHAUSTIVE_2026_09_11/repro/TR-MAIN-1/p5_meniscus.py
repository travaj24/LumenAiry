"""Probe 5: exit-vertex transfer sign on surfaces with CONVEX exit (sag>0)
and CONCAVE exit (sag<0); probe 2 extended to a fast biconvex."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from p2_opd_oracle import oracle_opd_on_exit_grid
WL = common.WL; k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

def run(tag, rx, AP, N=768, subs=(1, 8)):
    dx = 1.3*AP/N
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x)
    m = (X**2+Y**2) <= (0.45*AP)**2
    opl_or, _, _ = oracle_opd_on_exit_grid(rx, X, Y)
    E_in = np.ones((N, N), np.complex128)
    # sag of the LAST surface at the pupil edge, to show the sign at stake
    R2 = float(rx['surfaces'][-1]['radius'])
    h = 0.45*AP
    sag = (R2 - np.sign(R2)*np.sqrt(R2**2-h**2)) if np.isfinite(R2) else 0.0
    print(f"{tag}: exit R={R2*1e3 if np.isfinite(R2) else np.inf} mm, "
          f"sag(h={h*1e3:.2f}mm) = {sag*1e6:+.3f} um "
          f"-> n_exit*sag = {sag*1e9:+.1f} nm of OPL at stake")
    for sub in subs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL,
                                       dx=dx, ray_subsample=sub, n_workers=1,
                                       on_undersample='silent',
                                       min_coarse_samples_per_aperture=0,
                                       on_pool_memory='silent')
        r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
        A = np.stack([np.ones(r.size), X[m], Y[m]], 1)
        c,*_ = np.linalg.lstsq(A, r, rcond=None); rr = r-A@c
        print(f"   sub={sub}: raw |resid|max={np.abs(r).max()*1e9:.5f} nm  "
              f"RMS(piston+tilt removed)={rr.std()*1e9:.5f} nm")

run("neg meniscus (CONVEX exit, sag>0)",
    common.neg_meniscus(R1=-40e-3, R2=-25e-3, t=3e-3, ap=8e-3), 8e-3)
run("biconvex f/8 (CONCAVE exit, sag<0)",
    {'wavelength': WL, 'aperture_diameter': 8e-3,
     'surfaces': [
        {'radius': 60e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': '_AUD_GLASS', 'semi_diameter': 4e-3},
        {'radius': -60e-3, 'thickness': 0.0, 'glass_before': '_AUD_GLASS',
         'glass_after': 'air', 'semi_diameter': 4e-3}],
     'thicknesses': [4e-3], 'stop_index': 0}, 8e-3)
