"""Probe 8: Newton amp mask -- what do masked pixels get, and is the
NaN/zero-fill bleed contained at the mask rim?"""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from p2_opd_oracle import oracle_opd_on_exit_grid
WL = common.WL; k0 = 2*np.pi/WL
N = 512; AP = 8e-3; dx = 1.6*AP/N
rx = common.plano_convex(R=60e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R = np.hypot(X, Y)
# TOP-HAT pupil input, radius 2.4 mm (well inside the 4 mm aperture radius)
r_top = 2.4e-3
E_in = (R <= r_top).astype(np.complex128)
opl_or, _, _ = oracle_opd_on_exit_grid(rx, X, Y)
kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent')

for imap in (True, False):
  for mrel, dil in ((1e-4, 2), (1e-4, 0), (0.0, 2)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = apply_real_lens_traced(E_in, newton_amp_mask_rel=mrel,
                                   newton_mask_dilate_coarse_px=dil,
                                   inverse_map=imap, **kw)
    amp = np.abs(E)
    nz = amp > 0
    resid = np.angle(E*np.exp(-1j*k0*opl_or))
    bad = nz & (np.abs(resid) > 1e-3)          # > 1 mrad of phase error
    # amplitude carried by the wrong-phase pixels
    p_bad = float((amp[bad]**2).sum()); p_tot = float((amp[nz]**2).sum())
    print(f"imap={imap} mask_rel={mrel:g} dilate={dil}: "
          f"nonzero={nz.sum():6d}  wrong-phase(>1mrad)={bad.sum():5d} "
          f"({100*p_bad/max(p_tot,1e-300):.3e} % of power)  "
          f"max|resid|={np.abs(resid[nz]).max():.3e} rad  "
          f"NaN in E: {np.isnan(E).any()}")
    if bad.any():
        print(f"     wrong-phase pixels at r = {R[bad].min()*1e3:.4f} .. "
              f"{R[bad].max()*1e3:.4f} mm (top-hat rim at {r_top*1e3:.3f} mm); "
              f"their |E|/|E|max = {amp[bad].max()/amp[nz].max():.3e}")
