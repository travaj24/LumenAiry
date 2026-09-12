"""Probe 8 (v2): the Newton amp mask's effect measured as a DIFFERENCE against
the unmasked (newton_amp_mask_rel=0.0) field."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
WL = common.WL; k0 = 2*np.pi/WL
N = 512; AP = 8e-3; dx = 1.6*AP/N
rx = common.plano_convex(R=200e-3, t=4e-3, ap=AP)   # slow: f ~ 387 mm
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R = np.hypot(X, Y)
kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent')
for tag, E_in in (("top-hat r=2.4mm", (R <= 2.4e-3).astype(np.complex128)),
                  ("gaussian w=1.2mm",
                   np.exp(-(R/1.2e-3)**2).astype(np.complex128))):
  for imap in (True, False):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E0 = apply_real_lens_traced(E_in, newton_amp_mask_rel=0.0,
                                    inverse_map=imap, **kw)
        for dil in (0, 2, 4):
            Em = apply_real_lens_traced(E_in, newton_amp_mask_rel=1e-4,
                                        newton_mask_dilate_coarse_px=dil,
                                        inverse_map=imap, **kw)
            d = Em - E0
            p0 = float((np.abs(E0)**2).sum())
            both = (np.abs(E0) > 0) & (np.abs(Em) > 0)
            dph = np.angle(Em[both]*np.conj(E0[both]))
            bad = both & (np.abs(np.angle(Em*np.conj(E0))) > 1e-3)
            zeroed = (np.abs(E0) > 0) & (np.abs(Em) == 0)
            print(f"{tag} imap={imap} dilate={dil}: "
                  f"dE power/total = {float((np.abs(d)**2).sum())/p0:.3e}; "
                  f"pixels zeroed by mask = {int(zeroed.sum())} "
                  f"(power {float((np.abs(E0[zeroed])**2).sum())/p0:.3e}); "
                  f"wrong-phase>1mrad = {int(bad.sum())} "
                  f"(power {float((np.abs(E0[bad])**2).sum())/p0:.3e}, "
                  f"max dphi={np.abs(dph).max() if dph.size else 0:.3e} rad)")
            if bad.any():
                print(f"      their radii {R[bad].min()*1e3:.3f}-{R[bad].max()*1e3:.3f} mm,"
                      f" |E0|/|E0|max up to {np.abs(E0[bad]).max()/np.abs(E0).max():.3e}")
