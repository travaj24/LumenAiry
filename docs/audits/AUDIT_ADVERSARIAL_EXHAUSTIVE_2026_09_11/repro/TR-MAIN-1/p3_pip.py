"""Probe 3: preserve_input_phase=True with a strongly CURVED input through a
ZERO-POWER plate.  The exit phase must equal the input phase propagated
through the plate (which for a plate is the ASM result)."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
WL = common.WL; k0 = 2*np.pi/WL
N = 512; AP = 4e-3; dx = 1.6*AP/N
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
R2 = X**2+Y**2
w = 0.8e-3
rxp = common.plate(t=4e-3, ap=AP)
nG = la.get_glass_index('_AUD_GLASS', WL)
print(f"plate t=4mm n={nG}; grid dx={dx*1e6:.3f} um, w={w*1e3} mm")

for name, Rc in (("collimated", np.inf), ("converging R=-50mm", -50e-3),
                 ("converging R=-20mm", -20e-3), ("tilt 5 mrad", None)):
    if name.startswith("tilt"):
        ph_in = k0*5e-3*X
    else:
        ph_in = 0.0 if not np.isfinite(Rc) else k0*R2/(2*Rc)
    E_in = (np.exp(-R2/w**2)*np.exp(1j*ph_in)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ea = apply_real_lens(E_in, prescription=rxp, wavelength=WL, dx=dx)
        Et = apply_real_lens_traced(E_in, prescription=rxp, wavelength=WL,
                                    dx=dx, ray_subsample=8, n_workers=1,
                                    on_undersample='silent',
                                    on_noncollimated='off',
                                    min_coarse_samples_per_aperture=0,
                                    on_pool_memory='silent')
    m = (np.abs(Ea) > 1e-3*np.abs(Ea).max()) & (R2 < (0.45*AP)**2)
    d = np.angle(Et*np.conj(Ea))[m]
    print(f"  {name:22s}: phase(traced) - phase(apply_real_lens): "
          f"mean={d.mean():+.4e} rad  std={d.std():.4e} rad "
          f"(={d.std()/k0*1e9:.4f} nm)  max|dev|={np.abs(d-d.mean()).max():.3e} rad")
    ea = np.abs(Et)[m]; eb = np.abs(Ea)[m]
    print(f"      |E| rel diff rms = {np.sqrt((((ea-eb)/max(eb.max(),1e-30))**2).mean()):.3e}")
