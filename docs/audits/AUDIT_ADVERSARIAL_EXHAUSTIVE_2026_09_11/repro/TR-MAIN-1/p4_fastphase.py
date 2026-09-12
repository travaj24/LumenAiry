"""Probe 4: fast_analytic_phase=True vs the ASM plane-wave pass."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
from lumenairy.elements._lens_traced import _geometric_lens_phase
WL = common.WL; k0 = 2*np.pi/WL

def run(tag, rx, AP, N=1024):
    dx = 1.3*AP/N
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
    m = (X**2+Y**2) <= (0.45*AP)**2
    E_in = np.ones((N, N), np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Epw = apply_real_lens(np.ones_like(E_in), prescription=rx,
                              wavelength=WL, dx=dx)
        g = _geometric_lens_phase(rx, WL, dx, N)
        Ef = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                                    ray_subsample=8, n_workers=1,
                                    fast_analytic_phase=True,
                                    on_undersample='silent',
                                    min_coarse_samples_per_aperture=0,
                                    on_pool_memory='silent')
        Es = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                                    ray_subsample=8, n_workers=1,
                                    fast_analytic_phase=False,
                                    on_undersample='silent',
                                    min_coarse_samples_per_aperture=0,
                                    on_pool_memory='silent')
    d = np.angle(np.exp(1j*(g - np.angle(Epw))))[m]
    d = d - np.median(d)
    d = (d + np.pi) % (2*np.pi) - np.pi
    print(f"{tag} N={N} dx={dx*1e6:.3f}um")
    print(f"   _geometric_lens_phase - angle(ASM pw): rms={d.std():.5e} rad "
          f"= {d.std()/k0*1e9:8.3f} nm ; max={np.abs(d).max()/k0*1e9:8.3f} nm")
    dd = np.angle(Ef*np.conj(Es))[m]
    dd = dd - np.median(dd); dd = (dd+np.pi) % (2*np.pi) - np.pi
    print(f"   FIELD  fast vs full: rms={dd.std():.5e} rad = "
          f"{dd.std()/k0*1e9:8.3f} nm ; max={np.abs(dd).max()/k0*1e9:8.3f} nm")

run("f/32 plano-convex", common.plano_convex(R=100e-3, t=4e-3, ap=6e-3), 6e-3)
run("f/5 singlet (R=+-51.68, t=5, ap=24mm)", common.singlet_f5(name='_AUD_GLASS'), 24e-3, N=2048)
run("f/8 biconvex ap=8mm",
    {'wavelength': WL, 'aperture_diameter': 8e-3,
     'surfaces': [{'radius': 60e-3, 'thickness': 4e-3, 'glass_before': 'air',
                   'glass_after': '_AUD_GLASS', 'semi_diameter': 4e-3},
                  {'radius': -60e-3, 'thickness': 0.0,
                   'glass_before': '_AUD_GLASS', 'glass_after': 'air',
                   'semi_diameter': 4e-3}],
     'thicknesses': [4e-3], 'stop_index': 0}, 8e-3)
