"""Probe 13b: a STRONG asphere (tens of um of departure) vs fit order."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements.lenses import surface_sag_general
from p13_asphere import inv_opl, sag
WL = common.WL; k0 = 2*np.pi/WL
AP = 8e-3
for scale, lbl in ((100.0, 'STRONG'), (1000.0, 'VERY STRONG')):
    A = {4: -1.0e2*scale, 6: 5.0e4*scale, 8: -2.0e7*scale, 10: 8.0e9*scale}
    rxa = {'wavelength': WL, 'aperture_diameter': AP,
           'surfaces': [
             {'radius': 60e-3, 'thickness': 4e-3, 'glass_before':'air',
              'glass_after':'_AUD_GLASS', 'conic': -0.6, 'aspheric_coeffs': A,
              'semi_diameter': AP/2},
             {'radius': -60e-3, 'thickness': 0.0, 'glass_before':'_AUD_GLASS',
              'glass_after':'air', 'semi_diameter': AP/2}],
           'thicknesses': [4e-3], 'stop_index': 0}
    h = np.array([AP/2*0.45/0.5*0.9])
    dep = 1e6*(sag(h**2, 60e-3, -0.6, A) - sag(h**2, 60e-3, 0.0, None))[0]
    N = 512; dx = 1.3*AP/N
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
    m = (X**2+Y**2) <= (0.40*AP)**2
    opl_or = inv_opl(rxa, X, Y)
    print(f"{lbl}: aspheric departure at r=3.24mm = {dep:.3f} um")
    for order in (4, 6, 8, 10):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(np.ones((N,N), np.complex128),
                prescription=rxa, wavelength=WL, dx=dx, ray_subsample=1,
                newton_poly_order=order, n_workers=1, on_undersample='silent',
                min_coarse_samples_per_aperture=0, on_pool_memory='silent')
        r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
        A_ = np.stack([np.ones(r.size), X[m], Y[m]], 1)
        c,*_ = np.linalg.lstsq(A_, r, rcond=None); rr = r-A_@c
        print(f"   order={order:2d} (sub=1, pure Newton on the forward fit): "
              f"|resid|max={np.abs(r).max()*1e9:10.4f} nm  "
              f"rms={rr.std()*1e9:10.4f} nm")
