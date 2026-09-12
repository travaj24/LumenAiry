"""newton_fit='spline' returns an ALL-ZERO field whenever any launch ray is
vignetted (which writes NaN into the three launch grids that RectBivariateSpline
is built on).  Mechanism check."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy import raytrace as rt
WL = common.WL
AP = 8e-3; N = 384; dx = 1.3*AP/N
def rx_of(sd):
    s = [{'radius': 60e-3, 'thickness': 4e-3, 'glass_before': 'air',
          'glass_after': '_AUD_GLASS'},
         {'radius': -60e-3, 'thickness': 0.0, 'glass_before': '_AUD_GLASS',
          'glass_after': 'air'}]
    if sd is not None:
        for d in s: d['semi_diameter'] = sd
    return {'wavelength': WL, 'aperture_diameter': AP, 'surfaces': s,
            'thicknesses': [4e-3], 'stop_index': 0}
E_in = np.ones((N, N), np.complex128)
base = dict(wavelength=WL, dx=dx, ray_subsample=4, n_workers=1,
            on_undersample='silent', min_coarse_samples_per_aperture=0,
            on_pool_memory='silent', on_fit_domain_basis='silent')
print("launch_radius = 0.75*aperture =", 0.75*AP*1e3, "mm")
for sd, lbl in ((4e-3, 'semi_diameter=4mm (=aperture/2): marginal launch rays VIGNETTED'),
                (None, 'no semi_diameter: nothing vignetted'),
                (7e-3, 'semi_diameter=7mm > launch_radius*? '),):
    rx = rx_of(sd)
    # how many launch rays die?
    pres_no_ap = dict(rx); pres_no_ap.pop('aperture_diameter', None)
    surfs = rt.surfaces_from_prescription(pres_no_ap)
    lr = 0.75*AP; sub = 4
    nl = max(8, int(2*lr/(dx*sub)));  nl += (nl % 2 == 0)
    xs = np.linspace(-lr, lr, nl); Xi, Yi = np.meshgrid(xs, xs, indexing='ij')
    rays = rt._make_bundle(x=Xi.ravel(), y=Yi.ravel(),
                           L=np.zeros(Xi.size), M=np.zeros(Xi.size),
                           wavelength=WL)
    res = rt.trace(rays, surfs, WL, output_filter='last')
    ndead = int((~res.image_rays.alive).sum())
    out = {}
    with warnings.catch_warnings(record=True) as wl_:
        warnings.simplefilter('always')
        Ep = apply_real_lens_traced(E_in, prescription=rx, newton_fit='polynomial', **base)
        Es = apply_real_lens_traced(E_in, prescription=rx, newton_fit='spline', **base)
        ws = [str(m.message)[:60] for m in wl_]
    print(f"  {lbl}")
    print(f"     dead launch rays = {ndead}/{rays.x.size}; "
          f"nonzero pixels: polynomial={int((np.abs(Ep)>0).sum())}, "
          f"spline={int((np.abs(Es)>0).sum())}; warnings={len(ws)}")
