"""How much does the returned field's accuracy depend on whether the internal
inverse-characteristic model engages?  And is there ANY user-visible signal?"""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from p2_opd_oracle import oracle_opd_on_exit_grid
WL = common.WL; k0 = 2*np.pi/WL
AP = 8e-3; N = 768; dx = 1.3*AP/N
rx = {'wavelength': WL, 'aperture_diameter': AP,
      'surfaces': [
        {'radius': 60e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': '_AUD_GLASS', 'semi_diameter': 4e-3},
        {'radius': -60e-3, 'thickness': 0.0, 'glass_before': '_AUD_GLASS',
         'glass_after': 'air', 'semi_diameter': 4e-3}],
      'thicknesses': [4e-3], 'stop_index': 0}
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
m = (X**2+Y**2) <= (0.45*AP)**2
opl_or, _, _ = oracle_opd_on_exit_grid(rx, X, Y)
E_in = np.ones((N, N), np.complex128)
base = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
            on_undersample='silent', min_coarse_samples_per_aperture=0,
            on_pool_memory='silent')
cases = [('default (polynomial)', {}),
         ("newton_fit='spline'", {'newton_fit': 'spline',
                                  'on_fit_domain_basis': 'silent'}),
         ("inversion_method='fit'", {'inversion_method': 'fit'}),
         ("use_gpu-equivalent (inverse_map=False)", {'inverse_map': False})]
for tag, extra in cases:
    d = {}
    with warnings.catch_warnings(record=True) as wl_:
        warnings.simplefilter('always')
        try:
            E = apply_real_lens_traced(E_in, _imap_out=d, **base, **extra)
        except Exception as e:
            print(f"{tag}: {type(e).__name__}: {str(e)[:100]}"); continue
        warns = [str(mm.message)[:80] for mm in wl_]
    r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
    A = np.stack([np.ones(r.size), X[m], Y[m]], 1)
    c,*_ = np.linalg.lstsq(A, r, rcond=None); rr = r-A@c
    print(f"{tag:40s}: imap engaged={d.get('engaged')} guard={d.get('guard')}  "
          f"|resid|max={np.abs(r).max()*1e9:9.4f} nm  rms={rr.std()*1e9:9.4f} nm  "
          f"user warnings={len(warns)}")
