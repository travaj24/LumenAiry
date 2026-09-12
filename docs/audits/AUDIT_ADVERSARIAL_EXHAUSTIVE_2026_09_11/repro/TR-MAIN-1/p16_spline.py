"""Isolate the newton_fit='spline' accuracy against the polynomial default."""
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
base = dict(prescription=rx, wavelength=WL, dx=dx, n_workers=1,
            on_undersample='silent', min_coarse_samples_per_aperture=0,
            on_pool_memory='silent', on_fit_domain_basis='silent')
for fit in ('polynomial', 'spline'):
    for sub, imap in ((1, False), (8, False), (8, True)):
        d = {}
        with warnings.catch_warnings(record=True) as wl_:
            warnings.simplefilter('always')
            E = apply_real_lens_traced(E_in, newton_fit=fit, ray_subsample=sub,
                                       inverse_map=imap, _imap_out=d, **base)
            ws = [str(mm.message)[:70] for mm in wl_]
        r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
        A = np.stack([np.ones(r.size), X[m], Y[m]], 1)
        c,*_ = np.linalg.lstsq(A, r, rcond=None); rr = r-A@c
        print(f"{fit:11s} sub={sub} imap_req={imap}: engaged={d.get('engaged')} "
              f"|resid|max={np.abs(r).max()*1e9:10.4f} nm rms={rr.std()*1e9:10.4f} nm")
        if d and fit == 'spline' and sub == 8 and imap:
            print("    _imap_out:", {k: v for k, v in d.items()
                                     if k in ('engaged','guard','reason','why',
                                              'refused','degree','n_samples')})
            print("    keys:", sorted(d.keys())[:20])
        if ws: print("    warnings:", ws)
