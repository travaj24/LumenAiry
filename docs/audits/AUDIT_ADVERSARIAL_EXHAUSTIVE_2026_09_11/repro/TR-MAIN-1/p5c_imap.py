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
for imap in (True, False):
    for sub in (1, 4, 8, 16):
        d = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL,
                                       dx=dx, ray_subsample=sub, n_workers=1,
                                       on_undersample='silent',
                                       min_coarse_samples_per_aperture=0,
                                       on_pool_memory='silent',
                                       inverse_map=imap, _imap_out=d)
        r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
        print(f"inverse_map={imap} sub={sub:2d}: engaged={d.get('engaged')} "
              f"guard={d.get('guard')}  |resid|max={np.abs(r).max()*1e12:10.4f} pm"
              f"  rms={r.std()*1e12:10.4f} pm")
