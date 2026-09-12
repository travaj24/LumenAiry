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
print("dx=%.4f um  sub*dx=%.2f um  f_eff~%.2f mm" % (dx*1e6, 8*dx*1e6, 58.0))
print("predicted order-1 upsample err (sub=8) = %.3f nm"
      % ((8*dx)**2/(8*0.058)*1e9))
for sub in (1, 4, 8, 16):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E = apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                                   ray_subsample=sub, n_workers=1,
                                   on_undersample='silent',
                                   min_coarse_samples_per_aperture=0,
                                   on_pool_memory='silent')
    r = np.angle(E*np.exp(-1j*k0*opl_or))
    rr = r[m]/k0
    print(f"sub={sub}: |E|max={np.abs(E).max():.4f} nonzero={np.count_nonzero(E)} "
          f"resid max={np.abs(rr).max()*1e12:.4f} pm  rms={rr.std()*1e12:.4f} pm")
    # row cut through the centre
    row = N//2
    cut = r[row, :]/k0
    ii = np.where(m[row])[0]
    print("   row-cut resid (pm) at cols", ii[::max(1,len(ii)//8)][:9],
          np.round(cut[ii[::max(1,len(ii)//8)][:9]]*1e12, 4))
