"""Diagnose the newton_fit='spline' residual structure."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from p2_opd_oracle import oracle_opd_on_exit_grid
WL = common.WL; k0 = 2*np.pi/WL
AP = 8e-3; N = 512; dx = 1.3*AP/N
rx = {'wavelength': WL, 'aperture_diameter': AP,
      'surfaces': [
        {'radius': 60e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': '_AUD_GLASS', 'semi_diameter': 4e-3},
        {'radius': -60e-3, 'thickness': 0.0, 'glass_before': '_AUD_GLASS',
         'glass_after': 'air', 'semi_diameter': 4e-3}],
      'thicknesses': [4e-3], 'stop_index': 0}
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R = np.hypot(X, Y)
opl_or, _, _ = oracle_opd_on_exit_grid(rx, X, Y)
E_in = np.ones((N, N), np.complex128)
base = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=1, n_workers=1,
            on_undersample='silent', min_coarse_samples_per_aperture=0,
            on_pool_memory='silent', on_fit_domain_basis='silent')
warnings.simplefilter('ignore')
Ep = apply_real_lens_traced(E_in, newton_fit='polynomial', **base)
Es = apply_real_lens_traced(E_in, newton_fit='spline', **base)
# residual vs oracle along a radial cut
row = N//2
for tag, E in (('poly', Ep), ('spline', Es)):
    res = np.angle(E*np.exp(-1j*k0*opl_or))
    cut = res[row, N//2:]
    rr = R[row, N//2:]
    sel = rr < 0.5*AP
    print(f"{tag}: |resid| along +x cut (rad) at r(mm)=")
    idx = np.linspace(0, sel.sum()-1, 12).astype(int)
    print("   r:", np.round(rr[sel][idx]*1e3, 3))
    print("   e:", np.round(cut[sel][idx], 5))
print("\nspline vs poly field: max|dphase| =",
      float(np.abs(np.angle(Es[R < 0.45*AP]*np.conj(Ep[R < 0.45*AP]))).max()),
      " rms =", float(np.angle(Es[R<0.45*AP]*np.conj(Ep[R<0.45*AP])).std()))
print("nonzero: poly", int((np.abs(Ep)>0).sum()), " spline", int((np.abs(Es)>0).sum()))
