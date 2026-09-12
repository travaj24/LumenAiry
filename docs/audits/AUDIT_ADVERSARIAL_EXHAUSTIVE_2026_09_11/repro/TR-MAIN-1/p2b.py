"""Probe 2 (corrected): residual phasor against the oracle -- no 2D unwrap."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens, apply_real_lens_traced
from p2_opd_oracle import oracle_opd_on_exit_grid
WL = common.WL; k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

AP = 6e-3
rx = common.plano_convex(R=100e-3, t=4e-3, ap=AP)
print("f_paraxial =", 100e-3/(1.5168-1)*1e3, "mm ; f/# =", 100e-3/(1.5168-1)/AP)

for N in (512, 1024):
    dx = 1.3*AP/N
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x)
    mask = (X**2+Y**2) <= (0.45*AP)**2
    opl_or, xe, ye = oracle_opd_on_exit_grid(rx, X, Y)
    E_in = np.ones((N, N), dtype=np.complex128)
    for sub in (1, 8):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(
                E_in, prescription=rx, wavelength=WL, dx=dx,
                ray_subsample=sub, n_workers=1, on_undersample='silent',
                min_coarse_samples_per_aperture=0, on_pool_memory='silent')
        resid = np.angle(E*np.exp(-1j*k0*opl_or))       # in (-pi,pi]
        d = resid[mask]/k0
        A = np.stack([np.ones(d.size), X[mask], Y[mask]], 1)
        c,*_ = np.linalg.lstsq(A, d, rcond=None); r = d-A@c
        print(f"  N={N} dx={dx*1e6:.3f}um sub={sub}: "
              f"RMS = {r.std()*1e9:9.5f} nm  PV = {(r.max()-r.min())*1e9:9.5f} nm  "
              f"|resid|max(raw) = {np.abs(resid[mask]).max():.3e} rad")
