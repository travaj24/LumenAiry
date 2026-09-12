import warnings
import numpy as np
from lumenairy.elements import _lens_traced as T
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.io.prescriptions_builders import make_singlet
lam=587.6e-9; k0=2*np.pi/lam
for N,dx in ((256, 30e-6),):
    x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
    E=np.exp(-(X**2+Y**2)/(1.5e-3)**2).astype(np.complex128)
    P=make_singlet(0.100,-0.100,2e-3,'N-BK7',aperture=0.006)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ea=apply_real_lens(E,prescription=P,wavelength=lam,dx=dx)
        opl_b=T._opl_by_backward_trace(Ea,P,lam,dx,N,8)
        An=T.apply_real_lens_traced(E,prescription=P,wavelength=lam,dx=dx,ray_subsample=8,
              on_undersample='silent',on_noncollimated='silent')
        Ab=T.apply_real_lens_traced(E,prescription=P,wavelength=lam,dx=dx,ray_subsample=8,
              inversion_method='backward_trace',on_undersample='silent',on_noncollimated='silent')
    fin=np.isfinite(opl_b)
    print(f"N={N} dx={dx*1e6:.0f}um: backward OPL finite on {100*fin.mean():.1f}% of the grid"
          f"; inside the beam (r<2w={3e-3:.1e} m) finite on {100*fin[r<3e-3].mean():.1f}%", flush=True)
    for rr,lab in ((1.5e-3,'r<w '),(3.0e-3,'r<2w')):
        m=(r<rr)&fin&(np.abs(An)>1e-6*np.abs(An).max())
        d=np.angle(Ab*np.conj(An))[m]
        d=d-np.mean(d)
        print(f"   {lab}: backward-vs-newton exit phase rms {np.sqrt(np.mean(d**2)):.4e} rad"
              f" = {np.sqrt(np.mean(d**2))/k0*1e9:8.2f} nm rms OPD, max {np.abs(d).max()/k0*1e9:8.2f} nm"
              f"   (docstring claims ~35-40 nm)", flush=True)
