"""TR-INFRA probe 5: _opl_by_backward_trace under both thickness conventions."""
import warnings
import numpy as np
from lumenairy.elements import _lens_traced as T
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.io.prescriptions_builders import make_singlet
lam=587.6e-9; k0=2*np.pi/lam; N=128; dx=50e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x)
E=np.exp(-(X**2+Y**2)/(1.5e-3)**2).astype(np.complex128)
for label, thick in (('len(thick)=n-1', None), ('len(thick)=n  ', [2e-3, 50e-3])):
    P=make_singlet(0.100,-0.100,2e-3,'N-BK7',aperture=0.006)
    if thick: P['thicknesses']=thick
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ea=apply_real_lens(E,prescription=P,wavelength=lam,dx=dx)
        opl=T._opl_by_backward_trace(Ea,P,lam,dx,N,8)
    fin=np.isfinite(opl)
    print(f"  {label}: finite {100*fin.mean():5.1f}%  OPL range "
          f"[{np.nanmin(opl):+.6e}, {np.nanmax(opl):+.6e}] m"
          f"  PV = {np.nanmax(opl)-np.nanmin(opl):.6e} m", flush=True)
