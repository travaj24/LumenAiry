"""Quantify the float32 penalty of _lens_jax under JAX's DEFAULT config,
against an exact independent ray-trace OPD oracle, for a short and a LONG
prescription (long back gap => large absolute OPL => float32 cancellation)."""
import os, sys, warnings, numpy as np, jax
X64 = os.environ.get('X64','0')=='1'
if X64: jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
from lumenairy.glass import get_glass_index
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax as TJ
lam=0.5876e-6; k0=2*np.pi/lam; ng=float(get_glass_index('N-BK7',lam))
print('x64 =', jax.config.read('jax_enable_x64'))

def case(extra_gap, tag):
    R1,R2,d,AP = 25e-3, float('inf'), 3e-3, 6.0e-3
    surf=[{'radius':R1,'conic':0.0,'aspheric_coeffs':None,'radius_y':None,'conic_y':None,
           'aspheric_coeffs_y':None,'glass_before':'air','glass_after':'N-BK7'},
          {'radius':R2,'conic':0.0,'aspheric_coeffs':None,'radius_y':None,'conic_y':None,
           'aspheric_coeffs_y':None,'glass_before':'N-BK7','glass_after':'air'}]
    th=[d]
    if extra_gap>0:
        surf.append({'radius':float('inf'),'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                     'conic_y':None,'aspheric_coeffs_y':None,'glass_before':'air','glass_after':'air'})
        th=[d, extra_gap]
    rx={'name':'s','aperture_diameter':AP,'surfaces':surf,'thicknesses':th}
    N,dxv=256,20e-6
    x=(np.arange(N)-N/2)*dxv; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
    E=np.ones((N,N),dtype=np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eo=np.asarray(TJ(E,prescription=rx,wavelength=lam,dx=dxv,ray_subsample=4,
                         cheb_order=10,newton_iters=14))
    hh=np.linspace(0.0,0.5*AP*1.02,400001); hh[0]=1e-12
    xo,opl,L,Nz=trace_singlet(hh,R1,R2,d,ng,extra_gap)
    rho=np.abs(xo); ref_opd=np.interp(r,rho,opl-opl[0],left=0.0,right=np.nan)
    m=(np.abs(Eo)>0)&(r<0.45*AP)&np.isfinite(ref_opd)
    dphi=np.angle(np.exp(1j*(np.angle(Eo[m])-k0*ref_opd[m])))
    dphi=np.angle(np.exp(1j*(dphi-np.angle(np.sum(np.exp(1j*dphi))))))
    print('  %-34s total OPL ~%.4g m   OPD err rms %.5g waves  max %.5g waves   out dtype %s'
          %(tag, float(opl.max()), np.sqrt(np.mean(dphi**2))/(2*np.pi),
            np.abs(dphi).max()/(2*np.pi), Eo.dtype))
case(0.0,      'singlet only (OPL ~ 7 mm)')
case(0.20,     'singlet + 200 mm air gap')
case(1.00,     'singlet + 1 m air gap')
