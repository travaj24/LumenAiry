import sys, warnings
sys.path.insert(0, 'C:/tmp/lum_vc3/validation/probe_verify_c3')
sys.path.insert(0, 'C:/tmp/lum_vc3')
import numpy as np, vclib as V
V.anchor('C:/tmp/lum_vc3')
import lumenairy.propagators.carrier as CA
WL, FD = 1.31e-6, 8.0e-3
TKW = dict(on_undersample='silent', on_noncollimated='silent')
def singlet(R1,R2,d,glass,ap,name):
    return {'name':name,'aperture_diameter':ap,'thicknesses':[d],'surfaces':[
      {'radius':R1,'glass_before':'air','glass_after':glass,'conic':0.0,'radius_y':None,'conic_y':None,'aspheric_coeffs':None,'aspheric_coeffs_y':None},
      {'radius':R2,'glass_before':glass,'glass_after':'air','conic':0.0,'radius_y':None,'conic_y':None,'aspheric_coeffs':None,'aspheric_coeffs_y':None}]}
n,dx,w,r_in=256,60e-6,4.5e-3,60e-3
g=V.axis(n,dx); env=np.exp(-((g[None,:]**2+g[:,None]**2)/w**2)).astype(np.complex128)
p=singlet(60e-3,-60e-3,3e-3,'N-BK7',14e-3,'p')
groups=[{'prescription':p,'gap_before':20e-3},{'prescription':p,'gap_before':10e-3}]
def run(**kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r=CA.propagate_traced_carrier_chain(env,groups,WL,dx,r_in=r_in,ray_subsample=16,n_workers=1,traced_kwargs=TKW,**kw)
    return r,len(rec)
for tag,kw in [
  ('paraxial+collins',dict(final_distance=FD,final_leg='paraxial',transport='collins')),
  ('paraxial+sziklas',dict(final_distance=FD,final_leg='paraxial',transport='sziklas')),
  ('auto+collins',dict(final_distance=FD,final_leg='auto',transport='collins')),
  ('auto+sziklas',dict(final_distance=FD,final_leg='auto',transport='sziklas')),
  ('default_leg+collins',dict(final_distance=FD,transport='collins')),
  ('exact+collins',dict(final_distance=FD,final_leg='exact',transport='collins')),
]:
    try:
        r,nw=run(**kw)
        A=np.abs(np.asarray(r.field))**2
        dxo=V.pitch2(r.dx)[0]
        print(f'{tag:22s} dx={dxo*1e6:10.5f} N={A.shape[-1]:5d} centre={A[A.shape[-2]//2,A.shape[-1]//2]:.6f} peak={A.max():.6f} P={A.sum()*dxo*dxo:.6e} nw={nw}')
    except Exception as e:
        print(f'{tag:22s} RAISED {type(e).__name__}: {e}')
# also: the standalone free step from the exit plane
r0,_=run(final_distance=0.0, final_leg='paraxial')
# recover envelope
dx0=V.pitch2(r0.dx)[0]
envx=CA.carrier_referenced_envelope(np.asarray(r0.field), r0.R, WL, dx0)
for tr in ('collins','sziklas'):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        o=CA.propagate_carrier_referenced(envx, r0.R, FD, WL, dx0, transport=tr)
    A=np.abs(np.asarray(o.env))**2; d=V.pitch2(o.dx)[0]
    print(f'standalone {tr:9s} dx={d*1e6:10.5f} centre={A[A.shape[-2]//2,A.shape[-1]//2]:.6f} peak={A.max():.6f} P={A.sum()*d*d:.6e} nw={len(rec)}')
