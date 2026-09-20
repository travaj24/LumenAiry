import sys, warnings
sys.path.insert(0,'C:/tmp/lum_vc3/validation/probe_verify_c3'); sys.path.insert(0,'C:/tmp/lum_vc3')
import numpy as np, vclib as V
V.anchor('C:/tmp/lum_vc3')
import lumenairy.propagators.carrier as CA
WL=1.31e-6
def singlet(R1,R2,d,g,ap,nm):
    return {'name':nm,'aperture_diameter':ap,'thicknesses':[d],'surfaces':[
      {'radius':R1,'glass_before':'air','glass_after':g,'conic':0.0,'radius_y':None,'conic_y':None,'aspheric_coeffs':None,'aspheric_coeffs_y':None},
      {'radius':R2,'glass_before':g,'glass_after':'air','conic':0.0,'radius_y':None,'conic_y':None,'aspheric_coeffs':None,'aspheric_coeffs_y':None}]}
for n in (256,512,1024):
    dx=60e-6*256/n; g=V.axis(n,dx)
    env=np.exp(-((g[None,:]**2+g[:,None]**2)/4.5e-3**2)).astype(np.complex128)
    p=singlet(60e-3,-60e-3,3e-3,'N-BK7',14e-3,'p')
    gr=[{'prescription':p,'gap_before':20e-3},{'prescription':p,'gap_before':10e-3}]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        r=CA.propagate_traced_carrier_chain(env,gr,WL,dx,r_in=60e-3,ray_subsample=16,n_workers=1,
            traced_kwargs=dict(on_undersample='silent',on_noncollimated='silent'),final_leg='paraxial',final_distance=0.0)
    d=V.pitch2(r.dx)[0]; A=np.abs(np.asarray(r.field))**2
    x=V.axis(A.shape[-1],d); Px=A.sum(axis=0); tot=Px.sum()
    m2=float((Px*x**2).sum()/tot); w=2*np.sqrt(m2)
    # 1/e^2 intensity (= 1/e amplitude) radius read off the profile
    prof=Px/Px.max(); i=np.argmax(prof)
    half=np.where(prof[i:]<np.exp(-2.0))[0]
    r_e2 = x[i+half[0]] if half.size else float('nan')
    ev=CA.carrier_referenced_envelope(np.asarray(r.field),r.R,WL,d)
    rx,ry,tx,ty=CA._collins_input_box(ev,d,d,WL,CA._COLLINS_TAIL_FRAC)
    print("N=%4d exit dx=%9.5f um  R=%.6f m  w_2ndmoment=%.4f mm  r_1/e-marginal=%.4f mm  support(1e-6)=%.4f mm" % (n,d*1e6,r.R,w*1e3,r_e2*1e3,rx*1e3))
