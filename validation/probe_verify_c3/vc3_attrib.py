"""Attribute the merged-tree digest move: C5 item 1 (tau) or item 3 (replica_fill)?"""
import hashlib, sys, warnings
import numpy as np
import lumenairy as la
import lumenairy.propagators.carrier as CA
sys.path.insert(0,'tests/unit'); sys.path.insert(0,'tests')
import test_niche_d2_chain_multi as D2
def dig(a):
    a=np.ascontiguousarray(np.asarray(a)); return hashlib.sha256(a.tobytes()).hexdigest()[:16]
gA,gB,groups=D2._relay_groups()
A2,B2,C2,Dm=D2._group_abcd(gB,D2._WL)
R_A=D2._group_abcd(gA,D2._WL)[0]/D2._group_abcd(gA,D2._WL)[2]
R_g=R_A+D2._GAP; fd=-((A2*R_g+B2)/(C2*R_g+Dm))
env=D2._gauss(); carrier=la.TiltedCarrier(np.inf,-D2._TILT,0.0)
print('tau=',CA._GAP_KERNEL_ACCURACY_TAU)
for tr in ('sziklas','collins'):
    for fill in (None,'repeat','zero'):
        fr=dict(dx_out=D2._DXO,N_out=D2._NOUT,on_replica='ignore')
        if fill: fr['replica_fill']=fill
        kw=dict(r_in=carrier,ray_subsample=D2._RS,n_workers=D2._NW,
                final_distance=fd,traced_kwargs=D2._TKW,final_leg='paraxial',
                focus_readout=fr,transport=tr)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r=la.propagate_traced_carrier_chain(env,groups,D2._WL,D2._DX,**kw)
        print('  transport=%-8s replica_fill=%-7s digest=%s'%(tr,str(fill),dig(r.field)))
# and with tau forced off
CA._GAP_KERNEL_ACCURACY_TAU=None
print('tau forced None:')
for tr in ('sziklas','collins'):
    fr=dict(dx_out=D2._DXO,N_out=D2._NOUT,on_replica='ignore',replica_fill='repeat')
    kw=dict(r_in=carrier,ray_subsample=D2._RS,n_workers=D2._NW,final_distance=fd,
            traced_kwargs=D2._TKW,final_leg='paraxial',focus_readout=fr,transport=tr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=la.propagate_traced_carrier_chain(env,groups,D2._WL,D2._DX,**kw)
    print('  transport=%-8s replica_fill=repeat  digest=%s'%(tr,dig(r.field)))
