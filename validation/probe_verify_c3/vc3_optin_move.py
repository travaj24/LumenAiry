"""Does an EXISTING transport='collins' caller get told their answer moved?"""
import sys, warnings
import numpy as np
sys.path.insert(0,'tests/unit'); sys.path.insert(0,'tests')
import lumenairy.propagators.carrier as CA
try:
    import test_c3_collins_default as T; FIX=T._chain_fixture(); TKW=T._TKW
except Exception:
    import test_audit2609_b4_collins_transport as B4; FIX=B4._chain_fixture(); TKW=B4._CHAIN_TKW
env,dx,r_in,groups=FIX
for fd in (0.045,):
    kw=dict(r_in=r_in,ray_subsample=16,n_workers=1,traced_kwargs=TKW,
            final_leg='paraxial',final_distance=fd,
            focus_readout=dict(dx_out=0.5e-6,N_out=64),transport='collins')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r=CA.propagate_traced_carrier_chain(env,groups,1.31e-6,dx,**kw)
    st=r.stages[-1]
    print('tree', CA.__file__)
    print('fd=%.4g  |E|max=%.6f  route=%s reason=%s k1=%s'
          % (fd, float(np.abs(np.asarray(r.field)).max()),
             st.get('readout_route'), st.get('readout_route_reason'),
             st.get('readout_route_k1')))
    print('warnings emitted: %d' % len(rec))
    for w in rec: print('   ', type(w.message).__name__, str(w.message)[:150].replace('\n',' '))
