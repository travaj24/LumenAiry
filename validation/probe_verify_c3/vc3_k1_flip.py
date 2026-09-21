"""Does the shipped K1 <= 1.0 boundary DECIDE anything on a fixture in (1,2]?
And does the new fallback introduce a RAISE where 5.48.1's explicit
transport='collins' returned a field?"""
import sys, warnings
import numpy as np
sys.path.insert(0,'tests/unit'); sys.path.insert(0,'tests')
import lumenairy.propagators.carrier as CA
try:
    import test_c3_collins_default as T
    FIX = T._chain_fixture(); TKW = T._TKW
except Exception:
    import test_audit2609_b4_collins_transport as B4
    FIX = B4._chain_fixture(); TKW = B4._CHAIN_TKW
env, dx, r_in, groups = FIX
print('tree', CA.__file__)
for fd in (0.045, 0.05):
    for tr in ('collins', 'sziklas'):
        kw = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                  final_leg='paraxial', final_distance=fd,
                  focus_readout=dict(dx_out=0.5e-6, N_out=64), transport=tr)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx, **kw)
            st = r.stages[-1]
            print('fd=%-7.4g tr=%-8s OK  route=%-8s reason=%-15s k1=%s  |E|max=%.6g'
                  % (fd, tr, st.get('readout_route'), st.get('readout_route_reason'),
                     (None if st.get('readout_route_k1') is None
                      else round(st['readout_route_k1'], 5)),
                     float(np.abs(np.asarray(r.field)).max())))
        except Exception as exc:
            print('fd=%-7.4g tr=%-8s RAISE %s: %s'
                  % (fd, tr, type(exc).__name__, str(exc)[:100]))
