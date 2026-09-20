"""VERIFY-WP-C3 CLAIM 11e -- chain-level A/B across the four default states.

Runs the same representative chain calls and digests the returned field, the
readout route and the readout period, so the C3 default and the C5 tau can be
attributed separately.
"""
import hashlib, json, sys, warnings
import numpy as np
import lumenairy as la
import lumenairy.propagators.carrier as CA
sys.path.insert(0, 'tests/unit'); sys.path.insert(0, 'tests')
import test_niche_d2_chain_multi as D2

def dig(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]

gA, gB, groups = D2._relay_groups()
A2, B2, C2, Dm = D2._group_abcd(gB, D2._WL)
R_A = D2._group_abcd(gA, D2._WL)[0] / D2._group_abcd(gA, D2._WL)[2]
R_g = R_A + D2._GAP
fd = -((A2 * R_g + B2) / (C2 * R_g + Dm))
env = D2._gauss()
carrier = la.TiltedCarrier(np.inf, -D2._TILT, 0.0)

rows = []
for tr in ('sziklas', 'collins', None):
    for tag, fr in (('readout', dict(dx_out=D2._DXO, N_out=D2._NOUT,
                                     on_replica='ignore')),
                    ('readout+standoff', dict(dx_out=D2._DXO, N_out=D2._NOUT,
                                              on_replica='ignore',
                                              standoff=2e-3)),
                    ('no-readout', None)):
        kw = dict(r_in=carrier, ray_subsample=D2._RS, n_workers=D2._NW,
                  final_distance=fd, traced_kwargs=D2._TKW,
                  final_leg='paraxial', focus_readout=fr)
        if tr is not None:
            kw['transport'] = tr
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = la.propagate_traced_carrier_chain(env, groups, D2._WL,
                                                      D2._DX, **kw)
            st = r.stages[-1]
            rows.append(dict(transport=str(tr), case=tag, digest=dig(r.field),
                             route=st.get('readout_route'),
                             reason=st.get('readout_route_reason'),
                             k1=st.get('readout_route_k1'),
                             period=(None if st.get('readout_period') is None
                                     else float(np.asarray(
                                         st['readout_period']).ravel()[0]))))
        except Exception as exc:
            rows.append(dict(transport=str(tr), case=tag,
                             error='%s: %s' % (type(exc).__name__,
                                               str(exc)[:110])))
print('tau =', CA._GAP_KERNEL_ACCURACY_TAU, ' tree =', CA.__file__)
for r in rows:
    print('%-8s %-17s %-17s route=%-8s reason=%-15s k1=%-8s period=%s'
          % (r['transport'], r['case'], r.get('digest', r.get('error', ''))[:17],
             r.get('route'), r.get('reason'),
             (None if r.get('k1') is None else round(r['k1'], 5)),
             (None if r.get('period') is None else round(r['period'] * 1e6, 3))))
out = sys.argv[1] if len(sys.argv) > 1 else 'vc3_chain_ab.json'
json.dump({'tau': CA._GAP_KERNEL_ACCURACY_TAU, 'tree': CA.__file__,
           'rows': rows}, open(out, 'w'), indent=1, default=str)
print('wrote', out)
