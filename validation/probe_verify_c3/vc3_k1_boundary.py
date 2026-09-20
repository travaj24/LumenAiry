"""VERIFY-WP-C3 CLAIM 13c(i) -- is the readout's K1 <= 1.0 boundary pinned?

Scans ``final_distance`` on the WP-C3 test file's own chain fixture to find a
leg whose readout K1 lands in ``(1, 2]``.  Such a leg is routed 'sziklas' by
the shipped bar and 'collins' by the mutated one, so it is the fixture the
missing decision test needs.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, 'tests/unit'); sys.path.insert(0, 'tests')
import lumenairy.propagators.carrier as CA
import test_c3_collins_default as T

env, dx, r_in, groups = T._chain_fixture()
print('%-12s %-12s %-10s %-16s %s' % ('final_dist', 'K1', 'route', 'reason', 'period um'))
for fd in (0.040, 0.045, 0.050, 0.055, 0.058, 0.060, 0.062, 0.065, 0.070, 0.075):
    kw = dict(r_in=r_in, ray_subsample=16, n_workers=1, traced_kwargs=T._TKW,
              final_leg='paraxial', final_distance=fd,
              focus_readout=dict(dx_out=0.5e-6, N_out=64))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = CA.propagate_traced_carrier_chain(env, groups, 1.31e-6, dx,
                                                  transport='collins', **kw)
        st = r.stages[-1]
        k1 = st.get('readout_route_k1')
        print('%-12.4g %-12s %-10s %-16s %s'
              % (fd, ('%.5g' % k1) if k1 is not None else None,
                 st.get('readout_route'), st.get('readout_route_reason'),
                 (None if st.get('readout_period') is None else
                  round(float(np.asarray(st['readout_period']).ravel()[0])*1e6, 3))))
    except Exception as exc:
        print('%-12.4g ERROR %s: %s' % (fd, type(exc).__name__, str(exc)[:90]))
