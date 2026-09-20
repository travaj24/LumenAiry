"""VERIFY-WP-C3 CLAIM 12.3 -- the d2 readout PERIOD on both transports.

Re-measures the branch's claim "the Collins period is 3846.09 um against the
requested 2867.20 um window", and checks what the SHIPPED docstrings say about
the standoff key against what the branch's own code now does.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, 'tests/unit'); sys.path.insert(0, 'tests')
import lumenairy as la
import test_niche_d2_chain_multi as D2

print('requested window  N_out*dx_out = %.4f um' % (D2._NOUT * D2._DXO * 1e6))
print('tile window       TILE*dx_out  = %.4f um' % (D2._TILE * D2._DXO * 1e6))

gA, gB, groups = D2._relay_groups()
A2, B2, C2, Dm = D2._group_abcd(gB, D2._WL)
R_A = D2._group_abcd(gA, D2._WL)[0] / D2._group_abcd(gA, D2._WL)[2]
R_g = R_A + D2._GAP
R_B = (A2 * R_g + B2) / (C2 * Dm and 1 or 1)  # placeholder, recomputed below
R_B = (A2 * R_g + B2) / (C2 * R_g + Dm)
fd = -R_B
env = D2._gauss()
carrier = la.TiltedCarrier(np.inf, -D2._TILT, 0.0)

for tr in ('sziklas', 'collins'):
    kw = dict(r_in=carrier, ray_subsample=D2._RS, n_workers=D2._NW,
              final_distance=fd, traced_kwargs=D2._TKW,
              final_leg='paraxial', transport=tr)
    fr = dict(dx_out=D2._DXO, N_out=D2._NOUT, on_replica='ignore')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(env, groups, D2._WL, D2._DX,
                                                focus_readout=fr, **kw)
    st = res.stages[-1]
    per = st.get('readout_period')
    print('transport=%-8s readout_period=%s um  route=%s reason=%s k1=%s'
          % (tr, (np.round(np.asarray(per) * 1e6, 4).tolist()
                  if per is not None else None),
             st.get('readout_route'), st.get('readout_route_reason'),
             st.get('readout_route_k1')))

# does the STANDOFF key still get refused on collins?
fr2 = dict(dx_out=D2._DXO, N_out=D2._NOUT, on_replica='ignore',
           standoff=1e-3)
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env, groups, D2._WL, D2._DX, focus_readout=fr2,
            r_in=carrier, ray_subsample=D2._RS, n_workers=D2._NW,
            final_distance=fd, traced_kwargs=D2._TKW, final_leg='paraxial',
            transport='collins')
    st = res.stages[-1]
    print("standoff key on transport='collins': ACCEPTED, route=%s reason=%s"
          % (st.get('readout_route'), st.get('readout_route_reason')))
except Exception as exc:
    print("standoff key on transport='collins': REFUSED -> %s: %s"
          % (type(exc).__name__, str(exc)[:120]))
