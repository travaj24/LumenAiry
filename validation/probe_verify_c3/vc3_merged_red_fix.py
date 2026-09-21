"""The merged-tree red, and the two candidate remedies, measured."""
import sys
import numpy as np
sys.path.insert(0,'tests/unit'); sys.path.insert(0,'tests')
import lumenairy.propagators.carrier as CA
import test_wave5_h2_near_focus_table as T
from lumenairy import propagate_carrier_referenced, carrier_referenced_reconstruct
q_a = complex(0.0, -T.ZR); w_a = T._w_of_q(q_a); z = 1.5*T.ZR
w_out = T._w_of_q(q_a+z); dx = 10.0*w_out/T.N_IN
x = T._axis(T.N_IN, dx)
env = np.exp(-(x[None,:]**2 + x[:,None]**2)/w_a**2).astype(np.complex128)
theta = T.LAM/(np.pi*w_a); predicted = T.K*abs(z)*theta**4/8.0
print('tau=%r  predicted quartic=%.6e' % (CA._GAP_KERNEL_ACCURACY_TAU, predicted))
for tag, kw in (('AS SHIPPED (no transport, auto)', dict(gap_kernel='auto')),
                ("transport='sziklas', auto", dict(gap_kernel='auto', transport='sziklas')),
                ("default transport, gap_kernel='exact'", dict(gap_kernel='exact')),
                ("transport='collins', auto", dict(gap_kernel='auto', transport='collins'))):
    try:
        got = propagate_carrier_referenced(env, float('inf'), z, T.LAM, dx, **kw)
        fld = carrier_referenced_reconstruct(got.env, got.R, T.LAM, got.dx)
        xo = T._axis(T.N_IN, got.dx)
        rel = T._rel(fld, T._q_field(xo, xo, q_a, z))
        print('%-40s rel=%.6e  ratio=%.4f  %s' % (tag, rel, rel/predicted,
              'PASS' if 0.5 < rel/predicted < 2.0 else 'FAIL'))
    except Exception as e:
        print('%-40s RAISE %s: %s' % (tag, type(e).__name__, str(e)[:90]))
