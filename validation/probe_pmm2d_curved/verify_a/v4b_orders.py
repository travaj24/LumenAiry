"""V4b -- the far-field ORDER COUNT under a map: the zero-order Jones of
the stripe at n_orders 2 vs 3, the symmetry-forbidden cross-polarisation
|J[1, 0]| (a y-uniform stripe under an x-only map), the TE error vs the 1-D
oracle; harm_asym and sine 0.02 on the preimage walls, and the unmapped
per-layer solve on the same walls; M = 6, 8, 10.  Output:
v4b_orders_<build>.txt (printed lines)."""
import _vcommon as C
import numpy as np

f = C.HarmonicStretch(0.10, 0.04, 0.9)
cm = C.stretch_map(f)
cms = C.stretch_map(C.sine(0.02))
for M in (6, 8, 10):
    for tag, m in (("asym", cm), ("sine0.02", cms), ("none", None)):
        if m is None:
            a = C.ref_perlayer([C.cell("stripe")], M, C.XW, C.YW, n_orders=2)
            b = C.ref_perlayer([C.cell("stripe")], M, C.XW, C.YW, n_orders=3)
        else:
            a = C.stack_solve(m, [C.cell("stripe")], M, n_orders=2)
            b = C.stack_solve(m, [C.cell("stripe")], M, n_orders=3)
        print(M, tag, "dJ(no2 vs no3)=%.2e" % np.abs(a[3] - b[3]).max(),
              "Jxy_leak no3=%.2e" % abs(b[3][1, 0]),
              "Jyx=%.2e" % abs(b[3][0, 1]),
              "te_err=%.2e" % C.stripe_err(b[0], b[1], b[2], 1,
                                           C.oracle_1d("te")), flush=True)
