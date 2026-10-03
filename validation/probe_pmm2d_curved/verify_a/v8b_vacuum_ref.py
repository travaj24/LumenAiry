"""V8b -- is a vacuum spacer on top of the stack a no-op under a map?  A
PATTERNED all-vacuum layer (its own region eig) and a UNIFORM vacuum layer
(the shared geometric eig) of thickness 0.2 above the stripe, vs the stripe
alone: R / T and the zero-order Jones after the exp(2 i k0 d) phase.
Unmapped, identity map, sine 0.02 on a uniform (u, v) lattice, harm_asym on
the preimage walls; M = 5, 7.  Output: v8b_vacuum_ref_<build>.txt."""
import _vcommon as C
import numpy as np

from lumenairy.elements.pmm._curvemap import IdentityMap

d = 0.2
for tag, cm in (("none", None), ("ident3", IdentityMap(3, 3, C.P, C.P)),
                ("sine0.02_u3", C.stretch_map(C.sine(0.02),
                                              np.linspace(0, 1, 4),
                                              np.linspace(0, 1, 4))),
                ("asym_pre", C.stretch_map(C.HarmonicStretch(0.10, 0.04,
                                                             0.9)))):
    for M in (5, 7):
        a = C.stack_solve(cm, [C.cell("stripe")], M)
        b = C.stack_solve(cm, [C.cell("vac"), C.cell("stripe")], M,
                          depths=[d, C.DEPTH])
        b2 = C.stack_solve(cm, [1.0 + 0j, C.cell("stripe")], M,
                           depths=[d, C.DEPTH])
        ph = np.exp(2j * C.K0 * d)
        print(tag, M,
              "patterned-vac dRT=%.2e dJph=%.2e | uniform-vac dRT=%.2e "
              "dJph=%.2e | closure a=%.1e" % (
                  max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()),
                  np.abs(b[3] - a[3] * ph).max(),
                  max(np.abs(a[1] - b2[1]).max(), np.abs(a[2] - b2[2]).max()),
                  np.abs(b2[3] - a[3] * ph).max(), C.closure(a[1], a[2])),
              flush=True)
