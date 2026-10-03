"""Diagnostic: is the hand-built 4x4 map's slow self-convergence a property of
the MAP (its split disk cells) or of the device?  The LONE disk layer (layer
1 only, 0.3 thick, same half-spaces) on (a) the hand-built 4x4 map and (b)
the shipped Phase B/C 3x3 circle map (shapes=), at rung M.

  python v3m_lone.py <a|b> <M>
Output: v3m_lone_<a|b>_M<M>_<build>.json"""
import sys
import time

import v3m_geom as G
from _ve import closure, dump, solve

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle

arm, M = sys.argv[1], int(sys.argv[2])
t0 = time.perf_counter()
if arm == "a":
    tm, _ = G.build()
    c1, _ = G.cells()
    st = PMM2DStackPure(G.P, G.P, n_superstrate=G.N_SUP, n_substrate=G.N_SUB,
                        n_modes=M, n_orders=3, cmap=tm)
    st.add_layer(G.D1, eps_cell=c1)
else:
    st = PMM2DStackPure(G.P, G.P, n_superstrate=G.N_SUP, n_substrate=G.N_SUB,
                        n_modes=M, n_orders=3)
    st.add_layer(G.D1, shapes=[Circle(G.CX, G.CY, G.R, G.EPS_D)],
                 background_eps=1.0)
o, R, T, J = solve(st, G.WL)
out = dict(arm=arm, M=M, orders=o, R=R, T=T, closure=closure((o, R, T, J)),
           wall=time.perf_counter() - t0)
print(arm, M, out["closure"], f"{out['wall']:.1f}s", flush=True)
dump(f"v3m_lone_{arm}_M{M}", out)
