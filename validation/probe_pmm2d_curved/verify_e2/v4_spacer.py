"""V4 identity ladders (a VACUUM spacer on top of a pillar, air above: a
physical no-op), so the spacer-vs-alone difference is the mortar's own error
with NO reference error.

usage: v4_spacer.py <family> <M> <eps_pillar>
family 'shipped' -- SHIPPED separable mortar only (no map): square pillar
   x, y in 0.4 .. 0.8 (uniform 3 x 3 cell, as build_e2/e2_v_spacer_shipped),
   t 0.3; spacer t 0.25 eps 1 on 'grid1' (shipped default), 'offset3b'
   (the builder's walls 0.2 / 0.8 -- 0.8 COINCIDES with a pillar wall),
   'offset3' (walls 0.2 / 0.7: fully non-conforming, n_modes=M),
   'offset3q' (walls 0.2 / 0.7 at n_modes = M + 1), 'conforming' (grid=3).
family 'curved' -- the eps disk r 0.36 on the circle map; a vacuum-painted
   sinusoid layer (x0 0.6, A 0.12) on top kept on its OWN map
   (_e2_no_ride) -- a genuine curved mortar -- vs the circle alone (shared).
Also writes the flat-device R/T so the error can be normalised by the
scattering strength."""
import sys
import time

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

fam, M, E = sys.argv[1], int(sys.argv[2]), complex(sys.argv[3])
ARMS = (sys.argv[4].split(",") if len(sys.argv) > 4 else
        ["grid1", "offset3b", "offset3", "offset3q", "conforming"])
P = 1.2
kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M, n_orders=3)
out = dict(family=fam, M=M, eps=E)
def d(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))

if fam == "shipped":
    cell = np.ones((3, 3), complex)
    cell[1, 1] = E
    base = PMM2DStackPure(P, P, **kw)
    base.add_layer(0.3, eps_cell=cell)
    a = solve(base)
    out["alone_R"], out["alone_T"], out["orders"] = a[1], a[2], a[0]
    for arm in ARMS:
        t0 = time.perf_counter()
        st = PMM2DStackPure(P, P, layer_grids="per-layer", **kw)
        if arm == "grid1":
            st.add_layer(0.25, eps=1.0)
        elif arm == "offset3b":
            st.add_layer(0.25, eps=1.0, x_walls=[0.2, 0.8],
                         y_walls=[0.2, 0.8], n_modes=M)
        elif arm == "offset3":
            st.add_layer(0.25, eps=1.0, x_walls=[0.2, 0.7],
                         y_walls=[0.2, 0.7], n_modes=M)
        elif arm == "offset3q":
            st.add_layer(0.25, eps=1.0, x_walls=[0.2, 0.7],
                         y_walls=[0.2, 0.7], n_modes=M + 1)
        else:
            st.add_layer(0.25, eps=1.0, grid=3, n_modes=M)
        st.add_layer(0.3, eps_cell=cell)
        b = solve(st)
        out[arm] = dict(vs_alone=d(a, b), Ms=st._perlayer_modal_counts(),
                        R=b[1], T=b[2],
                        closure=np.abs(b[1].sum(1) + b[2].sum(1) - 1),
                        wall=time.perf_counter() - t0)
        print(arm, out[arm])
else:
    base = PMM2DStackPure(P, P, **kw)
    base.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, E)],
                   background_eps=1.0)
    a = solve(base)
    out["alone_R"], out["alone_T"], out["orders"] = a[1], a[2], a[0]
    t0 = time.perf_counter()
    st = PMM2DStackPure(P, P, layer_grids="per-layer", **kw)
    st.add_layer(0.25, shapes=[SinusoidalWall("x", 0.6, 0.12, eps=1.0)],
                 background_eps=1.0)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, E)],
                 background_eps=1.0)
    st._e2_no_ride = True
    b = solve(st)
    out["noride"] = dict(vs_alone=d(a, b), Ms=st._perlayer_modal_counts(),
                         closure=np.abs(b[1].sum(1) + b[2].sum(1) - 1),
                         wall=time.perf_counter() - t0)
    print("noride", out["noride"])
dump(f"v4_spacer_{fam}_{E.real:g}_M{M}" + ("_amp" if len(sys.argv) > 4 else ""), out)
