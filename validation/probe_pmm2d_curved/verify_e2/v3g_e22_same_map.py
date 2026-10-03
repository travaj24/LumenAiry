"""E2-2 on the verifier's geometries: the SAME map on both layers, joined
through the FORCED curved mortar, vs the shared-map solve.  Maps: a ROTATED
off-centre ellipse (3x3, sheared corners), an off-centre circle, an
off-centre FilletRect (5x5).  Layer 1 the shape (eps 3.4), layer 2 a lossy
film eps 3.1 + 0.12i on the same map.  Normal / oblique / conical.
Fail-before: _core.PMM2D_MORTAR_H_SWAP = False on the forced arm.
Usage: v3g_e22_same_map.py M"""
import sys
import time

import numpy as np
from _ve import dump
from v3g_fix import N_SUB, N_SUP, NREC, WL, P, PMM2DStackPure

from lumenairy.elements.pmm import _core
from lumenairy.elements.pmm.shapes2d import Circle, Ellipse, FilletRect, compile_shapes

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
FILM = 3.1 + 0.12j
SHAPES = {
    "ellipse_rot": Ellipse(0.52, 0.6, 0.33, 0.22, 3.4, angle=0.35),
    "circle_off": Circle(0.62, 0.47, 0.3, 3.4),
    "fillet_off": FilletRect(0.5, 0.58, 0.56, 0.44, 0.13, 3.4),
}


def run(mode, cell, cm, th, ph, swap=True):
    if mode == "shared":
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, cmap=cm)
        st.add_layer(0.28, eps_cell=cell)
        st.add_layer(0.21, eps=FILM)
    else:
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        st.add_layer(0.28, eps_cell=cell, cmap=cm)
        st.add_layer(0.21, eps=FILM, cmap=cm, n_modes=M)
    st.set_source(WL, theta=th, phi=ph)
    _core.PMM2D_MORTAR_H_SWAP = swap
    t0 = time.perf_counter()
    try:
        if mode == "forced":
            o, R, T, J = st._solve_per_layer(jones=True, retain_internal=False,
                                             force_mortar=True)
        else:
            o, R, T, J = st.solve()
    finally:
        _core.PMM2D_MORTAR_H_SWAP = True
    return (np.asarray(R), np.asarray(T), np.asarray(J),
            time.perf_counter() - t0)


def d(a, b, jones=True):
    v = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())
    if jones:
        v = max(v, np.abs(a[2] - b[2]).max())
    return float(v)


ONLY = sys.argv[2] if len(sys.argv) > 2 else None
out = {"M": M}
for name, shp in SHAPES.items():
    if ONLY and name != ONLY:
        continue
    cell, xw, yw, cm = compile_shapes(P, P, [shp], 1.0)
    for (th, ph) in ((0.0, 0.0), (0.33, 0.0), (0.33, 1.1)):
        nrec0 = len(NREC)
        s = run("shared", cell, cm, th, ph)
        p = run("perlayer", cell, cm, th, ph)
        f = run("forced", cell, cm, th, ph)
        nrec_f = NREC[nrec0:]
        x = run("forced", cell, cm, th, ph, swap=False)
        key = f"{name}_th{th}_ph{ph}"
        out[key] = dict(
            grid=list(cell.shape),
            perlayer_vs_shared=d(p, s),
            perlayer_bytes=bool(np.array_equal(p[0], s[0])
                                and np.array_equal(p[1], s[1])),
            forced_vs_shared=d(f, s),
            hswap_off_vs_shared=d(x, s, jones=False),
            closure_shared=np.abs(s[0].sum(1) + s[1].sum(1) - 1).tolist(),
            absorbed_shared=(1 - s[0].sum(1) - s[1].sum(1)).tolist(),
            forced_n=[r["n"] for r in nrec_f],
            forced_change=[r["change"] for r in nrec_f],
            wall_shared=s[3], wall_forced=f[3])
        print(key, {k: v for k, v in out[key].items()
                    if k not in ("closure_shared",)}, flush=True)
dump(f"v3g_e22_same_map_M{M}" + (f"_{ONLY}" if ONLY else ""), out)
