"""E2-2: two layers on the SAME map through the curved mortar = the
shared-grid solve.  A circle layer over a lossy film layer, both on the 3 x 3
circle map: (i) layer_grids='shared' with the stack map, (ii) per-layer with
the map given to each layer (identical grids -> the plain square match),
(iii) per-layer FORCED through the mortar (force_mortar=True: the dense
curved cross-mass between two identical mapped grids).  Normal, oblique and
conical incidence.  Fail-before: the forced mortar with the V1/V2 swap of
the H rows switched OFF (_core.PMM2D_MORTAR_H_SWAP = False), the defect the
shipped mortar's G3 gate guards -- invisible on a conforming SEPARABLE
interface (C1 = G1 cancels it), visible here because the curved H-row cross
operator is built swapped while the mass is not."""
import sys
import time

import numpy as np
from _common import CM, EPS_P, N_SUB, N_SUP, R_CIRC, WL, P, PMM2DStackPure, dump

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def circle_cell():
    e = np.ones((3, 3), complex)
    e[1, 1] = EPS_P
    return e


def run(mode, cm2=None, theta=0.0, phi=0.0):
    cm, _w = CM._circle_map_3x3(P, R_CIRC)
    cm2 = cm if cm2 is None else cm2
    if mode == "shared":
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, cmap=cm)
        st.add_layer(0.3, eps_cell=circle_cell())
        st.add_layer(0.2, eps=2.25 + 0.05j)
    else:
        st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                            n_modes=M, n_orders=3, layer_grids="per-layer")
        st.add_layer(0.3, eps_cell=circle_cell(), cmap=cm)
        st.add_layer(0.2, eps=2.25 + 0.05j, cmap=cm2, n_modes=M)
    st.set_source(WL, theta=theta, phi=phi)
    t0 = time.perf_counter()
    if mode == "forced":
        o, R, T, J = st._solve_per_layer(jones=True, retain_internal=False,
                                         force_mortar=True)
    else:
        o, R, T, J = st.solve()
    return np.asarray(R), np.asarray(T), np.asarray(J), time.perf_counter() - t0


out = {}
for (th, ph) in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.7)):
    Rs, Ts, Js, _ = run("shared", theta=th, phi=ph)
    Rp, Tp, Jp, _ = run("perlayer", theta=th, phi=ph)
    Rf, Tf, Jf, tf = run("forced", theta=th, phi=ph)
    d_pl = max(np.abs(Rp - Rs).max(), np.abs(Tp - Ts).max(),
               np.abs(Jp - Js).max())
    d_f = max(np.abs(Rf - Rs).max(), np.abs(Tf - Ts).max(),
              np.abs(Jf - Js).max())
    bytes_pl = bool(np.array_equal(Rp, Rs) and np.array_equal(Tp, Ts))
    from lumenairy.elements.pmm import _core
    _core.PMM2D_MORTAR_H_SWAP = False
    try:
        Rx, Tx, Jx, _ = run("forced", theta=th, phi=ph)
    finally:
        _core.PMM2D_MORTAR_H_SWAP = True
    d_x = max(np.abs(Rx - Rs).max(), np.abs(Tx - Ts).max())
    key = f"th{th}_ph{ph}"
    out[key] = dict(perlayer_vs_shared=d_pl, perlayer_bytes_equal=bytes_pl,
                    forced_mortar_vs_shared=d_f, forced_wall=tf,
                    hswap_off_vs_shared=d_x)
    print(key, out[key])
dump(f"e2_2_same_map_M{M}.json", out)
