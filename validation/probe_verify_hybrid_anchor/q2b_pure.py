"""Q2 (b) -- the PURE engine as the cross-engine arm, on a cell BOTH engines
express, bounded by the HYBRID's own ``n_orders`` ladder.

Separated from ``q2_anchor.py`` because the pure engine's cost is
``(n_segments * n_modes)`` per axis per component: ``n_modes = 7`` on a 6 x 6
cell is a 7000-dof dense eig.  ``n_modes = 4/5`` is the usable rung and is what
this measures.
"""
from __future__ import annotations

import time
import warnings

import _lib as L
import numpy as np
from q2_anchor import BASE6, TX, arms


def main():
    t0 = time.time()
    k0 = L.k0_of()
    W = (TX * L.DTHICK, 0.0)
    res = {}
    for mount in ("oblique25", "conical25_40"):
        th, ph = L.MOUNTS[mount]
        hy = {}
        for M in (5, 7, 9):
            st = L.hybrid(n_orders=M)
            st.add_layer(L.DTHICK, eps_cell=BASE6, slant=(TX, 0.0))
            st.set_source(L.WL, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.solve()
            hy[M] = (st.per_order_amplitudes("transmission"),
                     st.jones_transmission())
        row = {}
        from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
        for nm in (4, 5):
            pu = PMM2DStackPure(L.PX, L.PY, n_superstrate=L.NSUP,
                                n_substrate=L.NSUB, n_modes=nm, n_orders=3)
            pu.add_layer(L.DTHICK, eps_cell=BASE6, slant=(TX, 0.0))
            pu.set_source(L.WL, theta=th, phi=ph)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pu.solve(jones=True)
            pa = pu.per_order_amplitudes("transmission")
            pj = pu.jones_transmission()
            a5, j5 = hy[5]
            r = arms(a5, pa, W, k0)
            p0 = int(np.where((np.asarray(a5["orders"])[:, 0] == 0)
                              & (np.asarray(a5["orders"])[:, 1] == 0))[0][0])
            P0 = complex(np.exp(1j * k0 * (a5["kx"][p0] * W[0]
                                           + a5["ky"][p0] * W[1])))
            r["jones_shipped"] = L.jones_residual(j5, pj)
            r["jones_none"] = L.jones_residual(j5 / P0, pj)
            r["jones_conj"] = L.jones_residual(j5 * np.conj(P0) / P0, pj)
            row["n_modes%d" % nm] = r
        row["hybrid_own_step_5_7"] = L.amp_residual(hy[5][0], hy[7][0])[0]
        row["hybrid_own_step_7_9"] = L.amp_residual(hy[7][0], hy[9][0])[0]
        row["hybrid_own_jones_step_5_7"] = L.jones_residual(hy[5][1], hy[7][1])
        res[mount] = row
        print("==", mount)
        for k, v in row.items():
            print("   ", k, v)
    res["seconds"] = round(time.time() - t0, 1)
    L.dump("q2b_pure", res)


if __name__ == "__main__":
    main()
