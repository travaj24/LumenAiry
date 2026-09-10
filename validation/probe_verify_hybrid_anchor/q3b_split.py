"""Q3b -- the LAYER-SPLIT identity, done on the SAME cell in both arms (the
q3 run's first attempt rolled the lower half by one pixel instead of the half
walk, so it measured a different solid; kept out of q3 and redone here).

One slanted layer of depth ``d`` vs TWO slanted halves of ``d/2`` whose lower
half's TOP cell is the upper's rolled by half the walk.  The identity itself is
sha-exact; the point of the row is that it is BLIND to the walk SUM -- give
both arms only half the sum and they move together.
"""
from __future__ import annotations

import time

import _lib as L
import numpy as np
from q3_composition import build, sheared_layers

TX, D = 0.5, L.DTHICK           # walk = P/4
U = 8                           # 48-column cell: half the walk = 6 px
K_ORACLE = 15


def main():
    t0 = time.time()
    k0 = L.k0_of()
    W = TX * D
    cell = L.upsample(L.BASE, U)
    # the frame CONTINUES downward, so the lower half's cell is passed AS
    # WRITTEN -- the cascade already places it at the accumulated walk.  The
    # ROLLED arm below is the alternative reading, and it is measurably wrong.
    rolled = np.roll(cell, cell.shape[0] // 8, axis=0)   # P/8 = W/2
    out = {"walk_um": W * 1e6, "half_walk_pixels": cell.shape[0] // 8}
    for mount in ("oblique25", "conical25_40"):
        st1, _ = build([dict(thickness=D, eps_cell=cell, slant=(TX, 0.0))],
                       mount)
        st2, _ = build([dict(thickness=D / 2, eps_cell=cell, slant=(TX, 0.0)),
                        dict(thickness=D / 2, eps_cell=cell,
                             slant=(TX, 0.0))], mount)
        st3, _ = build([dict(thickness=D / 2, eps_cell=cell, slant=(TX, 0.0)),
                        dict(thickness=D / 2, eps_cell=rolled,
                             slant=(TX, 0.0))], mount)
        j1, j2 = st1.jones_transmission(), st2.jones_transmission()
        a1 = st1.per_order_amplitudes("transmission")
        a2 = st2.per_order_amplitudes("transmission")
        so, _ = build(sheared_layers(L.BASE, D, 0.25, 0.0, K_ORACLE), mount)
        so5, _ = build(sheared_layers(L.BASE, D, 0.25, 0.0, 5), mount)
        b = so.per_order_amplitudes("transmission")
        row = dict(
            jones_sha_equal=(L.sha(j1) == L.sha(j2)),
            per_order_sha_equal=(
                L.sha(np.concatenate([a1["Ex"], a1["Ey"]], axis=1))
                == L.sha(np.concatenate([a2["Ex"], a2["Ey"]], axis=1))),
            jones_gap=L.jones_residual(j2, j1),
            per_order_gap=L.amp_residual(a2, a1)[0],
            one_layer_vs_oracle=L.amp_residual(a1, b)[0],
            two_halves_vs_oracle=L.amp_residual(a2, b)[0],
            two_halves_ROLLED_vs_oracle=L.amp_residual(
                st3.per_order_amplitudes("transmission"), b)[0],
            oracle_step_K5_K15=L.amp_residual(
                so5.per_order_amplitudes("transmission"), b)[0],
            # the BLINDNESS: HALF the sum, both arms
            one_layer_half_sum=L.amp_residual(
                L.rephase(a1, (-W / 2, 0.0), k0), b)[0],
            two_halves_half_sum=L.amp_residual(
                L.rephase(a2, (-W / 2, 0.0), k0), b)[0],
            one_layer_no_sum=L.amp_residual(
                L.rephase(a1, (-W, 0.0), k0), b)[0],
            two_halves_no_sum=L.amp_residual(
                L.rephase(a2, (-W, 0.0), k0), b)[0])
        out[mount] = row
        print("==", mount)
        for k, v in row.items():
            print("   %-24s %s" % (k, v))
    out["seconds"] = round(time.time() - t0, 1)
    L.dump("q3b_split", out)


if __name__ == "__main__":
    main()
