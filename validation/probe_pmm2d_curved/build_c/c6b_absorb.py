"""C6b -- the absorption gate of C6 at the unit-test size (M = 3, merged 7 x 7
map): per-layer ``layer_absorption`` with the plain (metric-free) flux Gram
the stack retains, against the fail-before ``-R`` (Phase A's A7 defect) --
the LOSSLESS circle layer must absorb nothing.

  c6b_absorb.py <M>
Output: c6b_absorb_M<M>.json
"""
import sys

import _common as C
import numpy as np

from lumenairy.elements.pmm import Circle, FilletRect  # noqa: E402


def run(M):
    st = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                          n_substrate=C.N_SUB, n_modes=M, n_orders=3)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.2, 4.0)], background_eps=1.0)
    st.add_layer(0.2, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09,
                                         2.25 + 0.4j)], background_eps=1.0)
    st.set_source(C.WL)
    o, R, T, J = st.solve(retain_internal=True)
    A = np.asarray(st.layer_absorption())
    bal = 1 - np.asarray(R).sum(1) - np.asarray(T).sum(1)
    G = st._internal["G"]
    st._internal["G"] = -C.TS.Granet2DTransverseE(
        C.P, C.P, st.cmap.u_walls, st.cmap.v_walls, M,
        np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
    Ab = np.asarray(st.layer_absorption())
    st._internal["G"] = G
    res = {"env": C.env_record(), "M": M, "A": A.tolist(),
           "A_minusR": Ab.tolist(), "balance": bal.tolist(),
           "lossless_layer_abs": float(np.max(np.abs(A[0]))),
           "lossless_layer_abs_minusR": float(np.max(np.abs(Ab[0]))),
           "sum_vs_balance": float(np.max(np.abs(A.sum(0) - bal))),
           "sum_vs_balance_minusR": float(np.max(np.abs(Ab.sum(0) - bal)))}
    print(res, flush=True)
    C.dump(f"c6b_absorb_M{M}.json", res)


if __name__ == "__main__":
    run(int(sys.argv[1]))
