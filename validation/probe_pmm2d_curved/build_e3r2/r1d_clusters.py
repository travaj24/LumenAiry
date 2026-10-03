"""R1d: the clusters at gap 1e-6 of the square pillar, M given: members'
lam (complex), inner spread, and the branch-selector decision
(the forward flip of q = sqrt(g2)) at the exact and at the lifted points
t * split_rel (t = -+1, -+2) through the library's own lift.

    python r1d_clusters.py M SPLIT
"""
import sys

import numpy as np
from _r2 import P, PMM2DStackPure, dump, jnp

import lumenairy.elements.rcwa._core as RCC
from lumenairy.elements.pmm import Rect
from lumenairy.elements.pmm._core import _forward_branch_flip

M, SPLIT = int(sys.argv[1]), float(sys.argv[2])
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=[Rect(0.6, 0.6, 0.5, 0.5, 3.5)],
             background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()
sh = tw.sol_h
ref = tw.layers[0]["ref"]
out = {"M": M, "split": SPLIT}
for k, (L, G), shifts in (("geom", (sh.Stt - sh.Schur, -sh.Rmat),
                           [1.0, 1.45 ** 2]),
                          ("layer", (ref.Lmat, -ref.Rmat), [0.0])):
    L, G = jnp.asarray(L), jnp.asarray(G)
    A = jnp.linalg.solve(G, L)
    lam, V = jnp.linalg.eig(A)
    lam_n = np.asarray(lam)
    s = np.max(np.abs(lam_n))
    D = np.abs(lam_n[:, None] - lam_n[None, :]) + np.eye(lam_n.size) * 1e300
    mem = np.min(D, axis=1) <= 1e-6 * s
    nonex = mem & (np.min(D, axis=1) > 1e-12 * s)
    dN, anyc = RCC._eig_cluster_lift(lam, V, G, 1e-6, SPLIT)

    def flips(l):
        res = []
        for e in shifts:
            q = np.sqrt(np.asarray(l, dtype=complex) + e)
            res.append(np.asarray(_forward_branch_flip(q)) != q)
        return np.concatenate(res)
    f0 = flips(lam_n)
    rec = {"s": float(s), "members": int(mem.sum()),
           "nonexact_members": int(nonex.sum()),
           "nonexact_lams": [str(np.round(x, 9)) for x in lam_n[nonex]][:12],
           "max_rel_imag_members": float(np.max(np.abs(lam_n[mem].imag)) / s)}
    for t in (1.0, -1.0, 2.0, -2.0):
        lt = np.asarray(jnp.linalg.eigvals(A + t * dN))
        # match lifted to exact by nearest
        idx = [int(np.argmin(np.abs(lt - x))) for x in lam_n]
        ft = flips(lt[idx])
        moved = np.abs(lt[idx] - lam_n) / s
        rec[f"t={t}"] = {"branch_changes": int(np.sum(ft != f0)),
                         "max_move_rel": float(np.max(moved)),
                         "max_imag_change_rel": float(np.max(np.abs(
                             lt[idx].imag - lam_n.imag)) / s),
                         "min_gap_rel": float(np.min(
                             np.abs(lt[:, None] - lt[None, :])
                             + np.eye(lt.size) * 1e300) / s)}
    out[k] = rec
    print(k, rec, flush=True)
print(dump(f"r1d_clusters_M{M}_{SPLIT:g}.json", out))
