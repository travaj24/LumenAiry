"""D7 -- a TENSOR and a MAGNETIC layer in ONE merged-map stack.

Layer 1: a circle (r = 0.2) of the LC30 tensor in air; layer 2: a filleted
square (0.9 x 0.9, r = 0.09) of eps 2.25 with mu = diag(1.5, 1.5, 1.2) in
air -- Phase C's C6 geometry (one merged 7 x 7 map) with the materials made
anisotropic and magnetic.  P = 1.2, lambda 1, air above, n = 1.45 below.

* closure  -- everything lossless: |sum R + sum T - 1|;
* absorb   -- the LC made LOSSY (LC30 + 0.3i I) and, in a second arm, the
  magnetic layer's mu made lossy (mu_t 1.5 + 0.2i) instead: per-layer
  ``layer_absorption`` with the plain flux Gram; the LOSSLESS layer must
  absorb nothing and the sum must equal 1 - sum R - sum T to the
  discretisation level; fail-before = -R as the flux Gram (Phase A's A7
  defect);
* vacuum   -- layer 2 painted with VACUUM shapes (eps 1, mu 1 explicitly,
  i.e. through the magnetic tensor route) vs a uniform vacuum layer on the
  same merged map (a physical identity), and the same shape with mu = 1 vs
  WITHOUT mu (the magnetic route against the scalar route).

usage: python d7_stack.py <closure|absorb|vacuum> <M>
"""
import sys
import time

import _dcommon as D
import numpy as np

from lumenairy.elements.pmm import Circle, FilletRect

P, WL = 1.2, 1.0
MU2 = np.diag([1.5, 1.5, 1.2]).astype(complex)


def stack(M, eps1, eps2, mu2, layer2="shape", retain=False):
    st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                          n_modes=M, n_orders=3)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.2, eps1)],
                 background_eps=1.0)
    if layer2 == "shape":
        st.add_layer(0.2, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09, eps2,
                                             mu=mu2)], background_eps=1.0)
    elif layer2 == "uniform_vacuum":
        # the SAME merged map (the fillet's walls) needs the fillet in SOME
        # layer: put a vacuum fillet in a zero-contrast third layer-free way
        # -- a vacuum-painted fillet layer IS that, see run_vacuum
        raise ValueError(layer2)
    st.set_source(WL)
    o, R, T, J = st.solve(retain_internal=retain)
    return st, np.asarray(o), np.asarray(R), np.asarray(T)


def run_closure(M):
    t0 = time.perf_counter()
    st, o, R, T = stack(M, D.LC30, 2.25, MU2)
    res = {"M": M, "grid": list(st.cmap.shape),
           "closure": float(np.abs(R.sum(1) + T.sum(1) - 1).max()),
           "t": time.perf_counter() - t0}
    D.dump(f"d7_closure_M{M}.json", res)
    print(res, flush=True)


def _absorb(st, M, R, T):
    A = np.asarray(st.layer_absorption())
    bal = 1 - R.sum(1) - T.sum(1)
    G = st._internal["G"]
    st._internal["G"] = -D.TS.Granet2DTransverseE(
        P, P, st.cmap.u_walls, st.cmap.v_walls, M,
        np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
    Ab = np.asarray(st.layer_absorption())
    st._internal["G"] = G
    return A, Ab, bal


def run_absorb(M):
    out = {"M": M}
    for arm, e1, mu2, lossless in (
            ("lossy_tensor", D.LC30 + 0.3j * np.eye(3), MU2, 1),
            ("lossy_mu", D.LC30, np.diag([1.5 + 0.2j, 1.5 + 0.2j,
                                          1.2]).astype(complex), 0)):
        st, o, R, T = stack(M, e1, 2.25, mu2, retain=True)
        A, Ab, bal = _absorb(st, M, R, T)
        out[arm] = {"A": A.tolist(), "balance": bal.tolist(),
                    "lossless_layer": lossless,
                    "lossless_layer_abs": float(np.abs(A[lossless]).max()),
                    "lossless_layer_abs_minusR": float(
                        np.abs(Ab[lossless]).max()),
                    "sum_vs_balance": float(np.abs(A.sum(0) - bal).max()),
                    "sum_vs_balance_minusR": float(
                        np.abs(Ab.sum(0) - bal).max())}
        print(arm, {k: v for k, v in out[arm].items()
                    if k not in ("A", "balance")}, flush=True)
    D.dump(f"d7_absorb_M{M}.json", out)


def run_vacuum(M):
    """Layer 2 painted with a vacuum fillet carrying an explicit mu = 1
    (magnetic tensor route) vs the same fillet painted vacuum with no mu
    (scalar route) vs a uniform vacuum layer -- all on the SAME merged map
    (the fillet in layer 2 fixes the walls; a third, real, fillet layer
    keeps the map identical when layer 2 is made uniform)."""
    def mk(layer2):
        st = D.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                              n_modes=M, n_orders=3)
        st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.2, D.LC30)],
                     background_eps=1.0)
        if layer2 == "uniform":
            st.add_layer(0.2, eps=1.0)
        elif layer2 == "vac_mu":
            st.add_layer(0.2, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09,
                                                 1.0, mu=np.eye(3))],
                         background_eps=1.0, background_mu=1.0)
        elif layer2 == "vac":
            st.add_layer(0.2, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09,
                                                 1.0)], background_eps=1.0)
        elif layer2 == "paint11":
            st.add_layer(0.2, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09,
                                                 1.1)], background_eps=1.0)
        st.add_layer(0.25, shapes=[FilletRect(0.6, 0.6, 0.9, 0.9, 0.09,
                                              2.25, mu=MU2)],
                     background_eps=1.0)
        st.set_source(WL)
        o, R, T, J = st.solve(jones=True)
        return st.cmap.fingerprint, np.asarray(R), np.asarray(T), \
            np.asarray(J)
    runs = {k: mk(k) for k in ("uniform", "vac_mu", "vac", "paint11")}
    fp = {k: v[0] for k, v in runs.items()}

    def d(a, b):
        return float(max(np.abs(runs[a][1] - runs[b][1]).max(),
                         np.abs(runs[a][2] - runs[b][2]).max(),
                         np.abs(runs[a][3] - runs[b][3]).max()))
    res = {"M": M, "same_map": len(set(fp.values())) == 1,
           "vac_mu_vs_uniform": d("vac_mu", "uniform"),
           "vac_vs_uniform": d("vac", "uniform"),
           "vac_mu_vs_vac": d("vac_mu", "vac"),
           "paint11_vs_uniform": d("paint11", "uniform")}
    D.dump(f"d7_vacuum_M{M}.json", res)
    print(res, flush=True)


if __name__ == "__main__":
    {"closure": run_closure, "absorb": run_absorb,
     "vacuum": run_vacuum}[sys.argv[1]](int(sys.argv[2]))
